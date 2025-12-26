# SPDX-License-Identifier: Apache-2.0
"""
将 OpenAI MRCR 数据集转换为 vLLM Custom 格式。
重点：生成符合 Log-Normal 分布（平均值 6k，长尾至 65k）的 Prompt。

用法:
python mrcr-custom.py \
    --model-id  deepseek-ai/DeepSeek-V3.1-Terminus  \
    --output-file ./bench_data_lognormal.jsonl \
    --target-samples 3000 \
    --num-workers 16
"""

import argparse
import json
import os
import random
import math
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor
from typing import List, Tuple, Optional
from transformers import AutoTokenizer
from huggingface_hub import hf_hub_download
import tqdm

# --- 全局变量 ---
_tokenizer: Optional[AutoTokenizer] = None

def generate_length_distribution(
    num_samples: int, 
    target_mean: float, 
    min_val: int = 10, 
    max_val: int = 65536,
    sigma: float = 1.0 # 控制尾部的肥厚程度，1.0 适合文本分布
) -> np.ndarray:
    """
    生成对数正态分布的长度数组，并调整以匹配目标平均值。
    """
    print(f"正在生成分布 (目标均值: {target_mean}, Sigma: {sigma})...")
    
    # Log-Normal 的数学性质: Mean = exp(mu + sigma^2 / 2)
    # 推导 mu: mu = ln(target_mean) - sigma^2 / 2
    # 注意：这里的 target_mean 是期望的算术平均值
    mu = np.log(target_mean) - (sigma**2 / 2)
    
    # 生成数据
    lengths = np.random.lognormal(mean=mu, sigma=sigma, size=num_samples)
    
    # 截断范围 [min, max]
    lengths = np.clip(lengths, min_val, max_val)
    
    # 由于 clip 操作会改变平均值，进行简单的缩放修正
    current_mean = np.mean(lengths)
    scale_factor = target_mean / current_mean
    lengths = lengths * scale_factor
    
    # 再次截断并取整
    lengths = np.clip(lengths, min_val, max_val).astype(int)
    
    # 打印统计数据
    print("\n--- 长度分布统计 ---")
    print(f"样本数: {len(lengths)}")
    print(f"Mean (平均): {np.mean(lengths):.2f}")
    print(f"Median (中位): {np.median(lengths):.2f}")
    print(f"Min: {np.min(lengths)}")
    print(f"Max: {np.max(lengths)}")
    print(f"P99 (99%分位): {np.percentile(lengths, 99):.0f}")
    
    # 简单的 ASCII 直方图
    hist, bin_edges = np.histogram(lengths, bins=10)
    print("\n[分布直方图]")
    for i in range(len(hist)):
        bar = '#' * int(hist[i] / len(lengths) * 50)
        range_str = f"{int(bin_edges[i]):5d} - {int(bin_edges[i+1]):5d}"
        print(f"{range_str} | {bar} ({hist[i]})")
    print("-" * 30 + "\n")
    
    return lengths

def init_worker(model_id: str):
    global _tokenizer
    try:
        _tokenizer = AutoTokenizer.from_pretrained(
            model_id, 
            trust_remote_code=True,
            model_max_length=1000000 # 防止警告
        )
        if not _tokenizer.pad_token:
            _tokenizer.pad_token = _tokenizer.eos_token
    except Exception as e:
        print(f"Tokenizer init failed: {e}")

def process_row(args: Tuple[str, int]) -> str:
    """
    从源文本中截取指定长度的 Prompt。
    优化：先按字符切片，再 Tokenize，提升速度。
    """
    global _tokenizer
    json_prompt, target_len = args
    
    if _tokenizer is None:
        return ""

    try:
        # 1. 提取纯文本
        messages = json.loads(json_prompt)
        full_text = ""
        for m in messages:
            full_text += m.get("content", "") + "\n"
        
        # 2. 性能优化：字符级预截断
        # 假设 1 token ≈ 4 chars (中文可能更少，英文更多，取安全值)
        # 为了保证由足够的 token，我们截取 target_len * 6 个字符
        # 对于超短 target (如 10)，这避免了 encode 整个 100k 的 context
        char_limit = max(100, target_len * 6)
        if len(full_text) > char_limit:
            full_text = full_text[:char_limit]

        # 3. Tokenize
        token_ids = _tokenizer.encode(full_text, add_special_tokens=False)
        
        # 4. 精确截断
        if len(token_ids) < target_len:
            # 如果源文本不够长，直接返回（分布允许有些许误差）
            # 或者为了严谨，可以选择跳过这条，但在大量数据下无所谓
            final_ids = token_ids
        else:
            final_ids = token_ids[:target_len]
            
        # 5. Decode
        return _tokenizer.decode(final_ids)
        
    except Exception:
        return ""

def main():
    parser = argparse.ArgumentParser(description="生成符合 Log-Normal 分布的 vLLM 测试数据")
    parser.add_argument("--model-id", type=str, required=True, help="Tokenizer ID")
    parser.add_argument("--output-file", type=str, required=True)
    parser.add_argument("--target-samples", type=int, default=2000)
    parser.add_argument("--num-workers", type=int, default=os.cpu_count() or 4)
    
    # 分布控制参数
    parser.add_argument("--mean", type=int, default=6000, help="目标平均长度")
    parser.add_argument("--sigma", type=float, default=1.1, help="分布形状，越大尾巴越长 (建议 0.8 - 1.2)")
    parser.add_argument("--min-len", type=int, default=10, help="最小长度")
    parser.add_argument("--max-len", type=int, default=65536, help="最大长度")

    args = parser.parse_args()
    
    # 1. 下载/读取 MRCR 数据
    print("正在加载 OpenAI MRCR 数据集...")
    try:
        # MRCR 数据集很大，只需要其中的一部分即可
        # 2needle.parquet 包含了最长的数据
        df = pd.read_parquet(
            hf_hub_download(repo_id="openai/mrcr", filename="2needle.parquet", repo_type="dataset")
        )
        # 将所有 prompt 重复利用，直到足够覆盖 target_samples
        raw_prompts = df["prompt"].tolist()
        while len(raw_prompts) < args.target_samples:
            raw_prompts.extend(raw_prompts)
        
        # 随机打乱源数据，避免总是取到同一个开头
        random.shuffle(raw_prompts)
        source_data = raw_prompts[:args.target_samples]
        
    except Exception as e:
        print(f"数据加载失败: {e}")
        return

    # 2. 生成目标长度分布
    target_lengths = generate_length_distribution(
        args.target_samples, 
        target_mean=args.mean,
        min_val=args.min_len,
        max_val=args.max_len,
        sigma=args.sigma
    )
    
    # 3. 准备任务
    tasks = list(zip(source_data, target_lengths))
    
    # 4. 并行处理
    print(f"\n开始处理 {len(tasks)} 条数据...")
    results = []
    
    with ProcessPoolExecutor(max_workers=args.num_workers, initializer=init_worker, initargs=(args.model_id,)) as executor:
        futures = list(tqdm.tqdm(executor.map(process_row, tasks), total=len(tasks), desc="生成 Prompts"))
        for res in futures:
            if res and len(res.strip()) > 0:
                results.append(res)
                
    # 5. 写入文件
    print(f"正在写入 {args.output_file} ...")
    with open(args.output_file, "w", encoding="utf-8") as f:
        for prompt in results:
            f.write(json.dumps({"prompt": prompt}, ensure_ascii=False) + "\n")
            
    print("完成！")

if __name__ == "__main__":
    main()