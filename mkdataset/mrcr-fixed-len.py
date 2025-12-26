# SPDX-License-Identifier: Apache-2.0
"""
将 OpenAI MRCR 数据集转换为 vLLM Custom 格式。
重点：生成长度严格一致的数据集 (例如 5.5k tokens)。

用法:
python mkdataset/mrcr-fixed-len.py \
    --model-id deepseek-ai/DeepSeek-V3.1-Terminus \
    --output-file ./bench_data_fixed_5500.jsonl \
    --target-samples 1000 \
    --fixed-len 5500 \
    --num-workers 16
"""

import argparse
import json
import os
import random
from concurrent.futures import ProcessPoolExecutor
from typing import List, Tuple, Optional
import pandas as pd
import tqdm
from transformers import AutoTokenizer
from huggingface_hub import hf_hub_download

# --- 全局变量 ---
_tokenizer: Optional[AutoTokenizer] = None
_target_length: int = 5500

def init_worker(model_id: str, target_len: int):
    global _tokenizer
    global _target_length
    _target_length = target_len
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

def process_row(json_prompt: str) -> Optional[str]:
    """
    从源文本中截取指定固定长度的 Prompt。
    如果长度不足，返回 None。
    如果长度足够，截取到精确长度。
    """
    global _tokenizer
    global _target_length
    
    if _tokenizer is None:
        return None

    try:
        # 1. 提取纯文本
        # MRCR 数据集中的 prompt 列通常是 JSON 格式的消息列表
        try:
            messages = json.loads(json_prompt)
            full_text = ""
            for m in messages:
                full_text += m.get("content", "") + "\n"
        except json.JSONDecodeError:
            # 如果不是 JSON，尝试直接作为文本处理
            full_text = json_prompt
        
        # 2. 简单的字符级预筛选 (优化性能)
        # 假设 1 token 平均最少 1 个字符 (非常保守)，如果字符数少于 target_len，肯定不够
        if len(full_text) < _target_length:
            return None

        # 3. Tokenize
        token_ids = _tokenizer.encode(full_text, add_special_tokens=False)
        
        # 4. 长度检查与截断
        if len(token_ids) < _target_length:
            return None # 长度不足，跳过
            
        final_ids = token_ids[:_target_length]
            
        # 5. Decode
        # skip_special_tokens=True 可能更好，但这里保持原样以保留原文风味，
        # 只要没有截断在特殊 token 中间即可。
        return _tokenizer.decode(final_ids)
        
    except Exception:
        return None

def main():
    parser = argparse.ArgumentParser(description="生成固定长度的 vLLM 测试数据")
    parser.add_argument("--model-id", type=str, required=True, help="Tokenizer ID")
    parser.add_argument("--output-file", type=str, required=True)
    parser.add_argument("--target-samples", type=int, default=1000)
    parser.add_argument("--num-workers", type=int, default=os.cpu_count() or 4)
    parser.add_argument("--fixed-len", type=int, default=5500, help="目标固定长度 (tokens)")
    
    args = parser.parse_args()
    
    print(f"目标: 生成 {args.target_samples} 条长度严格为 {args.fixed_len} tokens 的数据。")

    # 1. 下载/读取 MRCR 数据
    print("正在加载 OpenAI MRCR 数据集...")
    try:
        # 使用 2needle.parquet 以获得较长文本
        parquet_path = hf_hub_download(repo_id="openai/mrcr", filename="2needle.parquet", repo_type="dataset")
        df = pd.read_parquet(parquet_path)
        
        # 假设我们需要比目标多一些的原始数据，以防过滤掉太短的
        # 这里简单加载所有数据，然后 shuffle
        raw_prompts = df["prompt"].tolist()
        random.shuffle(raw_prompts)
        
        print(f"原始数据池大小: {len(raw_prompts)}")
        
    except Exception as e:
        print(f"数据加载失败: {e}")
        return

    # 2. 并行处理
    print(f"\n开始处理...")
    valid_results = []
    
    # 我们分批处理，直到满足 target_samples
    chunk_size = args.target_samples * 2  # 每次尝试处理 2 倍于目标的数据量
    current_idx = 0
    
    with ProcessPoolExecutor(
        max_workers=args.num_workers, 
        initializer=init_worker, 
        initargs=(args.model_id, args.fixed_len)
    ) as executor:
        
        pbar = tqdm.tqdm(total=args.target_samples, desc="收集有效样本")
        
        while len(valid_results) < args.target_samples and current_idx < len(raw_prompts):
            batch = raw_prompts[current_idx : current_idx + chunk_size]
            current_idx += len(batch)
            
            # 提交任务
            futures = executor.map(process_row, batch)
            
            for res in futures:
                if res:
                    valid_results.append(res)
                    pbar.update(1)
                    if len(valid_results) >= args.target_samples:
                        break
            
            if len(valid_results) < args.target_samples and current_idx >= len(raw_prompts):
                print(f"\n警告: 遍历了所有原始数据，仅收集到 {len(valid_results)} 条有效样本。")
                break

        pbar.close()

    # 3. 截取正好需要的数量
    final_results = valid_results[:args.target_samples]
            
    # 4. 写入文件
    print(f"\n正在写入 {len(final_results)} 条数据到 {args.output_file} ...")
    with open(args.output_file, "w", encoding="utf-8") as f:
        for prompt in final_results:
            # vLLM Custom 格式: {"prompt": "..."}
            f.write(json.dumps({"prompt": prompt}, ensure_ascii=False) + "\n")
            
    print("完成！")

if __name__ == "__main__":
    main()
