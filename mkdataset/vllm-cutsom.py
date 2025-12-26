#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
将 LongBench (.jsonl) 和 ShareGPT (.json) 合并为 vLLM Custom 格式 (.jsonl)
并通过分层采样达到目标平均 prompt 长度。

Custom 格式示例:
{"prompt": "What is the capital of India?"}
{"prompt": "What is the capital of Iran?"}

用法:
python merge_to_custom_format.py \
    --longbench-dir ./datasets/longbench/ \
    --sharegpt-file ./datasets/ShareGPT_V3_unfiltered_cleaned_split.json \
    --output-file ./datasets/benchmark_custom_6k.jsonl \
    --model-id meta-llama/Llama-3.1-8B-Instruct \
    --target-size 3000 \
    --target-avg-prompt 6000 \
    --max-prompt-len 32000 \
    --num-workers 8
"""

import argparse
import json
import math
import os
import random
from concurrent.futures import ProcessPoolExecutor
from typing import List, Dict, Any, Tuple, Optional

import numpy as np
import pandas as pd
import tqdm
from transformers import AutoTokenizer

# 全局 tokenizer for multiprocessing
_tokenizer: Optional[AutoTokenizer] = None

# LongBench 数据集列表
LONGBENCH_DATASETS = [
    "narrativeqa", "qasper", "multifieldqa_en", "hotpotqa", "2wikimqa", 
    "musique", "gov_report", "qmsum", "multi_news", "trec", "triviaqa", 
    "samsum", "passage_retrieval_en", "lcc", "repobench-p"
]

def init_worker(model_id: str) -> None:
    """初始化每个工作进程的分词器"""
    global _tokenizer
    try:
        _tokenizer = AutoTokenizer.from_pretrained(
            model_id, 
            trust_remote_code=True,
            model_max_length=999999
        )
        if not _tokenizer.pad_token:
            _tokenizer.pad_token = _tokenizer.eos_token
    except Exception as e:
        print(f"工作进程初始化 tokenizer 失败: {e}")


def process_item(args: Tuple[str, str]) -> Optional[Tuple[str, int]]:
    """
    处理单个样本，返回 (prompt, prompt_len)
    args: (prompt_text, source_type)
    """
    global _tokenizer
    if _tokenizer is None:
        return None
    
    prompt_text, source_type = args
    
    try:
        # 计算 token 长度
        prompt_len = len(_tokenizer.encode(prompt_text))
        return (prompt_text, prompt_len)
    except Exception:
        return None


def load_longbench(
    longbench_dir: str,
    datasets: List[str],
    num_workers: int
) -> List[Tuple[str, int]]:
    """
    加载 LongBench 数据集并转换为 (prompt, prompt_len) 格式
    """
    print(f"\n=== 加载 LongBench 数据集 ===")
    
    all_prompts = []
    
    for dataset_name in datasets:
        filepath = os.path.join(longbench_dir, f"{dataset_name}.jsonl")
        
        if not os.path.exists(filepath):
            print(f"跳过不存在的文件: {filepath}")
            continue
        
        print(f"正在加载: {dataset_name}.jsonl")
        
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                lines = f.readlines()
            
            for line in lines:
                try:
                    item = json.loads(line)
                    
                    # 构造 prompt: context + input
                    context = item.get("context", "") or ""
                    input_text = item.get("input", "") or ""
                    
                    prompt = (context + "\n\n" + input_text).strip()
                    
                    if prompt:
                        all_prompts.append((prompt, "longbench"))
                        
                except json.JSONDecodeError:
                    continue
                    
        except Exception as e:
            print(f"读取 {filepath} 失败: {e}")
    
    print(f"LongBench 总计加载: {len(all_prompts):,} 个 prompts")
    
    # 并行计算 token 长度
    print("正在计算 LongBench prompts 的 token 长度...")
    analyzed_prompts = []
    
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = executor.map(process_item, all_prompts, chunksize=100)
        
        for result in tqdm.tqdm(futures, total=len(all_prompts), desc="分析 LongBench"):
            if result:
                analyzed_prompts.append(result)
    
    return analyzed_prompts


def load_sharegpt(
    sharegpt_file: str,
    num_workers: int
) -> List[Tuple[str, int]]:
    """
    加载 ShareGPT 数据集并转换为 (prompt, prompt_len) 格式
    """
    print(f"\n=== 加载 ShareGPT 数据集 ===")
    print(f"正在读取: {sharegpt_file}")
    
    try:
        with open(sharegpt_file, "r", encoding="utf-8") as f:
            sharegpt_data = json.load(f)
    except Exception as e:
        print(f"读取 ShareGPT 失败: {e}")
        return []
    
    print(f"ShareGPT 总对话数: {len(sharegpt_data):,}")
    
    # 提取所有用户 prompts
    all_prompts = []
    
    for item in sharegpt_data:
        if "conversations" not in item:
            continue
        
        conversations = item["conversations"]
        
        # 只取第一个用户 turn 作为 prompt
        for turn in conversations:
            if turn.get("from") in {"human", "user"}:
                prompt = turn.get("value", "").strip()
                if prompt:
                    all_prompts.append((prompt, "sharegpt"))
                break  # 只取第一个用户 turn
    
    print(f"ShareGPT 提取的 prompts: {len(all_prompts):,}")
    
    # 并行计算 token 长度
    print("正在计算 ShareGPT prompts 的 token 长度...")
    analyzed_prompts = []
    
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = executor.map(process_item, all_prompts, chunksize=100)
        
        for result in tqdm.tqdm(futures, total=len(all_prompts), desc="分析 ShareGPT"):
            if result:
                analyzed_prompts.append(result)
    
    return analyzed_prompts


def stratified_sampling(
    longbench_pool: List[Tuple[str, int]],
    sharegpt_pool: List[Tuple[str, int]],
    target_size: int,
    target_avg_prompt: int,
    max_prompt_len: int
) -> List[str]:
    """
    分层采样以达到目标平均长度
    """
    print(f"\n=== 分层采样策略 ===")
    
    # 过滤超长样本
    longbench_pool = [(p, l) for p, l in longbench_pool if l <= max_prompt_len]
    sharegpt_pool = [(p, l) for p, l in sharegpt_pool if l <= max_prompt_len]
    
    if not longbench_pool or not sharegpt_pool:
        print("错误: 过滤后的数据池为空")
        return []
    
    # 计算平均长度
    avg_long = np.mean([l for _, l in longbench_pool])
    avg_short = np.mean([l for _, l in sharegpt_pool])
    
    print(f"LongBench 池: {len(longbench_pool):,} 样本, 平均长度: {avg_long:.2f} tokens")
    print(f"ShareGPT 池: {len(sharegpt_pool):,} 样本, 平均长度: {avg_short:.2f} tokens")
    
    # 计算混合比例
    # x * avg_long + (1-x) * avg_short = target_avg
    if avg_long == avg_short:
        print("警告: 两个池的平均长度相同，使用 50:50 混合")
        x = 0.5
    else:
        x = (target_avg_prompt - avg_short) / (avg_long - avg_short)
        x = max(0.0, min(1.0, x))  # 限制在 [0, 1]
    
    num_long = int(target_size * x)
    num_short = target_size - num_long
    
    print(f"\n目标: {target_size} 样本, 平均长度 {target_avg_prompt} tokens")
    print(f"混合比例: {x:.4f}")
    print(f"  - LongBench: {num_long:,} ({num_long/target_size*100:.1f}%)")
    print(f"  - ShareGPT: {num_short:,} ({num_short/target_size*100:.1f}%)")
    
    # 执行采样
    if num_long > len(longbench_pool):
        print(f"警告: LongBench 样本不足，调整为 {len(longbench_pool)}")
        num_long = len(longbench_pool)
        num_short = target_size - num_long
    
    if num_short > len(sharegpt_pool):
        print(f"警告: ShareGPT 样本不足，调整为 {len(sharegpt_pool)}")
        num_short = len(sharegpt_pool)
    
    sampled_long = random.sample(longbench_pool, num_long)
    sampled_short = random.sample(sharegpt_pool, num_short)
    
    # 合并并打乱
    all_sampled = sampled_long + sampled_short
    random.shuffle(all_sampled)
    
    # 验证实际平均长度
    actual_avg = np.mean([l for _, l in all_sampled])
    print(f"\n实际结果:")
    print(f"  总样本数: {len(all_sampled):,}")
    print(f"  实际平均长度: {actual_avg:.2f} tokens")
    print(f"  目标平均长度: {target_avg_prompt} tokens")
    print(f"  偏差: {actual_avg - target_avg_prompt:+.2f} tokens")
    
    # 打印分布统计
    lengths = pd.Series([l for _, l in all_sampled])
    print(f"\n长度分布:")
    print(f"  Min: {lengths.min():,}")
    print(f"  25%: {lengths.quantile(0.25):.0f}")
    print(f"  50%: {lengths.quantile(0.50):.0f}")
    print(f"  75%: {lengths.quantile(0.75):.0f}")
    print(f"  95%: {lengths.quantile(0.95):.0f}")
    print(f"  Max: {lengths.max():,}")
    
    # 返回 prompts
    return [prompt for prompt, _ in all_sampled]


def main():
    parser = argparse.ArgumentParser(
        description="合并 LongBench 和 ShareGPT 为 vLLM Custom 格式 (JSONL)"
    )
    parser.add_argument(
        "--longbench-dir",
        type=str,
        required=True,
        help="LongBench .jsonl 文件所在目录"
    )
    parser.add_argument(
        "--sharegpt-file",
        type=str,
        required=True,
        help="ShareGPT .json 文件路径"
    )
    parser.add_argument(
        "--output-file",
        type=str,
        required=True,
        help="输出的 Custom 格式 .jsonl 文件路径"
    )
    parser.add_argument(
        "--model-id",
        type=str,
        required=True,
        help="用于 tokenization 的模型 ID"
    )
    parser.add_argument(
        "--target-size",
        type=int,
        default=3000,
        help="目标数据集大小（样本数）"
    )
    parser.add_argument(
        "--target-avg-prompt",
        type=int,
        default=6000,
        help="目标平均 prompt 长度（tokens）"
    )
    parser.add_argument(
        "--max-prompt-len",
        type=int,
        default=32000,
        help="最大允许的 prompt 长度（tokens）"
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=os.cpu_count() or 4,
        help="并行处理的进程数"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="随机种子"
    )
    parser.add_argument(
        "--longbench-datasets",
        type=str,
        default=None,
        help="要使用的 LongBench 数据集（逗号分隔），默认使用所有"
    )
    
    args = parser.parse_args()
    
    # 设置随机种子
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    # 确定要使用的 LongBench 数据集
    if args.longbench_datasets:
        datasets_to_use = [d.strip() for d in args.longbench_datasets.split(",")]
    else:
        datasets_to_use = LONGBENCH_DATASETS
    
    print(f"将使用以下 LongBench 数据集: {datasets_to_use}")
    
    # 初始化 tokenizer
    print(f"\n初始化 Tokenizer: {args.model_id}")
    init_worker(args.model_id)
    
    if _tokenizer is None:
        print("错误: Tokenizer 初始化失败")
        return
    
    # 加载数据
    longbench_pool = load_longbench(
        args.longbench_dir,
        datasets_to_use,
        args.num_workers
    )
    
    sharegpt_pool = load_sharegpt(
        args.sharegpt_file,
        args.num_workers
    )
    
    if not longbench_pool or not sharegpt_pool:
        print("错误: 数据加载失败")
        return
    
    # 分层采样
    final_prompts = stratified_sampling(
        longbench_pool,
        sharegpt_pool,
        args.target_size,
        args.target_avg_prompt,
        args.max_prompt_len
    )
    
    if not final_prompts:
        print("错误: 采样失败")
        return
    
    # 写入 Custom 格式 (JSONL)
    print(f"\n=== 写入输出文件 ===")
    print(f"输出文件: {args.output_file}")
    
    try:
        with open(args.output_file, "w", encoding="utf-8") as f:
            for prompt in final_prompts:
                json_line = json.dumps({"prompt": prompt}, ensure_ascii=False)
                f.write(json_line + "\n")
        
        print(f"✓ 成功写入 {len(final_prompts):,} 条数据")
        
        # 验证文件
        print(f"\n验证输出文件...")
        with open(args.output_file, "r", encoding="utf-8") as f:
            lines = f.readlines()
        print(f"✓ 文件包含 {len(lines):,} 行")
        
        # 显示前3个样本
        print(f"\n前3个样本预览:")
        for i, line in enumerate(lines[:3], 1):
            data = json.loads(line)
            prompt_preview = data["prompt"][:100] + "..." if len(data["prompt"]) > 100 else data["prompt"]
            print(f"{i}. {prompt_preview}")
        
    except Exception as e:
        print(f"错误: 写入文件失败 - {e}")


if __name__ == "__main__":
    main()