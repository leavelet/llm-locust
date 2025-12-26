#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
分析 LongBench 数据集的长度分布，以 5% 为统计区间输出分位数。

用法:
python analyze-longbench-distribution.py \
    --longbench-dir ./datasets/longbench/ \
    --datasets passage_retrieval_zh,trec \
    --model-id meta-llama/Llama-3.1-8B-Instruct
"""

import argparse
import json
import os
from typing import List, Tuple

import numpy as np
import pandas as pd
import tqdm
from transformers import AutoTokenizer


def load_longbench_dataset(filepath: str) -> List[str]:
    """
    加载单个 LongBench 数据集文件，返回 prompts 列表
    """
    prompts = []
    
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            try:
                item = json.loads(line)
                # 构造 prompt: context + input
                context = item.get("context", "") or ""
                input_text = item.get("input", "") or ""
                prompt = (context + "\n\n" + input_text).strip()
                if prompt:
                    prompts.append(prompt)
            except json.JSONDecodeError:
                continue
    
    return prompts


def compute_token_lengths(
    prompts: List[str], 
    tokenizer: AutoTokenizer
) -> List[int]:
    """
    计算所有 prompts 的 token 长度
    """
    lengths = []
    for prompt in tqdm.tqdm(prompts, desc="计算 token 长度"):
        try:
            length = len(tokenizer.encode(prompt))
            lengths.append(length)
        except Exception:
            continue
    return lengths


def print_distribution(
    dataset_name: str, 
    lengths: List[int]
) -> None:
    """
    打印长度分布统计，以 5% 为统计区间
    """
    if not lengths:
        print(f"\n=== {dataset_name} 数据集 ===")
        print("无有效数据")
        return
    
    s = pd.Series(lengths)
    
    print(f"\n{'='*50}")
    print(f"=== {dataset_name} 数据集 ===")
    print(f"{'='*50}")
    print(f"样本数:     {len(lengths):,}")
    print(f"平均长度:   {s.mean():.2f} tokens")
    print(f"中位数:     {s.median():.2f} tokens")
    print(f"标准差:     {s.std():.2f} tokens")
    print(f"最小长度:   {s.min():,} tokens")
    print(f"最大长度:   {s.max():,} tokens")
    
    print(f"\n长度分布 (5% 分位数):")
    print("-" * 30)
    
    # 以 5% 为统计区间：0%, 5%, 10%, ... 95%, 100%
    percentiles = list(range(0, 101, 5))
    for p in percentiles:
        if p == 0:
            value = s.min()
        elif p == 100:
            value = s.max()
        else:
            value = s.quantile(p / 100)
        print(f"  {p:3d}%:  {value:,.0f} tokens")


def main():
    parser = argparse.ArgumentParser(
        description="分析 LongBench 数据集的长度分布"
    )
    parser.add_argument(
        "--longbench-dir",
        type=str,
        default="./datasets/longbench/",
        help="LongBench .jsonl 文件所在目录"
    )
    parser.add_argument(
        "--datasets",
        type=str,
        default="passage_retrieval_zh,trec",
        help="要分析的数据集名称（逗号分隔）"
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default="gpt2",
        help="用于 tokenization 的模型 ID"
    )
    
    args = parser.parse_args()
    
    # 解析数据集列表
    datasets = [d.strip() for d in args.datasets.split(",")]
    
    # 加载 tokenizer
    print(f"正在加载 Tokenizer: {args.model_id}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_id, 
            trust_remote_code=True,
            model_max_length=999999
        )
        if not tokenizer.pad_token:
            tokenizer.pad_token = tokenizer.eos_token
    except Exception as e:
        print(f"错误: 无法加载 Tokenizer - {e}")
        return
    
    print(f"将分析以下数据集: {datasets}")
    
    # 分析每个数据集
    for dataset_name in datasets:
        filepath = os.path.join(args.longbench_dir, f"{dataset_name}.jsonl")
        
        if not os.path.exists(filepath):
            print(f"\n警告: 文件不存在 - {filepath}")
            continue
        
        print(f"\n正在加载: {dataset_name}.jsonl")
        prompts = load_longbench_dataset(filepath)
        print(f"加载了 {len(prompts):,} 个样本")
        
        lengths = compute_token_lengths(prompts, tokenizer)
        print_distribution(dataset_name, lengths)


if __name__ == "__main__":
    main()
