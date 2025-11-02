# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2024 vLLM project contributors
"""
合并 ShareGPT (短) 和 LongBench (长) 数据集，
并进行分层采样以达到目标大小和平均 prompt 长度。

用法:
python create_benchmark_dataset.py \
    --sharegpt-file ./datasets/ShareGPT_filtered.json \
    --longbench-file ./datasets/longbench_converted.json \
    --output-file ./datasets/final_benchmark_dataset_3k_6k.json \
    --model-id meta-llama/Llama-2-7b-hf \
    --target-size 3000 \
    --target-avg-prompt 6000 \
    --min-long-output-len 100 \
    --num-workers 16
"""

import argparse
import json
import math
import os
import random
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List, Dict, Any, Tuple, Optional
MAX_CONTEXT_LEN = 64000

import numpy as np
import pandas as pd  # type: ignore
import tqdm  # type: ignore
from transformers import AutoTokenizer  # type: ignore

# 用于工作进程的全局分词器
_tokenizer: Optional[AutoTokenizer] = None

def init_worker(model_id: str) -> None:
    """初始化每个工作进程的分词器"""
    global _tokenizer
    try:
        _tokenizer = AutoTokenizer.from_pretrained(
            model_id, 
            trust_remote_code=True,
            model_max_length=999999 # 禁用截断警告
        )
    except Exception as e:
        print(f"工作进程 {os.getpid()} 无法加载 tokenizer: {e}")

def analyze_item(item: Dict[str, Any]) -> Optional[Tuple[Dict[str, Any], int, int]]:
    """
    单个工作进程的函数。
    计算一个对话的 prompt 和 output token 长度。
    """
    global _tokenizer
    if _tokenizer is None:
        return None

    try:
        user_turn = item["conversations"][0]["value"]
        assistant_turn = item["conversations"][1]["value"]
        
        prompt_len = len(_tokenizer.encode(user_turn))
        output_len = len(_tokenizer.encode(assistant_turn))
        
        return (item, prompt_len, output_len)
    except Exception:
        # 捕获可能的编码错误或格式错误
        return None

def parallel_analyze(
    items: List[Dict[str, Any]], 
    num_workers: int,
    desc: str
) -> List[Tuple[Dict[str, Any], int, int]]:
    """使用多进程并行分析数据集"""
    
    analyzed_data: List[Tuple[Dict[str, Any], int, int]] = []
    
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        
        num_chunks = num_workers * 4
        chunksize = max(1, math.ceil(len(items) / num_chunks))
        
        futures = executor.map(analyze_item, items, chunksize=chunksize)
        
        progress_bar = tqdm.tqdm(
            futures, 
            total=len(items), 
            desc=f"分析 {desc}"
        )

        for result in progress_bar:
            if result:
                analyzed_data.append(result)
                
    return analyzed_data

def print_final_stats(
    dataset: List[Tuple[Dict[str, Any], int, int]],
    title: str = "最终数据集"
) -> None:
    """打印最终数据集的统计数据"""
    if not dataset:
        print(f"{title} 为空！")
        return

    items, prompt_lengths, output_lengths = zip(*dataset)
    
    s_prompts = pd.Series(prompt_lengths)
    s_outputs = pd.Series(output_lengths)

    print(f"\n--- {title}指标 ---")
    print(f"总对话数: {len(items):,}")

    print("\n[Prompt (输入) 长度]")
    print(f"  平均: {s_prompts.mean():.2f} tokens")
    print(f"  中位数: {s_prompts.median():.2f} tokens")
    print(f"  Min: {s_prompts.min():,}")
    print(f"  Max: {s_prompts.max():,}")
    print("\n  [Prompt 分布 (百分位数)]")
    print(s_prompts.describe(percentiles=[.1, .25, .5, .75, .9, .99]).to_string())

    print("\n[Output (回复) 长度]")
    print(f"  平均: {s_outputs.mean():.2f} tokens")
    print(f"  中位数: {s_outputs.median():.2f} tokens")
    print(f"  Min: {s_outputs.min():,}")
    print(f"  Max: {s_outputs.max():,}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="合并 ShareGPT 和 LongBench 并进行分层采样。"
    )
    # ... (添加所有参数)
    parser.add_argument("--sharegpt-file", required=True)
    parser.add_argument("--longbench-file", required=True)
    parser.add_argument("--output-file", required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--target-size", type=int, default=3000)
    parser.add_argument("--target-avg-prompt", type=int, default=6000)
    parser.add_argument("--min-long-output-len", type=int, default=100,
                        help="从 LongBench 采样时, 要求的最小回复 token 长度。")
    parser.add_argument("--num-workers", type=int, default=os.cpu_count() or 1)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    random.seed(args.seed)
    
    # --- 1. 预缓存 Tokenizer ---
    print(f"正在预缓存 Tokenizer '{args.model_id}'...")
    try:
        init_worker(args.model_id)
        global _tokenizer
        if _tokenizer is None:
            raise ValueError("Tokenizer 初始化失败")
        print("Tokenizer 预缓存成功。")
    except Exception as e:
        print(f"错误: 无法加载 Tokenizer。{e}")
        return
        
    # --- 2. 加载和分析 ShareGPT (短 Prompts) ---
    print(f"正在加载 {args.sharegpt_file}...")
    with open(args.sharegpt_file, "r", encoding="utf-8") as f:
        sharegpt_data = json.load(f)
    
    analyzed_sharegpt = parallel_analyze(
        sharegpt_data, args.num_workers, "ShareGPT"
    )
    del sharegpt_data # 释放内存
    
    if not analyzed_sharegpt:
        print("错误: ShareGPT 数据分析失败或为空。")
        return
        
    avg_sharegpt_prompt = np.mean([p_len for _, p_len, _ in analyzed_sharegpt])
    print(f"ShareGPT 池: {len(analyzed_sharegpt):,} 个样本, "
          f"平均 Prompt: {avg_sharegpt_prompt:.2f} tokens")

    # --- 3. 加载和分析 LongBench (长 Prompts) ---
    print(f"正在加载 {args.longbench_file}...")
    with open(args.longbench_file, "r", encoding="utf-8") as f:
        longbench_data = json.load(f)

    analyzed_longbench = parallel_analyze(
        longbench_data, args.num_workers, "LongBench"
    )
    del longbench_data # 释放内存

    if not analyzed_longbench:
        print("错误: LongBench 数据分析失败或为空。")
        return

    # --- 4. 过滤 LongBench 以满足长输出需求 ---
    longbench_pool = [
        (item, p_len, o_len) for item, p_len, o_len in analyzed_longbench
        if o_len >= args.min_long_output_len and o_len < MAX_CONTEXT_LEN
    ]
    del analyzed_longbench # 释放内存

    if not longbench_pool:
        print(f"错误: 没有 LongBench 样本满足最小输出长度 > "
              f"{args.min_long_output_len} tokens。")
        return
        
    avg_longbench_prompt = np.mean([p_len for _, p_len, _ in longbench_pool])
    print(f"LongBench 池 (已过滤长输出): {len(longbench_pool):,} 个样本, "
          f"平均 Prompt: {avg_longbench_prompt:.2f} tokens")

    # --- 5. 计算混合比例 ---
    # 求解: x * avg_long + (1-x) * avg_short = target
    try:
        x = ( (args.target_avg_prompt - avg_sharegpt_prompt) / 
              (avg_longbench_prompt - avg_sharegpt_prompt) )
    except ZeroDivisionError:
        print("错误: LongBench 和 ShareGPT 的平均长度相同。无法采样。")
        return

    if not (0 <= x <= 1):
        print(f"警告: 无法达到 {args.target_avg_prompt} 的平均值。")
        if x > 1:
            print("目标过高, 将全使用 LongBench。")
            x = 1.0
        else:
            print("目标过低, 将全使用 ShareGPT。")
            x = 0.0

    num_longbench = int(args.target_size * x)
    num_sharegpt = args.target_size - num_longbench

    print(f"\n--- 采样计划 (目标: {args.target_size} 样本, {args.target_avg_prompt} avg) ---")
    print(f"混合比例 (x): {x:.4f}")
    print(f"从 LongBench 抽取: {num_longbench} 个")
    print(f"从 ShareGPT 抽取: {num_sharegpt} 个")

    # --- 6. 执行采样 ---
    sharegpt_pool = analyzed_sharegpt # 重命名
    
    if num_longbench > len(longbench_pool):
        print(f"警告: LongBench 池中样本不足 ({len(longbench_pool)}), "
              f"只能抽取 {len(longbench_pool)} 个。")
        num_longbench = len(longbench_pool)
        # 重新计算 short 数量以保持总大小
        num_sharegpt = args.target_size - num_longbench

    if num_sharegpt > len(sharegpt_pool):
        print(f"警告: ShareGPT 池中样本不足 ({len(sharegpt_pool)}), "
              f"只能抽取 {len(sharegpt_pool)} 个。")
        num_sharegpt = len(sharegpt_pool)

    # 执行最终采样
    sampled_long = random.sample(longbench_pool, num_longbench)
    sampled_short = random.sample(sharegpt_pool, num_sharegpt)

    final_analyzed_dataset = sampled_long + sampled_short
    random.shuffle(final_analyzed_dataset)

    # --- 7. 保存并验证 ---
    final_dataset_items = [item for item, _, _ in final_analyzed_dataset]

    print(f"\n正在写入 {len(final_dataset_items):,} 条数据到: {args.output_file}")
    try:
        with open(args.output_file, "w", encoding="utf-8") as f:
            json.dump(final_dataset_items, f, ensure_ascii=False, indent=2)
        print("写入成功。")
    except Exception as e:
        print(f"错误: 无法写入 JSON 文件。{e}")

    # 打印最终验证统计数据
    print_final_stats(final_analyzed_dataset, "最终基准数据集")


if __name__ == "__main__":
    main()