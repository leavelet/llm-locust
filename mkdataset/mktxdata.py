#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
将 LongBench V1 (.jsonl), LongBench V2 (.json) 和 ShareGPT (.json) 合并为 vLLM Custom 格式 (.jsonl)
并通过直方图拟合（Histogram Matching）达到特定的长度分布。

包含功能：
1. 自动分桶统计。
2. 针对长文本不足的区间，自动从更长的文本中截断填充 (Cascade Truncation)。
3. 支持 "256 means <= 256" 的直方图定义。
4. 支持 LongBench V2 数据集以补充超长文本。

用法:
python merge_custom_distribution.py \
    --longbench-dir ./datasets/longbench/ \
    --longbench-v2-file ./datasets/longbenchv2.json \
    --sharegpt-file ./datasets/ShareGPT_V3_unfiltered_cleaned_split.json \
    --output-file ./datasets/benchmark_distribution_matched.jsonl \
    --model-id meta-llama/Llama-3.1-8B-Instruct \
    --target-size 4000 \
    --num-workers 16
"""

import argparse
import json
import os
import random
import bisect
import math
from concurrent.futures import ProcessPoolExecutor
from typing import List, Dict, Tuple, Optional
from collections import defaultdict

import numpy as np
import tqdm
from transformers import AutoTokenizer

# 全局 tokenizer for multiprocessing
_tokenizer_worker: Optional[AutoTokenizer] = None

# LongBench 数据集列表
LONGBENCH_DATASETS = [
    "narrativeqa", "qasper", "multifieldqa_en", "hotpotqa", "2wikimqa", 
    "musique", "gov_report", "qmsum", "multi_news", "trec", "triviaqa", 
    "samsum", "passage_retrieval_en", "lcc", "repobench-p"
]

# ==========================================
# 目标分布数据 (Bin End -> Weight)
# 使用 'total_data_count' 作为权重
# ==========================================
TARGET_DISTRIBUTION_RAW = [
    (256, 2272), (512, 1321), (768, 1071), (1024, 1105), (1280, 841), (1536, 843),
    (1792, 814), (2048, 592), (2304, 559), (2560, 427), (2816, 451), (3072, 386),
    (3328, 338), (3584, 347), (3840, 264), (4096, 293), (4352, 260), (4608, 293),
    (4864, 282), (5120, 252), (5376, 259), (5632, 242), (5888, 247), (6144, 226),
    (6400, 257), (6656, 267), (6912, 255), (7168, 234), (7424, 200), (7680, 196),
    (7936, 200), (8192, 255), (8448, 232), (8704, 213), (8960, 205), (9216, 158),
    (9472, 206), (9728, 187), (9984, 204), (10240, 210), (10496, 202), (10752, 163),
    (11008, 171), (11264, 137), (11520, 156), (11776, 132), (12032, 127), (12288, 128),
    (12544, 128), (12800, 130), (13056, 114), (13312, 70), (13568, 112), (13824, 103),
    (14080, 89), (14336, 98), (14592, 89), (14848, 66), (15104, 92), (15360, 75),
    (15616, 68), (15872, 68), (16128, 81), (16384, 55), (16640, 70), (16896, 64),
    (17152, 52), (17408, 50), (17664, 53), (17920, 45), (18176, 29), (18432, 33),
    (18688, 58), (18944, 42), (19200, 38), (19456, 56), (19712, 38), (19968, 23),
    (20224, 23), (20480, 45), (20736, 26), (20992, 34), (21248, 15), (21504, 22),
    (21760, 37), (22016, 27), (22272, 30), (22528, 37), (22784, 26), (23040, 24),
    (23296, 26), (23552, 39), (23808, 21), (24064, 21), (24320, 26), (24576, 31),
    (24832, 17), (25088, 14), (25344, 21), (25600, 18), (25856, 16), (26112, 15),
    (26368, 16), (26624, 20), (26880, 21), (27136, 25), (27392, 19), (27648, 12),
    (27904, 15), (28160, 10), (28416, 22), (28672, 16), (28928, 20), (29184, 19),
    (29440, 23), (29696, 26), (29952, 18), (30208, 10), (30464, 11), (30720, 4),
    (30976, 10), (31232, 15), (31488, 9), (31744, 10), (32000, 20), (32256, 8),
    (32512, 5), (32768, 14), (33024, 3), (33280, 11), (33536, 13), (33792, 4),
    (34048, 2), (34304, 5), (34560, 10), (34816, 2), (35072, 3), (35328, 5),
    (35584, 2), (35840, 13), (36096, 4), (36352, 5), (36864, 3), (37120, 4),
    (37376, 7), (37632, 5), (37888, 2), (38144, 10), (38400, 4), (38656, 2),
    (38912, 4), (39168, 2), (39424, 2), (39680, 3), (39936, 4), (40192, 2),
    (41216, 2), (41472, 4), (41728, 4), (41984, 2), (42240, 1), (42752, 4),
    (43776, 6), (44800, 4), (45312, 2), (45568, 2), (46336, 2), (46848, 4),
    (47360, 6), (48384, 4), (48640, 2), (48896, 2), (49152, 2), (49408, 2),
    (49664, 2), (50176, 2), (50944, 3), (51712, 2), (51968, 1), (52224, 3),
    (52480, 1), (54272, 2), (55296, 2), (55808, 2), (58368, 2), (58880, 2),
    (60416, 2), (60672, 2), (65792, 1), (71424, 2), (79104, 2), (81920, 2),
    (90368, 2), (99328, 2)
]

def init_worker(model_id: str) -> None:
    """初始化每个工作进程的分词器"""
    global _tokenizer_worker
    try:
        _tokenizer_worker = AutoTokenizer.from_pretrained(
            model_id, 
            trust_remote_code=True,
            model_max_length=999999
        )
        if not _tokenizer_worker.pad_token:
            _tokenizer_worker.pad_token = _tokenizer_worker.eos_token
    except Exception as e:
        print(f"工作进程初始化 tokenizer 失败: {e}")

def process_item(args: Tuple[str, str]) -> Optional[Tuple[str, int]]:
    """处理单个样本，返回 (prompt, prompt_len)"""
    global _tokenizer_worker
    if _tokenizer_worker is None:
        return None
    
    prompt_text, source_type = args
    try:
        # 只计算长度，不返回 heavy 的 input_ids
        tokens = _tokenizer_worker.encode(prompt_text, add_special_tokens=False)
        prompt_len = len(tokens)
        return (prompt_text, prompt_len)
    except Exception:
        return None

def truncate_prompt_text(
    tokenizer: AutoTokenizer, 
    prompt: str, 
    target_length: int
) -> str:
    """
    将 prompt 截断到 target_length。
    为了效率，先进行字符级估算截断，再进行 Token 级精确截断。
    """
    # 1. 快速字符级估算 (假设 1 token ~= 3-4 chars，保守取 2.5 避免截断过多)
    estimated_chars = int(target_length * 6) 
    if len(prompt) > estimated_chars:
        prompt = prompt[:estimated_chars]
        
    # 2. 精确 Token 级截断
    tokens = tokenizer.encode(prompt, add_special_tokens=False)
    if len(tokens) > target_length:
        tokens = tokens[:target_length]
        return tokenizer.decode(tokens)
    
    return prompt

def load_longbench(longbench_dir: str, datasets: List[str], num_workers: int) -> List[Tuple[str, int]]:
    """加载 LongBench V1 数据集 (.jsonl)"""
    print(f"\n=== 加载 LongBench V1 数据集 ===")
    all_prompts = []
    
    for dataset_name in datasets:
        filepath = os.path.join(longbench_dir, f"{dataset_name}.jsonl")
        if not os.path.exists(filepath):
            continue
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        item = json.loads(line)
                        context = item.get("context", "") or ""
                        input_text = item.get("input", "") or ""
                        prompt = (context + "\n\n" + input_text).strip()
                        if prompt:
                            all_prompts.append((prompt, "longbench_v1"))
                    except json.JSONDecodeError:
                        continue
        except Exception as e:
            print(f"读取 {filepath} 失败: {e}")
            
    print(f"LongBench V1 原始数据: {len(all_prompts):,} 条")
    
    analyzed_prompts = []
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = executor.map(process_item, all_prompts, chunksize=100)
        for result in tqdm.tqdm(futures, total=len(all_prompts), desc="Tokenizing LongBench V1"):
            if result:
                analyzed_prompts.append(result)
    return analyzed_prompts

def load_longbench_v2(filepath: str, num_workers: int) -> List[Tuple[str, int]]:
    """加载 LongBench V2 数据集 (.json)"""
    print(f"\n=== 加载 LongBench V2 数据集 ===")
    if not filepath or not os.path.exists(filepath):
        print(f"提示: 未指定 LongBench V2 文件或文件不存在: {filepath}")
        return []

    try:
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print(f"读取 LongBench V2 失败: {e}")
        return []
    
    all_prompts = []
    for item in data:
        # LongBench V2: context + question
        context = item.get("context", "") or ""
        question = item.get("question", "") or ""
        prompt = (context + "\n\n" + question).strip()
        if prompt:
            all_prompts.append((prompt, "longbench_v2"))
            
    print(f"LongBench V2 原始数据: {len(all_prompts):,} 条")
    
    analyzed_prompts = []
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = executor.map(process_item, all_prompts, chunksize=100)
        for result in tqdm.tqdm(futures, total=len(all_prompts), desc="Tokenizing LongBench V2"):
            if result:
                analyzed_prompts.append(result)
    return analyzed_prompts

def load_sharegpt(sharegpt_file: str, num_workers: int) -> List[Tuple[str, int]]:
    """加载 ShareGPT 数据集"""
    print(f"\n=== 加载 ShareGPT 数据集 ===")
    try:
        with open(sharegpt_file, "r", encoding="utf-8") as f:
            sharegpt_data = json.load(f)
    except Exception as e:
        print(f"读取 ShareGPT 失败: {e}")
        return []

    all_prompts = []
    for item in sharegpt_data:
        if "conversations" not in item: continue
        for turn in item["conversations"]:
            if turn.get("from") in {"human", "user"}:
                prompt = turn.get("value", "").strip()
                if prompt:
                    all_prompts.append((prompt, "sharegpt"))
                break 

    print(f"ShareGPT 原始数据: {len(all_prompts):,} 条")
    
    analyzed_prompts = []
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = executor.map(process_item, all_prompts, chunksize=100)
        for result in tqdm.tqdm(futures, total=len(all_prompts), desc="Tokenizing ShareGPT"):
            if result:
                analyzed_prompts.append(result)
    return analyzed_prompts

def distribution_sampling_cascade(
    pool: List[Tuple[str, int]],
    target_size: int,
    model_id: str,
    seed: int
) -> List[str]:
    """
    基于目标直方图分布进行采样，使用“向下级联截断”(Cascade Truncation) 策略。
    从最大的 bucket 开始处理，多余的或者原本更长的数据可以被截断放入较小的 bucket。
    """
    print(f"\n=== 执行分布匹配采样 (Target Size: {target_size}) ===")
    
    # 1. 准备目标分布数据
    bin_bounds = [x[0] for x in TARGET_DISTRIBUTION_RAW]
    bin_weights = [x[1] for x in TARGET_DISTRIBUTION_RAW]
    
    total_weight = sum(bin_weights)
    bin_probs = [w / total_weight for w in bin_weights]
    target_counts = [int(p * target_size) for p in bin_probs]
    
    # 补齐误差
    diff = target_size - sum(target_counts)
    if diff > 0:
        max_idx = np.argmax(bin_weights)
        target_counts[max_idx] += diff
        
    print(f"目标分布桶数量: {len(bin_bounds)}")
    
    # 2. 将原始数据进行自然分桶 (Natural Assignment)
    # pool_buckets: { bin_idx: [ (prompt, len), ... ] }
    pool_buckets = defaultdict(list)
    overflow_count = 0
    
    # 用于存储比最大 bin 还要大的数据
    global_overflow_pool = []
    
    for prompt, length in tqdm.tqdm(pool, desc="数据初始分桶"):
        idx = bisect.bisect_left(bin_bounds, length)
        if idx < len(bin_bounds):
            pool_buckets[idx].append((prompt, length))
        else:
            # 超过最大 bin 的数据，放入 global_overflow_pool，供最大 bin 使用
            global_overflow_pool.append((prompt, length))
            overflow_count += 1
            
    print(f"初始分桶完成。最大 bin 范围外数据: {overflow_count} (将作为截断候选)")
    
    # 3. 初始化主进程 Tokenizer (用于截断)
    print("初始化主进程 Tokenizer 用于截断操作...")
    main_tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    
    final_dataset_entries = [] # List[(prompt, len)]
    
    # 4. 从大到小遍历 Bucket (Cascade Down)
    current_overflow_pool = global_overflow_pool
    
    # 倒序遍历索引
    for idx in range(len(bin_bounds) - 1, -1, -1):
        target_n = target_counts[idx]
        natural_data = pool_buckets[idx]
        upper_bound = bin_bounds[idx]
        
        selected_for_this_bin = []
        
        # A. 优先取 Natural Data
        # 策略修改：为了最大化平均长度，在 Bin 内部优先选择长度较长（接近 Upper Bound）的样本
        # 而不是随机选取。这能显著提升最终的平均 Token 数。
        natural_data.sort(key=lambda x: x[1], reverse=True)
        
        if len(natural_data) >= target_n:
            # 数据充足，取最长的那部分，剩下的加入 overflow 传给下级
            selected_for_this_bin.extend(natural_data[:target_n])
            # 剩余的传给下级 (依然是比下一级 Bin 大的数据)
            current_overflow_pool.extend(natural_data[target_n:])
        else:
            # Natural 数据不够，全取
            selected_for_this_bin.extend(natural_data)
            needed = target_n - len(natural_data)
            
            # B. 从 Overflow Pool 中补足 (需要截断)
            random.shuffle(current_overflow_pool)
            
            if len(current_overflow_pool) >= needed:
                trunc_candidates = current_overflow_pool[:needed]
                current_overflow_pool = current_overflow_pool[needed:]
                
                # 执行截断
                for p_text, p_len in trunc_candidates:
                    truncated_text = truncate_prompt_text(main_tokenizer, p_text, upper_bound)
                    selected_for_this_bin.append((truncated_text, upper_bound))
            else:
                # 依然不够 (理论上 ShareGPT 很庞大，小桶不会缺；LongBench 在大桶可能会缺)
                for p_text, p_len in current_overflow_pool:
                    truncated_text = truncate_prompt_text(main_tokenizer, p_text, upper_bound)
                    selected_for_this_bin.append((truncated_text, upper_bound))
                current_overflow_pool = [] # 清空
                
                print(f"提示: Bin <={upper_bound} 即使截断后仍不足 (缺 {needed - len(current_overflow_pool)})")
        
        # 将本 Bin 选中的数据加入最终集
        final_dataset_entries.extend(selected_for_this_bin)
    
    # 5. 统计与验证
    final_prompts = [x[0] for x in final_dataset_entries]
    final_lens = [x[1] for x in final_dataset_entries] 
    
    print(f"\n采样结果统计:")
    print(f"  目标数量: {target_size}")
    print(f"  实际数量: {len(final_prompts)}")
    print(f"  平均长度: {np.mean(final_lens):.2f}")
    print(f"  最大长度: {np.max(final_lens)}")
    
    return final_prompts

def main():
    parser = argparse.ArgumentParser(description="合并数据并按特定分布采样")
    parser.add_argument("--longbench-dir", type=str, required=True)
    parser.add_argument("--longbench-v2-file", type=str, default=None, help="LongBench V2 .json 文件路径")
    parser.add_argument("--sharegpt-file", type=str, required=True)
    parser.add_argument("--output-file", type=str, required=True)
    parser.add_argument("--model-id", type=str, required=True)
    parser.add_argument("--target-size", type=int, default=3000)
    parser.add_argument("--num-workers", type=int, default=os.cpu_count() or 4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--longbench-datasets", type=str, default=None)
    
    args = parser.parse_args()
    
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    datasets_to_use = [d.strip() for d in args.longbench_datasets.split(",")] if args.longbench_datasets else LONGBENCH_DATASETS
    
    print(f"初始化 Worker Tokenizer: {args.model_id}")
    init_worker(args.model_id)
    
    # 1. 加载数据
    pool_long = load_longbench(args.longbench_dir, datasets_to_use, args.num_workers)
    pool_short = load_sharegpt(args.sharegpt_file, args.num_workers)
    
    pool_long_v2 = []
    if args.longbench_v2_file:
        pool_long_v2 = load_longbench_v2(args.longbench_v2_file, args.num_workers)
    
    full_pool = pool_long + pool_short + pool_long_v2
    print(f"\n总可用数据池: {len(full_pool):,} 条")
    
    if not full_pool:
        print("错误: 没有加载到任何数据")
        return

    # 2. 执行级联采样
    final_prompts = distribution_sampling_cascade(
        full_pool,
        args.target_size,
        args.model_id, # 传入 model_id 以便在主进程加载 tokenizer
        args.seed
    )
    
    # 3. 写入文件
    print(f"\n写入输出文件: {args.output_file}")
    with open(args.output_file, "w", encoding="utf-8") as f:
        for prompt in final_prompts:
            f.write(json.dumps({"prompt": prompt}, ensure_ascii=False) + "\n")
            
    print("完成!")

if __name__ == "__main__":
    main()