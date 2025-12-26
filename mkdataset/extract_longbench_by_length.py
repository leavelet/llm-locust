#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
从 LongBench 数据集中按 token 长度提取数据，分组保存为 JSON 文件。

每 1K token 区间取 3 条数据，范围 1K-10K（不含 10K）。

用法:
    python extract_longbench_by_length.py

输出:
    datasets/golong/1k.json  (1000-2000 tokens)
    datasets/golong/2k.json  (2000-3000 tokens)
    ...
    datasets/golong/9k.json  (9000-10000 tokens)
"""

import json
import os
import random
from collections import defaultdict
from pathlib import Path

# 配置
LONGBENCH_DIR = Path(__file__).parent.parent / "datasets" / "longbench"
OUTPUT_DIR = Path(__file__).parent.parent / "datasets" / "golong"
SAMPLES_PER_BUCKET = 3
SEED = 42

# 长度区间定义 (1K-10K，每个区间 1000 tokens)
LENGTH_BUCKETS = [
    (1000, 2000, "1k"),
    (2000, 3000, "2k"),
    (3000, 4000, "3k"),
    (4000, 5000, "4k"),
    (5000, 6000, "5k"),
    (6000, 7000, "6k"),
    (7000, 8000, "7k"),
    (8000, 9000, "8k"),
    (9000, 10000, "9k"),
]


def load_longbench_data():
    """加载所有 LongBench 数据集"""
    all_data = []
    
    if not LONGBENCH_DIR.exists():
        print(f"错误: LongBench 目录不存在: {LONGBENCH_DIR}")
        return all_data
    
    # 遍历所有 jsonl 文件
    jsonl_files = list(LONGBENCH_DIR.glob("*.jsonl"))
    print(f"找到 {len(jsonl_files)} 个 JSONL 文件")
    
    for filepath in jsonl_files:
        dataset_name = filepath.stem
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                for line_num, line in enumerate(f, 1):
                    try:
                        item = json.loads(line.strip())
                        # 构造 prompt: context + input
                        context = item.get("context", "") or ""
                        input_text = item.get("input", "") or ""
                        prompt = (context + "\n\n" + input_text).strip()
                        
                        length = item.get("length", 0)
                        
                        if prompt and length > 0:
                            all_data.append({
                                "prompt": prompt,
                                "length": length,
                                "dataset": dataset_name,
                                "input": input_text,
                            })
                    except json.JSONDecodeError:
                        continue
        except Exception as e:
            print(f"读取 {filepath} 失败: {e}")
    
    print(f"总共加载 {len(all_data)} 条数据")
    return all_data


def group_by_length(data):
    """按长度区间分组"""
    buckets = defaultdict(list)
    
    for item in data:
        length = item["length"]
        for min_len, max_len, bucket_name in LENGTH_BUCKETS:
            if min_len <= length < max_len:
                buckets[bucket_name].append(item)
                break
    
    # 打印统计
    print("\n长度分布统计:")
    for min_len, max_len, bucket_name in LENGTH_BUCKETS:
        count = len(buckets[bucket_name])
        print(f"  {bucket_name}: {count} 条 ({min_len}-{max_len} tokens)")
    
    return buckets


def sample_and_save(buckets):
    """从每个区间采样并保存"""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    random.seed(SEED)
    
    print("\n采样并保存:")
    for min_len, max_len, bucket_name in LENGTH_BUCKETS:
        items = buckets[bucket_name]
        
        if len(items) < SAMPLES_PER_BUCKET:
            print(f"  警告: {bucket_name} 只有 {len(items)} 条，不足 {SAMPLES_PER_BUCKET} 条")
            sampled = items
        else:
            sampled = random.sample(items, SAMPLES_PER_BUCKET)
        
        # 保存为 JSON 文件
        output_file = OUTPUT_DIR / f"{bucket_name}.json"
        output_data = [
            {
                "prompt": item["prompt"],
                "length": item["length"],
                "dataset": item["dataset"],
                "input": item["input"],
            }
            for item in sampled
        ]
        
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        
        print(f"  {bucket_name}.json: 保存 {len(sampled)} 条 (长度: {[item['length'] for item in sampled]})")
    
    print(f"\n输出目录: {OUTPUT_DIR}")


def main():
    print("=" * 60)
    print("从 LongBench 提取数据并按长度分组")
    print("=" * 60)
    
    # 加载数据
    data = load_longbench_data()
    if not data:
        return
    
    # 按长度分组
    buckets = group_by_length(data)
    
    # 采样并保存
    sample_and_save(buckets)
    
    print("\n完成!")


if __name__ == "__main__":
    main()
