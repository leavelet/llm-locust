# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2024 vLLM project contributors
"""
使用多进程分块处理 ShareGPT JSON 文件，对其进行过滤、分析和分桶。

V2: 增加了 Tokenizer 预缓存 (pre-caching) 步骤，以避免多进程
     并发下载时出现网络错误 (SSLEOFError)。

用法:
python analyze_and_filter_sharegpt_v2.py \
    --input-file sharegpt_original.json \
    --output-file sharegpt_filtered.json \
    --model-id meta-llama/Llama-2-7b-hf \
    --num-workers 16
"""

import argparse
import json
import os
import math
from typing import List, Dict, Any, Optional, Tuple
from concurrent.futures import ProcessPoolExecutor

import pandas as pd  # type: ignore
import tqdm  # type: ignore
from transformers import AutoTokenizer  # type: ignore

# --- 过滤标准 ---
# 1. 128k = 128 * 1024 = 131,072 tokens
MAX_CONTEXT_LEN = 128000
RESERVED_OUTPUT_TOKENS = 1_500
MAX_PROMPT_LEN = MAX_CONTEXT_LEN - RESERVED_OUTPUT_TOKENS  # 129,572 tokens

# 2. 去掉10 token以内的超短请求
MIN_PROMPT_LEN = 10  # 我们将保留 > 10 的, 即从 11 开始

# --- 角色定义 ---
HUMAN_ROLES = {"human", "user"}
ASSISTANT_ROLES = {"gpt", "bing", "chatgpt", "bard", "assistant"}

# --- 分桶定义 (用于采样分析) ---
BINS = [
    0, 
    1001, 
    2001, 
    4001, 
    6001, 
    8001, 
    12001, 
    16001, 
    32001, 
    64001,
    MAX_PROMPT_LEN + 1
]
BIN_LABELS = [
    "0-1k", 
    "1k-2k", 
    "2k-4k", 
    "4k-6k", 
    "6k-8k", 
    "8k-12k", 
    "12k-16k", 
    "16k-32k", 
    "32k-64k", 
    "64k+"
]

# 用于工作进程的全局分词器
_tokenizer: Optional[AutoTokenizer] = None

def init_worker(model_id: str, add_token: Optional[str]) -> None:
    """
    初始化每个工作进程的分词器。
    此时 model_id 对应的文件应该已经被主进程预缓存。
    """
    global _tokenizer
    try:
        _tokenizer = AutoTokenizer.from_pretrained(model_id, 
                                                   trust_remote_code=True)
        if not _tokenizer.pad_token:
            _tokenizer.pad_token = _tokenizer.eos_token
            
        if add_token:
            _tokenizer.add_special_tokens(
                {"additional_special_tokens": [add_token]}
            )
    except Exception as e:
        print(f"工作进程 {os.getpid()} 无法加载 tokenizer: {e}")

def process_item(item: Dict[str, Any]) -> Optional[Tuple[Dict[str, Any], int]]:
    """
    单个工作进程的函数。
    计算一个对话的 prompt tokens，并根据标准进行过滤。
    """
    global _tokenizer
    if _tokenizer is None:
        # 如果分词器未成功初始化，则跳过
        return None

    if "conversations" not in item:
        return None

    conversations = item["conversations"]
    
    current_prompt_tokens = 0

    for turn in conversations:
        if "from" not in turn or "value" not in turn:
            continue
            
        try:
            # 使用 encode 避免自动添加 BOS/EOS token
            # 这里的警告 (Token indices sequence length is longer...) 是无害的，
            # 因为我们稍后会过滤掉这些超长序列。
            num_tokens = len(_tokenizer.encode(turn["value"]))
        except Exception:
            continue  # 跳过无法编码的回合

        role = turn["from"]
        if role in HUMAN_ROLES:
            current_prompt_tokens += num_tokens
        # (我们只关心 prompt 长度)

    # 应用过滤条件
    if (current_prompt_tokens > MIN_PROMPT_LEN and
        current_prompt_tokens <= MAX_PROMPT_LEN):
        # 通过
        return (item, current_prompt_tokens)
    
    # 过滤掉
    return None

def print_stats_and_bins(
    prompt_lengths: List[int],
    total_original_count: int
) -> None:
    """打印过滤后数据集的统计数据和分桶情况"""
    
    filtered_count = len(prompt_lengths)
    print("\n--- 过滤统计 ---")
    print(f"原始数量: {total_original_count:,}")
    print(f"过滤后数量: {filtered_count:,}")
    print(f"已移除: {total_original_count - filtered_count:,} (占比: {(total_original_count - filtered_count) / total_original_count:.2%})")

    if filtered_count == 0:
        print("警告: 所有数据均被过滤，无剩余数据！")
        return
        
    s_prompts = pd.Series(prompt_lengths)

    print("\n--- 过滤后数据集指标 (基于 Tokens) ---")
    print("\n[Prompt (输入) 长度分布]")
    print(s_prompts.describe(percentiles=[.25, .5, .75, .9, .95, .99, .999]).to_string())

    # --- 分桶分析 ---
    print("\n--- 采样分桶分析 (Prompt 长度) ---")
    
    # 使用 pandas.cut 进行分桶
    # right=False 表示 [left, right) 区间
    s_binned = pd.cut(s_prompts, bins=BINS, labels=BIN_LABELS, right=False)
    
    bin_counts = s_binned.value_counts().sort_index()
    
    bin_df = pd.DataFrame({
        "区间": bin_counts.index,
        "数量": bin_counts.values
    })
    
    # 计算百分比
    bin_df["百分比"] = (bin_df["数量"] / filtered_count * 100).map("{:.2f}%".format)
    
    print(bin_df.to_string(index=False))
    print("-" * 30)
    print(f"总计: {bin_df['数量'].sum():,}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="使用多进程分块过滤、分析和分桶 ShareGPT JSON 文件。"
    )
    parser.add_argument(
        "--input-file", 
        required=True,
        help="输入的 ShareGPT 格式的 JSON 文件路径"
    )
    parser.add_argument(
        "--output-file", 
        required=True,
        help="过滤后输出的 JSON 文件路径"
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default="gpt2",
        help="用于分词的 Hugging Face 模型 ID (例如 'meta-llama/Llama-2-7b-hf')",
    )
    parser.add_argument(
        "--add-token",
        type=str,
        default=None,
        help="（可选）为分词器添加额外的特殊 token (例如 <|im_start|>)"
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=os.cpu_count() or 1,
        help="用于处理数据的进程数 (默认为 os.cpu_count())"
    )
    
    args = parser.parse_args()

    # 在主进程中预先加载/缓存 Tokenizer
    try:
        _tokenizer_main = AutoTokenizer.from_pretrained(
            args.model_id, 
            trust_remote_code=True,
            # 设一个超大值以抑制超长序列的警告 (例如 163488 > 128000)
            # 这些序列无论如何都会被我们的 MAX_PROMPT_LEN 过滤器捕获
            model_max_length=1_000_000  
        )
        
        if args.add_token:
            _tokenizer_main.add_special_tokens(
                {"additional_special_tokens": [args.add_token]}
            )
        
        # 我们不需要在主进程中保留这个实例
        del _tokenizer_main 
    except Exception as e:
        print(f"错误: 无法加载或预缓存 Tokenizer '{args.model_id}'。")
        print(f"详细错误: {e}")
        return


    print(f"正在加载输入文件: {args.input_file}")
    try:
        with open(args.input_file, "r", encoding="utf-8") as f:
            sharegpt_data = json.load(f)
    except Exception as e:
        print(f"错误: 无法读取或解析 JSON 文件。{e}")
        return

    if not isinstance(sharegpt_data, list):
        print("错误: JSON 顶层结构不是一个列表 (list)。")
        return

    original_count = len(sharegpt_data)
    print(f"成功加载 {original_count:,} 个对话片段。")
    print(f"过滤条件:")
    print(f"  - Prompt (输入) 长度 > {MIN_PROMPT_LEN} tokens")
    print(f"  - Prompt (输入) 长度 <= {MAX_PROMPT_LEN:,} tokens")

    filtered_items: List[Dict[str, Any]] = []
    prompt_lengths: List[int] = []

    # 传递给工作进程初始化函数的参数
    init_args = (args.model_id, args.add_token)

    with ProcessPoolExecutor(
        max_workers=args.num_workers,
        initializer=init_worker,
        initargs=init_args
    ) as executor:
        
        # 自动分块
        num_chunks = args.num_workers * 4
        chunksize = max(1, math.ceil(original_count / num_chunks))
        
        print(f"使用 {args.num_workers} 个进程, {num_chunks} 个任务块, 每块大小 {chunksize:,}")

        # executor.map 会自动处理分块和结果排序
        results = list(tqdm.tqdm(
            executor.map(process_item, sharegpt_data, chunksize=chunksize), 
            total=original_count, 
            desc="过滤与分词"
        ))

    # 过滤掉返回 None 的无效结果
    filtered_results = [res for res in results if res is not None]

    if not filtered_results:
        print("\n过滤完成。没有数据剩下。")
        return

    # 解包结果
    filtered_items, prompt_lengths = zip(*filtered_results)

    # 打印统计数据和分桶情况
    print_stats_and_bins(list(prompt_lengths), original_count)

    # 写入新的 JSON 文件
    print(f"\n正在写入 {len(filtered_items):,} 条过滤后的数据到: {args.output_file}")
    try:
        with open(args.output_file, "w", encoding="utf-8") as f:
            # list() 转换是必要的，因为 zip object 是一个迭代器
            json.dump(list(filtered_items), f, ensure_ascii=False, indent=2)
        print("写入完成。")
    except Exception as e:
        print(f"错误: 无法写入 JSON 文件。{e}")

if __name__ == "__main__":
    main()