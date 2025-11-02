# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2024 vLLM project contributors
"""
分析 ShareGPT 格式的 JSON 文件并报告各项指标。

用法:
python display_sharegpt_metrics.py <输入文件.json> --model-id <tokenizer_model_name>
"""

import argparse
import json
from typing import List, Dict, Any

import matplotlib.pyplot as plt
import pandas as pd  # type: ignore
import tqdm  # type: ignore
from transformers import AutoTokenizer  # type: ignore

# 定义 ShareGPT 格式中的角色
HUMAN_ROLES = {"human", "user"}
ASSISTANT_ROLES = {"gpt", "bing", "chatgpt", "bard", "assistant"}


def analyze_dataset(
    input_file: str, tokenizer: AutoTokenizer, plot_file: str
) -> None:
    """
    加载、处理 ShareGPT JSON 文件并打印指标。
    """
    print(f"使用 Tokenizer: {tokenizer.name_or_path}")
    
    # 1. 加载 JSON 文件
    print(f"正在读取文件: {input_file}")
    try:
        with open(input_file, "r", encoding="utf-8") as f:
            sharegpt_data = json.load(f)
    except Exception as e:
        print(f"错误: 无法读取或解析 JSON 文件。{e}")
        return

    if not isinstance(sharegpt_data, list):
        print("错误: JSON 顶层结构不是一个列表 (list)。")
        return

    print(f"成功加载 {len(sharegpt_data):,} 个对话片段。")

    # 2. 遍历和分词
    all_prompt_lengths: List[int] = []
    all_reply_lengths: List[int] = []
    all_total_lengths: List[int] = []

    print("正在处理对话并计算 Token 长度...")
    for item in tqdm.tqdm(sharegpt_data):
        if "conversations" not in item:
            print(f"警告: 跳过 item {item.get('id', 'N/A')}, 缺少 'conversations' 键。")
            continue

        conversations = item["conversations"]
        
        current_prompt_tokens = 0
        current_reply_tokens = 0

        for turn in conversations:
            if "from" not in turn or "value" not in turn:
                print(f"警告: 跳过 item {item.get('id', 'N/A')} 中的一个无效 turn。")
                continue
                
            try:
                # 使用 encode 而不是 __call__ 来避免自动添加 BOS/EOS 特殊 token
                num_tokens = len(tokenizer.encode(turn["value"]))
            except Exception as e:
                print(f"警告: Tokenizer 编码失败 (item {item.get('id', 'N/A')})。跳过此 turn。错误: {e}")
                continue

            role = turn["from"]
            if role in HUMAN_ROLES:
                current_prompt_tokens += num_tokens
            elif role in ASSISTANT_ROLES:
                current_reply_tokens += num_tokens
            # (忽略 "system" 角色或其他未知角色)

        # 存储这个对话的总长度
        all_prompt_lengths.append(current_prompt_tokens)
        all_reply_lengths.append(current_reply_tokens)
        all_total_lengths.append(current_prompt_tokens + current_reply_tokens)

    if not all_total_lengths:
        print("错误: 未找到有效对话数据。")
        return

    # 3. 计算和显示指标
    print("\n--- 数据集指标 (基于 Tokens) ---")
    
    # 转换为 Pandas Series 以便轻松获取统计数据
    s_prompts = pd.Series(all_prompt_lengths)
    s_replies = pd.Series(all_reply_lengths)
    s_totals = pd.Series(all_total_lengths)

    print(f"\n总对话数: {len(all_total_lengths):,}")

    # 指标 1: 平均长度
    print("\n[平均长度]")
    print(f"  平均 Prompt (输入) 长度: {s_prompts.mean():.2f} tokens")
    print(f"  平均 Reply (回复) 长度: {s_replies.mean():.2f} tokens")
    print(f"  平均 Total (总) 长度:   {s_totals.mean():.2f} tokens")

    # 指标 2: 最长长度
    print("\n[最长长度]")
    print(f"  最长 Prompt (输入) 长度: {s_prompts.max():,} tokens")
    print(f"  最长 Reply (回复) 长度: {s_replies.max():,} tokens")
    print(f"  最长 Total (总) 长度:   {s_totals.max():,} tokens")
    
    # 额外：分布情况
    print("\n[Prompt (输入) 长度分布]")
    print(s_prompts.describe(percentiles=[.25, .5, .75, .9, .95, .99]).to_string())

    # 4. 指标 3: 绘制 Prompt 长度分布图
    print(f"\n正在生成 Prompt 长度分布图 (保存至: {plot_file})...")
    try:
        plt.figure(figsize=(12, 6))
        # 使用对数刻度Y轴，因为长尾分布可能很明显
        plt.hist(s_prompts, bins=100, edgecolor='black', log=True)
        plt.title(f'Prompt 长度分布 (Tokenizer: {tokenizer.name_or_path})')
        plt.xlabel('Prompt 长度 (Tokens)')
        plt.ylabel('频次 (对数刻度)')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        # 添加均值和中位数的垂直线
        plt.axvline(s_prompts.mean(), color='red', linestyle='dashed', linewidth=2)
        plt.axvline(s_prompts.median(), color='orange', linestyle='dashed', linewidth=2)
        plt.legend({
            f'平均值: {s_prompts.mean():.2f}': 'red',
            f'中位数: {s_prompts.median():.2f}': 'orange',
            '分布': 'blue'
        })
        
        plt.savefig(plot_file)
        print(f"图表已保存到 {plot_file}")
    except Exception as e:
        print(f"错误: 无法生成图表。{e}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="分析 ShareGPT JSON 文件的指标。"
    )
    parser.add_argument(
        "input_file", 
        help="输入的 ShareGPT 格式的 JSON 文件路径"
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default="gpt2",
        help="用于分词的 Hugging Face CausalLM 模型 ID (例如 'meta-llama/Llama-2-7b-hf')",
    )
    parser.add_argument(
        "--plot-file",
        type=str,
        default="prompt_length_distribution.png",
        help="输出的 Prompt 长度分布图的文件名",
    )
    parser.add_argument(
        "--add-token",
        type=str,
        default=None,
        help="（可选）为分词器添加额外的特殊 token (例如 <|im_start|>)"
    )

    args = parser.parse_args()

    # 加载分词器
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
        
        if not tokenizer.pad_token:
            tokenizer.pad_token = tokenizer.eos_token
            
        if args.add_token:
            print(f"正在向分词器添加特殊 token: {args.add_token}")
            tokenizer.add_special_tokens({"additional_special_tokens": [args.add_token]})
            
    except Exception as e:
        print(f"错误: 无法加载 Tokenizer '{args.model_id}'。请确保已登录 HF (如果需要) 并且模型名称正确。")
        print(f"详细错误: {e}")
        return

    analyze_dataset(args.input_file, tokenizer, args.plot_file)


if __name__ == "__main__":
    main()