# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2024 vLLM project contributors
"""
将 LongBench 数据集 (.jsonl) 转换为 vLLM/ShareGPT 格式 (.json)。

用法:
python convert_longbench_to_sharegpt.py \
    --input-dir ./datasets/longbench/ \
    --output-file ./datasets/longbench_converted.json \
    --files "narrativeqa.jsonl qasper.jsonl gov_report.jsonl qmsum.jsonl multi_news.jsonl"
"""

import argparse
import json
import os
import uuid
from typing import List, Dict, Any

import tqdm  # type: ignore

# 推荐选择具有长答案的任务 (QA/摘要)
# 而不是分类任务 (如 lsht, trec)
DEFAULT_FILES_TO_PROCESS = []

longbench_datasets = ["narrativeqa", "qasper", "multifieldqa_en", "multifieldqa_zh", "hotpotqa", "2wikimqa", "musique", \
            "dureader", "gov_report", "qmsum", "multi_news", "vcsum", "trec", "triviaqa", "samsum", "lsht", \
            "passage_count", "passage_retrieval_en", "passage_retrieval_zh", "lcc", "repobench-p"]

for dataset in longbench_datasets:
    DEFAULT_FILES_TO_PROCESS.append(f"{dataset}.jsonl")


def convert_longbench_to_sharegpt(
    input_dir: str, 
    output_file: str,
    files_to_process: List[str]
) -> None:
    
    all_conversations: List[Dict[str, Any]] = []
    print(f"将要处理以下文件: {files_to_process}")

    for filename in tqdm.tqdm(files_to_process, desc="处理文件"):
        filepath = os.path.join(input_dir, filename)
        
        if not os.path.exists(filepath):
            print(f"警告: 文件 {filepath} 不存在, 已跳过。")
            continue

        dataset_name = filename.split('.')[0]
        
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                lines = list(f)  # 一次性读取所有行以便tqdm显示进度

            for i, line in enumerate(tqdm.tqdm(lines, desc=f"  {dataset_name}", leave=False)):
                try:
                    item = json.loads(line)
                except json.JSONDecodeError:
                    print(f"错误: 无法解析 {filename} 第 {i+1} 行。")
                    continue
                
                # 1. 构造 Prompt (User Turn)
                # 结合 'context' 和 'input'。
                # 对于某些任务 (如摘要)，context 可能为空，这是正常的。
                context = item.get("context", "") or ""
                input_text = item.get("input", "") or ""
                
                # 将 context 和 input 组合成一个 prompt
                user_content = (context + "\n\n" + input_text).strip()
                
                if not user_content:
                    # print(f"警告: 跳过 {filename} 第 {i+1} 行 (无 input/context)。")
                    continue

                # 2. 构造 Output (Assistant Turn)
                answers = item.get("answers")
                assistant_content = ""
                
                if isinstance(answers, list) and len(answers) > 0:
                    assistant_content = str(answers[0]) # 取第一个答案
                elif isinstance(answers, str):
                    assistant_content = answers

                # 如果没有答案，则使用用户同意的占位符
                if not assistant_content:
                    assistant_content = "Output placeholder"

                # 3. 构造唯一 ID
                # 优先使用 _id, 否则使用 dataset 和行号
                item_id = item.get("_id") or f"{dataset_name}-{i}"
                
                # 4. 格式化为 ShareGPT
                sharegpt_item = {
                    "id": item_id,
                    "conversations": [
                        {"from": "user", "value": user_content},
                        {"from": "assistant", "value": assistant_content},
                    ]
                }
                all_conversations.append(sharegpt_item)

        except Exception as e:
            print(f"处理文件 {filepath} 时出错: {e}")

    # 5. 写入到单个 JSON 文件
    print(f"\n转换完成。总共 {len(all_conversations):,} 个对话。")
    print(f"正在写入到: {output_file}")
    
    try:
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(all_conversations, f, ensure_ascii=False, indent=2)
        print("写入成功。")
    except Exception as e:
        print(f"错误: 无法写入到 {output_file}。 {e}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="将 LongBench .jsonl 转换为 vLLM ShareGPT .json 格式。"
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default="./datasets/longbench/",
        help="LongBench .jsonl 文件的输入目录。",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default="./datasets/longbench_converted.json",
        help="合并后的 ShareGPT 格式的 JSON 输出文件路径。",
    )
    parser.add_argument(
        "--files",
        type=str,
        default=" ".join(DEFAULT_FILES_TO_PROCESS),
        help=(
            "要处理的 .jsonl 文件列表 (用空格分隔)。"
            "推荐选择 QA 和摘要任务 (如 narrativeqa.jsonl) "
            "以获得更长的回复内容。"
        )
    )

    args = parser.parse_args()
    files_list = args.files.split()
    
    convert_longbench_to_sharegpt(
        args.input_dir, 
        args.output_file,
        files_list
    )

if __name__ == "__main__":
    main()