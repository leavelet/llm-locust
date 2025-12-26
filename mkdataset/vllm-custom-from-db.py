#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
从 MySQL/SQLite 数据库的 dataset 表提取 messages,
转换为 vLLM Custom 格式 ({"prompt": "..."})。

用法:
python mkdataset/vllm-custom-from-db.py \
    --db-path datasets/dataset.db \
    --output-file datasets/vllm_custom_dataset.jsonl \
    --model-id meta-llama/Llama-3.1-8B-Instruct \
    --max-prompt-len 32000 \
    --target-size 10000
"""

import argparse
import hashlib
import json
import logging
import os
import random
import sqlite3
from concurrent.futures import ProcessPoolExecutor
from typing import List, Dict, Optional, Tuple

import numpy as np
import tqdm
from transformers import AutoTokenizer

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 全局 tokenizer (用于多进程)
_tokenizer: Optional[AutoTokenizer] = None


def init_worker(model_id: str) -> None:
    """初始化每个工作进程的 tokenizer"""
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
        logger.error(f"工作进程初始化 tokenizer 失败: {e}")


def process_prompt_item(prompt: str) -> Optional[Tuple[str, int]]:
    """
    处理单个 prompt,计算 token 长度。
    
    Args:
        prompt: prompt 文本
        
    Returns:
        (prompt, token_length) 元组,或 None(失败时)
    """
    global _tokenizer
    if _tokenizer is None:
        return None
    
    try:
        token_len = len(_tokenizer.encode(prompt))
        return (prompt, token_len)
    except Exception:
        return None


def extract_prompt_from_messages(
    req_json_body: str,
    remove_uuid_prefix: bool = True
) -> Optional[str]:
    """
    从 req_json_body 中提取 messages 并拼接为单个 prompt。
    
    Args:
        req_json_body: JSON 格式的请求体字符串
        remove_uuid_prefix: 是否去除第一个 user message 的 UUID 前缀(前36字符)
        
    Returns:
        拼接后的 prompt 字符串,或 None(解析失败时)
    """
    try:
        data = json.loads(req_json_body)
        messages = data.get("messages", [])
        
        if not messages:
            return None
        
        # 拼接所有消息内容
        prompt_parts = []
        
        for i, msg in enumerate(messages):
            role = msg.get("role", "")
            content = msg.get("content", "")
            
            # 第一个 message 去除 UUID 前缀(不管是什么 role)
            if i == 0 and remove_uuid_prefix:
                if len(content) >= 36:
                    content = content[36:]
            
            # 拼接: role: content
            if content:
                prompt_parts.append(f"{role}: {content}")
        
        if not prompt_parts:
            return None
        
        # 用换行连接
        prompt = "\n".join(prompt_parts)
        return prompt.strip()
        
    except json.JSONDecodeError as e:
        logger.debug(f"JSON 解析失败: {e}")
        return None
    except Exception as e:
        logger.debug(f"处理 messages 时出错: {e}")
        return None


def load_prompts_from_db(
    db_path: str,
    remove_uuid_prefix: bool = True
) -> List[str]:
    """
    从数据库加载所有 prompts 并去重。
    
    Args:
        db_path: SQLite 数据库文件路径
        remove_uuid_prefix: 是否去除 UUID 前缀
        
    Returns:
        去重后的 prompts 列表
    """
    logger.info(f"正在从数据库加载数据: {db_path}")
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # 查询所有 req_json_body 非空的记录
    cursor.execute("SELECT req_json_body FROM dataset WHERE req_json_body IS NOT NULL")
    rows = cursor.fetchall()
    conn.close()
    
    total_count = len(rows)
    logger.info(f"数据库中共有 {total_count:,} 条记录")
    
    # 提取 prompts 并去重
    prompts_set = {}  # {md5_hash: prompt}
    error_count = 0
    
    logger.info("正在提取和去重 prompts...")
    for row in tqdm.tqdm(rows, desc="处理记录"):
        req_json_body = row[0]
        
        prompt = extract_prompt_from_messages(req_json_body, remove_uuid_prefix)
        
        if prompt is None:
            error_count += 1
            continue
        
        # MD5 去重
        prompt_hash = hashlib.md5(prompt.encode('utf-8')).hexdigest()
        if prompt_hash not in prompts_set:
            prompts_set[prompt_hash] = prompt
    
    unique_prompts = list(prompts_set.values())
    
    logger.info(f"提取完成:")
    logger.info(f"  - 总记录数: {total_count:,}")
    logger.info(f"  - 解析失败: {error_count:,}")
    logger.info(f"  - 有效 prompts: {total_count - error_count:,}")
    logger.info(f"  - 去重后: {len(unique_prompts):,}")
    
    return unique_prompts


def calculate_token_lengths(
    prompts: List[str],
    model_id: str,
    num_workers: int
) -> List[Tuple[str, int]]:
    """
    使用多进程并行计算所有 prompts 的 token 长度。
    
    Args:
        prompts: prompts 列表
        model_id: tokenizer 模型 ID
        num_workers: 并行进程数
        
    Returns:
        (prompt, token_length) 元组列表
    """
    logger.info(f"正在使用 {num_workers} 个进程并行计算 token 长度...")
    
    prompts_with_length = []
    
    # 使用 ProcessPoolExecutor 并行处理
    with ProcessPoolExecutor(
        max_workers=num_workers,
        initializer=init_worker,
        initargs=(model_id,)
    ) as executor:
        futures = executor.map(process_prompt_item, prompts, chunksize=100)
        
        for result in tqdm.tqdm(futures, total=len(prompts), desc="计算 token 长度"):
            if result:
                prompts_with_length.append(result)
    
    error_count = len(prompts) - len(prompts_with_length)
    if error_count > 0:
        logger.warning(f"有 {error_count} 个 prompts 计算 token 长度失败")
    
    return prompts_with_length


def filter_and_sample(
    prompts_with_length: List[Tuple[str, int]],
    min_prompt_len: int,
    max_prompt_len: int,
    target_size: Optional[int] = None
) -> List[str]:
    """
    过滤和采样 prompts。
    
    Args:
        prompts_with_length: (prompt, token_length) 元组列表
        min_prompt_len: 最小 token 长度
        max_prompt_len: 最大 token 长度
        target_size: 目标样本数量(None 表示不限制)
        
    Returns:
        最终的 prompts 列表
    """
    logger.info(f"正在过滤 prompts (长度范围: {min_prompt_len} - {max_prompt_len})...")
    
    # 过滤
    filtered = [
        (p, l) for p, l in prompts_with_length
        if min_prompt_len <= l <= max_prompt_len
    ]
    
    logger.info(f"过滤后剩余: {len(filtered):,} 个 prompts")
    
    if len(filtered) == 0:
        logger.warning("过滤后没有剩余样本!")
        return []
    
    # 采样
    if target_size and len(filtered) > target_size:
        logger.info(f"正在随机采样 {target_size:,} 个样本...")
        filtered = random.sample(filtered, target_size)
    
    # 打乱顺序
    random.shuffle(filtered)
    
    # 统计信息
    lengths = [l for _, l in filtered]
    logger.info(f"\n最终数据集统计:")
    logger.info(f"  样本数: {len(filtered):,}")
    logger.info(f"  平均长度: {np.mean(lengths):.2f} tokens")
    logger.info(f"  中位数长度: {np.median(lengths):.2f} tokens")
    logger.info(f"  最小长度: {min(lengths):,} tokens")
    logger.info(f"  最大长度: {max(lengths):,} tokens")
    logger.info(f"  25% 分位: {np.percentile(lengths, 25):.0f} tokens")
    logger.info(f"  75% 分位: {np.percentile(lengths, 75):.0f} tokens")
    logger.info(f"  95% 分位: {np.percentile(lengths, 95):.0f} tokens")
    
    return [p for p, _ in filtered]


def write_custom_format(
    prompts: List[str],
    output_file: str
) -> None:
    """
    写入 vLLM Custom 格式 (JSONL)。
    
    Args:
        prompts: prompts 列表
        output_file: 输出文件路径
    """
    logger.info(f"正在写入输出文件: {output_file}")
    
    try:
        with open(output_file, "w", encoding="utf-8") as f:
            for prompt in prompts:
                json_line = json.dumps({"prompt": prompt}, ensure_ascii=False)
                f.write(json_line + "\n")
        
        logger.info(f"✓ 成功写入 {len(prompts):,} 条数据")
        
        # 验证文件
        logger.info(f"验证输出文件...")
        with open(output_file, "r", encoding="utf-8") as f:
            lines = f.readlines()
        logger.info(f"✓ 文件包含 {len(lines):,} 行")
        
        # 显示前3个样本
        logger.info(f"\n前3个样本预览:")
        for i, line in enumerate(lines[:3], 1):
            data = json.loads(line)
            prompt_preview = data["prompt"][:150] + "..." if len(data["prompt"]) > 150 else data["prompt"]
            logger.info(f"{i}. {prompt_preview}")
        
    except Exception as e:
        logger.error(f"写入文件失败: {e}")
        raise


def main():
    parser = argparse.ArgumentParser(
        description="从数据库提取 messages 并转换为 vLLM Custom 格式 (JSONL)"
    )
    parser.add_argument(
        "--db-path",
        type=str,
        required=True,
        help="SQLite 数据库文件路径"
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
        default=None,
        help="用于 tokenization 的模型 ID (如果不指定则不计算 token 长度)"
    )
    parser.add_argument(
        "--min-prompt-len",
        type=int,
        default=1,
        help="最小允许的 prompt 长度(tokens),默认 1"
    )
    parser.add_argument(
        "--max-prompt-len",
        type=int,
        default=999999,
        help="最大允许的 prompt 长度(tokens),默认无限制"
    )
    parser.add_argument(
        "--target-size",
        type=int,
        default=None,
        help="目标数据集大小(样本数),默认不限制"
    )
    parser.add_argument(
        "--no-remove-uuid",
        action="store_true",
        help="不去除第一个 user message 的 UUID 前缀"
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=os.cpu_count() or 4,
        help="并行处理的进程数,默认为 CPU 核心数"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="随机种子,默认 42"
    )
    
    args = parser.parse_args()
    
    # 设置随机种子
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    # 检查数据库文件
    if not os.path.exists(args.db_path):
        logger.error(f"数据库文件不存在: {args.db_path}")
        return
    
    # 1. 从数据库加载 prompts
    prompts = load_prompts_from_db(
        args.db_path,
        remove_uuid_prefix=not args.no_remove_uuid
    )
    
    if not prompts:
        logger.error("没有提取到有效的 prompts")
        return
    
    # 2. 计算 token 长度(如果指定了 model_id)
    if args.model_id:
        logger.info(f"初始化 Tokenizer: {args.model_id}")
        prompts_with_length = calculate_token_lengths(
            prompts,
            args.model_id,
            args.num_workers
        )
        
        if not prompts_with_length:
            logger.error("计算 token 长度失败")
            return
        
        # 3. 过滤和采样
        final_prompts = filter_and_sample(
            prompts_with_length,
            args.min_prompt_len,
            args.max_prompt_len,
            args.target_size
        )
    else:
        logger.info("未指定 model_id,跳过 token 长度计算和过滤")
        # 如果指定了 target_size,仍然进行采样
        if args.target_size and len(prompts) > args.target_size:
            logger.info(f"正在随机采样 {args.target_size:,} 个样本...")
            final_prompts = random.sample(prompts, args.target_size)
            random.shuffle(final_prompts)
        else:
            final_prompts = prompts
            random.shuffle(final_prompts)
        
        logger.info(f"最终样本数: {len(final_prompts):,}")
    
    if not final_prompts:
        logger.error("没有最终的 prompts 输出")
        return
    
    # 4. 写入 Custom 格式
    write_custom_format(final_prompts, args.output_file)
    
    logger.info("\n✓ 转换完成!")


if __name__ == "__main__":
    main()
