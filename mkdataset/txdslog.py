import csv
import io
import json
import logging
import os
import sqlite3

from tz import count_tokens

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def merge_dataset():
    dataset = open("data/dataset.jsonl", "w")
    files = os.listdir("./data/caddy")
    for file in files:
        with open(os.path.join("./data/caddy", file), "r") as f:
            for line in f:
                data = json.loads(line)
                dataset.write(json.dumps(data, ensure_ascii=False) + "\n")


def insert_sqlite():
    conn = sqlite3.connect("data/dataset.db")
    c = conn.cursor()
    c.execute(
        """
        CREATE TABLE IF NOT EXISTS dataset (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts REAL NOT NULL,
            bytes_read INTEGER,
            duration REAL,
            size INTEGER,
            status INTEGER,
            resp_headers JSON CHECK(json_valid(resp_headers)),
            req_json_body JSON CHECK(json_valid(req_json_body))
        )
        """
    )
    with open("data/dataset.jsonl", "r") as f:
        for line in f:
            try:
                row = json.loads(line)
                ts = float(row.get("ts"))
                bytes_read = row.get("bytes_read")
                duration = row.get("duration")
                size = row.get("size")
                status = row.get("status")
                resp_headers = row.get("resp_headers")
                req_raw = row.get("req_json_body")
                if req_raw is not None:
                    try:
                        req_obj = json.loads(req_raw)
                        req_json = json.dumps(req_obj, ensure_ascii=False)
                    except Exception:
                        req_json = req_raw
                else:
                    req_json = None
                c.execute(
                    "INSERT INTO dataset (ts, bytes_read, duration, size, status, resp_headers, req_json_body) VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (
                        ts,
                        bytes_read,
                        duration,
                        size,
                        status,
                        json.dumps(resp_headers, ensure_ascii=False) if resp_headers is not None else None,
                        req_json,
                    ),
                )
            except Exception:
                import traceback
                traceback.print_exc()
                print(line)
    conn.commit()
    conn.close()
    
import hashlib

xsd=open('x.jsonl','w',encoding='utf8')
k={}
def extract_messages_to_csv(req_json_body: str) -> str | None:
    """
    从 req_json_body 中提取 messages 数组，并转换为 CSV 格式字符串。

    Args:
        req_json_body: JSON 格式的请求体字符串

    Returns:
        CSV 格式的字符串，或 None（解析失败时）
    """
    try:
        data = json.loads(req_json_body)
        messages = data.get("messages", [])

        if not messages:
            return None

        # 处理 messages，只保留 role 和 content
        processed_messages = []
        for i, msg in enumerate(messages):
            role = msg.get("role", "")
            content = msg.get("content", "")

            # 对第一条 message 的 content 去除 UUID 前缀（36个字符）
            if i == 0 and len(content) >= 36:
                content = content[36:]

            processed_messages.append({"role": role, "content": content})
        jjj = json.dumps(processed_messages)
        jjj_md5 = hashlib.md5(jjj.encode('utf-8')).hexdigest()
        if jjj_md5 not in k:
            xsd.write(f'{jjj}\n')
            k[jjj_md5]=1
        
        # 转换为 CSV 格式字符串（无表头）
        output = io.StringIO()
        writer = csv.writer(output)
        for msg in processed_messages:
            writer.writerow([msg["role"], msg["content"]])

        return output.getvalue()

    except json.JSONDecodeError as e:
        logger.warning(f"JSON 解析失败: {e}")
        return None
    except Exception as e:
        logger.warning(f"处理 messages 时出错: {e}")
        return None


def process_message_count(db_path: str = "data/dataset.db"):
    """
    从 dataset 表中查询所有记录，提取并转换 messages，
    统计相同数据出现的次数，存储到 message_count 表。

    Args:
        db_path: SQLite 数据库文件路径
    """
    logger.info(f"开始处理数据库: {db_path}")

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # 创建 message_count 表（如果不存在）
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS message_count (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            message TEXT UNIQUE NOT NULL,
            data_count INTEGER NOT NULL DEFAULT 1
        )
    """)
    logger.info("message_count 表已创建或已存在")

    # 查询所有 dataset 记录
    cursor.execute("SELECT id, req_json_body FROM dataset WHERE req_json_body IS NOT NULL")
    rows = cursor.fetchall()
    total_count = len(rows)
    logger.info(f"共查询到 {total_count} 条记录")

    # 处理每条记录
    processed_count = 0
    error_count = 0
    inserted_count = 0
    updated_count = 0

    for _, req_json_body in rows:
        processed_count += 1

        if processed_count % 1000 == 0:
            logger.info(f"处理进度: {processed_count}/{total_count}")

        # 提取并转换 messages
        csv_str = extract_messages_to_csv(req_json_body)

        if csv_str is None:
            error_count += 1
            continue

        # 使用 UPSERT 逻辑：如果存在则更新 data_count，否则插入新记录
        cursor.execute("""
            INSERT INTO message_count (message, data_count)
            VALUES (?, 1)
            ON CONFLICT(message) DO UPDATE SET
                data_count = data_count + 1
        """, (csv_str,))

        # 判断是插入还是更新
        if cursor.rowcount == 1:
            # 检查是否是新插入（通过 changes() 无法区分，这里简化处理）
            cursor.execute("SELECT data_count FROM message_count WHERE message = ?", (csv_str,))
            count = cursor.fetchone()[0]
            if count == 1:
                inserted_count += 1
            else:
                updated_count += 1

    conn.commit()

    # 统计最终结果
    cursor.execute("SELECT COUNT(*) FROM message_count")
    unique_count = cursor.fetchone()[0]

    cursor.execute("SELECT SUM(data_count) FROM message_count")
    total_data_count = cursor.fetchone()[0] or 0

    conn.close()

    logger.info(f"处理完成!")
    logger.info(f"  - 总记录数: {total_count}")
    logger.info(f"  - 成功处理: {processed_count - error_count}")
    logger.info(f"  - 解析失败: {error_count}")
    logger.info(f"  - 唯一消息数: {unique_count}")
    logger.info(f"  - 总计数: {total_data_count}")


def update_token_count(db_path: str = "data/dataset.db", tokenizer_path: str = "./tokenizer.json"):
    """
    为 message_count 表中的每条记录计算 token 数量并更新 token_count 列。

    Args:
        db_path: SQLite 数据库文件路径
        tokenizer_path: tokenizer.json 文件路径
    """
    logger.info(f"开始更新 token_count，数据库: {db_path}")

    # 检查 tokenizer 文件是否存在
    if not os.path.exists(tokenizer_path):
        logger.error(f"tokenizer 文件不存在: {tokenizer_path}")
        return

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # 检查 message_count 表是否存在
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='message_count'")
    if cursor.fetchone() is None:
        logger.error("message_count 表不存在，请先运行 process_message_count()")
        conn.close()
        return

    # 添加 token_count 列（如果不存在）
    try:
        cursor.execute("ALTER TABLE message_count ADD COLUMN token_count INTEGER")
        logger.info("已添加 token_count 列")
    except sqlite3.OperationalError as e:
        if "duplicate column name" in str(e).lower():
            logger.info("token_count 列已存在")
        else:
            logger.error(f"添加 token_count 列失败: {e}")
            conn.close()
            return

    # 查询所有记录
    cursor.execute("SELECT id, message FROM message_count")
    rows = cursor.fetchall()
    total_count = len(rows)
    logger.info(f"共查询到 {total_count} 条记录")

    # 处理每条记录
    processed_count = 0
    error_count = 0

    for row_id, message in rows:
        processed_count += 1

        if processed_count % 1000 == 0:
            logger.info(f"处理进度: {processed_count}/{total_count}")

        try:
            # 计算 token 数量
            token_count = count_tokens(message, tokenizer_path)

            if token_count < 0:
                # count_tokens 返回 -1 表示出错
                error_count += 1
                logger.warning(f"记录 ID {row_id} 计算 token 失败")
                continue

            # 更新记录
            cursor.execute(
                "UPDATE message_count SET token_count = ? WHERE id = ?",
                (token_count, row_id)
            )

        except Exception as e:
            error_count += 1
            logger.warning(f"记录 ID {row_id} 处理失败: {e}")

    conn.commit()

    # 统计结果
    cursor.execute("SELECT COUNT(*) FROM message_count WHERE token_count IS NOT NULL")
    updated_count = cursor.fetchone()[0]

    cursor.execute("SELECT SUM(token_count) FROM message_count WHERE token_count IS NOT NULL")
    total_tokens = cursor.fetchone()[0] or 0

    cursor.execute("SELECT AVG(token_count) FROM message_count WHERE token_count IS NOT NULL")
    avg_tokens = cursor.fetchone()[0] or 0

    conn.close()

    logger.info("处理完成!")
    logger.info(f"  - 总记录数: {total_count}")
    logger.info(f"  - 成功更新: {updated_count}")
    logger.info(f"  - 处理失败: {error_count}")
    logger.info(f"  - 总 token 数: {total_tokens}")
    logger.info(f"  - 平均 token 数: {avg_tokens:.2f}")


def analyze_token_distribution(
    db_path: str = "data/dataset.db",
    output_path: str = "data/token_distribution.csv",
    step: int = 256
):
    """
    分析 message_count 表中消息的 token 长度分布情况，并生成 CSV 文件。

    Args:
        db_path: SQLite 数据库文件路径
        output_path: 输出 CSV 文件路径
        step: 分组步长，默认 256
    """
    logger.info(f"开始分析 token 分布，数据库: {db_path}")
    logger.info(f"分组步长: {step}")

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # 检查 message_count 表是否存在
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='message_count'")
    if cursor.fetchone() is None:
        logger.error("message_count 表不存在，请先运行 process_message_count()")
        conn.close()
        return

    # 检查 token_count 列是否有数据
    cursor.execute("SELECT COUNT(*) FROM message_count WHERE token_count IS NOT NULL")
    valid_count = cursor.fetchone()[0]
    if valid_count == 0:
        logger.error("token_count 列没有数据，请先运行 update_token_count()")
        conn.close()
        return

    logger.info(f"有效记录数: {valid_count}")

    # 使用 SQL 进行分组统计
    # (token_count / step) * step 计算区间起始值
    cursor.execute(f"""
        SELECT
            (token_count / {step}) * {step} AS range_start,
            (token_count / {step}) * {step} + {step - 1} AS range_end,
            COUNT(*) AS message_count,
            SUM(data_count) AS total_data_count
        FROM message_count
        WHERE token_count IS NOT NULL
        GROUP BY (token_count / {step})
        ORDER BY range_start
    """)

    rows = cursor.fetchall()
    conn.close()

    logger.info(f"共生成 {len(rows)} 个区间")

    # 写入 CSV 文件
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        # 写入表头
        writer.writerow(["range_start", "range_end", "message_count", "total_data_count"])
        # 写入数据
        for row in rows:
            writer.writerow(row)

    logger.info(f"CSV 文件已保存: {output_path}")

    # 输出结果摘要
    total_messages = sum(row[2] for row in rows)
    total_data = sum(row[3] for row in rows)

    logger.info("分布统计摘要:")
    logger.info(f"  - 区间数量: {len(rows)}")
    logger.info(f"  - 总消息数: {total_messages}")
    logger.info(f"  - 总 data_count: {total_data}")

    # 显示前 10 个区间
    logger.info("前 10 个区间分布:")
    for row in rows[:10]:
        range_start, range_end, msg_count, data_count = row
        logger.info(f"  [{range_start:>6} - {range_end:>6}]: 消息数={msg_count:>6}, data_count={data_count:>8}")

    if len(rows) > 10:
        logger.info(f"  ... 共 {len(rows)} 个区间，详见 {output_path}")


def main():
    # merge_dataset()
    # clean_dataset()
    # insert_sqlite()
    process_message_count()
    # update_token_count()
    # analyze_token_distribution()


if __name__ == "__main__":
    main()
