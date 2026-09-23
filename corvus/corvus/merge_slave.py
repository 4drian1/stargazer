"""从机库幂等合并（merge_slave.py，主侧运行）。

背景：监控机（从机）只采集不抽取（无 Qwen），SQLite 与 raw 原文目录按与本机
**相同的绝对路径布局**（D:\stargazer\corvus）运行。本脚本在主机执行：
1. 把从机 data\raw 原文树 robocopy 合并进本机（同路径布局，raw_path 无需改写）；
2. ATTACH 从机 sqlite，按 raw_text_hash / (instrument_id,event_time) 幂等合并
   raw_news / market_observation / collection_run（raw_id 冲突交给 autoincrement，
   从机 raw_id 不保留，主机抽取只依赖原文与哈希，不依赖从机 id）。

用法（主机）：
    python -m corvus.merge_slave D:\\corvus_slave_backup   # 从机目录（含 corvus.sqlite 与 data\\raw）
"""
import shutil
import sqlite3
import sys
from pathlib import Path

MASTER_DB = Path(__file__).resolve().parent.parent / "data" / "corvus.sqlite"
MASTER_ROOT = MASTER_DB.parent.parent   # 主机 corvus 根（raw_path 改写目标前缀）

RAW_COLS = ("source_id", "title", "url", "raw_text_hash", "published_at",
            "collected_at", "raw_path", "content_license", "mime", "encoding",
            "source_time_raw", "tz_inference", "time_parse_status",
            "detail_status", "detail_mime", "data_status")
MKT_COLS = ("instrument_id", "event_time", "available_at", "open", "high",
            "low", "close", "volume", "source", "raw_hash", "quality_status")
RUN_COLS = ("source_id", "started_at", "finished_at", "status", "fetched_count",
            "inserted_count", "duplicate_count", "failed_count",
            "latest_published_at", "error_summary", "collector_version")


def merge(slave_root):
    slave_root = Path(slave_root).resolve()
    slave_db = slave_root / "data" / "corvus.sqlite"
    if not slave_db.exists():
        raise SystemExit(f"从机库不存在: {slave_db}")
    # 1) 原文树合并（同路径布局 → raw_path 字符串保持有效）
    src_raw = slave_root / "data" / "raw"
    dst_raw = MASTER_DB.parent / "raw"
    if src_raw.exists():
        shutil.copytree(src_raw, dst_raw, dirs_exist_ok=True)
        print(f"原文树已合并: {src_raw} -> {dst_raw}")

    conn = sqlite3.connect(str(MASTER_DB), timeout=120)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys=OFF")
    conn.execute(f"ATTACH DATABASE '{slave_db}' AS slave")
    try:
        # raw_path 前缀改写：从机安装路径 != 主机路径时，把已合并行的
        # raw_path 从从机前缀改写到主机前缀（原文树已按相同相对结构拷入）
        bs = chr(92)
        slave_prefix = str(slave_root).replace("/", bs).rstrip(bs)
        master_prefix = str(MASTER_ROOT).replace("/", bs).rstrip(bs)
        if slave_prefix.lower() != master_prefix.lower():
            n_rw = conn.execute(
                "UPDATE main.raw_news SET raw_path = ? || substr(raw_path, ?) "
                "WHERE raw_path LIKE ? AND raw_path IS NOT NULL",
                (master_prefix, len(slave_prefix) + 1,
                 slave_prefix + bs + "%")).rowcount
            print(f"raw_path 前缀改写: {n_rw} 条")
        # raw_news：按内容哈希去重，raw_id 交由 autoincrement（从机 id 不保留）
        n = conn.execute(
            f"INSERT INTO main.raw_news ({', '.join(RAW_COLS)}) "
            f"SELECT {', '.join(RAW_COLS)} FROM slave.raw_news "
            f"WHERE raw_text_hash NOT IN (SELECT raw_text_hash FROM main.raw_news)"
        ).rowcount
        print(f"raw_news 合并新增: {n}")
        # 镜像列（老库可能缺列，逐个尝试）
        for col, expr in (("body_source", "body_source"), ("body_status", "body_status"),
                          ("body_hash", "body_hash"),
                          ("mirror_match_method", "mirror_match_method"),
                          ("mirror_match_confidence", "mirror_match_confidence")):
            try:
                conn.execute(
                    f"UPDATE main.raw_news SET {col} = "
                    f"(SELECT s.{col} FROM slave.raw_news s "
                    f"WHERE s.raw_text_hash = main.raw_news.raw_text_hash) "
                    f"WHERE {col} IS NULL AND raw_text_hash IN "
                    f"(SELECT raw_text_hash FROM slave.raw_news)")
            except sqlite3.OperationalError:
                pass
        # market_observation：唯一索引 (instrument_id, event_time) → OR IGNORE
        n = conn.execute(
            f"INSERT OR IGNORE INTO main.market_observation ({', '.join(MKT_COLS)}) "
            f"SELECT {', '.join(MKT_COLS)} FROM slave.market_observation"
        ).rowcount
        print(f"market_observation 合并新增: {n}")
        # collection_run 账本：直接追加（审计日志允许重复语义，只追加不覆盖）
        try:
            n = conn.execute(
                f"INSERT INTO main.collection_run ({', '.join(RUN_COLS)}) "
                f"SELECT {', '.join(RUN_COLS)} FROM slave.collection_run"
            ).rowcount
            print(f"collection_run 追加: {n}")
        except sqlite3.OperationalError as e:
            print(f"collection_run 跳过（列差异）: {e}")
        conn.commit()
        conn.execute("DETACH DATABASE slave")
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()
    print("合并完成。下一步在主机跑抽取: python -m corvus.daily_run --mode extract")


def main(argv=None):
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    merge(sys.argv[1])


if __name__ == "__main__":
    main()
