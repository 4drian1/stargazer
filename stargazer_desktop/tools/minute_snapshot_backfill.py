# -*- coding: utf-8 -*-
"""分钟快照 T+1 路径回填。

每日盘后运行一次（或手动）：读取 minute_snapshots/*/index.csv 中
t1_filled=0 的条目，从 stockdb 拉该股票信号日之后最近一个交易日的
1 分钟K，写入对应 JSON 的 t1_bars 字段，并把 index 标记为 1。

用法：
    python tools/minute_snapshot_backfill.py            # 回填所有未处理日期
    python tools/minute_snapshot_backfill.py 20260903   # 只回填指定日期
"""
from __future__ import annotations

import csv
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tools.minute_snapshot import OUT_ROOT, fetch_minute_bars  # noqa: E402


def _next_trading_day_bars(code: str, yyyymmdd: str) -> list[dict]:
    """取信号日之后最近一个有分钟数据的交易日（向后找最多 10 个自然日）。"""
    from datetime import datetime, timedelta
    d = datetime.strptime(yyyymmdd, "%Y%m%d")
    for i in range(1, 11):
        nd = (d + timedelta(days=i)).strftime("%Y%m%d")
        bars = fetch_minute_bars(code, nd)
        if bars:
            return bars
    return []


def backfill_day(day_dir: str) -> tuple[int, int]:
    idx_path = os.path.join(day_dir, "index.csv")
    if not os.path.exists(idx_path):
        return 0, 0
    with open(idx_path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    done = fail = 0
    for row in rows:
        if row.get("t1_filled") == "1":
            continue
        fpath = os.path.join(day_dir, row["file"])
        try:
            with open(fpath, encoding="utf-8") as f:
                payload = json.load(f)
            if not payload.get("t1_bars"):
                bars = _next_trading_day_bars(row["code"], row["date"])
                if not bars:
                    fail += 1
                    continue
                payload["t1_bars"] = bars
                with open(fpath, "w", encoding="utf-8") as f:
                    json.dump(payload, f, ensure_ascii=False)
            row["t1_filled"] = "1"
            done += 1
        except Exception:
            fail += 1
    with open(idx_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["datetime", "date", "code", "sector",
                                          "trigger", "group", "n_bars", "file",
                                          "t1_filled"])
        w.writeheader()
        w.writerows(rows)
    return done, fail


def main() -> None:
    day = sys.argv[1] if len(sys.argv) > 1 else None
    dirs = [os.path.join(OUT_ROOT, day)] if day else sorted(
        d for d in glob.glob(os.path.join(OUT_ROOT, "*")) if os.path.isdir(d))
    total_done = total_fail = 0
    for d in dirs:
        done, fail = backfill_day(d)
        total_done += done
        total_fail += fail
        print(f"[{os.path.basename(d)}] 回填 {done} 条, 失败/无数据 {fail} 条")
    print(f"合计：回填 {total_done}，失败 {total_fail}")


if __name__ == "__main__":
    main()
