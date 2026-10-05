"""attach_realization.py — 标签成熟后独立 append realization（与 prediction 物理隔离）。

对 c4_shadow_prediction 中 realization 未齐（T+1/T+3/T+5）的簇：
  读日线（cache→stockdb）→ compute_cluster_labels（backfill 冻结口径）
  → mat_1d/3d/5d 任一成熟（对应交易日 ≤ 日历末日）且非 NULL
  → c4_shadow_realization 幂等 append（先有 prediction 行，顺序约束）。

红线：不回写 prediction；CENSORED_TAIL/标签缺失 标 label_status 不填布尔；
成熟判定以交易日历为准（label_start 后第 h 个交易日存在即成熟）。
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from backfill_pipeline import compute_cluster_labels  # noqa: E402
from _stockdb import DEFAULT_BASE, load_daily, make_opener  # noqa: E402
import pg_store  # noqa: E402

OUT = ROOT / "out"
CACHE = OUT / "daily_cache"
LABEL_VERSION = "c4_formal_v1"
HORIZONS = {1: "T+1", 3: "T+3", 5: "T+5"}


def log(m: str):
    print(f"[attach_realization {dt.datetime.now():%H:%M:%S}] {m}", flush=True)


def load_daily_mixed(code: str, label_start: int, opener) -> pd.DataFrame | None:
    p = CACHE / f"{code}.parquet"
    if p.exists():
        d = pd.read_parquet(p)
        if len(d) and int(label_start) in set(d["date"].astype(int).tolist()):
            return d
    try:
        return load_daily(opener, DEFAULT_BASE, code)
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="只处理前 N 条（调试用）")
    args = ap.parse_args()

    cal = json.loads((OUT / "calendar.json").read_text(encoding="utf-8"))
    max_cal = cal[-1]
    bench_p = OUT / "bench_000300.parquet"
    bench = pd.read_parquet(bench_p) if bench_p.exists() else None

    conn = pg_store.connect()
    pg_store.ensure_schema(conn)
    cur = conn.execute(
        f"SELECT p.event_cluster_id, p.materiality_model_version, p.code, "
        f"p.available_at, p.label_start, "
        f"ARRAY(SELECT label_horizon FROM {pg_store.SCHEMA}.c4_shadow_realization r "
        f" WHERE r.event_cluster_id=p.event_cluster_id "
        f" AND r.materiality_model_version=p.materiality_model_version) AS have "
        f"FROM {pg_store.SCHEMA}.c4_shadow_prediction p ORDER BY p.prediction_ts")
    rows = cur.fetchall()
    if args.limit:
        rows = rows[: args.limit]
    log(f"prediction 行 {len(rows)}，逐条检查 realization 成熟度")

    opener = make_opener()
    t0 = time.time()
    n_ins = n_skip = n_err = 0
    for cid, ver, code, avail_at, label_start, have in rows:
        have = set(have or [])
        need = {h for h in HORIZONS.values()} - have
        if not need or not label_start:
            n_skip += 1
            continue
        try:
            daily = load_daily_mixed(code, int(label_start), opener)
            if daily is None or daily.empty:
                log(f"  {cid} {code}: 无日线，跳过")
                n_skip += 1
                continue
            g = pd.DataFrame([{"event_cluster_id": cid, "code": code,
                               "announce_date": str(avail_at)}])
            lab = compute_cluster_labels(g, daily, cal, bench, max_cal)
            status = lab.get("label_status", "OK")
            pos = {d: i for i, d in enumerate(cal)}
            i = pos.get(int(label_start))
            if i is None:
                n_skip += 1
                continue
            for h, name in HORIZONS.items():
                if name not in need:
                    continue
                matured = (i + h) < len(cal) and cal[i + h] <= max_cal
                mat = lab.get(f"mat_{h}d")
                if not matured or mat is None:
                    continue  # 未成熟或标签不可用：不填 0，等成熟/留 NULL
                r = pg_store.attach_realization(conn, cid, ver, name, bool(mat),
                                                LABEL_VERSION,
                                                "OK" if status == "OK" else status)
                if r == "INSERTED":
                    n_ins += 1
            n_skip += 1
        except Exception as e:
            log(f"  {cid} {code}: ERROR {type(e).__name__}: {str(e)[:100]}")
            n_err += 1
    st = pg_store.stats(conn)
    conn.close()
    log(f"完成: 新 realization {n_ins}，跳过 {n_skip}，错误 {n_err} ({time.time()-t0:.0f}s)"
        f"｜ 库内累计 realization={st['realizations']}")


if __name__ == "__main__":
    main()
