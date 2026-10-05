"""formal_score.py — C4 收盘后正式评分链（路线 A 的 T+1 收盘段）。

时序（开发路线 #12.5 第④步）：
  T 盘后   : daily_ingest.py 落 events_{T}.jsonl（事实证据，无概率）
  T+1 收盘后: 本脚本对 T 的事件簇正式打分：
      补齐 label_start(T+1) 当日派生量（gap_return / volume_shock / sigma250，
      公式照抄 backfill_pipeline.compute_cluster_labels 冻结口径）
      → scoring_domain 资格门禁（禁标签侧字段）
      → score_event（v1.0.0 冻结模型，parity/hash 强制校验）
      → c4_shadow_prediction 幂等落库（永不 UPDATE）
  标签成熟后: attach_realization.py 独立 append realization（T+1/T+3/T+5）

健康指标（工程层，非模型闸门）逐事件落 out/formal_chain/health_{T}.jsonl：
  latency_ms / action / reason / missing_warnings / feature_nan_frac。

红线：不改模型/特征/阈值；被拒事件显式记录原因，不静默丢。
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from backfill_pipeline import load_calendar  # noqa: E402
from _stockdb import DEFAULT_BASE, load_daily, make_opener  # noqa: E402
import pg_store  # noqa: E402

EVENTS_DIR = ROOT / "out" / "daily_events"
CHAIN_DIR = ROOT / "out" / "formal_chain"
CACHE = ROOT / "out" / "daily_cache"
LABEL_VERSION = "c4_formal_v1"


def log(m: str):
    print(f"[formal_score {dt.datetime.now():%H:%M:%S}] {m}", flush=True)


def load_daily_mixed(code: str, label_start: int, opener) -> pd.DataFrame | None:
    """cache 优先；cache 不含 label_start 则走 stockdb（不回填 cache）。"""
    p = CACHE / f"{code}.parquet"
    if p.exists():
        d = pd.read_parquet(p)
        if len(d) and int(label_start) in set(d["date"].astype(int).tolist()):
            return d
    try:
        return load_daily(opener, DEFAULT_BASE, code)
    except Exception:
        return None


def market_fields(daily: pd.DataFrame, label_start: int) -> dict:
    """label_start 当日派生量（照抄冻结口径，见 compute_cluster_labels）。"""
    daily = daily.copy()
    daily["date"] = daily["date"].astype(int)
    pos = {d: i for i, d in enumerate(daily["date"].tolist())}
    if label_start not in pos:
        return {"gap_return": None, "volume_shock": None, "sigma250": None}
    idx = pos[label_start]
    row = daily.iloc[idx]
    pc = float(row["pre_close"]) if not pd.isna(row.get("pre_close")) else None
    gap = (float(row["open"]) / pc - 1) if pc else None
    prev20 = daily.iloc[max(0, idx - 20): idx]
    vm = float(pd.to_numeric(prev20["volume"], errors="coerce").mean()) \
        if len(prev20) >= 5 and "volume" in prev20.columns else None
    vshock = float(row["volume"]) / vm if vm else None
    prev = daily.iloc[max(0, idx - 250): idx]
    rets = pd.to_numeric(prev.get("pct_chg"), errors="coerce").dropna() / 100.0 \
        if "pct_chg" in prev.columns else pd.Series(dtype=float)
    sigma = float(rets.std()) if len(rets) >= 120 else None
    return {"gap_return": gap, "volume_shock": vshock, "sigma250": sigma}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", required=True, help="公告日 T（YYYYMMDD）= 对哪天的 events 打分")
    ap.add_argument("--as-of", default=None,
                    help="评分时点日，默认 = 各簇 label_start（T+1 收盘语义）")
    ap.add_argument("--out", default=str(CHAIN_DIR))
    args = ap.parse_args()

    from c4_shadow import score_event  # 延迟加载：先确保 PG/数据就绪

    ev_path = EVENTS_DIR / f"events_{args.date}.jsonl"
    if not ev_path.exists():
        raise SystemExit(f"无 {ev_path}（先跑 daily_ingest.py --date {args.date}）")
    events = [json.loads(l) for l in ev_path.open(encoding="utf-8")]
    log(f"date={args.date} 事件簇 {len(events)}")

    opener = make_opener()
    conn = pg_store.connect()
    pg_store.ensure_schema(conn)

    CHAIN_DIR.mkdir(parents=True, exist_ok=True)
    health = CHAIN_DIR / f"health_{args.date}.jsonl"
    hf = health.open("a", encoding="utf-8")
    t0 = time.time()
    n_scored = n_rejected = 0
    for ev in events:
        t1 = time.perf_counter()
        code = ev["code"]
        label_start = ev.get("label_start")
        action, reason, snap = "NOT_SCORED", "", None
        try:
            if not label_start:
                raise ValueError("label_start 缺失（日历未覆盖/无未来交易日），无法正式评分")
            daily = load_daily_mixed(code, int(label_start), opener)
            mf = market_fields(daily, int(label_start)) if daily is not None \
                else {"gap_return": None, "volume_shock": None, "sigma250": None}
            mf = {k: (np.nan if v is None else v) for k, v in mf.items()}
            rec = {
                "event_cluster_id": ev["event_cluster_id"],
                "code": code,
                "title": (ev.get("titles") or [""])[0],
                "announce_date": ev["announce_date"],
                "label_start": label_start,
                "n_announcements_in_cluster": ev["n_announcements_in_cluster"],
                "sigma250": mf["sigma250"],
                "gap_return": mf["gap_return"],
                "volume_shock": mf["volume_shock"],
                "category": ev.get("categories_all", [ev.get("category")])[0],
                "tier": ev["tier"],
                "attribution_quality": "HEAVY" if ev.get("heavy_flag")
                else ("MODERATE" if (ev.get("overlap_count") or 0) >= 1 else "CLEAN"),
            }
            as_of = int(args.as_of) if args.as_of else int(label_start)
            provider = (lambda c, _d=daily: _d) if daily is not None else None
            snap, diag = score_event(rec, as_of_date=as_of, daily_provider=provider)
            if diag["eligible"]:
                record = snap.to_record()
                record.update({
                    "prediction_ts": dt.datetime.now().isoformat(timespec="seconds"),
                    "code": code,
                    "label_start": label_start,
                    "evidence_ids": [f"daily_events/{args.date}/{ev['event_cluster_id']}"],
                })
                action = pg_store.upsert_prediction(conn, record)
                n_scored += 1
            else:
                action, reason = "REJECTED", ";".join(diag["problems"])
                n_rejected += 1
        except Exception as e:
            action, reason = "ERROR", f"{type(e).__name__}: {str(e)[:120]}"
            n_rejected += 1
        hf.write(json.dumps({
            "event_cluster_id": ev["event_cluster_id"], "code": code,
            "action": action, "reason": reason,
            "p_materiality": getattr(snap, "p_materiality", None),
            "latency_ms": round((time.perf_counter() - t1) * 1000, 1),
            "missing_warnings": getattr(snap, "missing_warnings", None) or [],
        }, ensure_ascii=False) + "\n")
    hf.close()
    st = pg_store.stats(conn)
    conn.close()
    log(f"完成: SCORED={n_scored} REJECTED/ERROR={n_rejected} "
        f"({time.time()-t0:.0f}s) ｜ 库内累计 prediction={st['predictions']} "
        f"realization={st['realizations']}")
    log(f"健康日志: {health}")


if __name__ == "__main__":
    main()
