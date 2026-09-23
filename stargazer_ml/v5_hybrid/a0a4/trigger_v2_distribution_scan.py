"""trigger_v2_distribution_scan.py — Step2: TIMING-TRIGGER-v2 触发分布 (不训练).

复用 a14_universe.parquet 的 zone 几何 / setup / atr_ref, 对每候选全天空跑 run_state_engine_v2,
取 FIRST MODEL_SCORE_REQUEST 的 enter_ts 作为 S, 统计四类指标 (用户 Step2 口径):

  T1 时间: 09:30 / 09:31-10:00 / 10:01-11:30 / 13:00+ ; median trigger time
  T2 路径: DIP_RECOVERY / DIP_RECLAIM / BOTTOM_RECOVERY / BOTTOM_RECLAIM / BOTTOM_STABILIZATION
  T3 拒绝: NEVER_TOUCH / STRUCTURAL_BREAKDOWN / TOUCH_NO_RECOVERY / TOUCH_NO_STABILIZATION / TAIL_CENSORED
  T4 Setup: BOTTOM / DIP ; dip_stage: PRE_PULLBACK / PULLBACK_ACTIVE / NEAR_ZONE

硬断言: 09:30 MODEL_SCORE_REQUEST 数必须 = 0 (否则实现错误, 直接非零退出).
输出: datasets/trigger_v2_distribution.json (不训练, 不写母表).
"""
from __future__ import annotations
import json
import sys
import time
import shutil
import argparse
from pathlib import Path
from collections import Counter, defaultdict

import numpy as np
import pandas as pd

try:
    import psutil
except Exception:  # noqa
    psutil = None

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
import a0a4_config as C  # noqa

sys.path.insert(0, str(C.PREFLIGHT))
from zone_state_engine import Zone  # noqa (冻结: 仅类型)
from zone_state_engine_v2 import run_state_engine_v2  # noqa (v2)
import latr_builder as L  # noqa (load_minute / load_adjustment_chain)
import trading_calendar as TC  # noqa


def _bucket(ts):
    if ts is None:
        return "NONE"
    hh = (ts // 10000) % 100
    mm = (ts // 100) % 100
    if hh == 9 and mm == 30:
        return "09:30"
    if (hh == 9 and mm >= 31) or (hh == 10 and mm == 0):
        return "09:31-10:00"
    if (hh == 10 and mm >= 1) or (hh == 11 and mm <= 30):
        return "10:01-11:30"
    if hh >= 13:
        return "13:00+"
    return "other"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=None, help="最多前 N 个候选 (smoke test)")
    ap.add_argument("--stride", type=int, default=None, help="跨全表等距抽样")
    ap.add_argument("--candidates", default=str(C.DATASETS / "a14_universe.parquet"))
    args = ap.parse_args()
    t0 = time.time()
    print(f"[v2scan] start n=? src={args.candidates}", flush=True)
    df = pd.read_parquet(args.candidates)
    if args.stride:
        df = df.iloc[::args.stride].copy()
    if args.n:
        df = df.head(args.n)
    print(f"[v2scan] rows={len(df)}", flush=True)

    # v2 gate 与冻结 ARMED 不同, 不能预筛 armed; 全量加载复权链
    codes = sorted({str(r.get("symbol") or r.get("code")).zfill(6)
                    for _, r in df.iterrows()})
    print(f"[v2scan] loading chains for {len(codes)} codes...", flush=True)
    chains = {}
    for i in range(0, len(codes), 300):
        part = L.load_adjustment_chain(codes[i:i + 300])
        chains.update(part)
        print(f"  [chains] {min(i + 300, len(codes))}/{len(codes)}", flush=True)

    minute_cache: dict = {}
    reasons = Counter()
    n_score = 0
    n_nine30 = 0
    bucket_counter = Counter()
    path_counter = Counter()
    setup_counter = Counter()
    dipstage_counter = Counter()
    score_by_setup = defaultdict(lambda: [0, 0])     # setup -> [scored, total]
    score_by_dipstage = defaultdict(lambda: [0, 0])  # dip_stage -> [scored, total]
    trigger_minutes = []

    for ci, (_, r) in enumerate(df.iterrows()):
        code = str(r.get("symbol") or r.get("code")).zfill(6)
        d_date = int(r["effective_trade_date"])
        plan_date = int(r["plan_date"])
        atr_ref = float(r["atr_ref"])
        setup = str(r.get("primary_setup", ""))
        dip_stage = r.get("dip_stage")

        chain = chains.get(code)
        if chain is None:
            reasons["no_chain"] += 1
            continue
        z_scale = (chain.factor_at(int(d_date)) or 1.0) / (chain.factor_at(int(plan_date)) or 1.0)
        key = (code, d_date)
        m = minute_cache.get(key)
        if m is None:
            m = L.load_minute(code, d_date)
            if minute_cache is not None:
                if len(minute_cache) > 400:
                    minute_cache.clear()
                minute_cache[key] = m
        if m is None or m.empty:
            reasons["minute_empty"] += 1
            continue
        try:
            zone = Zone(
                zone_id=str(r.get("primary_zone_id")),
                zone_source=str(r.get("primary_zone_source", "")),
                zone_lower=float(r["primary_zone_lower"]),
                zone_upper=float(r["primary_zone_upper"]),
                zone_mid=float(r["primary_zone_mid"]),
                zone_timestamp=int(plan_date),
                zone_width_atr=float(r.get("zone_band_atr", 0.0) or 0.0),
                zone_strength=int(r.get("zone_strength", 1) or 1),
                zone_source_count=1,
                zone_distance_atr=float(r.get("primary_zone_distance_atr", 0.0) or 0.0),
            )
        except Exception as e:  # noqa
            reasons[f"zone_err:{type(e).__name__}"] += 1
            continue

        snap = int(m["ts"].max())  # 全天空跑, 找 FIRST MODEL_SCORE_REQUEST
        out = run_state_engine_v2(
            zone, m, z_scale, setup, atr_ref, snap,
            float(r.get("open_dip_atr", 0.0) or 0.0),
            float(r.get("primary_zone_distance_atr", 0.0) or 0.0),
            dip_stage=dip_stage)

        setup_counter[setup] += 1
        dipstage_counter[str(dip_stage)] += 1
        score_by_setup[setup][1] += 1
        score_by_dipstage[str(dip_stage)][1] += 1

        if out.get("score_ts") is not None:
            s = int(out["score_ts"])
            n_score += 1
            b = _bucket(s)
            bucket_counter[b] += 1
            path_counter[out.get("score_path")] += 1
            trigger_minutes.append(s)
            score_by_setup[setup][0] += 1
            score_by_dipstage[str(dip_stage)][0] += 1
            if b == "09:30":
                n_nine30 += 1
        else:
            reasons[out.get("rejection")] += 1

        if (ci + 1) % 100 == 0:
            try:
                cc = shutil.disk_usage("C:\\")
                dd = shutil.disk_usage("D:\\")
                rss = (psutil.Process().memory_info().rss / 1e9) if psutil else float("nan")
                print(f"  [mon] n={ci + 1} scored={n_score} "
                      f"Cfree={cc.free / 1e9:.1f} Dfree={dd.free / 1e9:.1f} RSS={rss:.1f}", flush=True)
                if cc.free / 1e9 < 15:
                    print("!! C free < 15GB -> STOP", flush=True); sys.exit(7)
                if dd.free / 1e9 < 30:
                    print("!! D free < 30GB -> STOP", flush=True); sys.exit(7)
            except Exception:  # noqa
                pass

    n_total = len(df)
    rep = {
        "n_total": n_total,
        "n_scored": n_score,
        "overall_trigger_rate": round(n_score / n_total, 4) if n_total else 0,
        "n_0930": n_nine30,
        "assert_0930_zero": (n_nine30 == 0),
        "time_buckets": dict(bucket_counter),
        "paths": dict(path_counter),
        "rejections": dict(reasons),
        "setup_counts": dict(setup_counter),
        "dipstage_counts": dict(dipstage_counter),
        "trigger_rate_by_setup": {
            k: {"scored": v[0], "total": v[1],
                "rate": round(v[0] / v[1], 4) if v[1] else 0}
            for k, v in score_by_setup.items()},
        "trigger_rate_by_dipstage": {
            k: {"scored": v[0], "total": v[1],
                "rate": round(v[0] / v[1], 4) if v[1] else 0}
            for k, v in score_by_dipstage.items()},
    }
    if trigger_minutes:
        hm = [((t // 10000) % 100) * 60 + ((t // 100) % 100) for t in trigger_minutes]
        rep["median_trigger_min_of_day"] = float(np.median(hm))
    rep["elapsed_sec"] = round(time.time() - t0, 1)

    out_path = C.DATASETS / "trigger_v2_distribution.json"
    out_path.write_text(json.dumps(rep, ensure_ascii=False, indent=2), encoding="utf-8")
    print("[V2SCAN]", json.dumps(rep, ensure_ascii=False), flush=True)
    if n_nine30 != 0:
        print(f"!! ASSERT FAILED: {n_nine30} triggers at 09:30 (must be 0)", flush=True)
        sys.exit(1)
    print("[v2scan] assert_0930_zero OK", flush=True)


if __name__ == "__main__":
    main()
