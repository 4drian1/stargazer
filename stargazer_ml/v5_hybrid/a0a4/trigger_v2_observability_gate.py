"""trigger_v2_observability_gate.py — Step3: TIMING-TRIGGER-v2 特征可观测性门控 (不训练).

复用 a14_universe.parquet 的 zone 几何 / setup / atr / gap_atr, 对每候选全天空跑 run_state_engine_v2,
取第一次 MODEL_SCORE_REQUEST 的 S=score_ts 处冻结的 as-of-S 特征 (compute_features_asof, v2 trace 产出,
绝不调用冻结引擎), 按 "触发路径 × 时间桶" 双拆统计:

  每字段: available_rate (非空率) / nonzero_rate (非空且!=0) / median / p25 / p75 / std

路径: BOTTOM_STABILIZATION / BOTTOM_RECOVERY / BOTTOM_RECLAIM / DIP_RECOVERY / DIP_RECLAIM
时间: 09:31-10:00 / 10:01-11:30 / 13:00+

顺带核查 dip_stage=NaN 来源 (按 setup / 按 path), 以确认是否为 BOTTOM-only 可接受.

输出: datasets/trigger_v2_observability.json (不训练, 不写母表).
"""
from __future__ import annotations
import json
import sys
import time
import shutil
import argparse
from pathlib import Path
from collections import defaultdict

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
from zone_state_engine_v2 import run_state_engine_v2, V2_FEATURE_REGISTRY  # noqa (v2, 自带 compute_features_asof)
import latr_builder as L  # noqa (load_minute / load_adjustment_chain)
import trading_calendar as TC  # noqa

PATHS = ["BOTTOM_STABILIZATION", "BOTTOM_RECOVERY", "BOTTOM_RECLAIM",
         "DIP_RECOVERY", "DIP_RECLAIM"]
BUCKETS = ["09:31-10:00", "10:01-11:30", "13:00+"]

# Step3 重点字段 (A3 新5 + A3 价格 + A4 状态, 与用户清单一致)
FIELDS = [
    # A3 v2 新
    "realized_dip_atr", "current_price_dip_atr", "zone_penetration_low_atr",
    "current_zone_distance_atr", "current_break_depth_atr",
    # A3 价格
    "open_dip_atr", "dip_depth_5m_atr", "dip_depth_30m_atr",
    "dip_speed_5m", "dip_speed_15m", "dip_speed_30m", "total_dip_atr",
    # A4 状态 (v2 语义)
    "recovery_from_zone_low_atr", "recovery_atr", "dip_recovery_ratio",
    "reclaim_zone_lower", "reclaim_zone_mid", "reclaim_zone_upper", "reclaim_failure_count",
    "second_test_flag", "second_test_higher_low",
    "time_since_zone_touch", "time_below_zone",
    "lower_wick_ratio", "sell_volume_ratio", "rebound_volume_ratio", "vwap_reclaim",
    # window 完整性 flag (审计: partial vs full)
    "window_complete_5m", "window_complete_15m", "window_complete_30m",
    # zone geometry audit
    "zone_width_atr", "open_to_zone_upper_atr", "open_to_zone_mid_atr", "open_to_zone_lower_atr",
]

# 窗口字段 -> 其完整性 flag (用于 window completeness audit)
WINDOWED = {
    "open_dip_atr": "window_complete_15m",
    "dip_depth_5m_atr": "window_complete_5m",
    "dip_depth_30m_atr": "window_complete_30m",
    "dip_speed_5m": "window_complete_5m",
    "dip_speed_15m": "window_complete_15m",
    "dip_speed_30m": "window_complete_30m",
}
GEOMETRY_FIELDS = ["zone_width_atr", "open_to_zone_upper_atr",
                   "open_to_zone_mid_atr", "open_to_zone_lower_atr"]


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


def _stats(vals):
    arr = np.array([v for v in vals
                    if v is not None and not (isinstance(v, float) and np.isnan(v))],
                   dtype=float)
    n = len(vals)
    if n == 0 or len(arr) == 0:
        return {"n": n, "available_rate": 0.0, "nonzero_rate": 0.0,
                "median": None, "p10": None, "p25": None, "p75": None, "p90": None, "std": None}
    avail = len(arr) / n
    nonzero = float(np.mean(arr != 0))
    return {"n": n, "available_rate": round(avail, 4), "nonzero_rate": round(nonzero, 4),
            "median": round(float(np.median(arr)), 4),
            "p10": round(float(np.percentile(arr, 10)), 4),
            "p25": round(float(np.percentile(arr, 25)), 4),
            "p75": round(float(np.percentile(arr, 75)), 4),
            "p90": round(float(np.percentile(arr, 90)), 4),
            "std": round(float(np.std(arr)), 4)}


def _agg(records, field):
    return _stats([r["feats"].get(field) for r in records])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=None)
    ap.add_argument("--stride", type=int, default=None)
    ap.add_argument("--candidates", default=str(C.DATASETS / "a14_universe.parquet"))
    args = ap.parse_args()
    t0 = time.time()
    print(f"[obs_gate] start src={args.candidates}", flush=True)
    df = pd.read_parquet(args.candidates)
    if args.stride:
        df = df.iloc[::args.stride].copy()
    if args.n:
        df = df.head(args.n)
    print(f"[obs_gate] rows={len(df)}", flush=True)

    codes = sorted({str(r.get("symbol") or r.get("code")).zfill(6)
                    for _, r in df.iterrows()})
    print(f"[obs_gate] loading chains for {len(codes)} codes...", flush=True)
    chains = {}
    for i in range(0, len(codes), 300):
        part = L.load_adjustment_chain(codes[i:i + 300])
        chains.update(part)
        print(f"  [chains] {min(i + 300, len(codes))}/{len(codes)}", flush=True)

    minute_cache: dict = {}
    records = []
    reasons = defaultdict(int)
    n_score = 0

    for ci, (_, r) in enumerate(df.iterrows()):
        code = str(r.get("symbol") or r.get("code")).zfill(6)
        d_date = int(r["effective_trade_date"])
        plan_date = int(r["plan_date"])
        atr_ref = float(r["atr_ref"])
        setup = str(r.get("primary_setup", ""))
        dip_stage = r.get("dip_stage")
        gap_raw = r.get("gap_atr")
        gap_atr = float(gap_raw) if (gap_raw is not None and not (isinstance(gap_raw, float) and np.isnan(gap_raw))) else None

        chain = chains.get(code)
        if chain is None:
            reasons["no_chain"] += 1
            continue
        z_scale = (chain.factor_at(int(d_date)) or 1.0) / (chain.factor_at(int(plan_date)) or 1.0)
        m = minute_cache.get((code, d_date))
        if m is None:
            m = L.load_minute(code, d_date)
            if minute_cache is not None:
                if len(minute_cache) > 400:
                    minute_cache.clear()
                minute_cache[(code, d_date)] = m
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

        snap = int(m["ts"].max())
        out = run_state_engine_v2(
            zone, m, z_scale, setup, atr_ref, snap,
            float(r.get("open_dip_atr", 0.0) or 0.0),
            float(r.get("primary_zone_distance_atr", 0.0) or 0.0),
            dip_stage=dip_stage, gap_atr=gap_atr)

        if out.get("score_ts") is None:
            reasons[out.get("rejection")] += 1
            continue
        n_score += 1
        records.append({
            "path": out["score_path"],
            "bucket": _bucket(out["score_ts"]),
            "setup": setup,
            "dip_stage_nan": (dip_stage is None or (isinstance(dip_stage, float) and np.isnan(dip_stage))),
            "feats": out["features_asof_S"],
        })

        if (ci + 1) % 100 == 0:
            try:
                cc = shutil.disk_usage("C:\\"); dd = shutil.disk_usage("D:\\")
                rss = (psutil.Process().memory_info().rss / 1e9) if psutil else float("nan")
                print(f"  [mon] n={ci + 1} scored={n_score} Cfree={cc.free/1e9:.1f} "
                      f"Dfree={dd.free/1e9:.1f} RSS={rss:.1f}", flush=True)
                if cc.free / 1e9 < 15:
                    print("!! C free < 15GB -> STOP", flush=True); sys.exit(7)
                if dd.free / 1e9 < 30:
                    print("!! D free < 30GB -> STOP", flush=True); sys.exit(7)
            except Exception:  # noqa
                pass

    print(f"[obs_gate] scored={n_score}", flush=True)

    fields_overall = {f: _agg(records, f) for f in FIELDS}
    by_path = {}
    for p in PATHS:
        pr = [r for r in records if r["path"] == p]
        by_path[p] = {f: _agg(pr, f) for f in FIELDS}
    by_path_time = {}
    for p in PATHS:
        by_path_time[p] = {}
        for b in BUCKETS:
            sub = [r for r in records if r["path"] == p and r["bucket"] == b]
            by_path_time[p][b] = {f: _agg(sub, f) for f in FIELDS}

    # dip_stage NaN 来源
    nan_overall = sum(1 for r in records if r["dip_stage_nan"])
    by_setup_nan = {}
    for s in ("BOTTOM_SETUP", "INTRADAY_DIP_SETUP"):
        sub = [r for r in records if r["setup"] == s]
        nn = sum(1 for r in sub if r["dip_stage_nan"])
        by_setup_nan[s] = {"nan": nn, "non_nan": len(sub) - nn,
                           "rate": round(nn / len(sub), 4) if sub else 0.0}
    by_path_nan = {}
    for p in PATHS:
        sub = [r for r in records if r["path"] == p]
        nn = sum(1 for r in sub if r["dip_stage_nan"])
        by_path_nan[p] = {"nan": nn, "non_nan": len(sub) - nn,
                          "rate": round(nn / len(sub), 4) if sub else 0.0}

    # window completeness audit (partial vs full 窗口)
    window_complete_audit = {}
    for f, flag in WINDOWED.items():
        n_complete = sum(1 for r in records if r["feats"].get(flag) == 1)
        window_complete_audit[f] = {
            "complete_rate": round(n_complete / n_score, 4) if n_score else 0.0,
            "n_complete": n_complete,
            "by_path": {p: round(
                sum(1 for r in records if r["path"] == p and r["feats"].get(flag) == 1)
                / max(1, sum(1 for r in records if r["path"] == p)), 4) for p in PATHS},
        }

    # zone geometry audit (BOTTOM_STABILIZATION 重点 + 全量)
    def _pctiles(sub_records, field, ps=(10, 50, 90)):
        arr = np.array([r["feats"].get(field) for r in sub_records
                        if r["feats"].get(field) is not None
                        and not (isinstance(r["feats"].get(field), float) and np.isnan(r["feats"].get(field)))],
                       dtype=float)
        if len(arr) == 0:
            return {f"p{p}": None for p in ps}
        return {f"p{p}": round(float(np.percentile(arr, p)), 4) for p in ps}

    geometry_audit = {}
    for grp_name, grp_filter in (("BOTTOM_STABILIZATION", lambda r: r["path"] == "BOTTOM_STABILIZATION"),
                                 ("ALL_SCORED", lambda r: True)):
        sub = [r for r in records if grp_filter(r)]
        geometry_audit[grp_name] = {
            "n": len(sub),
            "fields": {gf: _pctiles(sub, gf) for gf in GEOMETRY_FIELDS},
            "zone_penetration_low_atr": _pctiles(sub, "zone_penetration_low_atr"),
            "realized_dip_atr": _pctiles(sub, "realized_dip_atr"),
        }

    rep = {
        "n_total": len(df),
        "n_scored": n_score,
        "fields_overall": fields_overall,
        "by_path": by_path,
        "by_path_time": by_path_time,
        "dip_stage_nan": {
            "overall_scored": {"nan": nan_overall, "non_nan": n_score - nan_overall,
                               "rate": round(nan_overall / n_score, 4) if n_score else 0.0},
            "by_setup": by_setup_nan,
            "by_path": by_path_nan,
        },
        "rejections": dict(reasons),
        "feature_registry": V2_FEATURE_REGISTRY,
        "window_completeness_audit": window_complete_audit,
        "zone_geometry_audit": geometry_audit,
        "elapsed_sec": round(time.time() - t0, 1),
    }

    out_path = C.DATASETS / "trigger_v2_observability.json"
    out_path.write_text(json.dumps(rep, ensure_ascii=False, indent=2), encoding="utf-8")
    _print_summary(rep)
    print(f"[obs_gate] wrote {out_path}", flush=True)


def _print_summary(rep):
    print("\n================ Step3 Observability Gate ================")
    print(f"n_total={rep['n_total']}  n_scored={rep['n_scored']}")
    # 1) 总体字段 available/nonzero
    print("\n[OVERALL] field: avail / nonzero(rate) / median (p25,p75) std")
    for f in FIELDS:
        s = rep["fields_overall"][f]
        print(f"  {f:28s} avail={s['available_rate']:.3f} nz={s['nonzero_rate']:.3f} "
              f"med={s['median']} ({s['p25']},{s['p75']}) sd={s['std']}")
    # 2) 重点 recovery 字段按 path 的 avail/nonzero
    key_fields = ["recovery_from_zone_low_atr", "dip_recovery_ratio", "reclaim_zone_lower",
                  "second_test_flag", "time_since_zone_touch", "time_below_zone",
                  "realized_dip_atr", "zone_penetration_low_atr"]
    print("\n[BY PATH] key recovery fields: avail / nonzero")
    hdr = "  {:24s}".format("field") + "".join(f"{p[:10]:>14s}" for p in PATHS)
    print(hdr)
    for f in key_fields:
        row = f"  {f:24s}"
        for p in PATHS:
            s = rep["by_path"][p][f]
            row += f"{s['available_rate']:.2f}/{s['nonzero_rate']:.2f}".rjust(14)
        print(row)
    # 3) dip_stage NaN
    print("\n[DIP_STAGE NaN]")
    print(f"  overall_scored: rate={rep['dip_stage_nan']['overall_scored']['rate']} "
          f"({rep['dip_stage_nan']['overall_scored']['nan']}/{rep['n_scored']})")
    for s, v in rep["dip_stage_nan"]["by_setup"].items():
        print(f"  by_setup {s:18s}: nan_rate={v['rate']} ({v['nan']}/{v['nan']+v['non_nan']})")
    for p, v in rep["dip_stage_nan"]["by_path"].items():
        print(f"  by_path  {p:22s}: nan_rate={v['rate']} ({v['nan']}/{v['nan']+v['non_nan']})")
    # 4) window completeness audit (防止 09:35 冒充 30min)
    print("\n[WINDOW COMPLETENESS] field: complete_rate (仅当 flag=1 才算完整窗口) ; by_path")
    for f, flag in WINDOWED.items():
        wc = rep["window_completeness_audit"][f]
        byp = " ".join(f"{p[:10]}:{wc['by_path'][p]:.2f}" for p in PATHS)
        print(f"  {f:22s} compl={wc['complete_rate']:.3f}  {byp}")
    # 5) zone geometry audit (解释 0.115 浅 dip vs 1.289 深 penetration)
    print("\n[ZONE GEOMETRY] BOTTOM_STABILIZATION p10/p50/p90 (atr 单位)")
    ga = rep["zone_geometry_audit"]["BOTTOM_STABILIZATION"]
    print(f"  n={ga['n']}")
    for gf in GEOMETRY_FIELDS:
        p = ga["fields"][gf]
        print(f"  {gf:24s} p10={p['p10']} p50={p['p50']} p90={p['p90']}")
    rt = ga["realized_dip_atr"]; tp = ga["zone_penetration_low_atr"]
    print(f"  {'realized_dip_atr':24s} p10={rt['p10']} p50={rt['p50']} p90={rt['p90']}")
    print(f"  {'zone_penetration_low':24s} p10={tp['p10']} p50={tp['p50']} p90={tp['p90']}")
    # 6) model eligibility
    print("\n[MODEL ELIGIBILITY] NOT eligible for v2-primary (reason):")
    for f, meta in V2_FEATURE_REGISTRY.items():
        if not meta.get("model_eligible"):
            print(f"  {f:24s} -> {meta.get('reason')}")
    print("===========================================================\n")


if __name__ == "__main__":
    main()
