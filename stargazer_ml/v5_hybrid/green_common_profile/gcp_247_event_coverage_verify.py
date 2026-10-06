# -*- coding: utf-8 -*-
"""首轮 247 事件逐事件只读覆盖核验 (2026-10-04).

零新增抓取: 仅在本地日线 parquet / 分钟缓存 / gate1c 复权缓存上重算.
核验维度 (用户裁决):
  - 入场执行元数据 (funnel fillable = 模拟候选, 非正式 FILLED)
  - E+1..H5 每个可能触发止损的交易日分钟路径 (非仅 E+3/E+5)
  - 预计顺延窗口 (缺失日 -> 无法评估止损/延迟)
  - 复权链本地可达性 (gate1c feb2026 缓存; 不可达 -> 待抓取清单)
  - 尾部截断 (H5 > 数据终点 20260928 -> CENSORED_TAIL)
按 train/val/test × A/B × 板块 报可用数与 NULL 原因; 缺数据先报去重请求清单与预算.
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

os.environ.setdefault("GATE1C_MINUTE_SOURCE", "stockdb")

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "step9_preflight"))

spec = importlib.util.spec_from_file_location(
    "gcp_dual_path_coverage_probe", HERE / "gcp_dual_path_coverage_probe.py")
P = importlib.util.module_from_spec(spec)
spec.loader.exec_module(P)

DATA_END = 20260928
INV_JSON = HERE / "gcp_round_b_small_sample_v1" / "dual_path_251_event_board_inventory.json"
ADJ_CACHE = ROOT / "opportunity_v2" / "gate1c" / "feb2026_adjustment_cache_v1" / "adj_cache.json"

t0 = time.time()
inv = json.loads(INV_JSON.read_text(encoding="utf-8"))
events = [e for e in inv["events"]
          if e["instrument_type"] == "STOCK"
          and e["exchange_board"] in ("SH_MAIN", "SZ_MAIN", "GEM", "STAR")]
print(f"首轮 247 范围: {len(events)}")

panel = P.get_daily_parquet()
panel["code"] = panel["code"].astype(str).str.zfill(6)
panel["date"] = panel["date"].astype(int)
cal = sorted(int(d) for d in panel["date"].unique())
cal_idx = {d: i for i, d in enumerate(cal)}
daily_by_code = {c: g.sort_values("date").reset_index(drop=True)
                 for c, g in panel.groupby("code")}
print(f"日线: {len(cal)} 交易日, 终点 {cal[-1]}")

minute_idx = P.load_minute_index()
print(f"分钟缓存索引: {len(minute_idx)} (code,date)")
grid = json.loads((P.OUT / "minute_grid.json").read_text(encoding="utf-8"))["grid"]

adj_local_codes = set()
if ADJ_CACHE.exists():
    adj_local_codes = set(json.loads(ADJ_CACHE.read_text(encoding="utf-8")).keys())
print(f"本地复权缓存码数: {len(adj_local_codes)}")


def atr14_last(d_sub):
    a = pd.Series(np.maximum(
        d_sub["high"].astype(float).to_numpy() - d_sub["low"].astype(float).to_numpy(),
        np.abs(d_sub["high"].astype(float).to_numpy()
               - np.concatenate([[d_sub["close"].astype(float).to_numpy()[0]],
                                 d_sub["close"].astype(float).to_numpy()[:-1]]))
    )).rolling(14).mean().to_numpy()
    return a


def entry_of(code, p, setup):
    """复算入场 (模拟候选, 非 FILLED). 返回 dict."""
    dc = daily_by_code.get(code)
    if dc is None:
        return {"entry_status": "NO_DAILY"}
    loc = int(dc["date"].searchsorted(p, side="right"))
    if loc < 20:
        return {"entry_status": "INSUFFICIENT_HISTORY"}
    d_sub = dc.iloc[:loc]
    atr_ref = P.E.build_atr_ref(d_sub)
    calibs = P.evaluate_calibers(d_sub, atr_ref)
    cb = calibs.get("main", {})
    if not cb.get("valid"):
        return {"entry_status": "NO_BOX"}
    i_p = cal_idx.get(p)
    if i_p is None or i_p + 1 >= len(cal):
        return {"entry_status": "NO_E"}
    e = cal[i_p + 1]
    mkeys = minute_idx.get((code, e))
    if not mkeys:
        return {"entry_status": "NO_MINUTE"}
    try:
        e_df_raw = pd.read_parquet(mkeys[0][0])
        day_int = e_df_raw["date"].astype(str).str[:8].astype(int)
        e_df = e_df_raw[day_int == e].sort_values("date").reset_index(drop=True)
        vd = P.E.validate_minute_day(e_df, grid, e)
        if not vd["ok"]:
            return {"entry_status": "MINUTE_INVALID", "reason": vd["reason"]}
    except Exception as ex:
        return {"entry_status": "MINUTE_READ_ERR", "reason": str(ex)[:80]}
    a14 = atr14_last(d_sub)
    amed60 = float(np.nanmedian(a14[-60:])) if len(a14) >= 14 else np.nan
    contraction = (atr_ref / amed60 if np.isfinite(amed60) and amed60 > 0 else np.nan)
    vol_arr = d_sub["volume"].astype(float).to_numpy()
    last10 = vol_arr[-10:]
    cv = float(np.std(last10, ddof=1) / np.mean(last10)) if np.mean(last10) > 0 else np.nan
    c_ok = (contraction is not None and contraction <= P.ATR_CONTRACTION_MAX)
    v_ok = (cv is not None and cv <= P.VOL_STABILITY_CV_MAX)
    views = P.scan_views(e_df, cb["upper"], cb["lower"], atr_ref, c_ok, v_ok)
    if views["action_ts"] is None:
        return {"entry_status": "NO_ACTION"}
    fill = P.fill_feasibility(e_df, views["action_ts"])
    if fill is None or fill["status"] != "FILLABLE":
        return {"entry_status": "SIM_BLOCKED", "fill_status": (fill["status"] if fill else None)}
    return {"entry_status": "SIM_CANDIDATE_FILLABLE",
            "action_ts": views["action_ts"], "fill_status": fill["status"]}


def day_coverage(code, d):
    """返回 (status, reason). status in DAY_OK / MINUTE_MISSING / DAY_INVALID / READ_ERR."""
    mkeys = minute_idx.get((code, d))
    if not mkeys:
        return "MINUTE_MISSING", None
    try:
        e_df_raw = pd.read_parquet(mkeys[0][0])
        day_int = e_df_raw["date"].astype(str).str[:8].astype(int)
        e_df = e_df_raw[day_int == d].sort_values("date").reset_index(drop=True)
        vd = P.E.validate_minute_day(e_df, grid, d)
        if vd["ok"]:
            return "DAY_OK", None
        return "DAY_INVALID", vd["reason"]
    except Exception as ex:
        return "READ_ERR", str(ex)[:80]


rows = []
minute_requests = set()   # (code, date) 缺失且 <=DATA_END
for e in events:
    code, Pd, Ee, setup, seg = e["code"], e["P"], e["E"], e["setup"], e["segment"]
    ent = entry_of(code, Pd, setup)
    i_E = cal_idx.get(Ee)
    scan = []
    for off in (1, 2, 3, 4, 5):
        if i_E is None or i_E + off >= len(cal):
            scan.append({"offset": off, "date": None, "status": "BEYOND_CALENDAR"})
        else:
            d = cal[i_E + off]
            st, rs = day_coverage(code, d)
            scan.append({"offset": off, "date": d, "status": st, "reason": rs})
            if st == "MINUTE_MISSING" and d <= DATA_END:
                minute_requests.add((code, d))
    H5_date = cal[i_E + 5] if (i_E is not None and i_E + 5 < len(cal)) else None
    censored = (H5_date is None) or (H5_date > DATA_END)
    adj = "LOCAL" if code in adj_local_codes else "NEEDS_FETCH"
    rows.append({"code": code, "P": Pd, "E": Ee, "setup": setup, "segment": seg,
                 "board": e["exchange_board"], "entry": ent, "scan": scan,
                 "censored": censored, "adj": adj})

print(f"逐事件核验完成, 耗时 {round(time.time()-t0,1)}s")

# ---- 聚合: seg × setup × board ----
agg = defaultdict(lambda: {"n": 0, "entry_sim_ok": 0, "exit_all5_ok": 0,
                          "day_missing": Counter(), "censored": 0,
                          "adj_local": 0, "adj_needs": 0,
                          "ready_no_fetch": 0, "entry_fail": Counter()})
by_seg = defaultdict(lambda: {"n": 0, "exit_all5_ok": 0, "censored": 0,
                              "adj_local": 0, "ready_no_fetch": 0})
for r in rows:
    key = (r["segment"], r["setup"], r["board"])
    a = agg[key]
    a["n"] += 1
    by_seg[r["segment"]]["n"] += 1
    ent_ok = r["entry"]["entry_status"] == "SIM_CANDIDATE_FILLABLE"
    if ent_ok:
        a["entry_sim_ok"] += 1
    else:
        a["entry_fail"][r["entry"]["entry_status"]] += 1
    days_ok = all(s["status"] == "DAY_OK" for s in r["scan"])
    if days_ok:
        a["exit_all5_ok"] += 1
        by_seg[r["segment"]]["exit_all5_ok"] += 1
    for s in r["scan"]:
        if s["status"] != "DAY_OK":
            a["day_missing"][s["status"]] += 1
    if r["censored"]:
        a["censored"] += 1
        by_seg[r["segment"]]["censored"] += 1
    if r["adj"] == "LOCAL":
        a["adj_local"] += 1
        by_seg[r["segment"]]["adj_local"] += 1
    else:
        a["adj_needs"] += 1
    ready = ent_ok and days_ok and (not r["censored"]) and r["adj"] == "LOCAL"
    if ready:
        a["ready_no_fetch"] += 1
        by_seg[r["segment"]]["ready_no_fetch"] += 1

segs = ["train", "val", "test"]
setups = ["A_SDP", "B_LVB"]
boards = ["SH_MAIN", "SZ_MAIN", "GEM", "STAR"]

print("\n=== 段 × A/B × 板块: 核验可用数 (exit_all5=5天分钟齐全; ready_no_fetch=入场OK+5天OK+未截断+复权本地) ===")
print(f"{'seg':6} {'setup':7} {'board':8} | {'n':>3} {'entryOK':>7} {'exit5':>5} {'cens':>4} {'adjL':>4} {'adjN':>4} {'ready':>5}")
for seg in segs:
    for setup in setups:
        for board in boards:
            a = agg[(seg, setup, board)]
            if a["n"] == 0:
                continue
            print(f"{seg:6} {setup:7} {board:8} | {a['n']:>3} {a['entry_sim_ok']:>7} "
                  f"{a['exit_all5_ok']:>5} {a['censored']:>4} {a['adj_local']:>4} "
                  f"{a['adj_needs']:>4} {a['ready_no_fetch']:>5}")
    # 段小计
    bs = by_seg[seg]
    print(f"  -- {seg} 小计: n={bs['n']} exit5_ok={bs['exit_all5_ok']} "
          f"censored={bs['censored']} adj_local={bs['adj_local']} ready_no_fetch={bs['ready_no_fetch']}")

print("\n=== 全量 (247) 汇总 ===")
tot = Counter()
for a in agg.values():
    tot["n"] += a["n"]; tot["entry_ok"] += a["entry_sim_ok"]
    tot["exit5_ok"] += a["exit_all5_ok"]; tot["censored"] += a["censored"]
    tot["adj_local"] += a["adj_local"]; tot["adj_needs"] += a["adj_needs"]
    tot["ready"] += a["ready_no_fetch"]
print(f"n={tot['n']} entry_sim_ok={tot['entry_ok']} exit_all5_ok={tot['exit5_ok']} "
      f"censored={tot['censored']} adj_local={tot['adj_local']} adj_needs={tot['adj_needs']} "
      f"ready_no_fetch={tot['ready']}")
print("入场失败分布:", dict(Counter(k for a in agg.values() for k in a["entry_fail"])))
print("退出日缺失状态:", dict(Counter(s for a in agg.values() for s in a["day_missing"])))

# ---- 去重请求清单与预算 ----
adj_codes_needed = sorted({r["code"] for r in rows if r["adj"] == "NEEDS_FETCH"})
print(f"\n=== 缺数据请求清单 (零抓取, 待裁决后预算) ===")
print(f"复权链待抓取码数 (去重): {len(adj_codes_needed)}  (每码 1 次 stockdb 复权 HTTP)")
print(f"分钟缺失 (code,day) 待抓取数 (去重, <=数据终点): {len(minute_requests)}")
minute_by_seg = Counter()
for (code, d) in minute_requests:
    seg = next((r["segment"] for r in rows if r["code"] == code and r["E"] and any(s["date"] == d for s in r["scan"])), "?")
    minute_by_seg[seg] += 1
print("  分钟缺失按段:", dict(minute_by_seg))

def _clean(o):
    if isinstance(o, Counter):
        return dict(o)
    if isinstance(o, dict):
        return {k: _clean(v) for k, v in o.items()}
    return o


def agg_clean(key):
    a = agg[key]
    return {"n": a["n"], "entry_sim_ok": a["entry_sim_ok"], "exit_all5_ok": a["exit_all5_ok"],
            "day_missing": dict(a["day_missing"]), "censored": a["censored"],
            "adj_local": a["adj_local"], "adj_needs": a["adj_needs"],
            "ready_no_fetch": a["ready_no_fetch"],
            "entry_fail": dict(a["entry_fail"])}


out = {
    "version": "gcp_247_coverage_verify_v1",
    "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    "network_requests": 0,
    "data_end": DATA_END,
    "n_events": len(rows),
    "summary": {
        "n": tot["n"], "entry_sim_ok": tot["entry_ok"], "exit_all5_ok": tot["exit5_ok"],
        "censored": tot["censored"], "adj_local": tot["adj_local"],
        "adj_needs": tot["adj_needs"], "ready_no_fetch": tot["ready"],
    },
    "by_segment_board": {f"{s}|{sp}|{b}": agg_clean((s, sp, b))
                         for s in segs for sp in setups for b in boards if agg[(s, sp, b)]["n"] > 0},
    "requests": {
        "adjustment_codes_needed": adj_codes_needed,
        "adjustment_codes_count": len(adj_codes_needed),
        "minute_missing_pairs": sorted([list(x) for x in minute_requests]),
        "minute_missing_count": len(minute_requests),
    },
    "rows": rows,
}
out = _clean(out)
OUT = HERE / "gcp_round_b_small_sample_v1" / "gcp_247_coverage_verify.json"
OUT.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"\n明细已写: {OUT.name}")
