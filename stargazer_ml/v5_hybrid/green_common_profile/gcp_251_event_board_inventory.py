# -*- coding: utf-8 -*-
"""251 条主口径模拟可成交事件 — 证券类型 / 板块清单 (2026-10-04).

零新请求: 复用 gcp_dual_path_coverage_probe 的漏斗逻辑与数据路径, 枚举
main 口径 fillable 事件 (code,E,setup,used_before), 再用 instrument_master
标注每个 code 的真实 exchange_board / instrument_type.

A_SDP / B_LVB 是**形态池**, 不是交易板块; 两者都含主板/创业板/科创板/北交所
及少量基金等. 本脚本只把 251 的板块真相报出来, 不做任何收益计算.

产出: 251 事件明细 JSON + 逐段×A/B×板块 聚合 + 北交所/非股票隔离清单.
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

# 必须在 import data_source_provenance 之前设定, 否则静默走镜像源
os.environ.setdefault("GATE1C_MINUTE_SOURCE", "stockdb")

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "step9_preflight"))

# ---- 以库方式加载探针模块 (仅执行顶层定义, 不跑 main()) ----
spec = importlib.util.spec_from_file_location(
    "gcp_dual_path_coverage_probe", HERE / "gcp_dual_path_coverage_probe.py")
P = importlib.util.module_from_spec(spec)
spec.loader.exec_module(P)

from instrument_master import instrument_info, authority_status  # noqa: E402

t0 = time.time()
CACHE = HERE / "_251_events_cache.json"
if CACHE.exists():
    print(">> 命中事件缓存, 跳过漏斗重算")
    events = json.loads(CACHE.read_text(encoding="utf-8"))
    panel = P.get_daily_parquet()
    daily_by_code = {c: g.sort_values("date").reset_index(drop=True)
                     for c, g in panel.groupby("code")}
else:
    panel = P.get_daily_parquet()
panel["code"] = panel["code"].astype(str).str.zfill(6)
panel["date"] = panel["date"].astype(int)
cal = sorted(int(d) for d in panel["date"].unique())
cal_idx = {d: i for i, d in enumerate(cal)}
daily_by_code = {c: g.sort_values("date").reset_index(drop=True)
                 for c, g in panel.groupby("code")}
print(f"日线: {len(cal)} 交易日, {len(daily_by_code)} 股")

pool_a = pd.read_parquet(P.POOL_DIR / "pool_a_sdp_members.parquet")
pool_b = pd.read_parquet(P.POOL_DIR / "pool_b_lvb_members.parquet")
pool_a["code"] = pool_a["code"].astype(str).str.zfill(6)
pool_b["code"] = pool_b["code"].astype(str).str.zfill(6)
members = sorted({(c, int(p), "A_SDP") for c, p in pool_a[["code", "date"]].itertuples(index=False)}
                 | {(c, int(p), "B_LVB") for c, p in pool_b[["code", "date"]].itertuples(index=False)})
print(f"池成员 (去重): {len(members)}")

used_keys = P.load_used_keys()
minute_idx = P.load_minute_index()
print(f"分钟缓存索引: {len(minute_idx)} (code,date)")
grid = json.loads((P.OUT / "minute_grid.json").read_text(encoding="utf-8"))["grid"]

# ---- 逐成员三口径箱体 (同探针) ----
member_evals = []
for code, p, setup in members:
    dc = daily_by_code.get(code)
    if dc is None:
        continue
    loc = int(dc["date"].searchsorted(p, side="right"))
    if loc < 20:
        continue
    d_sub = dc.iloc[:loc]
    atr_ref = P.E.build_atr_ref(d_sub)
    calibs = P.evaluate_calibers(d_sub, atr_ref)
    atr14 = pd.Series(np.maximum(
        d_sub["high"].astype(float).to_numpy() - d_sub["low"].astype(float).to_numpy(),
        np.abs(d_sub["high"].astype(float).to_numpy()
               - np.concatenate([[d_sub["close"].astype(float).to_numpy()[0]],
                                 d_sub["close"].astype(float).to_numpy()[:-1]]))
    )).rolling(14).mean().to_numpy()
    amed60 = float(np.nanmedian(atr14[-60:])) if len(atr14) >= 14 else np.nan
    contraction = (atr_ref / amed60 if np.isfinite(amed60) and amed60 > 0 else np.nan)
    vol_arr = d_sub["volume"].astype(float).to_numpy()
    last10 = vol_arr[-10:]
    cv = float(np.std(last10, ddof=1) / np.mean(last10)) if np.mean(last10) > 0 else np.nan
    member_evals.append({
        "code": code, "P": p, "setup": setup, "atr_ref": atr_ref,
        "calibs": calibs,
        "contraction": float(contraction) if np.isfinite(contraction) else None,
        "vol_cv10": cv if np.isfinite(cv) else None,
        "minute_era": p >= P.MINUTE_ERA_START,
    })

# ---- 2025+ 漏斗明细, 捕获 main 口径 fillable 事件 ----
era = [m for m in member_evals if m["minute_era"]]
events = []
seen_e = set()
for m in era:
    code, p, setup = m["code"], m["P"], m["setup"]
    i_p = cal_idx.get(p)
    if i_p is None or i_p + 1 >= len(cal):
        continue
    e = cal[i_p + 1]
    key_e = (code, e)
    if key_e in seen_e:
        continue
    cb = m["calibs"].get("main", {})
    if not cb.get("valid"):
        continue  # 仅 main 口径 (251 起点)
    mkeys = minute_idx.get((code, e))
    if not mkeys:
        continue  # 无分钟缓存 → 无法判定入场可成交
    try:
        e_df_raw = pd.read_parquet(mkeys[0][0])
        day_int = e_df_raw["date"].astype(str).str[:8].astype(int)
        e_df = e_df_raw[day_int == e].sort_values("date").reset_index(drop=True)
        vd = P.E.validate_minute_day(e_df, grid, e)
        if not vd["ok"]:
            continue
    except Exception:
        continue
    c_ok = (m["contraction"] is not None and m["contraction"] <= P.ATR_CONTRACTION_MAX)
    v_ok = (m["vol_cv10"] is not None and m["vol_cv10"] <= P.VOL_STABILITY_CV_MAX)
    views = P.scan_views(e_df, cb["upper"], cb["lower"], m["atr_ref"], c_ok, v_ok)
    if views["action_ts"] is None:
        continue
    fill = P.fill_feasibility(e_df, views["action_ts"])
    if fill is None or fill["status"] != "FILLABLE":
        continue
    seen_e.add(key_e)
    info = instrument_info(code)
    events.append({
        "code": code, "P": p, "E": e, "setup": setup,
        "segment": P.segment_of(e),
        "used_before": (code, e) in used_keys,
        "exchange_board": info["exchange_board"],
        "instrument_type": info["instrument_type"],
        "resolution_method": info["resolution_method"],
        "instrument_type_conflict": info["instrument_type_conflict"],
    })

print(f"\n[枚举完成] main 口径 fillable 事件 = {len(events)} "
      f"(应等于 251: 24 used + 227 unused)")
CACHE.write_text(json.dumps(events, ensure_ascii=False), encoding="utf-8")
print(f"耗时 {round(time.time()-t0,1)}s")

# ---- 聚合: 段 × A/B × 板块 ----
from collections import defaultdict
xs_ab_board = defaultdict(int)
xs_board = defaultdict(int)
xs_ab_type = defaultdict(int)
seg_tot = defaultdict(int)
bse_rows, nonstock_rows = [], []
for r in events:
    seg, setup, board, itype = r["segment"], r["setup"], r["exchange_board"], r["instrument_type"]
    xs_ab_board[(seg, setup, board)] += 1
    xs_board[(seg, board)] += 1
    xs_ab_type[(seg, setup, itype)] += 1
    seg_tot[seg] += 1
    if board == "BSE":
        bse_rows.append(r)
    if itype != "STOCK":
        nonstock_rows.append(r)

boards_order = ["SH_MAIN", "SZ_MAIN", "GEM", "STAR", "BSE", "FUND_LIST", "BOND_LIST", "UNKNOWN"]
segs = ["train", "val", "test"]
setups = ["A_SDP", "B_LVB"]

print("\n=== 段 × 板块 (BSE 隔离, 非股票单列) ===")
hdr = f"{'seg':6} | " + " | ".join(f"{b:8}" for b in ["SH_MAIN", "SZ_MAIN", "GEM", "STAR", "BSE", "非股票", "合计"])
print(hdr)
for seg in segs:
    row = []
    for b in ["SH_MAIN", "SZ_MAIN", "GEM", "STAR", "BSE"]:
        row.append(xs_board.get((seg, b), 0))
    nonstock = sum(v for (s, sp, t), v in xs_ab_type.items() if s == seg and t != "STOCK")
    row.append(nonstock)
    row.append(seg_tot[seg])
    print(f"{seg:6} | " + " | ".join(f"{x:8}" for x in row))

print("\n=== 段 × A/B × 板块 (STOCK 板块; 非股票并入'非股票') ===")
print(f"{'seg':6} {'setup':7} | " + " | ".join(f"{b:8}" for b in ["SH_MAIN", "SZ_MAIN", "GEM", "STAR", "BSE", "非股票", "小计"]))
for seg in segs:
    for setup in setups:
        row = []
        for b in ["SH_MAIN", "SZ_MAIN", "GEM", "STAR", "BSE"]:
            row.append(xs_ab_board.get((seg, setup, b), 0))
        nonstock = sum(v for (s, sp, t), v in xs_ab_type.items() if s == seg and sp == setup and t != "STOCK")
        row.append(nonstock)
        row.append(sum(xs_ab_board.get((seg, setup, b), 0) for b in boards_order[:5]) + nonstock)
        print(f"{seg:6} {setup:7} | " + " | ".join(f"{x:8}" for x in row))

print(f"\n=== 北交所隔离: {len(bse_rows)} 条 ===")
# 沪深主板+创业板+科创板 (首轮范围)
in_scope = [r for r in events if r["exchange_board"] in ("SH_MAIN", "SZ_MAIN", "GEM", "STAR")
            and r["instrument_type"] == "STOCK"]
print(f"首轮范围 (沪深主板+创业板+科创板, 仅STOCK): {len(in_scope)} 条")
print(f"  其中按板块: " + ", ".join(
    f"{b}={sum(1 for r in in_scope if r['exchange_board']==b)}"
    for b in ["SH_MAIN", "SZ_MAIN", "GEM", "STAR"]))
print(f"北交所: {len(bse_rows)} 条 (隔离, 首轮不纳入)")
print(f"非股票: {len(nonstock_rows)} 条 (隔离): " + ", ".join(
    f"{r['code']}({r['instrument_type']},{r['exchange_board']})" for r in nonstock_rows[:20])
    + (f" ...共{len(nonstock_rows)}" if len(nonstock_rows) > 20 else ""))

# ---- 落盘 ----
out = {
    "version": "gcp_251_event_board_inventory_v1",
    "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    "network_requests": 0,
    "total_events": len(events),
    "instrument_master": authority_status(),
    "segment_totals": dict(seg_tot),
    "by_segment_board": {f"{s}|{b}": c for (s, b), c in xs_board.items()},
    "by_segment_ab_board": {f"{s}|{sp}|{b}": c for (s, sp, b), c in xs_ab_board.items()},
    "in_scope_first_round": {
        "definition": "SH_MAIN+SZ_MAIN+GEM+STAR, instrument_type==STOCK",
        "count": len(in_scope),
        "by_board": {b: sum(1 for r in in_scope if r["exchange_board"] == b)
                     for b in ["SH_MAIN", "SZ_MAIN", "GEM", "STAR"]},
    },
    "bse_isolated": len(bse_rows),
    "nonstock_isolated": len(nonstock_rows),
    "nonstock_detail": [
        {"code": r["code"], "E": r["E"], "setup": r["setup"],
         "instrument_type": r["instrument_type"], "exchange_board": r["exchange_board"]}
        for r in nonstock_rows],
    "events": events,
}
OUT_JSON = HERE / "gcp_round_b_small_sample_v1" / "dual_path_251_event_board_inventory.json"
OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
OUT_JSON.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"\n明细已写: {OUT_JSON.name} ({len(events)} 事件)")
