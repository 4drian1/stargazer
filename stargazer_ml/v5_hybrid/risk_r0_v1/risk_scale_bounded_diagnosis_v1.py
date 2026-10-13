"""R0-V1 有界尺度诊断 v1（只读、零网络、不输出统一价）。

目的：对五类样本分别核实 日线字段(raw/除权口径) 的 原始口径、调整方向、锚点：
  1. 普通日（无公司行为）
  2. 公司行为日（来自 adj_cache 的 mult 事件）
  3. 分片边界（同一码跨 part_*.parquet 的连续性）
  4. 旧 32 只冲突码（高 pre_close 失配率码）
  5. 37 行 OHLC 异常

不声称根因已证明；只呈现证据，供人工判定字段口径。
"""
from __future__ import annotations
import json, glob, re
from pathlib import Path
import pandas as pd

DAILY = r"D:\a股数据\data_daily_parts"
CACHE = r"D:\stargazer\stargazer_ml\v5_hybrid\opportunity_v2\gate1c\feb2026_adjustment_cache_v1\adj_cache.json"
FULL = Path(r"D:\stargazer\stargazer_ml\v5_hybrid\risk_r0_v1\_scale_recon_full.json")
OUT = Path(r"D:\stargazer\stargazer_ml\v5_hybrid\risk_r0_v1\risk_scale_bounded_diagnosis_v1.json")

COLS = ["code", "date", "open", "high", "low", "close", "pre_close", "pct_chg"]

# ---- 候选样本码集合 ----
cache = json.loads(Path(CACHE).read_text(encoding="utf-8"))
# 冲突码（来自全量扫描 top20）
full = json.loads(FULL.read_text(encoding="utf-8"))
conflict_codes = [c for c, _, _, _ in full.get("high_mismatch_codes_top20", [])]
# 稳定大盘码
stable = ["000001", "600000", "000002", "600519", "000858", "601318"]
# 公司行为候选：2018-2026 间 |mult-1|>=0.03 的事件码（取前若干）
ca_candidates = []
for code, v in cache.items():
    if v.get("status") != "OK":
        continue
    for d, m in v.get("events", []):
        if 20180101 <= d <= 20261009 and abs(m - 1.0) >= 0.03:
            ca_candidates.append((code, d, m))
            break
ca_codes = [c for c, _, _ in ca_candidates[:12]]
# 异常码
anom_codes = list({str(a["code"]) for a in full.get("ohlc_anomaly_sample", [])})

sample_set = set(conflict_codes) | set(stable) | set(ca_codes) | set(anom_codes)
# 强制确保异常码与公司行为码都存在
for a in full.get("ohlc_anomaly_sample", []):
    sample_set.add(str(a["code"]))

# ---- 单遍收集样本码的全部日线行 ----
frames = []
for f in sorted(glob.glob(DAILY + r"\part_*.parquet")):
    df = pd.read_parquet(f, columns=COLS)
    df = df[df["code"].astype(str).isin(sample_set)]
    if len(df):
        frames.append(df)
data = pd.concat(frames, ignore_index=True)
data["code"] = data["code"].astype(str)
data = data.sort_values(["code", "date"]).reset_index(drop=True)
# 每码前一日收盘（原始口径锚点）
data["prev_close_raw"] = data.groupby("code")["close"].shift(1)

# 公司行为事件索引
ev_index = {}
for code, v in cache.items():
    if v.get("status") == "OK":
        ev_index[code] = {d: m for d, m in v.get("events", [])}

def classify_row(r):
    """返回该行的三类收益与判定。"""
    out = {}
    c, pc, pcr, pct = r["close"], r["pre_close"], r["prev_close_raw"], r["pct_chg"]
    ex_ret = (c / pc - 1) if (pd.notna(pc) and pc != 0) else None
    raw_ret = (c / pcr - 1) if (pd.notna(pcr) and pcr != 0) else None
    pct_dec = (pct / 100.0) if pd.notna(pct) else None
    out["ex_rights_ret"] = ex_ret
    out["raw_prev_ret"] = raw_ret
    out["pct_chg_dec"] = pct_dec
    # 判定 pct_chg 更接近哪个口径
    if pct_dec is not None:
        d_ex = abs(pct_dec - ex_ret) if ex_ret is not None else None
        d_raw = abs(pct_dec - raw_ret) if raw_ret is not None else None
        if d_ex is not None and d_raw is not None:
            out["pct_chg_matches"] = "ex_rights" if d_ex <= d_raw else "raw_prev"
            out["d_ex"] = d_ex
            out["d_raw"] = d_raw
    return out

def row_block(r):
    b = {
        "code": r["code"], "date": int(r["date"]),
        "open": r["open"], "high": r["high"], "low": r["low"], "close": r["close"],
        "pre_close": r["pre_close"], "pct_chg": r["pct_chg"],
        "prev_close_raw": r["prev_close_raw"],
    }
    b.update(classify_row(r))
    # 公司行为事件标记
    evs = ev_index.get(r["code"], {})
    d = int(r["date"])
    if d in evs:
        b["ca_event_mult"] = evs[d]
    return b

result = {"meta": {"purpose": "bounded scale diagnosis only; no unification output",
                   "sample_codes": len(sample_set),
                   "cache_codes": len(cache)}}

# 1. 普通日：选每个样本码一个无事件日
ordinary = []
for code, g in data.groupby("code"):
    evs = ev_index.get(code, {})
    cand = g[(g["date"].map(lambda d: d not in evs and (d - 1) not in evs and (d + 1) not in evs))
             & g["pre_close"].notna() & g["prev_close_raw"].notna() & g["pct_chg"].notna()]
    if len(cand):
        ordinary.append(row_block(cand.iloc[len(cand) // 2]))
result["1_ordinary_days"] = ordinary[:8]

# 2. 公司行为日（取 adj_cache 里 |mult-1|>=0.03 且在日线中的事件）
ca_rows = []
for code, d, m in ca_candidates:
    if code not in set(data["code"]):
        continue
    sub = data[(data["code"] == code) & (data["date"] == d)]
    if len(sub):
        b = row_block(sub.iloc[0])
        ca_rows.append(b)
    if len(ca_rows) >= 10:
        break
result["2_corporate_action_days"] = ca_rows

# 3. 分片边界：取一个码，展示其全序列相邻日收益并与事件对比（检测非事件跳变）
boundary = []
for code in ["000001", "600000", "600519"]:
    if code not in set(data["code"]):
        continue
    g = data[data["code"] == code].sort_values("date")
    evs = ev_index.get(code, {})
    seq = []
    prev = None
    for _, r in g.iterrows():
        d = int(r["date"])
        ret = (r["close"] / prev - 1) if prev else None
        is_ev = d in evs
        seq.append({"date": d, "close": r["close"], "prev_close_raw": r["prev_close_raw"],
                    "pre_close": r["pre_close"], "adj_ret_vs_prev": ret,
                    "ca_event": (evs[d] if is_ev else None)})
        prev = r["close"]
    # 仅报告相邻收益绝对值 > 0.11 且非事件日的“可疑跳变”
    spikes = [s for s in seq if s["adj_ret_vs_prev"] is not None
              and abs(s["adj_ret_vs_prev"]) > 0.11 and s["ca_event"] is None]
    boundary.append({"code": code, "n_rows": len(g),
                     "non_event_spikes_gt_11pct": spikes[:10]})
result["3_partition_boundary_continuity"] = boundary

# 4. 冲突码模式：top5 各取 3 个连续样本行
conflict_pattern = []
for code in conflict_codes[:5]:
    g = data[data["code"] == code].sort_values("date")
    if len(g) >= 3:
        pick = g.iloc[[0, len(g) // 2, len(g) - 1]]
        conflict_pattern.append({"code": code, "n_rows": len(g),
                                 "samples": [row_block(r) for _, r in pick.iterrows()]})
result["4_conflict_code_pattern"] = conflict_pattern

# 5. 37 异常行诊断
anom_detail = []
for a in full.get("ohlc_anomaly_sample", []):
    code, d = str(a["code"]), a["date"]
    sub = data[(data["code"] == code) & (data["date"] == d)]
    if len(sub):
        r = sub.iloc[0]
        issues = []
        for fld in ["open", "high", "low", "close"]:
            if pd.notna(r[fld]) and r[fld] <= 0:
                issues.append(f"{fld}<=0")
        if pd.notna(r["high"]) and pd.notna(r["low"]) and r["high"] < r["low"]:
            issues.append("high<low")
        if pd.notna(r["open"]) and pd.notna(r["high"]) and r["open"] > r["high"]:
            issues.append("open>high")
        if pd.notna(r["low"]) and pd.notna(r["open"]) and r["low"] > r["open"]:
            issues.append("low>open")
        anom_detail.append({"code": code, "date": d, "issues": issues,
                            "ohlc": [r["open"], r["high"], r["low"], r["close"]]})
result["5_ohlc_anomaly_detail"] = anom_detail

OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
print("WROTE", OUT, OUT.stat().st_size, "bytes")
print("ordinary", len(result["1_ordinary_days"]), "ca", len(result["2_corporate_action_days"]),
      "conflict", len(result["4_conflict_code_pattern"]), "anom", len(result["5_ohlc_anomaly_detail"]))
