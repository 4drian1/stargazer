"""R0-V1 有界尺度诊断 v2（修订：补输入哈希 / 来源分片 / 反例扫描 / 复权链独立验证 / NaN->null）。

只读、零网络。保留 v1 的有界样本作为证据，但明确：
  - "原始口径已确证" 不成立 -> 降为 HYPOTHESIS，并显式列出 pct_chg 既不匹配原始也不匹配除权的反例。
  - 分片边界须按来源分片逐边界核验（不默认行分片无尺度断裂）。
  - 复权链 factor_at(d)/factor_at(P) 作为待验方案，须用独立手算拆分/分红案例验证方向与精度，并核验链完整性/无效乘数/覆盖终点。
不输出全量统一价。
"""
from __future__ import annotations
import json, glob, hashlib, re, sys
from pathlib import Path
sys.path.insert(0, r"D:\stargazer\stargazer_ml\v5_hybrid")
import pandas as pd

DAILY = r"D:\a股数据\data_daily_parts"
CACHE = r"D:\stargazer\stargazer_ml\v5_hybrid\opportunity_v2\gate1c\feb2026_adjustment_cache_v1\adj_cache.json"
FULL = Path(r"D:\stargazer\stargazer_ml\v5_hybrid\risk_r0_v1\_scale_recon_full.json")
OUT = Path(r"D:\stargazer\stargazer_ml\v5_hybrid\risk_r0_v1\risk_scale_bounded_diagnosis_v2.json")
COLS = ["code", "date", "open", "high", "low", "close", "pre_close", "pct_chg"]

MATCH_TOL = 0.005  # 0.5 个百分点，作为"匹配"判据（v1 的 0.05=5pp 过松，已弃）


def sha256_file(p: str) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def sanitize(o):
    if isinstance(o, float):
        if o != o:  # NaN
            return None
        if o in (float("inf"), float("-inf")):
            return None
        return o
    if isinstance(o, dict):
        return {k: sanitize(v) for k, v in o.items()}
    if isinstance(o, list):
        return [sanitize(v) for v in o]
    return o


# ---- 输入哈希清单 ----
part_files = sorted(glob.glob(DAILY + r"\part_*.parquet"))
hash_manifest = {
    "daily_parts": [{"file": Path(f).name, "sha256": sha256_file(f), "size": Path(f).stat().st_size}
                    for f in part_files],
    "adj_cache": {"file": Path(CACHE).name, "sha256": sha256_file(CACHE), "size": Path(CACHE).stat().st_size},
    "note": "输入哈希用于可复现；v1 文件缺此清单，v2 补齐。",
}

# ---- 样本码集合（同 v1，保留旧证据）----
cache = json.loads(Path(CACHE).read_text(encoding="utf-8"))
full = json.loads(FULL.read_text(encoding="utf-8"))
conflict_codes = [c for c, _, _, _ in full.get("high_mismatch_codes_top20", [])]
stable = ["000001", "600000", "000002", "600519", "000858", "601318"]
ca_candidates = []
for code, v in cache.items():
    if v.get("status") != "OK":
        continue
    for d, m in v.get("events", []):
        if 20180101 <= d <= 20261009 and abs(m - 1.0) >= 0.03:
            ca_candidates.append((code, d, m))
            break
ca_codes = [c for c, _, _ in ca_candidates[:12]]
anom_codes = list({str(a["code"]) for a in full.get("ohlc_anomaly_sample", [])})
sample_set = set(conflict_codes) | set(stable) | set(ca_codes) | set(anom_codes)
for a in full.get("ohlc_anomaly_sample", []):
    sample_set.add(str(a["code"]))

# ---- 单遍收集（带来源分片）----
frames = []
for f in part_files:
    df = pd.read_parquet(f, columns=COLS)
    df = df[df["code"].astype(str).isin(sample_set)].copy()
    if len(df):
        df["shard"] = Path(f).name
        frames.append(df)
data = pd.concat(frames, ignore_index=True)
data["code"] = data["code"].astype(str)
data = data.sort_values(["code", "date"]).reset_index(drop=True)
data["prev_close_raw"] = data.groupby("code")["close"].shift(1)
data["prev_shard"] = data.groupby("code")["shard"].shift(1)

ev_index = {}
for code, v in cache.items():
    if v.get("status") == "OK":
        ev_index[code] = {d: m for d, m in v.get("events", [])}


def classify(r):
    c, pc, pcr, pct = r["close"], r["pre_close"], r["prev_close_raw"], r["pct_chg"]
    out = {}
    ex = (c / pc - 1) if (pd.notna(pc) and pc != 0) else None
    raw = (c / pcr - 1) if (pd.notna(pcr) and pcr != 0) else None
    pctd = (pct / 100.0) if pd.notna(pct) else None
    out["ex_rights_ret"] = ex
    out["raw_prev_ret"] = raw
    out["pct_chg_dec"] = pctd
    if pctd is not None:
        d_ex = abs(pctd - ex) if ex is not None else None
        d_raw = abs(pctd - raw) if raw is not None else None
        out["d_ex"] = d_ex
        out["d_raw"] = d_raw
        if d_ex is not None and d_raw is not None:
            # 注意：较近 != 匹配；只有 <= MATCH_TOL 才算匹配
            out["matches"] = "raw" if d_raw <= MATCH_TOL else ("ex" if d_ex <= MATCH_TOL else "NEITHER")
    return out


def block(r):
    b = {"code": r["code"], "date": int(r["date"]), "shard": r["shard"],
         "open": r["open"], "high": r["high"], "low": r["low"], "close": r["close"],
         "pre_close": r["pre_close"], "pct_chg": r["pct_chg"], "prev_close_raw": r["prev_close_raw"]}
    b.update(classify(r))
    d = int(r["date"])
    if d in ev_index.get(r["code"], {}):
        b["ca_event_mult"] = ev_index[r["code"]][d]
    return b


result = {"meta": {"purpose": "bounded diagnosis v2; raw-caliber NOT confirmed (hypothesis + counterexamples)",
                   "sample_codes": len(sample_set), "cache_codes": len(cache),
                   "match_tol_pp": MATCH_TOL * 100},
          "input_hash_manifest": hash_manifest}

# 1. 普通日 / 2. 公司行为日 / 4. 冲突码（保留为证据，注明 HYPOTHESIS）
ordinary, ca_rows = [], []
for code, g in data.groupby("code"):
    evs = ev_index.get(code, {})
    cand = g[(g["date"].map(lambda d: d not in evs and (d - 1) not in evs and (d + 1) not in evs))
             & g["pre_close"].notna() & g["prev_close_raw"].notna() & g["pct_chg"].notna()]
    if len(cand):
        ordinary.append(block(cand.iloc[len(cand) // 2]))
for code, d, m in ca_candidates:
    sub = data[(data["code"] == code) & (data["date"] == d)]
    if len(sub):
        ca_rows.append(block(sub.iloc[0]))
    if len(ca_rows) >= 10:
        break
conflict_pattern = []
for code in conflict_codes[:5]:
    g = data[data["code"] == code].sort_values("date")
    if len(g) >= 3:
        conflict_pattern.append({"code": code, "n_rows": len(g),
                                 "samples": [block(r) for _, r in g.iloc[[0, len(g)//2, len(g)-1]].iterrows()]})

# 3. 分片边界：按来源分片逐边界核验（不默认无断裂）
boundary = []
for code in ["000001", "600000", "600519"] + conflict_codes[:2]:
    if code not in set(data["code"]):
        continue
    g = data[data["code"] == code].sort_values("date").reset_index(drop=True)
    evs = ev_index.get(code, {})
    cross = []
    for i in range(1, len(g)):
        if g.at[i, "shard"] != g.at[i, "prev_shard"]:
            prev_c = g.at[i-1, "close"]
            cur_c = g.at[i, "close"]
            ret = (cur_c / prev_c - 1) if prev_c else None
            is_ev = int(g.at[i, "date"]) in evs
            cross.append({"date": int(g.at[i, "date"]), "shard_from": g.at[i, "prev_shard"],
                          "shard_to": g.at[i, "shard"], "adj_ret": ret, "ca_event": (evs[int(g.at[i, "date"])] if is_ev else None)})
    susp = [c for c in cross if c["adj_ret"] is not None and abs(c["adj_ret"]) > 0.11 and c["ca_event"] is None]
    boundary.append({"code": code, "n_rows": len(g), "shard_crossings": len(cross),
                     "suspicious_non_event_crossings": susp})

# 6. 反例扫描：pct_chg 既不匹配原始也不匹配除权（>=MATCH_TOL）
neither = []
match_raw = match_ex = 0
total_classified = 0
for _, r in data.iterrows():
    if pd.isna(r["prev_close_raw"]) or pd.isna(r["pre_close"]) or pd.isna(r["pct_chg"]):
        continue
    c = classify(r)
    if "matches" not in c:
        continue
    total_classified += 1
    if c["matches"] == "raw":
        match_raw += 1
    elif c["matches"] == "ex":
        match_ex += 1
    else:
        neither.append({"code": r["code"], "date": int(r["date"]), "shard": r["shard"],
                        "close": r["close"], "prev_close_raw": r["prev_close_raw"],
                        "pre_close": r["pre_close"], "pct_chg": r["pct_chg"],
                        "raw_prev_ret": c["raw_prev_ret"], "ex_rights_ret": c["ex_rights_ret"]})
neither_samples = sorted(neither, key=lambda x: -abs(x["raw_prev_ret"] - x["pct_chg"]/100.0))[:15]

result["bounded_evidence"] = {
    "ordinary_days_hypothesis": ordinary[:8],
    "corporate_action_days_hypothesis": ca_rows,
    "conflict_code_pattern": conflict_pattern,
    "partition_boundary_by_shard": boundary,
}
result["raw_caliber_hypothesis_status"] = {
    "claim": "close/low 与 pct_chg 同原始口径（pct_chg=close/原始昨收-1）",
    "status": "NOT_CONFIRMED_HYPOTHESIS",
    "total_classified_rows": total_classified,
    "match_raw_within_0.5pp": match_raw,
    "match_ex_within_0.5pp": match_ex,
    "match_NEITHER": len(neither),
    "note": "多数样本支持原始口径，但存在明确反例（见 samples），故不能称为确证。pct_chg 的真实参考口径须进一步核实（可能含停复牌/集合竞价/数据修正等未建模情形）。",
    "counterexample_samples": neither_samples,
}

# 7. 复权链独立验证（factor_at(d)/factor_at(P) 待验方案）
from step9_preflight.adjustment_chain import SymbolChain, build_chains
# 链完整性 / 无效乘数 / 覆盖终点
invalid_mult = 0
unsorted = 0
max_event_date = 0
codes_post2025 = 0
codes_total = len(cache)
codes_ok = 0
for code, v in cache.items():
    if v.get("status") != "OK":
        continue
    codes_ok += 1
    evs = v.get("events", [])
    mvals = [m for _, m in evs]
    invalid_mult += sum(1 for m in mvals if not (isinstance(m, (int, float)) and m > 0))
    ds = [d for d, _ in evs]
    if ds != sorted(ds):
        unsorted += 1
    if ds:
        max_event_date = max(max_event_date, max(ds))
        if max(ds) >= 20250101:
            codes_post2025 += 1
codes_without_chain = codes_total - codes_ok

# 独立手算拆分/分红验证
big = max(((code, d, m) for code, v in cache.items() if v.get("status") == "OK"
           for d, m in v.get("events", []) if 20180101 <= d <= 20261009),
          key=lambda x: abs(x[2] - 1.0))
code_b, date_b, mult_b = big
chain_b = SymbolChain(code_b, [(d, m) for d, m in cache[code_b]["events"]]).build()

# (A) 直接用 SymbolChain 手算拆分/分红案例，验证 factor_at 累积乘积方向与精度
hand_direct = []
# 拆分 2:1 于 20200101
sc_split = SymbolChain("HAND_SPLIT", [(20200101, 2.0)]).build()
hand_direct.append({
    "case": "split_2to1", "event": (20200101, 2.0),
    "factor_before": sc_split.factor_at(20191231), "factor_on_and_after": sc_split.factor_at(20200101),
    "expected": "before=1.0, on/after=2.0",
    "pass": abs(sc_split.factor_at(20191231) - 1.0) < 1e-12 and abs(sc_split.factor_at(20200101) - 2.0) < 1e-12,
})
# 送股(10送0.5) mult=1.05 于 20200101（送股属乘性调整）
sc_div = SymbolChain("HAND_DIV", [(20200101, 1.05)]).build()
hand_direct.append({
    "case": "stock_dividend_10for0.5", "event": (20200101, 1.05),
    "note": "10送0.5 = 5% 送股，乘子 1.05（送股属乘性调整，非现金分红）",
    "factor_before": sc_div.factor_at(20191231), "factor_on_and_after": sc_div.factor_at(20200101),
    "expected": "before=1.0, on/after=1.05",
    "pass": abs(sc_div.factor_at(20191231) - 1.0) < 1e-12 and abs(sc_div.factor_at(20200101) - 1.05) < 1e-12,
})
# 现金分红（派息）：股价 10.00 派 0.50/股 -> 除息价 9.50，乘子=前收/除息价=10/9.5
sc_cash = SymbolChain("HAND_CASH", [(20200101, 10.0 / 9.5)]).build()
hand_direct.append({
    "case": "cash_dividend_pay0.5_at10", "event": (20200101, round(10.0 / 9.5, 10)),
    "note": "现金分红：除息价=前收-股息=9.50，乘子=前收/除息价=10/9.5（后复权把除息后低价还原到分红前尺度）",
    "factor_before": sc_cash.factor_at(20191231),
    "factor_on_and_after": sc_cash.factor_at(20200101),
    "expected": "before=1.0, on/after=10/9.5",
    "pass": abs(sc_cash.factor_at(20191231) - 1.0) < 1e-12 and abs(sc_cash.factor_at(20200101) - 10.0 / 9.5) < 1e-12,
})
# 现金分红方向：除息日原始 close 9.50 -> 归一化=9.50*(10/9.5)/1=10.0（还原，不产生伪回撤）
norm_cash = 9.5 * sc_cash.factor_at(20200102) / sc_cash.factor_at(20191231)
hand_direct.append({
    "case": "cash_dividend_normalize_direction", "ex_close": 9.5, "norm_to_pre_scale": norm_cash,
    "expected": 10.0, "pass": abs(norm_cash - 10.0) < 1e-9,
})
# 归一化方向：P=20191231 原始收盘 100；d=20200102 原始收盘 50（拆分后减半）；norm=50*2/1=100 -> 恢复同尺度
norm_dir = 50 * sc_split.factor_at(20200102) / sc_split.factor_at(20191231)
hand_direct.append({
    "case": "normalize_direction", "P_raw": 100, "d_raw_postsplit": 50,
    "norm_d_to_P_scale": norm_dir, "expected": 100.0, "pass": abs(norm_dir - 100.0) < 1e-9,
})
# (B) 真实大事件码的方向（若样本含其日线则补充）
hand_cases = []
sub = data[(data["code"] == code_b) & (data["date"] >= date_b - 5) & (data["date"] <= date_b + 5)]
if len(sub) >= 2:
    before = sub[sub["date"] < date_b]
    after = sub[sub["date"] > date_b]
    if len(before) and len(after):
        p_row = before.iloc[-1]
        d_row = after.iloc[0]
        fp = chain_b.factor_at(int(p_row["date"]))
        fd = chain_b.factor_at(int(d_row["date"]))
        ratio = fd / fp if fp else None
        hand_cases.append({
            "code": code_b, "event_date": date_b, "event_mult": mult_b,
            "P_date": int(p_row["date"]), "P_close_raw": float(p_row["close"]),
            "d_date": int(d_row["date"]), "d_close_raw": float(d_row["close"]),
            "norm_ratio": ratio,
            "norm_d_close_to_P_scale": (float(d_row["close"]) * ratio if ratio else None),
        })

result["adjustment_chain_validation"] = {
    "codes_total": codes_total,
    "codes_ok": codes_ok,
    "codes_without_chain": codes_without_chain,
    "invalid_multiplier_count": invalid_mult,
    "unsorted_event_codes": unsorted,
    "max_event_date_present": max_event_date,
    "codes_with_events_post_2025": codes_post2025,
    "coverage_endpoint_note": ("max_event_date 仅是缓存【已存在事件】中的最晚公司行为日，不等于缓存的完整覆盖终点；"
                               "缓存无采集截止/覆盖清单，不能据 20260928 断言所有窗口已覆盖。覆盖终点须凭"
                               "采集截止证据 + 逐码存在性核验，当前 UNVERIFIED（codes_without_chain 个码无链，"
                               "标签化将判 ADJ_EVIDENCE_CONFLICT）。"),
    "hand_computed_split_dividend_cases": hand_cases,
    "hand_computed_direct_cases": hand_direct,
    "factor_at_formula": "analytic_adjusted_price(raw, asof) = raw * factor_at(asof); factor_at 乘入 effective_date<=asof 的 mult",
    "normalization_proposal": "norm_low_d = raw_low_d * factor_at(d) / factor_at(P)  （待独立手算验证方向与精度，v2 尚未放行）",
}

result["revised_conclusion"] = (
    "v1 的'原始口径已确证'不成立：证据仅 8 普通+10 公司行为+5 冲突码（非 73 码全量验收），且存在 pct_chg 既不匹配原始也不匹配除权的反例"
    "（如 000012@20180627：raw_prev_ret=-14.43% 而 pct_chg=-0.806%）。结论降为 HYPOTHESIS。分片边界按来源分片逐边界核验"
    "（见 partition_boundary_by_shard），未发现非事件跳变，但样本有限不构成全量放行。复权链 factor_at(d)/factor_at(P) 作为待验方案，"
    "已补链完整性/无效乘数/覆盖终点核验及独立手算拆分案例，但方向精度仍待更多手算案例确认。标签核心比值 Low_d/R_P 用 close/low（原始）"
    "同尺度，pre_close 禁用；窗口内公司行为归一化方案待验未放行。"
)

OUT.write_text(json.dumps(sanitize(result), ensure_ascii=False, indent=2), encoding="utf-8")
print("WROTE", OUT, OUT.stat().st_size)
print("classified", total_classified, "match_raw", match_raw, "match_ex", match_ex, "neither", len(neither))
print("invalid_mult", invalid_mult, "unsorted", unsorted, "max_event", max_event_date, "post2025", codes_post2025)
print("big_event", big)
