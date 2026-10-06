"""GREEN-COMMON-PROFILE-v0.1 第一轮A: 共性选池对比测试 (用户 2026-09-28 授权).

比较 (同外层宇宙/入场/退出/成本):
  OLD = pool_daily_v11 primary_tier∈{WATCH,HPW}
  A_SDP = 新强势首次分歧池   B_LVB = 新低位稳量底部池
交付 (规格《第一批必须交付》子集, 日线层):
  参数冻结表(已预注册) / 两类名单年度漏斗 / 缺失/常量/相关性报告 /
  因果截断对拍 / H3(主)+H5(复核) 成本后收益对比 / 集中度(月/股)
不做: 个性画像(P1)、流动性区域(P2)、P0-P3 消融(第一轮B)、训练。
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "step9_preflight"))

from gcp_v0_1_spec import SPEC, spec_sha256  # noqa: E402
from gcp_v0_1_pools import process_code  # noqa: E402
from trading_calendar import load_calendar  # noqa: E402
import data_source_provenance as DSP  # noqa: E402

POOL_V11 = HERE.parent / "green_watch_v0" / "pool_daily_v11.parquet"
OUT_DIR = HERE / "round_a_v0_1_impl_fix"        # v0.1 原产物保留于 round_a_v0_1/, 不覆盖
OUT_DIR.mkdir(exist_ok=True)
V01_DIR = HERE / "round_a_v0_1"                 # v0.1 原结果 (MAD 修复前), 仅读取对照
COMMON_START, COMMON_END = 20210104, 20260917   # 旧池窗口 (新池全史单独报漏斗)
BUY_F, SELL_F = 1.0013, 0.9982                  # INTRADAY_SD_COST_ASSUMPTIONS_V1


# ------------------------------------------------------------------ 标签
def label_events(ev: pd.DataFrame, cal, calpos, bars) -> pd.DataFrame:
    """研究近似日线标签: E=下一交易日开盘买入, H3/H5=收盘卖出, 一字板剔除, T+1 自动满足."""
    recs = []
    for r in ev.itertuples(index=False):
        d, code = int(r.date), str(r.code).zfill(6)
        stype = getattr(r, "setup_type", "NA")
        i = calpos.get(d)
        b = bars.get(code)
        if i is None or b is None or i + 6 >= len(cal):
            recs.append({"date": d, "code": code, "setup_type": stype, "status": "NO_CALENDAR_OR_BARS"})
            continue
        dmap, O, H, L, C = b
        E = cal[i + 1]
        if E not in dmap:
            recs.append({"date": d, "code": code, "setup_type": stype, "status": "UNFILLED_SUSPENDED"})
            continue
        e = dmap[E]
        o_e, h_e, l_e = O[e], H[e], L[e]
        if o_e == h_e == l_e:
            recs.append({"date": d, "code": code, "setup_type": stype, "status": "EXCLUDED_LIMIT_BOARD"})
            continue
        out = {"date": d, "code": code, "setup_type": stype, "status": "FILLED",
               "entry_date": E, "entry_open": float(o_e)}
        lows, highs = [l_e], [h_e]
        for tag, k in (("h3", 3), ("h5", 5)):
            xd = cal[i + 1 + k]
            if xd not in dmap:
                out[f"net_{tag}"] = None
                out[f"exit_date_{tag}"] = None
                continue
            x = dmap[xd]
            out[f"net_{tag}"] = float(C[x] * SELL_F / (o_e * BUY_F) - 1.0)
            out[f"exit_date_{tag}"] = xd
            lows.append(L[x]); highs.append(H[x])
            for dd in cal[i + 2:i + 1 + k]:            # 持有窗口内逐日 low/high (停牌bar不计)
                if dd in dmap:
                    j = dmap[dd]; lows.append(L[j]); highs.append(H[j])
        out["mae"] = float(min(lows) / o_e - 1.0) if lows else None
        out["mfe"] = float(max(highs) / o_e - 1.0) if highs else None
        recs.append(out)
    return pd.DataFrame(recs)


def econ_stats(lab: pd.DataFrame, tag: str) -> dict:
    f = lab[lab["status"] == "FILLED"]
    x = f[f[f"net_{tag}"].notna()] if "net_h3" in f else pd.DataFrame()
    s = x[f"net_{tag}"].astype(float)
    return {
        "n": int(len(s)), "mean": round(float(s.mean()), 6) if len(s) else None,
        "median": round(float(s.median()), 6) if len(s) else None,
        "win_rate": round(float((s > 0).mean()), 4) if len(s) else None,
        "p5": round(float(s.quantile(0.05)), 6) if len(s) else None,
        "p10": round(float(s.quantile(0.10)), 6) if len(s) else None,
        "mae_p50": round(float(x["mae"].median()), 6) if len(x) else None,
        "mfe_p50": round(float(x["mfe"].median()), 6) if len(x) else None,
    }


def main():
    t0 = time.time()
    cal = load_calendar()
    calpos = {d: i for i, d in enumerate(cal)}
    raw = DSP.get_daily_parquet()
    raw["code"] = raw["code"].astype(str).str.zfill(6)

    # bars 索引 (标签用)
    bars = {}
    for code, g in raw.groupby("code", sort=False):
        g = g.sort_values("date")
        bars[code] = ({int(d): i for i, d in enumerate(g["date"].to_numpy())},
                      g["open"].to_numpy(float), g["high"].to_numpy(float),
                      g["low"].to_numpy(float), g["close"].to_numpy(float))

    # ---------------- 构建两池 (全史) ----------------
    a_parts, b_parts = [], []
    for code, g in raw.groupby("code", sort=False):
        g = g.sort_values("date").reset_index(drop=True)
        a, b = process_code(g)
        if len(a):
            a_parts.append(a)
        if len(b):
            b_parts.append(b)
    pools_a = pd.concat(a_parts, ignore_index=True) if a_parts else pd.DataFrame()
    pools_b = pd.concat(b_parts, ignore_index=True) if b_parts else pd.DataFrame()
    print(f"pools built: A={len(pools_a)} B={len(pools_b)} ({time.time()-t0:.0f}s)", flush=True)
    pools_a.to_parquet(OUT_DIR / "pool_a_sdp_members.parquet", index=False)
    pools_b.to_parquet(OUT_DIR / "pool_b_lvb_members.parquet", index=False)

    # ---------------- 旧池 + 外层宇宙对齐 ----------------
    old = pd.read_parquet(POOL_V11, columns=["date", "code", "primary_tier", "primary_setup"])
    old = old[old["primary_tier"].isin(["WATCH", "HIGH_PRIORITY_WATCH"])].copy()
    old["code"] = old["code"].astype(str).str.zfill(6)
    outer_all = raw[["date", "code", "is_st", "float_mv", "amount", "volume"]].copy()
    outer_all["outer_ok"] = ((outer_all["is_st"].astype(float) == 0)
                             & (outer_all["float_mv"].astype(float) > 0)
                             & (outer_all["amount"].astype(float) >= 5e7)
                             & (outer_all["volume"].astype(float) > 0))
    outer_cnt = outer_all[outer_all["outer_ok"]].groupby("date").size()
    old = old.merge(outer_all[outer_all["outer_ok"]][["date", "code"]],
                    on=["date", "code"], how="inner")          # 旧池套同一外层宇宙
    old["setup_type"] = "OLD_WATCH_HPW"

    cw = (raw["date"] >= COMMON_START) & (raw["date"] <= COMMON_END)
    n_outer_cw = int(outer_all[cw & outer_all["outer_ok"]].shape[0])

    pools_a_cw = pools_a[(pools_a["date"] >= COMMON_START) & (pools_a["date"] <= COMMON_END)]
    pools_b_cw = pools_b[(pools_b["date"] >= COMMON_START) & (pools_b["date"] <= COMMON_END)]

    # ---------------- 漏斗 (年度) ----------------
    def yearly(df):
        if df.empty:
            return {}
        y = df.copy(); y["year"] = y["date"] // 10000
        return {int(k): int(v) for k, v in y.groupby("year").size().items()}

    ab_keys = set(zip(pools_a_cw["date"], pools_a_cw["code"]))
    bb_keys = set(zip(pools_b_cw["date"], pools_b_cw["code"]))
    ob_keys = set(zip(old["date"], old["code"]))
    funnel = {
        "universe_days": int(old["date"].nunique()),
        "outer_universe_rows_common_window": n_outer_cw,
        "per_day_mean": {
            "OLD": round(len(old) / max(old["date"].nunique(), 1), 1),
            "A_SDP": round(len(pools_a_cw) / max(old["date"].nunique(), 1), 1),
            "B_LVB": round(len(pools_b_cw) / max(old["date"].nunique(), 1), 1),
        },
        "yearly_counts": {"OLD": yearly(old), "A_SDP": yearly(pools_a_cw),
                          "B_LVB": yearly(pools_b_cw)},
        "yearly_counts_full_history_new_pools": {
            "A_SDP": yearly(pools_a), "B_LVB": yearly(pools_b)},
        "overlap": {
            "A_and_B": len(ab_keys & bb_keys),
            "OLD_and_A": len(ob_keys & ab_keys), "OLD_and_B": len(ob_keys & bb_keys),
            "OLD_and_AorB": len(ob_keys & (ab_keys | bb_keys)),
            "A_or_B": len(ab_keys | bb_keys),
        },
    }

    # ---------------- 标签 + 经济学 ----------------
    ev = pd.concat([old[["date", "code", "setup_type"]],
                    pools_a_cw[["date", "code", "setup_type"]],
                    pools_b_cw[["date", "code", "setup_type"]]], ignore_index=True)
    lab = label_events(ev, cal, calpos, bars)
    lab = lab.drop_duplicates(subset=["date", "code", "setup_type"])
    lab.to_parquet(OUT_DIR / "labels_research_v0.parquet", index=False)

    econ = {}
    for name, sub in lab.groupby("setup_type"):
        econ[name] = {"h3": econ_stats(sub, "h3"), "h5": econ_stats(sub, "h5"),
                      "status_counts": {k: int(v) for k, v in sub["status"].value_counts().items()}}
    # 组成视角: 仅旧池 vs 仅新池 (同股同日标签相同, 故差异来自名单组成)
    new_keys = ab_keys | bb_keys
    old_only = lab[(lab["setup_type"] == "OLD_WATCH_HPW")
                   & ~lab.apply(lambda r: (r["date"], r["code"]) in new_keys, axis=1)]
    new_only_a = lab[(lab["setup_type"] == "A_SDP")
                     & ~lab.apply(lambda r: (r["date"], r["code"]) in ob_keys, axis=1)]
    new_only_b = lab[(lab["setup_type"] == "B_LVB")
                     & ~lab.apply(lambda r: (r["date"], r["code"]) in ob_keys, axis=1)]
    econ["OLD_ONLY(not in A/B)"] = {"h3": econ_stats(old_only, "h3")}
    econ["A_ONLY(not in OLD)"] = {"h3": econ_stats(new_only_a, "h3")}
    econ["B_ONLY(not in OLD)"] = {"h3": econ_stats(new_only_b, "h3")}

    # 年度 H3 一致性 + 集中度 (事件占比 + 盈利贡献占比 + 剔除最佳股/月检验) + 日级聚类
    yearly_econ, concentration, day_level = {}, {}, {}
    for name, sub in lab.groupby("setup_type"):
        f = sub[(sub["status"] == "FILLED") & sub["net_h3"].notna()].copy()
        f["year"] = f["date"] // 10000
        yearly_econ[name] = {int(y): round(float(g["net_h3"].mean()), 5)
                             for y, g in f.groupby("year")}
        by_stock = f.groupby("code")["net_h3"].agg(["sum", "count", "mean"]).sort_values("sum", ascending=False)
        total_sum = float(f["net_h3"].sum())
        top10_ev_share = float(by_stock["count"].head(10).sum() / max(len(f), 1))
        # 盈利贡献占比 (修复#3): top10 股净收益合计 / 全池净收益合计 (含正负抵消, 同时报正值口径)
        top10_profit_share = (float(by_stock["sum"].head(10).sum() / total_sum)
                              if abs(total_sum) > 1e-12 else None)
        m = f.groupby(f["date"] // 100)["net_h3"].agg(["mean", "count"]).sort_values("mean", ascending=False)
        best_month = int(m.index[0]) if len(m) else None
        best_stock = str(by_stock.index[0]) if len(by_stock) else None
        # 剔除最佳股 / 最佳月后的 H3 均值 (是否被单一股/月撑起)
        excl_stock_mean = (float(f.loc[f["code"] != best_stock, "net_h3"].mean())
                           if best_stock else None)
        excl_month_mean = (float(f.loc[(f["date"] // 100) != best_month, "net_h3"].mean())
                           if best_month else None)
        # 日级聚类不确定性 (修复: 连续多日同一股票不能当独立证据): 按日聚合后再统计
        dser = f.groupby("date")["net_h3"].mean()
        dmean, dstd, dn = float(dser.mean()), float(dser.std(ddof=1)), int(len(dser))
        t_day = (dmean / (dstd / np.sqrt(dn))) if (dn > 1 and dstd > 0) else None
        mser = f.groupby(f["date"] // 100)["net_h3"].mean()
        mmean, mstd, mn = float(mser.mean()), float(mser.std(ddof=1)), int(len(mser))
        t_month = (mmean / (mstd / np.sqrt(mn))) if (mn > 1 and mstd > 0) else None
        concentration[name] = {
            "top10_stock_event_share": round(top10_ev_share, 4),
            "top10_stock_profit_share_of_total_sum": (round(top10_profit_share, 4)
                                                      if top10_profit_share is not None else None),
            "best_stock": best_stock,
            "best_stock_event_count": int(by_stock["count"].iloc[0]) if len(by_stock) else 0,
            "best_month": best_month,
            "best_month_mean": round(float(m["mean"].iloc[0]), 5) if len(m) else None,
            "worst_month_mean": round(float(m["mean"].iloc[-1]), 5) if len(m) else None,
            "h3_mean_excl_best_stock": round(excl_stock_mean, 6) if excl_stock_mean is not None else None,
            "h3_mean_excl_best_month": round(excl_month_mean, 6) if excl_month_mean is not None else None,
        }
        day_level[name] = {
            "n_days": dn, "daily_mean_of_daily_means": round(dmean, 6),
            "daily_std": round(dstd, 6), "t_stat_date_clustered": round(t_day, 3) if t_day else None,
            "n_months": mn, "monthly_mean": round(mmean, 6), "monthly_std": round(mstd, 6),
            "t_stat_month_clustered": round(t_month, 3) if t_month else None,
        }

    # ---------------- 特征质量报告 (A∪B 成员行) ----------------
    members = pd.concat([pools_a_cw, pools_b_cw], ignore_index=True)
    feat_cols = [c for c in members.columns if c not in ("date", "code", "setup_type", "div_date")]
    quality = {"n_rows": int(len(members)),
               "null_rate": {c: round(float(members[c].isna().mean()), 4) for c in feat_cols},
               "n_unique": {c: int(members[c].nunique(dropna=True)) for c in feat_cols}}
    num = members[feat_cols].select_dtypes(include=[np.number]).dropna(axis=1, how="all")
    num = num.loc[:, num.nunique(dropna=True) > 1]
    corr = num.corr(method="spearman").abs()
    pairs = []
    seen = set()
    for i in corr.index:
        for j in corr.columns:
            if i < j and corr.loc[i, j] > 0.8:
                pairs.append((i, j, round(float(corr.loc[i, j]), 3)))
    quality["high_corr_pairs_gt_0.8"] = pairs

    # ------------- 因果验收 (修复#2: 全量对拍 + 未来扰动, 5 码) -------------
    def _norm(df: pd.DataFrame) -> pd.DataFrame:
        """成员行标准化: 排序+重置索引+固定列序, 供逐位比较."""
        if "date" not in getattr(df, "columns", []):
            return pd.DataFrame()
        return df.sort_values("date").reset_index(drop=True)

    def _compare(full: pd.DataFrame, trunc: pd.DataFrame, upto: int) -> dict:
        """比较截至 upto 的候选: 键(date,code,div_date) + 全部特征值逐位一致."""
        f_ = _norm(full[full["date"] <= upto]) if len(full) else pd.DataFrame()
        t_ = _norm(trunc)
        if len(f_) != len(t_):
            return {"identical": False, "reason": f"row_count full={len(f_)} trunc={len(t_)}"}
        if len(f_) == 0:
            return {"identical": True, "n_rows": 0}
        cols = [c for c in f_.columns if c in t_.columns]
        f_, t_ = f_[cols], t_[cols]
        try:
            pd.testing.assert_frame_equal(f_, t_, check_exact=True)
            return {"identical": True, "n_rows": int(len(f_)), "cols_checked": len(cols)}
        except AssertionError as e:
            return {"identical": False, "reason": str(e)[:200]}

    causal_ok, causal_detail = True, []
    rng = np.random.default_rng(20260928)
    sample_codes = list(rng.choice(sorted(set(members["code"])), size=min(5, members["code"].nunique()),
                                   replace=False)) if len(members) else []
    for code in sample_codes:
        g = raw[raw["code"] == code].sort_values("date").reset_index(drop=True)
        rows = members[(members["code"] == code)].sort_values("date")
        D = int(rows["date"].max())
        a_full, b_full = process_code(g)
        a_trunc, b_trunc = process_code(g[g["date"] <= D].reset_index(drop=True))
        # (i) 截断重算: 键 + 全部特征值逐位一致 (不再只比数量)
        cmp_a = _compare(a_full, a_trunc, D)
        cmp_b = _compare(b_full, b_trunc, D)
        # (ii) 未来扰动: D 之后价格×1.07/量×2.5, 截至 D 的候选必须不变
        g_pert = g.copy()
        msk = g_pert["date"] > D
        for col in ("open", "high", "low", "close", "pre_close", "volume"):
            g_pert[col] = g_pert[col].astype(float)          # 整列先转 float, 避免 int 列掩码赋值报错
            g_pert.loc[msk, col] = g_pert.loc[msk, col] * (2.5 if col == "volume" else 1.07)
        a_pert, b_pert = process_code(g_pert)
        cmp_pa = _compare(a_pert, a_trunc, D)      # 与截断版比: 截至 D 须逐位一致
        cmp_pb = _compare(b_pert, b_trunc, D)
        same = all(c["identical"] for c in (cmp_a, cmp_b, cmp_pa, cmp_pb))
        causal_detail.append({"code": code, "D": D,
                              "truncation": {"A": cmp_a, "B": cmp_b},
                              "future_perturbation": {"A": cmp_pa, "B": cmp_pb},
                              "identical": bool(same)})
        causal_ok = causal_ok and same

    # ------------- B 池成员差异对照 (MAD 算法修复的影响面) -------------
    b_diff = {"note": "MAD 修复仅影响 B_LVB (vol_stability); A_SDP 不含该字段, 成员不变"}
    old_b_path = V01_DIR / "pool_b_lvb_members.parquet"
    if old_b_path.exists():
        b_old = pd.read_parquet(old_b_path, columns=["date", "code"])
        ko = set(zip(b_old["date"].astype(int), b_old["code"].astype(str)))
        kn = set(zip(pools_b["date"].astype(int) if len(pools_b) else [],
                     pools_b["code"].astype(str) if len(pools_b) else []))
        b_diff.update({"v0_1_members": len(ko), "v0_1_1_members": len(kn),
                       "kept": len(ko & kn), "removed": len(ko - kn), "added": len(kn - ko)})
        pools_b_cw = pools_b_cw  # 下方标签/统计均用修复后成员

    # ---------------- 报告 ----------------
    report = {
        "name": "GCP_V0_1_ROUND_A_IMPL_FIX (v0.1.1)",
        "spec_sha256": spec_sha256(),
        "spec_sha256_note": "spec 未改动, 与 v0.1 相同哈希 => 阈值零调整, 本轮只修实现与验收",
        "supersedes_for_review": "round_a_v0_1/GCP_V0_1_ROUND_A_REPORT.json (原结果保留归档)",
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "data_epoch": DSP.data_epoch_fingerprint() if hasattr(DSP, "data_epoch_fingerprint") else None,
        "common_window": [COMMON_START, COMMON_END],
        "implementation_fixes": {
            "1_mad_stability": "同一三日窗口内统一减该窗口中位数再取中位 (原实现=各日减各自滚动中位再滚动中位, 属算法bug); 阈值0.30不变, 仅影响B_LVB",
            "2_causal_check": "从'只比候选数量'升级为: 候选键+分歧日期+全部特征值逐位对拍(assert_frame_equal check_exact) + 未来数据扰动(D后价格x1.07/量x2.5, 截至D候选必须不变), 5码",
            "3_concentration": "补盈利贡献占比(top10股净收益/全池净收益)、剔除最佳股/最佳月后H3均值",
            "4_uncertainty": "新增日级/月级聚类统计: 按日聚合后 t 统计, 连续多日同股不再被当独立证据",
            "labels_unchanged": "标签仍为日线级粗代理(RESEARCH_APPROXIMATION); 正式验收须接既有执行引擎与公司行为约束(退出跌停/除权), 本轮不冒充",
        },
        "definition_clarifications": {
            "boll_lower_dist_signed": "B 池下轨条件为有符号距离<=1ATR: 跌破下轨很远(负值)同样通过, 不完全等于'接近下轨'; 此为 v0.1 预注册定义, 本轮不改",
            "a_pullback_volume": "A 池仅要求调整段量中位数<=0.8x分歧日量, 未要求逐日收缩; 若要逐日收缩属新定义, 须另立版本",
            "a_b_overlap": "A∩B=0 是本样本观察, 非数学必然(A强势条件在分歧前, 与当前低RSI不构成构造性互斥)",
        },
        "b_pool_membership_diff_vs_v0_1": b_diff,
        "funnel": funnel, "economics": econ, "yearly_h3_mean": yearly_econ,
        "concentration": concentration, "day_month_clustered": day_level,
        "feature_quality": quality,
        "causal_truncation_check": {"all_identical": causal_ok, "detail": causal_detail},
        "caveats": [
            "标签为日线级研究近似(RESEARCH_APPROXIMATION): 无分钟成交/限价逐笔核验, 一字板剔除, 停牌bar不计入MAE/MFE",
            "2024-2026 为多轮开发参与区间(#10.4), 该段结果不作为纯净测试; 本轮为描述性对比, 无时间隔离训练",
            "全部 FIRST_ROUND_PROPOSAL 阈值看结果前预注册且本轮未动; 证伪即整轮作废, 不做阈值搜索",
            "本报告只检验共性选池经济学(第一轮A); P0-P3 个性买点消融属第一轮B, 未运行",
            "裁决范围收窄(用户2026-09-28): 裸池次日开盘买入亏损≠否定'等到个性区域+盘中承接'的择时假设, 该假设尚未被检验",
        ],
    }
    (OUT_DIR / "GCP_V0_1_ROUND_A_REPORT.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    print(json.dumps({k: report[k] for k in
                      ("funnel", "economics", "concentration", "causal_truncation_check")},
                     ensure_ascii=False, indent=1, default=str))
    print(f"\nreport -> {OUT_DIR / 'GCP_V0_1_ROUND_A_REPORT.json'} ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
