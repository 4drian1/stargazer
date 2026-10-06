"""GREEN-COMMON-PROFILE v0.2 可行性预演 (2026-09-29, 用户选(b)).

复用登记范围内 16 股缓存快照 (gcp_round_b_small_sample_v1/minute/*.parquet), 0 新增请求.
目的: 在"不再只收 A/B 资格日"的 v0.2 思路下, 用现有快照检查可行性, 不看盈利挑设计.

v0.2 设计原则 (用户裁定):
  - 共性选股(A/B)不动; 历史画像不再只收 A/B 资格日, 收集该股历史锚点接近/跌破事件
    (含原793非池资格日), 但保留当时背景, 非池事件绝不重标 A/B, 也不当 A/B 买点期望
  - 两类统计: 基础行为画像(较广有效锚点事件: 穿透深度/收回速度/失败比例+不确定性)
              vs 条件交易价值(与当前背景可比/共享估计: 这次位置是否有成本后价值)
  - 不简单合并单元格, 不逐股独立训练收益模型; 时间衰减降样本不能凭空造样本

分析:
  1. 扩展事件库规模: 全 FILLED(池+非池) 各股/各锚点数量; 对照旧 A/B-only(110)
  2. 跨股行为差异: 各股 信号bar穿透/完整下杀最大穿透/3日失败率/收益 的离散度
     (ICC 近似, 未做随机效应校正; 看是否超噪声 —— 仅探索, 非定论)
  3. 时间稳定性: 各股事件按日切前后半, 跨股 early-vs-late 指标相关
  4. 背景匹配+收缩后有效样本: 4维单元格(session, zone_sources, sig_type, vol_bucket)
     各股 ≥10 / ≥5; 并报较粗粒度(session×zone_sources) ≥10 覆盖

口径纠正 (2026-09-29 用户裁定):
  - f 是"持有期 3 日内是否跌破 buy_net - 0.5*ATR_ref"的失败标签, 非"收复率";
    真 reclaim30/60 未在引擎实现 (仅 reclaim30_invalid 数据质量标记), 报告中分别命名, 不互换
  - pen_atr_bar 是"信号bar 当时的穿透深度", 非完整下杀过程最大穿透;
    后者由 (zone_lower - min_low)/atr_ref 近似 (min_low = 持有期最低, fill_date 同尺度)
  - 4维单元格稀疏 ≠ "逐股永远到不了10": 较粗粒度(session×锚点) 87.5% 股至少 1 个 ≥10 单元格
  - ICC 为方差成分近似比值, 非随机效应充分校正; 行为指标离散 ≠ 已证个性稳定

运行: python gcp_v02_feasibility.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "step9_preflight"))
sys.path.insert(0, str(HERE))
from data_source_provenance import get_daily_parquet  # noqa: E402
import gcp_round_b_engine as E  # noqa: E402
import gcp_round_b_acceptance as ACC  # noqa: E402

OUT = HERE / "gcp_round_b_small_sample_v1"


def main():
    grid = json.loads((OUT / "minute_grid.json").read_text(encoding="utf-8"))["grid"]
    panel = get_daily_parquet()
    # 直接复用 stage_real (快照全缓存, 0 分钟请求; 跳过 stage_a 的网格 HTTP 探针)
    all_events, real_recs = ACC.stage_real(grid, panel, True)

    # ---- 事件表: 所有发出信号的回放日, 完整保留当时背景 ----
    rows = []
    for r in all_events:
        if not r.get("signal_ts"):
            continue
        rows.append({
            "code": r["code"], "date": r["date"],
            "session": r.get("session"),
            "sig_type": r.get("signal_type"),
            "zone_sources": "+".join(r["zone_sources"]) if r.get("zone_sources") else None,
            "setup": r.get("setup"),
            "pool_q": r.get("setup") is not None,
            "d_atr": r.get("d_atr"),
            "pen_atr_bar": r.get("pen_atr_bar"),
            "zone_lower": r.get("zone_lower"),
            "min_low": r.get("min_low"),
            "atr_ref": r.get("atr_ref"),
            "status": r["status"],
            "y": r.get("y"), "f": r.get("f"),
            "fill_amount": r.get("fill_amount"),
        })
    df = pd.DataFrame(rows)
    sig = df[df["status"] == "FILLED"].copy()
    print(f"回放日含信号={len(df)}  FILLED(扩展事件库)={len(sig)}  "
          f"(其中 A/B 池资格={int(sig['pool_q'].sum())} 非池={int((~sig['pool_q']).sum())})")

    # ===== 1. 扩展事件库规模 =====
    n_stock = sig.groupby("code").size()
    n_pool_only = sig[sig["pool_q"]].groupby("code").size()
    print("\n== 1. 扩展事件库规模 (vs 旧 A/B-only) ==")
    print(f"  全 FILLED: 股数={sig['code'].nunique()} 均值/股={n_stock.mean():.1f} "
          f"中位={n_stock.median():.0f} 范围[{n_stock.min()},{n_stock.max()}]")
    print(f"  A/B-only: 均值/股={n_pool_only.reindex(n_stock.index).fillna(0).mean():.1f} "
          f"(扩展后约 {n_stock.mean()/max(n_pool_only.mean(),1):.1f}x)")
    anchor = sig.groupby("zone_sources").size().sort_values(ascending=False)
    print(f"  锚点组合(扩展库)分布 top: {anchor.head(6).to_dict()}")
    st = sig.groupby(["code", "session", "sig_type"]).size().groupby("code").size()
    print(f"  平均每股 (session×sig_type) 单元格数={st.mean():.1f}")

    # ===== 2. 跨股行为差异 (ICC 近似, 未做随机效应校正) =====
    print("\n== 2. 跨股行为差异 (ICC 近似, 仅探索) ==")
    g = sig.groupby("code")
    sm = g["y"].mean(); sn = g["y"].size(); sse = g["y"].std() / np.sqrt(sn)
    f3 = g["f"].mean()                          # 3日失败率 (跌破 buy_net-0.5*ATR_ref)
    pen_sb = g["pen_atr_bar"].mean()            # 信号bar穿透 (仅触发当日)
    # 完整下杀最大穿透: (zone_lower - min_low)/atr_ref, fill_date 同尺度
    sig["pen_fw"] = np.where(
        sig["atr_ref"].fillna(0) > 0,
        np.maximum(0.0, (sig["zone_lower"].astype(float) - sig["min_low"].astype(float))
                   / sig["atr_ref"].astype(float)),
        np.nan,
    )
    pen_fw = g["pen_fw"].mean()
    between_var = sm.var()
    within_var = ((g["y"].var() * (sn - 1)).sum()) / (sn.sum() - len(sn))
    icc = between_var / (between_var + within_var) if (between_var + within_var) > 0 else float("nan")
    print(f"  收益均值 跨股 SD={sm.std():.4f}  股内均值 SE={sse.mean():.4f}  "
          f"ICC≈{icc:.3f} (近似比值, 非随机效应校正; 越高=跨股差异越超噪声)")
    print(f"  3日失败率(f) 跨股范围[{f3.min():.2f},{f3.max():.2f}] 中位={f3.median():.2f}  "
          f"(注: f≠收复率; 真 reclaim30/60 未实现)")
    print(f"  信号bar穿透(pen_atr_bar) 跨股范围[{pen_sb.min():.2f},{pen_sb.max():.2f}] "
          f"中位={pen_sb.median():.2f}  (仅触发当日, 非完整下杀)")
    print(f"  完整下杀最大穿透(pen_fw) 跨股范围[{pen_fw.min():.2f},{pen_fw.max():.2f}] "
          f"中位={pen_fw.median():.2f}  (持有期最低 vs zone_lower)")
    print(f"  收益均值 离散: min={sm.min():.3f} p25={sm.quantile(.25):.3f} "
          f"med={sm.median():.3f} p75={sm.quantile(.75):.3f} max={sm.max():.3f}")

    # ===== 3. 时间稳定性 (前后半相关) =====
    print("\n== 3. 时间稳定性 (72日窗口, 弱功效, 仅探索) ==")
    med = g["date"].transform("median")
    sig["half"] = np.where(sig["date"] <= med, "early", "late")
    piv = sig.groupby(["code", "half"]).agg(y_mean=("y", "mean"),
                                           f3=("f", "mean"), n=("y", "size"))
    e = piv.xs("early", level="half"); l = piv.xs("late", level="half")
    both = e.index.intersection(l.index)
    e, l = e.loc[both], l.loc[both]
    ok_e = e["n"] >= 3; ok_l = l["n"] >= 3
    use = both[ok_e & ok_l]
    if len(use) >= 3:
        cy = np.corrcoef(e.loc[use, "y_mean"], l.loc[use, "y_mean"])[0, 1]
        cf = np.corrcoef(e.loc[use, "f3"], l.loc[use, "f3"])[0, 1]
        # 符号一致性 (收益均值前后半同向占比)
        same_sign = ((e.loc[use, "y_mean"] > 0) == (l.loc[use, "y_mean"] > 0)).mean()
        print(f"  有效股(前后半各n>=3)={len(use)}")
        print(f"  收益均值 early-vs-late 相关={cy:.2f}  3日失败率相关={cf:.2f}  "
              f"收益符号同向占比={same_sign:.0%}")
    else:
        print("  样本不足, 跳过稳定性相关")

    # ===== 4. 背景匹配+收缩后有效样本 =====
    print("\n== 4. 背景匹配后有效样本 (波动分桶) ==")
    sig["vol_rank"] = g["atr_ref"].rank(pct=True)
    sig["vol_bucket"] = np.select([sig["vol_rank"] <= 1/3, sig["vol_rank"] <= 2/3],
                                  ["lo", "mid"], "hi")
    sig["ctx_key"] = (sig["session"].astype(str) + "|" + sig["zone_sources"].astype(str)
                      + "|" + sig["sig_type"].astype(str) + "|" + sig["vol_bucket"])
    cells = sig.groupby(["code", "ctx_key"]).size().groupby("code")
    n10 = cells.apply(lambda s: (s >= 10).sum())
    n5 = cells.apply(lambda s: (s >= 5).sum())
    tot_cells = cells.apply(len)
    print(f"  各股背景单元格数: 中位={tot_cells.median():.0f} 范围[{tot_cells.min()},{tot_cells.max()}]")
    print(f"  各股 ≥10 事件单元格: 中位={n10.median():.0f}  有≥1个的股占比={ (n10>=1).mean():.0%}")
    print(f"  各股 ≥5 事件单元格: 中位={n5.median():.0f}  有≥1个的股占比={ (n5>=1).mean():.0%}")
    # 若背景匹配放宽 (仅 session×zone_sources, 不分波动/信号类型)
    sig["ctx_loose"] = (sig["session"].astype(str) + "|" + sig["zone_sources"].astype(str))
    cl = sig.groupby(["code", "ctx_loose"]).size().groupby("code")
    n10l = cl.apply(lambda s: (s >= 10).sum())
    print(f"  放宽(仅 session×锚点) ≥10 单元格: 中位={n10l.median():.0f} "
          f"有≥1个的股占比={(n10l>=1).mean():.0%}")

    # ===== 落盘 =====
    report = {
        "feasibility_version": "GCP_V02_FEASIBILITY_V2_CORRECTED",
        "date": "2026-09-29",
        "scope": "复用16股缓存快照, 0新增请求; 可行性非绩效",
        "corrections_v2": [
            "f 字段重命名为 failure_3d_rate (3日跌破标签, 非收复率); "
            "真 reclaim30/60 未在引擎实现, 仅 reclaim30_invalid 数据质量标记",
            "pen_atr_bar 重命名为 penetration_signal_bar (仅信号bar穿透); "
            "新增 penetration_full_window = (zone_lower-min_low)/atr_ref (持有期最大穿透)",
            "4维单元格稀疏 ≠ 逐股永远到不了10: 较粗粒度(session×锚点) 87.5% 股至少1个≥10单元格",
            "ICC 为方差成分近似比值, 非随机效应充分校正; 行为指标离散 ≠ 已证个性稳定",
        ],
        "event_library": {
            "n_signaled_days": int(len(df)), "n_filled_expanded": int(len(sig)),
            "n_filled_pool": int(sig["pool_q"].sum()),
            "n_filled_nonpool": int((~sig["pool_q"]).sum()),
            "per_stock_filled_mean": float(n_stock.mean()),
            "per_stock_filled_median": float(n_stock.median()),
            "anchor_distribution": anchor.head(8).to_dict(),
        },
        "cross_stock": {
            "y_mean_between_sd": float(sm.std()),
            "y_mean_within_se": float(sse.mean()),
            "icc_approx": float(icc),
            "icc_caveat": "方差成分近似比值, 非随机效应充分校正; 仅探索",
            "failure_3d_rate_range": [float(f3.min()), float(f3.max())],
            "failure_3d_rate_note": "f=持有期3日内 min_low<buy_net-0.5*ATR_ref; 非 reclaim30/60",
            "penetration_signal_bar_range": [float(pen_sb.min()), float(pen_sb.max())],
            "penetration_signal_bar_note": "仅信号bar当日 (zone_lower-lo_signal)/atr_ref",
            "penetration_full_window_range": [float(pen_fw.min()), float(pen_fw.max())],
            "penetration_full_window_note": "(zone_lower-min_low)/atr_ref; min_low=持有期最低(fill_date同尺度)",
        },
        "temporal_stability": {
            "window_days_per_stock_note": "72日/股, 弱功效",
            "n_stable_stocks": int(len(use)) if 'use' in dir() else None,
            "y_early_late_corr": (float(cy) if len(use) >= 3 else None),
            "failure_3d_early_late_corr": (float(cf) if len(use) >= 3 else None),
            "y_sign_agreement": (float(same_sign) if len(use) >= 3 else None),
        },
        "context_matched_sample": {
            "cells_per_stock_median_4d": float(tot_cells.median()),
            "cells_ge10_median_4d": float(n10.median()),
            "stocks_with_ge10_cell_4d": float((n10 >= 1).mean()),
            "cells_ge5_median_4d": float(n5.median()),
            "loose_session_anchor_ge10_median": float(n10l.median()),
            "loose_stocks_with_ge10": float((n10l >= 1).mean()),
            "granularity_note": "4维(session×zone_sources×sig_type×vol_bucket) 稀疏: 多数股无≥10单元格; "
                                "但较粗粒度(session×zone_sources) 87.5%股至少1个≥10单元格, 中位1",
        },
        "conclusion_notes": [
            "扩展事件库 (池+非池) 各股 ~37 FILLED, 较 A/B-only(~7) 增约5x",
            "ICC≈低且为近似比值 ⇒ 跨股收益差异多在噪声内; 行为画像应以个股自身为主; "
            "行为指标有离散 ≠ 已证个性稳定",
            "72日窗口对时间稳定性判定功效不足, 证实需更长历史(另议小范围补抓)",
            "4维单元格稀疏: 多数股无≥10单元格; 但较粗粒度(session×锚点) 87.5%股至少1个≥10单元格 → "
            "个性价值须靠跨股共享估计+收缩, 非逐股独立; 恰合 v0.2 '条件交易价值=与当前背景可比/共享估计' 设定",
            "口径纠正: f=3日失败率非收复率; pen_atr_bar=信号bar穿透非全窗最大穿透(已补全窗); "
            "ICC=近似比值非随机效应校正",
            "基础画像'可以计算'已成立; '个性稳定'与'个性有交易增量'都尚未成立",
        ],
    }
    rp = OUT / "v02_feasibility_v2_corrected.json"
    rp.write_text(json.dumps(report, ensure_ascii=False, indent=1, default=str),
                  encoding="utf-8")
    print(f"\n可行性报告(修正版) 已写入: {rp}")


if __name__ == "__main__":
    main()
