"""GCP v0.2 稳定性分析 v3 (REV2-F 第5步, 2026-09-29).

用户放行: 有限补抓已完成 (865/865, 0源缺), 现运行固定 60m 主检验的 v3 稳定性分析.
本轮仅稳定性检验, 非正式训练或策略晋级.

稳定性证据 (协议 §5.1): 早段画像预测晚段事件, 相比组统计是否降低 RMSE/Brier;
  bootstrap CI 不跨 0 的增量 = 稳定性证据.

主检验 (协议 §2.1, 预注册): W_60m
  - pen_full_window_60m (连续): RMSE
  - reclaim_60m (二值 RECLAIMED=1): Brier
辅助窗 (30m/5d/10d): 同样计算, 不单独判稳定性.

切分 (协议 §5):
  - 7 股补抓股: 早段=扩展日 (minute_ext), 晚段=原登记日 (run_sheet.minute_fetch_days)
  - 早段已闭环: 10d 窗结束 < 晚段起点 (cal 索引)
  - 晚段事件 60m 窗完整 (window_complete_60m=True)
  - 其他 9 股无早段, 不参与预测增量 (不进 stock_profile; 仍不计入组画像:
    组画像只用有早段的 7 股早段事件, 保持预测/对比同源)

行为有效事件入选 (协议 §1.1):
  - FILLED / NO_FILL_ZERO_QTY / NO_FILL_UNLABELED / EXIT_DEFERRED_FAILED / EXIT_UNAVAILABLE 入选
  - SOURCE_MISSING / INVALID_DAY / ADJ_UNKNOWN / HALTED_UP_APPROX / NO_SIGNAL 不入选
  - 行为有效性 ≠ 执行有效性; 收益标签才需成交

输出: gcp_round_b_small_sample_v1/v02_feasibility_v3.json
运行: python gcp_v02_stability_v3.py
"""
from __future__ import annotations

import bisect
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

OUT = HERE / "gcp_round_b_small_sample_v1"
SNAP = OUT / "minute"
SNAP_EXT = OUT / "minute_ext"
POOL_DIR = HERE / "round_a_v0_1_impl_fix"

# 行为有效状态 (协议 §1.1): 行情完整入选, 不依赖成交
BEHAVIOR_ELIGIBLE = {
    "FILLED", "NO_FILL_ZERO_QTY", "NO_FILL_UNLABELED",
    "EXIT_DEFERRED_FAILED", "EXIT_UNAVAILABLE",
}
W_MAIN = "60m"
W_AUX = ["30m", "5d", "10d"]
W_ALL = [W_MAIN] + W_AUX
B_BOOT = 1000
MIN_EVENTS = 10
RNG_SEED = 20260929


def prev_td(d: int, cal: list) -> int | None:
    i = bisect.bisect_left(cal, d)
    return cal[i - 1] if i > 0 else None


def is_behavior_eligible(rec: dict) -> bool:
    """行为有效事件: 含 signal_ts + 行情完整状态 + behavior 字段已计算 (pen_60m 非 None)."""
    if not rec.get("signal_ts"):
        return False
    if rec.get("status") not in BEHAVIOR_ELIGIBLE:
        return False
    return rec.get(f"pen_full_window_{W_MAIN}") is not None


def is_closed_10d(rec: dict, cal: list, ci: dict, late_start: int) -> bool:
    """早段事件已闭环: 10d 窗结束日 < 晚段起点 (cal 索引). 10d 是最长窗, 闭环即全窗闭环."""
    sig_day = int(str(rec["signal_ts"])[:8])
    if sig_day not in ci:
        return False
    end_ci_10d = ci[sig_day] + 10
    if end_ci_10d >= len(cal):
        return False  # 10d 窗超出 cal (未闭环)
    return cal[end_ci_10d] < late_start


def reclaim_to_binary(reclaim_val):
    """reclaim 状态转二值: RECLAIMED=1, NOT_RECLAIMED/ALREADY_ABOVE=0, AMBIGUOUS=None(排除)."""
    if reclaim_val == "RECLAIMED":
        return 1
    if reclaim_val in ("NOT_RECLAIMED", "ALREADY_ABOVE"):
        return 0
    return None


def stage_real_extended(grid, panel):
    """新回放函数: 拼接 minute+minute_ext, 回放所有有分钟数据的交易日.

    stage_real (gcp_round_b_acceptance.py L610) 只回放 st["minute_fetch_days"] (L685),
    不会处理扩展日 [20250102, lookback_start). 本函数对每股回放所有有 minute 数据的
    交易日 (含扩展日), setup 按"事件发生时池资格" (mem dict, prev_td(d)) 决定;
    非池日 setup=None, 不重标 A/B.

    返回 (all_events, cal, ci). 每事件含 phase (early/late/skip) + lookback_start + late_start.
    """
    sheet = json.loads((OUT / "run_sheet.json").read_text(encoding="utf-8"))
    cal = sorted(int(d) for d in panel["date"].unique())
    didx = {d: i for i, d in enumerate(cal)}
    ci = didx

    # 池资格 mem dict (按事件发生时 prev_td(d) 池资格)
    pool = pd.concat([
        pd.read_parquet(POOL_DIR / "pool_a_sdp_members.parquet")[["date", "code", "setup_type"]],
        pd.read_parquet(POOL_DIR / "pool_b_lvb_members.parquet")[["date", "code", "setup_type"]],
    ])
    pool["code"] = pool["code"].astype(str).str.zfill(6)
    mem = pool.groupby(["date", "code"])["setup_type"].apply(set).to_dict()

    # 复权链缓存 (已存在, 不再 HTTP)
    adj_f = OUT / "adj_cache.json"
    adj_cache = json.loads(adj_f.read_text(encoding="utf-8")) if adj_f.exists() else {}

    # 预加载 minute_ext 日期集合 (用于早段判定)
    ext_days_map = {}
    if SNAP_EXT.exists():
        for pf in SNAP_EXT.glob("*.parquet"):
            code = pf.stem
            df = pd.read_parquet(pf, columns=["date"])
            df["day"] = df["date"].astype(str).str[:8].astype(int)
            ext_days_map[code] = set(df["day"].unique().tolist())

    all_events = []
    for st in sheet["stocks"]:
        code = st["code"]
        lookback_start = st["lookback_start"]
        fetch_days_set = set(st["minute_fetch_days"])
        ext_days_set = ext_days_map.get(code, set())
        late_start = min(fetch_days_set) if fetch_days_set else lookback_start

        # daily (从 panel)
        daily = panel[panel["code"].astype(str).str.zfill(6) == code].sort_values("date")
        daily = daily[["date", "open", "high", "low", "close"]].copy()
        daily["date"] = daily["date"].astype(int)

        # 复权链
        ac = adj_cache.get(code, {"ok": True, "chain": {}})
        chain = {int(k): float(v) for k, v in ac.get("chain", {}).items()}
        adj_ok = bool(ac.get("ok", True))

        # 拼接 minute + minute_ext (旧输入保留不覆盖)
        parts = []
        snap_old = SNAP / f"{code}.parquet"
        if snap_old.exists():
            parts.append(pd.read_parquet(snap_old))
        snap_ext = SNAP_EXT / f"{code}.parquet"
        if snap_ext.exists():
            parts.append(pd.read_parquet(snap_ext))
        if not parts:
            continue
        md = pd.concat(parts, ignore_index=True).drop_duplicates("date")
        md["day"] = md["date"].astype(int).astype(str).str[:8].astype(int)
        by_day = {int(d): g for d, g in md.groupby("day")}

        ctx = {"code": code, "daily": daily, "date_idx": didx, "cal": cal,
               "chain": chain, "adj_ok": adj_ok}

        for d in sorted(by_day.keys()):
            # phase 判定: early=扩展日, late=原登记日, skip=未分类 (理论不存在)
            if d in ext_days_set:
                phase = "early"
            elif d in fetch_days_set:
                phase = "late"
            else:
                phase = "skip"
            if phase == "skip":
                continue
            # setup 按事件发生时 prev_td(d) 池资格
            pd_prev = prev_td(d, cal)
            setups = mem.get((pd_prev, code), set()) or {None}
            for s in sorted(setups, key=lambda x: (x is None, x)):
                rec = E.run_day({**ctx, "setup": s, "minute": by_day.get(d)}, d, grid)
                rec["phase"] = phase
                rec["lookback_start"] = lookback_start
                rec["late_start"] = late_start
                all_events.append(rec)
    return all_events, cal, ci


def _profile(events, w):
    """计算一组事件的 pen_W 均值 + reclaim_W 概率 (二值化)."""
    pens = [r.get(f"pen_full_window_{w}") for r in events
            if r.get(f"pen_full_window_{w}") is not None]
    reclaims = [reclaim_to_binary(r.get(f"reclaim_{w}")) for r in events]
    reclaims = [x for x in reclaims if x is not None]
    return {
        "pen_mean": float(np.mean(pens)) if pens else None,
        "reclaim_p": float(np.mean(reclaims)) if reclaims else None,
        "n_pen": len(pens), "n_reclaim": len(reclaims),
    }


def compute_prediction_deltas(early, late, cal, ci):
    """主检验 W_60m + 辅助窗: 早段画像预测晚段, 对比组统计.

    对每个窗口 W:
      - 个股预测值 = 早段该股 pen_W 均值 / reclaim_W 概率 (已闭环事件)
      - 组预测值 = 早段跨股 pen_W 均值 / reclaim_W 概率 (已闭环事件, 同源)
      - 晚段事件实际值: pen_W (float), reclaim_W (二值)
      - 误差: (预测 - 实际)^2
      - RMSE (pen) = sqrt(mean(误差)); Brier (reclaim) = mean(误差)
      - 增量 = 组误差 - 个股误差 (正=个股更准)
      - bootstrap CI (B=1000, 晚段事件 resample; CI 不跨0 = 显著)
    """
    # 早段已闭环事件 + 行为有效
    early_closed = [r for r in early
                    if is_behavior_eligible(r) and is_closed_10d(r, cal, ci, r["late_start"])]
    # 晚段事件 + 行为有效 + 60m 窗完整
    late_valid = [r for r in late
                  if is_behavior_eligible(r) and r.get(f"window_complete_{W_MAIN}", False)]

    # 早段按股分组 (只含有早段的股)
    early_by_code = {}
    for r in early_closed:
        early_by_code.setdefault(r["code"], []).append(r)
    # 组画像: 只用有早段画像的 7 股的早段事件 (保持预测/对比同源)
    group_profile = {w: _profile(early_closed, w) for w in W_ALL}

    # 个股画像
    stock_profile = {code: {w: _profile(evs, w) for w in W_ALL}
                     for code, evs in early_by_code.items()}

    results = {}
    for w in W_ALL:
        # 对每股晚段事件, 用该股早段画像预测; 无早段画像的股跳过 (不参与预测增量)
        sq_err_stock_pen, sq_err_group_pen = [], []
        sq_err_stock_reclaim, sq_err_group_reclaim = [], []
        per_stock_n = {}
        for r in late_valid:
            code = r["code"]
            sp = stock_profile.get(code)
            if sp is None:
                continue  # 该股无早段画像, 跳过
            actual_pen = r.get(f"pen_full_window_{w}")
            actual_reclaim = reclaim_to_binary(r.get(f"reclaim_{w}"))
            stock_pen_pred = sp[w]["pen_mean"]
            stock_reclaim_pred = sp[w]["reclaim_p"]
            group_pen_pred = group_profile[w]["pen_mean"]
            group_reclaim_pred = group_profile[w]["reclaim_p"]

            if actual_pen is not None and stock_pen_pred is not None and group_pen_pred is not None:
                sq_err_stock_pen.append((stock_pen_pred - actual_pen) ** 2)
                sq_err_group_pen.append((group_pen_pred - actual_pen) ** 2)
            if (actual_reclaim is not None and stock_reclaim_pred is not None
                    and group_reclaim_pred is not None):
                sq_err_stock_reclaim.append((stock_reclaim_pred - actual_reclaim) ** 2)
                sq_err_group_reclaim.append((group_reclaim_pred - actual_reclaim) ** 2)
            per_stock_n.setdefault(code, 0)
            per_stock_n[code] += 1

        rmse_stock = float(np.sqrt(np.mean(sq_err_stock_pen))) if sq_err_stock_pen else None
        rmse_group = float(np.sqrt(np.mean(sq_err_group_pen))) if sq_err_group_pen else None
        brier_stock = float(np.mean(sq_err_stock_reclaim)) if sq_err_stock_reclaim else None
        brier_group = float(np.mean(sq_err_group_reclaim)) if sq_err_group_reclaim else None

        # bootstrap CI (晚段事件 resample; stock/group 长度对齐)
        rng = np.random.default_rng(RNG_SEED)
        n_late = len(sq_err_stock_pen)
        deltas_pen, deltas_reclaim = [], []
        if n_late >= MIN_EVENTS:
            arr_sp = np.array(sq_err_stock_pen)
            arr_gp = np.array(sq_err_group_pen)
            for _ in range(B_BOOT):
                idx = rng.integers(0, n_late, n_late)
                deltas_pen.append(np.sqrt(arr_gp[idx].mean()) - np.sqrt(arr_sp[idx].mean()))
        n_late_r = len(sq_err_stock_reclaim)
        if n_late_r >= MIN_EVENTS:
            arr_sr = np.array(sq_err_stock_reclaim)
            arr_gr = np.array(sq_err_group_reclaim)
            for _ in range(B_BOOT):
                idx = rng.integers(0, n_late_r, n_late_r)
                deltas_reclaim.append(arr_gr[idx].mean() - arr_sr[idx].mean())

        ci_pen = (float(np.percentile(deltas_pen, 2.5)),
                  float(np.percentile(deltas_pen, 97.5))) if deltas_pen else None
        ci_reclaim = (float(np.percentile(deltas_reclaim, 2.5)),
                      float(np.percentile(deltas_reclaim, 97.5))) if deltas_reclaim else None

        results[w] = {
            "n_early_closed": len(early_closed),
            "n_late_valid_for_pen": n_late,
            "n_late_valid_for_reclaim": n_late_r,
            "n_stocks_with_early": len(stock_profile),
            "n_stocks_late_used": len(per_stock_n),
            "group_profile": group_profile[w],
            "rmse_stock": rmse_stock, "rmse_group": rmse_group,
            "rmse_delta_group_minus_stock": (rmse_group - rmse_stock)
                if (rmse_stock is not None and rmse_group is not None) else None,
            "rmse_delta_ci95": ci_pen,
            "rmse_delta_significant": (ci_pen[0] > 0) if ci_pen else False,
            "brier_stock": brier_stock, "brier_group": brier_group,
            "brier_delta_group_minus_stock": (brier_group - brier_stock)
                if (brier_stock is not None and brier_group is not None) else None,
            "brier_delta_ci95": ci_reclaim,
            "brier_delta_significant": (ci_reclaim[0] > 0) if ci_reclaim else False,
            "per_stock_late_n": per_stock_n,
        }
    return results, early_closed, late_valid, stock_profile


def compute_temporal_consistency(early, late, cal, ci):
    """早晚一致性 (辅助证据, §7.2.2): pen/reclaim 均值方向一致 + r>=0.5.

    要求每股早晚各>=3 事件 (低于 §5.3 的 10 门槛, 但一致性只看方向, 放宽到 3).
    """
    early_by_code, late_by_code = {}, {}
    for r in early:
        if is_behavior_eligible(r) and is_closed_10d(r, cal, ci, r["late_start"]):
            early_by_code.setdefault(r["code"], []).append(r)
    for r in late:
        if is_behavior_eligible(r) and r.get(f"window_complete_{W_MAIN}", False):
            late_by_code.setdefault(r["code"], []).append(r)

    consistency = {}
    for w in W_ALL:
        early_pens, late_pens = [], []
        early_reclaims, late_reclaims = [], []
        for code in set(early_by_code) & set(late_by_code):
            e, l = early_by_code[code], late_by_code[code]
            if len(e) < 3 or len(l) < 3:
                continue
            ep = [r.get(f"pen_full_window_{w}") for r in e
                  if r.get(f"pen_full_window_{w}") is not None]
            lp = [r.get(f"pen_full_window_{w}") for r in l
                  if r.get(f"pen_full_window_{w}") is not None]
            er = [reclaim_to_binary(r.get(f"reclaim_{w}")) for r in e]
            lr = [reclaim_to_binary(r.get(f"reclaim_{w}")) for r in l]
            er = [x for x in er if x is not None]
            lr = [x for x in lr if x is not None]
            if ep and lp:
                early_pens.append(np.mean(ep))
                late_pens.append(np.mean(lp))
            if er and lr:
                early_reclaims.append(np.mean(er))
                late_reclaims.append(np.mean(lr))
        pen_corr = (float(np.corrcoef(early_pens, late_pens)[0, 1])
                    if len(early_pens) >= 3 else None)
        reclaim_corr = (float(np.corrcoef(early_reclaims, late_reclaims)[0, 1])
                        if len(early_reclaims) >= 3 else None)
        consistency[w] = {
            "n_stocks": len(early_pens),
            "pen_early_late_corr": pen_corr,
            "reclaim_early_late_corr": reclaim_corr,
        }
    return consistency


def main():
    print("=" * 64)
    print("[GCP v0.2 稳定性分析 v3 — REV2-F 第5步]")
    print(f"主检验: W_{W_MAIN} (预注册); 辅助窗: {W_AUX} (不单独判)")
    print(f"稳定性证据 = 早晚预测误差增量 (bootstrap B={B_BOOT}, CI 不跨0)")
    print("=" * 64)

    grid = json.loads((OUT / "minute_grid.json").read_text(encoding="utf-8"))["grid"]
    panel = get_daily_parquet()

    all_events, cal, ci = stage_real_extended(grid, panel)
    early = [r for r in all_events if r.get("phase") == "early"]
    late = [r for r in all_events if r.get("phase") == "late"]
    skip = [r for r in all_events if r.get("phase") == "skip"]

    print(f"\n[回放] 总事件={len(all_events)} early={len(early)} late={len(late)} skip={len(skip)}")
    by_code_phase = {}
    for r in all_events:
        d = by_code_phase.setdefault(r["code"], {"early": 0, "late": 0})
        d[r["phase"]] = d.get(r["phase"], 0) + 1
    print("  各股 early/late 事件数 (含未触发信号/无效日):")
    for code in sorted(by_code_phase):
        e = by_code_phase[code].get("early", 0)
        l = by_code_phase[code].get("late", 0)
        flag = " ← 有早段" if e > 0 else ""
        print(f"    {code}: early={e} late={l}{flag}")

    early_eligible = [r for r in early if is_behavior_eligible(r)]
    late_eligible = [r for r in late if is_behavior_eligible(r)]
    print(f"\n[行为有效事件 (行情完整入选, 不依赖成交)]")
    print(f"  early_eligible={len(early_eligible)} late_eligible={len(late_eligible)}")

    print(f"\n[预测误差增量 (主检验 W_60m, bootstrap B={B_BOOT})]")
    deltas, early_closed, late_valid, stock_profile = compute_prediction_deltas(
        early, late, cal, ci)
    for w in W_ALL:
        d = deltas[w]
        role = "主检验" if w == W_MAIN else "辅助  "
        print(f"\n  -- 窗口 {w} ({role}) --")
        print(f"    n_early_closed={d['n_early_closed']} n_late_valid_for_pen={d['n_late_valid_for_pen']}"
              f" n_late_valid_for_reclaim={d['n_late_valid_for_reclaim']}"
              f" n_stocks_with_early={d['n_stocks_with_early']}")
        gp = d["group_profile"]
        pen_str = f"{gp['pen_mean']:.4f}" if gp['pen_mean'] is not None else "N/A"
        rec_str = f"{gp['reclaim_p']:.4f}" if gp['reclaim_p'] is not None else "N/A"
        print(f"    组画像(早段已闭环): pen_mean={pen_str} reclaim_p={rec_str}"
              f" (n_pen={gp['n_pen']} n_reclaim={gp['n_reclaim']})")
        if d["rmse_stock"] is not None:
            delta = d['rmse_delta_group_minus_stock']
            print(f"    pen RMSE: stock={d['rmse_stock']:.4f} group={d['rmse_group']:.4f}"
                  f" delta(group-stock)={delta:.4f}")
            if d["rmse_delta_ci95"]:
                lo, hi = d["rmse_delta_ci95"]
                sig = "显著(个股更准)" if lo > 0 else ("显著(组更准)" if hi < 0 else "不显著(CI跨0)")
                print(f"      delta CI95=[{lo:.4f}, {hi:.4f}] {sig}")
        else:
            print("    pen RMSE: N/A (样本不足)")
        if d["brier_stock"] is not None:
            delta = d['brier_delta_group_minus_stock']
            print(f"    reclaim Brier: stock={d['brier_stock']:.4f} group={d['brier_group']:.4f}"
                  f" delta={delta:.4f}")
            if d["brier_delta_ci95"]:
                lo, hi = d["brier_delta_ci95"]
                sig = "显著(个股更准)" if lo > 0 else ("显著(组更准)" if hi < 0 else "不显著(CI跨0)")
                print(f"      delta CI95=[{lo:.4f}, {hi:.4f}] {sig}")
        else:
            print("    reclaim Brier: N/A (样本不足)")

    print(f"\n[早晚一致性 (辅助证据, §7.2.2, pen/reclaim corr>=0.5)]")
    consistency = compute_temporal_consistency(early, late, cal, ci)
    for w in W_ALL:
        c = consistency[w]
        role = "主检验" if w == W_MAIN else "辅助  "
        print(f"  {w} ({role}): n_stocks={c['n_stocks']}"
              f" pen_corr={c['pen_early_late_corr']}"
              f" reclaim_corr={c['reclaim_early_late_corr']}")

    # 稳定性判定 (主检验 W_60m, 协议 §7.2)
    md = deltas[W_MAIN]
    mc = consistency[W_MAIN]
    print(f"\n[稳定性判定 (主检验 W_60m, 协议 §7.2 三项全达标)]")
    crit1 = bool(md.get("rmse_delta_significant") or md.get("brier_delta_significant"))
    crit2 = bool(((mc.get("pen_early_late_corr") or 0) >= 0.5) or
                  ((mc.get("reclaim_early_late_corr") or 0) >= 0.5))
    early_n_by_code = {}
    late_n_by_code = {}
    for r in early_eligible:
        if is_closed_10d(r, cal, ci, r["late_start"]):
            early_n_by_code[r["code"]] = early_n_by_code.get(r["code"], 0) + 1
    for r in late_eligible:
        if r.get(f"window_complete_{W_MAIN}", False):
            late_n_by_code[r["code"]] = late_n_by_code.get(r["code"], 0) + 1
    stocks_early_ge10 = [c for c in early_n_by_code if early_n_by_code[c] >= MIN_EVENTS]
    stocks_both_ge10 = [c for c in stocks_early_ge10 if late_n_by_code.get(c, 0) >= MIN_EVENTS]
    crit3 = len(stocks_both_ge10) >= 3
    print(f"  1. 预测增量 (核心): rmse_sig={md.get('rmse_delta_significant')}"
          f" brier_sig={md.get('brier_delta_significant')} -> {'达标' if crit1 else '未达标'}")
    print(f"  2. 早晚一致 (辅助): pen_corr={mc.get('pen_early_late_corr')}"
          f" reclaim_corr={mc.get('reclaim_early_late_corr')} -> {'达标' if crit2 else '未达标'}")
    print(f"  3. 样本充分: 早晚各>=10 的股={stocks_both_ge10} (需>=3) -> {'达标' if crit3 else '未达标'}")
    stability = crit1 and crit2 and crit3
    print(f"  => 个性稳定{'成立 (进 v0.2 主合同起草)' if stability else '尚未成立 (画像仅保留解释用途)'}")

    # 落盘
    report = {
        "report_version": "GCP_V02_STABILITY_V3",
        "date": "2026-09-29",
        "scope": "REV2-F 第5步: 固定 60m 主检验稳定性分析; 非正式训练或策略晋级",
        "protocol": "gcp_v02_stability_protocol.md REV2 (预注册 W_60m 为主检验)",
        "data_source": {
            "minute_old": "gcp_round_b_small_sample_v1/minute/ (16 股, 旧输入保留)",
            "minute_ext": "gcp_round_b_small_sample_v1/minute_ext/ (7 股补抓, 865/865 成功 0 源缺)",
            "backfill_report": "v02_backfill_execute_report.json",
        },
        "main_window": W_MAIN,
        "aux_windows": W_AUX,
        "bootstrap_B": B_BOOT,
        "min_events_per_phase": MIN_EVENTS,
        "counts": {
            "total_events": len(all_events),
            "early": len(early), "late": len(late), "skip": len(skip),
            "early_eligible": len(early_eligible),
            "late_eligible": len(late_eligible),
            "early_closed": len(early_closed),
            "late_valid_60m_complete": len(late_valid),
            "per_stock_phase": by_code_phase,
            "early_eligible_closed_n_by_code": early_n_by_code,
            "late_eligible_60m_complete_n_by_code": late_n_by_code,
            "stocks_early_ge10": stocks_early_ge10,
            "stocks_both_ge10": stocks_both_ge10,
        },
        "stock_profile_early": {c: {w: stock_profile[c][w] for w in W_ALL}
                                 for c in stock_profile},
        "prediction_deltas": deltas,
        "temporal_consistency": consistency,
        "stability_verdict": {
            "main_window": W_MAIN,
            "criteria_1_prediction_delta": crit1,
            "criteria_2_temporal_consistency": crit2,
            "criteria_3_sample_sufficient": crit3,
            "established": stability,
            "note": "个性稳定成立 = 三项全达标 (协议 §7.2); "
                    "未成立则画像仅保留解释用途, 不进买入决策",
        },
        "conclusion_notes": [
            "本轮为稳定性检验, 非正式训练或策略晋级",
            "早段=扩展日 (minute_ext), 晚段=原登记日 (run_sheet.minute_fetch_days)",
            "早段已闭环: 10d 窗结束 < 晚段起点 (cal 索引); 晚段事件 60m 窗完整",
            "行为有效事件入选: FILLED/NO_FILL_ZERO_QTY/NO_FILL_UNLABELED/EXIT_* (行情完整); "
            "行为有效性 ≠ 执行有效性",
            "组画像只用有早段画像的 7 股早段事件 (保持预测/对比同源)",
            "辅助窗 30m/5d/10d 仅报不判稳定性 (预注册 W_60m 为主检验)",
            "有预测增量才制定决策合同; 再验交易收益; 仍不重训/不接线上",
        ],
        "caveats": [
            "措辞约束: 上市日期代理/零量占位/冻结研究请求清单 — "
            "不用'新上市/已确认停牌/权威化'证明措辞",
            "补抓不用于证明交易规则有效",
            "2024 不补抓结论保留, 限定为'当前 canonical 源和研究范围不可取'",
        ],
    }
    rp = OUT / "v02_feasibility_v3.json"
    rp.write_text(json.dumps(report, ensure_ascii=False, indent=1, default=str),
                  encoding="utf-8")
    print(f"\n稳定性报告 v3 已写入: {rp}")


if __name__ == "__main__":
    main()
