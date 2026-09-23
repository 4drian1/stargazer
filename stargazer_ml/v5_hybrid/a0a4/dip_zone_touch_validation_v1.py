"""
DIP_ZONE_TOUCH_VALIDATION_V1
================================
目的：验证 "INTRADAY_DIP_SETUP + 第一次真实 ZONE_TOUCH" 是否是一个稳定的事件级 alpha，
      而不是训练模型。只做事件验证，不训练。

固定事件：
    INTRADAY_DIP_SETUP  AND  first real ZONE_TOUCH
Execution：
    signal = first ZONE_TOUCH
    actual_fill = first legal minute > signal   (来自 timing_frontier_v1.parquet 的 T0_TOUCH 记录)

必须回答的 5 个问题 (见用户裁定)：
  Q1 按 Fold / Year 拆            -> 多数年份为正？还是个别年份拉均值？
  Q2 Event-Matched Random         -> Touch 是否提供增量？(same-day, same-dip_stage DIP 候选)
  Q3 拆 DIP_ONLY / BOTH           -> 边缘来自 DIP 本身还是带 Bottom 属性的 DIP？
  Q4 拆 dip_stage                 -> PRE_PULLBACK / PULLBACK_ACTIVE / NEAR_ZONE
  Q5 pre/post 20260701 作为 Data-Regime Risk
                                   输出 n / coverage / raw / alpha / MFE / MAE

口径说明：
  * event_alpha / event_raw = frontier T0_TOUCH 的 fill->H1 (实际成交入场)
  * open_alpha  / open_raw  = a14 的 timing_alpha_h1 / timing_return_h1 (D-open->H1，用于与 random 公平比较)
  * Matched Random 用 (plan_date, dip_stage) 同窗 DIP 候选的 timing_alpha_h1 分布
    (float_mv bucket 匹配因 a14 未存 float_mv 而近似为 same-day same-stage；记为已知限制)
STATUS 输出：写入 datasets/dip_zone_touch_validation_v1.json
裁决 (v2026-09-22): FAIL_TO_PROMOTE  (不是 PROVEN_NO_EDGE)
  - matched-random 增量 CI 含 0 => 不足以升级 Trigger, 但不证伪 Touch 时刻 timing 增量
  - 需 DIP_TOUCH_EVENT_TIME_MATCHED_CONTROL_V1 用同日同 stage 同 minute-of-day 同执行规则补证
  - post-20260701 为负向 divergence, 但未匹配 float_mv, 尚不能排除 composition
"""
from __future__ import annotations
import sys, time, json
from pathlib import Path
import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
import a0a4_config as C  # noqa

REGIME_SPLIT = 20260701  # FLOAT_MV_DATA_REGIME_CHANGE


def _is_nan(x):
    return x is None or (isinstance(x, float) and np.isnan(x))


def _stats(df):
    n = len(df)
    if n == 0:
        return dict(n=0, event_alpha=None, event_raw=None, open_alpha=None, open_raw=None,
                    mfe=None, mae=None, loss=None)
    ev_a = df["alpha"].mean() if "alpha" in df else np.nan
    ev_r = df["stock_ret"].mean() if "stock_ret" in df else np.nan
    op_a = df["timing_alpha_h1"].mean() if "timing_alpha_h1" in df else np.nan
    op_r = df["timing_return_h1"].mean() if "timing_return_h1" in df else np.nan
    mfe = df["mfe"].mean() if "mfe" in df else np.nan
    mae = df["mae"].mean() if "mae" in df else np.nan
    loss = df["loss"].mean() * 100 if "loss" in df else np.nan
    return dict(n=n,
                event_alpha=(None if _is_nan(ev_a) else round(float(ev_a), 4)),
                event_raw=(None if _is_nan(ev_r) else round(float(ev_r), 4)),
                open_alpha=(None if _is_nan(op_a) else round(float(op_a), 4)),
                open_raw=(None if _is_nan(op_r) else round(float(op_r), 4)),
                mfe=(None if _is_nan(mfe) else round(float(mfe), 4)),
                mae=(None if _is_nan(mae) else round(float(mae), 4)),
                loss=(None if _is_nan(loss) else round(float(loss), 2)))


def _fmt(x):
    return f"{x * 100:+.2f}" if (x is not None and not _is_nan(x)) else "  --  "


def main():
    t0 = time.time()
    a = pd.read_parquet(C.DATASETS / "a14_universe.parquet")
    f = pd.read_parquet(C.DATASETS / "timing_frontier_v1.parquet")

    # ---- Touch group (DIP + first real ZONE_TOUCH, filled) ----
    touch = f[(f.setup == "INTRADAY_DIP_SETUP") & (f.arm == "T0_TOUCH") &
              (f.eligible == True) & (f.filled == True)].copy()  # noqa

    a14dip = a[a.primary_setup == "INTRADAY_DIP_SETUP"][
        ["sample_id", "timing_alpha_h1", "timing_return_h1", "is_bottom",
         "dip_stage", "plan_date", "effective_trade_date"]].copy()
    # slim copy for merge (avoid plan_date/dip_stage conflict with frontier's own columns)
    a14dip_merge = a14dip[["sample_id", "timing_alpha_h1", "timing_return_h1",
                           "is_bottom", "effective_trade_date"]].copy()
    touch = touch.merge(a14dip_merge, on="sample_id", how="left")

    touched_ids = set(touch["sample_id"])
    a14dip["touched"] = a14dip["sample_id"].isin(touched_ids)

    print(f"[dip_validate] touch n={len(touch)}  dip_pool={len(a14dip)}  "
          f"({time.time() - t0:.0f}s)", flush=True)

    out = {}
    out["overall"] = _stats(touch)
    print("\n=== OVERALL (DIP + ZONE_TOUCH, event entry fill->H1) ===")
    print(f"  n={out['overall']['n']}  event_alpha={_fmt(out['overall']['event_alpha'])}  "
          f"event_raw={_fmt(out['overall']['event_raw'])}  "
          f"open_alpha={_fmt(out['overall']['open_alpha'])}  "
          f"MFE={_fmt(out['overall']['mfe'])}  MAE={_fmt(out['overall']['mae'])}  loss%={out['overall']['loss']}")

    # ============ Q2: Event-Matched Random ============
    print("\n=== Q2  EVENT-MATCHED RANDOM (same plan_date + same dip_stage DIP pool) ===")
    cells = {}
    for (pd_, st), g in a14dip.groupby(["plan_date", "dip_stage"]):
        cells[(pd_, st)] = g

    inc, rmean, rp95, beat = [], [], [], []
    touch_open = []
    for _, row in touch.iterrows():
        key = (row["plan_date"], row["dip_stage"])
        g = cells.get(key)
        if g is None or len(g) <= 1:
            continue
        nt = g[(~g["touched"]) & (g["sample_id"] != row["sample_id"])]
        use = nt if len(nt) >= 5 else g[g["sample_id"] != row["sample_id"]]
        ta = use["timing_alpha_h1"].dropna()
        if len(ta) == 0:
            continue
        oa = row["timing_alpha_h1"]
        if _is_nan(oa):
            continue
        rm = float(ta.mean())
        rp = float(ta.quantile(0.95))
        inc.append(oa - rm)
        rmean.append(rm)
        rp95.append(rp)
        beat.append(1 if oa > rp else 0)
        touch_open.append(oa)

    inc = np.array(inc); rmean = np.array(rmean); rp95 = np.array(rp95); beat = np.array(beat)
    touch_open = np.array(touch_open)
    matched = dict(
        n_matched=int(len(inc)),
        touch_open_alpha_mean=round(float(touch_open.mean()), 4),
        random_mean=round(float(rmean.mean()), 4),
        random_p95=round(float(rp95.mean()), 4),
        incremental_alpha_mean=round(float(inc.mean()), 4),
        incremental_alpha_se=(round(float(inc.std(ddof=1) / np.sqrt(len(inc))), 5) if len(inc) > 1 else None),
        frac_beat_p95=round(float(beat.mean()), 4),
        incremental_ci95=(
            round(float(inc.mean() - 1.96 * inc.std(ddof=1) / np.sqrt(len(inc))), 4),
            round(float(inc.mean() + 1.96 * inc.std(ddof=1) / np.sqrt(len(inc))), 4)),
    ) if len(inc) else dict(n_matched=0)
    out["matched_random"] = matched
    print(f"  n_matched={matched.get('n_matched')}")
    print(f"  Touch open_alpha mean = {_fmt(matched.get('touch_open_alpha_mean'))}")
    print(f"  Random (same-day same-stage DIP) mean = {_fmt(matched.get('random_mean'))}   P95 = {_fmt(matched.get('random_p95'))}")
    print(f"  Incremental alpha (Touch - Random) mean = {_fmt(matched.get('incremental_alpha_mean'))}  "
          f"SE={matched.get('incremental_alpha_se')}  CI95={matched.get('incremental_ci95')}")
    print(f"  Fraction Touch > Random P95 = {matched.get('frac_beat_p95'):.3f}")

    # ============ Q1: Year / Fold ============
    print("\n=== Q1a  BY YEAR ===")
    by_year = {}
    for y, g in touch.groupby(touch["plan_date"] // 10000):
        s = _stats(g)
        by_year[int(y)] = s
        print(f"  {y}: n={s['n']:5d}  event_alpha={_fmt(s['event_alpha'])}  "
              f"open_alpha={_fmt(s['open_alpha'])}  MAE={_fmt(s['mae'])}  loss%={s['loss']}")
    out["by_year"] = by_year

    print("\n=== Q1b  WALK-FORWARD FOLDS (chronological quintiles) ===")
    tv = touch.sort_values("plan_date").reset_index(drop=True)
    tv["fold"] = pd.qcut(tv["plan_date"].rank(method="first"), 5, labels=["F1", "F2", "F3", "F4", "F5"])
    by_fold = {}
    for fk, g in tv.groupby("fold"):
        s = _stats(g)
        by_fold[str(fk)] = s
        print(f"  {fk}: n={s['n']:5d}  range[{g['plan_date'].min()}~{g['plan_date'].max()}]  "
              f"event_alpha={_fmt(s['event_alpha'])}  open_alpha={_fmt(s['open_alpha'])}")
    out["by_fold"] = by_fold

    # ============ Q3: DIP_ONLY / BOTH ============
    print("\n=== Q3  DIP_ONLY vs BOTH (is_bottom) ===")
    by_db = {}
    for lbl, g in [("DIP_ONLY", touch[~touch["is_bottom"]]), ("BOTH", touch[touch["is_bottom"]])]:
        s = _stats(g)
        by_db[lbl] = s
        print(f"  {lbl} (n={s['n']}): event_alpha={_fmt(s['event_alpha'])}  "
              f"open_alpha={_fmt(s['open_alpha'])}  MAE={_fmt(s['mae'])}  loss%={s['loss']}")
    out["by_dip_only_both"] = by_db

    # ============ Q4: dip_stage ============
    print("\n=== Q4  BY dip_stage ===")
    by_stage = {}
    for st, g in touch.groupby("dip_stage"):
        s = _stats(g)
        by_stage[str(st)] = s
        print(f"  {st}: n={s['n']:5d}  event_alpha={_fmt(s['event_alpha'])}  "
              f"open_alpha={_fmt(s['open_alpha'])}  MAE={_fmt(s['mae'])}")
    out["by_dip_stage"] = by_stage

    # ============ Q5: pre/post 20260701 regime risk ============
    print("\n=== Q5  pre/post 20260701  (Data-Regime Risk) ===")
    dip_total = a[a.primary_setup == "INTRADAY_DIP_SETUP"]
    by_regime = {}
    for tag, mask in [("pre", touch["plan_date"] < REGIME_SPLIT), ("post", touch["plan_date"] >= REGIME_SPLIT)]:
        g = touch[mask]
        s = _stats(g)
        # coverage = touched / all DIP candidates in that regime window
        cov_mask = (dip_total["plan_date"] < REGIME_SPLIT) if tag == "pre" else (dip_total["plan_date"] >= REGIME_SPLIT)
        cov = 100.0 * len(g) / max(1, int(cov_mask.sum()))
        by_regime[tag] = dict(**s, touch_rate_pct=round(cov, 2),
                              dip_candidates_in_window=int(cov_mask.sum()))
        print(f"  {tag}: n_touch={s['n']:5d}  dip_candidates={by_regime[tag]['dip_candidates_in_window']:5d}  "
              f"touch_rate={cov:.1f}%  event_alpha={_fmt(s['event_alpha'])}  "
              f"open_alpha={_fmt(s['open_alpha'])}  MAE={_fmt(s['mae'])}  loss%={s['loss']}")
    out["by_regime"] = by_regime

    # ---- verdict hint ----
    # 裁决 (用户裁定 v2026-09-22):
    #   DIP_ZONE_TOUCH_VALIDATION_V1 = FAIL_TO_PROMOTE  (不是 PROVEN_NO_EDGE)
    #   理由: matched-random 增量 CI 含 0, 仅说明 "Touch 不优于同日同 stage DIP 候选池",
    #         尚不能证伪 Touch 时刻本身的 timing 增量 (需 event-time matched control V1 补证).
    #   post-20260701: 观察到明显负向 regime divergence, 但尚不能排除 composition
    #     (未匹配 float_mv; 2026-07-01 是 float_mv 数据制度变化点; touch rate 相近仅说明触发比例未变).
    out["verdict"] = "FAIL_TO_PROMOTE"
    out["verdict_note"] = ("matched-random 增量 CI 含 0 => 不足以升级 Trigger; "
                           "但不能据此证伪 Touch 时刻 timing 增量 (需 event-time matched control V1)。")
    print("\n=== VERDICT HINTS ===")
    print(f"  VERDICT = FAIL_TO_PROMOTE  (非 PROVEN_NO_EDGE)")
    print(f"  event_alpha(overall)      = {_fmt(out['overall']['event_alpha'])}")
    print(f"  incremental vs random     = {_fmt(matched.get('incremental_alpha_mean'))}  "
          f"(CI95 {matched.get('incremental_ci95')})  -> CI 含 0")
    yr = {y: v['event_alpha'] for y, v in by_year.items()}
    print(f"  per-year event_alpha       = {yr}   (2026 拉高均值, 2025≈0)")
    print(f"  BOTH n={by_db['BOTH']['n']} (underpowered) -> DIP_ONLY dominates edge test")
    print(f"  post-20260701 event_alpha = {_fmt(by_regime['post']['event_alpha'])} "
          f"(n={by_regime['post']['n']}, touch_rate={by_regime['post']['touch_rate_pct']}%)  "
          f"=> 观察到负向 divergence, 但未匹配 float_mv, 尚不能排除 composition")

    (C.DATASETS / "dip_zone_touch_validation_v1.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(f"\n[dip_validate] wrote datasets/dip_zone_touch_validation_v1.json  ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
