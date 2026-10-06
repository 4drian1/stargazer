# -*- coding: utf-8 -*-
"""T5 双头标签引擎 v3（2026-09-09 冻结：风险头保持T+2窗口，机会头改3/5日经济账本）。

风险头主标签（不变）：r_sbr_t2 = stop_before_repair_through_earliest_sell_day——
entry(T+1盘中VWAP)→T+2收盘窗口内先触可执行止损(ks×ATR，跌停锁死bar不计执行)
再修复(+0.3ATR)；同bar双触悲观记STOP；与机会头共用同一价格路径。
辅助：r_stop_touched_t2 / r_stop_by_t2 / r_stop_locked_only_t2、
r_same_day_extreme_adverse_move（决策日旧口径0.6%，仅诊断）、r_mae_atr、
r_new_low_30m/60m（仅未截断样本）、r_vol_dump。

机会头主标签（本轮改口径）：不再用"先修复+0.3ATR"（o_rbs_t2 降级为辅助，
已证实与5日净收益弱对齐——修复快≈已反弹）。改为独立3日/5日交易账本
（规则逐条复制 minute_label_engine.scan_exit：T+2起扫、止损ks×ATR、
止盈1.0/1.5/2.0×ATR、跌停锁死日递延成交、缺口成交、超时按最后收盘、
扣COST；同分钟双触的TARGET/STOP归属与引擎一致——注意引擎同bar双触记
TARGET，与"悲观STOP"文档口径不同，ties列计数供审计）：
  o_net_ret_3d/5d        主场景(kt=2.0)扣费净收益
  o_t1_bt_3d/5d          target_before_stop（kt=1.0 先止盈）
  o_t15_bt_3d/5d         kt=1.5 先止盈
  o_stop_3d/5d           主场景先止损（stop_before_target）
  o_mfe/mae_3d/5d_atr    窗口最大涨幅/最大回撤（ATR单位）
  o_ttt_3d/5d            先止盈(kt=1.0)自T+2开盘起的交易分钟（NaN=未达）
  o_close_ret_3d/5d      买入持有至窗口末收盘的扣费收益（benchmark）
  o_timeout_ret_3d/5d    TIMEOUT 情形的扣费收益
  o_repair_but_loss_3d/5d 盘中修复成功(o_rbs_t2=1)但3/5日最终亏损（审计列）
  o_ties_3d/5d           同bar双触计数（口径审计）
机会头辅助（解释层，不作主标签）：o_rbs_t2、o_repair_by_t2、o_reclaim03/05、
o_vwap_reclaim、o_mfe_atr（决策日内）、o_target_before_stop_5d（=o_t1_bt_5d别名）。

30/60分钟标签的截断约束：cens30/cens60=True 时相应标签记 NaN，绝不记失败。
幂等；--samples/--out 跑 2025 与 2026。5d账本由 validate_ledger.py 对
minute_samples 的 outcome_t*/net_ret_t*/mfe/mae 交叉验证。
"""
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from minute_label_engine import COST, LIMIT_OF, load_code  # noqa: E402

OUT_DIR = "D:/stargazer/stargazer_ml/pullback_train"
SAMPLES = os.path.join(OUT_DIR, "minute_samples_2025.parquet")
OUT = os.path.join(OUT_DIR, "dual_head_labels_2025.parquet")
DPS = (945, 1000, 1030, 1330, 1400, 1430, 1450)
# 与引擎 KTARGS 完全一致（含3.0）：提前 break 时点必须与引擎相同，
# 否则全窗口 mfe/mae 与引擎样本表不一致（§18.30 修复，validate_ledger 校验）
KT_LEDG = (1.0, 1.5, 2.0, 3.0)


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def compute(cm, code, day_i, dp, entry, stop_px, rec03, rec05, atr_pct):
    f = {}
    idx = np.where(cm["day_idx"] == day_i)[0]
    t = cm["t"][idx].astype(int)
    h = cm["h"][idx].astype(float)
    l = cm["l"][idx].astype(float)
    c = cm["c"][idx].astype(float)
    v = cm["v"][idx].astype(float)
    a = cm["a"][idx].astype(float)
    i_dp = int(np.searchsorted(t, dp * 100, side="right")) - 1
    i_en = int(np.searchsorted(t, dp * 100, side="right"))  # entry后第一根
    if i_en >= len(t) or i_dp < 0:
        return None
    # entry 用该 bar 的 vwap，与标签引擎同口径
    if v[i_en] > 0:
        entry = float(a[i_en] / v[i_en])
    f["_entry"] = entry
    j0 = i_en + 1  # entry 之后的 bar（entry bar 本身不计路径）
    bars_left = len(t) - j0
    cens30 = bool(bars_left < 30)
    cens60 = bool(bars_left < 60)
    if not np.isfinite(entry) or entry <= 0 or not np.isfinite(atr_pct) \
            or atr_pct <= 0:
        return None
    f["cens30"] = cens30
    f["cens60"] = cens60
    if bars_left <= 0:
        return None
    lw = l[j0:]
    hw = h[j0:]
    cw = c[j0:]
    n = len(lw)
    low_dp = l[:i_dp + 1].min()          # DP 窗口内最低（决策时已知）
    # session VWAP（至当日收盘）
    cv = np.cumsum(v)
    with np.errstate(all="ignore"):
        vwap_all = np.cumsum(a) / np.where(cv > 0, cv, np.nan)
    f["r_day_close_below_vwap"] = float(c[-1] < vwap_all[-1]) \
        if np.isfinite(vwap_all[-1]) else np.nan
    # ---- 风险主标签：entry→T+2收盘（最早可卖日，含隔夜跳空；口径钉死见§18.24）
    lim = LIMIT_OF(code)
    path_l = [lw]
    path_c = [cw]
    if day_i + 1 < len(cm["days"]):
        idx2 = np.where(cm["day_idx"] == day_i + 1)[0]
        if len(idx2):
            path_l.append(cm["l"][idx2].astype(float))
            path_c.append(cm["c"][idx2].astype(float))
    cl = np.concatenate(path_l)
    cc = np.concatenate(path_c)
    pc = np.concatenate(([entry], cc[:-1]))
    with np.errstate(all="ignore"):
        ret_bar = cc / pc - 1
    locked = (cc == cl) & (ret_bar <= -lim + 0.005)
    touched = cl <= stop_px
    exec_stop = touched & ~locked          # 可成交的止损触及
    hit_rec = cc >= rec03
    any_stop = bool(exec_stop.any())
    any_rec = bool(hit_rec.any())
    any_touch = bool(touched.any())
    stop_i = int(np.argmax(exec_stop)) if any_stop else len(cl) + 1
    rec_i = int(np.argmax(hit_rec)) if any_rec else len(cc) + 1
    f["r_sbr_t2"] = float(any_stop and stop_i <= rec_i)   # 同bar悲观STOP
    f["o_rbs_t2"] = float(any_rec and rec_i < stop_i)     # 降级为辅助
    f["r_stop_by_t2"] = float(any_stop)
    f["r_stop_touched_t2"] = float(any_touch)
    f["r_stop_locked_only_t2"] = float(any_touch and not any_stop)
    f["o_repair_by_t2"] = float(any_rec)
    # ---- 同日极端标签（诊断）：决策日内先触止损再修复（正例率~0.6%） ----
    sd_touch = lw <= stop_px
    sd_rec = cw >= rec03
    sd_stop_i = int(np.argmax(sd_touch)) if sd_touch.any() else n + 1
    sd_rec_i = int(np.argmax(sd_rec)) if sd_rec.any() else n + 1
    f["r_same_day_extreme_adverse_move"] = float(sd_stop_i < sd_rec_i)
    # ---- 路径极值（决策日内） ----
    f["r_mae_atr"] = float((entry - lw.min()) / (entry * atr_pct))
    f["o_mfe_atr"] = float((hw.max() - entry) / (entry * atr_pct))
    f["o_vwap_reclaim"] = float((cw > vwap_all[j0:]).any()) \
        if np.isfinite(vwap_all[j0:]).any() else np.nan
    # ---- 30/60 分钟辅助（截断记NaN） ----
    f["r_new_low_30m"] = f["r_new_low_60m"] = np.nan
    f["o_reclaim03_30m"] = f["o_reclaim03_60m"] = np.nan
    f["o_reclaim05_30m"] = f["o_reclaim05_60m"] = np.nan
    for w, cens in ((30, cens30), (60, cens60)):
        if cens:
            continue
        seg_l = lw[:w]
        seg_c = cw[:w]
        f[f"r_new_low_{w}m"] = float(seg_l.min() < low_dp)
        up03 = (seg_c.max() - entry) / (entry * atr_pct)
        f[f"o_reclaim03_{w}m"] = float(up03 >= 0.3)
        f[f"o_reclaim05_{w}m"] = float(up03 >= 0.5)
    # ---- 放量下杀（向量化）：量>2×当日截至该bar中位 且 跌幅>0.5ATR ----
    med = pd.Series(v).expanding(min_periods=5).median().values
    prev_c = np.concatenate(([c[0]], c[:-1]))
    with np.errstate(all="ignore"):
        ret1b = c / prev_c - 1
    cond = (v > 2 * med) & (ret1b < -0.5 * atr_pct)
    cond[:5] = False
    f["r_vol_dump"] = float(cond[j0 + 1:].any()) if n > 1 else 0.0
    return f


def scan_ledger(cm, dp_day_i, n_days, entry, atr_pct, ks, pcb, lim):
    """T+2起扫 n_days 个可卖交易日（dp_day_i=T+1）。规则逐条复制
    minute_label_engine.scan_exit（含跌停锁死递延、缺口成交、超时收盘、COST）。
    返回 {res{kt:...}, mfe, mae, avail, need, ties, timeout_price}。"""
    nd = len(cm["days"])
    fut = [j for j in range(dp_day_i + 1, min(dp_day_i + 1 + n_days, nd))]
    if not len(fut):
        return None
    S = entry * (1 - ks * atr_pct)
    Tk = {kt: entry * (1 + kt * atr_pct) for kt in KT_LEDG}
    res = {kt: {"outcome": "TIMEOUT", "net_ret": np.nan, "touch": np.nan,
                "exit_day": len(fut)} for kt in KT_LEDG}
    mfe, mae = -np.inf, np.inf
    done = {kt: False for kt in KT_LEDG}
    pending_stop = False
    timeout_price = np.nan
    ties = 0
    for di, j in enumerate(fut, 1):
        m = cm["day_idx"] == j
        o = cm["o"][m].astype(float)
        h = cm["h"][m].astype(float)
        l = cm["l"][m].astype(float)
        c = cm["c"][m].astype(float)
        t = cm["t"][m]
        if not len(o):
            continue
        # mfe/mae 在循环内累计且受提前break截断——与引擎逐条一致（§18.30）
        mfe = max(mfe, float(h.max() / entry - 1))
        mae = min(mae, float(l.min() / entry - 1))
        timeout_price = float(c[-1])
        pc = pcb[j]
        locked = (np.isfinite(pc)
                  and (cm["day_close"][j] / pc - 1) <= -lim + 0.005
                  and cm["day_low"][j] == cm["day_close"][j])
        if pending_stop:
            px = float(o[0]) if o[0] < S else float(S)
            for kt in KT_LEDG:
                if not done[kt]:
                    res[kt].update(outcome="STOP",
                                   net_ret=float(px / entry - 1 - COST),
                                   touch=int(t[0]), exit_day=di)
                    done[kt] = True
            pending_stop = False
            if all(done.values()):
                break
            continue
        hit_stop = l <= S
        for kt in KT_LEDG:
            if done[kt]:
                continue
            T = Tk[kt]
            hit_t = h >= T
            if not hit_stop.any() and not hit_t.any():
                continue
            i_s = int(np.argmax(hit_stop)) if hit_stop.any() else 10 ** 9
            i_t = int(np.argmax(hit_t)) if hit_t.any() else 10 ** 9
            if i_s == i_t < 10 ** 9:
                ties += 1
            if i_s == 10 ** 9:  # 只触目标
                px = o[i_t] if o[i_t] > T else T
                res[kt].update(outcome="TARGET",
                               net_ret=float(px / entry - 1 - COST),
                               touch=int(t[i_t]), exit_day=di)
                done[kt] = True
            elif i_t == 10 ** 9 or i_s < i_t:  # 先触止损（同bar归属见ties审计）
                if o[i_s] < S:
                    if locked:
                        pending_stop = True
                        continue
                    res[kt].update(outcome="STOP",
                                   net_ret=float(o[i_s] / entry - 1 - COST),
                                   touch=int(t[i_s]), exit_day=di)
                    done[kt] = True
                else:
                    if locked:
                        pending_stop = True
                        continue
                    res[kt].update(outcome="STOP",
                                   net_ret=float(S / entry - 1 - COST),
                                   touch=int(t[i_s]), exit_day=di)
                    done[kt] = True
            else:  # 先触目标
                px = o[i_t] if o[i_t] > T else T
                res[kt].update(outcome="TARGET",
                               net_ret=float(px / entry - 1 - COST),
                               touch=int(t[i_t]), exit_day=di)
                done[kt] = True
        if all(done.values()):
            break
    for kt in KT_LEDG:
        if not done[kt]:
            res[kt]["net_ret"] = float(timeout_price / entry - 1 - COST)
    avail = sum(1 for j in fut if (cm["day_idx"] == j).any())
    return {"res": res,
            "mfe": mfe if np.isfinite(mfe) else np.nan,
            "mae": mae if np.isfinite(mae) else np.nan,
            "avail": avail, "need": len(fut), "ties": ties,
            "timeout_price": timeout_price}


def min_from_open(tt):
    return (tt // 10000) * 60 + (tt // 100) % 100 - 570


def ledger_features(cm, day_i, entry, atr_pct, ks, lim):
    """3d/5d 双账本 → 机会头经济标签 dict。"""
    days = cm["days"]
    pcb = np.full(len(days), np.nan)
    pcb[1:] = cm["day_close"][:-1]
    f = {}
    for h in (3, 5):
        sc = scan_ledger(cm, day_i, h, entry, atr_pct, ks, pcb, lim)
        if sc is None or sc["avail"] < sc["need"]:
            for k in (f"o_net_ret_{h}d", f"o_t1_bt_{h}d", f"o_t15_bt_{h}d",
                      f"o_stop_{h}d", f"o_timeout_{h}d", f"o_mfe_{h}d_atr",
                      f"o_mae_{h}d_atr", f"o_ttt_{h}d", f"o_close_ret_{h}d",
                      f"o_timeout_ret_{h}d", f"o_repair_but_loss_{h}d",
                      f"o_ties_{h}d"):
                f[k] = np.nan
            continue
        r2 = sc["res"][2.0]
        f[f"o_net_ret_{h}d"] = r2["net_ret"]
        f[f"o_t1_bt_{h}d"] = float(sc["res"][1.0]["outcome"] == "TARGET")
        f[f"o_t15_bt_{h}d"] = float(sc["res"][1.5]["outcome"] == "TARGET")
        f[f"o_stop_{h}d"] = float(r2["outcome"] == "STOP")
        f[f"o_timeout_{h}d"] = float(r2["outcome"] == "TIMEOUT")
        f[f"o_mfe_{h}d_atr"] = sc["mfe"] / atr_pct
        f[f"o_mae_{h}d_atr"] = sc["mae"] / atr_pct
        r1 = sc["res"][1.0]
        f[f"o_ttt_{h}d"] = float((r1["exit_day"] - 1) * 240
                                 + min_from_open(int(r1["touch"]))) \
            if r1["outcome"] == "TARGET" else np.nan
        f[f"o_close_ret_{h}d"] = float(sc["timeout_price"] / entry - 1 - COST)
        f[f"o_timeout_ret_{h}d"] = r2["net_ret"] \
            if r2["outcome"] == "TIMEOUT" else np.nan
        f[f"o_ties_{h}d"] = float(sc["ties"])
    return f


def main():
    global SAMPLES, OUT
    args = sys.argv[1:]
    if "--samples" in args:
        SAMPLES = args[args.index("--samples") + 1]
    if "--out" in args:
        OUT = args[args.index("--out") + 1]
    t0 = time.time()
    df = pd.read_parquet(SAMPLES)
    df = df[(df["dp"].isin(DPS)) & (df["entry_type"] == "vwap")
            & (df["censored"] == 0)].copy()
    g = (df.groupby(["code", "dp_date", "dp"])
           .agg(atr_pct=("atr_pct", "first"), ks=("ks", "first"),
                entry=("entry", "first"),
                outcome_t1=("outcome_t1.0", "first"),
                outcome_t2=("outcome_t2.0", "first"),
                net_ret2=("net_ret_t2.0", "first"))
           .reset_index())
    log(f"唯一 (code,dp_date,dp): {len(g):,}")
    rows, miss = [], 0
    cur, cm = None, None
    for _, r in g.iterrows():
        code = str(r["code"])
        if code != cur:
            cur = code
            cm = load_code(code)
        if cm is None:
            miss += 1
            continue
        hit = np.where(cm["days"] == int(r["dp_date"]))[0]
        if not len(hit):
            miss += 1
            continue
        atr_pct = float(r["atr_pct"])
        entry = float(r["entry"])
        ks = float(r["ks"])
        stop_px = entry * (1 - ks * atr_pct)
        rec03 = entry * (1 + 0.3 * atr_pct)
        rec05 = entry * (1 + 0.5 * atr_pct)
        day_i = int(hit[0])
        f = compute(cm, code, day_i, int(r["dp"]), entry, stop_px, rec03,
                    rec05, atr_pct)
        if f is None:
            miss += 1
            continue
        entry_used = f.pop("_entry")
        f.update(ledger_features(cm, day_i, entry_used, atr_pct, ks,
                                 LIMIT_OF(code)))
        for h in (3, 5):
            f[f"o_repair_but_loss_{h}d"] = float(
                f["o_rbs_t2"] == 1 and f[f"o_net_ret_{h}d"] < 0) \
                if np.isfinite(f[f"o_net_ret_{h}d"]) else np.nan
        rows.append({"code": code, "dp_date": int(r["dp_date"]),
                     "dp": int(r["dp"]),
                     "o_target_before_stop_5d": float(
                         r["outcome_t1"] == "TARGET"),
                     "y5": r["outcome_t2"], "nr5": float(r["net_ret2"]), **f})
    out = pd.DataFrame(rows)
    out.to_parquet(OUT, index=False)
    log(f"写入 {OUT}: {out.shape}；miss={miss}；用时 {time.time()-t0:.0f}s")
    for c in ("r_sbr_t2", "o_rbs_t2", "o_net_ret_3d", "o_net_ret_5d",
              "o_t1_bt_3d", "o_t1_bt_5d", "o_stop_3d", "o_stop_5d",
              "o_mfe_5d_atr", "o_mae_5d_atr", "o_repair_but_loss_3d",
              "o_repair_but_loss_5d", "o_ties_3d", "o_ties_5d",
              "r_stop_locked_only_t2", "r_same_day_extreme_adverse_move"):
        if c in out:
            log(f"  {c}: 均值 {out[c].mean():.4f} (NaN "
                f"{out[c].isna().mean():.1%})")
    log(f"cens30/cens60: {out['cens30'].mean():.2%}/{out['cens60'].mean():.2%}")


if __name__ == "__main__":
    main()
