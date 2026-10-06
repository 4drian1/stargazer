# -*- coding: utf-8 -*-
"""盘中路线第4步：分钟标签引擎（三决策点 09:45/10:00/10:30）。

**无未来函数口径**：日线信号在 T 收盘后产生 → 三个决策点都在 **T+1**（下一交易日），
entry = T+1 的 DP 后第一根 bar，退出评估从 T+2 起（T+1 买入不可当日卖），
horizon = T+2..T+6 共5个交易日。信号日 T 本身不参与特征（历史基线如20日
同窗口量比可以用 T 及以前的日子）。

对每个候选事件（minute_candidates_2025.parquet）× 每个决策点 DP：
- 特征只用 T+1 的 DP 前已完成的分钟K（无前视）；
- entry = DP 后第一根分钟 bar 的 open 或 vwap(amount/volume) 两种口径；
- barrier：Stop=Entry×(1−Ks×ATR_pct)，Target=Entry×(1+Kt×ATR_pct)，
  ATR_pct=atr_ref/entry_signal_close（尺度不变；日线与分钟同为 raw，§18.5）；
  A2来源 Ks=1.2，B0来源 Ks=1.5；4档目标 1.0/1.5/2.0/3.0 ATR；
- 同分钟双触按悲观记 STOP；跳空越过 barrier 按跳空价成交；
  止损日若整日跌停锁死（日聚合 close==low 且 ret≤−limit+0.5%）顺延，
  恢复日 open<S 按 open、否则按 S（与日线引擎同口径）；
- 输出 outcome/net_ret/first_touch/exit_day/mfe/mae/flags；
- 同事件三决策点共用 group_id，训练/验证切分必须按 group_id（不可拆散）。

重叠事件(A2+B0/LEGACY)按来源各生成一行（src∈{A2,B0,LEGACY}，Ks 不同；
LEGACY 止损同为 1.5 ATR）。
ATR 开盘冲击-承接特征组（open15_*，T+1 开盘第一根15分钟K，分母为信号日
冻结的 ATR_ref，任何 DP 下均无未来函数）；open15_relative_sector_ret 因
stockdb 无板块指数分钟数据留空（缺口已记录，M3 降级用日级板块强弱）。
除权窗口三标志（corp_action_* / price_scale_valid，§18.9）随样本带出，
跨除权窗口事件由下游消融脚本排除或单列。
输出：minute_samples_2025.parquet（每行=事件×决策点×entry口径）。
"""
import os
import sys
import time

import numpy as np
import pandas as pd

OUT_DIR = "D:/stargazer/stargazer_ml/pullback_train"
CACHE = os.environ.get(
    "MINUTE_CACHE_DIR", os.path.join(OUT_DIR, "minute_cache_clean"))
QUAR_PATH = os.path.join(OUT_DIR, "quarantine_2026.csv")
CAND = os.path.join(OUT_DIR, "minute_candidates_2025.parquet")
CAL_PATH = os.path.join(OUT_DIR, "market_daily.parquet")
OUT = os.path.join(OUT_DIR, "minute_samples_2025.parquet")
REPORT_PATH = os.path.join(OUT_DIR, "event_funnel_report.md")

DPS = (945, 1000, 1030, 1330, 1400, 1430, 1450)  # AM头+PM头（T3/T4，§18.17）
KTARGS = (1.0, 1.5, 2.0, 3.0)
COST = 0.002
KS_OF = {"A2": 1.2, "B0": 1.5, "LEGACY": 1.5}
LIMIT_OF = lambda c: (0.195 if c.startswith(("30", "68"))
                      else (0.295 if c.startswith(("8", "4", "92")) else 0.095))


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def board_limit(code):
    return LIMIT_OF(code)


# ---------------------------------------------------------------- code minutes
_code_cache = {}
_QUAR = None


def _quarantine():
    """隔离清单：code -> set(day)。存在即加载，load_code 剔除这些(code,day)。"""
    global _QUAR
    if _QUAR is None:
        _QUAR = {}
        if os.path.exists(QUAR_PATH):
            q = pd.read_csv(QUAR_PATH, dtype={"code": str, "day": np.int64})
            for c, g in q.groupby("code", sort=False):
                _QUAR[c] = set(g["day"].tolist())
    return _QUAR


def load_code(code):
    """code -> dict: days (sorted int YYYYMMDD), per-day dict of numpy arrays,
    plus day_close array aligned to days. 隔离(code,day)被剔除。"""
    if code in _code_cache:
        return _code_cache[code]
    p = os.path.join(CACHE, f"{code}.parquet")
    if not os.path.exists(p):
        _code_cache[code] = None
        return None
    df = pd.read_parquet(p)
    if not len(df):
        _code_cache[code] = None
        return None
    df = df.sort_values("date")
    days_all = ((df["date"].values) // 1000000).astype(int)
    days_all = np.unique(days_all)  # 含隔离日的完整日历（守卫用）
    qdays = _quarantine().get(code)
    if qdays:
        days_s = (df["date"].values // 1000000).astype(int)
        df = df[~np.isin(days_s, list(qdays))]
        if not len(df):
            _code_cache[code] = None
            return None
        df = df.sort_values("date")
    days = ((df["date"].values) // 1000000).astype(int)
    uniq, inv = np.unique(days, return_inverse=True)
    out = {"days": uniq, "days_all": days_all, "day_idx": inv,
           "t": df["date"].values % 1000000,
           "o": df["open"].values.astype(np.float64),
           "h": df["high"].values.astype(np.float64),
           "l": df["low"].values.astype(np.float64),
           "c": df["close"].values.astype(np.float64),
           "v": (df["volume"].values.astype(np.float64)
                 if "volume" in df else np.zeros(len(df))),
           "a": (df["amount"].values.astype(np.float64)
                 if "amount" in df else np.zeros(len(df)))}
    dc = np.full(len(uniq), np.nan)
    for i in range(len(uniq)):
        m = inv == i
        dc[i] = out["c"][m][-1]
    out["day_close"] = dc
    out["day_open"] = np.array([out["o"][inv == i][0] for i in range(len(uniq))])
    out["day_high"] = np.array([out["h"][inv == i].max() for i in range(len(uniq))])
    out["day_low"] = np.array([out["l"][inv == i].min() for i in range(len(uniq))])
    _code_cache[code] = out
    return out


# ---------------------------------------------------------------- features
def dp_features(cm, day_i, dp, pre_close, atr_px):
    """DP前已完成分钟特征。atr_px = pre_close*ATR_pct（元/ATR单位）。"""
    m = (cm["day_idx"] == day_i) & (cm["t"] <= dp * 100)
    if not m.any():
        return None
    o, h, l, c = cm["o"][m], cm["h"][m], cm["l"][m], cm["c"][m]
    v, a = cm["v"][m], cm["a"][m]
    last = c[-1]
    hi_s, lo_s = h.max(), l.min()
    f = {
        "bars": int(m.sum()),
        "last": float(last),
        "downside_atr": float((lo_s - pre_close) / atr_px),
        "shock_atr": float((hi_s - pre_close) / atr_px),
        "recovery": float((last - lo_s) / max(hi_s - lo_s, 1e-9)),
        "lower_wick_atr": float((np.minimum(o, c) - l).max() / atr_px),
        "vwap_reclaim_atr": float((last - (a.sum() / max(v.sum(), 1.0))) / atr_px),
        "kill_speed_atr": float((np.diff(np.concatenate([[pre_close], l])).min()
                                 / atr_px)),
    }
    m15 = (cm["day_idx"] == day_i) & (cm["t"] < 94500)
    if m15.any():
        f["first15_vol"] = float(cm["v"][m15].sum())
        f["first15_low_atr"] = float((cm["l"][m15].min() - pre_close) / atr_px)
        f["first15_range_atr"] = float((cm["h"][m15].max() - cm["l"][m15].min())
                                       / atr_px)
    else:
        f["first15_vol"], f["first15_low_atr"], f["first15_range_atr"] = np.nan, np.nan, np.nan
    # 9:45后是否继续创新低（相对开盘前15分钟低点）
    if dp > 945:
        m_after = (cm["day_idx"] == day_i) & (cm["t"] > 94500) & (cm["t"] <= dp * 100)
        f["new_low_after_945"] = int(m_after.any() and cm["l"][m_after].min() < cm["l"][m15].min()) \
            if m15.any() else 0
    else:
        f["new_low_after_945"] = 0
    # 二次回踩承接：最低价出现后，后续低点是否守住 low-0.25*ATR
    i_min = int(np.argmin(l))
    if len(l) - i_min - 1 > 0:
        f["dip_hold"] = int(l[i_min + 1:].min() >= lo_s - 0.25 * atr_px)
    else:
        f["dip_hold"] = 1
    return f


def hist_vol_ratio(cm, day_i, first15_vol):
    """前20个交易日同窗口(前15分钟)成交量中位数比。"""
    if not np.isfinite(first15_vol):
        return np.nan
    vals = []
    for j in range(max(0, day_i - 20), day_i):
        m = (cm["day_idx"] == j) & (cm["t"] < 94500)
        if m.any():
            vals.append(cm["v"][m].sum())
    if len(vals) < 10:
        return np.nan
    return float(first15_vol / np.median(vals))


def open15_features(cm, day_i, pre_close, atr_px):
    """T+1 开盘第一根15分钟K的 ATR 开盘冲击-承接特征组。

    分母 ATR_ref = 信号日冻结的 atr_ref（元/ATR单位）；pre_close = 信号日
    收盘（分钟日聚合 raw 尺度）。只用 09:30-09:45 已完成的bar，任何决策点
    (09:45/10:00/10:30)下都无未来函数。返回 None 表示无该bar。
    """
    m15 = (cm["day_idx"] == day_i) & (cm["t"] < 94500)
    if not m15.any():
        return None
    o = float(cm["o"][m15][0]); h = float(cm["h"][m15].max())
    l = float(cm["l"][m15].min()); c = float(cm["c"][m15][-1])
    v = float(cm["v"][m15].sum()); a = float(cm["a"][m15].sum())
    rng = max(h - l, 1e-9)
    tr = max(h - l, abs(h - pre_close), abs(l - pre_close))
    vwap = a / max(v, 1.0)
    return {
        "open15_tr_atr_ratio": tr / atr_px,
        "open15_gap_atr_ratio": abs(o - pre_close) / atr_px,
        "open15_range_atr_ratio": (h - l) / atr_px,
        "open15_downside_atr_ratio": max(0.0, o - l) / atr_px,
        "open15_upside_atr_ratio": (h - o) / atr_px,
        "open15_recovery_ratio": (c - l) / rng,
        "open15_close_location": (c - o) / max(h - o, o - l, 1e-9),
        "open15_lower_wick_ratio": (min(o, c) - l) / rng,
        "open15_vwap_reclaim": (c - vwap) / atr_px,
        "open15_vol": v,
    }


def add_interactions(row):
    """三个交互项（依赖 open15_relative_sector_ret 先填入）。"""
    shk = row.get("open15_tr_atr_ratio", np.nan)
    row["shock_recovery_ix"] = shk * row.get("open15_recovery_ratio", np.nan)
    row["downside_vwap_ix"] = (row.get("open15_downside_atr_ratio", np.nan)
                               * row.get("open15_vwap_reclaim", np.nan))
    row["shock_sector_ix"] = shk * row.get("open15_relative_sector_ret", np.nan)


# ---------------------------------------------------------------- labels
def scan_exit(cm, day_i, cal_days, entry, atr_pct, ks, pre_close_by_day, lim,
              code):
    """从 T+1 起扫 5 个交易日。返回 dict（对每个 target 的 outcome 等）
    及全窗口 mfe/mae。pre_close_by_day: 每日的 pre_close（raw）。"""
    nd = len(cm["days"])
    fut = [j for j in range(day_i + 1, min(day_i + 6, nd))]
    if not len(fut):
        return None
    S = entry * (1 - ks * atr_pct)
    Tk = {kt: entry * (1 + kt * atr_pct) for kt in KTARGS}
    res = {kt: {"outcome": "TIMEOUT", "net_ret": np.nan, "touch": np.nan,
                "exit_day": len(fut), "fill": "timeout_close"} for kt in KTARGS}
    mfe, mae = -np.inf, np.inf
    done = {kt: False for kt in KTARGS}
    pending_stop = False
    timeout_price = np.nan
    for di, j in enumerate(fut, 1):
        m = cm["day_idx"] == j
        o, h, l, c = cm["o"][m], cm["h"][m], cm["l"][m], cm["c"][m]
        t = cm["t"][m]
        if not len(o):
            continue
        mfe = max(mfe, float(h.max() / entry - 1))
        mae = min(mae, float(l.min() / entry - 1))
        timeout_price = float(c[-1])
        # 跌停锁死日判定（日聚合）
        pc = pre_close_by_day[j]
        locked = (np.isfinite(pc)
                  and (cm["day_close"][j] / pc - 1) <= -lim + 0.005
                  and cm["day_low"][j] == cm["day_close"][j])
        if pending_stop:
            # 恢复可卖日：open<S 按 open，否则按 S
            if o[0] < S:
                for kt in KTARGS:
                    if not done[kt]:
                        res[kt].update(outcome="STOP",
                                       net_ret=float(o[0] / entry - 1 - COST),
                                       touch=int(t[0]), exit_day=di,
                                       fill="stop_deferred_open")
                        done[kt] = True
            else:
                for kt in KTARGS:
                    if not done[kt]:
                        res[kt].update(outcome="STOP",
                                       net_ret=float(S / entry - 1 - COST),
                                       touch=int(t[0]), exit_day=di,
                                       fill="stop_deferred_stop")
                        done[kt] = True
            pending_stop = False
            if all(done.values()):
                break
            continue
        hit_stop = l <= S
        for kt in KTARGS:
            if done[kt]:
                continue
            T = Tk[kt]
            hit_t = h >= T
            if not hit_stop.any() and not hit_t.any():
                continue
            i_s = int(np.argmax(hit_stop)) if hit_stop.any() else 10 ** 9
            i_t = int(np.argmax(hit_t)) if hit_t.any() else 10 ** 9
            if i_s == 10 ** 9:  # 只触目标
                px = o[i_t] if o[i_t] > T else T
                res[kt].update(outcome="TARGET",
                               net_ret=float(px / entry - 1 - COST),
                               touch=int(t[i_t]), exit_day=di,
                               fill="target_gap" if o[i_t] > T else "target_level")
                done[kt] = True
            elif i_t == 10 ** 9 or i_s < i_t:  # 先触止损（同分钟双触悲观记STOP）
                if o[i_s] < S:
                    if locked:
                        pending_stop = True
                        continue
                    res[kt].update(outcome="STOP",
                                   net_ret=float(o[i_s] / entry - 1 - COST),
                                   touch=int(t[i_s]), exit_day=di,
                                   fill="stop_open")
                    done[kt] = True
                else:
                    if locked:
                        pending_stop = True
                        continue
                    res[kt].update(outcome="STOP",
                                   net_ret=float(S / entry - 1 - COST),
                                   touch=int(t[i_s]), exit_day=di,
                                   fill="stop_level")
                    done[kt] = True
            else:  # 先触目标
                px = o[i_t] if o[i_t] > T else T
                res[kt].update(outcome="TARGET",
                               net_ret=float(px / entry - 1 - COST),
                               touch=int(t[i_t]), exit_day=di,
                               fill="target_gap" if o[i_t] > T else "target_level")
                done[kt] = True
        if all(done.values()):
            break
    for kt in KTARGS:
        if not done[kt]:
            if pending_stop:
                res[kt].update(fill="stop_approx_close")
            res[kt]["net_ret"] = float(timeout_price / entry - 1 - COST)
    n_days = sum(1 for j in fut
                 if (cm["day_idx"] == j).any())
    return {"res": res, "mfe": mfe if np.isfinite(mfe) else np.nan,
            "mae": mae if np.isfinite(mae) else np.nan,
            "avail_days": n_days, "need_days": len(fut)}


# ---------------------------------------------------------------- main
def main():
    global CAND, OUT, REPORT_PATH
    args = sys.argv[1:]
    if "--cand" in args:
        CAND = args[args.index("--cand") + 1]
    if "--out" in args:
        OUT = args[args.index("--out") + 1]
    if "--no-report" in args:
        REPORT_PATH = None
    t0 = time.time()
    cal = sorted(int(d) for d in pd.read_parquet(CAL_PATH)["date"].unique())
    cand = pd.read_parquet(CAND)
    cand = cand[~cand["minute_unavailable"]].reset_index(drop=True)
    # 展开为 (code,date,src) 行
    flag_cols = [c for c in ("corp_action_on_signal_day",
                             "corp_action_during_horizon",
                             "price_scale_valid", "source_modes")
                 if c in cand.columns]
    rows = []
    for _, r in cand.iterrows():
        base = tuple(r[c] for c in flag_cols)
        if pd.notna(r["a2_event_id"]):
            rows.append((r["code"], int(r["date"]), "A2",
                         float(r["atr_ref"]), float(r["entry_qfq"]),
                         r["a2_event_id"]) + base)
        if pd.notna(r["b0_event_id"]):
            rows.append((r["code"], int(r["date"]), "B0",
                         float(r["atr_ref"]), float(r["entry_qfq"]),
                         r["b0_event_id"]) + base)
        if pd.notna(r["legacy_event_id"]):
            rows.append((r["code"], int(r["date"]), "LEGACY",
                         float(r["atr_ref"]), float(r["entry_qfq"]),
                         r["legacy_event_id"]) + base)
    ev = pd.DataFrame(rows, columns=["code", "date", "src", "atr_ref",
                                     "entry_close", "src_event_id"] + flag_cols)
    log(f"events expanded: {len(ev):,} (src: {ev['src'].value_counts().to_dict()})")

    out_rows = []
    cur_code, cm = None, None
    for i, r in ev.iterrows():
        if r["code"] != cur_code:
            cur_code = r["code"]
            cm = load_code(cur_code)
        if cm is None:
            continue
        days = cm["days"]
        hit = np.where(days == r["date"])[0]
        if not len(hit):
            continue
        day_i = int(hit[0])
        lim = board_limit(cur_code)
        atr_pct = r["atr_ref"] / r["entry_close"]
        atr_px = r["entry_close"] * atr_pct
        # 决策日在 T+1：信号 T 收盘后产生，最早 T+1 盘中才可买
        dp_i = day_i + 1
        if dp_i >= len(days):
            continue
        # 守卫：真实 T+1（含隔离日的完整日历）若被隔离，丢弃事件，
        # 不允许 dp_date 静默滑移到 T+2（§18.14 隔离偏差检查）
        hit_a = np.where(cm["days_all"] == r["date"])[0]
        if not len(hit_a) or int(hit_a[0]) + 1 >= len(cm["days_all"]):
            continue
        if days[dp_i] != cm["days_all"][int(hit_a[0]) + 1]:
            continue
        ks = KS_OF[r["src"]]
        # 特征日(T+1)的 pre_close = 信号日收盘（分钟日聚合，raw 尺度）
        pc = float(cm["day_close"][day_i])
        # 开盘15分钟 ATR 冲击-承接特征组（与 DP 无关，一次计算）
        of15 = open15_features(cm, dp_i, pc, atr_px)
        if of15 is not None:
            of15 = dict(of15)
            of15["open15_volume_ratio"] = hist_vol_ratio(cm, dp_i,
                                                         of15["open15_vol"])
            # 板块指数分钟数据 stockdb 无覆盖 → 留空（缺口，M3 降级日级强弱）
            of15["open15_relative_sector_ret"] = np.nan
            add_interactions(of15)
        # 每日 pre_close 序列（用于锁死判定）
        pcb = np.full(len(days), np.nan)
        pcb[0] = np.nan
        pcb[1:] = cm["day_close"][:-1]
        for dp in DPS:
            f = dp_features(cm, dp_i, dp, pc, atr_px)
            if f is None:
                continue
            f["vol_ratio15"] = hist_vol_ratio(cm, dp_i, f.get("first15_vol", np.nan))
            # entry：T+1 的 DP 后第一根 bar
            m_after = (cm["day_idx"] == dp_i) & (cm["t"] > dp * 100)
            if not m_after.any():
                continue
            j0 = int(np.where(m_after)[0][0])
            e_open = float(cm["o"][j0])
            e_vwap = (float(cm["a"][j0]) / cm["v"][j0]) if cm["v"][j0] > 0 else e_open
            for etype, entry in (("open", e_open), ("vwap", e_vwap)):
                # 退出从 T+2 起扫5个交易日（scan_exit 从 day_i+1 开始）
                sc = scan_exit(cm, dp_i, days, entry, atr_pct, ks, pcb, lim,
                               cur_code)
                if sc is None:
                    continue
                row = {
                    "group_id": i, "code": cur_code, "date": r["date"],
                    "dp_date": int(days[dp_i]),
                    "src": r["src"], "src_event_id": str(r["src_event_id"]),
                    "dp": dp, "entry_type": etype, "entry": entry,
                    "ks": ks, "atr_pct": float(atr_pct),
                    "sig_close": float(r["entry_close"]), "pre_close": pc,
                    "mfe": sc["mfe"], "mae": sc["mae"],
                    "avail_days": sc["avail_days"],
                    "need_days": sc["need_days"],
                    "censored": int(sc["avail_days"] < sc["need_days"]),
                }
                row.update({k: v for k, v in f.items()})
                if of15 is not None:
                    row.update(of15)
                for k in flag_cols:
                    row[k] = r[k]
                for kt in KTARGS:
                    rr = sc["res"][kt]
                    row[f"outcome_t{kt}"] = rr["outcome"]
                    row[f"net_ret_t{kt}"] = rr["net_ret"]
                    row[f"touch_t{kt}"] = rr["touch"]
                    row[f"exit_day_t{kt}"] = rr["exit_day"]
                    row[f"fill_t{kt}"] = rr["fill"]
                out_rows.append(row)
        if (i + 1) % 5000 == 0:
            log(f"{i+1}/{len(ev)} rows={len(out_rows)} "
                f"{time.time()-t0:.0f}s")

    df = pd.DataFrame(out_rows)
    df.to_parquet(OUT, index=False)
    log(f"wrote {OUT}: {df.shape}")

    # ---------------- 报告 §18.7（--no-report 时跳过，如 2026 复核跑） ----------------
    if REPORT_PATH is None:
        log(f"done {time.time()-t0:.0f}s (no report)")
        return
    with open(REPORT_PATH, encoding="utf-8") as f:
        old = f.read()
    idx = old.find("\n### 18.7 分钟标签引擎")
    if idx != -1:
        old = old[:idx] + "\n"
    n_ev = df["group_id"].nunique()
    L = ["\n### 18.7 分钟标签引擎（第4步，三决策点）\n",
         "每行=事件×决策点(09:45/10:00/10:30)×entry口径(open/vwap)。"
         "**决策点在 T+1**（信号 T 收盘后产生，T+1 才可买，无未来函数）；特征只用"
         "T+1 的 DP 前分钟；entry=DP 后第一根 bar；退出从 T+2 起评（T+1 买入不可"
         "当日卖）；horizon=T+2..T+6 共5交易日；barrier按 ATR_pct 比例法（日线分钟"
         "同raw尺度，§18.5）；同分钟双触悲观记STOP；止损日跌停锁死顺延（与日线引擎"
         "同口径）。同事件三DP共用 group_id，切分按 group_id 防拆散。\n",
         f"- 展开事件：{len(ev):,}（{ev['src'].value_counts().to_dict()}）",
         f"- 覆盖事件（至少1个DP样本）：{n_ev:,} / {len(ev):,}",
         f"- 样本行：{len(df):,}（DP分布 {df['dp'].value_counts().to_dict()}）",
         f"- entry口径分布 {df['entry_type'].value_counts().to_dict()}",
         f"- censored（未来不足5日）: {int(df['censored'].sum()):,}",
         "- 特征组v1：downside/shock_atr、recovery、lower_wick_atr、"
         "vwap_reclaim_atr、kill_speed_atr、first15_low/range_atr、"
         "vol_ratio15(前20日同窗口基线)、new_low_after_945、dip_hold",
         "- 特征组v2(ATR开盘冲击-承接)：open15_tr/gap/range/downside/upside_atr_ratio、"
         "recovery_ratio、close_location、lower_wick_ratio、vwap_reclaim、"
         "volume_ratio + 交互 shock_recovery/downside_vwap/shock_sector_ix；"
         "open15_relative_sector_ret 因板块分钟数据缺失留空（缺口记录）",
         "- src∈{A2,B0,LEGACY}（LEGACY止损1.5ATR）；随样本带出除权三标志"
         "（§18.9），跨除权窗口事件由下游排除或单列",
         "- 标签：4档目标×(outcome/net_ret/touch/exit_day/fill) + mfe/mae",
         "\n四口径对照（收盘买入/固定DP/open15规则/盘中模型）见 "
         "minute_four_way.py 输出的 §18.8。"]
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(old.rstrip("\n") + "\n" + "\n".join(L) + "\n")
    log(f"done {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
