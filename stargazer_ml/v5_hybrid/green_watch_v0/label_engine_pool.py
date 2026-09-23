"""label_engine_pool.py — 退出标签扩到 GREEN-WATCH 池宇宙（EXIT-GREEN-v2 ①② 共用引擎）。

事件输入：pool_daily.parquet 的 WATCH+ 事件（primary_tier ∈ WATCH/HIGH_PRIORITY_WATCH）。
规则：严格复刻 green_exit_v1/build_exit_labels.py 的 run_exit 语义（exit_dyn_spec_v1 冻结）：
  T+1 开盘入场（涨停顺延至 T+5）、ATR_ref=median(ATR14[T-4..T]) 冻结、
  硬止损 clip(1.5ATR,3%,6%)、同 bar 悲观序、高水位只用当前及之前 bar、T+1 不可卖、
  跌停锁死顺延 +3 日开盘出场、prev_close 链式口径、is_st 用 parts 列（偏离注明）、
  2025+ 有分钟缓存走 MINUTE_EXACT（T+5 用 14:51–14:55 VWAP）否则 DAILY_APPROX、
  成本 ROUND_TRIP_COST=0.003（旧口径，注明不重算）。

变体表（v1 四变体 + EXIT-GREEN-v2 消融 E0–E5）：
  X0_CONTROL/X1/X1_RF/X2_STAGED：冻结 v1 四变体（E2≡X1_daily、E0≡X0_daily，仍单独算以便对齐）。
  E0=止损+T+5；E1=止损+二档trailing+T+5；E2=止损+一二档trailing+T+5；
  E3=E1+反弹失败；E4=止损+T+10；E5=E1+T+10。E 系全历史强制 DAILY_APPROX（设计 §43）。

实现：X0 走 numpy 快路径；其余逐 bar 调原引擎 ExitState（v1 变体）或 ExitStateV2
（E 系，step 逻辑逐行复制自冻结 ExitState 并加 use_trail1/use_trail2/use_rf 门控）。
多进程按 code 分组。X2 staging 语义与原引擎 finish() 一致（staging 后任意退出按
blended 计并标 REBOUND_FAILURE_STAGED）。

运行：
  python label_engine_pool.py --variants X0_CONTROL,X1,X1_RF,X2_STAGED --outdir labels_pool
  python label_engine_pool.py --variants E0,E1,E2,E3,E4,E5 --outdir labels_pool_e
"""
from __future__ import annotations
import argparse
import glob
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent
GE = Path("d:/stargazer/stargazer_ml/v5_hybrid/green_exit_v1")
sys.path.insert(0, str(GE))
from build_exit_labels import (  # noqa: E402  冻结语义单一事实源
    ExitState, STOP_MULT, STOP_MIN, STOP_MAX, ROUND_TRIP_COST,
    PRECISION_MIN_YEAR, BLOCK_MAX_DELAY,
    LIMIT_UP_MULT, LIMIT_DOWN_MULT, ST_LIMIT_UP_MULT, ST_LIMIT_DOWN_MULT,
)

POOL_F = BASE / "pool_daily.parquet"
MINUTE_DIR = Path("d:/stargazer/stargazer_ml/pullback_train/minute_cache_clean")
BARS_GLOB = "D:/a股数据/data_daily_parts/part_*.parquet"

# ---------------------------------------------------------------- 变体表
# daily=True 强制 DAILY_APPROX（E 系，设计 §43「日线近似全历史」）
VARIANTS = {
    "X0_CONTROL": dict(tmax=4, trail1=False, trail2=False, rf=False, staged=False,
                       v1=True, fast=True),
    "X1":         dict(tmax=4, trail1=True,  trail2=True,  rf=False, staged=False,
                       v1=True),
    "X1_RF":      dict(tmax=4, trail1=True,  trail2=True,  rf=True,  staged=False,
                       v1=True),
    "X2_STAGED":  dict(tmax=4, trail1=True,  trail2=True,  rf=True,  staged=True,
                       v1=True),
    "E0": dict(tmax=4, trail1=False, trail2=False, rf=False, staged=False, daily=True),
    "E1": dict(tmax=4, trail1=False, trail2=True,  rf=False, staged=False, daily=True),
    "E2": dict(tmax=4, trail1=True,  trail2=True,  rf=False, staged=False, daily=True),
    "E3": dict(tmax=4, trail1=False, trail2=True,  rf=True,  staged=False, daily=True),
    "E4": dict(tmax=9, trail1=False, trail2=False, rf=False, staged=False, daily=True),
    "E5": dict(tmax=9, trail1=False, trail2=True,  rf=False, staged=False, daily=True),
}

COLS = ["date", "code", "open", "high", "low", "close", "is_st"]


class ExitStateV2(ExitState):
    """E 系变体的门控状态机：step() 逐行复制冻结 ExitState.step()，
    仅把 trail1/trail2/反弹失败 三分支改为显式开关（v2 设计矩阵所需）。"""

    def __init__(self, entry, atr_ref, stop, use_trail1, use_trail2, use_rf):
        super().__init__(entry, atr_ref, stop, "E")
        self.use_trail1, self.use_trail2, self.use_rf = use_trail1, use_trail2, use_rf

    def step(self, o, h, l, c, dt, can_sell):
        self._last_dt = dt
        e = self.entry
        t1p, t1d, t2p, t2t, rfd, rfr, rff = self.thresholds()
        old_hw = self.hw
        peak = old_hw / e - 1
        if not self.activated1 and peak >= t1p:
            self.pending1 = True
        if not self.activated2 and peak >= t2p:
            self.pending2 = True
        rf_trig = False
        if self.use_rf and not self.rf_disabled:
            if not self.damaged and l <= e * (1 - rfd):
                self.damaged = True
                self.reb_low = l
                self.reb_high = h
            elif self.damaged:
                self.reb_low = min(self.reb_low, l)
                self.reb_high = max(self.reb_high, h)
                if (self.reb_high - self.reb_low) >= rfr and \
                   (self.reb_high - c) >= rff and c < e:
                    rf_trig = True
        if not can_sell:
            self._commit(old_hw, h)
            return None
        if o <= self.stop:
            return ("STOP", o, dt)
        if l <= self.stop:
            return ("STOP", self.stop, dt)
        if rf_trig:
            return ("REBOUND_FAILURE", c, dt)
        if self.use_trail2 and self.activated2:
            trail = old_hw * (1 - t2t)
            if o <= trail:
                return ("TRAIL2", o, dt)
            if l <= trail:
                return ("TRAIL2", trail, dt)
        elif self.use_trail1 and self.activated1:
            dd = old_hw * (1 - t1d)
            if o <= dd:
                return ("TRAIL1", o, dt)
            if l <= dd:
                return ("TRAIL1", dd, dt)
        self._commit(old_hw, h)
        return None


# ---------------------------------------------------------------- 数据装载
def load_parts(codes):
    parts = [pd.read_parquet(f, columns=COLS) for f in sorted(glob.glob(BARS_GLOB))]
    df = pd.concat(parts, ignore_index=True)
    df["code"] = df["code"].astype(str).str.zfill(6)
    df = df[df["code"].isin(codes)]
    df = df.sort_values(["code", "date"]).reset_index(drop=True)
    out = {}
    for c, g in df.groupby("code", sort=False):
        out[c] = dict(
            date=g["date"].to_numpy(np.int64),
            open=g["open"].to_numpy(np.float64),
            high=g["high"].to_numpy(np.float64),
            low=g["low"].to_numpy(np.float64),
            close=g["close"].to_numpy(np.float64),
            is_st=g["is_st"].fillna(False).to_numpy(bool),
        )
    return out


def _day_bounds(day):
    idx = np.flatnonzero(np.r_[True, day[1:] != day[:-1], True])
    return [(day[idx[k]], idx[k], idx[k + 1]) for k in range(len(idx) - 1)]


def load_minute_slices(code):
    f = MINUTE_DIR / f"{code}.parquet"
    if not f.exists():
        return None
    df = pd.read_parquet(f, columns=["date", "open", "high", "low", "close",
                                     "amount", "volume"])
    df = df.sort_values("date")
    ts = df["date"].to_numpy(np.int64)
    day = ts // 1000000
    out, vwap = {}, {}
    for d, s, e in _day_bounds(day):
        sl = slice(s, e)
        out[int(day[s])] = (df["open"].to_numpy(np.float64)[sl],
                            df["high"].to_numpy(np.float64)[sl],
                            df["low"].to_numpy(np.float64)[sl],
                            df["close"].to_numpy(np.float64)[sl],
                            ts[sl])
        vb = df.iloc[s:e]
        mask = (vb["date"] % 1000000 >= 145100) & (vb["date"] % 1000000 <= 145500)
        vw = vb[mask]
        v = float(vw["amount"].sum() / vw["volume"].sum()) \
            if len(vw) and vw["volume"].sum() > 0 else float(vb["close"].iloc[-1])
        vwap[int(day[s])] = v
    return out, vwap


# ---------------------------------------------------------------- 事件级语义
def _prep_code(px):
    n = len(px["date"])
    pc = np.empty(n)
    pc[0] = px["close"][0]
    pc[1:] = px["close"][:-1]
    tr = np.maximum.reduce([px["high"] - px["low"],
                            np.abs(px["high"] - pc), np.abs(px["low"] - pc)])
    atr14 = pd.Series(tr).rolling(14, min_periods=14).mean().to_numpy()
    return pc, atr14


def _atr_ref(atr14, entry_idx):
    need = np.arange(entry_idx - 5, entry_idx)
    valid = need[(need >= 13) & (need < len(atr14))]
    if valid.size == 0:
        return None
    return float(np.median(atr14[valid]))


def _limit_mults(st):
    return (ST_LIMIT_UP_MULT, ST_LIMIT_DOWN_MULT) if st else (LIMIT_UP_MULT, LIMIT_DOWN_MULT)


def _locked_dn(close_p, low_p, pc, st):
    if not pc or pc <= 0:
        return False
    _, dn = _limit_mults(st)
    return close_p <= pc * dn + 1e-9 and abs(close_p - low_p) < 1e-6


def _find_entry(px, pc, i, n):
    for k in range(1, 6):
        cand = i + k
        if cand >= n:
            return None
        up, _ = _limit_mults(bool(px["is_st"][cand]))
        if pc[cand] > 0 and px["open"][cand] >= pc[cand] * up - 1e-9:
            continue
        if cand + 1 >= n:
            return None
        return cand
    return None


def _base_rec(entry_date, entry, atr_ref, stop):
    return dict(entry_date=int(entry_date), entry_time=int(entry_date),
                entry_price=float(entry), atr_ref=float(atr_ref),
                initial_stop_price=float(stop))


def _mk_finish(rb, entry, staged_first):
    """生成 finish 闭包（X2 staging 后任意退出按 blended 计，同原引擎 finish()）。"""
    def finish(reason, price, exit_time, blocked, precision, extra=None):
        if staged_first is not None:
            blended = 0.5 * (staged_first / entry - 1) + 0.5 * (price / entry - 1)
            gross, reason2 = blended, "REBOUND_FAILURE_STAGED"
            rec = dict(rb)
            rec.update(exit_time=int(exit_time), exit_price=float(price),
                       exit_reason=reason2, exit_blocked=int(blocked),
                       gross_return=float(gross), cost=ROUND_TRIP_COST,
                       net_return=float(gross - ROUND_TRIP_COST),
                       label_precision=precision)
        else:
            gross = price / entry - 1
            rec = dict(rb)
            rec.update(exit_time=int(exit_time), exit_price=float(price),
                       exit_reason=reason, exit_blocked=int(blocked),
                       gross_return=float(gross), cost=ROUND_TRIP_COST,
                       net_return=float(gross - ROUND_TRIP_COST),
                       label_precision=precision)
        if extra:
            rec.update(extra)
        return rec
    return finish


def _blocked_defer(px, pc, day_idx, n, reason, precision, finish):
    for dd in range(day_idx + 1, min(day_idx + 1 + BLOCK_MAX_DELAY, n)):
        if not _locked_dn(px["close"][dd], px["low"][dd], pc[dd], bool(px["is_st"][dd])):
            return finish(reason, px["open"][dd], int(px["date"][dd]), 1, precision), dd
    dd = min(day_idx + BLOCK_MAX_DELAY, n - 1)
    return finish(reason, px["open"][dd], int(px["date"][dd]), 1, precision), dd


# ---- 通用动态变体驱动（X1 / X1_RF / X2_STAGED / E0–E5）----
def run_dyn(px, pc, entry_idx, atr_ref, min_slice, vwap_map, n, cfg):
    entry = float(px["open"][entry_idx])
    stop_dist = min(max(STOP_MULT * atr_ref / entry, STOP_MIN), STOP_MAX)
    stop = entry * (1 - stop_dist)
    t5 = min(entry_idx + cfg["tmax"], n - 1)
    if t5 <= entry_idx:
        return None
    if cfg.get("v1"):
        st = ExitState(entry, atr_ref, stop,
                       "X2_STAGED" if cfg["staged"] else ("X1_RF" if cfg["rf"] else "X1"))
    else:
        st = ExitStateV2(entry, atr_ref, stop, cfg["trail1"], cfg["trail2"], cfg["rf"])
    staged_first = None
    has_min = (min_slice is not None) and not cfg.get("daily")
    precision = "MINUTE_EXACT" if has_min else "DAILY_APPROX"
    rb = _base_rec(px["date"][entry_idx], entry, atr_ref, stop)
    finish = _mk_finish(rb, entry, None)   # staged_first 动态处理见下
    blocked = None

    def finish2(reason, price, exit_time, blocked_, precision_, extra=None):
        f = _mk_finish(rb, entry, staged_first)
        return f(reason, price, exit_time, blocked_, precision_, extra)

    ed = entry_idx
    st.step(float(px["open"][ed]), float(px["high"][ed]), float(px["low"][ed]),
            float(px["close"][ed]), int(px["date"][ed]), can_sell=False)
    for d in range(entry_idx + 1, t5 + 1):
        cdate = int(px["date"][d])
        sellable = not _locked_dn(px["close"][d], px["low"][d], pc[d],
                                  bool(px["is_st"][d]))
        ms = min_slice.get(cdate) if has_min else None
        if ms is not None:
            mo, mh, ml, mc, mts = ms
            done = False
            for j in range(len(mts)):
                if d == t5 and int(mts[j] % 1000000) >= 145000:
                    if not sellable:
                        blocked = ("TIME_EXIT", d)
                        done = True
                        break
                    return finish2("TIME_EXIT", vwap_map[cdate], int(mts[j]), 0,
                                   precision, dict(holding_days=d - entry_idx,
                                                   holding_minutes=(d - entry_idx) * 390,
                                                   high_watermark=st.hw,
                                                   activation_time=st.t1_act or 0))
                r = st.step(mo[j], mh[j], ml[j], mc[j], int(mts[j]), can_sell=True)
                if r:
                    if cfg["staged"] and r[0] in ("REBOUND_FAILURE", "REBOUND_FAILURE_STAGED") \
                            and staged_first is None:
                        staged_first = r[1]
                        st.staged = True
                        st.rf_disabled = True
                        st.activated2 = True
                        continue
                    if not sellable:
                        blocked = (r[0], d)
                        done = True
                        break
                    return finish2(r[0], r[1], int(r[2]), 0, precision,
                                   dict(holding_days=d - entry_idx,
                                        holding_minutes=(d - entry_idx) * 390,
                                        high_watermark=st.hw,
                                        activation_time=st.t1_act or 0))
            if done:
                break
        else:
            r = st.step(float(px["open"][d]), float(px["high"][d]),
                        float(px["low"][d]), float(px["close"][d]), cdate,
                        can_sell=True)
            if r:
                if cfg["staged"] and r[0] in ("REBOUND_FAILURE", "REBOUND_FAILURE_STAGED") \
                        and staged_first is None:
                    staged_first = r[1]
                    st.staged = True
                    st.rf_disabled = True
                    st.activated2 = True
                    continue
                if not sellable:
                    blocked = (r[0], d)
                    break
                return finish2(r[0], r[1], int(r[2]), 0, precision,
                               dict(holding_days=d - entry_idx,
                                    holding_minutes=(d - entry_idx) * 390,
                                    high_watermark=st.hw,
                                    activation_time=st.t1_act or 0))
    if blocked is not None:
        r, dd = _blocked_defer(px, pc, blocked[1], n, blocked[0], precision, finish2)
        if r:
            r.update(holding_days=dd - entry_idx,
                     holding_minutes=(dd - entry_idx) * 390,
                     high_watermark=st.hw, activation_time=st.t1_act or 0)
        return r
    if _locked_dn(px["close"][t5], px["low"][t5], pc[t5], bool(px["is_st"][t5])):
        for dd in range(t5 + 1, min(t5 + 1 + BLOCK_MAX_DELAY, n)):
            if not _locked_dn(px["close"][dd], px["low"][dd], pc[dd],
                              bool(px["is_st"][dd])):
                return finish2("TIME_EXIT", px["open"][dd], int(px["date"][dd]), 1,
                               precision, dict(holding_days=dd - entry_idx,
                                               holding_minutes=(dd - entry_idx) * 390,
                                               high_watermark=st.hw,
                                               activation_time=st.t1_act or 0))
        dd = min(t5 + BLOCK_MAX_DELAY, n - 1)
        return finish2("TIME_EXIT", px["open"][dd], int(px["date"][dd]), 1, precision,
                       dict(holding_days=dd - entry_idx,
                            holding_minutes=(dd - entry_idx) * 390,
                            high_watermark=st.hw, activation_time=st.t1_act or 0))
    if has_min and t5 in min_slice:
        mts = min_slice[t5][4]
        return finish2("TIME_EXIT", vwap_map[int(px["date"][t5])], int(mts[-1]), 0,
                       precision, dict(holding_days=t5 - entry_idx,
                                       holding_minutes=(t5 - entry_idx) * 390,
                                       high_watermark=st.hw,
                                       activation_time=st.t1_act or 0))
    return finish2("TIME_EXIT", px["close"][t5], int(px["date"][t5]), 0, precision,
                   dict(holding_days=t5 - entry_idx,
                        holding_minutes=(t5 - entry_idx) * 390,
                        high_watermark=st.hw, activation_time=st.t1_act or 0))


# ---- X0 快路径（等价 ExitState 在 X0 下的逐 bar 结果；E0 每日线亦走通用驱动对照）----
def run_x0(px, pc, entry_idx, atr_ref, min_slice, vwap_map, n):
    entry = float(px["open"][entry_idx])
    stop_dist = min(max(STOP_MULT * atr_ref / entry, STOP_MIN), STOP_MAX)
    stop = entry * (1 - stop_dist)
    t5 = min(entry_idx + 4, n - 1)
    if t5 <= entry_idx:
        return None
    has_min = min_slice is not None
    precision = "MINUTE_EXACT" if has_min else "DAILY_APPROX"
    rb = _base_rec(px["date"][entry_idx], entry, atr_ref, stop)
    finish = _mk_finish(rb, entry, None)
    blocked = None
    for d in range(entry_idx + 1, t5 + 1):
        cdate = int(px["date"][d])
        sellable = not _locked_dn(px["close"][d], px["low"][d], pc[d],
                                  bool(px["is_st"][d]))
        ms = min_slice.get(cdate) if has_min else None
        trig = None
        if ms is not None:
            mo, mh, ml, mc, mts = ms
            if d == t5:
                cut = int(np.searchsorted(mts % 1000000, 145000))
                for j in range(cut):
                    if mo[j] <= stop:
                        trig = ("STOP", mo[j], int(mts[j]))
                        break
                    if ml[j] <= stop:
                        trig = ("STOP", stop, int(mts[j]))
                        break
                if trig is None:
                    if not sellable:
                        blocked = ("TIME_EXIT", d)
                        break
                    exit_ts = int(mts[cut]) if cut < len(mts) else int(mts[-1])
                    return finish("TIME_EXIT", vwap_map[cdate], exit_ts, 0,
                                  precision, dict(holding_days=d - entry_idx,
                                                  holding_minutes=(d - entry_idx) * 390))
            else:
                for j in range(len(mts)):
                    if mo[j] <= stop:
                        trig = ("STOP", mo[j], int(mts[j]))
                        break
                    if ml[j] <= stop:
                        trig = ("STOP", stop, int(mts[j]))
                        break
        if trig is None and ms is None:
            if px["open"][d] <= stop:
                trig = ("STOP", px["open"][d], cdate)
            elif px["low"][d] <= stop:
                trig = ("STOP", stop, cdate)
        if trig is not None:
            if not sellable:
                blocked = (trig[0], d)
                break
            return finish(trig[0], trig[1], trig[2], 0, precision,
                          dict(holding_days=d - entry_idx,
                               holding_minutes=(d - entry_idx) * 390))
    if blocked is not None:
        r, dd = _blocked_defer(px, pc, blocked[1], n, blocked[0], precision, finish)
        if r:
            r.update(holding_days=dd - entry_idx,
                     holding_minutes=(dd - entry_idx) * 390)
        return r
    if _locked_dn(px["close"][t5], px["low"][t5], pc[t5], bool(px["is_st"][t5])):
        for dd in range(t5 + 1, min(t5 + 1 + BLOCK_MAX_DELAY, n)):
            if not _locked_dn(px["close"][dd], px["low"][dd], pc[dd],
                              bool(px["is_st"][dd])):
                return finish("TIME_EXIT", px["open"][dd], int(px["date"][dd]), 1,
                              precision, dict(holding_days=dd - entry_idx,
                                              holding_minutes=(dd - entry_idx) * 390))
        dd = min(t5 + BLOCK_MAX_DELAY, n - 1)
        return finish("TIME_EXIT", px["open"][dd], int(px["date"][dd]), 1, precision,
                      dict(holding_days=dd - entry_idx,
                           holding_minutes=(dd - entry_idx) * 390))
    if has_min and t5 in min_slice:
        mts = min_slice[t5][4]
        return finish("TIME_EXIT", vwap_map[int(px["date"][t5])], int(mts[-1]), 0,
                      precision, dict(holding_days=t5 - entry_idx,
                                      holding_minutes=(t5 - entry_idx) * 390))
    return finish("TIME_EXIT", px["close"][t5], int(px["date"][t5]), 0, precision,
                  dict(holding_days=t5 - entry_idx,
                       holding_minutes=(t5 - entry_idx) * 390))


def run_event(px, pc, atr14, loc, sd, n, min_slice, vwap_map, variants):
    i = loc.get(int(sd))
    if i is None:
        return None
    entry_idx = _find_entry(px, pc, i, n)
    if entry_idx is None:
        return None
    atr_ref = _atr_ref(atr14, entry_idx)
    if atr_ref is None or not np.isfinite(atr_ref) or atr_ref <= 0:
        return None
    use_min = int(sd) >= PRECISION_MIN_YEAR * 10000
    ms = min_slice if use_min else None
    vm = vwap_map if use_min else None
    out = {}
    for v in variants:
        cfg = VARIANTS[v]
        if v == "X0_CONTROL":
            r = run_x0(px, pc, entry_idx, atr_ref, ms, vm, n)
        else:
            r = run_dyn(px, pc, entry_idx, atr_ref, ms, vm, n, cfg)
        if r is not None:
            r["strategy_variant"] = v
        out[v] = r
    return out


# ---------------------------------------------------------------- per-code 驱动
def process_code(args):
    code, px, signal_dates, variants = args
    n = len(px["date"])
    pc, atr14 = _prep_code(px)
    loc = {int(d): i for i, d in enumerate(px["date"])}
    recs = []
    has25 = any(int(sd) >= PRECISION_MIN_YEAR * 10000 for sd in signal_dates)
    minute = load_minute_slices(code) if has25 else None
    min_slice, vwap_map = minute if minute else (None, None)
    for sd in signal_dates:
        res = run_event(px, pc, atr14, loc, sd, n, min_slice, vwap_map, variants)
        if res is None:
            continue
        for v, r in res.items():
            if r is None:
                continue
            r["event_id"] = f"GW:{code}:{int(sd)}"
            r["code"] = code
            r["signal_date"] = int(sd)
            r["exit_signal_time"] = r["exit_time"]
            recs.append(r)
    return recs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--limit-codes", type=int, default=0)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--tier", default="WATCH+", help="WATCH+ = 只 WATCH 及以上；ALL = 全部池行")
    args = ap.parse_args()
    variants = tuple(v for v in args.variants.split(",") if v in VARIANTS)
    assert variants, f"未知变体：{args.variants}"
    outdir = BASE / args.outdir
    t0 = time.time()
    print("[1/4] 池事件 …", flush=True)
    cols = ["date", "code", "primary_tier"]
    pool = pd.read_parquet(POOL_F, columns=cols)
    pool["code"] = pool["code"].astype(str).str.zfill(6)
    if args.tier == "WATCH+":
        pool = pool[pool["primary_tier"].isin(["WATCH", "HIGH_PRIORITY_WATCH"])]
    codes = sorted(pool["code"].unique())
    if args.limit_codes:
        codes = codes[: args.limit_codes]
    ev_by_code = {c: g["date"].to_numpy() for c, g in
                  pool[pool["code"].isin(codes)].groupby("code", sort=False)}
    print(f"  事件 {sum(len(v) for v in ev_by_code.values())}，票 {len(codes)}"
          f"（tier={args.tier}，variants={variants}）", flush=True)

    print("[2/4] 装载日线 …", flush=True)
    px_all = load_parts(set(codes))
    print(f"  {len(px_all)} 票行情，用时 {time.time()-t0:.0f}s", flush=True)

    print("[3/4] 事件级标签（多进程）…", flush=True)
    items = [(c, px_all[c], ev_by_code.get(c, np.array([])), variants)
             for c in codes if c in px_all and len(ev_by_code.get(c, []))]
    del px_all
    all_recs = []
    if args.workers > 1:
        import multiprocessing as mp
        with mp.Pool(args.workers) as p:
            for k, recs in enumerate(p.imap_unordered(process_code, items,
                                                      chunksize=8)):
                all_recs.extend(recs)
                if k % 300 == 0:
                    print(f"  {k}/{len(items)} codes, recs={len(all_recs)}, "
                          f"{time.time()-t0:.0f}s", flush=True)
    else:
        for k, it in enumerate(items):
            all_recs.extend(process_code(it))
            if k % 100 == 0:
                print(f"  {k}/{len(items)}, recs={len(all_recs)}", flush=True)
    df = pd.DataFrame(all_recs)
    print(f"  标签 {len(df)} 行（{df['strategy_variant'].value_counts().to_dict()}）",
          flush=True)

    print("[4/4] 落盘 …", flush=True)
    outdir.mkdir(parents=True, exist_ok=True)
    for prec in ("DAILY_APPROX", "MINUTE_EXACT"):
        sub = df[df["label_precision"] == prec]
        if sub.empty:
            continue
        out = outdir / f"exit_labels_pool_{prec.lower()}.parquet"
        sub.to_parquet(out, index=False)
        print(f"  {prec}: {len(sub)} 行 -> {out}", flush=True)
    print(f"完成，用时 {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
