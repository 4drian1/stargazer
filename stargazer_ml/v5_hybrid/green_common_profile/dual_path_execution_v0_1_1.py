# -*- coding: utf-8 -*-
"""双路径执行引擎 v0.1.1 (DUAL_PATH_EXECUTION_V0_1_1) —— 独立新执行路径.

严格按 contracts/dual_path_execution_contract_v0_1_1_final.md 实现; 不覆盖旧
gcp_round_b_engine.label_event 的 y/f. 所有模拟成交标 RESEARCH_APPROXIMATION.

覆盖:
  - 时间语义: 14 位 ts, 交易网格 [09:31,11:30]/[13:01,14:56] 为成交槽, 09:30/15:00 为源观测
  - 入场 (§4): S 后下一合法 bar open; 容量 q_cap=floor(0.01*vol); 资金 q_cash=floor(50000/(p*1.0013));
              板块数量规则 (主板/创业板 100 递增; 科创 200 起 1 股递增); 限锁/零量不成交
  - T+1 + 硬止损 (§5): D_stop=clip(1.5*atr/sim_fill,3%,6%); StopPrice=sim_fill*(1-D_stop);
              正式止损自 E 下一交易日; 触发 bar 不成交, 下一合法 bar open 卖出; 跳空 09:31 处理
  - 预定退出 (§5): 计划日 14:55 -> 下一合法 14:56 open; STOP 优先
  - 卖出/顺延 (§6): 1% 容量; 部分卖出; 顺延起点 d0=首次退出指令日, 到 d0+5 交易日 14:56; 不重置
  - 限价 (§7): 主/创/科 + ST 日期分段 (Decimal round_half_up); IPO 前 5 日 NO_DAILY_LIMIT
  - 缺槽/隔离 (§8): 扫描窗内任一缺槽即拒; 状态字典

H3/H5 为同入场复制的独立持仓轨道, 各保 status/reason/net/依赖终点.
"""
from __future__ import annotations

import json
import sys
import glob
import math
import os
import hashlib
from pathlib import Path
from decimal import Decimal, ROUND_HALF_UP

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import gcp_round_b_engine as E  # 复用 build_atr_ref/build_zones/detect_signal/hhmm/validate_minute_day

# ---- 冻结参数 (与 manifest 一致) ----
BUY_MULT = 1.0013
SELL_MULT = 0.9982
ORDER_CAP = 50000.0
PART_FRAC = 0.01
STOP_MULT = 1.5
STOP_PCT_MIN = 0.03
STOP_PCT_MAX = 0.06
PLANNED_EXIT_HHMM = 1455
HORIZONS = [3, 5]
DEFER_MAX_DAYS = 5
DATA_CUTOFF = 20260928
ST_MAIN_NEW_RULE = 20260706
TICK = 0.01

# 成交槽范围 (HHMMSS)
FILL_RANGE = [((93100, 113000), (130100, 145600))]
GRID_PATH = HERE / "gcp_round_b_small_sample_v1" / "minute_grid.json"
DAILY_PARTS = os.environ.get("STARGAZER_DATA_ROOT", r"D:/a股数据") + r"\data_daily_parts"
# 分钟缓存目录 (顺序即读取优先级; by_pair 为 realtest 采集的持仓窗口日, 优先于常规切片)
MIN_DIRS = [
    HERE / "gcp_round_b_small_sample_v1" / "minute_collect_realtest" / "by_pair",
    HERE / "gcp_round_b_small_sample_v1" / "minute_ext",
    HERE / "gcp_round_b_small_sample_v1" / "minute",
    HERE / "gcp_round_b_small_sample_v1" / "minute_car",
]

_CAL = None
_GRID = None


# ================= 网格 / 日历 =================
def load_minute_grid():
    global _GRID
    if _GRID is None:
        _GRID = json.loads(GRID_PATH.read_text(encoding="utf-8"))
    return _GRID


def day_grid(date: int) -> list:
    g = load_minute_grid()
    if int(date) < int(g.get("grid_1300_from", 20260624)):
        return list(g["grid_241"])
    return list(g["grid"])


def fill_slots(date: int) -> list:
    """该日期的成交槽 (HHMMSS, 升序), 排除 09:30/15:00 观测槽."""
    exp = day_grid(date)
    out = []
    for s in exp:
        hm = s // 100
        if (931 <= hm <= 1130) or (1301 <= hm <= 1456):
            out.append(s)
    return out


def load_calendar():
    global _CAL
    if _CAL is None:
        files = sorted(glob.glob(os.path.join(DAILY_PARTS, "part_*.parquet")))
        import pyarrow.dataset as ds
        tbl = ds.dataset(files, format="parquet").to_table(columns=["date"]).column("date").to_pylist()
        _CAL = sorted({int(d) for d in tbl})
    return _CAL


def next_trading_day(d):
    cal = load_calendar()
    later = [x for x in cal if x > int(d)]
    return later[0] if later else None


# ================= 数据加载 =================
def _sha256_file(p) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_daily(code: str):
    """读该 code 日线; 返回 (df, provenance[含路径+哈希的依赖 part])."""
    code = str(code).zfill(6)
    parts = sorted(glob.glob(os.path.join(DAILY_PARTS, "part_*.parquet")))
    out, prov = [], []
    import pyarrow.parquet as pq
    import pyarrow.compute as pc
    import pyarrow as pa
    cols = ["date", "code", "open", "high", "low", "close", "volume",
            "pre_close", "amount", "is_st"]
    for f in parts:
        pf = Path(f)
        try:
            t = pq.ParquetFile(pf).read(columns=cols)
        except Exception:
            continue
        sub = t.filter(pc.equal(t.column("code"), pa.scalar(code)))
        if sub.num_rows:
            out.append(sub.to_pandas())
            prov.append({"path": str(pf), "sha256": _sha256_file(pf)})
    if not out:
        return pd.DataFrame(), prov
    df = pd.concat(out, ignore_index=True).drop_duplicates("date").sort_values("date")
    return df, prov


def _merge_minute(by_day, df, prefer_day=None):
    df = df.copy()
    df["date"] = df["date"].astype(int)
    df["day"] = df["date"] // 1000000
    if prefer_day is not None:
        df = df[df["day"] == int(prefer_day)]
    for day, sub in df.groupby("day"):
        day = int(day)
        sub = sub.sort_values("date").reset_index(drop=True)
        by_day.setdefault(day, sub)  # 先到优先 (by_pair 目录排在 MIN_DIRS 首位)


def load_minute_code(code: str):
    """读该 code 全部分钟 (各日). 支持两种命名:
       A) {code}.parquet (整码切片, minute/minute_ext/minute_car)
       B) {code}_{YYYYMMDD}.parquet (by_pair, realtest 持仓窗口日)
    返回 (by_day{day:df}, provenance[路径+哈希+命名]). 记录每个依赖文件路径与哈希."""
    code = str(code).zfill(6)
    by_day, prov = {}, []
    for d in MIN_DIRS:
        d = Path(d)
        if not d.exists():
            continue
        f = d / f"{code}.parquet"
        if f.exists():
            try:
                df = pd.read_parquet(f)
            except Exception:
                df = None
            if df is not None and len(df):
                prov.append({"path": str(f), "sha256": _sha256_file(f), "naming": "code"})
                _merge_minute(by_day, df)
        for fp in sorted(d.glob(f"{code}_*.parquet")):
            try:
                day = int(fp.stem.split("_")[-1])
            except ValueError:
                continue
            try:
                df = pd.read_parquet(fp)
            except Exception:
                continue
            if df is None or not len(df):
                continue
            prov.append({"path": str(fp), "sha256": _sha256_file(fp),
                         "naming": "by_pair", "day": day})
            _merge_minute(by_day, df, prefer_day=day)
    return by_day, prov


# ================= 限价 (§7) =================
def limit_amplitude(board: str, is_st: bool, trade_date: int) -> float | None:
    if board in ("SH_MAIN", "SZ_MAIN"):
        if is_st:
            return 0.05 if int(trade_date) < ST_MAIN_NEW_RULE else 0.10
        return 0.10
    if board in ("GEM", "STAR"):
        return 0.20  # 含 ST
    return None  # OUT_OF_SCOPE / 未知板块


def _round2(x):
    return float(Decimal(str(x)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))


def compute_limit(prev_close: float, board: str, is_st: bool, trade_date: int, no_daily_limit: bool = False):
    if no_daily_limit:
        return None, None, "NO_DAILY_LIMIT", "IPO first 5 days / no daily limit evidence"
    amp = limit_amplitude(board, is_st, trade_date)
    if amp is None:
        return None, None, "OUT_OF_SCOPE", "board not in scope"
    upper = _round2(prev_close * (1 + amp))
    lower = _round2(prev_close * (1 - amp))
    if upper <= prev_close:
        upper = _round2(prev_close + TICK)
    if lower >= prev_close:
        lower = _round2(prev_close - TICK)
    if lower < TICK:
        lower = TICK
    return upper, lower, "OK", None


# ================= 入场 (§4) =================
def _board_qty(board: str, q_raw: float) -> int:
    if board in ("SH_MAIN", "SZ_MAIN", "GEM"):
        return 100 * int(q_raw // 100) if q_raw >= 100 else 0
    if board == "STAR":
        return int(q_raw) if q_raw >= 200 else 0
    return 0


def attempt_entry(minute_E: pd.DataFrame, signal_slot: int, board: str, is_st: bool,
                  prev_close_E: float, day: int) -> dict:
    """S 后下一合法 bar open 买入; 返回 entry dict 或含 status 的失败 dict."""
    slots = fill_slots(day)
    df = minute_E.sort_values("date").reset_index(drop=True)
    ts = df["date"].astype(int).to_numpy()
    o = df["open"].astype(float).to_numpy()
    h = df["high"].astype(float).to_numpy()
    lo = df["low"].astype(float).to_numpy()
    c = df["close"].astype(float).to_numpy()
    v = df["volume"].astype(float).to_numpy()
    slot_of = {int(t) % 1000000: i for i, t in enumerate(ts)}
    upper, lower, lstate, lreason = compute_limit(prev_close_E, board, is_st, day)
    reasons = []
    for s in slots:
        if s <= signal_slot:
            continue
        i = slot_of.get(s)
        if i is None:
            # §8/§4: 入场必要前缀或等待成交期间出现未知缺槽 -> 停止对应入场, 保持 NULL,
            # 不跳过缺槽后在更晚 bar 宣布成交 (不插值/不猜价/不默认停牌)
            reasons.append(f"{s} missing_slot_entry_window")
            return {"status": "ENTRY_DATA_MISSING",
                    "reason": f"entry window missing slot {s} (prefix continuity broken)",
                    "reasons": reasons, "limit_state": lstate}
        if v[i] <= 0:
            reasons.append(f"{s} zero_vol")
            continue
        op = o[i]
        # 限锁判定
        if upper is not None and abs(op - upper) < 1e-6:
            reasons.append(f"{s} open_at_limit_up")
            continue
        if abs(h[i] - lo[i]) < 1e-9 and abs(lo[i] - c[i]) < 1e-9 and upper is not None \
                and (abs(op - upper) < 1e-6 or (lower is not None and abs(op - lower) < 1e-6)):
            reasons.append(f"{s} locked_bar")
            continue
        q_cap = math.floor(PART_FRAC * v[i])
        q_cash = math.floor(ORDER_CAP / (op * BUY_MULT))
        q_raw = min(q_cap, q_cash)
        qty = _board_qty(board, q_raw)
        if qty <= 0:
            reasons.append(f"{s} zero_qty q_cap={q_cap} q_cash={q_cash}")
            continue
        price = float(op)
        amount = qty * price * BUY_MULT
        return {
            "sim_fill_ts": int(ts[i]), "sim_fill_price": price, "shares": int(qty),
            "amount": float(amount), "q_cap": int(q_cap), "q_cash": int(q_cash),
            "execution_assumption": "NEXT_LEGAL_BAR_OPEN; RESEARCH_APPROXIMATION",
            "limit_state": lstate, "limit_reason": lreason,
            "upper": upper, "lower": lower, "status": "SIM_FILLED",
        }
    return {"status": "NO_FILL_EOD", "reasons": reasons, "limit_state": lstate}


# ================= 轨道执行 (§5/§6) =================
def _try_sell(bar, pending_kind, remaining, board, prev_close, is_st, C_sell, sells):
    """尝试在 bar open 卖出 (限锁判定 + 1% 容量 + 板块数量规则). 返回 (C_sell, remaining, sold).
    is_st 取该交易日权威值 (Bug4: 不得硬编码 False)."""
    up, lo, _, _ = compute_limit(prev_close, board, is_st, bar["date"])
    op = bar["open"]
    blocked = False
    if up is not None and abs(op - lo) < 1e-9 and abs(lo - bar["close"]) < 1e-9 and \
            abs(bar["high"] - lo) < 1e-9:
        blocked = True
    elif up is not None and abs(op - up) < 1e-9:
        blocked = True
    if blocked or bar["volume"] <= 0:
        return C_sell, remaining, False
    q_cap = math.floor(PART_FRAC * bar["volume"])
    if board in ("SH_MAIN", "SZ_MAIN", "GEM"):
        sell_q = 100 * int(min(remaining, q_cap) // 100)
    else:
        # 科创: >=200 股 1 股递增; 不足 200 为零股, 持仓<200 时按零股卖出(不得膨胀到 200 超持仓)
        sell_q = int(min(remaining, q_cap))
        if 0 < sell_q < 200:
            sell_q = 200 if (q_cap >= 200 and remaining >= 200) else sell_q
    sell_q = min(sell_q, remaining)  # 防御: 卖出量不得超过当前持仓
    if sell_q <= 0:
        return C_sell, remaining, False
    C_sell += sell_q * op * SELL_MULT
    remaining -= sell_q
    sells.append({"date": bar["date"], "slot": bar["slot"], "price": op,
                  "shares": sell_q, "kind": pending_kind})
    return C_sell, remaining, True


def _day_fill_bars(minute_df):
    df = minute_df.sort_values("date").reset_index(drop=True)
    ts = df["date"].astype(int).to_numpy()
    o = df["open"].astype(float).to_numpy()
    h = df["high"].astype(float).to_numpy()
    lo = df["low"].astype(float).to_numpy()
    c = df["close"].astype(float).to_numpy()
    vol = df["volume"].astype(float).to_numpy()
    out = []
    for k in range(len(df)):
        s = int(ts[k]) % 1000000
        if (93100 <= s <= 113000) or (130100 <= s <= 145600):
            out.append({"date": int(ts[k]) // 1000000, "slot": s, "open": float(o[k]),
                        "high": float(h[k]), "low": float(lo[k]), "close": float(c[k]),
                        "volume": float(vol[k])})
    return out


def execute_track(entry: dict, stop_price: float, board: str, is_st_map: dict,
                  cal: list, minute_by_day: dict, plan_offset: int,
                  pre_close_map: dict, data_cutoff: int = DATA_CUTOFF) -> dict:
    """单轨道 (H3/H5): T+1 起逐日扫描止损/预定退出/顺延/部分卖出.

    关键: 一旦全部卖出即停止扫描 -> 缺口前已闭环的交易标签保持有效 (§8);
    仅当未退出时遇到缺失/非法日才判 EXIT_DATA_MISSING. 顺延上限 d0+5 交易日.
    """
    track = "H3" if plan_offset == 3 else "H5"
    E_day = entry["sim_fill_ts"] // 1000000
    ci = {d: i for i, d in enumerate(cal)}
    if E_day not in ci:
        return {"track": track, "status": "EXIT_DATA_MISSING", "reason": "E_NOT_IN_CAL",
                "sells": [], "net": None, "label_end_ts": None, "last_fill_ts": None,
                "observed_endpoint_ts": None, "missing_dependencies": [{"type": "entry_not_in_calendar", "date": E_day}]}
    ei = ci[E_day]
    plan_date = cal[ei + plan_offset] if ei + plan_offset < len(cal) else None
    remaining = entry["shares"]
    C_buy = entry["amount"]
    C_sell = 0.0
    sells = []
    pending_exit = False
    pending_kind = None
    d0 = None
    stop_triggered = False
    exited = False
    last_bar = None
    last_fill_ts = None
    close_ref = None
    if plan_date is not None and plan_date in minute_by_day:
        refs = minute_by_day[plan_date][minute_by_day[plan_date]["date"].astype(int) % 1000000 == 150000]
        if len(refs):
            close_ref = float(refs["close"].astype(float).iloc[0])

    def _cap_win():
        if d0 is not None and ci[d0] + DEFER_MAX_DAYS < len(cal):
            return cal[ci[d0] + DEFER_MAX_DAYS]
        return None

    i = ei + 1
    while i < len(cal) and cal[i] <= data_cutoff and remaining > 0:
        d = cal[i]
        # Fix1: 顺延上限在读取/成交前判定 (d0+5 含; 超出即拒, 绝不在超限日成交/绕过上限)
        if d0 is not None and ci[d] > ci[d0] + DEFER_MAX_DAYS:
            return _finalize_track(track, entry, stop_price, plan_date, d0, sells, remaining,
                                   C_buy, C_sell, "EXIT_DEFERRED_CAP", "exceed d0+5",
                                   last_bar, close_ref, last_fill_ts,
                                   [{"type": "deferral_cap", "d0": d0, "window_end": _cap_win()}])
        if d not in minute_by_day or minute_by_day[d] is None:
            return _finalize_track(track, entry, stop_price, plan_date, d0, sells, remaining,
                                   C_buy, C_sell, "EXIT_DATA_MISSING", "MISSING_MINUTE_DAY",
                                   last_bar, close_ref, last_fill_ts,
                                   [{"type": "minute_day_missing", "date": d}])
        # Fix3: 未知 ST 即隔离 (不得默认非 ST)
        is_st_d = is_st_map.get(d)
        if is_st_d is None:
            return _finalize_track(track, entry, stop_price, plan_date, d0, sells, remaining,
                                   C_buy, C_sell, "EXIT_DATA_MISSING", "MISSING_ST",
                                   last_bar, close_ref, last_fill_ts,
                                   [{"type": "is_st_unknown", "date": d}])
        if d not in pre_close_map or pre_close_map[d] is None:
            # 退出日限价依据必须是该日可见 pre_close; 未知即隔离
            return _finalize_track(track, entry, stop_price, plan_date, d0, sells, remaining,
                                   C_buy, C_sell, "EXIT_DATA_MISSING", "MISSING_PRECLOSE",
                                   last_bar, close_ref, last_fill_ts,
                                   [{"type": "pre_close_missing", "date": d}])
        prev_close_d = pre_close_map[d]
        v = E.validate_minute_day(minute_by_day[d], day_grid(d), d)
        if v["dups"] or v["unexpected"]:
            return _finalize_track(track, entry, stop_price, plan_date, d0, sells, remaining,
                                   C_buy, C_sell, "EXIT_DATA_MISSING",
                                   f"DAY_INVALID dups={v['dups']} unexpected={v['unexpected']}",
                                   last_bar, close_ref, last_fill_ts,
                                   [{"type": "day_invalid", "date": d,
                                     "detail": f"DAY_INVALID dups={v['dups']} unexpected={v['unexpected']}"}])
        # §8 缺槽判定 (新执行路径, 不沿用 validate_minute_day 的 missing<=2 容忍):
        #   必要扫描窗 = T+1 起至该轨道最后卖出 bar 完成; 窗内缺预期分钟 (含 1-2 槽) 即拒, 缺 1 根或 2 根同等;
        #   当日完成退出的, 完成 bar 之后的缺槽 (含 14:57-15:00 集合竞价/盘后, 不在本轮成交模型) 不否定已闭环交易;
        #   不插值/不猜缺失价/不用未来完整 bar 证明此前未触发.
        inwin_missing = [s for s in v["missing"] if s <= 145600]
        first_missing = inwin_missing[0] if inwin_missing else None
        # Fix2: 09:30 跳空止损观测 (open 低于止损即触发, 下一合法 bar open 成交)
        m = minute_by_day[d]
        def _obs(slot):
            r = m[m["date"].astype(int) % 1000000 == slot]
            return float(r["open"].astype(float).iloc[0]) if len(r) else None
        o930 = _obs(93000)
        if not stop_triggered and o930 is not None and o930 <= stop_price:
            stop_triggered = True
            pending_exit = True
            pending_kind = "STOP"
            if d0 is None:
                d0 = d
        day_bars = _day_fill_bars(minute_by_day[d])
        for bar in day_bars:
            d, s = bar["date"], bar["slot"]
            # 缺槽边界: 未闭环时本 bar 处于当日首个窗内缺槽或之后 -> 该缺槽在必要扫描窗内 -> NULL
            #          (缺槽前未闭环, 不能用完成 bar 之后的完整数据反证此前未触发止损)
            if remaining > 0 and first_missing is not None and s >= first_missing:
                return _finalize_track(track, entry, stop_price, plan_date, d0, sells, remaining,
                                       C_buy, C_sell, "EXIT_DATA_MISSING",
                                       f"DAY_GAP_IN_WINDOW first_missing={first_missing}",
                                       last_bar, close_ref, last_fill_ts,
                                       [{"type": "day_gap", "date": d,
                                         "detail": f"first_missing<={145600}: {first_missing}",
                                         "missing": inwin_missing}])
            # Fix2: 在 14:56 bar 起始建立预定退出指令 (等效 14:55 观测), 使其在本 bar open 成交,
            #        不取决于 14:56 当根 bar 是否破止损 (时间顺序正确, 旧逻辑会取消本应成交的预定卖单)
            if d == plan_date and s == 145600 and not pending_exit and not stop_triggered and not exited:
                pending_exit = True
                pending_kind = "TIME"
                if d0 is None:
                    d0 = d
            # 1) 上一观测/bar 触发的 pending 退出 -> 本 bar open 成交
            if pending_exit and remaining > 0:
                C_sell, remaining, did = _try_sell(bar, pending_kind, remaining, board,
                                                   prev_close_d, is_st_d, C_sell, sells)
                if did:
                    # bar["date"] 为交易日(8位), slot 为时分秒(6位) -> 完整 14 位时间戳
                    last_fill_ts = int(d) * 1000000 + int(s)
                if remaining == 0:
                    exited = True
                    pending_exit = False
            # 2) 本 bar 触发新指令 (已闭环则跳过); TIME 指令已于 14:56 bar 起始建立, 此处不再触发
            if remaining > 0 and not exited:
                if not stop_triggered and bar["low"] <= stop_price:
                    stop_triggered = True
                    pending_exit = True
                    pending_kind = "STOP"
                    if d0 is None:
                        d0 = d
            last_bar = (d, s)
        # 当日扫描完毕仍未闭环且存在窗内缺槽 -> 该缺槽在必要扫描窗内 (T+1 至最后卖出 bar) -> NULL
        if not exited and first_missing is not None:
            return _finalize_track(track, entry, stop_price, plan_date, d0, sells, remaining,
                                   C_buy, C_sell, "EXIT_DATA_MISSING",
                                   f"DAY_PARTIAL_GAP missing<={145600} n={len(inwin_missing)}",
                                   last_bar, close_ref, last_fill_ts,
                                   [{"type": "day_gap", "date": d,
                                     "detail": f"missing_slots<={145600}: {inwin_missing[:12]}",
                                     "missing": inwin_missing}])
        i += 1
    if exited:
        return _finalize_track(track, entry, stop_price, plan_date, d0, sells, remaining,
                               C_buy, C_sell, "SIM_COMPLETE", None, last_bar, close_ref, last_fill_ts, [])
    if d0 is not None:
        return _finalize_track(track, entry, stop_price, plan_date, d0, sells, remaining,
                               C_buy, C_sell, "EXIT_DEFERRED_CAP", "scan end residual",
                               last_bar, close_ref, last_fill_ts,
                               [{"type": "deferral_cap", "d0": d0, "window_end": _cap_win()}])
    return _finalize_track(track, entry, stop_price, plan_date, d0, sells, remaining,
                           C_buy, C_sell, "SIM_COMPLETE", "no exit triggered",
                           last_bar, close_ref, last_fill_ts, [])


def _finalize_track(track, entry, stop_price, plan_date, d0, sells, remaining, C_buy,
                    C_sell, status, reason, last_bar, close_ref, last_fill_ts=None,
                    missing_deps=None) -> dict:
    full = (remaining == 0)
    net = (C_sell / C_buy - 1.0) if full else None
    pnl = (C_sell - C_buy) if full else None
    # 已观察终点: 扫描到的最末 bar 完成时刻 (bar 完成时间口径 = ts + 1 分钟)
    observed_endpoint_ts = None
    if last_bar is not None:
        observed_endpoint_ts = int(last_bar[0]) * 1000000 + int(last_bar[1]) + 100
    # 有效标签终点: 仅完全退出且全部依赖通过时生成; 否则为 NULL (不允许带残仓进入训练)
    # FixB: 成交使用整根分钟量, 终点遵守 bar 完成时间口径 (last_fill_ts + 1 分钟)
    if full and last_fill_ts is not None:
        label_end_ts = last_fill_ts + 100
    else:
        label_end_ts = None
    return {
        "track": track, "status": status, "reason": reason,
        "sim_fill": {"ts": entry["sim_fill_ts"], "price": entry["sim_fill_price"],
                     "shares": entry["shares"], "amount": entry["amount"]},
        "stop_price": round(stop_price, 4),
        "planned_exit_date": plan_date,
        "deferral_d0": d0,
        "sells": sells, "remaining_shares": remaining,
        "C_buy": round(C_buy, 4), "C_sell": round(C_sell, 4),
        "pnl": (round(pnl, 4) if pnl is not None else None),
        "net": (round(net, 6) if net is not None else None),
        "close_ref_Hn": close_ref,
        "fully_exited": full,
        # FixB: 分别记录最后成交、已观察终点与缺失依赖; 有效标签终点仅完全退出时生成
        "last_fill_ts": last_fill_ts,
        "observed_endpoint_ts": observed_endpoint_ts,
        "missing_dependencies": (missing_deps if missing_deps is not None else []),
        "label_end_ts": label_end_ts,
    }


# ================= 事件编排 =================
def execute_event(code: str, E_day: int, board: str, is_st=None, setup: str = None,
                  signal_ts: int = None, minute_by_day: dict = None, daily_df=None,
                  cal: list = None, data_cutoff: int = DATA_CUTOFF) -> dict:
    """单事件执行 (新路径).

    - signal_ts: 冻结双路径信号 (manifest 的 S). 提供则严格使用, 绝不回退旧 detect_signal 重算
      (Bug2: 重算不一致必须报错, 不能换成旧信号继续执行).
    - is_st: 默认 None -> 从日线 is_st 列按日派生权威值 (Bug4: 不得硬编码 False).
    - 记录所有依赖文件 (路径+SHA256) 供审计.
    """
    cal = cal or load_calendar()
    if daily_df is None:
        daily_df, daily_prov = load_daily(code)
    else:
        daily_prov = []
    if minute_by_day is None:
        minute_by_day, minute_prov = load_minute_code(code)
    else:
        minute_prov = []
    out = {"code": code, "E": E_day, "board": board, "setup": setup,
           "is_st_source": "daily_derived" if is_st is None else "param"}
    # 日线派生: 各交易日 pre_close 与 is_st (权威, 用于限价与卖出)
    pre_close_map, is_st_map = {}, {}
    for _, row in daily_df.iterrows():
        dd = int(row["date"])
        pc = row.get("pre_close")
        pre_close_map[dd] = float(pc) if pd.notna(pc) else None
        # Fix3: 日线 is_st 缺失/未知 -> None (未知即隔离, 不得默认非 ST)
        st = row.get("is_st")
        is_st_map[dd] = bool(int(st)) if (pd.notna(st) and str(st) not in ("", "nan", "None")) else None
    # 信号
    upto_P = daily_df[daily_df["date"].astype(int) < E_day].sort_values("date")
    if upto_P.empty:
        return {**out, "status": "ENTRY_DATA_MISSING", "reason": "no daily before E",
                "dependencies": {"daily_parts": daily_prov, "minute_files": minute_prov}}
    atr_ref = E.build_atr_ref(upto_P)
    zones = E.build_zones(upto_P, atr_ref)
    ref_price = float(upto_P["close"].astype(float).iloc[-1])
    mE = minute_by_day.get(E_day)
    if mE is None:
        return {**out, "status": "ENTRY_DATA_MISSING", "reason": "no E-day minute",
                "dependencies": {"daily_parts": daily_prov, "minute_files": minute_prov}}
    if signal_ts is not None:
        out["signal_ts"] = int(signal_ts)
        out["signal_source"] = "frozen_manifest_S"
        out["signal_type"] = "FROZEN"
    else:
        # 仅隔离对照在无冻结 S 时回退旧 detect_signal; 普通事件必须传冻结 S
        sig = E.detect_signal(mE, zones, atr_ref, ref_price)
        if sig is None:
            return {**out, "status": "NO_SIGNAL",
                    "dependencies": {"daily_parts": daily_prov, "minute_files": minute_prov}}
        out["signal_ts"] = sig["signal_ts"]
        out["signal_type"] = sig.get("signal_type")
        out["signal_source"] = "detect_signal_fallback"
    out["atr_ref"] = round(float(atr_ref), 4) if np.isfinite(atr_ref) else None
    # 入场限价参考 (E 日 pre_close, 可见)
    prev_close_E = pre_close_map.get(E_day)
    if prev_close_E is None:
        prior = [v for k, v in pre_close_map.items() if k < E_day]
        prev_close_E = float(prior[-1]) if prior else ref_price
        out["limit_ref_note"] = "E-day pre_close missing; used prior close"
    # FixA: 入场日 is_st 未知 -> 返回 ENTRY_METADATA_UNKNOWN, 保持 NULL; 不得默认非 ST 后宣布成交
    # (此处为规则模拟成交, 无真实订单证明; 缺失 ST 影响限价与入场可行性, 必须隔离)
    def _null_track(t):
        return {"track": t, "status": "ENTRY_METADATA_UNKNOWN", "reason": "entry_metadata_unknown",
                "sim_fill": None, "stop_price": None, "planned_exit_date": None, "deferral_d0": None,
                "sells": [], "remaining_shares": None, "C_buy": None, "C_sell": None,
                "pnl": None, "net": None, "close_ref_Hn": None, "fully_exited": False,
                "last_fill_ts": None, "observed_endpoint_ts": None,
                "missing_dependencies": [{"type": "entry_metadata_unknown"}], "label_end_ts": None}
    if is_st is not None:
        is_st_E = bool(is_st)
    else:
        _e_st = is_st_map.get(E_day)
        if _e_st is None:
            out["dependencies"] = {"daily_parts": daily_prov, "minute_files": minute_prov}
            out["entry"] = {"status": "ENTRY_METADATA_UNKNOWN", "reason": "E-day is_st unknown",
                            "sim_fill_ts": None, "sim_fill_price": None, "shares": 0, "amount": None}
            out["H3"] = _null_track("H3")
            out["H5"] = _null_track("H5")
            out["status"] = "ENTRY_METADATA_UNKNOWN"
            return out
        is_st_E = _e_st
    ent = attempt_entry(mE, int(str(out["signal_ts"])[-6:]), board, is_st_E, prev_close_E, E_day)
    out["entry"] = ent
    if ent.get("status") != "SIM_FILLED":
        out["status"] = ent.get("status")
        out["dependencies"] = {"daily_parts": daily_prov, "minute_files": minute_prov}
        return out
    # 止损价
    D_stop = max(STOP_PCT_MIN, min(STOP_PCT_MAX, STOP_MULT * atr_ref / ent["sim_fill_price"]))
    stop_price = ent["sim_fill_price"] * (1 - D_stop)
    out["entry"]["D_stop"] = round(D_stop, 4)
    out["entry"]["stop_price"] = round(stop_price, 4)
    # 退出限价依据 = 各日后继日可见 pre_close (Bug3: 不得用当日 15:00 收盘价)
    h3 = execute_track(ent, stop_price, board, is_st_map, cal, minute_by_day, 3,
                       pre_close_map, data_cutoff)
    h5 = execute_track(ent, stop_price, board, is_st_map, cal, minute_by_day, 5,
                       pre_close_map, data_cutoff)
    out["H3"] = h3
    out["H5"] = h5
    out["status"] = "EXECUTED"
    out["dependencies"] = {"daily_parts": daily_prov, "minute_files": minute_prov}
    return out


if __name__ == "__main__":
    print("dual_path_execution_v0_1_1 module loaded; use execute_event(...)")
