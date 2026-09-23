"""事件后价格反应标签（price_labels.py，规格 minimal_event_price_loop_v1 第 4/5/6 节）。

口径要点：
- label_start 只由 available_at 决定（盘中→下一分钟；盘前→当日开盘；盘后/周末→下一交易日开盘）；
- 日线标签用开盘价/收盘价从 parquet 重算，可从原始数据整体重算（换 label_version 即可）；
- 30 分钟与反身性字段用 akshare 分钟线（仅抓事件窗口 ±2 交易日，缓存 parquet，失败留 NULL 不编造）；
- 标签一律是"事件后反应"，重叠事件写 event_overlap 并给 attribution_quality；
- 窗口未走完 → CENSORED_TAIL，字段 NULL，绝不填 0；
- 全部写入以 (event_id, entity_id, label_version) 幂等，重复运行不新增行。

用法：python -m corvus.price_labels
"""
import hashlib
import json
import re
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from .db import get_conn as get_sqlite
from .event_loop import DATA_PARTS
from .pgstore import get_pg_conn, init_pg_schema

LABEL_VERSION = "event_loop_v1"
MINUTE_CACHE = Path(__file__).resolve().parent.parent / "data" / "minute_cache"
BENCH_CACHE = Path(__file__).resolve().parent.parent / "data" / "benchmark_000300.parquet"
BENCH_ID = "000300"          # 沪深300（v1 统一大盘基准；板块基准暂缺，留 NULL 不冒充）
HORIZONS = (1, 3, 5)
STOCK_PRICE_VERSION = "daily_parts_27parquet"

_cal_cache = None
_price_cache = {}


# ---------------------------------------------------------------- 数据装载
def trading_calendar(bench=None):
    """交易日历 = 日线 parquet 日期 ∪ 沪深300指数日期。
    个股数据有更新延迟（增量构建滞后），指数（新浪）更新最快——
    日历必须以交易所级数据为准，否则事件会被误判成'无未来交易日'。"""
    global _cal_cache
    if _cal_cache is None:
        dates = set()
        for f in sorted(DATA_PARTS.glob("part_*.parquet")):
            df = pd.read_parquet(f, columns=["date"])
            dates.update(int(x) for x in df["date"].unique())
        if bench is not None:
            dates.update(int(x) for x in bench["date"].unique())
        _cal_cache = sorted(dates)
    return _cal_cache


def price_data_max_date():
    return trading_calendar()[-1]


def load_stock_daily(code):
    """从 27 个 parquet 分片按 code 过滤拼全日线（前复权口径由数据集保证）。"""
    if code in _price_cache:
        return _price_cache[code]
    frames = []
    for f in sorted(DATA_PARTS.glob("part_*.parquet")):
        df = pd.read_parquet(f, filters=[("code", "=", code)])
        if len(df):
            frames.append(df)
    df = (pd.concat(frames, ignore_index=True).sort_values("date")
          if frames else pd.DataFrame())
    _price_cache[code] = df
    return df


def load_benchmark():
    """沪深300 日线。东财指数接口（push2）在本网络被断连，用新浪 stock_zh_index_daily
    （2002 起全历史）。缓存 parquet，缓存最新日期落后太多时重捞。"""
    if BENCH_CACHE.exists():
        df = pd.read_parquet(BENCH_CACHE)
        if int(df["date"].max()) >= 20260901:
            return df
    import akshare as ak
    df = ak.stock_zh_index_daily(symbol="sh000300")
    df["date"] = df["date"].astype(str).str.replace("-", "").astype(int)
    df = df[["date", "open", "close", "high", "low"]].sort_values("date")
    BENCH_CACHE.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(BENCH_CACHE, index=False)
    return df


def load_minute(code, day):
    """事件日 ±2 交易日的 1 分钟线。东财分钟接口（80.push2）在本网络被断连，
    改用新浪 stock_zh_a_minute（仅近 ~8 个交易日，历史事件拿不到就诚实返回 None）。
    缓存 parquet，失败不编造。"""
    day_s = str(day)
    cache = MINUTE_CACHE / f"{code}_{day_s}.parquet"
    if cache.exists():
        return pd.read_parquet(cache)
    if code.startswith(("4", "8", "92")):   # 北交所新浪分钟不支持
        return None
    prefix = "sh" if code.startswith("6") else "sz"
    try:
        import akshare as ak
        df = ak.stock_zh_a_minute(symbol=f"{prefix}{code}", period="1", adjust="")
        if df is None or df.empty:
            return None
        df["dt"] = pd.to_datetime(df["day"])
        end = pd.Timestamp(f"{_shift_day(day, +2)} 15:30")
        df = df[(df["dt"] >= pd.Timestamp(f"{day_s} 09:00")) & (df["dt"] <= end)]
        if df.empty:
            return None
        for c in ("open", "close", "high", "low", "volume"):
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df = df.dropna(subset=["open", "close"])
        if df.empty:
            return None
        MINUTE_CACHE.mkdir(parents=True, exist_ok=True)
        df[["dt", "open", "close", "high", "low", "volume"]].to_parquet(cache, index=False)
        return df[["dt", "open", "close", "high", "low", "volume"]].reset_index(drop=True)
    except Exception:
        return None


def _shift_day(day, n):
    cal = trading_calendar()
    i = cal.index(int(day)) if int(day) in cal else None
    if i is None:
        return str(int(day) + n)
    return str(cal[min(max(i + n, 0), len(cal) - 1)])


# ---------------------------------------------------------------- 标签起点
def resolve_label_start(available_at, cal):
    """available_at（ISO 字符串）→ (label_start_dt, basis)。规格第 4 节。"""
    dt = pd.Timestamp(str(available_at))
    if dt.tzinfo is not None:                      # PG timestamptz 带时区，分钟线为 naive
        dt = dt.tz_convert("Asia/Shanghai").tz_localize(None)
    day = int(dt.strftime("%Y%m%d"))
    t = dt.time()
    if day in cal:
        if t <= pd.Timestamp("09:30").time():
            return dt.replace(hour=9, minute=30), "PREMARKET_OPEN"
        if t < pd.Timestamp("15:00").time():
            nxt = (dt + timedelta(minutes=1)).replace(second=0, microsecond=0)
            return nxt, "INTRADAY_NEXT_MINUTE"
        # 收盘后 → 下一交易日（若中间隔周末/节假日，basis 记 WEEKEND_NEXT_DAY）
        i = cal.index(day)
        nxt_day = cal[i + 1] if i + 1 < len(cal) else None
        if nxt_day:
            from datetime import date as _date
            gap_days = (_date(nxt_day // 10000, nxt_day // 100 % 100, nxt_day % 100)
                        - _date(day // 10000, day // 100 % 100, day % 100)).days
            basis = "WEEKEND_NEXT_DAY" if gap_days > 1 else "POST_CLOSE_NEXT_DAY"
            return pd.Timestamp(str(nxt_day)).replace(hour=9, minute=30), basis
        return None, "NO_FUTURE_TRADE_DAY"
    # 非交易日（周末/节假日）→ 下一交易日开盘
    future = [d for d in cal if d > day]
    if not future:
        return None, "NO_FUTURE_TRADE_DAY"
    return pd.Timestamp(str(future[0])).replace(hour=9, minute=30), "WEEKEND_NEXT_DAY"


# ---------------------------------------------------------------- 单事件标签
def _ret(close_path, entry):
    return float(close_path / entry - 1) if entry else None


def compute_event_labels(pg, eid, code, available_at, bench, cal):
    """返回 dict：日线 + 分钟线标签（可算才算，None 不编造）。"""
    start_dt, basis = resolve_label_start(available_at, cal)
    if start_dt is None:
        return {"label_start": None, "label_start_basis": basis,
                "label_status": "CENSORED_TAIL", "horizon_complete": False}
    entry_day = int(start_dt.strftime("%Y%m%d"))
    df = load_stock_daily(code)
    if df.empty or entry_day not in set(df["date"]):
        # 区分"真停牌"（数据期内无 bar）与"数据滞后"（bar 尚未入库）
        lag = df.empty or entry_day > int(df["date"].max())
        out = {"label_start": start_dt.isoformat(), "label_start_basis": basis,
               "suspended": None if lag else True,   # 滞后不下结论，绝不冒充停牌
               "label_status": "CENSORED_TAIL",
               "horizon_complete": False}
        # 日线滞后但分钟线（新浪，近8日）可得 → 30m 与反身性字段照常算
        minute = load_minute(code, entry_day)
        if minute is not None and len(minute):
            _minute_fields(out, minute, start_dt, float(
                minute[minute["dt"] >= start_dt].iloc[0]["open"])
                if len(minute[minute["dt"] >= start_dt]) else None, df)
        return out
    df = df.reset_index(drop=True)
    idx = int(df.index[df["date"] == entry_day][0])
    row = df.loc[idx]

    # 入场价：盘前/盘后/周末起点 = 当日开盘；盘中起点优先用分钟线成交价
    entry = float(row["open"])
    minute = None
    if basis == "INTRADAY_NEXT_MINUTE":
        minute = load_minute(code, entry_day)
        if minute is not None:
            after = minute[minute["dt"] >= start_dt]
            if len(after):
                entry = float(after.iloc[0]["open"])

    out = {"label_start": start_dt.isoformat(), "label_start_basis": basis,
           "entry_price": entry, "benchmark_market": BENCH_ID,
           "gap_return": float(row["open"] / row["pre_close"] - 1)
           if basis in ("PREMARKET_OPEN", "WEEKEND_NEXT_DAY") and row.get("pre_close") else None,
           "limit_locked": _is_limit_locked(code, row),
           "suspended": False}

    # 日线收益与路径标签
    last_i = len(df) - 1
    close_entry = float(row["close"])
    out["ret_to_close"] = _ret(close_entry, entry)
    bdf = bench.set_index("date")
    b_entry_row = bdf.loc[entry_day] if entry_day in bdf.index else None
    b_entry = float(b_entry_row["open"]) if b_entry_row is not None else None

    # 除权守卫：开盘/昨收缺口超过涨跌停带而 pct_chg 温和 → 除权断尺度日
    # （实例：600019 20260916 送转，open 5.78 vs 前收 16.72，pct_chg +0.35%）
    corp_days = set()
    for k in range(idx + 1, last_i + 1):
        rk = df.loc[k]
        pc_, oc_ = rk.get("pre_close"), float(rk["open"])
        pch = rk.get("pct_chg")
        if pc_ and abs(oc_ / float(pc_) - 1) > 0.115 and pch is not None                 and not (isinstance(pch, float) and pd.isna(pch)) and abs(float(pch)) < 9.5:
            corp_days.add(k)

    def daily_factor(k):
        """第 k 日（相对前收）的调整感知日收益因子；除权断尺度日退化为 0 收益占位。"""
        rk = df.loc[k]
        pch = rk.get("pct_chg")
        if pch is not None and not (isinstance(pch, float) and pd.isna(pch)):
            return 1.0 + float(pch) / 100.0
        if k in corp_days:
            return 1.0
        prev = df.loc[k - 1]
        return float(rk["close"]) / float(prev["close"])

    complete = True
    chain = 1.0
    for h in HORIZONS:
        j = idx + h
        if j > last_i:
            complete = False
            continue                      # 缺数据 → 留 NULL（CENSORED_TAIL 判定看 complete）
        for k in range(idx + h - HORIZONS[HORIZONS.index(h) - 1] if h > HORIZONS[0] else idx + 1,
                       j + 1):
            if k > idx + (HORIZONS[HORIZONS.index(h) - 1] if h > HORIZONS[0] else 0):
                pass
        r = df.loc[j]
        # 日收益链式复权：跨除权日用 pct_chg 链，杜绝尺度断裂造假收益
        chain = 1.0
        for k in range(idx + 1, j + 1):
            chain *= daily_factor(k)
        out[f"ret_{h}d"] = float((float(df.loc[idx]["close"]) / entry) * chain - 1)
        if any(k in corp_days for k in range(idx, j + 1)):
            out["corp_action_during_horizon"] = True   # 窗口含除权日：mfe/mae 不可比
        else:
            seg = df.loc[idx:j]
            out[f"mfe_{h}d"] = float(seg["high"].max() / entry - 1)
            out[f"mae_{h}d"] = float(seg["low"].min() / entry - 1)
        if b_entry is not None and int(r["date"]) in bdf.index:
            b_ret = float(bdf.loc[int(r["date"])]["close"] / b_entry - 1)
            out[f"abnormal_ret_market_{h}d"] = out[f"ret_{h}d"] - b_ret
    out["horizon_complete"] = complete
    out["label_status"] = "OK" if complete else "CENSORED_TAIL"

    # 量能/波动结构
    prev = df.loc[max(0, idx - 20):idx - 1]
    if len(prev) >= 5:
        out["volume_shock"] = float(row["volume"] / prev["volume"].mean()) \
            if prev["volume"].mean() else None
        if len(prev) >= 10:
            base_vol = prev["close"].pct_change().std()
            seg = df.loc[idx:min(idx + 5, last_i)]
            new_vol = seg["close"].pct_change().std()
            out["volatility_expansion"] = float(new_vol / base_vol) if base_vol else None

    # 分钟线字段（30m + 反身性现象；拿不到就 NULL）
    if minute is None:
        minute = load_minute(code, entry_day)
    if minute is not None and len(minute):
        _minute_fields(out, minute, start_dt, entry, df, idx)
    return out


def _minute_fields(out, minute, start_dt, entry, daily, idx=None):
    """30 分钟与反身性字段（从分钟线算；entry 为 None 时用起点后第一分钟开盘价）。"""
    after = minute[minute["dt"] >= start_dt]
    if len(after) < 2:
        return
    p0 = entry if entry else float(after.iloc[0]["open"])
    m30 = after.iloc[min(29, len(after) - 1)]
    out["ret_30m"] = float(m30["close"] / p0 - 1)
    seg30 = after.iloc[:30]
    out["mfe_30m"] = float(seg30["high"].max() / p0 - 1)
    out["mae_30m"] = float(seg30["low"].min() / p0 - 1)
    out["initial_direction"] = ("UP" if after.iloc[1]["close"] > p0 * 1.001
                                else "DOWN" if after.iloc[1]["close"] < p0 * 0.999
                                else "FLAT")
    out["initial_shock_atr"] = _shock_atr(after, daily, idx) if idx is not None else None
    out["reversal_30m"] = _reversal(seg30, p0)
    out["reclaim_pre_event_price"] = bool(after["close"].max() >= p0) \
        if out.get("initial_direction") == "DOWN" else None


def _is_limit_locked(code, row):
    """近似涨跌停封板：涨幅触板且振幅极小。v1 近似口径，只作标签不作交易依据。"""
    pct = row.get("pct_chg")
    if pct is None or (isinstance(pct, float) and np.isnan(pct)):
        return None
    limit = 19.8 if code.startswith(("300", "301", "688")) else 9.8
    return bool(abs(float(pct)) >= limit and float(row.get("amplitude") or 9) < 1.0)


def _shock_atr(after, daily, idx):
    """初始冲击幅度（ATR 倍数）：前 30 分钟最大不利偏离 / 前 20 日 ATR。"""
    tr = []
    for i in range(max(1, idx - 19), idx + 1):
        r = daily.loc[i]
        pc = daily.loc[i - 1]["close"] if i > 0 else r["open"]
        tr.append(max(r["high"] - r["low"], abs(r["high"] - pc), abs(r["low"] - pc)))
    atr = float(np.mean(tr)) if tr else None
    if not atr:
        return None
    p0 = float(after.iloc[0]["open"])
    depth = abs(float(after.iloc[:30]["low"].min()) - p0)
    return float(depth / atr)


def _reversal(seg, p0):
    """反转强度：30 分钟内自极值点的回拉幅度（占初始冲击的比例）。"""
    low_i = seg["low"].idxmin()
    low_v = float(seg.loc[low_i, "low"])
    shock = p0 - low_v
    if shock <= 0:
        return None
    tail = seg.loc[low_i:]
    return float((tail["high"].max() - low_v) / shock)


# ---------------------------------------------------------------- 重叠
def write_overlaps(pg, cal):
    """同一 DIRECT_STOCK、标签窗口（±5 交易日）内多事件 → event_overlap。"""
    rows = pg.execute(
        """SELECT l1.event_id, l2.event_id, l1.entity_id,
                  e1.available_at::date AS d1, e2.available_at::date AS d2
           FROM event_entity_link l1
           JOIN event_entity_link l2 ON l1.entity_id = l2.entity_id
                                    AND l1.event_id < l2.event_id
           JOIN information_event e1 ON e1.event_id = l1.event_id
           JOIN information_event e2 ON e2.event_id = l2.event_id
           WHERE l1.entity_type = 'DIRECT_STOCK'""").fetchall()
    pos = {d: i for i, d in enumerate(cal)}
    from collections import defaultdict
    by_pair = defaultdict(list)
    for a, b, ent, d1, d2 in rows:
        i1, i2 = pos.get(int(str(d1).replace("-", ""))), pos.get(int(str(d2).replace("-", "")))
        if i1 is None or i2 is None or abs(i1 - i2) > 5:
            continue
        by_pair[(a, ent)].append(b)
    n = 0
    # CLEAN 事件也留一行自记录（other_event_id=自身，count=0），保证每事件 overlap 字段可审计
    all_pairs = pg.execute(
        """SELECT DISTINCT l.event_id, l.entity_id FROM event_entity_link l
           WHERE l.entity_type='DIRECT_STOCK'""").fetchall()
    have = {(a, ent) for (a, ent) in by_pair}
    for a, ent in all_pairs:
        if (a, ent) not in have:
            pg.execute(
                """INSERT INTO event_overlap
                   (event_id, other_event_id, entity_id, overlap_window,
                    overlap_event_count, overlap_event_ids, macro_overlap,
                    sector_overlap, attribution_quality, label_version)
                   VALUES (%s,%s,%s,%s,0,'[]',FALSE,FALSE,'CLEAN',%s)
                   ON CONFLICT DO NOTHING""",
                (a, a, ent, "±5交易日", LABEL_VERSION))
            n += 1
    for (a, ent), others in by_pair.items():
        others = sorted(set(others))
        heavy = len(others) >= 3
        q = "HEAVY" if heavy else ("MODERATE" if others else "CLEAN")
        for b in others:
            pg.execute(
                """INSERT INTO event_overlap
                   (event_id, other_event_id, entity_id, overlap_window,
                    overlap_event_count, overlap_event_ids, macro_overlap,
                    sector_overlap, attribution_quality, label_version)
                   VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s) ON CONFLICT DO NOTHING""",
                (a, b, ent, "±5交易日", len(others), json.dumps(others),
                 False, False, q, LABEL_VERSION))
            n += 1
    pg.commit()
    return n


# ---------------------------------------------------------------- 主流程
def run_labels(verbose=True):
    pg = get_pg_conn()
    sl = get_sqlite()
    try:
        init_pg_schema(pg)
        run_id = pg.execute(
            """INSERT INTO event_label_run (label_version, params, price_data_version, code_version)
               VALUES (%s,%s,%s,%s) RETURNING run_id""",
            (LABEL_VERSION,
             json.dumps({"horizons": list(HORIZONS), "bench": BENCH_ID}),
             STOCK_PRICE_VERSION, _code_version())).fetchone()[0]
        pg.commit()

        cal = trading_calendar(load_benchmark())
        bench = load_benchmark()
        links = pg.execute(
            """SELECT l.event_id, l.entity_id, e.available_at
               FROM event_entity_link l JOIN information_event e ON e.event_id=l.event_id
               WHERE l.entity_type='DIRECT_STOCK' AND l.mapping_method IN
                     ('RULE_CODE_OFFICIAL','RULE_NAME_MATCH')""").fetchall()

        # 窗口内是否有分红除权等公司行为（DISTRIBUTION 事件同股）
        corp = {r[0] for r in pg.execute(
            """SELECT DISTINCT l2.entity_id FROM event_entity_link l2
               JOIN information_event e2 ON e2.event_id=l2.event_id
               WHERE e2.canonical_event_type='DISTRIBUTION'
                 AND l2.entity_type='DIRECT_STOCK'""").fetchall()}

        stats = {"OK": 0, "CENSORED_TAIL": 0, "SUSPENDED": 0, "FAILED": 0}
        for eid, code, avail in links:
            try:
                out = compute_event_labels(pg, eid, code, avail, bench, cal)
                if not out.get("label_start"):
                    pg.rollback()
                    stats["FAILED"] += 1
                    if verbose:
                        print(f"  [SKIP] event={eid} {code}: 日历内无未来交易日", flush=True)
                    continue
                out["corp_action_during_horizon"] = code in corp
                _upsert_reaction(pg, eid, code, out)
                stats["SUSPENDED" if out.get("suspended") else out["label_status"]] += 1
            except Exception as exc:
                pg.rollback()          # 单条失败不毒化整批事务
                stats["FAILED"] += 1
                if verbose:
                    print(f"  [FAIL] event={eid} {code}: {type(exc).__name__} {exc}", flush=True)
        n_ov = write_overlaps(pg, cal)
        pg.execute(
            "UPDATE event_label_run SET finished_at=now(), status='SUCCESS', stats=%s WHERE run_id=%s",
            (json.dumps({"reactions": stats, "overlaps": n_ov,
                         "max_cal_date": str(cal[-1])}), run_id))
        pg.commit()
        if verbose:
            print(f"标签：{stats}；重叠记录 {n_ov}；日历最大交易日 {cal[-1]}")
        return stats
    finally:
        sl.close()
        pg.close()


def _upsert_reaction(pg, eid, code, out):
    cols = ["ret_30m", "ret_to_close", "ret_1d", "ret_3d", "ret_5d",
            "abnormal_ret_market_30m", "abnormal_ret_market_1d",
            "abnormal_ret_market_3d", "abnormal_ret_market_5d",
            "mfe_30m", "mfe_1d", "mfe_3d", "mfe_5d",
            "mae_30m", "mae_1d", "mae_3d", "mae_5d",
            "gap_return", "volatility_expansion", "volume_shock",
            "reversal_30m", "reversal_1d", "initial_direction",
            "initial_shock_atr", "liquidity_sweep_depth", "reclaim_vwap",
            "reclaim_pre_event_price", "reversal_strength", "reversal_speed",
            "post_reversal_ret_3d", "post_reversal_ret_5d",
            "limit_locked", "suspended", "corp_action_during_horizon"]
    keys = ["event_id", "entity_type", "entity_id", "label_start", "label_start_basis",
            "entry_price", "benchmark_market", "horizon_complete", "label_status",
            "label_version", "price_data_version", "source_hash", "code_version"] + cols
    vals = [eid, "DIRECT_STOCK", code, out.get("label_start"),
            out.get("label_start_basis"), out.get("entry_price"),
            out.get("benchmark_market"), out.get("horizon_complete", False),
            out.get("label_status", "CENSORED_TAIL"), LABEL_VERSION,
            STOCK_PRICE_VERSION, None, _code_version()] + [out.get(c) for c in cols]
    sets = ", ".join(f"{k} = EXCLUDED.{k}" for k in keys[3:])
    pg.execute(
        f"""INSERT INTO event_price_reaction ({', '.join(keys)})
            VALUES ({', '.join(['%s'] * len(keys))})
            ON CONFLICT (event_id, entity_id, label_version)
            DO UPDATE SET {sets}, generated_at = now()""", vals)


def _code_version():
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:12]


def main(argv=None):
    run_labels()


if __name__ == "__main__":
    main()
