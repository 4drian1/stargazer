"""外围行情采集器。

三个数据源各一个函数，统一入口 backfill(instrument_id, start, end)。
- 原始响应落盘 data/raw/{provider}/{instrument_id}_{date}.json（只追加，不覆盖）
- 解析后写 market_observation，available_at 按 visibility 规则计算
- quality_status 初值 RAW，按注册表 quality_rule 校验（±30% 标 SUSPECT），缺失值存 NULL
"""
import hashlib
import json
import os
from datetime import datetime
from pathlib import Path

import requests
import yaml

from .db import get_conn
from .visibility import BEIJING, compute_available_at

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "data" / "raw"
CONFIG_PATH = BASE_DIR / "config" / "instruments.yaml"

RATE_LIMIT_PCT = 0.30  # quality_rule 统一阈值：单日涨跌幅超 ±30% 标 SUSPECT


def load_registry():
    with open(CONFIG_PATH, encoding="utf-8") as f:
        return {i["instrument_id"]: i for i in yaml.safe_load(f)["instruments"]}


def _save_raw(provider, instrument_id, date_str, payload):
    """原始响应只追加落盘。"""
    d = RAW_DIR / provider
    d.mkdir(parents=True, exist_ok=True)
    path = d / f"{instrument_id}_{date_str}.json"
    if path.exists():
        return str(path)  # 已存在则不覆盖
    path.write_text(json.dumps(payload, ensure_ascii=False, default=str), encoding="utf-8")
    return str(path)


def _raw_hash(payload):
    return hashlib.sha256(json.dumps(payload, ensure_ascii=False, default=str).encode("utf-8")).hexdigest()


# ---------------- FRED ----------------

def fetch_fred(symbol, start, end):
    """FRED 日频序列（requests 直连，key 只从环境变量读）。"""
    key = os.environ.get("FRED_API_KEY")
    if not key:
        raise RuntimeError("缺少环境变量 FRED_API_KEY")
    url = "https://api.stlouisfed.org/fred/series/observations"
    params = {
        "series_id": symbol,
        "api_key": key,
        "file_type": "json",
        "observation_start": start,
        "observation_end": end,
    }
    resp = requests.get(url, params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    rows = []
    for ob in data.get("observations", []):
        date_str = ob["date"]
        raw_path = _save_raw("fred", symbol, date_str, ob)
        val = ob["value"]
        close = None if val in (".", "", None) else float(val)  # FRED 缺失值是 "."，存 NULL
        rows.append({
            "date": date_str, "open": None, "high": None, "low": None,
            "close": close, "volume": None,
            "raw_hash": _raw_hash(ob), "raw_path": raw_path,
        })
    return rows


# ---------------- yfinance ----------------

def fetch_yfinance(symbol, start, end):
    """yfinance 日线 OHLCV。日期按交易所时区换算北京时间存。"""
    import yfinance as yf

    df = yf.download(symbol, start=start, end=end, progress=False, auto_adjust=False)
    if df.empty:
        return []
    rows = []
    for idx, r in df.iterrows():
        date_str = idx.strftime("%Y-%m-%d")
        payload = {c: (None if v != v else float(v)) for c, v in r.items()}  # NaN -> None
        raw_path = _save_raw("yfinance", symbol, date_str, {"date": date_str, **payload})
        rows.append({
            "date": date_str,
            "open": payload.get("Open"), "high": payload.get("High"),
            "low": payload.get("Low"), "close": payload.get("Close"),
            "volume": payload.get("Volume"),
            "raw_hash": _raw_hash(payload), "raw_path": raw_path,
        })
    return rows


# ---------------- akshare（2026-09-11 实测接口，参数见 instruments.yaml ak_interface） ----------------

def _df_to_rows(df, date_col, ohlcv_map, provider, symbol):
    """akshare DataFrame 统一转 rows；date 列转 YYYY-MM-DD，数值缺失/NaN 存 None。"""
    import pandas as pd
    rows = []
    for _, r in df.iterrows():
        d = str(r.get(date_col))[:10].replace("/", "-")
        payload = {"date": d}
        out = {"date": d, "open": None, "high": None, "low": None,
               "close": None, "volume": None}
        for k, col in ohlcv_map.items():
            v = r.get(col)
            v = None if v is None or (isinstance(v, float) and v != v) else float(v)
            out[k] = v
            payload[col] = v
        raw_path = _save_raw(provider, symbol, d, payload)
        out["raw_hash"] = _raw_hash(payload)
        out["raw_path"] = raw_path
        rows.append(out)
    return rows


def fetch_akshare(instrument_id, symbol, start, end, ak_interface=None):
    """akshare 适配器。ak_interface 决定调哪个实测接口，不编造不存在的接口名。"""
    import akshare as ak
    if not symbol or symbol in ("TODO", "null"):
        raise NotImplementedError(
            f"{instrument_id}: akshare 接口参数待核实（见 instruments.yaml notes）")
    start_s, end_s = start.replace("-", ""), end.replace("-", "")

    if ak_interface == "stock_us_daily":
        df = ak.stock_us_daily(symbol=symbol)
        df = df[(df["date"].astype(str) >= start) & (df["date"].astype(str) <= end)]
        return _df_to_rows(df, "date", {"open": "open", "high": "high", "low": "low",
                                        "close": "close", "volume": "volume"},
                           "akshare_sina_us", symbol)
    if ak_interface == "stock_hk_index_daily_sina":
        df = ak.stock_hk_index_daily_sina(symbol=symbol)
        df = df[(df["date"].astype(str) >= start) & (df["date"].astype(str) <= end)]
        return _df_to_rows(df, "date", {"open": "open", "high": "high", "low": "low",
                                        "close": "close", "volume": "volume"},
                           "akshare_sina_hkidx", symbol)
    if ak_interface == "futures_foreign_hist":
        df = ak.futures_foreign_hist(symbol=symbol)
        df = df[(df["date"].astype(str) >= start) & (df["date"].astype(str) <= end)]
        return _df_to_rows(df, "date", {"open": "open", "high": "high", "low": "low",
                                        "close": "close", "volume": "volume"},
                           "akshare_sina_gfut", symbol)
    if ak_interface == "spot_hist_sge":
        df = ak.spot_hist_sge(symbol=symbol)
        df = df[(df["date"].astype(str) >= start) & (df["date"].astype(str) <= end)]
        return _df_to_rows(df, "date", {"open": "open", "high": "high", "low": "low",
                                        "close": "close"}, "akshare_sge", symbol)
    if ak_interface == "currency_boc_sina":
        df = ak.currency_boc_sina(symbol=symbol, start_date=start_s, end_date=end_s)
        # 牌价无 OHLC，用中行折算价作 close（口径见 instruments.yaml notes）
        df = df.rename(columns={"日期": "date", "中行折算价": "close"})
        df["date"] = df["date"].astype(str).str[:10]
        return _df_to_rows(df, "date", {"close": "close"}, "akshare_boc_fx", symbol)
    raise NotImplementedError(
        f"{instrument_id}: 未实现的 ak_interface={ak_interface}（可用的：stock_us_daily / "
        f"stock_hk_index_daily_sina / futures_foreign_hist / currency_boc_sina）")


# ---------------- 统一入口 ----------------

def _check_quality(prev_close, close):
    """quality_rule：单日涨跌幅超 ±30% 标 SUSPECT。"""
    if prev_close and close is not None:
        if abs(close / prev_close - 1) > RATE_LIMIT_PCT:
            return "SUSPECT"
    return "RAW"


def backfill(instrument_id, start, end, db_path=None):
    """回填 [start, end] 到 market_observation。返回写入条数。"""
    reg = load_registry()
    if instrument_id not in reg:
        raise KeyError(f"未注册序列: {instrument_id}")
    row = dict(reg[instrument_id], instrument_id=instrument_id)
    provider = row["provider"]
    symbol = row["provider_symbol"]

    if provider == "FRED":
        rows = fetch_fred(symbol, start, end)
    elif provider == "yfinance":
        rows = fetch_yfinance(symbol, start, end)
    elif provider == "akshare":
        rows = fetch_akshare(instrument_id, symbol, start, end, row.get("ak_interface"))
    elif provider in ("MISSING", None):
        raise NotImplementedError(
            f"{instrument_id}: 数据源缺失（见 instruments.yaml notes），如实报缺失，不冒充")
    else:
        raise NotImplementedError(f"provider 未实现: {provider}")

    conn = get_conn(db_path)
    n = 0
    prev_close = None
    try:
        for r in rows:
            if r["close"] is None:
                continue  # 无 close 的 bar 不入库（缺失存 NULL 不等于造数）
            event_time = datetime.fromisoformat(r["date"]).replace(tzinfo=BEIJING).isoformat()
            available_at = compute_available_at(event_time, row).isoformat()
            q = _check_quality(prev_close, r["close"])
            cur = conn.execute(
                """INSERT OR IGNORE INTO market_observation
                   (instrument_id, event_time, available_at, open, high, low, close, volume,
                    source, raw_hash, quality_status)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
                (instrument_id, event_time, available_at, r["open"], r["high"], r["low"],
                 r["close"], r["volume"], provider.lower(), r["raw_hash"], q))
            prev_close = r["close"]
            n += cur.rowcount  # 幂等：重复键跳过不计
        conn.commit()
    finally:
        conn.close()
    return n


def smoke_akshare():
    """akshare 冒烟：上证指数日线（stock_zh_index_daily 为长期存在的真实接口）。"""
    import akshare as ak
    return ak.stock_zh_index_daily(symbol="sh000001").tail(5)


if __name__ == "__main__":
    import sys
    if len(sys.argv) >= 4:
        print("写入条数:", backfill(sys.argv[1], sys.argv[2], sys.argv[3]))
    else:
        print(__doc__)
