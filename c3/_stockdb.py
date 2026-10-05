"""stockdb HTTP 最小客户端（c3_pack 自包含，不依赖 corvus/PG）。

实测结论（2026-09-27，本机 stockdb 127.0.0.1:7899）：
- GET /?cmd=get&t=<table>:<code>:<date>，Accept: application/json；
  返回 JSON 或 msgpack（先尝试 JSON 解码，失败按 msgpack 解）。
- 单条：t=日k:600519:20260916 -> dict。
- 全历史：t=日k:600519:* -> [[key, dict], ...]。
- 通配只支持"代码段尾glob"：日k:600* 有效；日k:*:20260916 与 日k:*:* 返回空。
- GET /?cmd=keys&t=日k:600* -> 全部 key 字符串列表（量级大，需流式解析）。
- 日k 字段（实测）：amount, amplitude, close, code, date, float_mv, float_share,
  high, is_st, low, name, open, pb, pct_chg, pe_ttm, pre_close, total_mv,
  total_share, turnover, vol_ratio, volume。
- ProxyHandler({}) 绕过系统代理（监控机环境必需）。
"""
from __future__ import annotations

import json
import os
import urllib.parse
import urllib.request

import msgpack

DEFAULT_BASE = os.environ.get("STOCKDB_BASE", "http://127.0.0.1:7899")


def make_opener():
    return urllib.request.build_opener(urllib.request.ProxyHandler({}))


def _decode(raw: bytes):
    try:
        return json.loads(raw.decode("utf-8", errors="replace"))
    except Exception:
        return msgpack.unpackb(raw, raw=False, strict_map_key=False)


def http_get_raw(opener, base: str, table: str, code: str = "*", date: str = "*",
                 timeout: int = 120):
    t = urllib.parse.quote(f"{table}:{code}:{date}")
    url = f"{base}/?cmd=get&t={t}"
    req = urllib.request.Request(url, headers={"Accept": "application/json"})
    with opener.open(req, timeout=timeout) as r:
        return _decode(r.read())


def recs_of(obj):
    """stockdb 列表响应 -> [dict, ...]（兼容 [key, rec] 与纯 dict 列表）。"""
    if isinstance(obj, list):
        out = [it[1] for it in obj
               if isinstance(it, (list, tuple)) and len(it) >= 2 and isinstance(it[1], dict)]
        if not out and obj and isinstance(obj[0], dict):
            out = obj
        return out
    if isinstance(obj, dict):
        return [obj]
    return []


def iter_keys(opener, base: str, pattern: str, timeout: int = 300):
    """流式迭代 cmd=keys 响应（msgpack Unpacker 增量解析，内存不随 key 数膨胀）。"""
    url = f"{base}/?cmd=keys&t={urllib.parse.quote(pattern)}"
    req = urllib.request.Request(url, headers={"Accept": "application/json"})
    with opener.open(req, timeout=timeout) as r:
        unpacker = msgpack.Unpacker(r, raw=False, strict_map_key=False)
        for item in unpacker:
            if isinstance(item, (list, tuple)):
                for k in item:
                    yield k
            elif isinstance(item, str):
                yield item


def universe_via_keys(base: str = DEFAULT_BASE, prefixes: tuple = ("0", "3", "6", "4", "8", "9")):
    """经 HTTP keys 流式取全 A 代码（stockdb 日k 表覆盖 2000 起含退市）。"""
    opener = make_opener()
    codes = set()
    for p in prefixes:
        for k in iter_keys(opener, base, f"日k:{p}*"):
            parts = k.split(":")
            if len(parts) == 3:
                codes.add(parts[1])
    return sorted(codes)


def universe_via_parts(stockdb_root: str):
    """降级路径：读 {STOCKDB_ROOT}/data_daily_parts/part_*.parquet 的 code 列。"""
    import glob
    import pandas as pd
    codes = set()
    for p in sorted(glob.glob(os.path.join(stockdb_root, "data_daily_parts", "part_*.parquet"))):
        df = pd.read_parquet(p, columns=["code"])
        for c in df["code"].astype(str):
            codes.add(c.zfill(6))
    return sorted(codes)


def load_daily(opener, base: str, code: str, timeout: int = 120):
    """单票日k全历史 -> DataFrame（按 date 升序）。"""
    import pandas as pd
    recs = recs_of(http_get_raw(opener, base, "日k", code, "*", timeout=timeout))
    if not recs:
        return pd.DataFrame()
    df = pd.DataFrame(recs)
    if "date" in df.columns:
        df = df.sort_values("date").reset_index(drop=True)
    return df
