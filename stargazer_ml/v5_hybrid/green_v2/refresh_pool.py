# -*- coding: utf-8 -*-
"""盘后链路前置：探测 stockdb 最新交易日 -> 把缺失日期增量写入 data_daily_parts
-> 重跑 build_pool 增量。被 plan_bridge「盘后选股」在 model_shadow_draft 之前调用。

用法: _dev_venv\python -X utf8 refresh_pool.py
环境: stockdb 引擎须在运行（127.0.0.1:7899）。
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

BASE = "http://127.0.0.1:7899"


def _parts_dir() -> Path:
    """日线目录：env STARGAZER_DAILY_PARTS（mquote/build_pool 同名先例）→ 仓库根
    data_daily_parts（监控机布局）→ 本机 D:/a股数据 缺省。"""
    import os
    v = os.environ.get("STARGAZER_DAILY_PARTS")
    if v:
        return Path(v)
    here = Path(__file__).resolve().parent          # green_v2/
    root = here.parents[2]                           # 仓库根
    for cand in (root / "data_daily_parts", Path(r"D:/a股数据/data_daily_parts")):
        if cand.exists():
            return cand
    return Path(r"D:/a股数据/data_daily_parts")


PARTS = _parts_dir()
PROBE_CODE = "600000"
OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))

import time as _t

FIELDS = ["date", "code", "name", "open", "high", "low", "close", "pre_close",
          "volume", "amount", "turnover", "pct_chg", "amplitude", "is_st",
          "vol_ratio", "total_share", "float_share", "total_mv", "float_mv",
          "pe_ttm", "pb"]


def get(t, timeout=30):
    import msgpack
    url = BASE + "/?cmd=get&t=" + urllib.parse.quote(t)
    req = urllib.request.Request(url, headers={"Accept": "application/x-msgpack"})
    raw = OPENER.open(req, timeout=timeout).read()
    d = msgpack.unpackb(raw, raw=False, strict_map_key=False)
    return {"data": d} if isinstance(d, list) else d


def latest_stockdb_date():
    d = get(f"日k:{PROBE_CODE}:*")
    rows = d.get("data") if isinstance(d, dict) else d
    if not rows:
        raise SystemExit("stockdb 无 600000 日线数据（引擎未跑数据更新？）")
    return max(int(r[1]["date"]) for r in rows)


def latest_parts_date():
    best = 0
    for f in glob.glob(str(PARTS / "part_*.parquet")):
        try:
            import pyarrow.parquet as pq
            d = pq.read_table(f, columns=["date"]).to_pandas()["date"].max()
            best = max(best, int(d))
        except Exception:
            continue
    return best


def fetch_day(day: str):
    import pandas as pd
    out = PARTS / f"part_incr_{day}.parquet"
    if out.exists():
        print(f"[skip] {out.name} 已存在")
        return
    uni = get("股票代码")
    codes = []
    for v in uni.values():
        if isinstance(v, list):
            codes.extend(str(x) for x in v)
    codes = list(dict.fromkeys(codes))

    def one(code):
        for attempt in range(3):
            try:
                d = get(f"日k:{code}:{day}*")
                rows = d.get("data") if isinstance(d, dict) else d
                if not rows:
                    return []
                records = [rec for _k, rec in rows]
                if any(int(rec['date']) != int(day) or str(rec['code']).zfill(6) != code
                       for rec in records):
                    raise ValueError('返回的股票/日期与请求不符')
                return [[rec.get(f) for f in FIELDS] for rec in records]
            except Exception as exc:
                if attempt < 2:
                    _t.sleep(2)
                else:
                    raise RuntimeError(f'{code}/{day} 请求失败，停止同步，不写部分分片：{exc}') from exc

    all_rows = []
    with ThreadPoolExecutor(max_workers=8) as ex:
        for rows in ex.map(one, codes):
            all_rows.extend(rows)
    df = pd.DataFrame(all_rows, columns=FIELDS)
    if df.empty:
        print(f"[warn] {day} 无数据（非交易日？）")
        return
    dtypes = {"date": "int64", "code": "str", "name": "str", "volume": "int64",
              "amount": "int64", "is_st": "bool", "total_share": "int64",
              "float_share": "int64", "total_mv": "int64", "float_mv": "int64"}
    df = df.astype({k: v for k, v in dtypes.items() if k in df.columns})
    df = df.drop_duplicates(["date", "code"]).sort_values(["date", "code"])
    temp = out.with_suffix('.parquet.tmp')
    df.to_parquet(temp, index=False)
    temp.replace(out)
    print(f"[ok] {out.name} shape={df.shape}")


def main(daily_only=False, through_date=None):
    sdb = latest_stockdb_date()
    target = through_date or sdb
    if target > sdb:
        raise SystemExit(f'stockdb 最新={sdb}，尚未更新到所需收盘日 {target}')
    parts = latest_parts_date()
    print(f"stockdb 最新={sdb} | data_daily_parts 最新={parts}")
    if target <= parts:
        print("无需增量")
    else:
        # 逐日补（跨节假日多日缺失也能补）
        import datetime as dt
        d0 = dt.datetime.strptime(str(parts), "%Y%m%d") + dt.timedelta(days=1)
        d1 = dt.datetime.strptime(str(target), "%Y%m%d")
        cur = d0
        while cur <= d1:
            fetch_day(cur.strftime("%Y%m%d"))
            cur += dt.timedelta(days=1)
    if latest_parts_date() < target:
        raise SystemExit(f'同步后日线仍未到 {target}，不生成新名单')
    if daily_only:
        print(f'[done] 日线已同步到 {target}；仅同步分片，不重建观察池')
        return
    import subprocess
    ml = Path(__file__).resolve().parent.parent
    pool_script = ml / "green_watch_v0" / "build_pool.py"
    py = Path(sys.executable).parent.parent / ".venv" / "Scripts" / "python.exe"
    if not py.exists():
        import os
        py = Path(os.environ.get("STARGAZER_ML_PYTHON",
                                r"D:\stargazer\.venv\Scripts\python.exe"))
    print(f"[pool] {py} build_pool.py --date {sdb}")
    r = subprocess.run([str(py), "-X", "utf8", str(pool_script), "--date", str(sdb)],
                       capture_output=True, text=True, encoding="utf-8", errors="replace",
                       timeout=900)
    tail = (r.stdout or "")[-800:] + (r.stderr or "")[-400:]
    print(tail)
    if r.returncode != 0:
        raise SystemExit(f"build_pool 失败: {tail[-300:]}")
    lj = ml / "green_watch_v0" / "pool_latest.json"
    info = json.loads(lj.read_text(encoding="utf-8"))
    print(f"[done] pool_date={info['pool_date']} next={info['next_trading_date']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--daily-only', action='store_true')
    parser.add_argument('--through-date', type=int)
    args = parser.parse_args()
    main(daily_only=args.daily_only, through_date=args.through_date)
