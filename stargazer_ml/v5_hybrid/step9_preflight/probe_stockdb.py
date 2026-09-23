"""probe_stockdb — 直连 free-stockdb HTTP 接口，探明 Gate 1 七项数据集真实可用性。

用 HTTP 接口（127.0.0.1:7899）而非 .pyd，规避 Python 版本约束，可在 _dev_venv 直接跑。
用途：先把"stockdb 到底有哪些表 / 覆盖到哪"摸清楚，再据此写真实的 data_availability_audit。

运行: python probe_stockdb.py
"""
from __future__ import annotations
import json
import sys
import time
import urllib.parse
import urllib.request

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:  # noqa
    pass

try:
    import msgpack
except ImportError:
    msgpack = None

BASE = "http://127.0.0.1:7899"


def get(table: str, code: str = "*", date: str = "*", timeout: int = 30):
    t = urllib.parse.quote(f"{table}:{code}:{date}")
    url = f"{BASE}/?cmd=get&t={t}"
    headers = {"Accept": "application/json, application/x-msgpack"}
    try:
        import requests
        resp = requests.get(url, headers=headers, timeout=timeout)
        return {"_raw": resp.content, "_headers": dict(resp.headers), "_via": "requests"}
    except ImportError:
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return {"_raw": r.read(), "_headers": dict(r.headers), "_via": "urllib"}


def _parse(raw: bytes):
    """stockdb 默认返回 msgpack；若带 Accept 协商成 JSON 则走 json。"""
    if msgpack is not None:
        try:
            return msgpack.unpackb(raw, raw=False, strict_map_key=False)
        except Exception:  # noqa
            pass
    return json.loads(raw.decode("utf-8", errors="replace"))


def analyze(label: str, table: str, code: str = "*", date: str = "*"):
    t0 = time.time()
    res = get(table, code, date)
    dt = time.time() - t0
    if "__error__" in res:
        print(f"[{label}] ERROR {res['__error__']} ({dt:.1f}s)")
        return None
    raw = res["_raw"]
    print(f"[{label}] via={res['_via']} len={len(raw)} head={raw[:8].hex()} "
          f"ct={res['_headers'].get('Content-Type')} ({dt:.1f}s)")
    try:
        obj = _parse(raw)
    except Exception as e:  # noqa
        print(f"   parse-fail: {type(e).__name__}: {e}")
        return {"raw": raw}
    if isinstance(obj, dict):
        total = sum(len(v) if isinstance(v, list) else 1 for v in obj.values())
        print(f"   DICT keys={list(obj.keys())} total_codes={total}")
        return {"dict": True, "total": total}
    if isinstance(obj, list):
        recs = [it[1] for it in obj if isinstance(it, (list, tuple)) and len(it) >= 2 and isinstance(it[1], dict)]
        cols = set()
        for r in recs[:60]:
            cols.update(r.keys())
        dates = [r.get("date") for r in recs if "date" in r]
        dmin, dmax = (min(dates), max(dates)) if dates else (None, None)
        print(f"   LIST rows={len(obj)} cols={sorted(cols)} date=[{dmin},{dmax}]")
        return {"n": len(obj), "cols": sorted(cols), "dmin": dmin, "dmax": dmax}
    print(f"   unknown type {type(obj)}")
    return {"obj": obj}


def main():
    print("=== 0. universe ===")
    analyze("universe", "股票代码")
    print("\n=== 1. daily 600000 (含 float_mv/is_st?) ===")
    analyze("daily", "日k", "600000")
    print("\n=== 2. minute 600000 ===")
    analyze("minute", "分钟k", "600000")
    print("\n=== 3. adjustments 600000 ===")
    analyze("adj", "复权", "600000")
    print("\n=== 4. index probes (日k) ===")
    for c in ["sh000001", "sh000300", "sz399001", "sz399006", "sh000688"]:
        analyze(f"idx_{c}", "日k", c)
    print("\n=== 5. index minute probes ===")
    for c in ["sh000001", "sz399006"]:
        analyze(f"idxmin_{c}", "分钟k", c)
    print("\n=== 6. delisted ===")
    analyze("delisted", "退市")
    print("\n=== 7. sector/board probes ===")
    analyze("board_600000", "板块", "600000")
    analyze("sw_600000", "申万", "600000")
    analyze("ind_600000", "行业", "600000")
    print("\n=== 8. halt/trading status probe ===")
    # 尝试常见停牌/停复牌表名
    for nm in ["停牌", "停复牌", "暂停上市", "特别处理", "股票状态"]:
        analyze(f"status_{nm}", nm, "600000")


if __name__ == "__main__":
    main()
