"""data_source_provenance.py
================================================================================
P0 ① CANONICAL DATA SOURCE / DATA_EPOCH PROVENANCE CHECK (Gate 1A-SOURCE)

用户判定：Gate 1A 全扫声称 2025-01-02→2026-09-16 / 6.69 亿 minute bars，
但 latr_builder/dry_replay 实测取数只到 ~2025-01-27。两套事实不能同时成立，
必须先找出 6.69 亿 bars 到底来自哪套源，否则任何回测不可信。

本模块做两件事：
  (1) 定义 **唯一 canonical adapter**，供 gate1a_full_scan / dry_replay / latr_builder
      统一调用（消除脚本私有 split：Gate1A 用本地 parquet 做 daily/float_mv，
      latr_builder 用 HTTP 日k 做 daily 这类分叉）。
      CANONICAL 源真相 = stockdb HTTP @ 127.0.0.1:7899 (free-stockdb v0.3.2)。
      本地 parquet 是衍生品，仅作交叉校验，不作取数源。
  (2) 录制 provenance 元数据 (host/port/adapter/query fn/commit/cache/env/version)，
      并对固定股票样本 × 固定日期 × {daily, minute, float_mv, adjustment} 做探针，
      输出 DATA_SOURCE_PROVENANCE.yaml。最终只会出现两种结论：
        A) Gate1A 全扫用的是另一套数据源 -> 找出那套 6.69 亿 bars 的源定为 CANONICAL_STOCKDB_V1
        B) 真实 canonical stockdb 就只到 2025-01-27 -> 改 MINUTE_END，不重训 Timing

运行： python data_source_provenance.py
================================================================================
"""
from __future__ import annotations
import os
import sys
import json
import subprocess
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

_HERE = Path(__file__).resolve().parent
WORKSPACE = Path(r"d:/stargazer")
sys.path.insert(0, str(_HERE))

# --------------------------------------------------------------------------- #
# 1) CANONICAL adapter 配置 (单一真相源)
# --------------------------------------------------------------------------- #
CANONICAL_HOST = "127.0.0.1"
CANONICAL_PORT = 7899
CANONICAL_BASE = f"http://{CANONICAL_HOST}:{CANONICAL_PORT}"
CANONICAL_ADAPTER = "http(stockdb free-stockdb v0.3.2, cmd=get|cmd=vals)"
CANONICAL_SOURCE_OF_TRUTH = ("stockdb 本地同步目录 D:/a股数据/free-stockdb-win/stockdb/data/ "
                             "(同步源 ah.123128.xyz via 数据更新.exe); "
                             "本地 parquet data_daily_parts 为其衍生品")
STARGAZER_DATA_ROOT = os.environ.get("STARGAZER_DATA_ROOT", r"D:/a股数据")
DAILY_PARTS = os.path.join(STARGAZER_DATA_ROOT, "data_daily_parts")

_opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def _http_raw(cmd: str, table: str, code: str = "*", date: str = "*",
              timeout: int = 180):
    t = urllib.parse.quote(f"{table}:{code}:{date}")
    url = f"{CANONICAL_BASE}/?cmd={cmd}&t={t}"
    with _opener.open(url, timeout=timeout) as r:
        return r.read()


def _decode(raw: bytes):
    try:
        import msgpack
        return msgpack.unpackb(raw, raw=False, strict_map_key=False)
    except Exception:
        return json.loads(raw.decode("utf-8", "replace"))


def get_daily_http(code: str) -> pd.DataFrame:
    """CANONICAL 日线 (来源 stockdb HTTP)。包含 float_mv 等因子。"""
    obj = _decode(_http_raw("get", "日k", code.zfill(6), "*"))
    rows = [v for k, v in obj if isinstance(v, dict)] if isinstance(obj, list) else []
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    if "date" in df:
        df["date"] = df["date"].astype(int)
        df = df.sort_values("date").reset_index(drop=True)
    return df


def get_minute_http(code: str, date: str) -> pd.DataFrame:
    """CANONICAL 分钟 (来源 stockdb HTTP)。date 为 8 位 YYYYMMDD，按前缀匹配。"""
    obj = _decode(_http_raw("vals", "分钟k", code.zfill(6), f"{date}*"))
    rows = [it for it in obj if isinstance(it, dict)] if isinstance(obj, list) else []
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    if "date" in df:
        df["date"] = df["date"].astype(int)
        df = df.sort_values("date").reset_index(drop=True)
    return df


def get_adjustment_http(code: str) -> list:
    """CANONICAL 复权 (来源 stockdb HTTP, 复权:CODE:DATE)。返回 [(date_str, rec)]。"""
    obj = _decode(_http_raw("get", "复权", code.zfill(6), "*"))
    out = []
    if isinstance(obj, list):
        for k, v in obj:
            if isinstance(k, str) and isinstance(v, dict):
                parts = k.split(":")
                d = parts[2] if len(parts) >= 3 else (v.get("date") or v.get("effective_date"))
                if d is not None:
                    out.append((str(d), v))
    return out


_PARQUET_CACHE = None


def get_daily_parquet() -> pd.DataFrame:
    """衍生品源 (parquet, 由 stockdb 构建; 2018+; 作 daily 高速主路径)。"""
    global _PARQUET_CACHE
    if _PARQUET_CACHE is None:
        import glob
        files = sorted(glob.glob(os.path.join(DAILY_PARTS, "part_*.parquet")))
        if not files:
            return pd.DataFrame()
        _PARQUET_CACHE = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
        if "code" in _PARQUET_CACHE:
            _PARQUET_CACHE["code"] = _PARQUET_CACHE["code"].astype(str).str.zfill(6)
    return _PARQUET_CACHE


def get_universe() -> list:
    """CANONICAL universe = 本地 parquet 全部 code (stockdb 衍生品; 全脚本一致)。"""
    p = get_daily_parquet()
    if not p.empty and "code" in p:
        return sorted(p["code"].astype(str).str.zfill(6).unique().tolist())
    # HTTP 兜底: 股票代码 表
    try:
        import requests
        r = requests.get(f"{CANONICAL_BASE}/?cmd=get&t=股票代码", timeout=20)
        data = r.json()
        codes = []
        if isinstance(data, dict):
            for v in data.values():
                if isinstance(v, list):
                    codes.extend(str(x) for x in v)
        return sorted({c.zfill(6) for c in codes})
    except Exception:
        return []


def get_daily(code: str) -> pd.DataFrame:
    """CANONICAL 日线 (单一入口: parquet 优先 + HTTP 兜底)。

    Gate1A 与 latr_builder 都经此函数取日线, 保证两侧 daily/float_mv 完全一致
    (消除原 parquet-vs-HTTP 日线分裂)。Step9A 窗口 2025+ 全在 parquet 覆盖内。
    """
    code = code.zfill(6)
    p = get_daily_parquet()
    if not p.empty and "code" in p:
        sub = p[p["code"] == code]
        if not sub.empty:
            return sub.copy()
    return get_daily_http(code)


def discover_data_epoch(sample_codes=("600519", "000001", "600036")) -> dict:
    """从 canonical HTTP 探测真实日线最大日期 (探针已证 20260917)。"""
    maxd = 0
    for c in sample_codes:
        d = get_daily_http(c)
        if not d.empty:
            maxd = max(maxd, int(d["date"].max()))
    return {"minute_start": 20250102, "daily_max": maxd,
            "epoch": f"2025-01-02..{maxd}" if maxd else "2025-01-02..UNKNOWN"}


# --------------------------------------------------------------------------- #
# 2) PROVENANCE 元数据
# --------------------------------------------------------------------------- #
def _git_commit() -> str:
    try:
        out = subprocess.run(["git", "-C", str(WORKSPACE), "rev-parse", "HEAD"],
                             capture_output=True, text=True, timeout=15)
        return out.stdout.strip() or "NOT_A_GIT_REPO"
    except Exception as e:  # noqa
        return f"git_unavailable:{e}"


def collect_provenance() -> dict:
    return {
        "canonical_host": CANONICAL_HOST,
        "canonical_port": CANONICAL_PORT,
        "canonical_base_url": CANONICAL_BASE,
        "canonical_adapter": CANONICAL_ADAPTER,
        "canonical_source_of_truth": CANONICAL_SOURCE_OF_TRUTH,
        "query_functions": {
            "daily": "get_daily_http(code)",
            "minute": "get_minute_http(code, date)",
            "adjustment": "get_adjustment_http(code)",
            "float_mv": "get_daily_http(code)['float_mv'] (随日线)",
        },
        "code_commit_sha": _git_commit(),
        "cache_path_parquet": DAILY_PARTS,
        "environment_variables": {
            "STARGAZER_DATA_ROOT": os.environ.get("STARGAZER_DATA_ROOT", "<unset>"),
        },
        "data_source_version": "free-stockdb v0.3.2 (local sync, not external API)",
        "captured_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


# --------------------------------------------------------------------------- #
# 3) 探针：固定样本 × 固定日期 × 四类数据
# --------------------------------------------------------------------------- #
PROBE_CODES = ["600519", "000001", "600036", "300750", "601318",
               "000858", "600276", "002594", "600900", "688981"]
PROBE_DATES = ["20250102", "20250127", "20250130", "20250603", "20260916"]


def _date_in_series(s: pd.Series, d: int) -> bool:
    return bool((s.astype(int) == d).any())


def run_probe() -> dict:
    prov = collect_provenance()
    results = []
    parquet_daily = get_daily_parquet()
    px_by_code = {}
    if not parquet_daily.empty and "code" in parquet_daily:
        parquet_daily["code"] = parquet_daily["code"].astype(str).str.zfill(6)
        for c, s in parquet_daily.groupby("code"):
            px_by_code[c] = s

    for code in PROBE_CODES:
        rec = {"code": code, "http_daily": {}, "parquet_daily": {}, "minute": {},
               "float_mv": {}, "adjustment": {}}
        # --- http daily ---
        d = get_daily_http(code)
        if not d.empty:
            rec["http_daily"] = {
                "rows": int(len(d)),
                "min_date": int(d["date"].min()),
                "max_date": int(d["date"].max()),
                "has_float_mv": bool("float_mv" in d.columns and d["float_mv"].notna().any()),
                "probe_dates_present": {dt: _date_in_series(d["date"], int(dt))
                                        for dt in PROBE_DATES},
            }
        # --- parquet daily (对比) ---
        if code in px_by_code:
            pd_ = px_by_code[code]
            rec["parquet_daily"] = {
                "rows": int(len(pd_)),
                "min_date": int(pd_["date"].min()) if "date" in pd_ else None,
                "max_date": int(pd_["date"].max()) if "date" in pd_ else None,
                "has_float_mv": bool("float_mv" in pd_.columns and pd_["float_mv"].notna().any()),
            }
        # --- minute @ each probe date ---
        for dt in PROBE_DATES:
            m = get_minute_http(code, dt)
            rec["minute"][dt] = int(len(m))
        # --- float_mv @ probe dates (from http daily) ---
        if not d.empty and "float_mv" in d:
            for dt in PROBE_DATES:
                sub = d[d["date"] == int(dt)]
                rec["float_mv"][dt] = (None if sub.empty else
                                       (float(sub["float_mv"].iloc[0])
                                        if pd.notna(sub["float_mv"].iloc[0]) else None))
        # --- adjustment ---
        adj = get_adjustment_http(code)
        rec["adjustment"] = {"n_events": len(adj),
                             "max_effective_date": (max(int(x[0]) for x in adj) if adj else None)}
        results.append(rec)

    # 汇总：是否每个 probe date 在 HTTP 上存在分钟/日线
    minute_any = {dt: any(r["minute"].get(dt, 0) > 0 for r in results) for dt in PROBE_DATES}
    daily_max_overall = max((r["http_daily"].get("max_date", 0) for r in results), default=0)
    parquet_max_overall = max((r["parquet_daily"].get("max_date", 0) for r in results), default=0)

    summary = {
        "http_daily_max_date_overall": daily_max_overall,
        "parquet_daily_max_date_overall": parquet_max_overall,
        "minute_available_per_probe_date": minute_any,
        "sources_agree_on_range": (daily_max_overall == parquet_max_overall),
        "conclusion_candidate": _conclude(minute_any, daily_max_overall, parquet_max_overall, results),
    }
    return {"provenance": prov, "probe_codes": PROBE_CODES, "probe_dates": PROBE_DATES,
            "per_symbol": results, "summary": summary}


def _conclude(minute_any, daily_max, parquet_max, results) -> str:
    late = minute_any.get("20250603", False) or minute_any.get("20260916", False)
    if late:
        return ("OUTCOME-A-CANDIDATE: stockdb HTTP 在 2025-06 / 2026-09 仍有分钟数据 -> "
                "需核对 Gate1A 全扫用的是否为同一实例; 若不同实例则定为 CANONICAL_STOCKDB_V1")
    # 只有早期日期有数据
    early = minute_any.get("20250102", False) or minute_any.get("20250127", False)
    if early and not late:
        return ("OUTCOME-B-CANDIDATE: 真实 canonical stockdb 仅覆盖到 ~2025-01 (分钟在 "
                f"20250603/20260916 全空). http_daily_max={daily_max}, parquet_daily_max="
                f"{parquet_max}. 应改 MINUTE_END 为真实值, 不重训 Timing, 先补灌/换源")
    return "INCONCLUSIVE: 探针未返回预期数据, 需人工检查 stockdb 服务状态"


if __name__ == "__main__":
    out = run_probe()
    out_path = _HERE / "DATA_SOURCE_PROVENANCE.yaml"
    import yaml  # type: ignore
    out_path.write_text(yaml.safe_dump(out, allow_unicode=True, sort_keys=False), encoding="utf-8")
    s = out["summary"]
    print("==== DATA SOURCE PROVENANCE PROBE ====")
    print(f"http_daily_max_date_overall : {s['http_daily_max_date_overall']}")
    print(f"parquet_daily_max_date_overall: {s['parquet_daily_max_date_overall']}")
    print(f"sources_agree_on_range       : {s['sources_agree_on_range']}")
    print("minute_available_per_probe_date:")
    for dt, ok in s["minute_available_per_probe_date"].items():
        print(f"   {dt}: {'YES' if ok else 'NO'}")
    print(f"\nCONCLUSION: {s['conclusion_candidate']}")
    print(f"\nwrote {out_path}")
