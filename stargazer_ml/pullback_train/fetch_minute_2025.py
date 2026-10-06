# -*- coding: utf-8 -*-
"""盘中路线第3步：候选股票2025分钟数据增量下载器。

- 范围：minute_candidates_2025.parquet 中非北交所股票（北交所无分钟覆盖）；
- 窗口：2025全年（信号基线+信号日+T+5的主体）+ 202601*（12月末信号的T+5溢出）；
  2025年1月信号的"前25日基线"因分钟数据2025-01-02才开始而截断 → 标签引擎
  以完整性标志处理，不在下载端补；
- HTTP前缀拉取：t=分钟k:{code}:2025* 与 t=分钟k:{code}:202601*；
- 缓存：minute_cache/{code}.parquet（原始分钟，未加工），已存在则跳过（断点续拉）；
- meta：minute_cache/meta.json 记录每 code 状态(bars/retries/error/pulled_at)；
  run_meta.json 记录数据版本、拉取时间、源服务地址、候选清单文件；
- 完成后输出覆盖率报告（报告 §18.6）与非北交所候选事件分钟覆盖率(验收≥95%)。

用法： python fetch_minute_2025.py [--limit N] [--workers 12]
"""
import json
import os
import sys
import threading
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd

OUT_DIR = "D:/stargazer/stargazer_ml/pullback_train"
CACHE = os.path.join(OUT_DIR, "minute_cache")
META_PATH = os.path.join(CACHE, "meta.json")
RUN_META = os.path.join(CACHE, "run_meta.json")
CAND = os.path.join(OUT_DIR, "minute_candidates_2025.parquet")
REPORT_PATH = os.path.join(OUT_DIR, "event_funnel_report.md")
BASE = os.environ.get("STOCKDB_URL", "http://127.0.0.1:7899")
PREFIXES = ("2025*", "202601*")
MAX_RETRY = 3
TIMEOUT = 120

_lock = threading.Lock()
_meta = {}


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def fetch_prefix(code, prefix):
    t = urllib.parse.quote(f"分钟k:{code}:{prefix}")
    with urllib.request.urlopen(f"{BASE}/?cmd=get&t={t}", timeout=TIMEOUT) as r:
        data = json.loads(r.read().decode("utf-8"))
    return [it[1] for it in data
            if isinstance(it, list) and len(it) >= 2 and isinstance(it[1], dict)]


def fetch_code(code):
    if os.path.exists(os.path.join(CACHE, f"{code}.parquet")):
        return code, "cached", 0, ""
    rows, retries = [], 0
    for pf in PREFIXES:
        for attempt in range(MAX_RETRY):
            try:
                rows.extend(fetch_prefix(code, pf))
                break
            except Exception as e:  # noqa: BLE001
                retries += 1
                if attempt == MAX_RETRY - 1:
                    return code, "error", retries, f"{pf}: {type(e).__name__}"
                time.sleep(1.5 * (attempt + 1))
    df = pd.DataFrame(rows)
    if len(df):
        df = df.drop_duplicates(subset=["date"]).sort_values("date")
        for c in ("open", "high", "low", "close", "volume", "amount"):
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")
        df["date"] = df["date"].astype(np.int64)
        df.to_parquet(os.path.join(CACHE, f"{code}.parquet"), index=False)
    else:
        # 无数据也落空标记，避免重复请求
        pd.DataFrame({"date": pd.Series(dtype="int64")}).to_parquet(
            os.path.join(CACHE, f"{code}.parquet"), index=False)
    return code, ("ok" if len(df) else "empty"), retries, ""


def main():
    limit = None
    workers = 12
    args = sys.argv[1:]
    if "--limit" in args:
        limit = int(args[args.index("--limit") + 1])
    if "--workers" in args:
        workers = int(args[args.index("--workers") + 1])

    os.makedirs(CACHE, exist_ok=True)
    if os.path.exists(META_PATH):
        with open(META_PATH, encoding="utf-8") as f:
            _meta.update(json.load(f))
    cand = pd.read_parquet(CAND)
    codes = sorted(c for c in cand["code"].unique() if not c.startswith(("8", "4", "92")))
    if limit:
        todo = [c for c in codes
                if not os.path.exists(os.path.join(CACHE, f"{c}.parquet"))][:limit]
    else:
        todo = [c for c in codes
                if not os.path.exists(os.path.join(CACHE, f"{c}.parquet"))]
    log(f"codes total={len(codes)} todo={len(todo)} workers={workers}")

    t0 = time.time()
    done = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(fetch_code, c): c for c in todo}
        for fut in as_completed(futs):
            code, status, retries, err = fut.result()
            with _lock:
                _meta[code] = {"status": status, "bars": None, "retries": retries,
                               "error": err,
                               "pulled_at": time.strftime("%Y-%m-%d %H:%M:%S")}
                done += 1
                if done % 50 == 0 or done == len(todo):
                    with open(META_PATH, "w", encoding="utf-8") as f:
                        json.dump(_meta, f, ensure_ascii=False)
                    el = time.time() - t0
                    log(f"{done}/{len(todo)} elapsed={el:.0f}s "
                        f"eta={(el/done)*(len(todo)-done):.0f}s")
    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump(_meta, f, ensure_ascii=False)
    with open(RUN_META, "w", encoding="utf-8") as f:
        json.dump({"data_version": "minute_1m_2025+v202601",
                   "source": BASE, "pulled_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                   "candidates": os.path.basename(CAND),
                   "prefixes": list(PREFIXES)}, f, ensure_ascii=False)

    # ---------------- 覆盖率验收 ----------------
    status = pd.Series({k: v["status"] for k, v in _meta.items()})
    log(f"status: {status.value_counts().to_dict()}")
    cov, tot = 0, 0
    ev = cand[~cand["minute_unavailable"]]
    got = set(status[status.isin(["ok", "cached"])].index)
    cov = int(ev["code"].isin(got).sum())
    tot = len(ev)
    log(f"event minute coverage: {cov}/{tot} = {cov/max(tot,1)*100:.2f}%")

    with open(REPORT_PATH, encoding="utf-8") as f:
        old = f.read()
    idx = old.find("\n### 18.6 分钟数据下载与覆盖")
    if idx != -1:
        old = old[:idx] + "\n"
    L = ["\n### 18.6 分钟数据下载与覆盖（第3步）\n",
         f"- 范围：{len(codes)} 只非北交所候选股，前缀 {list(PREFIXES)}，"
         f"缓存 minute_cache/{{code}}.parquet（原始分钟，加工特征另存）",
         f"- 状态分布：{status.value_counts().to_dict()}",
         f"- 非北交所候选事件分钟覆盖率：{cov}/{tot} = {cov/max(tot,1)*100:.2f}%"
         f"（验收线 ≥95%）",
         f"- run_meta：{json.load(open(RUN_META, encoding='utf-8'))}"]
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(old.rstrip("\n") + "\n" + "\n".join(L) + "\n")
    log(f"done in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
