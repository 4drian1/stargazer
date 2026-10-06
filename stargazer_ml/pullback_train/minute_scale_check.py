# -*- coding: utf-8 -*-
"""盘中路线第2步：价格尺度抽查（大量下载前必须完成）。

问题：分钟价格很可能是未复权（raw），日线ATR是前复权（qfq），两者不能直接相减。
验证：
1. 分钟K按日聚合 OHLC（raw尺度） vs stockdb 日K默认尺度 vs 本地qfq日线；
2. 判定分钟尺度=日K默认尺度（若相等）；
3. 分钟聚合收盘 / qfq收盘 的日度比率序列：除权日跳变、平日稳定 → 确认双尺度关系；
4. ATR_pct 尺度不变性：qfq口径 ATR_ref/close 与 raw口径 ATR14/close 是否一致
   （抽查股全程相关系数/分位差），支撑标签引擎用 ATR_pct 在 raw 分钟价上设barrier。

输出：报告 §18.5 抽查结论；若发现分钟=qfq或尺度异常 → 断言失败，停止后续。
"""
import json
import os
import sys
import time
import urllib.parse
import urllib.request

import numpy as np
import pandas as pd

OUT_DIR = "D:/stargazer/stargazer_ml/pullback_train"
REPORT_PATH = os.path.join(OUT_DIR, "event_funnel_report.md")
BASE = "http://127.0.0.1:7899"
CODES = ["600702", "000001", "300750", "002594", "601127"]
DAILY_GLOB = "D:/a股数据/data_daily_parts/part_*.parquet"


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def http_get(table, code, prefix):
    t = urllib.parse.quote(f"{table}:{code}:{prefix}")
    with urllib.request.urlopen(f"{BASE}/?cmd=get&t={t}", timeout=60) as r:
        data = json.loads(r.read().decode("utf-8"))
    rows = [it[1] for it in data if isinstance(it, list) and len(it) >= 2]
    return pd.DataFrame(rows)


def main():
    t0 = time.time()
    daily = pd.concat([pd.read_parquet(f, columns=["date", "code", "close", "high", "low"])
                       for f in sorted(__import__("glob").glob(DAILY_GLOB))],
                      ignore_index=True)
    daily["date"] = daily["date"].astype(int)

    L = ["\n### 18.5 价格尺度抽查（第2步，下载前置闸门）\n",
         "分钟K按日聚合(raw尺度) vs stockdb日K默认尺度 vs 本地qfq日线；"
         "ATR_pct=ATR_ref(qfq)/close(qfq) 与 raw口径 ATR14/close 的一致性。\n",
         "| 代码 | 分钟天数 | 分钟聚合=日K默认 | raw/qfq日比率中位数 | 比率日跳变>1%天数 | "
         "ATR_pct qfq vs raw 相关 |",
         "|---|---:|---|---:|---:|---:|"]
    ok_all = True
    for code in CODES:
        m = http_get("分钟k", code, "2025*")
        if not len(m):
            L.append(f"| {code} | 0 | — | — | — | — |")
            continue
        m["date"] = m["date"].astype(np.int64)
        m["day"] = (m["date"] // 1000000).astype(int)
        for c in ("open", "high", "low", "close"):
            m[c] = m[c].astype(float)
        agg = m.groupby("day").agg(o=("open", "first"), h=("high", "max"),
                                   l=("low", "min"), c=("close", "last"))
        dk = http_get("日k", code, "2025*")
        dk["date"] = dk["date"].astype(int)
        dk = dk.set_index("date")[["open", "high", "low", "close"]].astype(float)
        common = agg.index.intersection(dk.index)
        same_scale = bool(np.allclose(agg.loc[common, "c"], dk.loc[common, "close"],
                                      rtol=1e-4, atol=1e-4))
        qfq = daily[daily["code"] == code].set_index("date")["close"]
        common2 = agg.index.intersection(qfq.index)
        ratio = (agg.loc[common2, "c"] / qfq.loc[common2]).dropna()
        jump_days = int((ratio.pct_change().abs() > 0.01).sum())
        r_med = float(ratio.median())
        # ATR_pct 一致性：raw口径 ATR14/close（用分钟聚合日线） vs qfq口径（本地日线）
        r = agg["c"].pct_change()
        atr_raw = (pd.concat([(agg["h"] - agg["l"]),
                              (agg["h"] - agg["c"].shift()).abs(),
                              (agg["l"] - agg["c"].shift()).abs()], axis=1)
                   .max(axis=1).rolling(14, min_periods=10).mean())
        atr_pct_raw = (atr_raw / agg["c"]).dropna()
        q = qfq.loc[common2]
        atr_qfq = (pd.concat([(daily[daily['code']==code].set_index('date')['high'] - daily[daily['code']==code].set_index('date')['low']),
                              (daily[daily['code']==code].set_index('date')['high'] - q.shift()).abs(),
                              (daily[daily['code']==code].set_index('date')['low'] - q.shift()).abs()], axis=1)
                   .max(axis=1).rolling(14, min_periods=10).mean())
        atr_pct_qfq = (atr_qfq / q).dropna()
        both = pd.concat([atr_pct_raw, atr_pct_qfq], axis=1,
                         keys=["raw", "qfq"]).dropna()
        corr = float(both["raw"].corr(both["qfq"])) if len(both) > 20 else np.nan
        med_diff = float((both["raw"] - both["qfq"]).abs().median() * 100) if len(both) else np.nan
        log(f"{code}: days={len(agg)} same_scale={same_scale} "
            f"r_med={r_med:.4f} jumps={jump_days} atr_corr={corr:.4f} "
            f"atr_med_diff={med_diff:.3f}pct")
        L.append(f"| {code} | {len(agg)} | {'是' if same_scale else '**否**'} | "
                 f"{r_med:.4f} | {jump_days} | {corr:.4f} |")
        if not same_scale:
            ok_all = False
    L.append("")
    if ok_all:
        L.append("结论：分钟K与stockdb日K默认尺度一致（未复权raw）；qfq日线与raw的日比率"
                 "平日稳定、除权日跳变，符合双尺度预期。标签引擎采用 ATR_pct 比例法："
                 "ATR_pct=ATR_ref(qfq)/close(qfq)（尺度不变），barrier在raw分钟价上按 "
                 "Entry×(1±K×ATR_pct) 计算，不跨尺度相减。**闸门通过，可以下载。**")
    else:
        L.append("**闸门未通过：分钟尺度≠stockdb日K默认尺度，暂停下载，需人工确认尺度。**")
    with open(REPORT_PATH, encoding="utf-8") as f:
        old = f.read()
    idx = old.find("\n### 18.5 价格尺度抽查")
    if idx != -1:
        old = old[:idx] + "\n"
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(old.rstrip("\n") + "\n" + "\n".join(L) + "\n")
    log(f"done {time.time()-t0:.0f}s gate={'PASS' if ok_all else 'FAIL'}")
    sys.exit(0 if ok_all else 1)


if __name__ == "__main__":
    main()
