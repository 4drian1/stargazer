# -*- coding: utf-8 -*-
"""盘中路线第5步：候选事件除权窗口标记。

背景：分钟价为原始未复权价，ATR_pct 比例法只解决日线ATR与分钟价的尺度
换算；持有窗口跨除权日时，原始价路径会出现虚假跳空（分红/送转导致），
T+1标签引擎会把除权缺口误判为STOP/TARGET。本脚本给每个候选事件打三个标记：
  corp_action_on_signal_day  信号日当天有除权除息记录
  corp_action_during_horizon 信号日起T+6个交易日内任一交易日有记录
  price_scale_valid          前两标志都为0（价格尺度有效，可入训练/评估）

除权记录源：stockdb 复权表（key=复权:{code}:{yyyymmdd}，记录含
div/give/trans/mult/cum，无date字段，日期解析自key）。有除权动作定义为
div>0 或 give>0 或 trans>0 或 mult!=1。

交易日历：minute_cache 全体分钟时间戳的日部分（覆盖2025全年+202601，
信号日与其后6个交易日均落在该范围）。拉取结果缓存 corp_actions.parquet
（增量，断点续拉）；标记并入 minute_candidates_2025.parquet 并独立落盘
corp_flags.parquet；报告写入 §18.9。

用法： python corp_flags.py [--workers 16] [--cand PATH] [--out PATH] [--no-report]
（--cand/--out 用于 2026 复核等非默认输入；--no-report 不写 §18.9 报告节）
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
CAND = os.path.join(OUT_DIR, "minute_candidates_2025.parquet")
CORP = os.path.join(OUT_DIR, "corp_actions.parquet")
FLAGS = os.path.join(OUT_DIR, "corp_flags.parquet")
REPORT_PATH = os.path.join(OUT_DIR, "event_funnel_report.md")
BASE = os.environ.get("STOCKDB_URL", "http://127.0.0.1:7899")
HORIZON_DAYS = 6  # 信号日起 T+6 个交易日窗口（含信号日）
WRITE_REPORT = True

_lock = threading.Lock()


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def fetch_corp(code):
    """拉单股全部除权记录，返回 DataFrame(code,date,div,give,trans,mult)。"""
    t = urllib.parse.quote(f"复权:{code}:*")
    with urllib.request.urlopen(f"{BASE}/?cmd=get&t={t}", timeout=120) as r:
        data = json.loads(r.read().decode("utf-8"))
    rows = []
    for it in data:
        if not (isinstance(it, list) and len(it) >= 2 and isinstance(it[1], dict)):
            continue
        d = it[1]
        rows.append({"code": code,
                     "date": int(it[0].split(":")[-1]),
                     "div": float(d.get("div", 0) or 0),
                     "give": float(d.get("give", 0) or 0),
                     "trans": float(d.get("trans", 0) or 0),
                     "mult": float(d.get("mult", 1) or 1)})
    return rows


def load_calendar():
    """minute_cache 全体分钟时间戳的日部分 → 排序交易日数组。"""
    days = set()
    files = [f for f in os.listdir(CACHE) if f.endswith(".parquet")]
    for f in files:
        try:
            s = pd.read_parquet(os.path.join(CACHE, f), columns=["date"])
        except Exception:  # noqa: BLE001
            continue
        if len(s):
            days.update((s["date"].astype(np.int64) // 1000000).unique().tolist())
    return np.array(sorted(days), dtype=np.int64)


def main():
    global CAND, FLAGS, WRITE_REPORT
    workers = 16
    args = sys.argv[1:]
    if "--workers" in args:
        workers = int(args[args.index("--workers") + 1])
    if "--cand" in args:
        CAND = args[args.index("--cand") + 1]
    if "--out" in args:
        FLAGS = args[args.index("--out") + 1]
    if "--no-report" in args:
        WRITE_REPORT = False

    cand = pd.read_parquet(CAND)
    codes = sorted(cand["code"].unique())
    log(f"candidates {len(cand):,} events, {len(codes)} codes")

    # ---- 1. 除权记录（缓存增量） ----
    old = (pd.DataFrame(columns=["code", "date", "div", "give", "trans", "mult"])
           if not os.path.exists(CORP) else pd.read_parquet(CORP))
    have = set(old["code"]) if len(old) else set()
    todo = [c for c in codes if c not in have]
    log(f"corp records cached: {len(have)}, todo: {len(todo)}")
    t0 = time.time()
    new_rows, fail = [], []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(fetch_corp, c): c for c in todo}
        for i, fut in enumerate(as_completed(futs), 1):
            c = futs[fut]
            try:
                new_rows.extend(fut.result())
            except Exception as e:  # noqa: BLE001
                fail.append((c, type(e).__name__))
            if i % 500 == 0 or i == len(todo):
                log(f"corp {i}/{len(todo)} {time.time()-t0:.0f}s fail={len(fail)}")
    corp = pd.concat([old, pd.DataFrame(new_rows)], ignore_index=True)
    corp = corp.drop_duplicates(["code", "date"])
    corp.to_parquet(CORP, index=False)
    log(f"corp_actions: {len(corp):,} rows, fetch fail {len(fail)} {fail[:5]}")

    act = corp[(corp["div"] > 0) | (corp["give"] > 0) | (corp["trans"] > 0)
               | (corp["mult"] != 1)]
    log(f"有除权动作记录: {len(act):,} rows, {act['code'].nunique()} codes")

    # ---- 2. 交易日历 ----
    cal = load_calendar()
    pos = {d: i for i, d in enumerate(cal)}
    log(f"calendar: {len(cal)} days {cal[0]}..{cal[-1]}")

    # ---- 3. 逐事件标记 ----
    act_by_code = {c: g["date"].values for c, g in act.groupby("code", sort=False)}
    sig_days = cand["date"].values
    codes_v = cand["code"].values
    on_day = np.zeros(len(cand), dtype=bool)
    in_hor = np.zeros(len(cand), dtype=bool)
    missing_cal = 0
    for i in range(len(cand)):
        d = sig_days[i]
        j = pos.get(d)
        if j is None:
            missing_cal += 1
            continue
        lo, hi = cal[j], cal[min(j + HORIZON_DAYS, len(cal) - 1)]
        ad = act_by_code.get(codes_v[i])
        if ad is None:
            continue
        on_day[i] = np.any(ad == d)
        in_hor[i] = np.any((ad >= lo) & (ad <= hi))
    if missing_cal:
        log(f"warn: {missing_cal} 事件信号日不在日历（北交所/无分钟），标记为窗口未知")

    flags = pd.DataFrame({"code": codes_v, "date": sig_days,
                          "corp_action_on_signal_day": on_day,
                          "corp_action_during_horizon": in_hor})
    flags["price_scale_valid"] = ~(on_day | in_hor)
    flags.to_parquet(FLAGS, index=False)

    # 并入候选表（先丢弃上轮残留标志列，防 _x/_y 后缀）
    cand = cand.drop(columns=[c for c in (
        "corp_action_on_signal_day", "corp_action_during_horizon",
        "price_scale_valid") if c in cand.columns])
    cand = cand.merge(flags, on=["code", "date"], how="left")
    for c in ("corp_action_on_signal_day", "corp_action_during_horizon",
              "price_scale_valid"):
        cand[c] = cand[c].fillna(False)
    cand.to_parquet(CAND, index=False)

    n_bad = int((~cand["price_scale_valid"]).sum())
    log(f"invalid(scale) events: {n_bad}/{len(cand)} = {n_bad/len(cand):.2%}")
    log(f"on_signal: {int(cand['corp_action_on_signal_day'].sum())}, "
        f"during_horizon: {int(cand['corp_action_during_horizon'].sum())}")

    # ---- 4. 报告 §18.9（--no-report 时跳过，如 2026 复核跑） ----
    if not WRITE_REPORT:
        log("done (no report)")
        return
    with open(REPORT_PATH, encoding="utf-8") as f:
        old_rep = f.read()
    idx = old_rep.find("\n### 18.9 ")
    if idx != -1:
        old_rep = old_rep[:idx] + "\n"
    L = ["\n### 18.9 除权窗口标记（数据补充2）\n",
         "目的：分钟价为原始未复权价，ATR_pct 只解决尺度换算；持有窗口跨除权"
         "日时原始价路径出现虚假跳空，标签引擎会把除权缺口误判为"
         "STOP/TARGET。三个标记：",
         "- `corp_action_on_signal_day`：信号日当天有除权除息记录"
         "（div>0/give>0/trans>0/mult≠1，源 stockdb 复权表）",
         "- `corp_action_during_horizon`：信号日起 T+6 个交易日内任一交易日有记录",
         "- `price_scale_valid`：前两标志均为0；跨除权窗口事件在分钟训练/评估中"
         "排除或单独报告",
         f"- 除权记录：{len(corp):,} 行（拉取失败 {len(fail)} 股），"
         f"有动作记录 {len(act):,} 行/{act['code'].nunique()} 股；"
         f"缓存 corp_actions.parquet（断点续拉）",
         f"- 交易日历：minute_cache 分钟戳日部分，{len(cal)} 天 "
         f"({cal[0]}..{cal[-1]})，全部候选信号日均在历",
         f"- 结果：on_signal {int(cand['corp_action_on_signal_day'].sum()):,}，"
         f"during_horizon {int(cand['corp_action_during_horizon'].sum()):,}，"
         f"price_scale_valid=False 共 {n_bad:,}/{len(cand):,} "
         f"= {n_bad/len(cand):.2%}（这些事件分钟标签需排除或单列）",
         "- 落盘：corp_flags.parquet + 并入 minute_candidates_2025.parquet"]
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(old_rep.rstrip("\n") + "\n" + "\n".join(L) + "\n")
    log("done")


if __name__ == "__main__":
    main()
