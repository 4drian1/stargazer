# -*- coding: utf-8 -*-
"""绿盘低吸 pullback_train - 第3.1步第9条：板块/市场数据拉取（东财接口，curl_cffi 直连）

拉取 12 个关注板块（同花顺口径 -> 东财概念/行业名候选，按优先级取第一个能拉到的）：
  - 板块指数日线历史 2018-01 至今 -> sector_daily.parquet (每板块每日一行)
  - 板块当前成分股快照 -> board_cons.json  【幸存者偏差，报告需显著标注】
  - 沪深300 / 创业板指 / 上证指数日线 -> market_daily.parquet (每日一行)

东财对该出口 IP 有限频封锁（突发式 Connection reset，冷却数分钟），因此：
  - 必须用 curl_cffi impersonate=chrome（requests 的 TLS 指纹基本必死）
  - 请求间 sleep 6-10s
  - 分板块缓存到 _fetch_cache/，成功即落盘；外层轮询 + 长冷却（180-240s）重试缺失项
  - 幂等：重跑只补缺失缓存，最终产物覆盖写
"""
import json
import os
import sys
import time
import random

os.environ["NO_PROXY"] = "*"
os.environ.pop("HTTP_PROXY", None)
os.environ.pop("HTTPS_PROXY", None)

import pandas as pd
from curl_cffi import requests as creq

OUT_DIR = "D:/stargazer/stargazer_ml/pullback_train"
CACHE_DIR = os.path.join(OUT_DIR, "_fetch_cache")
SECTOR_DAILY_PATH = os.path.join(OUT_DIR, "sector_daily.parquet")
MARKET_DAILY_PATH = os.path.join(OUT_DIR, "market_daily.parquet")
CONS_PATH = os.path.join(OUT_DIR, "board_cons.json")
META_PATH = os.path.join(OUT_DIR, "fetch_sector_meta.json")

UT = "fa5fd1943c7b386f172d6893dbfba10b"
UT_CLIST = "bd1d9ddb04089700cf9c27f6f7426281"

SECTORS = {
    "储能": ["储能概念"],
    "半导体": ["半导体"],
    "CPO": ["CPO概念", "共封装光学"],
    "数据中心": ["数据中心"],
    "农业种植": ["农业种植"],
    "创新药": ["创新药"],
    "油气": ["油气设服", "油气开采"],
    "军工": ["军工"],
    "房地产": ["房地产"],
    "通信服务": ["通信服务"],
    "通用设备": ["通用设备"],
    "存储芯片": ["存储芯片"],
}

INDEXES = {"sh000001": "1.000001", "sh000300": "1.000300", "cyb399006": "0.399006"}

sys.stdout.reconfigure(encoding="utf-8")
os.makedirs(CACHE_DIR, exist_ok=True)


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def get_json(url, params, tries=3, timeout=25):
    last = None
    for i in range(tries):
        try:
            r = creq.get(url, params=params, impersonate="chrome", timeout=timeout)
            if r.status_code == 200:
                return r.json()
            last = f"http {r.status_code}"
        except Exception as e:
            last = f"{type(e).__name__}: {str(e)[:60]}"
        time.sleep(3 * (i + 1) + random.uniform(0, 2))
    raise RuntimeError(last)


def cpath(name):
    return os.path.join(CACHE_DIR, name)


def cache_exists(name):
    p = cpath(name)
    return os.path.exists(p) and os.path.getsize(p) > 10


def fetch_board_map(fs, kind):
    out = {}
    pn = 1
    while True:
        j = get_json("https://push2delay.eastmoney.com/api/qt/clist/get", {
            "pn": pn, "pz": 100, "po": 1, "np": 1, "ut": UT_CLIST, "fltt": 2,
            "invt": 2, "fid": "f3", "fs": fs, "fields": "f12,f14"})
        data = j.get("data") or {}
        diff = data.get("diff") or []
        for d in diff:
            out[d["f14"]] = d["f12"]
        total = data.get("total", 0)
        if pn * 100 >= total or not diff:
            break
        pn += 1
        time.sleep(random.uniform(2.0, 4.0))
    log(f"board map ({kind}): {len(out)} boards")
    return out


def fetch_kline(secid, fqt):
    j = get_json("https://push2his.eastmoney.com/api/qt/stock/kline/get", {
        "secid": secid, "ut": UT, "fields1": "f1,f2,f3,f4,f5,f6",
        "fields2": "f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61",
        "klt": "101", "fqt": fqt, "beg": "20180101", "end": "20500101",
        "lmt": "1000000"})
    data = j.get("data") or {}
    klines = data.get("klines") or []
    if not klines:
        return None
    rows = [k.split(",") for k in klines]
    dfk = pd.DataFrame(rows, columns=[
        "date", "open", "close", "high", "low", "volume", "amount",
        "amplitude", "pct_chg", "change", "turnover"])
    dfk["date"] = dfk["date"].str.replace("-", "").astype(int)
    for c in dfk.columns:
        if c != "date":
            dfk[c] = pd.to_numeric(dfk[c], errors="coerce")
    return dfk


def fetch_cons(bk):
    codes = []
    pn = 1
    while True:
        j = get_json("https://push2delay.eastmoney.com/api/qt/clist/get", {
            "pn": pn, "pz": 100, "po": 1, "np": 1, "ut": UT_CLIST, "fltt": 2,
            "invt": 2, "fid": "f3", "fs": f"b:{bk}", "fields": "f12,f14"})
        data = j.get("data") or {}
        diff = data.get("diff") or []
        codes.extend(d["f12"] for d in diff)
        total = data.get("total", 0)
        if pn * 100 >= total or not diff:
            break
        pn += 1
        time.sleep(random.uniform(2.0, 4.0))
    return codes


def ensure_sector(canon, candidates):
    """返回 (status, info)；缓存命中直接返回。每步成功即写缓存。"""
    kf, cf = cpath(f"kline_{canon}.parquet"), cpath(f"cons_{canon}.json")
    if cache_exists(f"kline_{canon}.parquet") and cache_exists(f"cons_{canon}.json"):
        return "cached", {}
    concept = industry = None
    if cache_exists("map_concept.json"):
        concept = json.load(open(cpath("map_concept.json"), encoding="utf-8"))
    if cache_exists("map_industry.json"):
        industry = json.load(open(cpath("map_industry.json"), encoding="utf-8"))
    if concept is None or industry is None:
        concept = fetch_board_map("m:90+t:3+f:!50", "concept")
        json.dump(concept, open(cpath("map_concept.json"), "w", encoding="utf-8"))
        time.sleep(random.uniform(2.0, 4.0))
        industry = fetch_board_map("m:90+t:2+f:!50", "industry")
        json.dump(industry, open(cpath("map_industry.json"), "w", encoding="utf-8"))
        time.sleep(random.uniform(2.0, 4.0))
        concept = json.load(open(cpath("map_concept.json"), encoding="utf-8"))
        industry = json.load(open(cpath("map_industry.json"), encoding="utf-8"))
    bk = src = em_name = None
    for cand in candidates:
        if cand in concept:
            bk, src, em_name = concept[cand], "concept", cand
            break
        if cand in industry:
            bk, src, em_name = industry[cand], "industry", cand
            break
    if bk is None:
        return "nomatch", {}
    if not cache_exists(f"kline_{canon}.parquet"):
        log(f"[{canon}] -> {em_name} ({src}, {bk}); fetching kline ...")
        dfk = fetch_kline(f"90.{bk}", "0")
        time.sleep(random.uniform(3.0, 6.0))
        if dfk is None or len(dfk) < 100:
            return "badkline", {}
        dfk.to_parquet(kf, index=False)
    if not cache_exists(f"cons_{canon}.json"):
        log(f"[{canon}] fetching cons ...")
        codes = fetch_cons(bk)
        json.dump(codes, open(cf, "w", encoding="utf-8"))
        time.sleep(random.uniform(3.0, 6.0))
    return "ok", {"em_name": em_name, "bk": bk, "src": src}


def main():
    t0 = time.time()
    deadline = t0 + 100 * 60
    os.makedirs(OUT_DIR, exist_ok=True)

    meta = {"matched": {}, "failed": [], "cons_count": {}, "hist_range": {}}
    pending = list(SECTORS.keys())
    rnd = 0
    while pending and time.time() < deadline:
        rnd += 1
        log(f"=== round {rnd}: pending={pending}")
        nxt = []
        for canon in pending:
            try:
                status, info = ensure_sector(canon, SECTORS[canon])
                if status == "ok":
                    meta["matched"][canon] = info
                elif status == "cached":
                    pass
                elif status == "nomatch":
                    meta["failed"].append(canon)
                    log(f"[FAIL] {canon}: no candidate name matched")
                    continue
                else:
                    raise RuntimeError(status)
            except Exception as e:
                log(f"[{canon}] round failed: {str(e)[:100]}; cooling down")
                nxt.append(canon)
                time.sleep(random.uniform(180, 240))
        pending = nxt
    if pending:
        log(f"WARNING: still pending after deadline: {pending}")

    # 从缓存重建 meta
    for canon in list(SECTORS.keys()):
        if cache_exists(f"kline_{canon}.parquet") and cache_exists(f"cons_{canon}.json"):
            dfk = pd.read_parquet(cpath(f"kline_{canon}.parquet"))
            meta["hist_range"][canon] = [int(dfk["date"].min()), int(dfk["date"].max())]
            meta["cons_count"][canon] = len(json.load(open(cpath(f"cons_{canon}.json"), encoding="utf-8")))
    # matched 信息: 若本轮未拉到(缓存来自历史), 从 kline 缓存推断 em 名称留空
    for canon in meta["cons_count"]:
        if canon not in meta["matched"] and canon not in meta["failed"]:
            meta["matched"][canon] = {"em_name": "(cache)", "bk": "", "src": ""}

    # 指数: 若已有 market_daily.parquet 且 3 个指数列齐全则跳过
    mkt_ok = False
    if os.path.exists(MARKET_DAILY_PATH):
        try:
            mm = pd.read_parquet(MARKET_DAILY_PATH)
            mkt_ok = all(f"{k}_close" in mm.columns for k in INDEXES) and len(mm) > 2000
        except Exception:
            mkt_ok = False
    idx_frames = []
    if not mkt_ok:
        for key, secid in INDEXES.items():
            cache = cpath(f"idx_{key}.parquet")
            if not cache_exists(f"idx_{key}.parquet"):
                try:
                    log(f"fetching index {key} ...")
                    dfk = fetch_kline(secid, "1")
                    time.sleep(random.uniform(3.0, 6.0))
                    if dfk is not None:
                        dfk.to_parquet(cache, index=False)
                except Exception as e:
                    log(f"index {key} failed: {str(e)[:80]}")
            if cache_exists(f"idx_{key}.parquet"):
                dfk = pd.read_parquet(cache)
                dfk = dfk.rename(columns={c: f"{key}_{c}" for c in dfk.columns if c != "date"})
                idx_frames.append(dfk)
                meta["hist_range"][key] = [int(dfk["date"].min()), int(dfk["date"].max())]
        if len(idx_frames) == len(INDEXES):
            mkt = idx_frames[0]
            for f in idx_frames[1:]:
                mkt = mkt.merge(f, on="date", how="outer")
            mkt = mkt.sort_values("date").reset_index(drop=True)
            mkt.to_parquet(MARKET_DAILY_PATH, index=False)
            log(f"wrote {MARKET_DAILY_PATH}: {len(mkt)} rows")
    else:
        mm = pd.read_parquet(MARKET_DAILY_PATH)
        for key in INDEXES:
            meta["hist_range"][key] = [int(mm["date"].min()), int(mm["date"].max())]
        log("market_daily.parquet already complete, skipped index fetch")

    # 汇总产物
    frames = []
    cons_dict = {}
    for canon in list(SECTORS.keys()):
        if cache_exists(f"kline_{canon}.parquet") and cache_exists(f"cons_{canon}.json"):
            dfk = pd.read_parquet(cpath(f"kline_{canon}.parquet"))
            dfk["sector"] = canon
            info = meta["matched"].get(canon, {})
            dfk["em_name"], dfk["bk_code"], dfk["src"] = \
                info.get("em_name", ""), info.get("bk", ""), info.get("src", "")
            frames.append(dfk)
            cons_dict[canon] = json.load(open(cpath(f"cons_{canon}.json"), encoding="utf-8"))
    if frames:
        sec = pd.concat(frames, ignore_index=True)
        sec.to_parquet(SECTOR_DAILY_PATH, index=False)
        log(f"wrote {SECTOR_DAILY_PATH}: {len(sec)} rows, {sec['sector'].nunique()} sectors")
    with open(CONS_PATH, "w", encoding="utf-8") as f:
        json.dump(cons_dict, f, ensure_ascii=False, indent=1)
    meta["failed"] = [c for c in SECTORS if c not in cons_dict]
    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=1)
    log(f"wrote {CONS_PATH}; matched={len(cons_dict)}, failed={meta['failed']}")
    log(f"done in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
