"""backfill_pipeline.py — Corvus-C2 公告回填 + 反应标签（监控机可移植自包含版）。

数据源：
- 公告：akshare 巨潮个股接口（实测 2026-09-27：列=代码/简称/公告标题/公告时间/公告链接，
  公告时间为日期粒度；多年窗口单请求可用，7 年窗口 600519 实测 472 行/2.6s；
  北交所可用；退市股 KeyError 或 0 行 -> 如实记录，见 DEVIATIONS.md 存活者偏差）。
- 行情/ universe：stockdb HTTP（字段与通配语义见 _stockdb.py 头注释）。

用法：
  python backfill_pipeline.py --phase all --start 20190101 --end 20251231 --out out
  python backfill_pipeline.py --phase label --out out          # 只对已有 raw 重算标签
  python backfill_pipeline.py --spotcheck 5 --out out          # 标签抽查（重算对拍）

纪律：
- 限速（--request-interval，默认 0.7s；实测 burst 7.2 req/s 不封，但保守限速）；
- 断点续跑（progress.json 按 code|window 记录）；
- 幂等（去重键 code+announce_date+title_hash；raw 追加 + 载入去重）；
- 逐票处理，峰值内存 <4GB（--profile-mem 用 tracemalloc 报告）。
"""
from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import hashlib
import io
import json
import os
import random
import sys
import time
import tracemalloc
from pathlib import Path

import pandas as pd

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from _stockdb import (DEFAULT_BASE, load_daily, make_opener,
                      universe_via_keys, universe_via_parts)
from _taxonomy import map_canonical, tier_of

LABEL_VERSION_V1 = "CORVUS_LABEL_V1_PRE_CORP_FIX"
LABEL_VERSION = "CORVUS_LABEL_V2_CORP_STARTDAY_FIX"
ANCHOR_CODES = ("000001", "600000", "300750", "688981", "601398", "830799")


# ---------------------------------------------------------------- 基础工具
def log(msg: str):
    print(f"[{dt.datetime.now():%H:%M:%S}] {msg}", flush=True)


def sha(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()


def atomic_write_json(path: Path, obj):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    tmp.replace(path)


def ymd(s: str) -> str:
    return s.replace("-", "")


def windows_of(start: str, end: str, years: int) -> list[tuple[str, str]]:
    """[start,end] 按自然年分块，每块宽度 = years 年，连续覆盖（左闭右闭）。

    例：start=20190101, end=20251231, years=3 ->
        [(20190101,20211231),(20220101,20241231),(20250101,20251231)]
    注：旧实现为“宽 1 年、步长 years”，会漏年（只取 2019/2022/2025），已修正。
    """
    s, e = int(start), int(end)
    out = []
    y = s // 10000
    while True:
        a = max(s, y * 10000 + 101)
        if a > e:
            break
        b = min(e, (y + years - 1) * 10000 + 1231)
        if a <= b:
            out.append((str(a), str(b)))
        y += years
    return out


# ---------------------------------------------------------------- 巨潮拉取
def fetch_window(code: str, w0: str, w1: str, retries: int = 3):
    """单票单窗口。返回 (rows, error)。akshare 的 tqdm 进度条抑制到内存。"""
    import akshare as ak
    last_err = None
    for attempt in range(retries):
        try:
            with contextlib.redirect_stderr(io.StringIO()):
                df = ak.stock_zh_a_disclosure_report_cninfo(
                    symbol=code, market="沪深京",
                    start_date=w0, end_date=w1, keyword="")
            if df is None or df.empty:
                return [], None
            need = {"代码", "简称", "公告标题", "公告时间", "公告链接"}
            missing = need - set(df.columns)
            if missing:
                return [], f"列结构变化: 缺 {sorted(missing)}, 实际 {list(df.columns)}"
            recs = []
            for _, r in df.iterrows():
                title = str(r["公告标题"]).strip()
                d = ymd(str(r["公告时间"])[:10])
                th = sha(title)[:12]
                recs.append({
                    "code": str(r["代码"]).zfill(6),
                    "name": str(r["简称"]),
                    "title": title,
                    "announce_date": d,
                    "announce_time": d,          # 巨潮个股接口仅日期粒度（DEVIATION D1）
                    "raw_url": str(r["公告链接"]),
                    "title_hash": th,
                    "dedup_key": f"{str(r['代码']).zfill(6)}|{d}|{th}",
                    "window": f"{w0}-{w1}",
                })
            return recs, None
        except KeyError as e:
            return [], f"PERMANENT: secid 映射缺失({e})，退市/北交老代码可能如此"
        except Exception as e:                       # 网络类错误退避重试
            last_err = f"{type(e).__name__}: {str(e)[:120]}"
            time.sleep(2 * (attempt + 1))
    return [], last_err


def phase_fetch(args, progress: dict, raw_path: Path):
    # universe
    uni_path = Path(args.universe_cache)
    if args.refresh_universe or not uni_path.exists():
        log("universe: stockdb HTTP keys 流式取码 ...")
        t0 = time.time()
        try:
            codes = universe_via_keys(args.base)
            src = "stockdb_http_keys"
        except Exception as e:
            root = os.environ.get("STOCKDB_ROOT", "")
            if not root:
                raise RuntimeError(f"HTTP keys 失败({e}) 且未设 STOCKDB_ROOT，无降级路径")
            log(f"HTTP keys 失败({e})，降级 STOCKDB_ROOT={root}")
            codes = universe_via_parts(root)
            src = f"stockdb_parts:{root}"
        uni_path.write_text("\n".join(codes), encoding="utf-8")
        log(f"universe: {len(codes)} codes via {src} ({time.time()-t0:.0f}s)")
    else:
        codes = [l.strip() for l in uni_path.read_text(encoding="utf-8").splitlines() if l.strip()]
        log(f"universe: {len(codes)} codes (cache {uni_path})")
    if args.codes:
        keep = set(args.codes.split(","))
        codes = [c for c in codes if c in keep]
    if args.limit:
        codes = codes[: args.limit]
    log(f"本次处理 {len(codes)} 只票")

    wins = windows_of(args.start, args.end, args.window_years)
    log(f"窗口: {wins[0]} ... {wins[-1]}  共{len(wins)}个/票")

    done = progress.setdefault("done", {})
    errors = progress.setdefault("errors", {})
    raw_f = raw_path.open("a", encoding="utf-8")
    n_req = 0
    t_start = time.time()
    try:
        for i, code in enumerate(codes):
            for (w0, w1) in wins:
                key = f"{code}|{w0}|{w1}"
                if key in done:
                    continue
                recs, err = fetch_window(code, w0, w1)
                n_req += 1
                if err and err.startswith("PERMANENT"):
                    errors[key] = err
                    done[key] = {"rows": 0, "ts": time.time(), "permanent_error": True}
                elif err:
                    errors[key] = err
                    continue                      # 不进 done，下次重跑重试
                else:
                    for r in recs:
                        raw_f.write(json.dumps(r, ensure_ascii=False) + "\n")
                    raw_f.flush()
                    done[key] = {"rows": len(recs), "ts": time.time()}
                time.sleep(args.request_interval)
            if (i + 1) % 50 == 0:
                atomic_write_json(progress_path(args.out), progress)
                el = time.time() - t_start
                log(f"进度 {i+1}/{len(codes)} 票, 请求 {n_req}, "
                    f"{n_req/el:.2f} req/s, 累计错误 {len(errors)}")
    finally:
        raw_f.close()
    atomic_write_json(progress_path(args.out), progress)
    el = time.time() - t_start
    log(f"fetch 完成: {n_req} 请求, {el:.0f}s, {n_req/el:.2f} req/s, 错误 {len(errors)} 条")
    return codes


# ---------------------------------------------------------------- 日历 / 基准
def load_calendar(args, opener) -> list[int]:
    p = Path(args.out) / "calendar.json"
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    dates = set()
    for c in ANCHOR_CODES:
        try:
            df = load_daily(opener, args.base, c)
            if len(df):
                dates.update(int(x) for x in df["date"])
        except Exception as e:
            log(f"calendar anchor {c} 失败: {e}")
    if not dates:
        raise RuntimeError("交易日历构建失败（anchor 全失败）")
    cal = sorted(dates)
    atomic_write_json(p, cal)
    return cal


def load_bench(args):
    """沪深300 日线（新浪，缓存）。失败返回 None（abnormal 列留 NULL，不编造）。"""
    p = Path(args.out) / "bench_000300.parquet"
    if p.exists():
        return pd.read_parquet(p)
    try:
        import akshare as ak
        with contextlib.redirect_stderr(io.StringIO()):
            df = ak.stock_zh_index_daily(symbol="sh000300")
        df["date"] = df["date"].astype(str).str.replace("-", "").astype(int)
        df = df[["date", "open", "close"]].sort_values("date")
        df.to_parquet(p, index=False)
        return df
    except Exception as e:
        log(f"WARN: 基准指数拉取失败，abnormal 列将留 NULL: {e}")
        return None


# ---------------------------------------------------------------- 标签计算
def resolve_label_start(announce_date: int, cal: list[int], assume_premarket: bool):
    """corvus 规格第4节（available_at 粒度退化到日，见 DEVIATION D1）。
    默认（盘后假设）：label_start = 第一个 > announce_date 的交易日。
    --assume-premarket：announce_date 为交易日则当日开盘，否则下一交易日开盘。"""
    pos = None
    for d in cal:
        if d > announce_date or (assume_premarket and d == announce_date):
            pos = d
            break
    if pos is None:
        return None, "NO_FUTURE_TRADE_DAY"
    if assume_premarket and pos == announce_date:
        return pos, "PREMARKET_OPEN"
    basis = "POST_CLOSE_NEXT_DAY" if pos == _next_natural_day(announce_date) \
        else "WEEKEND_NEXT_DAY"
    return pos, basis


def _next_natural_day(d0: int) -> int:
    y, m, dd = d0 // 10000, d0 // 100 % 100, d0 % 100
    nxt = (dt.date(y, m, dd) + dt.timedelta(days=1))
    return nxt.year * 10000 + nxt.month * 100 + nxt.day


def compute_cluster_labels(g: pd.DataFrame, daily: pd.DataFrame, cal: list[int],
                           bench, max_cal: int, assume_premarket: bool = False):
    """单票单簇（(code, announce_date) 分组）标签。corvus 口径见 DEVIATIONS D2-D4。"""
    adate = int(g["announce_date"].iloc[0])
    start, basis = resolve_label_start(adate, cal, assume_premarket)
    out = {"event_cluster_id": g["event_cluster_id"].iloc[0],
           "label_start": start, "label_start_basis": basis,
           "n_announcements_in_cluster": int(len(g)),
           "label_version": LABEL_VERSION}
    if start is None:
        out["label_status"] = "NO_FUTURE_TRADE_DAY"
        return out
    if daily.empty:
        out["label_status"] = "NO_PRICE_DATA"
        return out
    dates = [int(x) for x in daily["date"]]
    pos = {d: i for i, d in enumerate(dates)}
    if start not in pos:
        # 个股停牌（日历有、个股无）-> 顺延到复牌首日（RESUME_FIRST_TRADE 语义）
        fut = [d for d in dates if d > start]
        if not fut:
            out["label_status"] = "NO_FUTURE_TRADE_DAY"
            return out
        start = fut[0]
        basis = "RESUME_FIRST_TRADE"
        out["label_start"] = start
        out["label_start_basis"] = basis
    idx = pos[start]
    row = daily.iloc[idx]
    entry = float(row["open"])
    pc = float(row["pre_close"]) if not pd.isna(row.get("pre_close")) else None
    out["entry_price"] = entry
    out["gap_return"] = (entry / pc - 1) if pc else None

    # 除权守卫（照抄 corvus price_labels.daily_factor）
    # FIX(corp_start_day): 守卫窗口必须包含 label_start 当天(idx)。
    # 原实现只查 idx+1..idx+5，若除权/除息发生在 T 当天，首段收益链
    # (ret_1d_pc 等) 会直接吞下除权跳空，制造虚假 materiality。
    n = len(daily)
    corp = set()
    for k in range(idx, min(n, idx + 6)):   # 含 T：idx .. idx+5
        rk = daily.iloc[k]
        pc_, oc_ = rk.get("pre_close"), float(rk["open"])
        pch = rk.get("pct_chg")
        if pc_ and abs(oc_ / float(pc_) - 1) > 0.115 and pch is not None \
           and not pd.isna(pch) and abs(float(pch)) < 9.5:
            corp.add(k)

    def factor(k):
        rk = daily.iloc[k]
        pch = rk.get("pct_chg")
        if pch is not None and not pd.isna(pch):
            return 1.0 + float(pch) / 100.0
        if k == idx and pc:
            # 首日前复权因子缺失时，用 收/前收 锚定（前收口径基准）
            return float(rk["close"]) / pc
        if k in corp:
            return 1.0
        return float(rk["close"]) / float(daily.iloc[k - 1]["close"])

    # vs 前收口径（任务口径）：label_start 日起 h 个交易日收益链
    # 首因子含 (1+pct_chg) → 结果 = close(idx+h-1)/pre_close(idx) - 1（跨除权不断尺度）
    for h in (1, 3, 5):
        j = idx + h - 1
        if j >= n:
            out[f"ret_{h}d_pc"] = None
            continue
        chain = 1.0
        for k in range(idx, j + 1):
            chain *= factor(k)
        out[f"ret_{h}d_pc"] = float(chain - 1)
    # corvus entry 口径：ret_h = close(idx)/entry * chain(idx+1..idx+h) - 1
    for h in (1, 3, 5):
        j = idx + h
        if j >= n:
            out[f"ret_{h}d"] = None
            out[f"horizon_complete"] = False
            continue
        chain = 1.0
        for k in range(idx + 1, j + 1):
            chain *= factor(k)
        out[f"ret_{h}d"] = float(float(daily.iloc[idx]["close"]) / entry * chain - 1)
        if any(x in corp for x in range(idx, j + 1)):
            out[f"mfe_{h}d"] = None
            out[f"mae_{h}d"] = None
        else:
            seg = daily.iloc[idx: j + 1]
            out[f"mfe_{h}d"] = float(seg["high"].max() / entry - 1)
            out[f"mae_{h}d"] = float(seg["low"].min() / entry - 1)

    # ---- corp 审计字段（显式输出，便于量化除权污染）----
    # tag 为 EX_RIGHTS_SUSPECTED（检测为 open/前收跳空 >11.5% 且 pct_chg 正常，
    # 无法进一步区分分红/送转，故为“疑似”）。守卫已把除权日中性化(factor=1.0)，
    # 标签数值已校正；但 canonical 契约要求：除权窗口样本 label_valid=False、
    # reason=CORPORATE_ACTION_WINDOW、training_eligible=False，由训练/生产侧统一按
    # label_valid 过滤，不再依赖各脚本各自硬编码排除（见 canonicalize_dataset.py）。
    out["corp_action_on_label_start"] = bool(idx in corp)
    out["corp_action_within_horizon"] = bool(len(corp) > 0)
    out["corp_action_type"] = ("EX_RIGHTS_SUSPECTED" if corp else None)
    out["corp_action_date"] = (
        ",".join(str(dates[k]) for k in sorted(corp)) if corp else None)
    if corp:
        out["label_valid"] = False
        out["label_invalid_reason"] = "CORPORATE_ACTION_WINDOW"
        out["training_eligible"] = False
    else:
        out["label_valid"] = True
        out["label_invalid_reason"] = None
        out["training_eligible"] = True

    # 波动基线：前 250 交易日 pct_chg std（≥120 个有效值）
    prev = daily.iloc[max(0, idx - 250): idx]
    rets = pd.to_numeric(prev.get("pct_chg"), errors="coerce").dropna() / 100.0 \
        if "pct_chg" in prev.columns else pd.Series(dtype=float)
    if len(rets) >= 120:
        sig = float(rets.std())
        out["sigma250"] = sig
        mats = {}
        for h in (1, 3, 5):
            r = out.get(f"ret_{h}d_pc")
            mats[h] = bool(abs(r) > 2 * sig) if (r is not None and sig > 0) else None
            out[f"mat_{h}d"] = mats[h]
        known = [v for v in mats.values() if v is not None]
        out["materiality"] = any(known) if known else None
    else:
        out["sigma250"] = None
        for h in (1, 3, 5):
            out[f"mat_{h}d"] = None
        out["materiality"] = None

    r3 = out.get("ret_3d_pc")
    out["direction_3d"] = ("UP" if r3 > 0.001 else "DOWN" if r3 < -0.001 else "FLAT") \
        if r3 is not None else None

    # 量能/波动结构（corvus 第5节）
    prev20 = daily.iloc[max(0, idx - 20): idx]
    if len(prev20) >= 5 and "volume" in daily.columns:
        vm = float(pd.to_numeric(prev20["volume"], errors="coerce").mean())
        out["volume_shock"] = float(row["volume"]) / vm if vm else None
    base = rets.iloc[-20:] if len(rets) >= 10 else None
    seg5 = daily.iloc[idx: min(n, idx + 6)]
    if base is not None and len(seg5) >= 2 and "pct_chg" in seg5.columns:
        bv = float(base.std())
        nv = pd.to_numeric(seg5["pct_chg"], errors="coerce").dropna()
        nv = float(nv.std()) if len(nv) >= 2 else None
        out["volatility_expansion"] = (nv / bv) if (bv and nv) else None

    # 大盘异常收益（bench=沪深300；拿不到留 NULL）
    if bench is not None:
        b = bench.set_index("date")["close"]
        def bclose(d):
            return float(b.loc[d]) if d in b.index else None
        b0 = bclose(start)
        for h in (1, 3, 5):
            j = idx + h - 1
            if j >= n:
                out[f"abnormal_{h}d"] = None
                continue
            bh = bclose(int(daily.iloc[j]["date"]))
            pcv = float(daily.iloc[idx]["pre_close"]) \
                if not pd.isna(daily.iloc[idx].get("pre_close")) else None
            if b0 and bh and pcv and out.get(f"ret_{h}d_pc") is not None:
                out[f"abnormal_{h}d"] = float(out[f"ret_{h}d_pc"] - (bh / b0 - 1))
            else:
                out[f"abnormal_{h}d"] = None

    out["label_status"] = "OK" if out.get("ret_5d_pc") is not None else "CENSORED_TAIL"
    return out


def add_overlap(df: pd.DataFrame, cal: list[int]) -> pd.DataFrame:
    """同票标签窗口 ±5 交易日重叠计数（corvus 规格第7节）。
    簇级计数：同簇多公告只算一次，且不把本簇计入。"""
    pos = {d: i for i, d in enumerate(cal)}
    cl = (df[["event_cluster_id", "code", "label_start"]]
          .drop_duplicates("event_cluster_id"))
    counts = {}
    for code, g in cl.groupby("code", sort=False):
        idxs = [pos.get(int(d)) if pd.notna(d) else None for d in g["label_start"]]
        arr = [i for i in idxs if i is not None]
        for cid, ci in zip(g["event_cluster_id"], idxs):
            counts[cid] = (sum(1 for cj in arr if abs(cj - ci) <= 5) - 1) \
                if ci is not None else 0
    df["overlap_count"] = df["event_cluster_id"].map(counts).fillna(0).astype(int)
    df["attribution_quality"] = df["overlap_count"].map(
        lambda k: "HEAVY" if k >= 3 else ("MODERATE" if k >= 1 else "CLEAN"))
    return df


# ---------------------------------------------------------------- label 主流程
def daily_cache_path(out: str, code: str) -> Path:
    return Path(out) / "daily_cache" / f"{code}.parquet"


def get_daily_cached(args, opener, code: str) -> pd.DataFrame:
    p = daily_cache_path(args.out, code)
    if p.exists():
        return pd.read_parquet(p)
    df = load_daily(opener, args.base, code)
    if len(df):
        p.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(p, index=False)
    return df


def phase_label(args):
    out = Path(args.out)
    raw_path = out / "raw_announcements.jsonl"
    if not raw_path.exists():
        raise SystemExit(f"无 {raw_path}，先跑 --phase fetch")
    log("载入 raw + 幂等去重 ...")
    seen, recs = set(), []
    with raw_path.open(encoding="utf-8") as f:
        for line in f:
            try:
                r = json.loads(line)
            except Exception:
                continue
            if r["dedup_key"] in seen:
                continue
            seen.add(r["dedup_key"])
            recs.append(r)
    df = pd.DataFrame(recs)
    log(f"raw 去重后 {len(df)} 条（去重键 code|date|title_hash）")

    cats = df["title"].map(map_canonical)
    df["category"] = [c for c, _ in cats]
    df["tier"] = df["category"].map(tier_of)
    df.loc[df["category"] == "UNMAPPED", "tier"] = "P2"
    df["matched_keyword"] = [k for _, k in cats]
    df["event_cluster_id"] = [
        sha(f"{c}|{d}")[:12] for c, d in zip(df["code"], df["announce_date"])]

    opener = make_opener()
    cal = load_calendar(args, opener)
    max_cal = cal[-1]
    bench = load_bench(args)
    log(f"calendar {cal[0]}..{max_cal} ({len(cal)} 日), bench={'OK' if bench is not None else 'NULL'}")

    label_rows = []
    codes = sorted(df["code"].unique())
    t0 = time.time()
    for i, code in enumerate(codes):
        gd = df[df["code"] == code]
        daily = get_daily_cached(args, opener, code)
        for _, g in gd.groupby("announce_date", sort=True):
            try:
                label_rows.append(compute_cluster_labels(g, daily, cal, bench, max_cal,
                                                         args.assume_premarket))
            except Exception as e:
                label_rows.append({"event_cluster_id": g["event_cluster_id"].iloc[0],
                                   "label_status": f"FAILED:{type(e).__name__}",
                                   "n_announcements_in_cluster": int(len(g)),
                                   "label_version": LABEL_VERSION})
        if (i + 1) % 500 == 0:
            log(f"label {i+1}/{len(codes)} codes ({time.time()-t0:.0f}s)")
    log(f"label 完成: {len(label_rows)} 簇 / {len(codes)} 票 ({time.time()-t0:.0f}s)")

    lab = pd.DataFrame(label_rows)
    ev = df.merge(lab, on="event_cluster_id", how="left")
    ev = add_overlap(ev, cal)

    ev.to_parquet(out / "events_SUPERSEDED_V2_labeled.parquet", index=False)
    stats = build_stats(ev, progress_path(args.out))
    atomic_write_json(out / "source_stats.json", stats)
    log(f"events_SUPERSEDED_V2_labeled.parquet: {len(ev)} 行 -> {out/'events_SUPERSEDED_V2_labeled.parquet'}")

    write_completeness_checklist(ev, out, args.completeness_sample)
    print_stats_summary(stats)


def build_stats(ev: pd.DataFrame, prog_path: Path) -> dict:
    prog = json.loads(prog_path.read_text(encoding="utf-8")) if prog_path.exists() else {}
    err = prog.get("errors", {})
    permanent = sum(1 for v in err.values() if str(v).startswith("PERMANENT"))
    labeled = ev[ev["label_status"] == "OK"]

    def vc(col, frame=None):
        s = (frame if frame is not None else ev)[col].value_counts(dropna=False)
        return {str(k): int(v) for k, v in s.items()}

    by_year = labeled.assign(y=labeled["announce_date"].astype(str).str[:4])["y"] \
                     .value_counts().sort_index()
    return {
        "label_version": LABEL_VERSION,
        "announcement_rows": int(len(ev)),
        "clusters": int(ev["event_cluster_id"].nunique()),
        "codes_with_events": int(ev["code"].nunique()),
        "fetch_windows_total": int(len(prog.get("done", {}))),
        "fetch_windows_permanent_error": permanent,
        "tier_counts": vc("tier"),
        "category_counts_top20": vc("category"),
        "label_status_counts": vc("label_status"),
        "attribution_counts": vc("attribution_quality"),
        "materiality_counts": vc("materiality", labeled),
        "direction_counts": vc("direction_3d", labeled),
        "events_by_year": {str(k): int(v) for k, v in by_year.items()},
        "trade_days_covered": int(labeled["label_start"].nunique()),
        "generated_at": dt.datetime.now().isoformat(timespec="seconds"),
    }


def print_stats_summary(s: dict):
    log("=== 源完整性统计 ===")
    for k in ("announcement_rows", "clusters", "codes_with_events",
              "fetch_windows_total", "fetch_windows_permanent_error",
              "trade_days_covered"):
        log(f"  {k}: {s[k]}")
    log(f"  tier: {s['tier_counts']}")
    log(f"  label_status: {s['label_status_counts']}")
    log(f"  materiality: {s['materiality_counts']}")
    log(f"  direction: {s['direction_counts']}")
    log(f"  分年: {s['events_by_year']}")


def write_completeness_checklist(ev: pd.DataFrame, out: Path, n_sample: int):
    ok = ev[ev["label_status"] == "OK"]
    if ok.empty:
        return
    months = ok.assign(m=ok["announce_date"].astype(str).str[:6])
    pick = months.groupby(["code", "m"]).size().reset_index(name="n")
    sample = pick.sample(min(n_sample, len(pick)), random_state=42)
    lines = ["# C2 完整性核对清单（人工 vs 巨潮官网）", "",
             "核对方法：打开 http://www.cninfo.com.cn/new/commonUrl/pageOfSearch?url=disclosure/list/search",
             "按 股票代码+关键词空+起止日期=当月 查询，比对条数与标题；差异记录在下表。", ""]
    for _, r in sample.iterrows():
        sub = months[(months["code"] == r["code"]) & (months["m"] == r["m"])]
        lines.append(f"## {r['code']} {sub['name'].iloc[0] if 'name' in sub else ''} "
                     f"{r['m']}  本包 {r['n']} 条")
        for _, e in sub.head(20).iterrows():
            lines.append(f"- {e['announce_date']} {e['title']}  {e['raw_url']}")
        lines.append("")
    (out / "completeness_checklist.md").write_text("\n".join(lines), encoding="utf-8")
    log(f"核对清单: {out/'completeness_checklist.md'} ({len(sample)} 票×月)")


# ---------------------------------------------------------------- 标签抽查
def spotcheck(args):
    out = Path(args.out)
    ev = pd.read_parquet(out / "events_SUPERSEDED_V2_labeled.parquet")
    ok = ev[ev["label_status"] == "OK"]
    sample = ok.sample(min(args.spotcheck, len(ok)), random_state=7)
    opener = make_opener()
    print("=== 标签抽查（独立重算 vs events_SUPERSEDED_V2_labeled.parquet） ===")
    for _, e in sample.iterrows():
        df = load_daily(opener, args.base, e["code"])
        df = df[df["date"].astype(int) >= int(e["announce_date"]) - 10].reset_index(drop=True)
        idx = df.index[df["date"].astype(int) == int(e["label_start"])]
        if not len(idx):
            print(f"{e['event_cluster_id']} {e['code']}: label_start 无 bar，跳过")
            continue
        idx = int(idx[0])
        chain = 1.0
        for k in range(idx, min(idx + 3, len(df))):
            chain *= 1 + float(df.iloc[k]["pct_chg"]) / 100.0
        r3 = chain - 1
        r1 = float(df.iloc[idx]["pct_chg"]) / 100.0
        print(f"  {e['code']} {e['announce_date']} '{e['title'][:30]}'")
        print(f"    重算: ret_1d_pc={r1:+.4f} ret_3d_pc={r3:+.4f} | "
              f"落盘: ret_1d_pc={e['ret_1d_pc']:+.4f} ret_3d_pc={e['ret_3d_pc']:+.4f} | "
              f"match={abs(r1-e['ret_1d_pc'])<1e-9 and abs(r3-e['ret_3d_pc'])<1e-9}")


# ---------------------------------------------------------------- main
def progress_path(out: str) -> Path:
    return Path(out) / "progress.json"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--phase", choices=["fetch", "label", "all"], default="all")
    ap.add_argument("--start", default="20190101")
    ap.add_argument("--end", default="20251231")
    ap.add_argument("--window-years", type=int, default=3)
    ap.add_argument("--out", default="out")
    ap.add_argument("--base", default=DEFAULT_BASE, help="STOCKDB_BASE 覆盖")
    ap.add_argument("--request-interval", type=float, default=0.7)
    ap.add_argument("--limit", type=int, default=0, help="只跑前 N 票（开发验证）")
    ap.add_argument("--codes", default="", help="只跑这些代码,逗号分隔")
    ap.add_argument("--universe-cache", default=os.path.join("out", "universe.txt"))
    ap.add_argument("--refresh-universe", action="store_true")
    ap.add_argument("--assume-premarket", action="store_true",
                    help="公告按盘前可见处理（默认盘后，见 DEVIATIONS D1）")
    ap.add_argument("--completeness-sample", type=int, default=5)
    ap.add_argument("--spotcheck", type=int, default=0, help="抽查 N 例标签后退出")
    ap.add_argument("--profile-mem", action="store_true")
    args = ap.parse_args()

    Path(args.out).mkdir(parents=True, exist_ok=True)

    if args.profile_mem:
        tracemalloc.start()

    if args.spotcheck:
        spotcheck(args)
        return

    prog_p = progress_path(args.out)
    progress = json.loads(prog_p.read_text(encoding="utf-8")) if prog_p.exists() \
        else {"version": 1, "done": {}, "errors": {}}

    if args.phase in ("fetch", "all"):
        phase_fetch(args, progress, Path(args.out) / "raw_announcements.jsonl")
    if args.phase in ("label", "all"):
        phase_label(args)

    if args.profile_mem:
        cur, peak = tracemalloc.get_traced_memory()
        log(f"tracemalloc: current={cur/2**20:.0f}MB peak={peak/2**20:.0f}MB")


if __name__ == "__main__":
    main()
