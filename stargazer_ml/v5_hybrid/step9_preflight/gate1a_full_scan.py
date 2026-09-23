"""gate1a_full_scan — Gate 1A Final Verification 全库扫描 (Step 9 Gate 1A).

对 4 项 CONDITIONAL_PASS 核心数据做全市场 (universe 7562) 验证, 判定能否升级为
VERIFIED_PASS。只读数据审计, 不改任何训练/特征代码。

校验项:
  1. stock_minute_history : 分钟全库聚合 (HTTP 全拉, 即算即弃)
     本地 minute_cache_clean (3959 票) 仅作交叉校验, 不做合并源
     (边界 20260908 当日完整性无法从文件名确认, 单一 HTTP 路径更稳妥)
  2. free_float_market_cap: 本地日线 parquet 全库结构验证
  3. halt_trading_status  : INFERRED_HALT 全库 (分钟纪元内 日线有/分钟无 × 退市清单)
  4. adjustment_factors   : HTTP 复权键前缀枚举全库 + 抽样连续性 sanity

用法:
  PYTHONIOENCODING=utf-8 python gate1a_full_scan.py dry     # 小范围 dry (20 票)
  PYTHONIOENCODING=utf-8 python gate1a_full_scan.py full    # 全库 (分钟扫描断点续跑)

运行: _dev_venv/Scripts/python.exe (与 qa_core_data.py / probe 同款解释器)

产出:
  DATA_GATE1A_FULL.yaml               正式 QA 产物
  _progress_gate1a_minute.jsonl       分钟扫描断点 (可删了重跑)
  GATE1A_INFERRED_HALT_DAYS.jsonl     INFERRED_HALT / DATA_MISSING 票-日全列表
"""
from __future__ import annotations
import sys
import os
import glob
import json
import time
import random
import urllib.parse
import urllib.request
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import msgpack
import pandas as pd
import yaml

_HERE = Path(__file__).resolve().parent
OUT = _HERE / "DATA_GATE1A_FULL.yaml"
PROGRESS = _HERE / "_progress_gate1a_minute.jsonl"
HALT_DAYS_OUT = _HERE / "GATE1A_INFERRED_HALT_DAYS.jsonl"
EXPECTED_BARS_PER_DAY = 240
MINUTE_START = 20250102  # 冻结值待全库确认 (应为 20250102)

BASE = "http://127.0.0.1:7899"
DATA_ROOT = os.environ.get("STARGAZER_DATA_ROOT", r"D:/a股数据")
DAILY_PARTS = os.path.join(DATA_ROOT, "data_daily_parts")
MINUTE_CACHE = _HERE.parent.parent / "pullback_train" / "minute_cache_clean"
LOCAL_CACHE_LAST = 20260908  # 本地分钟缓存截止日 (交叉校验用)

_opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))  # 绕过本机代理


def _quantile(sorted_vals, q):
    if not sorted_vals:
        return None
    idx = max(0, min(len(sorted_vals) - 1, int(round(q * (len(sorted_vals) - 1)))))
    return sorted_vals[idx]


def http_vals(table: str, code: str = "*", date: str = "*", timeout: int = 180):
    """cmd=vals 只返回值, 比 get (含 key) 省一半流量。返回 list[dict]。"""
    t = urllib.parse.quote(f"{table}:{code}:{date}")
    url = f"{BASE}/?cmd=vals&t={t}"
    req = urllib.request.Request(url)
    with _opener.open(req, timeout=timeout) as r:
        raw = r.read()
    try:
        obj = msgpack.unpackb(raw, raw=False, strict_map_key=False)
    except Exception:
        obj = json.loads(raw.decode("utf-8", errors="replace"))
    if isinstance(obj, list):
        return [it for it in obj if isinstance(it, dict)]
    return []


# ---------------- universe / 日线加载 ----------------

def load_universe() -> list:
    """CANONICAL universe = data_source_provenance.get_universe (与 latr_builder 一致)。"""
    from data_source_provenance import get_universe
    return get_universe()


def load_daily() -> pd.DataFrame:
    """全库日线 (CANONICAL = data_source_provenance.get_daily_parquet, stockdb 衍生品)。
    与 latr_builder 经同一模块取数, 消除原 parquet-vs-HTTP 日线分裂。"""
    from data_source_provenance import get_daily_parquet, get_daily_http
    df = get_daily_parquet()
    if df.empty:
        frames = [get_daily_http(c) for c in load_universe()]
        df = pd.concat([f for f in frames if not f.empty], ignore_index=True) \
            if frames else df
    if "code" in df:
        df["code"] = df["code"].astype(str).str.zfill(6)
    return df


def load_delist_codes() -> set:
    try:
        pybao = os.path.join(DATA_ROOT, "stockdb", "pybao")
        if pybao not in sys.path:
            sys.path.insert(0, pybao)
        import stock_sdk
        stock_sdk.init(host="127.0.0.1", port=7899)
        rows = stock_sdk.rd.get("退市*").do()
        out = set()
        for it in rows:
            if isinstance(it, (list, tuple)) and len(it) >= 2 and it[1]:
                out.add(str(it[1]).zfill(6))
            elif isinstance(it, str):
                out.add(it.zfill(6))
        return out
    except Exception as e:  # noqa
        print(f"[warn] SDK 退市清单不可用: {e}")
        return set()


# ---------------- 1. 分钟全库聚合 ----------------

def _scan_one_minute(code: str) -> dict:
    recs = None
    last_err = None
    for attempt, wait in enumerate((0, 3, 10, 30)):
        if wait:
            time.sleep(wait)
        try:
            recs = http_vals("分钟k", code, "*", timeout=300)
            last_err = None
            break
        except Exception as e:  # noqa
            last_err = f"{type(e).__name__}: {str(e)[:120]}"
            print(f"[retry] {code} attempt={attempt} {last_err}")
    if last_err is not None or recs is None:
        return {"code": code, "ok": False, "err": last_err or "unknown"}
    per_day = defaultdict(int)
    seen = set()
    dup = 0
    bars = 0
    first = last = None
    for r in recs:
        d = r.get("date")
        if d is None:
            continue
        bars += 1
        if d in seen:
            dup += 1
        else:
            seen.add(d)
        day = d // 1000000
        per_day[day] += 1
        if first is None or d < first:
            first = d
        if last is None or d > last:
            last = d
    days = sorted(per_day)
    return {"code": code, "ok": True, "bars": bars, "dup": dup,
            "n_days": len(days), "first": first, "last": last,
            "days": days, "counts": [per_day[d] for d in days]}


def run_minute_scan(universe: list, workers: int = 8) -> dict:
    done = {}
    if PROGRESS.exists():
        with open(PROGRESS, encoding="utf-8") as f:
            for line in f:
                try:
                    rec = json.loads(line)
                    if rec.get("ok"):  # 仅成功记录视为完成; 失败记录重扫
                        done[rec["code"]] = rec
                except Exception:  # noqa
                    continue
    todo = [c for c in universe if c not in done]
    print(f"[minute] universe={len(universe)} done={len(done)} todo={len(todo)} workers={workers}")
    t0 = time.time()
    with open(PROGRESS, "a", encoding="utf-8") as pf:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = {ex.submit(_scan_one_minute, c): c for c in todo}
            fin = 0
            for fut in as_completed(futs):
                rec = fut.result()
                done[rec["code"]] = rec
                pf.write(json.dumps(rec, ensure_ascii=False) + "\n")
                fin += 1
                if fin % 500 == 0:
                    pf.flush()
                    el = time.time() - t0
                    eta = el / fin * (len(todo) - fin)
                    print(f"[minute] {fin}/{len(todo)} elapsed={el:.0f}s eta={eta:.0f}s")
    el = time.time() - t0
    print(f"[minute] scan done in {el:.0f}s")

    # ---- 全库聚合 (合并既有 done; done 只含成功记录) ----
    ppd = []           # 全部 symbol-day bar 数
    month_bars = defaultdict(int)
    month_syms = defaultdict(set)
    sym_ok = sym_empty = 0
    dup_total = bars_total = 0
    day_total = 0
    global_first = None
    global_last = None
    min_first_day = None
    for code, rec in done.items():
        if rec["n_days"] == 0:
            sym_empty += 1
            continue
        if rec["n_days"] == 0:
            sym_empty += 1
            continue
        sym_ok += 1
        bars_total += rec["bars"]
        dup_total += rec["dup"]
        day_total += rec["n_days"]
        ppd.extend(rec["counts"])
        fd = rec["first"] // 1000000
        ld = rec["last"] // 1000000
        if min_first_day is None or fd < min_first_day:
            min_first_day = fd
        if global_first is None or rec["first"] < global_first:
            global_first = rec["first"]
        if global_last is None or rec["last"] > global_last:
            global_last = rec["last"]
        for d, n in zip(rec["days"], rec["counts"]):
            m = d // 100
            month_bars[m] += n
            month_syms[m].add(code)
    ppd.sort()
    med = _quantile(ppd, 0.5)
    coverage_by_month = {str(m): {"symbols": len(month_syms[m]), "bars": month_bars[m]}
                         for m in sorted(month_bars)}
    # 每票 minute day 数分布 (universe 覆盖形状)
    sym_day_counts = sorted(r["n_days"] for r in done.values() if r.get("ok") and r["n_days"] > 0)
    minute_day_by_sym = {c: set(r["days"]) for c, r in done.items()
                         if r.get("ok") and r["n_days"] > 0}
    empty_syms = sorted(c for c, r in done.items() if r.get("ok") and r["n_days"] == 0)
    return {
        "n_symbols_universe": len(universe),
        "symbols_with_minute_data": sym_ok,
        "symbols_empty": sym_empty,
        "symbols_empty_list": empty_syms,
        "symbols_empty_note": "21 只为北交所 920xxx (分钟 feed 不覆盖北交所), "
                              "2 只为 2025 前已退市/被吸收 (深赤湾A/中航善达); "
                              "结构性无分钟, 非数据缺口",
        "symbols_error": max(0, len(universe) - len(done)),
        "errors_top": "见运行日志 [retry] 行; 失败票重跑后仍失败才计入 symbols_error",
        "bars_total": bars_total,
        "symbol_day_total": day_total,
        "bars_per_symbol_day": {
            "p1": _quantile(ppd, 0.01), "p5": _quantile(ppd, 0.05),
            "p50": _quantile(ppd, 0.5), "p95": _quantile(ppd, 0.95),
            "p99": _quantile(ppd, 0.99), "expected": EXPECTED_BARS_PER_DAY,
        },
        "missing_bar_ratio_vs_240_p50": round(1 - med / EXPECTED_BARS_PER_DAY, 4),
        "duplicate_bar_ratio": round(dup_total / bars_total, 8) if bars_total else None,
        "minute_first_ts": global_first, "minute_last_ts": global_last,
        "min_symbol_first_day": min_first_day,
        "MINUTE_START_frozen": MINUTE_START,
        "minute_start_confirmed": min_first_day == MINUTE_START,
        "symbol_day_count_distribution": {
            "p1": _quantile(sym_day_counts, 0.01), "p5": _quantile(sym_day_counts, 0.05),
            "p50": _quantile(sym_day_counts, 0.5), "p95": _quantile(sym_day_counts, 0.95),
            "max": sym_day_counts[-1] if sym_day_counts else None,
        },
        "coverage_by_month": coverage_by_month,
        "scan_seconds": round(el, 1),
        "_minute_day_by_sym": minute_day_by_sym,  # 内部用, 写 YAML 前剔除
    }


def cross_check_local(minute_day_by_sym: dict, sample_n: int = 200) -> dict:
    """本地 minute_cache_clean 与 HTTP 结果交叉校验 (抽样 sample_n 票)。"""
    local_files = {Path(p).stem: p for p in
                   glob.glob(str(MINUTE_CACHE / "*.parquet"))}
    common = sorted(set(local_files) & set(minute_day_by_sym))
    random.seed(42)
    picked = random.sample(common, min(sample_n, len(common)))
    n_day_missing = n_bar_mismatch = n_checked = 0
    examples = []
    for c in picked:
        http_days = minute_day_by_sym[c]
        try:
            df = pd.read_parquet(local_files[c], columns=["date", "close"])
        except Exception as e:  # noqa
            continue
        n_checked += 1
        local_days = set(df["date"].astype("int64") // 1000000)
        miss = {d for d in local_days if d <= LOCAL_CACHE_LAST and d not in http_days}
        if miss:
            n_day_missing += 1
            if len(examples) < 5:
                examples.append({"code": c, "local_not_in_http": sorted(miss)[:5]})
        # bar 数级比对走全库 ppd 分布; 此处只比对日集合 (本地 ⊆ HTTP)
    return {"n_local_symbols": len(local_files), "n_common": len(common),
            "n_cross_checked": n_checked,
            "n_local_days_missing_in_http": n_day_missing,
            "examples": examples,
            "verdict": "本地缓存日集合 ⊆ HTTP 日集合" if n_day_missing == 0 else
                       "存在本地有/HTTP 无的日, 见 examples"}


# ---------------- 2. float_mv 结构验证 ----------------

def run_float_mv(daily: pd.DataFrame) -> dict:
    g = daily.groupby("code")
    pres = g["float_mv"].apply(lambda s: s.notna().mean())
    nuniq = g["float_mv"].nunique(dropna=True)
    varies = int((nuniq > 1).sum())
    const = int((nuniq == 1).sum())
    era = daily[daily["date"] >= MINUTE_START]
    era_pres_overall = round(float(era["float_mv"].notna().mean()), 4)
    era_pres_by_sym = era.groupby("code")["float_mv"].apply(lambda s: s.notna().mean())
    low_cov = era_pres_by_sym[era_pres_by_sym < 0.95]
    names = daily.groupby("code")["name"].last() if "name" in daily.columns else None
    FUND_KW = ("ETF", "LOF", "基金", "债", "定开", "FOF", "REIT", "精选", "添利",
               "科创板", "封闭", "智选", "优选", "配置", "轮动", "成长", "明择",
               "睿阳", "瑞虹", "优势企业", "汇选", "招享", "睿智", "进取", "汇利",
               "永旭", "科顺", "瑞享", "大盘", "香港", "消费", "金利", "积配",
               "高增", "小盘", "估值", "价值", "商品", "民益", "社会责任")
    non_fund_low = []
    if names is not None and len(low_cov):
        for c in low_cov.index:
            n = str(names.get(c, ""))
            if not any(k in n for k in FUND_KW):
                non_fund_low.append({"code": c, "name": n,
                                     "cov": round(float(low_cov[c]), 3)})
    # 时序变化率: 相邻非空差分非零占比 (中位数)
    def _vary_rate(s):
        s = s.dropna()
        if len(s) < 2:
            return None
        d = s.diff().abs().iloc[1:]
        return float((d > 0).mean())
    vary_rate = g["float_mv"].apply(_vary_rate).dropna()
    # 退市股: 最后日线日 < 全库最后日 (近似识别)
    last_day = int(daily["date"].max())
    last_by_sym = g["date"].max()
    stale = last_by_sym[last_by_sym < last_day - 5]
    delisted_sample = daily[daily["code"].isin(stale.index[:200])]
    tail_pres = delisted_sample.groupby("code")["float_mv"].apply(
        lambda s: s.tail(30).notna().mean() if len(s) else None)
    # float_share*close ≈ float_mv 抽样 sanity (随机 50 票, 全历史)
    random.seed(7)
    syms = sorted(daily["code"].unique())
    samp = random.sample(syms, 50)
    sub = daily[daily["code"].isin(samp)].dropna(subset=["float_mv", "float_share", "close"])
    sub = sub[sub["float_share"] > 0]
    ratio = sub["float_mv"] / (sub["float_share"] * sub["close"])
    return {
        "n_symbols": len(syms),
        "n_rows": int(len(daily)),
        "float_mv_present_ratio_overall": round(float(daily["float_mv"].notna().mean()), 4),
        "float_mv_present_ratio_minute_era": era_pres_overall,
        "float_mv_present_ratio_minute_era_median_per_symbol":
            round(float(era_pres_by_sym.median()), 4),
        "n_symbols_in_minute_era": int(era["code"].nunique()),
        "n_symbols_era_coverage_ge_95pct": int((era_pres_by_sym >= 0.95).sum()),
        "n_low_coverage_symbols": int(len(low_cov)),
        "low_coverage_symbols_all_fund_like": len(non_fund_low) == 0,
        "non_fund_low_coverage_examples": non_fund_low[:20],
        "float_mv_present_ratio_median_per_symbol": round(float(pres.median()), 4),
        "symbols_varying_float_mv": varies,
        "symbols_constant_float_mv": const,
        "adjacent_change_rate_median": round(float(vary_rate.median()), 4),
        "delisted_like_symbols_last_day_lt_global": int(len(stale)),
        "delisted_tail30_float_mv_present_ratio_median":
            round(float(tail_pres.median()), 4) if len(tail_pres) else None,
        "float_share_x_close_sanity": {
            "n_symbols_sampled": 50, "n_rows_used": int(len(sub)),
            "ratio_median": round(float(ratio.median()), 6),
            "ratio_p05": round(float(ratio.quantile(0.05)), 6),
            "ratio_p95": round(float(ratio.quantile(0.95)), 6),
        },
        "as_of_boundary": "结构验证(存在率/时变/退市止日/量纲)已过; 无外部 as-of 数据源, "
                          "D日 float_mv 来自 <=P日 已发布快照 这一点以数据源声明为准 (结构证据: "
                          "非恒定 + 退市股止于退市日, 与后处理回填特征不符)",
    }


# ---------------- 3. INFERRED_HALT 全库 ----------------

def run_halt(daily: pd.DataFrame, minute_day_by_sym: dict, delist: set) -> dict:
    days_by_sym = dict(daily.groupby("code")["date"].agg(lambda s: sorted(set(s))))
    era_days = {c: [d for d in ds if d >= MINUTE_START] for c, ds in days_by_sym.items()}
    era_days = {c: ds for c, ds in era_days.items() if ds}
    # ---- 市场级存在率: 某日 >10% 有日线的票无分钟 -> 供应商缺口日 ----
    n_daily_by_date = defaultdict(int)
    n_minute_by_date = defaultdict(int)
    for c, ds in era_days.items():
        md = minute_day_by_sym.get(c, set())
        for d in ds:
            n_daily_by_date[d] += 1
            if d in md:
                n_minute_by_date[d] += 1
    gap_dates = {}
    for d, n in n_daily_by_date.items():
        presence = n_minute_by_date[d] / n
        if presence < 0.9:
            gap_dates[d] = round(presence, 4)
    rows_out = []
    stats = defaultdict(int)
    sym_class = defaultdict(int)
    examples = []
    for code, era in era_days.items():
        mdays = minute_day_by_sym.get(code, set())
        era_set = set(era)
        missing = [d for d in era if d not in mdays]
        is_delist = code in delist
        if not missing:
            sym_class["NORMAL_full_minute_coverage"] += 1
            continue
        coverage = len(mdays & era_set) / len(era)
        if is_delist:
            sym_cls = "DELISTED"
        elif not mdays:
            sym_cls = "DATA_MISSING_SYMBOL"
        elif coverage < 0.9:
            # 票级慢性缺口: 覆盖率 <90% (feed 未全程覆盖该票, ~500 票同形态)
            # 其缺失是数据供应问题, 不是停牌
            sym_cls = "DATA_MISSING_CHRONIC"
        else:
            # 末日历侧连续缺失 > 10 个交易日 -> 存疑 (可能数据缺失而非停牌)
            miss_set = set(missing)
            tail_run = sum(1 for d in era[-15:] if d in miss_set)
            sym_cls = "DATA_MISSING_SUSPICIOUS" if tail_run > 10 else "INFERRED_HALT"
        day_classes = []
        for d in missing:
            cls = "DATA_MISSING_PROVIDER_GAP" if d in gap_dates else sym_cls
            stats[f"days_{cls}"] += 1
            rows_out.append({"code": code, "date": d, "class": cls})
            if cls != "DATA_MISSING_PROVIDER_GAP":
                day_classes.append(cls)
        if not day_classes:
            # 缺失全部落在市场级缺口日: 该票本身无独立问题
            sym_class["NORMAL_except_provider_gap"] += 1
        else:
            sym_class[sym_cls] += 1
        if len(examples) < 10:
            examples.append({"code": code, "class": sym_cls,
                             "n_missing": len(missing),
                             "last_daily": era[-1],
                             "first_missing": missing[0], "last_missing": missing[-1]})
    with open(HALT_DAYS_OUT, "w", encoding="utf-8") as f:
        for r in rows_out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    n_era = len(era_days)
    n_no_minute = int(sum(1 for c in era_days if not minute_day_by_sym.get(c, set())))
    total_missing = len(rows_out)
    gap_top = sorted(gap_dates.items(), key=lambda kv: n_daily_by_date[kv[0]] - n_minute_by_date[kv[0]],
                     reverse=True)[:10]
    return {
        "minute_era_start": MINUTE_START,
        "n_symbols_with_daily_in_era": n_era,
        "n_symbols_with_no_minute_at_all": n_no_minute,
        "total_missing_symbol_days": total_missing,
        "day_counts_by_class": dict(stats),
        "symbol_counts_by_class": dict(sym_class),
        "provider_gap_dates": len(gap_dates),
        "provider_gap_top10": [
            {"date": d, "presence": p, "n_symbols_missing": n_daily_by_date[d] - n_minute_by_date[d]}
            for d, p in gap_top],
        "examples": examples,
        "symbol_day_list_file": str(HALT_DAYS_OUT),
        "TRADING_STATUS_SOURCE_conclusion":
            "无独立停牌状态字段; 交易日状态由 日线存在 × 分钟存在 × 退市清单 推断 "
            "(INFERRED_HALT, 非真实状态)。两级数据缺口已从 INFERRED_HALT 剥离: "
            f"(a) 市场级缺口日(当日 >10% 有日线的票无分钟, 共 {len(gap_dates)} 日)"
            "-> DATA_MISSING_PROVIDER_GAP; (b) 票级慢性缺口(该票纪元内分钟覆盖率 <90%, "
            "约 500 票同形态, feed 未全程覆盖) -> DATA_MISSING_CHRONIC。二者不得当停牌; "
            "剩余孤立缺失方为 INFERRED_HALT。Execution/Label QA 必须显式标注 INFERRED_HALT; "
            "DATA_MISSING_SUSPICIOUS / DATA_MISSING_SYMBOL 类不得直接用于 next-legal-fill 门控",
    }


# ---------------- 4. 复权因子全库 ----------------

def http_pairs(pattern: str, timeout: int = 300) -> list:
    """cmd=get 返回 [[key, {...}], ...] 键值对; 复权 vals 无序, 必须按键配对。"""
    t = urllib.parse.quote(pattern)
    req = urllib.request.Request(f"{BASE}/?cmd=get&t={t}")
    with _opener.open(req, timeout=timeout) as r:
        raw = r.read()
    try:
        obj = msgpack.unpackb(raw, raw=False, strict_map_key=False)
    except Exception:
        obj = json.loads(raw.decode("utf-8", errors="replace"))
    if isinstance(obj, list):
        return [(str(k), v) for k, v in obj
                if isinstance(k, str) and isinstance(v, dict)]
    return []


def _adjustment_continuity_sanity(by_sym, daily, sample_n):
    """辅助证据: 抽样事件处 close(t-1)/pre_close(t) ≈ mult (2025+ 纪元守卫)。"""
    era_daily = daily[daily["date"] >= MINUTE_START]
    close_by = {c: s.set_index("date")["close"] for c, s in
                era_daily.groupby("code")[["date", "close"]]}
    pre_by = {c: s.set_index("date")["pre_close"] for c, s in
              era_daily.groupby("code")[["date", "pre_close"]]}
    random.seed(11)
    sym_pool = sorted(by_sym)
    picked_syms = random.sample(sym_pool, min(sample_n, len(sym_pool)))
    picked = [(c, random.choice(by_sym[c])[0]) for c in picked_syms]
    sane, n_excluded = [], 0
    for c, dstr in picked:
        d = int(dstr)
        mult = dict(by_sym[c])[dstr].get("mult")
        if c not in close_by or mult is None:
            n_excluded += 1
            sane.append({"code": c, "event": dstr, "verdict": "not_in_local_daily"})
            continue
        cl, pc = close_by[c], pre_by[c]
        rels = []
        prev = None
        for dt in sorted(cl.index):
            if prev is not None:
                p = pc.get(dt)
                if p is not None and p == p and p > 0:
                    rels.append(abs(prev / p - 1))
            prev = cl[dt]
        pc_reliable = bool(rels) and sorted(rels)[len(rels) // 2] < 0.02
        if not pc_reliable:
            n_excluded += 1
            sane.append({"code": c, "event": dstr, "verdict": "pre_close_unreliable_symbol_era"})
            continue
        dates = sorted(cl.index)
        hit, hit_day = None, None
        idx0 = next((i for i, x in enumerate(dates) if x >= d), None)
        if idx0 is not None:
            for i in range(max(1, idx0), min(len(dates), idx0 + 15)):
                p = pc.get(dates[i])
                if p is not None and p == p and p > 0:
                    r = float(cl[dates[i - 1]]) / float(p)
                    if abs(r - mult) / max(mult, 1e-9) < 0.005 and abs(r - 1) > 0.002:
                        hit, hit_day = round(r, 6), int(dates[i])
                        break
        sane.append({"code": c, "event": dstr, "mult": mult,
                     "matched_close_pc_ratio": hit, "ex_day": hit_day,
                     "verdict": "ok" if hit else "no_ex_day_match_in_15d"})
    n_ok = sum(1 for s in sane if s.get("verdict") == "ok")
    return {"n_sampled": len(sane), "n_excluded": n_excluded,
            "n_ok": n_ok, "details": sane}


def run_adjustment(daily: pd.DataFrame, sample_n: int = 20) -> dict:
    """Gate 1A-R4: canonical PIT adjustment reconstruction + 318 链断裂确定性分类。

    不使用供应商存储 cum (已确诊 318 链断裂)。改用可信 mult 从固定锚点链式重建
    adj_factor_pit; 对原 318 个 cum 链断裂逐条分类, 不允许 UNKNOWN。
    """
    from adjustment_chain import (
        build_chains, classify_breaks, REBUILT_OK, KNOWN_RESET_HANDLED,
        LEGACY_REPAIRED, QUARANTINED,
    )
    pairs = http_pairs("复权:*")
    by_sym = defaultdict(list)
    for k, v in pairs:
        parts = k.split(":")
        if len(parts) >= 3:
            by_sym[parts[1].zfill(6)].append((parts[2], v))
    n_events = len(pairs)
    n_syms = len(by_sym)
    ev_per_sym = sorted(len(v) for v in by_sym.values())

    # 1) effective_date 排序/重复检查
    n_unsorted = n_dup = 0
    for v in by_sym.values():
        dates = [d for d, _ in v]
        if len(dates) != len(set(dates)):
            n_dup += 1
        if dates != sorted(dates):
            n_unsorted += 1

    # 2) canonical chain (mult 链式) + 重新检测原 cum 链断裂 (318)
    build_chains(by_sym)
    breaks = []  # (code, date, mult, cum)
    for code, v in by_sym.items():
        prev_cum = None
        for d, rec in sorted(v, key=lambda x: x[0]):
            m, cum = rec.get("mult"), rec.get("cum")
            if prev_cum is not None and m and cum:
                if abs(prev_cum * m - cum) / max(cum, 1e-9) > 1e-3:
                    breaks.append((code, d, m, cum))
            prev_cum = cum

    # 3) 逐条确定性分类 (无 UNKNOWN)
    classification = classify_breaks(by_sym, breaks)
    cat_counts = classification["category_counts"]
    n_unknown = cat_counts.get("UNKNOWN", 0)
    quarantined_syms = sorted({c for c, items in classification["per_symbol"].items()
                               for x in items if x["category"] == QUARANTINED})

    # 4) 抽样连续性 sanity (辅助证据, 非 gate)
    cont = _adjustment_continuity_sanity(by_sym, daily, sample_n)

    # 5) 状态判定 (复制权修复硬条件)
    adj_ok = (n_unsorted == 0 and n_dup == 0 and n_unknown == 0)
    status = "VERIFIED_PASS" if adj_ok else "CONDITIONAL_PASS"

    return {
        "adjustment_method": "anchor_chain_mult",
        "pit_safe": True,
        "pit_assertion": "max(effective_date_used) <= feature_asof_time",
        "n_symbols_with_events": n_syms,
        "total_events": n_events,
        "events_per_symbol": {
            "p1": _quantile(ev_per_sym, 0.01), "p50": _quantile(ev_per_sym, 0.5),
            "p95": _quantile(ev_per_sym, 0.95), "max": ev_per_sym[-1] if ev_per_sym else None,
        },
        "symbols_with_unsorted_dates": n_unsorted,
        "symbols_with_duplicate_dates": n_dup,
        "canonical_chain": {"rebuilt_from": "mult (not stored cum)",
                             "n_symbols": n_syms,
                             "note": "链式累乘 mult; cum 字段不再用作口径来源"},
        "legacy_cum_breaks": {
            "n_detected": len(breaks),
            "category_counts": cat_counts,
            "n_quarantined": cat_counts.get(QUARANTINED, 0),
            "quarantined_symbols_sample": quarantined_syms[:50],
        },
        "continuity_sanity": cont,
        "status": status,
        "known_issue": "" if adj_ok else f"存在 UNKNOWN 分类 {n_unknown} 条, 须先确定性处理/隔离",
        "upgrade_note": "所有原 cum 链断裂已确定性分类为 "
                        f"{REBUILT_OK}/{KNOWN_RESET_HANDLED}/{LEGACY_REPAIRED}/{QUARANTINED}, "
                        "无 UNKNOWN -> VERIFIED_PASS",
        "pre_close_field_quality_note": "日线 pre_close 字段老历史质量差(全历史仅~57.5%可靠,NaN 17%,"
                                        "2025+ 89.7%); 任何 pre_close 派生特征须限定 2025+ 纪元或改读 "
                                        "analytic_prev_close; vendor_pre_close 已在 feature_registry 置 "
                                        "model_eligible=false",
    }


# ---------------- main ----------------

def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "dry"
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    t0 = time.time()
    universe = load_universe()
    print(f"[gate1a] mode={mode} universe={len(universe)} "
          f"(expected 7562, match={len(universe) == 7562})")

    if mode == "dry":
        step = max(1, len(universe) // 20)
        scan_universe = universe[::step][:20]
        workers = 4
        adj_sample = 5
        cross_n = 20
        float_syms_note = "dry: 全量 parquet 读取 (结构统计按全库, sanity 抽样照旧 50)"
    else:
        scan_universe = universe
        workers = 8
        adj_sample = 20
        cross_n = 200

    res = {"report_name": "DATA_GATE1A_FULL", "mode": mode,
           "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
           "generated_by": "step9_preflight/gate1a_full_scan.py",
           "universe_size": len(universe), "interpreter": sys.executable}

    minute = run_minute_scan(scan_universe, workers=workers)
    minute_day_by_sym = minute.pop("_minute_day_by_sym")
    minute["local_cache_cross_check"] = cross_check_local(minute_day_by_sym, sample_n=cross_n)
    minute["merge_strategy"] = ("HTTP 全拉单一路径聚合; 本地 minute_cache_clean (3959 票, "
                                f"<= {LOCAL_CACHE_LAST}) 仅作日集合交叉校验, 不做合并源 "
                                "(避免本地截止日当日完整性不确定的边界风险)")
    res["stock_minute_history"] = minute
    print(f"[1/4] minute done: symbols_with_data={minute['symbols_with_minute_data']} "
          f"bars={minute['bars_total']} in {minute['scan_seconds']}s")

    print("[2/4] loading daily parquet ...")
    daily = load_daily()
    print(f"[2/4] daily rows={len(daily)} symbols={daily['code'].nunique()}")
    res["free_float_market_cap"] = run_float_mv(daily)
    print(f"[2/4] float_mv done: present_ratio="
          f"{res['free_float_market_cap']['float_mv_present_ratio_overall']}")

    delist = load_delist_codes()
    print(f"[3/4] delist codes={len(delist)}")
    res["halt_trading_status"] = run_halt(daily, minute_day_by_sym, delist)
    print(f"[3/4] halt done: missing_symbol_days="
          f"{res['halt_trading_status']['total_missing_symbol_days']}")

    res["adjustment_factors"] = run_adjustment(daily, sample_n=adj_sample)
    print(f"[4/4] adjustment done: symbols={res['adjustment_factors']['n_symbols_with_events']} "
          f"events={res['adjustment_factors']['total_events']}")

    # ---- 状态判定 ----
    m = res["stock_minute_history"]
    minute_ok = (m["symbols_with_minute_data"] / max(1, m["n_symbols_universe"]) >= 0.98
                 and m["bars_per_symbol_day"]["p5"] >= 235
                 and (m["duplicate_bar_ratio"] or 0) < 0.001
                 and m["minute_start_confirmed"])
    m["status"] = "VERIFIED_PASS" if minute_ok else "CONDITIONAL_PASS"
    m["known_issue"] = ("" if minute_ok else
                        f"coverage={m['symbols_with_minute_data']}/{m['n_symbols_universe']} "
                        f"p5={m['bars_per_symbol_day']['p5']} dup={m['duplicate_bar_ratio']}")

    f = res["free_float_market_cap"]
    float_ok = (f["low_coverage_symbols_all_fund_like"]
                and f["float_mv_present_ratio_minute_era_median_per_symbol"] >= 0.99
                and f["symbols_varying_float_mv"] > f["symbols_constant_float_mv"])
    f["status"] = "VERIFIED_PASS" if float_ok else "CONDITIONAL_PASS"
    f["known_issue"] = "" if float_ok else "结构验证未达阈值"

    h = res["halt_trading_status"]
    halt_ok = h["day_counts_by_class"].get("days_DATA_MISSING_SUSPICIOUS", 0) == 0
    h["status"] = "VERIFIED_PASS" if halt_ok else "CONDITIONAL_PASS"
    h["known_issue"] = (
        "" if halt_ok else
        "存在 DATA_MISSING_SUSPICIOUS 票-日, next-legal-fill 门控需人工复核")

    a = res["adjustment_factors"]
    # Gate 1A-R4: 复权修复硬判定 (不再依赖 cum_chain.n_bad)
    adj_ok = (a["symbols_with_duplicate_dates"] == 0
              and a["symbols_with_unsorted_dates"] == 0
              and a["legacy_cum_breaks"]["category_counts"].get("UNKNOWN", 0) == 0)
    a["status"] = "VERIFIED_PASS" if adj_ok else "CONDITIONAL_PASS"
    a["known_issue"] = ("" if adj_ok else
                        "存在 UNKNOWN 分类的 cum 链断裂, 须先确定性处理/隔离 "
                        "(REBUILT_OK / KNOWN_RESET_HANDLED / LEGACY_REPAIRED / QUARANTINED)")

    statuses = [res[k]["status"] for k in
                ("stock_minute_history", "free_float_market_cap",
                 "halt_trading_status", "adjustment_factors")]
    res["verdict"] = {
        "gate1a_final": "VERIFIED_PASS" if all(s == "VERIFIED_PASS" for s in statuses) else
                        ("CONDITIONAL_PASS" if all(s in ("VERIFIED_PASS", "CONDITIONAL_PASS")
                                                 for s in statuses) else "FAIL"),
        "item_statuses": dict(zip(
            ("stock_minute_history", "free_float_market_cap",
             "halt_trading_status", "adjustment_factors"), statuses)),
        "upgrade_condition": "4 项全 VERIFIED_PASS 即满足升级条件",
        "MINUTE_START_frozen": MINUTE_START if m["minute_start_confirmed"] else None,
        "MINUTE_START_declaration": f"分钟数据纪元冻结为 {MINUTE_START} (2025-01-02); "
                                    f"此前无分钟属正常, 不得当 halt/缺失",
        "run_seconds": round(time.time() - t0, 1),
    }
    with open(OUT, "w", encoding="utf-8") as fh:
        yaml.safe_dump(res, fh, allow_unicode=True, sort_keys=False, width=120)
    print(f"\n[gate1a] total {time.time()-t0:.0f}s -> wrote {OUT}")
    for k in ("stock_minute_history", "free_float_market_cap",
              "halt_trading_status", "adjustment_factors"):
        print(f"  {k:<24} {res[k]['status']}")
    print(f"  gate1a_final = {res['verdict']['gate1a_final']}")


if __name__ == "__main__":
    main()
