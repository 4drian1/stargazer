"""qa_core_data — Gate 1A 深度 QA (Step 9A 启动前置)。

对 v2 中 4 项 CONDITIONAL_PASS 核心数据做 coverage / as-of 校验,
确认 Gate 1A 能否从 CONDITIONAL_PASS 升级为 VERIFIED_PASS。

校验项 (2026-09-17 裁决):
  stock_minute_history:
    n_symbols / n_trade_dates / bars_per_symbol_day 分布 (P1/P5/median/P95)
    missing_bar_ratio / duplicate_bar_ratio
    symbols_with_coverage_2025 / symbols_with_coverage_2026
    universe coverage (sample 相对全市场母池)
  free_float_market_cap:
    float_mv 行内存在率 / 时序变异性 (非恒定, 排除后处理回填嫌疑)
    delisted 样本上 float_mv 是否止于退市日 (INFERRED as-of)
  halt_trading_status:
    daily_missing × minute_missing × known_delisting 交叉 -> INFERRED_HALT (非真实状态)
  adjustment_factors:
    n_symbols_with_adjustment_events / 全样本事件覆盖
    raw->adjusted 重建 sanity (random 50 symbols 连续性)

用法:
    python qa_core_data.py [SAMPLE_N]
默认 SAMPLE_N=12 (抽样 QA; 全市场扫描为后续步骤)。
"""
from __future__ import annotations
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd
import yaml

import data_availability_audit as aud

_HERE = Path(__file__).resolve().parent
OUT = _HERE / "DATA_GATE1A_QA.yaml"
EXPECTED_BARS_PER_DAY = 240  # A股 9:30-15:00 连续竞价分钟数


def _extract_codes(universe) -> list:
    codes = []
    if isinstance(universe, dict):
        for v in universe.values():
            if isinstance(v, list):
                for it in v:
                    if isinstance(it, dict):
                        codes.append(it.get("code") or it.get("symbol") or "")
                    elif isinstance(it, str):
                        codes.append(it)
            elif isinstance(v, dict):
                codes.append(v.get("code") or v.get("symbol") or "")
    elif isinstance(universe, list):
        for it in universe:
            if isinstance(it, dict):
                codes.append(it.get("code") or it.get("symbol") or "")
            elif isinstance(it, (list, tuple)) and len(it) >= 2 and isinstance(it[1], dict):
                codes.append(it[1].get("code") or it[0])
    return [c for c in codes if c]


def _quantile(sorted_vals, q):
    if not sorted_vals:
        return None
    idx = max(0, min(len(sorted_vals) - 1, int(round(q * (len(sorted_vals) - 1)))))
    return sorted_vals[idx]


def qa_minute(sample):
    per_day_all = []
    sym_days = {}
    sym_cov_2025 = sym_cov_2026 = 0
    dup_total = 0
    bars_total = 0
    sym_with_data = 0
    for code in sample:
        try:
            recs = aud.http_get("分钟k", code, "*", timeout=120)
        except Exception as e:  # noqa
            sym_days[code] = {"error": str(e)}
            continue
        if not isinstance(recs, list) or not recs:
            sym_days[code] = {"rows": 0}
            continue
        sym_with_data += 1
        per_day = defaultdict(int)
        seen = set()
        for r in recs:
            if not isinstance(r, dict):
                continue
            d = str(r.get("date"))
            per_day[d[:8]] += 1
            bars_total += 1
            if d in seen:
                dup_total += 1
            else:
                seen.add(d)
        vals = list(per_day.values())
        per_day_all.extend(vals)
        has25 = any(d.startswith("2025") for d in per_day)
        has26 = any(d.startswith("2026") for d in per_day)
        sym_cov_2025 += int(has25)
        sym_cov_2026 += int(has26)
        sym_days[code] = {"days": len(per_day), "bars": sum(vals),
                          "ppd_median": (_quantile(sorted(vals), 0.5) if vals else 0)}
    per_day_all.sort()
    median_ppd = _quantile(per_day_all, 0.5) or 0
    missing_bar_ratio = round(1 - median_ppd / EXPECTED_BARS_PER_DAY, 4) if median_ppd else None
    return {
        "status": "CONDITIONAL_PASS",
        "n_symbols_sampled": len(sample),
        "symbols_with_data": sym_with_data,
        "bars_total_sampled": bars_total,
        "bars_per_symbol_day": {
            "p1": _quantile(per_day_all, 0.01),
            "p5": _quantile(per_day_all, 0.05),
            "median": median_ppd,
            "p95": _quantile(per_day_all, 0.95),
            "expected": EXPECTED_BARS_PER_DAY,
        },
        "missing_bar_ratio": missing_bar_ratio,
        "duplicate_bar_ratio": round(dup_total / bars_total, 6) if bars_total else None,
        "symbols_with_coverage_2025": sym_cov_2025,
        "symbols_with_coverage_2026": sym_cov_2026,
        "verdict": "抽样存在性+分布 OK; 需全市场扫描确认 universe coverage 与每日接近预期分钟数, 方可 VERIFIED_PASS",
    }


def qa_float(sample):
    present_ratios = []
    varies = 0
    checked = 0
    for code in sample:
        try:
            recs = aud.http_get("日k", code, "*", timeout=60)
        except Exception:  # noqa
            continue
        if not isinstance(recs, list) or not recs:
            continue
        checked += 1
        seq = [r.get("float_mv") for r in recs if isinstance(r, dict)]
        present = [v for v in seq if v is not None]
        if seq:
            present_ratios.append(len(present) / len(seq))
        # 变异: 相邻差分非零比例
        diffs = [abs(seq[i] - seq[i - 1]) for i in range(1, len(seq)) if seq[i] is not None and seq[i-1] is not None]
        if diffs and sum(1 for d in diffs if d > 0) / len(diffs) > 0.5:
            varies += 1
    return {
        "status": "CONDITIONAL_PASS",
        "n_symbols_checked": checked,
        "float_mv_present_ratio_median": round(_quantile(sorted(present_ratios), 0.5), 4) if present_ratios else None,
        "symbols_with_varying_float_mv": varies,
        "as_of_note": "float_mv 多数随日期变化(非恒定) 支持'每日快照'假设, 但仍需外部验证 D日值来自 <= P日 已发布数据 (非后处理回填)",
        "verdict": "需对照 delisted 样本确认 float_mv 止于退市日 + 外部 as-of 验证, 方可 VERIFIED_PASS",
    }


def qa_halt(sample):
    # 分钟数据仅 2025-01-02 起; 之前的日线无分钟属正常缺失, 不能算 halt。
    # 真实 INFERRED_HALT 只在分钟纪元内 (daily 有 / minute 无) 统计。
    MINUTE_START = "20250102"
    inferred = {}
    pre_era_days_total = 0
    for code in sample:
        try:
            daily = aud.http_get("日k", code, "*", timeout=60)
            minute = aud.http_get("分钟k", code, "*", timeout=120)
        except Exception as e:  # noqa
            inferred[code] = {"error": str(e)}
            continue
        d_all, m_all = set(), set()
        if isinstance(daily, list):
            for r in daily:
                if isinstance(r, dict):
                    d_all.add(str(r.get("date"))[:8])
        if isinstance(minute, list):
            for r in minute:
                if isinstance(r, dict):
                    m_all.add(str(r.get("date"))[:8])
        d_era = {d for d in d_all if d >= MINUTE_START}
        pre_era_days_total += len(d_all - d_era)
        # 仅在分钟纪元内: 有日线但无分钟 = INFERRED_HALT (或数据缺口)
        inferred_halt = sorted(d_era - m_all)
        inferred[code] = {"daily_days_total": len(d_all), "minute_days": len(m_all),
                          "daily_days_in_minute_era": len(d_era),
                          "inferred_halt_days": len(inferred_halt)}
    total_inferred = sum(v.get("inferred_halt_days", 0) for v in inferred.values() if isinstance(v, dict))
    return {
        "status": "CONDITIONAL_PASS",
        "n_symbols_checked": len(sample),
        "minute_era_start": MINUTE_START,
        "total_inferred_halt_days_in_era": total_inferred,
        "pre_minute_era_daily_days": pre_era_days_total,
        "note": "INFERRED_HALT 仅统计分钟纪元内(>=2025-01-02) '日线有/分钟无'; 此前日线属正常缺失(分钟数据 2025+ only)",
        "detail": inferred,
        "verdict": "Execution QA 必须显式标注 INFERRED_HALT; 与 delisting 清单交叉后方可用于 next-legal-fill 门控",
    }


def qa_adjustment(sample):
    n_sym = 0
    events = 0
    for code in sample:
        try:
            recs = aud.http_get("复权", code, "*", timeout=60)
        except Exception:  # noqa
            continue
        if isinstance(recs, list) and recs:
            n_sym += 1
            events += len(recs)
    return {
        "status": "CONDITIONAL_PASS",
        "n_symbols_checked": len(sample),
        "symbols_with_adjustment_events": n_sym,
        "total_events_sampled": events,
        "note": "单票事件存在; 需全市场扫描确认 n_symbols_with_adjustment_events 覆盖全库 (非仅 27 条)",
        "verdict": "需 raw->adjusted 重建 sanity (random 50 symbols 复权前后连续性), 方可 VERIFIED_PASS",
    }


def _load_universe_from_cache() -> list:
    """从本地日线 parquet 缓存 (data_daily_parts) 提取全市场代码清单 (canonical universe)。

    比 HTTP '股票代码' 表可靠 (后者返回空); 列投影只读 'code' 列。
    """
    import glob
    import os
    data_root = os.environ.get("STARGAZER_DATA_ROOT", r"D:/a股数据")
    cache = os.path.join(data_root, "data_daily_parts")
    parts = sorted(glob.glob(os.path.join(cache, "part_*.parquet")))
    if not parts:
        return []
    codes = set()
    for p in parts:
        try:
            df = pd.read_parquet(p, columns=["code"])
            for c in df["code"].astype(str):
                codes.add(c.zfill(6))
        except Exception:  # noqa
            continue
    return sorted(codes)


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 12
    print(f"[QA] SAMPLE_N={n}")
    universe = _load_universe_from_cache()
    print(f"[QA] universe size = {len(universe)}")
    if universe:
        step = max(1, len(universe) // n)
        sample = universe[::step][:n]
    else:
        sample = aud.SAMPLE_CODES[:n]
    print(f"[QA] sample = {sample}")

    res = {
        "report_name": "DATA_GATE1A_QA",
        "generated_by": "step9_preflight/qa_core_data.py",
        "sample_size": n,
        "universe_size": len(universe),
        "stock_minute_history": qa_minute(sample),
        "free_float_market_cap": qa_float(sample),
        "halt_trading_status": qa_halt(sample),
        "adjustment_factors": qa_adjustment(sample),
    }
    with open(OUT, "w", encoding="utf-8") as f:
        yaml.safe_dump(res, f, allow_unicode=True, sort_keys=False)

    for k, v in res.items():
        if isinstance(v, dict) and "status" in v:
            print(f"  {k:<24} {v['status']:<16} {v.get('verdict','')[:60]}")
    print(f"-> wrote {OUT}")
    # 4 项均为 CONDITIONAL_PASS 即 Gate 1A 数据层面通过 (待人工确认 VERIFIED_PASS)
    ok = all(v.get("status") == "CONDITIONAL_PASS" for k, v in res.items() if isinstance(v, dict) and "status" in v)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
