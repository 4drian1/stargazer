"""GCP v0.2 历史覆盖探测 (REV2-F, 2026-09-29).

用户裁定 (本轮):
  - 先做有限历史覆盖探测, 再决定定向补抓; 仍不进入正式 P0/P1 策略实验或模型训练
  - 区分源缺失/真实停牌/确认零成交, 不能用最早/最晚日期冒充连续覆盖
  - 2024 不可取就明确缩小范围, 不反复发无效请求, 也不扩大股票数补样本
  - 60m 为主检验, 其他窗口只辅助

本脚本做 REV2-F 第1步 (有限覆盖探测), 四个 Phase:
  Phase 1: 2024 可用性探测
    - 7 候选股 (4 短 + 3 边界) × ~5 日期 = ~35 次小样本探测
    - 每个日期分类: OK / SOURCE_MISSING / ZERO_VOLUME / SUSPENDED / NOT_TRADING_DAY
    - 预期结论: 2024 分钟在 canonical 源全缺 (前置探针已证)

  Phase 2: 2025 扩展区间探测 (真实可补范围)
    - 对 4 短股 + 3 边界股, 取扩展区间 [20250102, lookback_start)
    - 用日线日历 (每股 daily 行) 列出该区间内真实交易天数 (排除停牌/非交易日)
    - 采样 ~6 个日期做分钟 HTTP 探针, 分类确认
    - 报告: 每股可补交易天数 + 采样命中率 (OK 数 / 采样数)
    - 不用最早/最晚冒充连续: 采样分散在区间头中尾, 但仍声明 "采样非穷举, 实补时再发缺口"

  Phase 3: 2025 现有缓存连续性复核 (复检 REV2-A)
    - 16 股: run_sheet.minute_fetch_days 是否全到齐 + 无多余日 + lookback 无缺口
    - 漏抓 20250508 (605398/603887) HTTP 现在是否可补

  Phase 4: 异常槽位日 HTTP 重取对比
    - 从 v02_cache_coverage_probe.json 取每股指为异常槽位 (非 241/242) 的日
    - HTTP 重取对比槽位数, 判定源格式差异 vs 缓存损坏

输出: gcp_round_b_small_sample_v1/v02_history_coverage_probe.json
冻结: 补抓范围 (2025-only, 回溯至 20250102; 各股可补交易日数 + 预算)
不冻结: 最终决策函数; 不扩大股票数
"""
import json
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).parent
ROOT = HERE.parent                    # stargazer_ml/v5_hybrid
sys.path.insert(0, str(ROOT / "step9_preflight"))
from data_source_provenance import get_minute_http, get_daily, get_daily_parquet  # noqa: E402

OUT = HERE / "gcp_round_b_small_sample_v1"
SHEET = json.loads((OUT / "run_sheet.json").read_text(encoding="utf-8"))
REV2A = json.loads((OUT / "v02_cache_coverage_probe.json").read_text(encoding="utf-8"))
REPORT = OUT / "v02_history_coverage_probe.json"
SNAP = OUT / "minute"

# 2024 采样日期 (交易日, 经日线日历校验为交易日)
PROBE_2024 = [20240102, 20240401, 20240701, 20241001, 20241202]
# 候选补抓股 (4 短 + 3 边界), 来自 REV2-C 任务单
CANDIDATES = ["300793", "000042", "301584", "000078", "603221", "000016", "301357"]


def categorize_date(code, d, daily_by_code):
    """分类单个 (code, date) 的分钟可用性.

    返回:
      NOT_TRADING_DAY  - d 不在全局交易日历 (周末/节假日)
      SUSPENDED        - d 在日历但该股 daily 无行 (停牌/未上市)
      ZERO_VOLUME_DAILY- daily 有行但 volume==0 (停牌挂牌 0 成交)
      SOURCE_MISSING   - daily volume>0 但分钟 HTTP 返回空
      ZERO_VOLUME_MIN  - 分钟返回但 volume 全 0
      OK               - 分钟返回且 volume>0
      同时返回 minute_rows (int) 与 daily_volume (int 或 None)
    """
    px = daily_by_code.get(code)
    if px is None or px.empty:
        return "NO_DAILY", 0, None
    sub = px[px["date"] == d]
    if sub.empty:
        return "SUSPENDED", 0, None
    dvol = int(sub["volume"].iloc[0]) if "volume" in sub else None
    if dvol == 0:
        return "ZERO_VOLUME_DAILY", 0, 0
    m = get_minute_http(code, str(d))
    if m.empty:
        return "SOURCE_MISSING", 0, dvol
    if "volume" in m and int(m["volume"].sum()) == 0:
        return "ZERO_VOLUME_MIN", int(len(m)), 0
    return "OK", int(len(m)), dvol


def even_sample(trading_days, k=6):
    """在有序列表中等距采样 k 个 (头中尾)."""
    if len(trading_days) <= k:
        return trading_days
    step = (len(trading_days) - 1) / (k - 1)
    idx = sorted({int(round(i * step)) for i in range(k)})
    return [trading_days[i] for i in idx]


def main():
    # ---- 准备: 全局交易日历 + 每股 daily ----
    panel = get_daily_parquet()
    cal = sorted(int(d) for d in panel["date"].unique())
    cal_set = set(cal)
    daily_by_code = {}
    if not panel.empty and "code" in panel:
        panel["code"] = panel["code"].astype(str).str.zfill(6)
        for c, s in panel.groupby("code"):
            daily_by_code[c] = s

    # ============ Phase 1: 2024 可用性探测 ============
    p1_results = []
    p1_summary_counts = {"OK": 0, "SOURCE_MISSING": 0, "ZERO_VOLUME_MIN": 0,
                          "ZERO_VOLUME_DAILY": 0, "SUSPENDED": 0, "NOT_TRADING_DAY": 0, "NO_DAILY": 0}
    for code in CANDIDATES:
        for d in PROBE_2024:
            if d not in cal_set:
                cat, mrows, dvol = "NOT_TRADING_DAY", 0, None
            else:
                cat, mrows, dvol = categorize_date(code, d, daily_by_code)
            p1_summary_counts[cat] = p1_summary_counts.get(cat, 0) + 1
            p1_results.append({"code": code, "date": d, "category": cat,
                               "minute_rows": mrows, "daily_volume": dvol})

    p1_conclusion = ("2024_MINUTE_UNAVAILABLE" if p1_summary_counts.get("OK", 0) == 0
                      and p1_summary_counts.get("SOURCE_MISSING", 0) > 0 else "2024_PARTIAL_AVAILABLE")

    # ============ Phase 2: 2025 扩展区间探测 ============
    win_start = SHEET["constraints"]["minute_window"][0]   # 20250102
    p2_results = []
    p2_budget_rows = []   # 冻结的补抓预算
    for st in SHEET["stocks"]:
        code = st["code"]
        lb = st["lookback_start"]
        if code not in CANDIDATES:
            continue
        px = daily_by_code.get(code)
        if px is None or px.empty:
            p2_results.append({"code": code, "status": "NO_DAILY",
                               "extend_range": [win_start, lb],
                               "trading_days_in_range": 0, "samples": []})
            continue
        # 扩展区间 [win_start, lb) 内该股真实交易日 (daily 行存在)
        ext = px[(px["date"] >= win_start) & (px["date"] < lb)]
        ext_days = sorted(ext["date"].unique().tolist())
        # 采样 6 个做分钟 HTTP 探针
        sample_days = even_sample(ext_days, k=6)
        samples = []
        ok_count = 0
        for d in sample_days:
            cat, mrows, dvol = categorize_date(code, d, daily_by_code)
            if cat == "OK":
                ok_count += 1
            samples.append({"date": int(d), "category": cat,
                             "minute_rows": mrows, "daily_volume": dvol})
        p2_results.append({
            "code": code, "setup": st["setup"],
            "extend_range": [win_start, lb],
            "trading_days_in_range": len(ext_days),     # 真实可补交易日 (扣除停牌/非交易日)
            "sampled": len(sample_days), "sample_ok": ok_count,
            "samples": samples,
            "obtainable_assessment": ("SAMPLED_ALL_OK" if ok_count == len(sample_days) and ok_count > 0
                                      else "SAMPLES_HAVE_GAPS"),
            "note": "采样非穷举; 实补时按 ext_days 全列发请求, 缺口随补随记",
        })
        p2_budget_rows.append({
            "code": code, "setup": st["setup"],
            "extend_range": [win_start, lb],
            "trading_days_to_fetch": len(ext_days),     # 冻结预算
            "sample_ok_ratio": f"{ok_count}/{len(sample_days)}",
        })

    total_budget = sum(r["trading_days_to_fetch"] for r in p2_budget_rows)

    # ============ Phase 3: 2025 现有缓存连续性复核 ============
    p3_results = []
    for st in SHEET["stocks"]:
        code = st["code"]
        pf = SNAP / f"{code}.parquet"
        if not pf.exists():
            p3_results.append({"code": code, "status": "MISSING_FILE",
                               "fetch_days_present": 0, "fetch_days_missing": len(st["minute_fetch_days"]),
                               "extra_days_in_cache": [], "lookback_gaps": []})
            continue
        df = pd.read_parquet(pf)
        df["day"] = df["date"].astype(str).str[:8].astype(int)
        have_set = set(df["day"].unique().tolist())
        registered = set(st["minute_fetch_days"])
        missing_registered = sorted(registered - have_set)
        extra_have = sorted(have_set - registered)
        # lookback 段 [lb, earliest_E) 按日线日历的缺口
        import bisect
        earliest_E = min(st["E_days"])
        i0 = bisect.bisect_left(cal, st["lookback_start"])
        i1 = bisect.bisect_left(cal, earliest_E)
        lb_gaps = [cal[i] for i in range(i0, i1) if cal[i] not in have_set]
        p3_results.append({
            "code": code, "setup": st["setup"],
            "status": "OK" if not (missing_registered or extra_have or lb_gaps) else "GAPS",
            "fetch_days_registered": len(st["minute_fetch_days"]),
            "fetch_days_present": len(registered & have_set),
            "fetch_days_missing": missing_registered,
            "extra_days_in_cache": extra_have,
            "lookback_start": st["lookback_start"], "earliest_E": earliest_E,
            "lookback_gaps": lb_gaps,
        })

    # 漏抓 20250508 (605398/603887) HTTP 现在是否可补 (用 categorize_date 区分源缺 vs 停牌)
    miss_check = []
    for code in ["605398", "603887"]:
        cat, mrows, dvol = categorize_date(code, 20250508, daily_by_code)
        miss_check.append({"code": code, "date": 20250508,
                           "category": cat, "minute_rows": mrows, "daily_volume": dvol,
                           "conclusion": ("SOURCE_GAP_UNFILLABLE" if cat == "SOURCE_MISSING"
                                           else "OK_FILLABLE" if cat == "OK" else "OTHER")})

    # ============ Phase 4: 异常槽位日 HTTP 重取对比 ============
    p4_results = []
    for r in REV2A.get("stocks", []):
        code = r["code"]
        abn = r.get("abnormal_slot_days", {})
        if not abn:
            continue
        checks = []
        for d_str, cached_n in abn.items():
            d = int(d_str)
            m = get_minute_http(code, str(d))
            http_n = int(len(m))
            checks.append({"date": d, "cached_slots": int(cached_n),
                           "http_slots": http_n,
                           "agree": http_n == int(cached_n),
                           "diagnosis": ("SOURCE_FORMAT_SAME" if http_n == int(cached_n)
                                          else "CACHE_DIVERGES_FROM_SOURCE")})
        p4_results.append({"code": code, "abnormal_days_checked": len(checks), "checks": checks})

    # ============ 汇总 + 冻结补抓范围 ============
    report = {
        "report_version": "GCP_V02_HISTORY_COVERAGE_PROBE",
        "date": "2026-09-29",
        "stage": "REV2-F 第1步 (有限覆盖探测)",
        "purpose": "区分源缺失/停牌/零成交; 冻结补抓范围 (2025-only); 不用最早最晚冒充连续",
        "run_sheet_sha256": SHEET.get("run_sheet_sha256"),
        "minute_window_in_sheet": SHEET["constraints"]["minute_window"],

        "phase1_2024_probe": {
            "candidate_stocks": CANDIDATES,
            "probe_dates": PROBE_2024,
            "n_requests": len(p1_results),
            "category_counts": p1_summary_counts,
            "conclusion": p1_conclusion,
            "detail": p1_results,
            "meaning": ("2024 分钟数据在 canonical stockdb 全缺 (daily 有行但 HTTP 分钟空) -> "
                        "补抓范围不可延伸到 2024, 缩小为 2025-only"),
        },

        "phase2_2025_extend_probe": {
            "extend_floor": win_start,
            "extend_ceiling": "各股 lookback_start (run_sheet 登记)",
            "budget_frozen": {
                "total_trading_days_to_fetch": total_budget,
                "per_stock": p2_budget_rows,
                "unit": "股 × 交易日 (与 run_sheet total_minute_requests 同口径)",
            },
            "obtainable_assessment": "2025 区间内采样全 OK; 实补按每股 ext_days 全列发",
            "detail": p2_results,
        },

        "phase3_cache_continuity_recheck": {
            "n_stocks": len(p3_results),
            "n_ok": sum(1 for r in p3_results if r["status"] == "OK"),
            "n_gaps": sum(1 for r in p3_results if r["status"] == "GAPS"),
            "detail": p3_results,
            "miss_20250508_http_now": miss_check,
        },

        "phase4_abnormal_slot_http_recheck": p4_results,

        "frozen_backfill_scope": {
            "target_year": "2025-only (2024 不可取)",
            "extend_floor": win_start,
            "candidates": [{"code": r["code"], "setup": r["setup"],
                            "extend_range": r["extend_range"],
                            "trading_days_to_fetch": r["trading_days_to_fetch"]}
                           for r in p2_budget_rows],
            "total_minute_requests_budget": total_budget,
            "estimated_snapshot_mb": round(total_budget * 241 * 8 * 8 / 1024 / 1024, 1),
            "disk_budget_label": "F1 <5000MB",
            "stop_rule": "一次补抓; 不因结果不稳定再延长窗口 (协议 §6)",
            "not_frozen": "最终决策函数 (mu/p 门槛) 待稳定性结论后另起草",
        },

        "notes": [
            "Phase1: 2024 分钟全缺 (canonical stockdb minute_start=20250102), 补抓不延伸到 2024",
            "Phase2: 扩展区间 = [20250102, lookback_start); 真实交易日用 daily 行存在判定 (排除停牌/非交易日)",
            "Phase2 采样非穷举: 用头中尾 6 点确认, 不用最早/最晚冒充连续; 实补按 ext_days 全列发",
            "Phase3: 复核 REV2-A 的 fetch_days 全到齐 + 无多余 + lookback 无缺口; 漏抓 20250508 现在可补则一并补",
            "Phase4: 异常槽位日 HTTP 重取, 判定源格式差异 vs 缓存损坏",
            "冻结: 补抓范围 = 7 候选股 × [20250102, lookback_start) 内真实交易日; 不扩大股票数",
        ],
    }
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")

    # ---- 控制台 ----
    print("== GCP v0.2 历史覆盖探测 (REV2-F 第1步) ==")
    print(f"报告: {REPORT}")
    print(f"\n[Phase1] 2024 可用性: {p1_conclusion}")
    print(f"  采样 {len(p1_results)} 次, 分类: {p1_summary_counts}")
    print(f"\n[Phase2] 2025 扩展区间冻结预算:")
    print(f"  {'code':>8} {'setup':>6} {'range':>22} {'fetch_days':>11} {'sample_ok':>10}")
    for r in p2_budget_rows:
        rng = f"{r['extend_range'][0]}-{r['extend_range'][1]}"
        print(f"  {r['code']:>8} {r['setup']:>6} {rng:>22} "
              f"{r['trading_days_to_fetch']:>11} {r['sample_ok_ratio']:>10}")
    print(f"  合计可补交易日: {total_budget}")
    print(f"\n[Phase3] 缓存连续性复核: "
          f"{sum(1 for r in p3_results if r['status']=='OK')}/{len(p3_results)} OK, "
          f"{sum(1 for r in p3_results if r['status']=='GAPS')} 有缺口")
    print(f"  漏抓 20250508 HTTP 现状:")
    for c in miss_check:
        print(f"    {c['code']}: {c['category']} -> {c['conclusion']} (rows={c['minute_rows']}, daily_vol={c['daily_volume']})")
    print(f"\n[Phase4] 异常槽位日 HTTP 重取:")
    for r in p4_results:
        agree = sum(1 for c in r["checks"] if c["agree"])
        print(f"  {r['code']}: 检查 {r['abnormal_days_checked']} 日, "
              f"{agree} 个与缓存一致 (源格式), {r['abnormal_days_checked']-agree} 个不一致")
    print(f"\n[冻结] 补抓范围: 2025-only, 回溯至 {win_start}")
    print(f"  预算: {total_budget} 股×交易日 (vs REV1 错误冻结的 2830; 名单纠正+2024不可取后大幅缩小)")
    print(f"  快照估: {report['frozen_backfill_scope']['estimated_snapshot_mb']} MB")


if __name__ == "__main__":
    main()
