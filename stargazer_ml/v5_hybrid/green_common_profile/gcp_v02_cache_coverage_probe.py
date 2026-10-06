"""GCP v0.2 现有缓存覆盖核验脚本 (REV2-A, 2026-09-29).

用户裁定 (看过可行性结果后的修订):
  1. 补抓名单存在股票代码错配 (旧任务单写 000021/000057/000084/000093,
     真实应为 603221/301357/301584/603887 等), 必须从原始 run_sheet.json 生成.
  2. 缓存最早/最晚日期不代表中间连续覆盖, 须核验.
  3. 原清单分钟研究范围从 2025-01-02 起; 补到 2024 须先探测真实覆盖,
     不能直接批准 2830 股日请求.

本脚本只做第 1 步 (REV2-A): 核验现有缓存的
  - 真实 16 股代码 vs 缓存文件名映射
  - run_sheet 登记的 minute_fetch_days 是否全部抓到
  - lookback 回看段 (lb_start -> earliest_E) 的中间连续性 (按日线交易日历)
  - 每日槽位数 (13:00 源格式感知, 241/242)
  - 现有最早/最晚缓存日 (供第 3 步补抓预算引用, 不在此冻结)

**不冻结补抓范围与预算** (属第 3 步, 待 REV2-D/E 验收后再探测).
仅输出结构化 JSON 供后续任务单与协议引用.
"""
import bisect
import json
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).parent
ROOT = HERE.parent                    # stargazer_ml/v5_hybrid
sys.path.insert(0, str(ROOT / "step9_preflight"))
from data_source_provenance import get_daily_parquet  # noqa: E402

OUT = HERE / "gcp_round_b_small_sample_v1"
SNAP = OUT / "minute"
SHEET = json.loads((OUT / "run_sheet.json").read_text(encoding="utf-8"))
REPORT = OUT / "v02_cache_coverage_probe.json"


def main():
    panel = get_daily_parquet()
    cal = sorted(int(d) for d in panel["date"].unique())
    cal_idx = {d: i for i, d in enumerate(cal)}

    def cal_range_gaps(have_set, start, end_exclusive):
        """[start, end_exclusive) 交易日区间内 have_set 缺失的交易日 (按日线日历)."""
        i0 = bisect.bisect_left(cal, start)
        i1 = bisect.bisect_left(cal, end_exclusive)   # exclusive 上界
        if i0 >= i1:
            return []
        return [cal[i] for i in range(i0, i1) if cal[i] not in have_set]

    # ---- 1. 代码-文件名映射核验 ----
    sheet_codes = [s["code"] for s in SHEET["stocks"]]
    snap_files = {f.stem for f in SNAP.glob("*.parquet")} if SNAP.exists() else set()
    missing_files = [c for c in sheet_codes if c not in snap_files]
    extra_files = [c for c in snap_files if c not in sheet_codes]
    # 旧任务单错配代码 (用户核对发现, 此处显式标注)
    stale_codes_in_old_task_order = ["000021", "000057", "000084", "000093"]
    stale_not_in_sheet = [c for c in stale_codes_in_old_task_order if c not in sheet_codes]

    # ---- 2. 逐股核验 ----
    rows = []
    for st in SHEET["stocks"]:
        code = st["code"]
        pf = SNAP / f"{code}.parquet"
        if not pf.exists():
            rows.append({"code": code, "setup": st["setup"],
                         "status": "MISSING_FILE",
                         "registered_fetch_days": len(st["minute_fetch_days"]),
                         "have_days": 0, "missing_registered": [],
                         "lookback_gap_days": [], "slot_dist": {}})
            continue
        df = pd.read_parquet(pf)
        df["day"] = df["date"].astype(int).astype(str).str[:8].astype(int)
        have_days = sorted(df["day"].unique().tolist())
        have_set = set(have_days)
        registered = set(st["minute_fetch_days"])
        missing_registered = sorted(registered - have_set)            # 登记日未抓到
        extra_have = sorted(have_set - registered)                    # 缓存有但未登记 (应空)
        d_min, d_max = have_days[0], have_days[-1]
        # lookback 段连续性: [lb_start, earliest_E) (run_sheet 设计为连续交易日)
        lb = st["lookback_start"]
        earliest_E = min(st["E_days"])
        lb_gaps = cal_range_gaps(have_set, lb, earliest_E)
        # 每日槽位分布 (13:00 源格式: <20260624 期望 241, >=20260624 期望 242)
        slot_counts = df.groupby("day").size()
        slot_dist = slot_counts.value_counts().to_dict()
        # 槽位异常日 (<241 或 >242, 或中间值)
        abnormal_slots = {}
        for d, n in slot_counts.items():
            if n not in (241, 242):
                abnormal_slots[int(d)] = int(n)
        rows.append({
            "code": code, "setup": st["setup"],
            "status": "OK" if not (missing_registered or lb_gaps or abnormal_slots) else "GAPS",
            "registered_fetch_days": len(st["minute_fetch_days"]),
            "have_days": len(have_days),
            "date_min": d_min, "date_max": d_max,
            "missing_registered": missing_registered,           # 登记但没抓到
            "extra_have": extra_have,                           # 缓存有但未登记 (应空)
            "lookback_start": lb, "earliest_E": earliest_E,
            "lookback_gap_days": lb_gaps,                       # 回看段中间缺口 (关键)
            "n_lookback_gaps": len(lb_gaps),
            "slot_dist": {str(k): int(v) for k, v in slot_dist.items()},
            "abnormal_slot_days": abnormal_slots,              # 槽位数非 241/242 的日
            "E_days": st["E_days"],
            "lookback_span_months": round(
                (cal_idx.get(d_max, 0) - cal_idx.get(lb, 0) + 1) / 21.0, 2) if lb in cal_idx and d_max in cal_idx else None,
        })

    # ---- 3. 汇总 ----
    n_ok = sum(1 for r in rows if r["status"] == "OK")
    n_gaps = sum(1 for r in rows if r["status"] == "GAPS")
    short_span = [r for r in rows if r.get("lookback_span_months") is not None
                  and r["lookback_span_months"] < 6]
    report = {
        "report_version": "GCP_V02_CACHE_COVERAGE_PROBE",
        "date": "2026-09-29",
        "stage": "REV2-A (看过可行性结果后的修订)",
        "purpose": "核验现有 16 股缓存的代码-文件名映射 + 中间覆盖连续性; 不冻结补抓范围",
        "run_sheet_sha256": SHEET.get("run_sheet_sha256"),
        "minute_window_in_sheet": SHEET["constraints"]["minute_window"],
        "code_mapping_check": {
            "sheet_codes": sheet_codes,
            "stale_codes_in_old_task_order": stale_codes_in_old_task_order,
            "stale_not_in_current_sheet": stale_not_in_sheet,   # 这 4 个代码不在 run_sheet 也不在缓存
            "missing_cache_files": missing_files,               # 应空
            "extra_cache_files": extra_files,                    # 应空
            "mapping_ok": not missing_files and not extra_files,
        },
        "summary": {
            "n_stocks": len(rows),
            "n_ok": n_ok, "n_with_gaps": n_gaps,
            "n_short_span_lt6m": len(short_span),
            "short_span_codes": [r["code"] for r in short_span],
        },
        "stocks": rows,
        "notes": [
            "lookback_gap_days = 回看段 (lb_start -> earliest_E-1) 按日线交易日历的缺口; 应空 (run_sheet 设计为连续抓)",
            "E_days 之间的间隔是设计性的 (只在 E 日抓分钟), 不算缓存缺口",
            "slot_dist: <20260624 期望 241 槽 (无 13:00), >=20260624 期望 242 槽 (含 13:00 源格式)",
            "abnormal_slot_days: 槽位数非 241/242 的日 (数据质量异常, 需排查)",
            "date_min/date_max 仅端点, 不代表中间连续; 看 lookback_gap_days 判断连续性",
            "补抓范围与预算不在此冻结 (属第 3 步, 待 REV2-D/E 引擎字段补全+合成测试验收后再探测)",
        ],
    }
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")

    # 控制台打印
    print(f"== GCP v0.2 缓存覆盖核验 (REV2-A) ==")
    print(f"报告: {REPORT}")
    print(f"代码-文件名映射: {'OK' if report['code_mapping_check']['mapping_ok'] else 'FAIL'}")
    print(f"旧任务单错配代码 (不在当前 run_sheet): {stale_not_in_sheet}")
    print(f"汇总: {n_ok}/{len(rows)} OK, {n_gaps} 有缺口, {len(short_span)} 股跨度<6月")
    print(f"{'code':>8} {'setup':>6} {'min':>10} {'max':>10} {'span_m':>7} "
          f"{'reg':>5} {'have':>5} {'lb_gap':>7} {'abn':>4}")
    for r in rows:
        span = r.get("lookback_span_months")
        print(f"{r['code']:>8} {r['setup']:>6} {r.get('date_min','-'):>10} "
              f"{r.get('date_max','-'):>10} {span if span is not None else '-':>7} "
              f"{r['registered_fetch_days']:>5} {r['have_days']:>5} "
              f"{r['n_lookback_gaps']:>7} {len(r['abnormal_slot_days']):>4}")
    if short_span:
        print(f"\n短跨度股 (<6月, 候选补抓, 但范围不在此冻结): "
              f"{[r['code'] for r in short_span]}")


if __name__ == "__main__":
    main()
