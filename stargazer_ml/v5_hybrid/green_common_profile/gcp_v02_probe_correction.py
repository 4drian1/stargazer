"""GCP v0.2 历史覆盖探测 — 纠正版 (REV2-F-correction, 2026-09-29).

用户驳回 v1 探测 (gcp_v02_history_coverage_probe.py) 的两处口径问题:
  1. Phase4 只比较缓存与 HTTP 的行数, 未逐位比对时间戳. 不能据此判"源格式差异, 按实际
     槽位处理放行". 须找出缺的是哪个时间戳: 端点差异(0930/1130/1300/1500)按版本化网格;
     交易时段内部缺口(如 1124/1029/1410)则涉及该缺口的窗口必须 INVALID.
  2. Phase2 把"日线有行"当"真实交易日已扣停牌", 证据不足. 000016 扩展区间仅 2 行,
     301584 长达十个月仅 34 行. 须核对上市日期(daily earliest 作代理)+ 日线连续性;
     对 daily 缺行日期用分钟 HTTP 验证(0 量=停牌占位 / 有量=日线覆盖缺口须补); 未知日期
     单列, 不冒充停牌. 865 是 daily 已列出的请求数, 不一定是完整扩展范围.

本脚本做轻量核对并冻结**明确的 (code, date) 请求清单 + SHA256**, 不再只冻结总数.
2024 不补抓的结论保留, 但限定为"当前 canonical 源和研究范围不可取".

输出: gcp_round_b_small_sample_v1/v02_history_probe_correction.json
      + v02_backfill_request_list.txt  (每行 code,date; SHA256 同步写 JSON)
"""
import hashlib
import json
import sys
from bisect import bisect_left
from pathlib import Path

import pandas as pd

HERE = Path(__file__).parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "step9_preflight"))
from data_source_provenance import get_minute_http, get_daily, get_daily_parquet  # noqa: E402

OUT = HERE / "gcp_round_b_small_sample_v1"
SHEET = json.loads((OUT / "run_sheet.json").read_text(encoding="utf-8"))
REV2A = json.loads((OUT / "v02_cache_coverage_probe.json").read_text(encoding="utf-8"))
REPORT = OUT / "v02_history_probe_correction.json"
LIST_TXT = OUT / "v02_backfill_request_list.txt"
SNAP = OUT / "minute"

CANDIDATES = ["300793", "000042", "301584", "000078", "603221", "000016", "301357"]
WIN_START = SHEET["constraints"]["minute_window"][0]   # 20250102


def expected_slot_set(with_1300):
    """标准交易时段 HHMMSS 集. 不含13:00=241(0930-1130 + 1301-1500); 含13:00=242."""
    am = pd.date_range("09:30", "11:30", freq="1min")
    pm = (pd.date_range("13:00", "15:00", freq="1min") if with_1300
          else pd.date_range("13:01", "15:00", freq="1min"))
    out = set()
    for t in am:
        out.add(int(t.strftime("%H%M%S")))
    for t in pm:
        out.add(int(t.strftime("%H%M%S")))
    return out


E241 = expected_slot_set(False)
E242 = expected_slot_set(True)


def categorize_minute(code, d):
    """对 (code, d) 探针分钟 HTTP, 返回 (category, rows, vol_sum)."""
    m = get_minute_http(code, str(d))
    if m.empty:
        return "SOURCE_MISSING", 0, 0
    vol = int(m["volume"].sum()) if "volume" in m else 0
    if vol == 0:
        return "ZERO_VOLUME_PLACEHOLDER", int(len(m)), 0
    return "OK", int(len(m)), vol


def main():
    panel = get_daily_parquet()
    panel["code"] = panel["code"].astype(str).str.zfill(6)
    cal = sorted(int(x) for x in panel["date"].unique())
    cal_set = set(cal)
    daily_by_code = {c: s for c, s in panel.groupby("code")}

    # ============ Phase A: 上市代理 + 日线连续性 + daily缺行分钟验证 ============
    phaseA = []
    fetch_dates_by_code = {}   # code -> sorted list[int] (最终补抓清单)
    for st in SHEET["stocks"]:
        code = st["code"]
        if code not in CANDIDATES:
            continue
        lb = st["lookback_start"]
        px = daily_by_code.get(code)
        if px is None or px.empty:
            phaseA.append({"code": code, "status": "NO_DAILY"})
            fetch_dates_by_code[code] = []
            continue
        earliest_daily = int(px["date"].min())
        # 扩展区间起点 = max(窗口起点, 上市代理). 上市晚的股(如301584)从上市日起, 不从20250102
        ext_start = max(WIN_START, earliest_daily)
        ext_end = lb
        daily_present = sorted(int(x) for x in px["date"].unique()
                               if ext_start <= x < ext_end)
        cal_in_range = [x for x in cal if ext_start <= x < ext_end]
        daily_missing = [x for x in cal_in_range if x not in set(daily_present)]

        # 上市前探测: 若 earliest_daily > WIN_START, 探一个月前确认无分钟(证上市而非daily覆盖缺口)
        pre_ipo_check = None
        if earliest_daily > WIN_START:
            # 取 earliest_daily 前 ~21 交易日的一个点
            i = bisect_left(cal, earliest_daily)
            probe_d = cal[max(0, i - 21)] if i >= 21 else cal[0]
            cat, rows, vol = categorize_minute(code, probe_d)
            pre_ipo_check = {"probe_date": int(probe_d), "category": cat,
                              "rows": rows, "vol": vol,
                              "confirms_listing": cat in ("SOURCE_MISSING", "ZERO_VOLUME_PLACEHOLDER")}

        # daily 缺行日期: 分钟 HTTP 验证 (0量=停牌占位 / 有量=日线覆盖缺口须补)
        missing_probe = []
        extra_fetch_from_gap = []
        for d in daily_missing:
            cat, rows, vol = categorize_minute(code, d)
            missing_probe.append({"date": int(d), "category": cat, "rows": rows, "vol": vol})
            if cat == "OK":
                extra_fetch_from_gap.append(int(d))   # daily 缺但分钟有量 -> 补
            # ZERO_VOLUME_PLACEHOLDER / SOURCE_MISSING -> 不补 (停牌占位 / 源缺)

        fetch_dates = sorted(set(daily_present) | set(extra_fetch_from_gap))
        fetch_dates_by_code[code] = fetch_dates
        phaseA.append({
            "code": code, "setup": st["setup"],
            "earliest_daily": earliest_daily,
            "listing_proxy_note": ("新上市: daily earliest=%d > 窗口起点, 扩展从上市日起" % earliest_daily
                                   if earliest_daily > WIN_START else "上市早于窗口起点"),
            "pre_ipo_check": pre_ipo_check,
            "extend_range": [ext_start, ext_end],
            "calendar_trading_days": len(cal_in_range),
            "daily_present": len(daily_present),
            "daily_missing": len(daily_missing),
            "missing_probe": missing_probe,
            "missing_categorization": {
                "ZERO_VOLUME_PLACEHOLDER": sum(1 for m in missing_probe if m["category"] == "ZERO_VOLUME_PLACEHOLDER"),
                "SOURCE_MISSING": sum(1 for m in missing_probe if m["category"] == "SOURCE_MISSING"),
                "OK_daily_gap": sum(1 for m in missing_probe if m["category"] == "OK"),
            },
            "fetch_dates_count": len(fetch_dates),
            "fetch_dates_are": "daily_present + daily_missing中分钟有量的(OK_daily_gap)",
            "note": "865 v1 只数 daily_present; 本纠正版加入 OK_daily_gap 并排除 0量/源缺",
        })

    # ============ Phase B: 异常槽位日 时间戳级比对 (纠正 v1 只比行数) ============
    phaseB = []
    for r in REV2A.get("stocks", []):
        code = r["code"]
        abn = r.get("abnormal_slot_days", {})
        if not abn:
            continue
        # 加载缓存
        pf = SNAP / f"{code}.parquet"
        if not pf.exists():
            continue
        cdf = pd.read_parquet(pf)
        cdf["day"] = cdf["date"].astype(str).str[:8].astype(int)
        cdf["hm"] = cdf["date"].astype(str).str[8:].astype(int)
        checks = []
        for d_str, _cached_n in abn.items():
            d = int(d_str)
            csub = cdf[cdf["day"] == d]
            cache_set = set(csub["hm"].unique().tolist()) if not csub.empty else set()
            m = get_minute_http(code, str(d))
            if m.empty:
                checks.append({"date": d, "cache_slots": len(cache_set), "http_slots": 0,
                               "cache_http_same": False, "missing_vs_expected": "HTTP_EMPTY"})
                continue
            m["hm"] = m["date"].astype(str).str[8:].astype(int)
            http_set = set(m["hm"].unique().tolist())
            # 选期望集: >=20260624 用242, 否则241 (13:00源格式)
            E = E242 if d >= 20260624 else E241
            miss_cache = sorted(E - cache_set)
            miss_http = sorted(E - http_set)
            # 端点 vs 盘中
            endpoints = {93000, 113000, 130000, 150000}
            miss_is_endpoint = all(x in endpoints for x in miss_cache) if miss_cache else False
            diag = ("ENDPOINT_VERSION_DIFF" if miss_is_endpoint
                    else "INTRADAY_GAP_WINDOW_INVALID" if miss_cache else "FULL_GRID")
            checks.append({
                "date": d, "cache_slots": len(cache_set), "http_slots": len(http_set),
                "cache_http_same": cache_set == http_set,
                "missing_from_cache_vs_expected": miss_cache,
                "missing_from_http_vs_expected": miss_http,
                "diagnosis": diag,
                "impact": ("端点缺: 按版本化网格处理, 窗口可补齐端点" if miss_is_endpoint
                            else "盘中缺口: 涉及该缺口的窗口必须 window_complete=false (INVALID)" if miss_cache
                            else "完整: 正常处理"),
            })
        n_intraday = sum(1 for c in checks if c.get("diagnosis") == "INTRADAY_GAP_WINDOW_INVALID")
        phaseB.append({"code": code, "abnormal_days_checked": len(checks),
                        "n_intraday_gap": n_intraday,
                        "n_endpoint_diff": sum(1 for c in checks if c.get("diagnosis") == "ENDPOINT_VERSION_DIFF"),
                        "checks": checks})

    # ============ 冻结明确 (code, date) 请求清单 + SHA256 ============
    request_lines = []
    per_code_summary = []
    for code in CANDIDATES:
        dates = fetch_dates_by_code.get(code, [])
        for d in dates:
            request_lines.append(f"{code},{d}")
        per_code_summary.append({"code": code, "n_dates": len(dates),
                                  "range": [dates[0], dates[-1]] if dates else None})
    request_lines.sort()
    list_text = "\n".join(request_lines) + ("\n" if request_lines else "")
    sha256 = hashlib.sha256(list_text.encode("utf-8")).hexdigest()
    LIST_TXT.write_text(list_text, encoding="utf-8")

    total = len(request_lines)

    report = {
        "report_version": "GCP_V02_HISTORY_PROBE_CORRECTION",
        "date": "2026-09-29",
        "stage": "REV2-F 纠正版 (用户驳回 v1 两处口径后)",
        "purpose": "纠正 v1: (1)Phase4 时间戳级比对 (2)Phase2 上市代理+daily缺行分钟验证; 冻结明确(code,date)清单+SHA256",
        "v1_flaws_fixed": [
            "v1 Phase4 只比行数 -> 本版逐位比对 HHMMSS, 区分端点差(版本化网格) vs 盘中缺口(窗口INVALID)",
            "v1 Phase2 把daily有行当真实交易日 -> 本版用earliest_daily作上市代理, daily缺行用分钟HTTP验证(0量=停牌/有量=日线覆盖缺口须补)",
        ],
        "phaseA_listing_and_continuity": phaseA,
        "phaseB_slot_timestamp_recheck": phaseB,
        "frozen_request_list": {
            "total_code_date_pairs": total,
            "per_code": per_code_summary,
            "list_file": str(LIST_TXT.name),
            "sha256": sha256,
            "format": "每行 code,date (code 6位, date 8位 YYYYMMDD); 按字典序排序后 SHA256",
            "composition": "daily_present 在 [max(20250102,earliest_daily), lookback_start) + daily缺行中分钟有量(OK_daily_gap)",
            "excluded": "daily缺行中 0量占位(ZERO_VOLUME_PLACEHOLDER,停牌) + 源缺(SOURCE_MISSING) + 上市前(pre-IPO)",
        },
        "key_findings": {
            "301584_is_new_ipo": "earliest_daily=20250925, 扩展从上市日起非20250102; pre-IPO 日期N/A不计",
            "000016_8_missing": "daily缺8日均返回241条0量bar -> 停牌占位, 不补",
            "301357_10_missing": "daily缺10日(20250526-20250609)均返回241条0量bar -> 停牌占位, 不补",
            "abnormal_slots_are_intraday_gaps": "非端点(1124/1029/1410等); 涉及该缺口的窗口必须INVALID, v1'按实际槽位处理'错误",
        },
        "scope_limit": "2024不补抓结论保留, 限定为'当前canonical源和研究范围不可取', 非所有来源都无2024分钟",
        "next": [
            "按 v02_backfill_request_list.txt 执行一次补抓; 已有缓存复用; 源缺保留原因不插值不无限重试",
            "新快照追加保存, 旧验收输入保留; 逐日验证网格/价格/量额单位",
            "生成v3稳定性报告: 60m主检验+时间切分+闭环边界, 与组统计比预测误差",
            "有预测增量才制定决策合同, 再验交易收益; 仍不重训/不接线上",
        ],
    }
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")

    # ---- 控制台 ----
    print("== GCP v0.2 历史探测纠正版 (REV2-F-correction) ==")
    print(f"报告: {REPORT}")
    print(f"清单: {LIST_TXT}")
    print(f"\n[PhaseA] 上市代理 + daily缺行验证:")
    print(f"  {'code':>8} {'earliest':>9} {'ext_range':>22} {'cal':>5} {'daily':>5} {'miss':>5} {'0量':>4} {'OKgap':>6} {'fetch':>6}")
    for a in phaseA:
        rng = f"{a['extend_range'][0]}-{a['extend_range'][1]}"
        mc = a.get("missing_categorization", {})
        print(f"  {a['code']:>8} {a['earliest_daily']:>9} {rng:>22} "
              f"{a['calendar_trading_days']:>5} {a['daily_present']:>5} {a['daily_missing']:>5} "
              f"{mc.get('ZERO_VOLUME_PLACEHOLDER',0):>4} {mc.get('OK_daily_gap',0):>6} {a['fetch_dates_count']:>6}")
    print(f"\n[PhaseB] 异常槽位日时间戳级比对:")
    for b in phaseB:
        print(f"  {b['code']}: 检查{b['abnormal_days_checked']}日, 盘中缺口(INVALID){b['n_intraday_gap']}, 端点差{b['n_endpoint_diff']}")
    print(f"\n[冻结清单] {total} 个 (code,date) 对, SHA256={sha256[:16]}...")
    for s in per_code_summary:
        r = f"{s['range'][0]}-{s['range'][1]}" if s['range'] else "none"
        print(f"  {s['code']}: {s['n_dates']} 日 [{r}]")


if __name__ == "__main__":
    main()
