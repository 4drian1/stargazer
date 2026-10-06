"""
双路径 E 日网格异常 · 逐分钟只读核对 (①)
========================================
目标: 厘清覆盖损失, 不把异常样本"救回"漏斗; 原始缓存/冻结切分/现有报告均不改.

做法:
- 读采集报告的 grid_anomaly_detail (127 条).
- 对每条 (code,E) 复用引擎 E.validate_minute_day 取缺失精确槽 (HHMMSS);
  缺失根数按日期对应 241/242 槽网格计算 (day<20260624→241, 否则 242).
- 复用探针 scan_views 在"不完整数据"上扫出 WATCH/S1/S2/action 位置,
  将缺失分钟交叉到: 信号本身 / S2 前 20 分钟滚动量比窗 / 下一分钟模拟成交.
- 三桶分类:
    明确无影响      = 已触发行动且关键依赖(信号ts/量比窗/下一分钟)完整(缺失落在别处)
    影响关键依赖    = 缺失分钟恰好落在信号ts / S2量比窗 / 下一分钟成交
    无法判断        = 缺失在盘中(会话槽)但不在可证信号位; 或 No_box/无信号 → 不能断言无影响
- 12 条 271 槽单列(part B): 不入 115 条的分桶; 核对盘中是否完整 + 盘后多余分钟的量价,
  来源待与北交所规则/日线对账 (2025 北交所盘后固定价交易尚未实施, 撤回"盘后固定价"定性).

network_requests = 0. 仅读本地缓存/报告/日线/池, 写一份新审计 JSON + 打印.
"""

from __future__ import annotations
import sys, os, json, time
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import gcp_round_b_engine as E
import gcp_dual_path_coverage_probe as P

OUT = P.OUT
REPORT_PATH = OUT / "dual_path_e_day_collect_report.json"
GRID_PATH = OUT / "minute_grid.json"
AUDIT_JSON = OUT / "dual_path_e_day_per_minute_audit.json"

# ---------------- 槽工具 ----------------
def slot_hm(slot: int):
    s = f"{int(slot):06d}"
    return int(s[0:2]), int(s[2:4])

def hm_str(slot: int) -> str:
    h, m = slot_hm(slot)
    return f"{h:02d}:{m:02d}"

def to_min(slot: int) -> int:
    h, m = slot_hm(slot)
    return h * 60 + m

def from_min(mm: int) -> int:
    h, m = divmod(mm, 60)
    return h * 10000 + m * 100

# ---------------- 加载 ----------------
def main():
    t0 = time.time()
    print("=" * 64)
    print("双路径 E 日网格异常 · 逐分钟只读核对 (①, 零新请求)")
    print("=" * 64)

    report = json.loads(REPORT_PATH.read_text(encoding="utf-8"))
    anomalies = report.get("grid_anomaly_detail", [])
    print(f"采集报告异常条数: {len(anomalies)}")

    grid = json.loads(GRID_PATH.read_text(encoding="utf-8"))["grid"]  # 242 槽

    panel = P.get_daily_parquet()
    panel["code"] = panel["code"].astype(str).str.zfill(6)
    panel["date"] = panel["date"].astype(int)
    cal = sorted(int(d) for d in panel["date"].unique())
    cal_idx = {d: i for i, d in enumerate(cal)}
    daily_by_code = {c: g.sort_values("date").reset_index(drop=True)
                     for c, g in panel.groupby("code")}

    pool_a = pd.read_parquet(P.POOL_DIR / "pool_a_sdp_members.parquet")
    pool_b = pd.read_parquet(P.POOL_DIR / "pool_b_lvb_members.parquet")
    pool_a["code"] = pool_a["code"].astype(str).str.zfill(6)
    pool_b["code"] = pool_b["code"].astype(str).str.zfill(6)
    # 按 (code, P) 归属单一 setup, 与探针漏斗口径一致 (同一 code 可在不同 P 日分属 A/B)
    pa = set((c, int(p)) for c, p in pool_a[["code", "date"]].itertuples(index=False))
    pb = set((c, int(p)) for c, p in pool_b[["code", "date"]].itertuples(index=False))

    minute_idx = P.load_minute_index()

    # ---------------- 逐条审计 ----------------
    part_a = []          # 115 条 (237-240 + 36)
    part_b = []          # 12 条 271
    for g in anomalies:
        code = str(g["code"]).zfill(6)
        e = int(g["E"])
        reason = g.get("reason", "")
        p_day = cal[cal_idx[e] - 1] if (e in cal_idx and cal_idx[e] >= 1) else None
        setup = ("A_SDP" if (code, p_day) in pa else
                 ("B_LVB" if (code, p_day) in pb else "UNKNOWN"))
        seg = P.segment_of(e)

        dc = daily_by_code.get(code)
        box_valid = False
        upper = lower = atr_ref = None
        if dc is not None and e in cal_idx and cal_idx[e] >= 1:
            p_day = cal[cal_idx[e] - 1]
            loc = int(dc["date"].searchsorted(p_day, side="right"))
            if loc >= 20:
                d_sub = dc.iloc[:loc]
                atr_ref = E.build_atr_ref(d_sub)
                calibs = P.evaluate_calibers(d_sub, atr_ref)
                mcb = calibs.get("main", {})
                box_valid = bool(mcb.get("valid"))
                upper = mcb.get("upper")
                lower = mcb.get("lower")

        mkeys = minute_idx.get((code, e))
        e_df = None
        if mkeys:
            raw = pd.read_parquet(mkeys[0][0])
            day_int = raw["date"].astype(str).str[:8].astype(int)
            e_df = raw[day_int == e].sort_values("date").reset_index(drop=True)

        v = E.validate_minute_day(e_df, grid, e)
        present = len(e_df) if e_df is not None else 0
        expected = v["expected_slots"]
        missing = v["missing"]            # 盘中缺失 HHMMSS
        postclose = v["unexpected"]       # 多出(盘后) HHMMSS

        # 分组
        if present > expected:
            group = "extended_271"
        elif len(missing) >= 100:
            group = "severe_36"
        else:
            group = "partial_237_240"

        rec = {
            "code": code, "E": e, "setup": setup, "segment": seg,
            "reason_raw": reason, "group": group,
            "present": present, "expected": expected,
            "missing_count": len(missing),
            "missing_hhmm": [hm_str(s) for s in missing],
            "postclose_count": len(postclose),
            "postclose_hhmm": [hm_str(s) for s in postclose],
            "dups": [hm_str(s) for s in v["dups"]],
            "box_valid": box_valid,
        }

        if group == "extended_271":
            # Part B: 独立对账, 不随 115 放行
            insession_missing = len(missing)  # 应为 0
            pc_vol = 0.0
            pc_pmin = pc_pmax = None
            last_close = None
            if e_df is not None:
                e_df["hhmm"] = e_df["date"].apply(E.hhmm)
                pc = e_df[e_df["hhmm"] >= 150100]
                ins = e_df[e_df["hhmm"] <= 150000]
                pc_vol = float(pc["volume"].sum()) if len(pc) else 0.0
                if len(pc):
                    pc_pmin = float(pc["close"].min())
                    pc_pmax = float(pc["close"].max())
                if len(ins):
                    last_close = float(ins.sort_values("date").iloc[-1]["close"])
            rec.update({
                "insession_missing": insession_missing,
                "postclose_vol": round(pc_vol, 0),
                "postclose_close_min": pc_pmin,
                "postclose_close_max": pc_pmax,
                "insession_last_close_1500": last_close,
                "note": "盘中槽完整(缺失=0); 多出为 15:01-15:30 分钟; "
                        "2025 北交所盘后固定价交易尚未实施(2026 修订才新增), 来源待量价/日线对账",
            })
            part_b.append(rec)
            continue

        # Part A: 115 条
        views = None
        if box_valid and e_df is not None and upper is not None:
            views = P.scan_views(e_df, upper, lower, atr_ref, True, True)
        rec["views"] = {k: views.get(k) for k in
                        ("watch", "s1_ts", "s2_ts", "action", "action_ts", "s2_reason")
                        } if views else None

        # 相对四个锚点的位置: 完整监控窗 / 首次行动 / 量比回看窗 / 下一分钟成交
        # 监控窗 [09:35,14:55] (HHMM int); 行动=action_ts; 量比窗=S2 前20min; 下一分钟成交=action_ts+1min
        MON_START, MON_END = 935, 1455
        action_ts = views.get("action_ts") if views else None
        action_slot = int(str(int(action_ts))[-6:]) if action_ts is not None else None
        s2_ts = views.get("s2_ts") if views else None
        vol_win = None
        if s2_ts is not None:
            s2_slot = int(str(int(s2_ts))[-6:])
            vol_win = (from_min(to_min(s2_slot) - 20), s2_slot)  # (起,止) 含
        next_fill = from_min(to_min(action_slot) + 1) if action_slot is not None else None

        gap_positions = []
        for m in missing:
            hhmm_int = (m // 10000) * 100 + ((m // 100) % 100)
            if hhmm_int < MON_START:
                mon_rel = "pre_monitor"      # 监控窗前(如 09:30/09:31)
            elif hhmm_int > MON_END:
                mon_rel = "post_monitor"     # 监控窗后(如 15:00)
            else:
                mon_rel = "in_monitor"
            if action_slot is None:
                act_rel = "no_action"
            elif m < action_slot:
                act_rel = "before_action"    # 行动之前(含 S2 前20min量比窗)
            elif m == action_slot:
                act_rel = "at_action"
            elif m == next_fill:
                act_rel = "at_next_fill"     # 下一分钟成交(依赖末端)
            else:
                act_rel = "after_action"     # 晚于全部依赖
            gap_positions.append({"hhmm": hm_str(m), "monitor": mon_rel, "action": act_rel})
        rec["gap_positions"] = gap_positions

        # 重新分级 (① 收紧版)
        # 可证明不影响该行动 := 存在已观察行动 且 每个缺口分钟都严格晚于 下一分钟成交
        #   (即晚于该行动及其全部依赖; 缺口若在行动之前/量比窗/下一分钟, 则无法排除更早触发)
        # 否则(缺口在行动前/可能隐藏更早触发/当天无观察信号) => 无法判断
        # 36 槽严重不完整 => severe_incomplete (单列, 不挽救)
        if group == "severe_36":
            bucket = "severe_incomplete"
        elif action_slot is not None and next_fill is not None and len(missing) > 0 \
                and all(m > next_fill for m in missing):
            bucket = "可证明不影响该行动"
        else:
            bucket = "无法判断"
        rec["bucket"] = bucket
        part_a.append(rec)

    # ---------------- 汇总 ----------------
    buckets_order = ["可证明不影响该行动", "无法判断", "severe_incomplete"]
    crosstab = {}
    totals = {b: 0 for b in buckets_order}
    for r in part_a:
        key = f"{r['segment']}|{r['setup']}"
        crosstab.setdefault(key, {b: 0 for b in buckets_order})
        crosstab[key][r["bucket"]] += 1
        totals[r["bucket"]] += 1

    group_counts = {}
    for r in part_a:
        group_counts[r["group"]] = group_counts.get(r["group"], 0) + 1

    print(f"\n[Part A] 盘中缺失组 (115 条): group={group_counts}")
    print(f"  三桶合计: " + ", ".join(f"{b}={totals[b]}" for b in buckets_order))
    print("\n  按 段|池 分桶:")
    for key in sorted(crosstab):
        c = crosstab[key]
        print(f"    {key}: " + ", ".join(f"{b}={c[b]}" for b in buckets_order))

    # 聚焦: val/test × A/B 数量变化
    print("\n  [聚焦] val/test × A/B (可证明不影响 / 无法判断 / severe):")
    for seg in ("val", "test"):
        for setup in ("A_SDP", "B_LVB"):
            key = f"{seg}|{setup}"
            if key in crosstab:
                c = crosstab[key]
                print(f"    {seg} {setup}: 可证={c['可证明不影响该行动']} "
                      f"无法判={c['无法判断']} severe={c['severe_incomplete']}")

    print("\n  逐条 (code E 段 池 group 缺失数 缺失分钟 bucket):")
    for r in part_a:
        miss = ",".join(r["missing_hhmm"]) if r["missing_hhmm"] else "-"
        print(f"    {r['code']} {r['E']} {r['segment']} {r['setup']} "
              f"{r['group']} miss={r['missing_count']} [{miss}] {r['bucket']}")

    print(f"\n[Part B] 271 槽扩展组 (独立对账, 不随 115 放行): {len(part_b)} 条")
    insession_all_ok = all(r["insession_missing"] == 0 for r in part_b)
    print(f"  盘中(≤15:00)缺失全为 0: {insession_all_ok}")
    for r in part_b:
        pc = ",".join(r["postclose_hhmm"][:6]) + ("..." if len(r["postclose_hhmm"]) > 6 else "")
        print(f"    {r['code']} {r['E']} {r['segment']} {r['setup']} "
              f"present={r['present']} 盘后多={r['postclose_count']} [{pc}] "
              f"盘后量={r['postclose_vol']:.0f} 15:00收={r['insession_last_close_1500']}")

    out = {
        "audit": "per_minute_grid_anomaly_v1",
        "network_requests": 0,
        "rules": {
            "expected_grid": "day<20260624→241 槽(无13:00); 否则 242 槽 (引擎 expected_grid)",
            "three_buckets": "可证明不影响该行动=存在已观察行动且每个缺口分钟都严格晚于该行动的全部依赖(含下一分钟成交); "
                             "无法判断=缺口在行动之前/可能隐藏更早触发/影响特征/当天无观察信号; "
                             "severe_incomplete=36槽严重不完整(单列, 不挽救); "
                             "本次不含'影响关键依赖'独立桶——缺口若在依赖位置已归入无法判断(无法证明无更早触发)",
            "grouping": "present>expected→extended_271(单列PartB); "
                        "盘中缺失≥100→severe_36; 否则 partial_237_240(PartA)",
            "split_points_unchanged": "train E<=20260305 / val 20260309..20260609 / test E>=20260610",
        },
        "collect_report_anomaly_total": len(anomalies),
        "part_a_in_session_missing": {
            "n": len(part_a),
            "group_counts": group_counts,
            "bucket_totals": totals,
            "by_segment_setup": crosstab,
            "records": part_a,
        },
        "part_b_extended_271": {
            "n": len(part_b),
            "insession_complete_all": insession_all_ok,
            "note": "盘中槽完整; 多出分钟为 15:01-15:30; 来源待与北交所规则/日线对账, "
                    "不随 115 条放行, 不计入正式有效样本",
            "records": part_b,
        },
        "elapsed_s": round(time.time() - t0, 1),
    }
    AUDIT_JSON.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n审计 JSON: {AUDIT_JSON.name}  (耗时 {out['elapsed_s']}s, network_requests=0)")


if __name__ == "__main__":
    main()
