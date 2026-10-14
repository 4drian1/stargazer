"""R1 覆盖盘点 — 只读、零网络、零训练、零阈值变更.

基于 v0.1.1 执行合同 (dual_path_execution_contract_v0_1_1_final.md),
对 **1,013 研究人口** (gcp_247_full_population_inventory_v3.json, FROZEN) 做逐事件覆盖盘点.

人口来源 (FROZEN, 不混批次):
  - gcp_247_full_population_inventory_v3.json — 1,013 人口 + 互斥四类状态
  - gcp_247_ruleA_trigger_records_v1.json    — 187 action 触发记录 (S1/S2/双触发)
  - gcp_247_return_path_coverage_v1.json     — 374 收益路径轨道 (H3/H5 × 187)

注: 251/247 是不同批次 (251 事件缓存 → 排除 BSE/ETF → 247), 与 1,013 研究人口不重叠;
    仅作旁注说明 (v3 records 中 in_251/in_247 标志交叉).

六项修正 (保留):
  1. 精确对账人口: 1,050 → 排除 37 ETF/BSE (人口层) → 1,013 研究人口 →
     互斥四类 (187/277/245/304) → action → fillable → label_available
  2. 逐事件前向收益: v3 exit_path_coverage (E+1..E+5 四定点) +
     return_path_coverage_v1 (H3/H5 含顺延) 双口径, 逐事件分类
  3. 正确基线: 全买/全拒/固定时间入场, 同 v0.1.1 口径, 仅改变入场决策
  4. WATCH 非天然负样本: NOT_ACTED / NOT_PROFITABLE / INSUFFICIENT_DATA 三态
  5. 已收紧缺槽规则: 按日期版本 (241/242), 按依赖链, 不放松 127 异常日
  6. DIP_REPAIR 仅按冻结定义; 本批 ruleA 冻结件仅 S1/S2/双触发, DIP_REPAIR 不在册
     S1≠BASE_HOLD 未经核实, 标 PENDING_VERIFICATION
"""
from __future__ import annotations

import hashlib
import json
import time
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).parent
OUT = HERE / "gcp_round_b_small_sample_v1"
CONTRACTS = HERE / "contracts"
REPORT_JSON = OUT / "r1_coverage_inventory.json"

# FROZEN 人口来源 (1,013 研究人口, 不混批次)
V3_INVENTORY = OUT / "gcp_247_full_population_inventory_v3.json"
RULEA_TRIGGERS = OUT / "gcp_247_ruleA_trigger_records_v1.json"
RETURN_PATH_COVERAGE = OUT / "gcp_247_return_path_coverage_v1.json"

# 251/247 批次 (旁注, 不混入; 不同批次)
EVENTS_CACHE_251 = HERE / "_251_events_cache.json"

# v0.1.1 §2 冻结参数
DATA_CUTOFF_CONTRACT = 20260928


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def safe_sha256(path) -> str:
    """安全哈希: 文件不存在返回 'MISSING'."""
    p = Path(path)
    try:
        return sha256_file(p)
    except Exception:
        return "MISSING"


def main():
    t0 = time.time()
    print("=" * 70)
    print("R1 覆盖盘点 (只读, 零网络, 零训练, 零阈值变更)")
    print("人口来源: gcp_247_full_population_inventory_v3.json (FROZEN 1,013 人口)")
    print("合同: dual_path_execution_contract_v0_1_1_final.md")
    print("=" * 70)

    # ================= 1. 加载 FROZEN 人口来源 =================
    v3 = json.loads(V3_INVENTORY.read_text(encoding="utf-8"))
    ruleA = json.loads(RULEA_TRIGGERS.read_text(encoding="utf-8"))
    ret_cov = json.loads(RETURN_PATH_COVERAGE.read_text(encoding="utf-8"))

    v3_records = v3["records"]
    n_population = len(v3_records)
    print(f"FROZEN v3 inventory: {n_population} 研究人口 "
          f"(header study_population_n={v3.get('study_population_n')})")

    # 索引 ruleA records by (code, E) — 交叉校验 action_kind / first_action_ts / double_trigger
    ruleA_by_key: dict = {}
    for r in ruleA["records"]:
        key = (str(r["code"]).zfill(6), int(r["E"]))
        ruleA_by_key[key] = r
    print(f"ruleA trigger records: {len(ruleA['records'])} 条 "
          f"(n_action={ruleA.get('n_action')}, n_double_trigger={ruleA.get('n_double_trigger')})")

    # 索引 return_path_coverage tracks by (code, E, track) — H3/H5 含顺延的逐日覆盖
    return_tracks: dict = {}
    for t in ret_cov.get("tracks", []):
        key = (str(t["code"]).zfill(6), int(t["E"]), t["track"])
        return_tracks[key] = t
    print(f"return_path_coverage tracks: {len(ret_cov.get('tracks', []))} 条 "
          f"(n_tracks={ret_cov.get('n_tracks')}, n_action_candidates={ret_cov.get('n_action_candidates')})")

    # ================= 2. 251/247 批次旁注 (不混入) =================
    n_251_batch = 0
    n_247_batch = 0
    n_bse_251 = 0
    n_etf_251 = 0
    if EVENTS_CACHE_251.exists():
        events_251 = json.loads(EVENTS_CACHE_251.read_text(encoding="utf-8"))
        n_251_batch = len(events_251)
        bse = [e for e in events_251 if e.get("exchange_board") == "BSE"]
        etf = [e for e in events_251
               if e.get("instrument_type") == "ETF"
               or e.get("exchange_board") == "FUND_LISTING"]
        n_bse_251 = len(bse)
        n_etf_251 = len(etf)
        n_247_batch = len([e for e in events_251
                           if e.get("exchange_board") != "BSE"
                           and e.get("instrument_type") != "ETF"
                           and e.get("exchange_board") != "FUND_LISTING"])
    print(f"\n[251/247 批次旁注 (不混入 1,013 人口)]")
    print(f"  251 事件缓存: {n_251_batch} → 排除 BSE={n_bse_251} + ETF={n_etf_251} → "
          f"{n_247_batch} 在范围 (不同批次)")
    print(f"  注: 251/247 与 1,013 研究人口不重叠; v3 records 中 in_251/in_247 "
          f"仅作交叉参考标志")

    # ================= 3. FROZEN 输入 SHA256 =================
    input_hashes = {
        "gcp_247_full_population_inventory_v3.json": safe_sha256(V3_INVENTORY),
        "gcp_247_ruleA_trigger_records_v1.json": safe_sha256(RULEA_TRIGGERS),
        "gcp_247_return_path_coverage_v1.json": safe_sha256(RETURN_PATH_COVERAGE),
        "_251_events_cache.json (旁注批次, 不混入)": safe_sha256(EVENTS_CACHE_251),
    }
    contract_path = CONTRACTS / "dual_path_execution_contract_v0_1_1_final.md"
    input_hashes["dual_path_execution_contract_v0_1_1_final.md"] = safe_sha256(contract_path)
    full_pop_contract = CONTRACTS / "full_population_rule_compare_contract_v1.md"
    input_hashes["full_population_rule_compare_contract_v1.md"] = safe_sha256(full_pop_contract)

    # ================= 4. 逐事件分析 (1,013 条) =================
    per_event = []
    for v3_rec in v3_records:
        code = str(v3_rec["code"]).zfill(6)
        E_day = int(v3_rec["E"])
        P_day = int(v3_rec["P"])
        setup = v3_rec["setup"]
        segment = v3_rec["segment"]
        mutual = v3_rec.get("mutual", "")
        in_251 = bool(v3_rec.get("in_251", False))
        in_247 = bool(v3_rec.get("in_247", False))
        e_day_cov = v3_rec.get("e_day_coverage", "UNKNOWN")
        watch_flag = v3_rec.get("watch")
        s1_ts = v3_rec.get("s1_ts")
        s2_ts = v3_rec.get("s2_ts")
        reason = v3_rec.get("reason")          # v3 自带 (S1/S2/双触发/E日缺槽)
        first_action = v3_rec.get("first_action")
        double_trigger_v3 = bool(v3_rec.get("double_trigger", False))
        exit_cov = v3_rec.get("exit_path_coverage") or {}

        # 从 ruleA 冻结件交叉获取 (与 v3 reason 互校)
        ruleA_rec = ruleA_by_key.get((code, E_day), {})
        action_kind = ruleA_rec.get("action_kind") or reason
        first_action_ts = ruleA_rec.get("first_action_ts") or first_action
        double_trigger = bool(ruleA_rec.get("double_trigger", double_trigger_v3))

        rec = {
            "code": code, "P": P_day, "E": E_day, "setup": setup,
            "segment": segment, "in_251": in_251, "in_247": in_247,
            "mutual_status": mutual,
            "e_day_coverage": e_day_cov,
            "watch": watch_flag,
            "s1_ts": s1_ts, "s2_ts": s2_ts,
            "action_kind": action_kind,
            "first_action_ts": first_action_ts,
            "double_trigger": double_trigger,
        }

        # ---- 互斥状态分类 + 事件类型 (按 v3 FROZEN 标签, 不重算触发) ----
        if mutual == "1_有行动":
            # 行动事件: 按 action_kind 映射 event_type
            if action_kind == "S1":
                event_type = "BASE_HOLD_S1"
                mapping_verified = False  # 修正6: S1→BASE_HOLD 待核实
            elif action_kind == "S2":
                event_type = "BASE_BREAKOUT_S2"
                mapping_verified = False  # 修正6: S2→BASE_BREAKOUT 待核实
            elif action_kind and "双触发" in str(action_kind):
                event_type = "BASE_HOLD_S1"  # 首触发 S1
                mapping_verified = False
            elif action_kind == "DIP_REPAIR":
                event_type = "DIP_REPAIR"
                mapping_verified = True
            else:
                event_type = "UNKNOWN_ACTION"
                mapping_verified = False
            action_ts = first_action_ts
            dip_repair_def_status = "NOT_TRACKED_IN_RULE_A"  # 修正6: ruleA 仅 S1/S2
        elif mutual == "2_WATCH但无行动":
            event_type = "WATCH_ONLY"
            action_ts = None
            mapping_verified = True
            dip_repair_def_status = "NA_NON_ACTION"
        elif mutual == "3_未WATCH且无行动":
            event_type = "NO_ACTION"
            action_ts = None
            mapping_verified = True
            dip_repair_def_status = "NA_NON_ACTION"
        elif mutual == "4_数据不足无法判断":
            event_type = "NO_ACTION"
            action_ts = None
            mapping_verified = True
            dip_repair_def_status = "NA_DATA_INSUFFICIENT"
        else:
            event_type = "UNKNOWN"
            action_ts = None
            mapping_verified = True
            dip_repair_def_status = "NA"

        # 触发列表 (修正6: 本批 ruleA 无 DIP_REPAIR)
        triggers = []
        if s1_ts is not None:
            triggers.append("S1")
        if s2_ts is not None:
            triggers.append("S2")
        if action_kind == "DIP_REPAIR":
            triggers.append("DIP_REPAIR")
        if not triggers:
            trigger_status = "NONE"
        elif len(triggers) == 1:
            trigger_status = "SINGLE"
        else:
            trigger_status = "BOTH"

        rec["event_type"] = event_type
        rec["event_type_mapping_verified"] = mapping_verified
        rec["trigger_status"] = trigger_status
        rec["triggers"] = triggers
        rec["dip_repair_def_status"] = dip_repair_def_status
        rec["action_ts"] = action_ts

        # ---- 可成交可行性 (行动事件: E 日分钟完整 + 有首触发时点) ----
        fill_status = "NO_ACTION"
        if mutual == "1_有行动":
            if e_day_cov == "COMPLETE" and action_ts is not None:
                fill_status = "FILLABLE"
            elif e_day_cov == "GAP":
                fill_status = "E_DAY_GAP"
            elif e_day_cov == "NO_FILE":
                fill_status = "NO_E_DAY_MINUTE"
            else:
                fill_status = "UNKNOWN_E_DAY"
        elif mutual == "4_数据不足无法判断":
            fill_status = "DATA_INSUFFICIENT"
        rec["fill_status"] = fill_status

        # ---- 前向收益覆盖 (修正2: 逐事件, 双口径) ----
        # 口径1: v3 exit_path_coverage (E+1/E+2/E+3/E+5 四定点, 无顺延)
        exit_e1 = exit_cov.get("E+1", "NO_FILE") if exit_cov else "NO_FILE"
        exit_e2 = exit_cov.get("E+2", "NO_FILE") if exit_cov else "NO_FILE"
        exit_e3 = exit_cov.get("E+3", "NO_FILE") if exit_cov else "NO_FILE"
        exit_e5 = exit_cov.get("E+5", "NO_FILE") if exit_cov else "NO_FILE"
        rec["exit_path_coverage"] = {
            "E+1": exit_e1, "E+2": exit_e2, "E+3": exit_e3, "E+5": exit_e5,
        }

        # 口径2: return_path_coverage_v1 轨道 (H3/H5, 含顺延逐日核对)
        h3_track = return_tracks.get((code, E_day, "H3"), {})
        h5_track = return_tracks.get((code, E_day, "H5"), {})
        h3_complete = bool(h3_track.get("all_checked_complete", False))
        h5_complete = bool(h5_track.get("all_checked_complete", False))
        h3_has_gap = bool(h3_track.get("has_gap_in_window", False))
        h5_has_gap = bool(h5_track.get("has_gap_in_window", False))
        h3_deferral_unchecked = int(h3_track.get("deferral_unchecked", 0))
        h5_deferral_unchecked = int(h5_track.get("deferral_unchecked", 0))

        # 提取 H3/H5 实际日期 (checked_dates 中最后日期 ≈ 计划退出日)
        h3_dates = [d.get("date") for d in h3_track.get("checked_dates", [])]
        h5_dates = [d.get("date") for d in h5_track.get("checked_dates", [])]
        h3_date = h3_dates[-1] if h3_dates else None
        h5_date = h5_dates[-1] if h5_dates else None

        rec["H3_date"] = h3_date
        rec["H5_date"] = h5_date
        rec["h3_track_complete"] = h3_complete
        rec["h5_track_complete"] = h5_complete
        rec["h3_has_gap"] = h3_has_gap
        rec["h5_has_gap"] = h5_has_gap
        rec["h3_deferral_unchecked"] = h3_deferral_unchecked
        rec["h5_deferral_unchecked"] = h5_deferral_unchecked

        # 分钟路径缺口 (行动事件: 用于 data_gaps)
        minute_path_missing = []
        if mutual == "1_有行动":
            for slot, st in [("E+1", exit_e1), ("E+2", exit_e2),
                             ("E+3", exit_e3), ("E+5", exit_e5)]:
                if st in ("NO_FILE", "GAP"):
                    minute_path_missing.append({"slot": slot, "status": st})
        rec["minute_path_missing"] = minute_path_missing
        rec["minute_path_ok"] = (len(minute_path_missing) == 0)

        # ---- 标签可用性分类 ----
        if mutual in ("2_WATCH但无行动", "3_未WATCH且无行动"):
            # 修正4: WATCH/无行动 非天然负样本 → NOT_ACTED
            decision_outcome = "NOT_ACTED"
            label_status = "NOT_APPLICABLE"
            label_reason = ("WATCH_ONLY_OBSERVATION" if mutual == "2_WATCH但无行动"
                            else "NO_TRIGGER")
        elif mutual == "4_数据不足无法判断":
            decision_outcome = "INSUFFICIENT_DATA"
            label_status = "MISSING"
            label_reason = "DATA_INSUFFICIENT"
        elif mutual == "1_有行动":
            # 行动事件: 需要前向收益 (修正2: 逐事件)
            if fill_status != "FILLABLE":
                decision_outcome = "INSUFFICIENT_DATA"
                label_status = "MISSING"
                label_reason = f"NO_FILL:{fill_status}"
            elif not h3_track or not h5_track:
                decision_outcome = "INSUFFICIENT_DATA"
                label_status = "MISSING"
                label_reason = "NO_RETURN_PATH_TRACK"
            elif not h3_complete and not h5_complete:
                decision_outcome = "INSUFFICIENT_DATA"
                label_status = "MISSING"
                label_reason = "H3_H5_BOTH_INCOMPLETE"
            elif not h5_complete:
                decision_outcome = "INSUFFICIENT_DATA"
                label_status = "PARTIAL"
                label_reason = "H5_TRACK_INCOMPLETE"
            elif not h3_complete:
                decision_outcome = "INSUFFICIENT_DATA"
                label_status = "PARTIAL"
                label_reason = "H3_TRACK_INCOMPLETE"
            elif h3_deferral_unchecked > 0 or h5_deferral_unchecked > 0:
                decision_outcome = "INSUFFICIENT_DATA"
                label_status = "PARTIAL"
                label_reason = "DEFERRAL_UNCHECKED"
            else:
                decision_outcome = "ACTED_LABEL_AVAILABLE"
                label_status = "LABEL_AVAILABLE"
                label_reason = "OK"
        else:
            decision_outcome = "INSUFFICIENT_DATA"
            label_status = "MISSING"
            label_reason = "UNKNOWN_MUTUAL_STATE"

        rec["label_status"] = label_status
        rec["label_reason"] = label_reason
        rec["decision_outcome"] = decision_outcome

        # ---- 网格异常状态 (修正5: 按日期版本, 按依赖链) ----
        rec["grid_anomaly_E"] = (e_day_cov == "GAP")
        rec["grid_anomaly_note"] = ("按日期版本判定 (241/242槽); "
                                    "依赖链缺槽判定留执行阶段 (修正5)")

        per_event.append(rec)

    # ================= 5. 聚合: segment × event_type × label_status =================
    agg = defaultdict(int)
    for r in per_event:
        key = (r["segment"], r["event_type"], r["label_status"])
        agg[key] += 1
    agg_out = {f"{seg}|{et}|{ls}": n for (seg, et, ls), n in sorted(agg.items())}

    # ================= 6. 人口对账链 (修正1: 精确对账, 不混批次) =================
    excl_layer = v3.get("instrument_exclusion_population_layer", {})
    n_excluded_etf_bse = excl_layer.get("excluded_n", 37)
    n_total_1050 = n_population + n_excluded_etf_bse  # 1,050

    # 互斥四类 (FROZEN)
    n_class1_action = sum(1 for r in per_event if r["mutual_status"] == "1_有行动")
    n_class2_watch = sum(1 for r in per_event if r["mutual_status"] == "2_WATCH但无行动")
    n_class3_no_watch = sum(1 for r in per_event if r["mutual_status"] == "3_未WATCH且无行动")
    n_class4_insufficient = sum(1 for r in per_event if r["mutual_status"] == "4_数据不足无法判断")

    # 行动子链
    n_action_events = sum(1 for r in per_event
                          if r["event_type"] in ("BASE_HOLD_S1", "BASE_BREAKOUT_S2", "DIP_REPAIR"))
    n_fillable = sum(1 for r in per_event if r.get("fill_status") == "FILLABLE")
    n_label_available = sum(1 for r in per_event if r.get("label_status") == "LABEL_AVAILABLE")

    # 触发联合 (修正1: 分别计数, 不混批次)
    trigger_union = {
        "S1": sum(1 for r in per_event if "S1" in r.get("triggers", [])),
        "S2": sum(1 for r in per_event if "S2" in r.get("triggers", [])),
        "DIP_REPAIR": sum(1 for r in per_event if "DIP_REPAIR" in r.get("triggers", [])),
        "any_trigger": sum(1 for r in per_event if r.get("trigger_status") != "NONE"),
        "double_trigger": sum(1 for r in per_event if r.get("double_trigger")),
    }

    # 251/247 批次交叉 (不混入, 仅旁注)
    n_in_251 = sum(1 for r in per_event if r.get("in_251"))
    n_in_247 = sum(1 for r in per_event if r.get("in_247"))

    # 质量隔离 (修正1: 单列)
    quality_isolation = {
        "E_day_COMPLETE": sum(1 for r in per_event if r.get("e_day_coverage") == "COMPLETE"),
        "E_day_GAP": sum(1 for r in per_event if r.get("e_day_coverage") == "GAP"),
        "E_day_NO_FILE": sum(1 for r in per_event if r.get("e_day_coverage") == "NO_FILE"),
        "E_day_anomaly": sum(1 for r in per_event if r.get("grid_anomaly_E")),
    }

    # ================= 7. 三种样本类型 + 有效日期数 =================
    sample_types = {}
    for et in ("DIP_REPAIR", "BASE_HOLD_S1", "BASE_BREAKOUT_S2", "WATCH_ONLY", "NO_ACTION"):
        subset = [r for r in per_event if r["event_type"] == et]
        sample_types[et] = {
            "count": len(subset),
            "label_available": sum(1 for r in subset if r.get("label_status") == "LABEL_AVAILABLE"),
            "unique_E_dates": len(set(r["E"] for r in subset)),
            "fillable": sum(1 for r in subset if r.get("fill_status") == "FILLABLE"),
        }

    # ================= 8. 数据缺口集 (行动事件 E+1..E+5 NO_FILE/GAP) =================
    data_gaps = []
    for r in per_event:
        if r["mutual_status"] != "1_有行动":
            continue
        for m in r.get("minute_path_missing", []):
            data_gaps.append({"code": r["code"], "slot": m["slot"],
                              "status": m["status"], "E": r["E"],
                              "reason": "MINUTE_GAP_OR_NO_FILE"})
    seen_gaps = set()
    data_gaps_unique = []
    for g in data_gaps:
        k = (g["code"], g["E"], g["slot"])
        if k not in seen_gaps:
            seen_gaps.add(k)
            data_gaps_unique.append(g)
    n_gap_codes = len(set(g["code"] for g in data_gaps_unique))
    n_gap_slots = len(data_gaps_unique)

    # ================= 9. 基线定义 (修正3) =================
    baseline_definitions = {
        "strategy_baseline_A_all_buy": {
            "desc": "对每个 S1/S2/DIP_REPAIR 信号都执行买入, 同 v0.1.1 执行口径",
            "entry_decision": "always act on trigger",
            "exit": "H3/H5 + 硬止损 per v0.1.1 §5",
            "T_plus_1": True,
            "caliber": "v0.1.1 (same as model arm)",
        },
        "strategy_baseline_B_all_reject": {
            "desc": "从不执行, 全程持有现金",
            "entry_decision": "never act",
            "exit": "N/A (cash)",
            "T_plus_1": "N/A",
            "caliber": "v0.1.1 (same as model arm)",
        },
        "simple_rule_baseline_fixed_time": {
            "desc": "E 日 09:31 买入, 同 T+1 退出口径 (H3/H5+硬止损)",
            "entry_decision": "buy at 09:31 on E day",
            "exit": "H3/H5 + 硬止损 per v0.1.1 §5",
            "T_plus_1": True,
            "caliber": "v0.1.1 (same as model arm)",
        },
        "note": ("两臂 (模型与基线) 使用同一 v0.1.1 执行口径, 仅改变入场决策; "
                 "不设'E日买E日卖' (违反 T+1); 不设常数概率模型"),
    }

    # ================= 10. WATCH 三态 (修正4) =================
    watch_states = {
        "NOT_ACTED": sum(1 for r in per_event if r.get("decision_outcome") == "NOT_ACTED"),
        "ACTED_LABEL_AVAILABLE": sum(1 for r in per_event
                                     if r.get("decision_outcome") == "ACTED_LABEL_AVAILABLE"),
        "INSUFFICIENT_DATA": sum(1 for r in per_event
                                 if r.get("decision_outcome") == "INSUFFICIENT_DATA"),
        "note": ("NOT_PROFITABLE 需执行后才能判定 (本轮不执行); "
                 "不按日终是否触发回填特征 (修正4); "
                 "WATCH (277) 与 未WATCH (245) 均归 NOT_ACTED"),
    }

    # ================= 11. DIP_REPAIR 定义状态 (修正6) =================
    dip_repair_def_summary = {
        "NOT_TRACKED_IN_RULE_A": sum(1 for r in per_event
                                     if r.get("dip_repair_def_status") == "NOT_TRACKED_IN_RULE_A"),
        "NA_NON_ACTION": sum(1 for r in per_event
                              if r.get("dip_repair_def_status") == "NA_NON_ACTION"),
        "NA_DATA_INSUFFICIENT": sum(1 for r in per_event
                                    if r.get("dip_repair_def_status") == "NA_DATA_INSUFFICIENT"),
        "S1_to_BASE_HOLD_mapping": {
            "PENDING_VERIFICATION": sum(1 for r in per_event
                                        if r.get("event_type") == "BASE_HOLD_S1"
                                        and not r.get("event_type_mapping_verified")),
            "VERIFIED": sum(1 for r in per_event
                            if r.get("event_type") == "BASE_HOLD_S1"
                            and r.get("event_type_mapping_verified")),
        },
        "note": ("ruleA 冻结件仅 S1/S2/双触发S1+S2, 无 DIP_REPAIR; "
                 "DIP_REPAIR 定义 (detect_signal_v02) 在本批不适用 (修正6); "
                 "S1→BASE_HOLD 映射需显式核实 (PENDING_VERIFICATION)"),
    }

    # ================= 12. 缺槽规则状态 (修正5) =================
    grid_rule_status = {
        "date_version_rule": ("241 槽 (<20260624) / 242 槽 (>=20260624); "
                              "13:00 槽为源格式差异, 非行情缺失"),
        "dependency_chain_rule": ("缺槽在闭环交易依赖前 → 不可评估; "
                                  "在依赖链闭环后 → 不误罚; "
                                  "逐事件依赖链判定留执行阶段"),
        "anomaly_days_not_relaxed": "127 异常日不重新讨论放松 (修正5)",
        "E_day_COMPLETE": sum(1 for r in per_event if r.get("e_day_coverage") == "COMPLETE"),
        "E_day_GAP": sum(1 for r in per_event if r.get("e_day_coverage") == "GAP"),
        "E_day_NO_FILE": sum(1 for r in per_event if r.get("e_day_coverage") == "NO_FILE"),
    }

    # ================= 13. 报告 =================
    report = {
        "version": "r1_coverage_inventory_v2_from_1013_population",
        "date": "2026-10-08",
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "contract_ref": "dual_path_execution_contract_v0_1_1_final.md",
        "population_source": "gcp_247_full_population_inventory_v3.json (FROZEN 1,013)",
        "network_requests": 0,
        "training_performed": False,
        "threshold_changes": 0,
        "corrections_incorporated": 6,
        "frozen_input_hashes": input_hashes,
        "contract_data_cutoff": DATA_CUTOFF_CONTRACT,
        "population_reconciliation": {
            "total_1050": n_total_1050,
            "excluded_ETF_BSE": n_excluded_etf_bse,
            "study_population_1013": n_population,
            "class1_action": n_class1_action,
            "class2_watch_no_action": n_class2_watch,
            "class3_no_watch_no_action": n_class3_no_watch,
            "class4_data_insufficient": n_class4_insufficient,
            "action_events": n_action_events,
            "fillable": n_fillable,
            "label_available": n_label_available,
            "chain": (f"{n_total_1050} → {n_population} (−{n_excluded_etf_bse} ETF/BSE 人口层) → "
                      f"{n_class1_action} action → {n_fillable} fillable → "
                      f"{n_label_available} label_available"),
            "mutex_chain": (f"1,013 = {n_class1_action}(1_有行动) + "
                            f"{n_class2_watch}(2_WATCH但无行动) + "
                            f"{n_class3_no_watch}(3_未WATCH且无行动) + "
                            f"{n_class4_insufficient}(4_数据不足无法判断)"),
        },
        "batch_251_247_separate": {
            "note": "251/247 是不同批次, 不混入 1,013 研究人口",
            "events_251_cache": n_251_batch,
            "excluded_BSE_in_251": n_bse_251,
            "excluded_ETF_in_251": n_etf_251,
            "in_scope_247": n_247_batch,
            "v3_records_in_251_cross": n_in_251,
            "v3_records_in_247_cross": n_in_247,
            "explanation": ("251 事件缓存 → 排除 BSE/ETF → 247 在范围, 是另一批次; "
                            "1,013 研究人口来自 1,050 箱体候选人口层排除 37 ETF/BSE; "
                            "两者不重叠, v3 records 中 in_251/in_247 标志仅作交叉参考"),
        },
        "trigger_union": trigger_union,
        "quality_isolation": quality_isolation,
        "aggregate_segment_x_event_x_label": agg_out,
        "sample_types": sample_types,
        "watch_states": watch_states,
        "dip_repair_def_summary": dip_repair_def_summary,
        "grid_rule_status": grid_rule_status,
        "baseline_definitions": baseline_definitions,
        "data_gaps": {
            "n_unique_gaps": len(data_gaps_unique),
            "n_gap_codes": n_gap_codes,
            "n_gap_slots": n_gap_slots,
            "gaps": data_gaps_unique[:200],
        },
        "per_event": per_event,
        "elapsed_s": round(time.time() - t0, 1),
    }

    REPORT_JSON.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    # ================= stdout 摘要 =================
    print("\n" + "=" * 70)
    print("[人口对账链] (修正1: 精确对账, 不混批次)")
    print(f"  {n_total_1050} → {n_population} (−{n_excluded_etf_bse} ETF/BSE 人口层) → "
          f"{n_class1_action} action → {n_fillable} fillable → "
          f"{n_label_available} label_available")
    print(f"  互斥四类: 行动={n_class1_action} / WATCH无行动={n_class2_watch} / "
          f"未WATCH无行动={n_class3_no_watch} / 数据不足={n_class4_insufficient}")

    print("\n[251/247 批次旁注 (不混入)]")
    print(f"  {n_251_batch} → 排除 BSE={n_bse_251} + ETF={n_etf_251} → "
          f"{n_247_batch} 在范围 (不同批次)")
    print(f"  v3 records 中 in_251={n_in_251}, in_247={n_in_247} (交叉参考)")

    print("\n[触发联合] (分别计数)")
    for k, v in trigger_union.items():
        print(f"  {k}: {v}")

    print("\n[质量隔离]")
    for k, v in quality_isolation.items():
        print(f"  {k}: {v}")

    print("\n[三种样本类型 + 有效日期数]")
    for et, d in sample_types.items():
        print(f"  {et}: count={d['count']}, fillable={d['fillable']}, "
              f"label_available={d['label_available']}, unique_E_dates={d['unique_E_dates']}")

    print("\n[聚合 segment × event_type × label_status]")
    for k, n in sorted(agg_out.items()):
        print(f"  {k}: {n}")

    print(f"\n[数据缺口] {len(data_gaps_unique)} 个唯一缺口 (行动事件 E+1..E+5), "
          f"{n_gap_codes} 股")

    print("\n[WATCH 三态] (修正4)")
    for k, v in watch_states.items():
        if isinstance(v, int):
            print(f"  {k}: {v}")

    print("\n[DIP_REPAIR 定义状态] (修正6)")
    for k, v in dip_repair_def_summary.items():
        if isinstance(v, dict):
            print(f"  {k}:")
            for k2, v2 in v.items():
                print(f"    {k2}: {v2}")
        elif isinstance(v, int):
            print(f"  {k}: {v}")

    print("\n[缺槽规则] (修正5)")
    for k, v in grid_rule_status.items():
        if isinstance(v, int):
            print(f"  {k}: {v}")
        else:
            print(f"  {k}: {v}")

    print("\n[基线定义] (修正3)")
    for k, v in baseline_definitions.items():
        if isinstance(v, dict):
            print(f"  {k}: {v['desc']}")

    print(f"\n耗时 {report['elapsed_s']}s; 报告: {REPORT_JSON.name}")


if __name__ == "__main__":
    main()
