"""
R1 DIP_REPAIR 缓存扫描 + S1->BASE_HOLD 映射核验 + 65 对缺口清单冻结
================================================================
零网络请求, 只读现有缓存, 零拟合, 零采集.

输入:
- gcp_247_full_population_inventory_v3.json (1,013 人口, FROZEN)
- gcp_247_ruleA_trigger_records_v1.json (S1/S2/双触发冻结件)
- r1_coverage_inventory.json (R1 盘点, 含 65 对缺口原始清单)
- 现有 minute/minute_ext/minute_car 缓存 (只读)
- D:/a股数据/data_daily_parts/ 日线分片 (只读)

输出: gcp_round_b_small_sample_v1/r1_dip_scan_and_gap_freeze.json

裁决依据:
- 合同 §2.4 优先级: BASE_HOLD(S1) > BASE_BREAKOUT(S2) > DIP_REPAIR(S0)
- detect_signal_v02 与 §2.1 DIP_REPAIR 定义一致 (BREAK/ALREADY_BELOW/RECLAIMED_AT_S/APPROACH)
- v3 冻结件 reason 字段: "S1"/"S2"/"双触发S1+S2"/null = action_kind 等价物
  注: v3 第 212-215 行 first_action=min(s1,s2) 按时间首触发, 与 §2.4 优先级裁决
  不完全等价; 双触发 2 条按 §2.4 应取 S1 (即使 s2 早于 s1), 本脚本独立按 §2.4 裁决.

DIP_REPAIR=0 的含义:
- "未追踪" 不是 "没有这种机会"; 缓存缺失记 UNCLASSIFIABLE, 不据此外推
- 与 S1/S2 重叠按 §2.4 归 S1/S2, 不重复增加独立样本
"""
from __future__ import annotations

import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "step9_preflight"))
sys.path.insert(0, str(HERE))

import gcp_dual_path_coverage_probe as PROBE  # noqa: E402
import gcp_round_b_engine as E  # noqa: E402
from gcp_car_v0_2_extractor import detect_signal_v02  # noqa: E402

RUN = HERE / "gcp_round_b_small_sample_v1"
CONTRACTS = HERE / "contracts"
DATA_CUTOFF = 20260928


def sha256_file(p: Path) -> str:
    import hashlib
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    t0 = time.time()
    print("=" * 70)
    print("R1 DIP_REPAIR 扫描 + S1->BASE_HOLD 映射核验 + 65 对缺口冻结")
    print("零网络请求, 只读现有缓存")
    print("=" * 70)

    # ---- 加载冻结输入 ----
    v3_path = RUN / "gcp_247_full_population_inventory_v3.json"
    rulea_path = RUN / "gcp_247_ruleA_trigger_records_v1.json"
    r1_path = RUN / "r1_coverage_inventory.json"

    v3 = json.loads(v3_path.read_text(encoding="utf-8"))
    rulea = json.loads(rulea_path.read_text(encoding="utf-8"))
    r1 = json.loads(r1_path.read_text(encoding="utf-8"))

    v3_recs = v3["records"]
    rulea_recs = rulea["records"]
    assert len(v3_recs) == 1013, f"v3 人口 != 1013: {len(v3_recs)}"
    assert len(rulea_recs) == 1013, f"ruleA 人口 != 1013: {len(rulea_recs)}"

    # ruleA 与 v3 按 (code, E) 对账
    rulea_map = {(r["code"], int(r["E"])): r for r in rulea_recs}
    v3_map = {(r["code"], int(r["E"])): r for r in v3_recs}
    for k in v3_map:
        assert k in rulea_map, f"v3 有而 ruleA 无: {k}"

    # ---- 日线 panel + 日历 ----
    panel = PROBE.get_daily_parquet()
    panel["code"] = panel["code"].astype(str).str.zfill(6)
    panel["date"] = panel["date"].astype(int)
    cal = sorted(int(d) for d in panel["date"].unique())
    cal_idx = {d: i for i, d in enumerate(cal)}
    daily_by_code = {c: g.sort_values("date").reset_index(drop=True)
                     for c, g in panel.groupby("code")}
    print(f"日线: {len(cal)} 交易日, {len(daily_by_code)} 股")
    print(f"日线终点 (实际快照): {cal[-1]} (合同 DATA_CUTOFF={DATA_CUTOFF})")

    # ---- 分钟缓存索引 (复用 PROBE.load_minute_index 逻辑, 独立实现以可审计) ----
    MIN_DIRS = [RUN / "minute", RUN / "minute_ext", RUN / "minute_car"]
    FALLBACK_DIR = ROOT / "pullback_train" / "minute_cache_clean"
    minute_idx: dict = {}
    file_sizes = []
    for directory in MIN_DIRS + ([FALLBACK_DIR] if FALLBACK_DIR.exists() else []):
        if not directory.exists():
            continue
        for pf in directory.glob("*.parquet"):
            try:
                sz = pf.stat().st_size
                file_sizes.append(sz)
                m = pd.read_parquet(pf, columns=["date"])
                if "date" in m and not m.empty:
                    code = pf.stem
                    for d in pd.unique(m["date"].astype(str).str[:8].astype(int)):
                        minute_idx.setdefault((code, int(d)), []).append((pf, directory.name))
            except Exception:
                pass

    print(f"分钟缓存索引: {len(minute_idx)} (code,date); 文件数: {len(file_sizes)}")
    if file_sizes:
        avg_file_bytes = int(np.mean(file_sizes))
        max_file_bytes = int(np.max(file_sizes))
        min_file_bytes = int(np.min(file_sizes))
        p50 = int(np.percentile(file_sizes, 50))
        p95 = int(np.percentile(file_sizes, 95))
    else:
        avg_file_bytes = max_file_bytes = min_file_bytes = p50 = p95 = 0
    print(f"文件字节: avg={avg_file_bytes}, min={min_file_bytes}, p50={p50}, p95={p95}, max={max_file_bytes}")

    def load_day(code, day):
        """同 v3 load_day: 返回 (df, source_path) 或 (None, path/None)."""
        cands = minute_idx.get((code, day))
        if not cands:
            return None, None
        # 取首个候选 (多份时不挑价格更有利的)
        pf, src = cands[0]
        try:
            df = pd.read_parquet(pf)
            df = df[df["date"].astype(int) // 1000000 == day].sort_values("date").reset_index(drop=True)
            return df, pf
        except Exception:
            return None, pf

    # ---- 逐事件扫描 DIP_REPAIR + §2.4 优先级裁决 ----
    per_event = []
    dip_view_counter = Counter()
    resolved_counter = Counter()
    overlap_counter = Counter()
    unclassifiable_counter = Counter()
    rulea_action_kind_counter = Counter()

    # v3 reason -> action_kind 标准化 (与 ruleA action_kind 对账)
    def reason_to_action_kind(reason):
        if reason == "S1":
            return "S1"
        if reason == "S2":
            return "S2"
        if reason and "双触发" in reason:
            return "DOUBLE"
        if reason is None:
            return None
        # 类别 4 的各种原因 (INSUFFICIENT_HISTORY / NO_FILE / GAP / BEYOND_CUTOFF / main口径箱体无效)
        return None

    for r in v3_recs:
        code = r["code"]
        E_date = int(r["E"])
        P_date = int(r["P"])
        setup = r["setup"]
        seg = r["segment"]
        s1_ts = r.get("s1_ts")
        s2_ts = r.get("s2_ts")
        reason_v3 = r.get("reason")
        mutual = r.get("mutual")
        e_day_cov = r.get("e_day_coverage")

        action_kind_v3 = reason_to_action_kind(reason_v3)
        rulea_action_kind_counter[action_kind_v3] += 1

        # 与 ruleA 冻结件对账
        rk = rulea_map.get((code, E_date), {})
        rulea_action_kind = rk.get("action_kind")
        rulea_first_action_ts = rk.get("first_action_ts")
        rulea_double = rk.get("double_trigger")
        # 核对 v3 reason 与 ruleA action_kind 一致性
        ak_match = (action_kind_v3 == rulea_action_kind) or (
            action_kind_v3 == "DOUBLE" and rulea_action_kind == "DOUBLE")

        # ---- 构建 zones / atr_ref / ref_price ----
        dc = daily_by_code.get(code)
        i_e = cal_idx.get(E_date, -1)

        dip_view = "NOT_COMPUTED"
        dip_signal_ts = None
        dip_zone_info = None

        if i_e < 0 or dc is None:
            dip_view = "E_NOT_IN_CAL"
        else:
            lookback = cal[max(0, i_e - 250):i_e]
            d_sub = dc[dc["date"].isin(lookback)].sort_values("date").reset_index(drop=True)
            if len(d_sub) < 25:
                dip_view = "INSUFFICIENT_HISTORY"
            else:
                try:
                    atr_ref = float(E.build_atr_ref(d_sub))
                    zones = E.build_zones(d_sub, atr_ref)
                    ref_price = float(d_sub["close"].iloc[-1])
                    if not np.isfinite(atr_ref) or atr_ref <= 0 or not zones:
                        dip_view = "ZONE_BUILD_FAIL"
                    else:
                        # 加载 E 日分钟
                        e_df, _ = load_day(code, E_date)
                        if e_df is None or len(e_df) == 0:
                            dip_view = "NO_CACHE"
                        else:
                            sig = detect_signal_v02(e_df, zones, atr_ref, ref_price)
                            if sig is None:
                                dip_view = "NO_SIGNAL"
                            else:
                                dip_view = sig["signal_type"]
                                dip_signal_ts = int(sig["signal_ts"])
                                dip_zone_info = {
                                    "zone_id": sig.get("zone_id"),
                                    "zone_upper": round(float(sig["zone_upper"]), 4),
                                    "zone_lower": round(float(sig["zone_lower"]), 4),
                                    "d_atr": round(float(sig["d_atr"]), 4),
                                    "pen_atr_bar": round(float(sig["pen_atr_bar"]), 4),
                                }
                except Exception as ex:
                    dip_view = "EXC"
                    dip_signal_ts = None

        # ---- §2.4 优先级裁决 (独立于 v3 时间首触发) ----
        # 1. S1 候选存在 -> BASE_HOLD
        # 2. S2 候选存在 (无 S1) -> BASE_BREAKOUT
        # 3. 无 S1/S2, detect_signal_v02 触发 BREAK/ALREADY_BELOW -> DIP_REPAIR
        # 其余: NO_ACTION / UNCLASSIFIABLE
        s1_present = s1_ts is not None
        s2_present = s2_ts is not None
        dip_trigger = dip_view in ("BREAK", "ALREADY_BELOW")

        if s1_present:
            resolved = "BASE_HOLD"
            basis = "§2.4(1) S1 候选存在 → BASE_HOLD 优先 (即使 S2 或 DIP 更早触发)"
        elif s2_present:
            resolved = "BASE_BREAKOUT"
            basis = "§2.4(2) S2 候选存在且无 S1 → BASE_BREAKOUT"
        elif dip_trigger:
            resolved = "DIP_REPAIR"
            basis = f"§2.4(3) 无 S1/S2, detect_signal_v02={dip_view} → DIP_REPAIR"
        elif dip_view == "RECLAIMED_AT_S":
            resolved = "NO_ACTION"
            basis = "DIP_REPAIR 视图 RECLAIMED_AT_S 不触发买入候选 (§2.1)"
        elif dip_view == "NO_SIGNAL":
            resolved = "NO_ACTION"
            basis = "无 S1/S2/DIP_REPAIR 触发"
        elif dip_view in ("NO_CACHE", "E_NOT_IN_CAL", "INSUFFICIENT_HISTORY",
                          "ZONE_BUILD_FAIL", "EXC", "NOT_COMPUTED"):
            resolved = "UNCLASSIFIABLE"
            basis = f"无法判定 DIP_REPAIR 视图: {dip_view}"
        else:
            resolved = "UNCLASSIFIABLE"
            basis = f"未知 dip_view: {dip_view}"

        overlap = dip_trigger and (s1_present or s2_present)
        # 重叠细分
        if overlap:
            if s1_present and s2_present:
                overlap_kind = "DIP_OVERLAP_S1_AND_S2"
            elif s1_present:
                overlap_kind = "DIP_OVERLAP_S1"
            else:
                overlap_kind = "DIP_OVERLAP_S2"
        else:
            overlap_kind = "NO_OVERLAP"

        # §2.4 裁决 vs v3 时间首触发的差异 (仅双触发且 s2<s1 时不同)
        if s1_present and s2_present:
            # 双触发: §2.4 取 S1; v3 取 min(s1, s2)
            v3_first = "S2" if (s2_ts < s1_ts) else "S1"
            sec24_diff = (v3_first != "S1")
            sec24_note = (f"双触发 §2.4 取 S1; v3 first_action={v3_first}; "
                          f"{'一致' if not sec24_diff else '不一致 (v3 时间首触发, §2.4 优先级裁决)'}")
        else:
            sec24_diff = False
            sec24_note = "无歧义"

        unclass_reason = None if resolved != "UNCLASSIFIABLE" else dip_view

        per_event.append({
            "code": code, "E": E_date, "P": P_date, "setup": setup, "segment": seg,
            "mutual": mutual,
            "e_day_coverage": e_day_cov,
            "s1_ts": s1_ts, "s2_ts": s2_ts,
            "action_kind_v3": action_kind_v3,
            "ruleA_action_kind": rulea_action_kind,
            "v3_ruleA_action_kind_match": ak_match,
            "ruleA_first_action_ts": rulea_first_action_ts,
            "ruleA_double_trigger": rulea_double,
            "dip_view": dip_view,
            "dip_signal_ts": dip_signal_ts,
            "dip_zone_info": dip_zone_info,
            "resolved_branch": resolved,
            "resolution_basis": basis,
            "overlap_with_s1s2": overlap,
            "overlap_kind": overlap_kind,
            "sec24_vs_v3_diff": sec24_diff,
            "sec24_note": sec24_note,
            "unclassifiable_reason": unclass_reason,
        })

        dip_view_counter[dip_view] += 1
        resolved_counter[resolved] += 1
        if overlap:
            overlap_counter[overlap_kind] += 1
        if resolved == "UNCLASSIFIABLE":
            unclassifiable_counter[unclass_reason] += 1

    # ---- 65 对缺口清单冻结 ----
    r1_gaps = r1["data_gaps"]["gaps"]
    assert len(r1_gaps) == 65, f"r1 gaps != 65: {len(r1_gaps)}"

    # 已知确定性源洞 (从 project_memory 与历史轮次整理; 不默认重抓能修复)
    # 20250508 (605398, 603887) = canonical 源缺口
    CANONICAL_SOURCE_HOLES = {
        ("605398", 20250508): "canonical 源缺口 (daily 有成交, HTTP 分钟空, 重试无效)",
        ("603887", 20250508): "canonical 源缺口 (daily 有成交, HTTP 分钟空, 重试无效)",
    }

    gaps_frozen = []
    accumulated_bytes = 0
    for g in r1_gaps:
        code = g["code"]
        E_date = int(g["E"])
        slot = g["slot"]  # E+1/E+2/E+3/E+5
        status = g["status"]  # GAP / NO_FILE
        offset = int(slot.replace("E+", ""))

        # slot 对应实际日期
        i_e = cal_idx.get(E_date, -1)
        if i_e >= 0 and i_e + offset < len(cal):
            actual_date = cal[i_e + offset]
        else:
            actual_date = None

        # 用途: 哪个轨道依赖
        # H3 依赖 E+1,E+2,E+3; H5 依赖 E+1,E+2,E+3,E+5
        if offset in (1, 2, 3):
            use = "H3+H5"
        elif offset == 5:
            use = "H5_only"
        else:
            use = "UNKNOWN"

        # 已知源洞核对
        known_hole_key = (code, actual_date) if actual_date else None
        source_note = CANONICAL_SOURCE_HOLES.get(known_hole_key)
        known_source_hole = source_note is not None

        # 预计新增字节: 优先从该 code 现有文件实际大小估算, 否则用全局平均
        code_files = [pf for (c, d), lst in minute_idx.items() if c == code for pf, _ in lst]
        if code_files:
            try:
                code_avg = int(np.mean([pf.stat().st_size for pf in code_files]))
            except Exception:
                code_avg = avg_file_bytes
        else:
            code_avg = avg_file_bytes

        accumulated_bytes += code_avg

        gaps_frozen.append({
            "code": code,
            "E": E_date,
            "slot": slot,
            "actual_date": actual_date,
            "offset_days": offset,
            "use": use,
            "cache_status": status,
            "known_source_hole": known_source_hole,
            "source_note": source_note,
            "estimated_bytes_per_file": code_avg,
            "accumulated_bytes": accumulated_bytes,
            "existing_cache_files_for_code": len(code_files),
        })

    # ---- 汇总 ----
    seg_setup_resolved = defaultdict(lambda: defaultdict(int))
    for ev in per_event:
        seg_setup_resolved[f"{ev['segment']}/{ev['setup']}"][ev["resolved_branch"]] += 1

    seg_resolved = defaultdict(lambda: defaultdict(int))
    for ev in per_event:
        seg_resolved[ev["segment"]][ev["resolved_branch"]] += 1

    # ---- 待执行验证项标注 (152 可用 = 执行标签成立? 本轮未跑执行器) ----
    pending_exec_verify = {
        "note": ("R1 报告 'label_available=152' 是文件层判定 (有缓存即视为可用); "
                 "用户要求收紧为执行标签成立 (完全退出/残仓为零/依赖有效/label_end_ts 合法/跨段隔离). "
                 "本轮未跑 v0.1.1 执行器, 逐轨道执行标签成立状态 PENDING."),
        "exec_label_check_pending": True,
        "r1_file_label_available": r1["population_reconciliation"]["label_available"],
        "track_h3_main_target": True,
        "track_h5_auxiliary": True,
        "rule": "H5 缺失不连带删除有效 H3; 522 未行动不是亏损负样本 (无反事实标签)",
    }

    out = {
        "phase": "R1_DIP_SCAN_AND_GAP_FREEZE",
        "version": "r1_dip_scan_and_gap_freeze_v1",
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
        "contract_refs": {
            "dual_path_event_execution_contract_v0_1": "contracts/dual_path_event_execution_contract.md (§2.1 DIP_REPAIR, §2.4 优先级)",
            "dual_path_execution_contract_v0_1_1": "contracts/dual_path_execution_contract_v0_1_1_final.md",
            "full_population_rule_compare_contract_v1": "contracts/full_population_rule_compare_contract_v1.md",
        },
        "network_requests": 0,
        "training_performed": False,
        "data_collection_performed": False,
        "frozen_input_hashes": {
            "gcp_247_full_population_inventory_v3.json": "a2c53545efc331e64438a590d155cae25d79048c79e2d8c4ec4166e9e018d0f8",
            "gcp_247_ruleA_trigger_records_v1.json": "f6c4c997003fe3b5191666a1a120feffdba7f9fcb0289ba0a9d339c89b98dc6e",
            "r1_coverage_inventory.json": "<sha256_at_runtime>",
            "gcp_car_v0_2_extractor.py": "<detect_signal_v02 冻结函数>",
        },
        "population_n": 1013,
        "dip_repair_definition_audit": {
            "detect_signal_v02_vs_contract_2_1": "CONSISTENT",
            "audit_details": [
                "BREAK: low<=zone_lower 且 close<zone_upper ✓ (实现 L62-73)",
                "ALREADY_BELOW: 首监控 bar open<=zone_lower 且 close<zone_upper ✓ (实现 L50-60)",
                "RECLAIMED_AT_S: low<=zone_lower 且 close>=zone_upper (同 bar/跳空收回, 不触发买入候选) ✓ (实现 L52-60, L64-72)",
                "APPROACH 仅观察不触发 ✓ (实现 L74-77)",
                "zone 使用 build_zones (0.35·ATR 链式合并) + target_zone 冻结区域 ✓",
                "S=触发 bar 结束; S 时可见字段=v0.2 已冻结字段 ✓",
            ],
            "note": ("DIP_REPAIR=0 在 R1 报告中是 '未追踪' 而非 '没有机会'; "
                     "本扫描用 detect_signal_v02 在 1,013 人口缓存内重新追踪, "
                     "与 S1/S2 重叠按 §2.4 归 S1/S2, 不重复增加独立样本."),
        },
        "sec24_priority_vs_v3_timefirst": {
            "contract_rule": "§2.4: BASE_HOLD(S1) > BASE_BREAKOUT(S2) > DIP_REPAIR(S0); 同 (code,E) 同 bar 多满足时按优先级取一条",
            "v3_implementation": "v3 第 212-215 行: first_action=min(s1,s2) 按时间首触发, reason='双触发S1+S2' for 双触发",
            "difference": "双触发 2 条若 s2<s1, v3 first_action=S2, 但 §2.4 应取 S1; 本扫描按 §2.4 独立裁决",
            "double_trigger_count": sum(1 for ev in per_event if ev["ruleA_double_trigger"]),
            "sec24_diff_count": sum(1 for ev in per_event if ev["sec24_vs_v3_diff"]),
        },
        "dip_view_distribution": dict(dip_view_counter),
        "resolved_branch_distribution": dict(resolved_counter),
        "overlap_distribution": dict(overlap_counter),
        "unclassifiable_reasons": dict(unclassifiable_counter),
        "ruleA_action_kind_distribution": dict(rulea_action_kind_counter),
        "segment_x_resolved": {seg: dict(c) for seg, c in seg_resolved.items()},
        "segment_setup_x_resolved": {k: dict(v) for k, v in seg_setup_resolved.items()},
        "gap_freeze": {
            "n_pairs": len(gaps_frozen),
            "n_unique_codes": len(set(g["code"] for g in gaps_frozen)),
            "total_estimated_bytes": accumulated_bytes,
            "total_estimated_kib": round(accumulated_bytes / 1024, 2),
            "total_estimated_mib": round(accumulated_bytes / (1024 * 1024), 4),
            "avg_file_bytes_used": avg_file_bytes,
            "file_size_stats": {
                "min": min_file_bytes, "p50": p50, "p95": p95, "max": max_file_bytes,
                "avg": avg_file_bytes,
            },
            "use_distribution": dict(Counter(g["use"] for g in gaps_frozen)),
            "cache_status_distribution": dict(Counter(g["cache_status"] for g in gaps_frozen)),
            "known_source_holes": sum(1 for g in gaps_frozen if g["known_source_hole"]),
            "canonical_source_holes_listed": [g for g in gaps_frozen if g["known_source_hole"]],
            "note": ("约 11.4KB 是每文件估算; 65 对约 0.72MiB 是按每文件独立计算的简单上限; "
                     "实际可能因同 code 多日合并到同一 parquet 而略低; "
                     "已有确定性源洞 (canonical 20250508) 不默认重抓能修复; "
                     "新目录不重置累计预算."),
            "gaps": gaps_frozen,
        },
        "pending_exec_label_verification": pending_exec_verify,
        "per_event": per_event,
        "elapsed_s": round(time.time() - t0, 1),
    }

    # r1_coverage_inventory.json 哈希填实
    try:
        out["frozen_input_hashes"]["r1_coverage_inventory.json"] = sha256_file(r1_path)
    except Exception:
        pass

    out_path = RUN / "r1_dip_scan_and_gap_freeze.json"
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n输出: {out_path}")
    print(f"耗时: {out['elapsed_s']}s")
    print(f"\n=== 汇总 ===")
    print(f"DIP_REPAIR view 分布: {dict(dip_view_counter)}")
    print(f"resolved_branch 分布: {dict(resolved_counter)}")
    print(f"overlap 分布: {dict(overlap_counter)}")
    print(f"无法归类原因: {dict(unclassifiable_counter)}")
    print(f"缺口冻结: {len(gaps_frozen)} 对, "
          f"预计 {accumulated_bytes} 字节 ({accumulated_bytes/1024:.1f} KiB)")
    print(f"已知源洞: {sum(1 for g in gaps_frozen if g['known_source_hole'])} 个")
    print(f"待执行验证: {pending_exec_verify['exec_label_check_pending']}")


if __name__ == "__main__":
    main()
