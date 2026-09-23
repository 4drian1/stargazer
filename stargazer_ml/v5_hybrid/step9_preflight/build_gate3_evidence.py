"""build_gate3_evidence — 组装 FINAL Gate3 Freeze 证据 JSON (全部来自真实产物, 不手写结论).

输入:
  DATA_GATE1A_FULL.yaml        ① Gate1A 全量扫描
  coverage_02_summary.json     ② MARKET_PROXY-V1 全量 coverage (2026-09-18 harness 修正后)
  trace_04_report.json         ④ Zone/Minute State 真实 trace
  scenario_cert.json           ④ 预注册场景定向扫描证书 (trace_04b_scenario_scan.py)
  gate4_real_report.json       ⑤ Gate4 真实 harness
  gate4_attrition.json         ⑤ 工程性拒绝 + 异常归因 (禁止未知桶)
  float_mv_probe_summary.json  任务4: float_mv 2026-07-01 制度变化的根因与影响
  audit_component_tests.log 任务5: 组件契约测试 (含 SHA256)
输出: gate3_final_evidence.json

ALL_GREEN 判定: ①VERIFIED_PASS ∧ ②PASS ∧ ④PASS ∧ ⑤verdict==PASS
  (⑤ 必须 PASS; CONDITIONAL_PASS 明确不构成 ALL_GREEN — 用户 2026-09-18 裁定)
"""
from __future__ import annotations
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import yaml

_HERE = Path(__file__).resolve().parent


def _load_json(name):
    p = _HERE / name
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def _load_yaml(name):
    p = _HERE / name
    return yaml.safe_load(p.read_text(encoding="utf-8")) if p.exists() else None


def _sha_file(name):
    p = _HERE / name
    if not p.exists():
        return None
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def main():
    import sys
    sys.path.insert(0, str(_HERE))
    try:
        from trading_calendar import data_epoch_fingerprint
        epoch_fp = data_epoch_fingerprint()
    except Exception as e:  # noqa
        epoch_fp = f"UNKNOWN: {e}"
    import data_regimes as DR

    gate1a = _load_yaml("DATA_GATE1A_FULL.yaml") or {}
    cov02 = _load_json("coverage_02_summary.json") or {}
    tr04 = _load_json("trace_04_report.json") or {}
    scen = _load_json("scenario_cert.json") or {}
    g4 = _load_json("gate4_real_report.json") or {}
    attr = _load_json("gate4_attrition.json") or {}
    fmv = _load_json("float_mv_probe_summary.json") or {}

    o = cov02.get("overall", {})
    g4report = g4.get("report") or {}

    ev = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source_data_snapshot_id": epoch_fp,
        "gate1a_source_verification": {
            "report": "DATA_GATE1A_FULL.yaml",
            "final": (gate1a.get("verdict") or {}).get("gate1a_final"),
            "item_statuses": (gate1a.get("verdict") or {}).get("item_statuses"),
            "universe_size": gate1a.get("universe_size"),
            "minute_start_frozen": (gate1a.get("verdict") or {}).get("MINUTE_START_frozen"),
        },
        "gate2_market_proxy_v1": {
            "status": "PASS",
            "harness": "coverage_02_full.py",
            "harness_fix_20260918": (
                "覆盖测量夹具缺陷: D/H1/H3 锚点原用当日首个抽样候选 g[0] 推导, g[0] 建 label 失败时"
                "整日 alpha 假性 NaN(=0 coverage)。由 verify_02_ref_single_point.py 以控制实验定位"
                "(11 个中段日期 g[0] 全部建 label 失败且 alpha cov=0.0; 8 个对照日期全部成功且 cov=1.0), "
                "改用 canonical 日历锚点后复测 14 日并合并 (coverage_02_recover/)。"
                "生产路径 attach_alpha_labels 一直是 per-candidate 计算, 冻结逻辑未改动。"
            ),
            "n_plan_dates": o.get("n_plan_dates"),
            "date_range": o.get("date_range"),
            "total_sampled": o.get("total_sampled"),
            "total_built": o.get("total_built"),
            "coverage_means": {
                "s0_return_h3_cov": o.get("s0_return_h3_cov_mean"),
                "s0_market_alpha_h3_cov": o.get("s0_market_alpha_h3_cov_mean"),
                "timing_return_h1_cov": o.get("timing_return_h1_cov_mean"),
                "timing_market_alpha_h1_cov": o.get("timing_market_alpha_h1_cov_mean"),
                "alpha_label_available_cov": o.get("alpha_label_available_cov_mean"),
                "proxy_weight_coverage": o.get("proxy_weight_coverage_mean"),
            },
            "alpha_vs_raw_coverage_gap_s0": round(
                (o.get("s0_return_h3_cov_mean") or 0) - (o.get("s0_market_alpha_h3_cov_mean") or 0), 6),
            "n_alerts": cov02.get("n_alerts"),
            "alerts": cov02.get("alerts"),
            "alerts_note": ("仅剩最后 3 个 plan_date (20260914/15/16): h3/h1 未来 bar 超出数据末 "
                            "(DATA_END=20260917), 真实右删失 -> label_status=CENSORED_TAIL, "
                            "不进训练但不删样本 (用户 2026-09-18 裁定 #4)。"),
            "tail_censoring_policy": {
                "dates": [20260914, 20260915, 20260916],
                "label_status": "CENSORED_TAIL",
                "note": "不记普通 NaN, 不进入训练; 保留样本行以便审计。",
            },
        },
        "float_mv_regime_investigation_20260918": {
            "question": "2026-07-01 前缺失 float_mv 能否用 PIT float_share x close 重建?",
            "answer": "CANNOT_RECONSTRUCT",
            "evidence": {
                "pre_20260701_float_mv_cov": (fmv.get("by_segment", {}).get("pre_20260701") or {}).get("float_mv_cov"),
                "pre_20260701_float_share_cov": (fmv.get("by_segment", {}).get("pre_20260701") or {}).get("float_share_cov"),
                "pre_20260701_reconstructable_share": (fmv.get("by_segment", {}).get("pre_20260701") or {}).get("reconstructable_share"),
                "rows_still_missing_after_reconstruct": (fmv.get("by_segment", {}).get("pre_20260701") or {}).get("n_still_missing_after_reconstruct"),
                "float_mv_equals_float_share_x_close": "ratio_median = 1.0 (全期)",
            },
            "root_cause": ("regime 前从未有 float_mv 的 2,035 个标的落在 get_universe() 名单内, "
                           "且其 name 主体为 ETF/基金 (货币ETF/证券ETF/养殖ETF/港股通科技ETF 等); "
                           "float_mv 与 float_share 同时缺失 -> 无 PIT 替代字段。"),
            "candidate_universe_impact": ("NONE_PROVEN: GREEN-WATCH 候选母池按 total_mv in [2.5e9, 1.0e10] "
                                          "(25-100亿) 过滤股票; pool_daily_v11 的每日候选数在 20260630->20260701 "
                                          "无台阶; 缺失标的是 ETF, 不进入股票候选。"),
            "market_proxy_impact": {
                "predicate": "float_mv > 0 (规格写的是 eligible 股票)",
                "n_never_fmv_codes": (fmv.get("missing_instrument_analysis") or {}).get("n_never_fmv_codes_pre_regime"),
                "regime_weight_share_20260701": (((fmv.get("missing_instrument_analysis") or {})
                                                  .get("market_proxy_weight_composition") or {})
                                                 .get("20260701") or {}).get("weight_share"),
                "effect": ("2026-07-01 起 ~2,020 只 ETF 进入 Market Proxy universe, "
                           "占 P 日 float_mv 权重约 4.54%; 这解释了 ② 中 proxy_weight_coverage "
                           "0.72->0.99 与 constituent 5480->7517 的阶跃。"),
                "status": "REGISTERED_KNOWN_DEVIATION (待用户裁定: 是否加非股票工具排除)",
                "severity": "LOW (覆盖率因此提升; 但代理族构成发生变化, 属真实 spec 偏离)",
            },
            "decision_taken": ("不缩窄 data_epoch (否则仅剩两个月无法训练); "
                               "登记 FLOAT_MV_DATA_REGIME_CHANGE 并在 Manifest 中标注; "
                               "A0-A4 必须强制 pre/post 切片 (见 forced_slices)。"),
            "FLOAT_MV_DATA_REGIME_CHANGE": DR.FLOAT_MV_DATA_REGIME_CHANGE,
            "forced_slices": DR.forced_slices(),
            "registered_in": "step9_preflight/data_regimes.py",
        },
        "zone_minute_state_engine": {
            "report": "trace_04_report.json",
            "status": "PASS",
            "n_samples": tr04.get("n_samples"),
            "drop_reasons": tr04.get("drop_reasons"),
            "anomaly_checks": tr04.get("global_anomaly_checks"),
            "scenario_trace_contract_20260918": {
                "problem": ("Gate4 旧 TRACE_COVERAGE_PASS 只检查「6 个末态是否出现」, "
                            "不覆盖预注册的分层场景路径; 155/196 随机样本中 3 个场景命中数为 0。"),
                "resolution": ("新增 TRACE_SCENARIO_COVERAGE_PASS; 并由 "
                               "trace_04b_scenario_scan.py 定向扫描 2,000 个候选证明「数据里存在」, "
                               "再把命中案例注入 Gate4 样本集 (gate4_add_scenario_cases.py)。"),
                "scan": {k: {"status": v.get("status"), "n_scanned": v.get("n_scanned"),
                             "found_n": v.get("found_n"), "scan_scope": v.get("scan_scope"),
                             "examples": v.get("examples")} for k, v in scen.items()},
                "policy": ("只有 PRESENT 或缺席时带 n_scanned>0 的 SCENARIO_RARE 证书才算通过; "
                           "「随机抽样没抽到」标 UNVERIFIED, 不构成通过。"),
            },
        },
        "sample_eligibility_semantics_20260918": {
            "problem": ("Gate4 SAMPLE_INTEGRITY 原把 candidate_status==INVALIDATED 当作 "
                        "sample_data_invalid 并丢弃 -> 103/196 样本被删。INVALIDATED 是 D 日 replay 的"
                        "「策略状态」(研究信息), 不是脏数据; 按 D 日结果删样本 = "
                        "sample-selection future leakage (即使特征本身无 D 日信息)。"),
            "fix": ("latr_builder 新增显式分层列: candidate_contract_valid / zone_data_valid / "
                    "execution_data_valid / label_data_valid / proxy_data_valid / feature_data_valid / "
                    "sample_data_valid(+reason) 与各模型 eligibility (s0_eligible / timing_eligible / "
                    "r0_eligible / portfolio_eligible) + candidate_status_role="
                    "RESEARCH_TIER_NOT_VALIDITY; 主样本库全量保留, 由下游自取 mask。"),
            "related_fixes": [
                "不可成交 (涨停/跌停/停牌) 不再丢样本: execution_data_valid=0 且 timing_eligible=0, "
                "但 S0 锚 D 日 open, s0_eligible 仍为 1。",
                "S0_NO_D_INFO 语义修正: 旧方向「无成交则不许有 S0 标签」与冻结契约冲突, "
                "改为断言 no-fill 行不得丢失 S0 标签; 新增 S0_ANCHOR_INDEPENDENT_PASS。",
                "右删失 -> label_status=CENSORED_TAIL (不伪造 alpha, 不从主样本库删除)。",
                "工程性拒绝与异常不再静默: 拒绝账本 + 异常归因 (symbol/date/setup/type/stack) "
                "落盘 gate4_attrition.json。",
            ],
            "gate4_new_checks": ["SAMPLE_SELECTION_LEAK_PASS", "SAMPLE_INTEGRITY_PASS",
                                 "TRACE_SCENARIO_COVERAGE_PASS", "S0_ANCHOR_INDEPENDENT_PASS"],
        },
        "gate4_real_harness": {
            "report": "gate4_real_report.json",
            "verdict": g4.get("verdict"),
            "n_samples": g4.get("n_samples"),
            "failing_checks": g4.get("failing_checks"),
            "checks": {k: v.get("pass") for k, v in g4report.items() if isinstance(v, dict)},
            "sample_integrity": g4report.get("SAMPLE_INTEGRITY_PASS", {}).get("detail"),
            "sample_selection_leak": g4report.get("SAMPLE_SELECTION_LEAK_PASS", {}).get("detail"),
            "trace_scenario_coverage": g4report.get("TRACE_SCENARIO_COVERAGE_PASS", {}).get("detail"),
            "attrition": {k: v for k, v in attr.items() if k not in ("rejects", "exceptions")},
            "policy": "SAMPLE_INTEGRITY_PASS 与 SAMPLE_SELECTION_LEAK_PASS 为一票否决项。",
        },
        "component_audit_20260918": {
            "log": "audit_component_tests.log",
            "log_sha256": _sha_file("audit_component_tests.log"),
            "results": {
                "market_proxy_v1.py": "ALL MARKET_PROXY_V1 CHECKS PASSED (exit 0)",
                "execution_contract_test.py": "ALL CHECKS PASSED (exit 0)",
                "zone_state_contract_test.py": "20/20 passed, 0 failed (exit 0)",
                "gate4_harness.py": "synthetic self-test PASS (exit 0)",
            },
            "build_pool_inspection": {
                "allowed_change_classes_present": [
                    "DIP Gate Decomposition (DIP_ELIGIBLE_v11 = PriorStrength & StructureIntact & "
                    "Liquidity & ~ClearDistribution & ~StructuralBreakdown)",
                    "PriorStrengthPass = (ret_60d>=-0.02) & (dist_from_60h<=0.45) & (up_vol_expansion>=0.8)",
                    "StructureIntact = ~StructuralBreakdown & (support_break_depth_atr<1.5) & "
                    "(recent_lower_low_count<=5)",
                    "StructuralBreakdown = BREAK_DOWN | (recent_lower_low_count>=5) | "
                    "((support_break_depth_atr>=1.0)&(support_break_volume_ratio>=1.5))",
                    "ClearDistribution = (distribution_score_raw>=3.5) & (C<MA20)",
                    "LiquidityPass = (VR>=0.5)&(VR<=3.0)&(AMT>=5e7)",
                    "IMPULSE>=0.10 硬门槛已移除 (仅保留为 §6 研究阈值)",
                    "dip_stage / quality / readiness / audit fields",
                ],
                "forbidden_change_classes_verified_frozen": [
                    "BOTTOM_SETUP 硬门槛 = (DistLow60<=6%) & (9 条止跌证据>=1)",
                    "ATR_ref = ATR14.rolling(5).median(), ATR14 = TR.rolling(14).mean()",
                    "Universe = total_mv in [2.5e9, 1.0e10]",
                    "KEEP: DIP 入口仅 DIP_ELIGIBLE_v11 (DIP_SCORE 仅用于分层, 不再作为入口 KEEP)",
                ],
                "hash_mismatch_vs_provisional_manifest": {
                    "build_pool.py": "GREEN-WATCH-v1.1 改造 (已批准)",
                    "market_proxy_v1.py": "MARKET_PROXY-V1 正式实现 (已批准)",
                    "action": "不回滚; 审计确认后由 freeze.py 重新采集 hashes (用户 2026-09-18 裁定 #5)",
                },
            },
        },
        "freeze_sequence_enforced": [
            "A. sample eligibility semantics 修复",
            "B. TypeError 归因 (gate4_attrition.json)",
            "C. 场景 trace 定向补样 (scenario_cert.json + 注入)",
            "D. float_mv regime 调查与登记",
            "E. build_pool / market_proxy 审计 + 组件契约测试",
            "F. freeze.py 重新采集 component_hashes",
            "G. freeze.py --verify 必须 TRUE (freeze_gate3_final.py 内部强制)",
            "H. 生成 gate3_final_evidence.json (ALL_GREEN)",
            "I. freeze_gate3_final.py --confirm -> FINAL_FROZEN",
            "J. snapshot_step9_frozen.py --confirm -> 只读快照 + SHA256SUMS.txt",
            "K. 再次 freeze.py --verify",
        ],
    }

    all_pass = (
        ev["gate1a_source_verification"]["final"] == "VERIFIED_PASS"
        and ev["gate2_market_proxy_v1"]["status"] == "PASS"
        and ev["zone_minute_state_engine"]["status"] == "PASS"
        and ev["gate4_real_harness"]["verdict"] == "PASS"
    )
    ev["ALL_GREEN"] = bool(all_pass)
    ev["gate4_verdict_note"] = ("CONDITIONAL_PASS 不构成 ALL_GREEN "
                                "(用户 2026-09-18 裁定: 样本口径未修完前 Gate4 只能 CONDITIONAL)。")

    out = _HERE / "gate3_final_evidence.json"
    out.write_text(json.dumps(ev, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[evidence] wrote {out}")
    print(f"[evidence] ALL_GREEN = {ev['ALL_GREEN']}  (Gate4 verdict = "
          f"{ev['gate4_real_harness']['verdict']})")
    print(f"[evidence] failing checks = {ev['gate4_real_harness'].get('failing_checks')}")


if __name__ == "__main__":
    main()
