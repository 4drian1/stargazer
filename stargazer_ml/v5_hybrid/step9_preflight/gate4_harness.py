"""gate4_harness.py — ⑤ 正式 Gate4 程序化检查.

对样本 DataFrame (samples_df) 程序化执行全部检查. __main__ 用合成 sdf 自测
harness 逻辑 (无需 stockdb).

检查项:
  PIPELINE_PASS               : 样本 schema 完整 (含 validity/eligibility 新契约列)
  TIME_TRAVEL_PASS            : max(source_ts) <= snapshot_ts (无未来信息)
  NO_FUTURE_LEAK              : state_transitions 时间戳均 <= snapshot
  S0_NO_D_INFO                : S0 标签不得以「是否成交」为条件 (锚 D 日 open, 非 fill)
                                [语义修正 2026-09-18: 旧方向 (无成交则不许有 S0 标签) 会
                                 迫使丢弃不可成交候选 = sample-selection leakage]
  S0_ANCHOR_INDEPENDENT_PASS  : 非删失样本必须都有 S0 标签; 无成交样本不得有 timing 标签
  TIMING_FILL_CORRECT         : timing 标签来自 actual_fill (exec_status==FILLED)
  ALPHA_INTERVAL_ALIGNED      : s0_alpha_h3 ≈ s0_return_h3 - market_proxy_return_h3 (同区间)
  TRACE_COVERAGE_PASS         : 末态状态集合覆盖 (仅「状态是否出现」)
  TRACE_SCENARIO_COVERAGE_PASS: 预注册场景路径覆盖 (PRESENT 或带证书的 RARE_PROVEN)
  SAMPLE_INTEGRITY_PASS       : 每层数据有效性 n_in/n_valid/n_dropped + 显式 drop_reason;
                                candidate_status 只作研究分层, 绝不作筛选 (veto)
  SAMPLE_SELECTION_LEAK_PASS  : eligibility 公式不得引用 candidate_status;
                                INVALIDATED 行必须被保留 (veto)

运行:
  python gate4_harness.py            # 合成自测
  python -c "import gate4_harness as G; G.run_gate4(samples_df, scenario_cert=cert)"
"""
from __future__ import annotations
import json
from typing import Dict, Any, Optional, Tuple

import numpy as np
import pandas as pd

SNAPSHOT_MINUTE = 150000  # 与 latr_builder 一致 (14 位 HHMMSS 后缀)

REQUIRED_COLUMNS = [
    "symbol", "plan_date", "effective_trade_date", "primary_setup", "d_date",
    "snapshot_time", "atr_ref", "open_dip_atr", "zone_mid", "zone_lower", "zone_upper",
    "actual_fill", "exec_status", "fill_ts", "signal_time",
    "s0_return_h3", "s0_alpha_h3", "timing_return_h1", "timing_alpha_h1",
    "market_proxy_return_h3", "proxy_timing_return",
    "final_state", "state_transitions_json", "zone_valid", "zone_error",
    # 2026-09-18 新契约: 数据有效性与各模型 eligibility 显式分离
    "candidate_status", "candidate_status_role",
    "candidate_contract_valid", "candidate_contract_invalid_reason",
    "zone_data_valid", "zone_invalid_reason",
    "execution_data_valid", "execution_invalid_reason",
    "label_data_valid", "label_invalid_reason",
    "proxy_data_valid", "proxy_invalid_reason",
    "feature_data_valid", "feature_invalid_reason",
    "sample_data_valid", "sample_invalid_reason", "label_status",
    "s0_eligible", "timing_eligible", "r0_eligible", "portfolio_eligible",
    "second_test_flag", "second_test_higher_low",
]

# 数据/工程有效性层: (flag 列, reason 列)
VALIDITY_LAYERS = [
    ("candidate_contract", "candidate_contract_valid", "candidate_contract_invalid_reason"),
    ("zone", "zone_data_valid", "zone_invalid_reason"),
    ("execution", "execution_data_valid", "execution_invalid_reason"),
    ("label", "label_data_valid", "label_invalid_reason"),
    ("proxy", "proxy_data_valid", "proxy_invalid_reason"),
    ("feature", "feature_data_valid", "feature_invalid_reason"),
]

# 预注册场景路径 (用户 ④ 分层 trace 合同)
SCENARIO_DEFS = {
    "DIP_TO_BREAKDOWN": "INTRADAY_DIP_SETUP 触 Zone 后 BREAKDOWN",
    "SECOND_TEST_HIGHER_LOW": "ZONE_TOUCH 后 recovery, 二次测试抬高低点",
    "LATE_SESSION_TOUCH": "14:00 之后才首次触及 Zone",
}


def _snap_ts(row) -> int:
    return int(f"{int(row['d_date'])}{SNAPSHOT_MINUTE}")


def _transitions(row):
    v = row.get("state_transitions_json")
    if isinstance(v, str):
        try:
            return json.loads(v)
        except Exception:
            return []
    return v if isinstance(v, list) else []


def _first_touch_ts(row) -> Optional[int]:
    for t in _transitions(row):
        if t.get("state") == "ZONE_TOUCH":
            return int(t.get("state_enter_ts", 0) or 0)
    return None


# ---------------- 检查项 ----------------
def check_pipeline(sdf: pd.DataFrame) -> Tuple[bool, Dict]:
    missing = [c for c in REQUIRED_COLUMNS if c not in sdf.columns]
    return (len(missing) == 0, {"missing_columns": missing, "n_required": len(REQUIRED_COLUMNS)})


def check_time_travel(sdf: pd.DataFrame) -> Tuple[bool, Dict]:
    snap = sdf.apply(_snap_ts, axis=1)
    bad = sdf[sdf["_max_source_ts"] > snap]
    return (len(bad) == 0, {"n_violations": int(len(bad))})


def check_no_future_leak(sdf: pd.DataFrame) -> Tuple[bool, Dict]:
    viol = 0
    for _, r in sdf.iterrows():
        snap = _snap_ts(r)
        for t in _transitions(r):
            if t.get("state_enter_ts", 0) > snap or t.get("state_exit_ts", 0) > snap:
                viol += 1
    return (viol == 0, {"transition_future_leak": int(viol)})


def check_s0_no_d_info(sdf: pd.DataFrame) -> Tuple[bool, Dict]:
    """S0 标签不得以「是否成交」为必要条件 (冻结契约: S0 锚 D 日 open, Timing 锚 actual_fill)。

    2026-09-18 语义修正: 旧方向 (exec_status!=FILLED 却有 s0 标签 = violation) 与冻结
    契约冲突, 且会迫使按 D 日执行结果丢样本 -> 改为断言 no-fill 行仍保有 S0 标签。
    """
    nofill = sdf["execution_data_valid"] == 0
    n_nofill = int(nofill.sum())
    lost = int((nofill & sdf["s0_return_h3"].isna()
                & (sdf["label_status"] != "CENSORED_TAIL")).sum())
    detail = {"no_fill_rows": n_nofill,
              "no_fill_rows_with_s0_label": int((nofill & sdf["s0_return_h3"].notna()).sum()),
              "no_fill_rows_lost_s0_label_not_censored": lost,
              "note": "S0 锚 D 日 open; 不成交不得导致 S0 样本被丢弃 (仅 CENSORED_TAIL 可缺)"}
    return (lost == 0, detail)


def check_s0_anchor_independent(sdf: pd.DataFrame) -> Tuple[bool, Dict]:
    """S0 与 Timing 锚点严格分离: 非删失样本必须有 S0 标签; 无成交不得有 Timing 标签。"""
    not_censored = sdf["label_status"] != "CENSORED_TAIL"
    s0_missing = int((not_censored & sdf["s0_return_h3"].isna()).sum())
    timing_without_fill = int(((sdf["execution_data_valid"] == 0)
                               & sdf["timing_return_h1"].notna()).sum())
    detail = {"s0_missing_but_not_censored": s0_missing,
              "timing_label_without_fill": timing_without_fill,
              "n_censored_tail": int((~not_censored).sum())}
    return (s0_missing == 0 and timing_without_fill == 0, detail)


def check_timing_fill(sdf: pd.DataFrame) -> Tuple[bool, Dict]:
    bad = sdf[(sdf["execution_data_valid"] == 0) & sdf["timing_return_h1"].notna()]
    return (len(bad) == 0, {"timing_label_without_fill": int(len(bad))})


def check_alpha_aligned(sdf: pd.DataFrame) -> Tuple[bool, Dict]:
    a = sdf["s0_alpha_h3"]; r = sdf["s0_return_h3"]; p = sdf["market_proxy_return_h3"]
    both = a.notna() & r.notna() & p.notna()
    diff = (a - r + p).abs()
    max_diff = float(diff[both].max()) if both.any() else 0.0
    return (max_diff < 1e-3, {"max_alpha_misalign": round(max_diff, 8), "n_checked": int(both.sum())})


def check_trace_coverage(sdf: pd.DataFrame) -> Tuple[bool, Dict]:
    """仅检查「末态状态集合是否出现」, 不代表场景路径覆盖 (见 TRACE_SCENARIO_COVERAGE_PASS)。"""
    required_states = {"MODEL_SCORE_REQUEST", "BREAKDOWN", "ZONE_TOUCH", "ZONE_APPROACH", "WATCH"}
    present = set(sdf["final_state"].dropna().unique())
    missing = sorted(required_states - present)
    return (len(missing) == 0, {"missing_states": missing, "present_states": sorted(present),
                                "scope": "STATE_PRESENCE_ONLY"})


def _scenario_present(sdf: pd.DataFrame) -> Dict[str, int]:
    out = {k: 0 for k in SCENARIO_DEFS}
    for _, r in sdf.iterrows():
        touch = _first_touch_ts(r)
        if (r.get("primary_setup") == "INTRADAY_DIP_SETUP" and touch is not None
                and r.get("final_state") == "BREAKDOWN"):
            out["DIP_TO_BREAKDOWN"] += 1
        if int(r.get("second_test_flag") or 0) == 1 and int(r.get("second_test_higher_low") or 0) == 1:
            out["SECOND_TEST_HIGHER_LOW"] += 1
        if touch is not None and (touch % 1000000) >= 140000 and r.get("final_state") == "ZONE_TOUCH":
            out["LATE_SESSION_TOUCH"] += 1
    return out


def check_trace_scenario_coverage(sdf: pd.DataFrame,
                                  scenario_cert: Optional[dict] = None
                                  ) -> Tuple[bool, Dict]:
    """预注册场景路径覆盖.

    每个场景状态:
      PRESENT      样本内出现
      RARE_PROVEN  样本内 0, 但定向扫描证书 (scenario_cert) 证明数据中确实极罕见/不存在
      UNVERIFIED   样本内 0 且无证书 -> 不算通过 (sampling 没抽到 ≠ 数据里没有)
    """
    present = _scenario_present(sdf)
    cert = scenario_cert or {}
    detail = {"present_in_samples": present, "scenarios": {}}
    ok = True
    for scen in SCENARIO_DEFS:
        c = cert.get(scen) or {}
        if present[scen] > 0:
            status = "PRESENT"
        elif c.get("status") in ("SCENARIO_RARE", "NOT_FOUND_PROVEN") and c.get("scan_scope") \
                and int(c.get("n_scanned", 0)) > 0:
            status = "RARE_PROVEN"
        else:
            status = "UNVERIFIED"
            ok = False
        detail["scenarios"][scen] = {"desc": SCENARIO_DEFS[scen], "status": status,
                                     "n_present": present[scen],
                                     "cert": {k: c.get(k) for k in
                                              ("status", "scan_scope", "n_scanned", "found_n",
                                               "generated_at")} if c else None}
    return (ok, detail)


def check_sample_integrity(sdf: pd.DataFrame) -> Tuple[bool, Dict]:
    """数据有效性分层 + 研究分层分离 (用户 2026-09-18 裁定)。

    veto 条件:
      1. 任一有效性层 drop 但没有显式 reason (禁止 NULL/未知桶)
      2. sample_data_valid 与声明的公式不一致
      3. candidate_status 被用于筛选 (见 SAMPLE_SELECTION_LEAK_PASS)
    """
    layers = {}
    unreasoned = {}
    for name, flag_col, reason_col in VALIDITY_LAYERS:
        f = sdf[flag_col].fillna(0).astype(int) == 1
        r = sdf[reason_col]
        miss = int((~f & r.isna()).sum())
        unreasoned[name] = miss
        layers[name] = {
            "n_in": int(len(sdf)), "n_valid": int(f.sum()), "n_dropped": int((~f).sum()),
            "drop_reason": (r[~f].fillna("<NULL>").astype(str).value_counts().to_dict()
                            if (~f).any() else {}),
            "unreasoned_drops": miss,
        }
    # 复合有效性与声明公式一致性
    exp = np.ones(len(sdf), dtype=int)
    for c in ("candidate_contract_valid", "zone_data_valid", "feature_data_valid",
              "label_data_valid", "proxy_data_valid"):
        exp &= (sdf[c].fillna(0).astype(int) == 1).to_numpy()
    formula_mismatch = int((exp != sdf["sample_data_valid"].fillna(0).astype(int).to_numpy()).sum())

    ze = sdf["zone_error"].fillna("").astype(str)
    ze_count = int((ze != "").sum())
    tier = sdf["candidate_status"].fillna("<NULL>").astype(str).value_counts().to_dict()

    integrity = {
        "layers": layers,
        "unreasoned_drops": unreasoned,
        "sample_valid_formula_mismatch": formula_mismatch,
        "research_tier": {"column": "candidate_status", "distribution": tier,
                          "role": "RESEARCH_TIER_NOT_VALIDITY",
                          "used_as_sample_filter": False},
        "zone_error_count": ze_count,
        "zone_error_rate": round(ze_count / len(sdf), 6) if len(sdf) else 0.0,
        "zone_error_by_reason": ze[ze != ""].value_counts().to_dict(),
        "eligibility_counts": {k: int(sdf[k].fillna(0).astype(int).sum())
                               for k in ("s0_eligible", "timing_eligible",
                                         "r0_eligible", "portfolio_eligible")},
    }
    all_ok = (sum(unreasoned.values()) == 0 and formula_mismatch == 0)
    return (all_ok, integrity)


def check_selection_leak(sdf: pd.DataFrame) -> Tuple[bool, Dict]:
    """样本选择不得引用 candidate_status (事后幸存者偏差守卫)。"""
    n = len(sdf)
    exp_s0 = ((sdf["sample_data_valid"].fillna(0).astype(int) == 1)
              & sdf["s0_return_h3"].notna()).astype(int)
    s0_mismatch = int((exp_s0 != sdf["s0_eligible"].fillna(0).astype(int)).sum())
    exp_t = ((sdf["sample_data_valid"].fillna(0).astype(int) == 1)
             & (sdf["execution_data_valid"].fillna(0).astype(int) == 1)
             & sdf["timing_return_h1"].notna()
             & sdf["timing_alpha_h1"].notna()).astype(int)
    t_mismatch = int((exp_t != sdf["timing_eligible"].fillna(0).astype(int)).sum())

    inv = sdf["candidate_status"].fillna("").astype(str) == "INVALIDATED"
    n_inv = int(inv.sum())
    n_inv_valid = int((inv & (sdf["sample_data_valid"].fillna(0).astype(int) == 1)).sum())
    n_inv_s0 = int((inv & (sdf["s0_eligible"].fillna(0).astype(int) == 1)).sum())
    # 明显被筛掉: 样本量足够大却一个 INVALIDATED 都没有, 或全部 INVALIDATED 的数据均无效
    filtered = (n >= 20 and n_inv == 0) or (n_inv > 0 and n_inv_valid == 0)
    detail = {
        "s0_eligible_formula_mismatch": s0_mismatch,
        "timing_eligible_formula_mismatch": t_mismatch,
        "n_candidate_status_invalidated": n_inv,
        "n_invalidated_sample_data_valid": n_inv_valid,
        "n_invalidated_s0_eligible": n_inv_s0,
        "candidate_status_used_as_sample_filter": bool(filtered),
        "n_dropped_by_candidate_status": 0,
        "note": "INVALIDATED 是 D 日 replay 结果, 属研究信息; 据此删样本 = selection leakage",
    }
    leak = s0_mismatch + t_mismatch + (1 if filtered else 0)
    return (leak == 0, detail)


CHECKS = {
    "PIPELINE_PASS": check_pipeline,
    "TIME_TRAVEL_PASS": check_time_travel,
    "NO_FUTURE_LEAK": check_no_future_leak,
    "S0_NO_D_INFO": check_s0_no_d_info,
    "S0_ANCHOR_INDEPENDENT_PASS": check_s0_anchor_independent,
    "TIMING_FILL_CORRECT": check_timing_fill,
    "ALPHA_INTERVAL_ALIGNED": check_alpha_aligned,
    "TRACE_COVERAGE_PASS": check_trace_coverage,
    "TRACE_SCENARIO_COVERAGE_PASS": check_trace_scenario_coverage,
    "SAMPLE_INTEGRITY_PASS": check_sample_integrity,
    "SAMPLE_SELECTION_LEAK_PASS": check_selection_leak,
}


def run_gate4(sdf: pd.DataFrame, scenario_cert: Optional[dict] = None) -> Dict[str, Any]:
    """对样本 DataFrame 执行全部 Gate4 检查, 返回报告 dict."""
    report = {}
    for name, fn in CHECKS.items():
        if name == "TRACE_SCENARIO_COVERAGE_PASS":
            ok, detail = fn(sdf, scenario_cert)
        else:
            ok, detail = fn(sdf)
        report[name] = {"pass": bool(ok), "detail": detail}
    report["OVERALL"] = all(v["pass"] for v in report.values())
    return report


# ---------------- 合成自测 (无需 stockdb) ----------------
def _fake_clean_sdf(n: int = 60) -> pd.DataFrame:
    rows = []
    for i in range(n):
        rows.append({
            "symbol": "000001.SZ", "plan_date": 20260105, "effective_trade_date": 20260106,
            "primary_setup": "BOTTOM_SETUP", "d_date": 20260106,
            "snapshot_time": 20260106150000, "atr_ref": 2.0, "open_dip_atr": 0.2,
            "zone_mid": 101.0, "zone_lower": 100.0, "zone_upper": 102.0,
            "actual_fill": 10.0, "exec_status": "FILLED", "fill_ts": 20260106093100,
            "signal_time": 20260106093100,
            "s0_return_h3": 0.05, "s0_alpha_h3": 0.02, "timing_return_h1": 0.03,
            "timing_alpha_h1": 0.01, "market_proxy_return_h3": 0.03, "proxy_timing_return": 0.02,
            "final_state": "MODEL_SCORE_REQUEST",
            "state_transitions_json": json.dumps(
                [{"state": "ZONE_TOUCH", "state_enter_ts": 20260106100000,
                  "state_exit_ts": 20260106103000, "trigger_reason": "x",
                  "zone_id": "Z1", "snapshot_ts": 20260106150000}]),
            "zone_valid": 1, "zone_error": None,
            "candidate_status": "PASS", "candidate_status_role": "RESEARCH_TIER_NOT_VALIDITY",
            "candidate_contract_valid": 1, "candidate_contract_invalid_reason": None,
            "zone_data_valid": 1, "zone_invalid_reason": None,
            "execution_data_valid": 1, "execution_invalid_reason": None,
            "label_data_valid": 1, "label_invalid_reason": None,
            "proxy_data_valid": 1, "proxy_invalid_reason": None,
            "feature_data_valid": 1, "feature_invalid_reason": None,
            "sample_data_valid": 1, "sample_invalid_reason": None, "label_status": "OK",
            "s0_eligible": 1, "timing_eligible": 1, "r0_eligible": 1, "portfolio_eligible": 1,
            "second_test_flag": 0, "second_test_higher_low": 0,
            "_max_source_ts": 20260106150000,
        })
    df = pd.DataFrame(rows)
    # 状态覆盖
    for i, st in enumerate(["MODEL_SCORE_REQUEST", "BREAKDOWN", "ZONE_TOUCH",
                            "ZONE_APPROACH", "WATCH"]):
        df.loc[i, "final_state"] = st
    # 场景覆盖: DIP->BREAKDOWN / 二次测试抬高低点 / 午后触 Zone
    df.loc[0, "primary_setup"] = "INTRADAY_DIP_SETUP"
    df.loc[0, "final_state"] = "BREAKDOWN"
    df.loc[1, "second_test_flag"] = 1
    df.loc[1, "second_test_higher_low"] = 1
    df.loc[2, "final_state"] = "ZONE_TOUCH"
    df.loc[2, "state_transitions_json"] = json.dumps(
        [{"state": "ZONE_TOUCH", "state_enter_ts": 20260106143000,
          "state_exit_ts": 20260106150000, "trigger_reason": "x",
          "zone_id": "Z1", "snapshot_ts": 20260106150000}])
    # 保留 INVALIDATED 样本 (证明未被筛选)
    for i in range(3, 12):
        df.loc[i, "candidate_status"] = "INVALIDATED"
    return df


def _fake_broken_sdf() -> pd.DataFrame:
    df = _fake_clean_sdf(12)
    df.loc[0, "_max_source_ts"] = 20260106160000            # 未来信息
    df.loc[1, "state_transitions_json"] = json.dumps(
        [{"state": "WATCH", "state_enter_ts": 20260106160000, "state_exit_ts": 20260106160000,
          "trigger_reason": "x", "zone_id": "Z1", "snapshot_ts": 20260106150000}])
    df.loc[2, "zone_valid"] = 0
    df.loc[2, "zone_error"] = "KeyError: 'volume'"           # zone_error -> invalid
    df.loc[2, "zone_data_valid"] = 0
    df.loc[2, "zone_invalid_reason"] = "KeyError: 'volume'"
    df.loc[2, "sample_data_valid"] = 0
    df.loc[2, "s0_eligible"] = 0
    df.loc[2, "timing_eligible"] = 0
    df.loc[3, "exec_status"] = "NOT_FILLED"                 # 无成交却有 timing 标签
    df.loc[3, "execution_data_valid"] = 0
    df.loc[3, "execution_invalid_reason"] = "exec_status=NOT_FILLED"
    df.loc[3, "timing_eligible"] = 0
    df.loc[3, "timing_return_h1"] = 0.02                    # 非法: 无成交不得有 timing 标签
    df.loc[4, "zone_data_valid"] = 0                          # 无效但无 reason -> 未知桶
    df.loc[4, "sample_data_valid"] = 0
    df.loc[4, "s0_eligible"] = 0
    df.loc[4, "timing_eligible"] = 0
    return df


def _self_test():
    clean = _fake_clean_sdf()
    r_clean = run_gate4(clean)
    assert r_clean["OVERALL"], f"clean sdf should PASS all: " \
        f"{ {k: v['detail'] for k, v in r_clean.items() if k != 'OVERALL' and not v['pass']} }"
    broken = _fake_broken_sdf()
    r_broken = run_gate4(broken)
    assert not r_broken["OVERALL"], "broken sdf should FAIL Gate4"
    assert not r_broken["TIME_TRAVEL_PASS"]["pass"], "future ts must fail TIME_TRAVEL"
    assert not r_broken["NO_FUTURE_LEAK"]["pass"], "transition future leak must fail"
    assert not r_broken["SAMPLE_INTEGRITY_PASS"]["pass"], "unreasoned drop must fail integrity"
    assert not r_broken["TIMING_FILL_CORRECT"]["pass"], "timing label without fill must fail"
    assert r_broken["SAMPLE_SELECTION_LEAK_PASS"]["pass"], \
        "broken sdf still retains INVALIDATED rows -> no selection leak"
    # 全 INVALIDATED 被干掉 (n>=20) -> 必须报 selection leak
    import copy
    d2 = copy.deepcopy(clean)
    d2 = d2[d2["candidate_status"] != "INVALIDATED"].reset_index(drop=True)
    d2 = pd.concat([d2] * 3, ignore_index=True)
    assert not run_gate4(d2)["SAMPLE_SELECTION_LEAK_PASS"]["pass"], \
        "dropping all INVALIDATED rows must be flagged"
    # 无证书且样本内 0 场景 -> UNVERIFIED
    d3 = copy.deepcopy(clean)
    d3["second_test_flag"] = 0
    d3["second_test_higher_low"] = 0
    r3 = run_gate4(d3)
    assert not r3["TRACE_SCENARIO_COVERAGE_PASS"]["pass"], "missing scenario w/o cert must fail"
    r4 = run_gate4(d3, scenario_cert={"SECOND_TEST_HIGHER_LOW": {
        "status": "SCENARIO_RARE", "scan_scope": "2018-2026 all DIP candidates",
        "n_scanned": 12345, "found_n": 0}})
    assert r4["TRACE_SCENARIO_COVERAGE_PASS"]["pass"], "certified RARE must pass"
    return r_clean, r_broken


if __name__ == "__main__":
    r_clean, r_broken = _self_test()
    print("[gate4] harness self-test PASS (clean=OVERALL_PASS, broken=FAIL as expected)")
    print(f"[gate4] clean selection-leak: "
          f"{json.dumps(r_clean['SAMPLE_SELECTION_LEAK_PASS']['detail'], ensure_ascii=False)}")
    print(f"[gate4] clean trace-scenario: "
          f"{json.dumps(r_clean['TRACE_SCENARIO_COVERAGE_PASS']['detail']['present_in_samples'])}")
    print("[gate4] NOTE: 真实执行需 stockdb; 场景证书由 trace_04b_scenario_scan.py 生成。")
