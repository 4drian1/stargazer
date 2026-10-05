"""deployment_acceptance.py — C4_DEPLOYMENT_ACCEPTANCE_V1（运行时语义验收，不涉及训练）。

验证 6 件事并写 artifacts/c4_deployment_acceptance_passed.json：
1. IDEMPOTENCY       同 (event_cluster_id, model_version) 不产生两个不同 prediction
2. APPEND-ONLY       realization 独立 append 到第二张表，prediction 表永不 UPDATE
3. MODEL HASH VERIFY manifest hash 与本地 artifact 不一致 → scorer 拒绝启动（含负例）
4. EVENT-TIME ORDER  prediction_ts >= available_at 且 feature_asof <= prediction_ts
5. RETRY SAFETY      ingest 重试/重启不重复写多条 shadow prediction（unique key upsert）
6. FAILURE EXPLICIT  feature 缺失 / entity 无效 / 非 P0 → 显式 REJECTED/NOT_SCORED，不静默丢

真实 DB 须物理拆成两张表：c4_shadow_prediction(PK event_cluster_id, model_version) 永不 UPDATE；
c4_shadow_realization(PK event_cluster_id, model_version, label_horizon) 独立 append。
"""
from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
import datetime as dt
from dataclasses import replace
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from c4_shadow import score_event, MaterialityModel
from corvus.materiality_v1.shadow_store import ShadowStore

ART = Path(__file__).resolve().parent / "artifacts"
OUT = "out"


def _load_real_cluster():
    ev = pd.read_parquet(os.path.join(OUT, "event_cluster.parquet"))
    return ev[(ev["tier"] == "P0") & (ev["label_status"] == "OK") &
              (ev["attribution_quality"] != "HEAVY") & (ev["category"] != "DISTRIBUTION")].iloc[0].to_dict()


def check_idempotency(rec):
    s1, _ = score_event(rec, as_of_date=99991231)
    s2, _ = score_event(rec, as_of_date=99991231)
    p_equal = (s1.p_materiality == s2.p_materiality) and \
              (s1.materiality_model_version == s2.materiality_model_version)
    store = ShadowStore()
    store.upsert_prediction(s1)
    store.upsert_prediction(s2)
    one_row = len(store) == 1
    # 注入不同结果应被拒
    tampered = replace(s1, p_materiality=(s1.p_materiality or 0) + 0.5)
    reject = False
    try:
        store.upsert_prediction(tampered)
    except RuntimeError:
        reject = True
    return (p_equal and one_row and reject), {
        "p_equal": p_equal, "store_one_row": one_row, "tamper_rejected": reject}


def check_append_only(rec):
    s, _ = score_event(rec, as_of_date=99991231)
    store = ShadowStore()
    store.upsert_prediction(s)
    store.attach_realized_label(s.event_cluster_id, s.materiality_model_version, "T+5", True)
    row = store.get(s.event_cluster_id, s.materiality_model_version)
    pred_unchanged = row["p_materiality"] == s.p_materiality
    realized = store.get_realization(s.event_cluster_id, s.materiality_model_version, "T+5")
    realized_set = (realized is not None) and (realized["realized_label"] is True)
    # 预测表字段不携带 realization 信息（物理隔离）
    pred_not_mutated = "later_realized_label" not in row
    return (pred_unchanged and realized_set and pred_not_mutated), {
        "pred_unchanged": pred_unchanged, "realized_attached": realized_set,
        "pred_not_mutated": pred_not_mutated}


def check_model_hash_verify():
    ok_pos = True
    try:
        MaterialityModel()
    except RuntimeError:
        ok_pos = False
    mp = ART / "model_manifest.json"
    tmp = Path(tempfile.mkdtemp()) / "tampered_manifest.json"
    m = json.loads(mp.read_text(encoding="utf-8"))
    first = m["model_files"][0]
    m["model_sha256"][Path(first).name] = "0" * 64
    tmp.write_text(json.dumps(m, ensure_ascii=False), encoding="utf-8")
    rejected = False
    try:
        MaterialityModel(str(tmp))
    except RuntimeError:
        rejected = True
    shutil.rmtree(tmp.parent, ignore_errors=True)
    return (ok_pos and rejected), {"load_ok": ok_pos, "tamper_rejected": rejected}


def check_event_time_order(rec):
    s, _ = score_event(rec, as_of_date=99991231)
    pred_date = int(s.created_ts[:10].replace("-", ""))
    ok = (pred_date >= s.available_at) and (s.feature_asof is not None) and \
         (s.feature_asof <= pred_date)
    s2, d2 = score_event(rec, as_of_date=20000101)  # as_of 早于 available_at
    future_flagged = (not d2["eligible"]) and ("available_at_in_future" in d2["problems"])
    return (ok and future_flagged), {
        "pred_ge_avail": pred_date >= s.available_at,
        "feat_asof_le_pred": s.feature_asof <= pred_date,
        "future_asof_rejected": future_flagged}


def check_retry_safety(rec):
    s, _ = score_event(rec, as_of_date=99991231)
    store = ShadowStore()
    for _ in range(3):  # 模拟 ingest 重试 + 进程重启
        store.upsert_prediction(s)
    return (len(store) == 1), {"rows_after_3_upserts": len(store)}


def check_failure_explicit():
    ev = pd.read_parquet(os.path.join(OUT, "event_cluster.parquet"))
    nonp0 = ev[ev["tier"] != "P0"].iloc[0].to_dict()
    s_n, d_n = score_event(nonp0, as_of_date=99991231)
    nonp0_explicit = (not d_n["eligible"]) and bool(d_n["problems"]) and \
                     s_n.calibration_status == "DOMAIN_INELIGIBLE"
    bad = nonp0.copy(); bad["code"] = float("nan")
    s_b, d_b = score_event(bad, as_of_date=99991231)
    entity_explicit = (not d_b["eligible"]) and ("entity_unmapped" in d_b["problems"])
    fake = nonp0.copy(); fake["code"] = "999999"; fake["tier"] = "P0"
    s_f, d_f = score_event(fake, as_of_date=99991231)
    feat_explicit = (not d_f["eligible"]) and s_f.calibration_status == "DOMAIN_OR_FEATURE_FAILED"
    return (nonp0_explicit and entity_explicit and feat_explicit), {
        "nonp0_explicit": nonp0_explicit, "entity_explicit": entity_explicit,
        "feature_explicit": feat_explicit}


def main():
    rec = _load_real_cluster()
    results = {}
    results["1_IDEMPOTENCY"], results["1_detail"] = check_idempotency(rec)
    results["2_APPEND_ONLY"], results["2_detail"] = check_append_only(rec)
    results["3_MODEL_HASH_VERIFY"], results["3_detail"] = check_model_hash_verify()
    results["4_EVENT_TIME_ORDER"], results["4_detail"] = check_event_time_order(rec)
    results["5_RETRY_SAFETY"], results["5_detail"] = check_retry_safety(rec)
    results["6_FAILURE_EXPLICIT"], results["6_detail"] = check_failure_explicit()

    passed = {k: v for k, v in results.items() if not k.endswith("_detail")}
    all_ok = all(passed.values())
    ART.mkdir(parents=True, exist_ok=True)
    if not all_ok:
        print("DEPLOYMENT ACCEPTANCE FAILED:")
        for k, v in passed.items():
            print(f"  {'PASS' if v else 'FAIL'}  {k}")
        raise SystemExit(1)
    flag = {
        "name": "C4_DEPLOYMENT_ACCEPTANCE_V1",
        "passed_at": dt.datetime.now().isoformat(timespec="seconds"),
        "checks": {k: ("PASS" if v else "FAIL") for k, v in passed.items()},
        "details": {k: v for k, v in results.items() if k.endswith("_detail")},
        "unique_key": "(event_cluster_id, materiality_model_version)",
        "note": "不涉及训练；仅验证实时运行语义。真实 DB 须以该唯一键做 upsert/约束。",
    }
    (ART / "c4_deployment_acceptance_passed.json").write_text(
        json.dumps(flag, ensure_ascii=False, indent=1), encoding="utf-8")
    print("DEPLOYMENT ACCEPTANCE PASSED (6/6)；已写", ART / "c4_deployment_acceptance_passed.json")


if __name__ == "__main__":
    main()
