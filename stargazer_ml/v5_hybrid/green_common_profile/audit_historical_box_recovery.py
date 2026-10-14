"""Offline provenance/reconstruction audit. Never publishes training fields or fits.

Original tables remain immutable. Successful reconstruction is candidate evidence,
not new authorization or proof that the entire execution path has been accepted.
"""
from collections import Counter
from datetime import datetime, timedelta
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent
RUN = BASE / "gcp_round_b_small_sample_v1"
sys.path.insert(0, str(RUN))
import trigger_fields_common as common
import gcp_dual_path_coverage_probe as probe
import gcp_round_b_engine as engine
import dual_path_execution_v0_1_1 as executor
from forward_training_cli import save


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def s1_time(frame, lower, atr):
    """Only reproduce frozen S1; do not run the unresolved S2 volume algorithm."""
    for row in frame.sort_values("date").itertuples():
        if engine.MONITOR_START <= engine.hhmm(row.date) <= engine.MONITOR_END:
            if (row.low <= lower + probe.LOWER_TOL * atr and row.close >= lower
                    and not row.open == row.high == row.low == row.close):
                return int(row.date)
    return None


def reconstruct(daily, minute, row, p_date, grid):
    """Local reconstruction, independently compare frozen times/ATR/signal close."""
    result = {"code": row["code"], "E": row["E"], "segment": row["segment"],
              "status": "PENDING", "reason": None}
    history = daily[daily.date <= p_date].sort_values("date").reset_index(drop=True)
    if history.empty or history.date.duplicated().any():
        return {**result, "reason": "DAILY_MISSING_OR_DUPLICATE"}
    atr = engine.build_atr_ref(history)
    box = probe.evaluate_calibers(history, atr).get("main", {})
    if not box.get("valid"):
        return {**result, "reason": "FROZEN_BOX_RULE_INVALID"}
    if not np.isfinite(atr) or abs(atr - row["atr_ref"]) > 1e-9:
        return {**result, "reason": "ATR_FROZEN_VALUE_MISMATCH"}
    signal, decision = row["signal_bar_ts"], row["decision_available_ts"]
    correct_decision = int((datetime.strptime(str(signal), "%Y%m%d%H%M%S")
                            + timedelta(minutes=1)).strftime("%Y%m%d%H%M%S"))
    if decision != correct_decision:
        return {**result, "reason": "DECISION_TIMESTAMP_MISMATCH"}
    if minute.empty or minute.date.duplicated().any():
        return {**result, "reason": "MINUTE_MISSING_OR_DUPLICATE"}
    if (minute.date.astype("int64") // 1000000 != row["E"]).any():
        return {**result, "reason": "MINUTE_DATE_MISMATCH"}
    # Frozen historical certification starts at the monitor window, not session open.
    # A future source protocol may require a larger prefix; this does not redefine it.
    slots = engine.expected_grid(grid, row["E"])
    required = {row["E"] * 1000000 + slot for slot in slots
                if 93500 <= slot <= signal % 1000000}
    prefix = minute[minute.date <= signal].copy()
    if required - set(prefix.date.astype(int)):
        return {**result, "reason": "FIRST_DECISION_PREFIX_GAP"}
    prices = prefix[["open", "high", "low", "close", "volume", "amount"]].astype(float)
    if (not np.isfinite(prices.to_numpy()).all()
            or (prices[["open", "high", "low", "close"]] <= 0).any().any()
            or (prices[["volume", "amount"]] < 0).any().any()
            or (prices.low > prices[["open", "close"]].min(axis=1)).any()
            or (prices.high < prices[["open", "close"]].max(axis=1)).any()):
        return {**result, "reason": "INVALID_PREFIX_OHLCV"}
    first = s1_time(prefix, box["lower"], atr)
    if first != signal or s1_time(minute, box["lower"], atr) != signal:
        return {**result, "reason": "FIRST_S1_FROZEN_TIMESTAMP_MISMATCH"}
    bar = prefix[prefix.date == signal].iloc[0]
    if abs(float(bar.close) - row["signal_close"]) > 1e-9:
        return {**result, "reason": "SIGNAL_CLOSE_FROZEN_VALUE_MISMATCH"}
    # Negative test: removing the signal bar must remove this first trigger.
    if s1_time(prefix[prefix.date < signal], box["lower"], atr) is not None:
        return {**result, "reason": "EARLIER_TRIGGER_FOUND"}
    changed = minute.copy()
    changed.loc[changed.date > signal, ["open", "high", "low", "close"]] *= 1.5
    if s1_time(changed, box["lower"], atr) != signal:
        return {**result, "reason": "FUTURE_PERTURBATION_CHANGED_FIRST_TRIGGER"}
    return {**result, "status": "RECONSTRUCTION_CANDIDATE", "reason": None,
            "P": p_date, "signal_bar_ts": signal, "decision_available_ts": decision,
            "atr_ref": float(atr), "box_lower": float(box["lower"]), "box_upper": float(box["upper"]),
            "pen_atr_bar": float((box["lower"] - bar.low) / atr),
            "d_atr": float((bar.close - box["upper"]) / atr),
            "width_atr": float((box["upper"] - box["lower"]) / atr),
            "signal_close": float(bar.close), "prefix_consistency": "PASS",
            "negative_prefix_test": "PASS", "future_perturbation_test": "PASS",
            "old_region_comparison": "NO_BASELINE",
            "historical_boundary_capture": False, "training_eligible": False}


def main():
    destination = RUN / "historical_box_provenance_v0_3"
    if destination.exists():
        raise RuntimeError("NEW_OUTPUT_DIRECTORY_REQUIRED")
    manifest_path = RUN / "trigger_fields_manifest_v0_2.json"
    manifest = read(manifest_path)
    extra_inputs = [RUN / name for name in ("trigger_fields_table_v0_2.json", "r1_fit_manifest_v0_1_1.json",
                                            "gcp_247_phase1_frozen_inputs.json", "minute_grid.json")]
    extra_inputs.append(Path(__file__))
    extra_hashes = {str(p): digest(p) for p in extra_inputs}
    checks = [{"path": x["path"], "sha256": x["sha256"],
               "match": Path(x["path"]).is_file() and digest(x["path"]) == x["sha256"]}
              for x in manifest["inputs"]]
    if not all(x["match"] for x in checks):
        raise RuntimeError("FROZEN_INPUT_CHANGED:" + str([x["path"] for x in checks if not x["match"]]))
    fit_manifest = read(RUN / "r1_fit_manifest_v0_1_1.json")
    for name in ("gcp_dual_path_coverage_probe.py", "gcp_round_b_engine.py", "dual_path_execution_v0_1_1.py"):
        if digest(BASE / name) != fit_manifest["sha256"][name]:
            raise RuntimeError("FROZEN_ALGORITHM_CHANGED:" + name)
    labels = read(RUN / "r1_exec_labels_v0_1_2.json")
    original_daily = read(RUN / "gcp_247_phase1_frozen_inputs.json")["daily_parts"]["files"]
    historical_daily_hashes = {x["file"]: x["sha256"] for x in original_daily}
    all_fields = read(RUN / "trigger_fields_table_v0_2.json")
    targets = [r for r in all_fields if r["branch"] == "BASE_HOLD"]
    if len(targets) != 80 or len(all_fields) != 124:
        raise RuntimeError("FROZEN_POPULATION_CHANGED")
    branch = {(r["code"], r["E"]): r for r in read(RUN / "r1_branch_at_decision_v0_1_2.json")["events"]}
    population = {(r["code"], r["E"]): r for r in read(RUN / "r1_dip_scan_and_gap_freeze.json")["per_event"]}
    feature_rows = read(RUN / "r1_fit_features_v0_1_1.json")["H3_primary"]
    feature_index = {(r["code"], r["E"]): r for r in feature_rows}
    grid = read(RUN / "minute_grid.json")["grid"]
    # Use only the originally frozen files, never glob newly refreshed daily inputs.
    daily_files = [Path(x["path"]) for x in manifest["inputs"]
                   if Path(x["path"]).parent.name == "data_daily_parts"]
    codes = {r["code"] for r in targets}
    pieces, calendar = [], set()
    for path in daily_files:
        part = pd.read_parquet(path, columns=["code", "date", "open", "high", "low", "close", "volume"])
        part["code"] = part.code.astype(str).str.zfill(6)
        part["date"] = part.date.astype(int)
        if path.name in historical_daily_hashes:
            if digest(path) != historical_daily_hashes[path.name]:
                raise RuntimeError("ORIGINAL_DAILY_SNAPSHOT_CHANGED:" + path.name)
        elif (part.date <= max(r["P"] for r in population.values())).any():
            raise RuntimeError("NEW_PART_OVERLAPS_HISTORICAL_P_WINDOW:" + path.name)
        calendar.update(part.date.unique())
        pieces.append(part[part.code.isin(codes)])
    daily = pd.concat(pieces, ignore_index=True)
    by_code = {code: frame for code, frame in daily.groupby("code")}
    calendar = sorted(calendar)
    cal_pos = {day: i for i, day in enumerate(calendar)}
    frozen_paths = {str(Path(x["path"]).resolve()): x for x in manifest["inputs"]}
    minute_cache = {}
    def original_day(code, day):
        for directory in ("minute", "minute_ext", "minute_car"):
            path = (RUN / directory / (code + ".parquet")).resolve()
            if str(path) not in frozen_paths:
                continue
            if path not in minute_cache:
                minute_cache[path] = pd.read_parquet(path)
            frame = minute_cache[path]
            sub = frame[frame.date.astype("int64") // 1000000 == day].sort_values("date")
            if len(sub):
                return sub, {"path": str(path), "sha256": frozen_paths[str(path)]["sha256"]}
        return pd.DataFrame(), None
    records = []
    loaded, current_hashes = {}, []
    for row in targets:
        key = row["code"], row["E"]
        p_date = population[key]["P"]
        b = branch[key]
        minute, evidence = original_day(*key)
        result = reconstruct(by_code.get(row["code"], pd.DataFrame(columns=["date"])), minute, row, p_date, grid)
        if (b["branch_at_decision"] != "BASE_HOLD" or b["s1_ts"] != row["signal_bar_ts"]
                or b["s1_dec"] != row["decision_available_ts"] or b["cert"] != "CERTIFIED"):
            result.update(status="PENDING", reason="FROZEN_BRANCH_BINDING_MISMATCH")
        if row["E"] not in cal_pos or cal_pos[row["E"]] == 0 or calendar[cal_pos[row["E"]] - 1] != p_date:
            result.update(status="PENDING", reason="P_E_CALENDAR_MISMATCH")
        result["signal_source"] = b["s1_s2_source"]
        result["minute_evidence"] = evidence
        original_sources = labels["code_provenance"].get(row["code"], [])
        if isinstance(original_sources, dict):
            original_sources = [original_sources]
        if evidence is None or not any(str(Path(x["path"]).resolve()) == evidence["path"]
                                       and x["sha256"] == evidence["sha256"] for x in original_sources):
            result.update(status="PENDING", reason="ORIGINAL_EXECUTION_MINUTE_SOURCE_NOT_BOUND")
        result["original_minute_hash_bound"] = result["reason"] != "ORIGINAL_EXECUTION_MINUTE_SOURCE_NOT_BOUND"
        if row["code"] not in loaded:
            loaded[row["code"]] = executor.load_minute_code(row["code"])
            current_hashes.extend(loaded[row["code"]][1])
        effective = loaded[row["code"]][0].get(row["E"])
        columns = ["date", "open", "high", "low", "close", "volume", "amount"]
        left = minute[minute.date <= row["signal_bar_ts"]][columns].sort_values("date").reset_index(drop=True)
        right = (effective[effective.date <= row["signal_bar_ts"]][columns].sort_values("date").reset_index(drop=True)
                 if effective is not None else pd.DataFrame(columns=columns))
        equivalent = (left.shape == right.shape and np.array_equal(left.to_numpy(), right.to_numpy()))
        result["effective_executor_prefix_match"] = bool(equivalent)
        if not equivalent:
            result.update(status="PENDING", reason="EFFECTIVE_MINUTE_SOURCE_CONFLICT")
        records.append(result)
    # Do not recalculate returns: count previously admitted labels only.
    usable = {(r["code"], r["E"]) for r in all_fields if r["branch"] == "DIP_REPAIR"}
    usable.update((r["code"], r["E"]) for r in records if r["status"] == "RECONSTRUCTION_CANDIDATE")
    coverage = {}
    for segment in ("train", "val", "test"):
        rows = [r for key, r in feature_index.items() if key in usable and r["segment"] == segment]
        coverage[segment] = {"events": len(rows), "positive": sum(r["label"] == 1 for r in rows),
                             "nonpositive": sum(r["label"] == 0 for r in rows),
                             "E_dates": len({r["E"] for r in rows})}
    # Inventory existing first blockers, not a full future-path budget or a request list.
    physical = {}
    for event in labels["records"]:
        for track in ("H3", "H5"):
            t = event.get(track) or {}
            for day in t.get("md", []):
                item = physical.setdefault((event["code"], day), {"code": event["code"], "date": day, "dependencies": []})
                item["dependencies"].append({"E": event["E"], "segment": event["segment"], "track": track,
                                               "original_reason": t.get("rsn")})
    inventory = []
    for key, item in sorted(physical.items()):
        code, day = key
        if code not in loaded:
            loaded[code] = executor.load_minute_code(code)
            current_hashes.extend(loaded[code][1])
        frame = loaded[code][0].get(day)
        if frame is None:
            item.update(current_status="NO_FILE", missing_slots=None)
        else:
            expected = set(executor.day_grid(day))
            slots = frame.date.astype("int64") % 1000000
            gaps = sorted(expected - set(slots))
            bad = slots.duplicated().any() or bool(set(slots) - expected)
            numeric = frame[["open", "high", "low", "close", "volume", "amount"]].astype(float)
            bad = bool(bad or not np.isfinite(numeric.to_numpy()).all()
                       or (numeric[["open", "high", "low", "close"]] <= 0).any().any()
                       or (numeric[["volume", "amount"]] < 0).any().any()
                       or (numeric.volume > 0).sum() == 0)
            item.update(current_status="INVALID" if bad else ("INTRA_HOLE" if gaps else "GRID_COMPLETE_ONLY"), missing_slots=gaps)
        inventory.append(item)
    result = {"status": "AUDIT_ONLY_NOT_TRAINING_RELEASE", "network_requests": 0, "fitted": False,
              "original_table_unchanged": True, "frozen_manifest_sha256": digest(manifest_path),
              "frozen_inputs_checked": len(checks), "algorithm_sha256": {
                  name: fit_manifest["sha256"][name] for name in
                  ("gcp_dual_path_coverage_probe.py", "gcp_round_b_engine.py", "dual_path_execution_v0_1_1.py")},
              "additional_input_hashes": extra_hashes, "original_daily_files_verified": len(original_daily),
              "auditor_sha256": extra_hashes[str(Path(__file__))],
              "reconstruction_counts": dict(Counter(r["status"] for r in records)),
              "population_if_reconstruction_accepted": coverage, "records": records,
              "exit_first_blocker_counts": dict(Counter(x["current_status"] for x in inventory)),
              "exit_first_blockers": inventory, "exit_current_file_evidence": current_hashes,
              "limitations": ["Region values have no historical stored baseline; reconstructed candidates only",
                              "Existing labels retained, not re-executed or newly certified",
                              "First blockers do not cover unvisited paths after each blocker",
                              "Historical validation/test already observed, not new blind evidence"],
              "training_ready": False}
    # Check every original frozen input again before publishing the audit.
    if any(digest(x["path"]) != x["sha256"] for x in manifest["inputs"]):
        raise RuntimeError("INPUT_CHANGED_DURING_AUDIT")
    if any(digest(path) != value for path, value in extra_hashes.items()):
        raise RuntimeError("ADDITIONAL_INPUT_CHANGED_DURING_AUDIT")
    if any(digest(x["path"]) != x["sha256"] for x in current_hashes):
        raise RuntimeError("EXIT_CACHE_CHANGED_DURING_AUDIT")
    payload = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False).encode("utf-8")
    summary = ("# 历史箱体来源核验（只读计算）\n\n"
               "状态：AUDIT_ONLY_NOT_TRAINING_RELEASE。零网络、零采集、零拟合，旧字段表/人口/收益未改变。\n\n"
               f"冻结输入复验 {len(checks)} 项。80条BASE_HOLD重建：{result['reconstruction_counts']}。\n\n"
               f"若候选重建证据获接受，原H3同人口字段覆盖：{coverage}。这是覆盖盘点，不是拟合门裁决。\n\n"
               f"原执行标签的物理首阻点现状：{result['exit_first_blocker_counts']}。不构成补采请求，也不是完整退出路径覆盖证明。\n\n"
               "区域旧值无baseline；本件只证明原算法与冻结触发时刻/ATR/信号收盘及前缀测试一致。下一步需另批字段构建发布与最终训练准入核验，不能直接拿本件拟合。\n")
    common.assert_budget(len(payload) + len(summary.encode("utf-8")))
    save(destination / "report.json", payload)
    save(destination / "report.md", summary.encode("utf-8"))
    print(json.dumps({k: result[k] for k in ("reconstruction_counts", "population_if_reconstruction_accepted",
                                           "exit_first_blocker_counts", "training_ready")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
