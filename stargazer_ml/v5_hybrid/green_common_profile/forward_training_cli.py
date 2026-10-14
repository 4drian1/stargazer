"""Fail-closed offline handoff: check a frozen prospective dataset, then fit on request.

No acquisition, no live monitoring, no threshold tuning, no automatic training.
The evidence format is explicit; legacy history cannot silently impersonate T0 data.
"""
import argparse
from datetime import datetime, timedelta
import hashlib
import io
import json
import math
import os
from pathlib import Path
import sys

BASE = Path(__file__).resolve().parent
ROOTS = (BASE / "gcp_round_b_small_sample_v1", BASE / "gcp_round_b_bg_value_v1",
         BASE.parent / "green_v2" / "shadow_bypass")
BASE_NUM = ["contraction", "vol_cv10"]
BASE_CAT = ["setup", "branch"]
EXTRA_NUM = ["pen_atr_bar", "d_atr", "width_atr"]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def ts(value):
    if type(value) is not int or len(str(value)) != 14:
        raise ValueError("FULL_TIMESTAMP_REQUIRED")
    datetime.strptime(str(value), "%Y%m%d%H%M%S")
    return value


def finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def verify_config(config):
    if config.get("status") != "SIGNED" or config.get("fit_authorized") is not True:
        raise ValueError("SIGNED_FIT_CONTRACT_AND_AUTHORIZATION_REQUIRED")
    for role in ("dataset", "labels", "population", "source_acceptance", "protocol", "fit_contract"):
        item = config["inputs"][role]
        if sha(item["path"]) != item["sha256"]:
            raise ValueError("INPUT_HASH_MISMATCH:" + role)
    for path, digest in config["code_hashes"].items():
        if sha(path) != digest:
            raise ValueError("CODE_HASH_MISMATCH:" + path)
    if str(Path(__file__).resolve()) not in config["code_hashes"]:
        raise ValueError("TRAINER_CODE_NOT_FROZEN")
    executor = str((BASE / "dual_path_execution_v0_1_1.py").resolve())
    if config["code_hashes"].get(executor) != config["executor_sha256"]:
        raise ValueError("EXECUTOR_CODE_NOT_FROZEN")
    source = read(config["inputs"]["source_acceptance"]["path"])
    if source.get("real_time_ready") is not True or source.get("real_trade_day_verified") is not True:
        raise ValueError("LIVE_SOURCE_NOT_ACCEPTED")
    if config.get("branch_scope") != ["DIP_REPAIR", "BASE_HOLD"]:
        raise ValueError("TWO_BRANCH_SCOPE_REQUIRED")
    for window in config["windows"].values():
        if ts(window["start"]) > ts(window["end"]):
            raise ValueError("INVALID_WINDOW")
    if config["windows"]["train"]["end"] >= config["windows"]["val"]["start"]:
        raise ValueError("OVERLAPPING_WINDOWS")
    ts(config["t0_ts"])
    if config["t0_ts"] > config["windows"]["train"]["start"]:
        raise ValueError("TRAIN_WINDOW_BEFORE_T0")
    if "test" in config["windows"] and config["windows"]["val"]["end"] >= config["windows"]["test"]["start"]:
        raise ValueError("OVERLAPPING_WINDOWS")
    if config.get("thresholds_frozen_before_t0") is not True:
        raise ValueError("UNFROZEN_THRESHOLDS")
    if config.get("model") != {"C": 0.1, "threshold": 0.5, "seed": 20261008}:
        raise ValueError("FIXED_MODEL_CONFIGURATION_REQUIRED")
    for segment in ("train", "val"):
        gates = config["gates"][segment]
        for key in ("events", "positive", "nonpositive", "dates"):
            if type(gates[key]) is not int or gates[key] <= 0:
                raise ValueError("UNSET_SAMPLE_GATE:" + segment + ":" + key)
        for key in ("C_det", "F_int", "L3_avl"):
            if not finite(gates[key]) or not 0 < gates[key] <= 1:
                raise ValueError("UNSET_COVERAGE_GATE:" + key)
        for branch in config["branch_scope"]:
            for key in ("events", "positive", "nonpositive", "dates"):
                value = config["branch_gates"][segment][branch][key]
                if type(value) is not int or value <= 0:
                    raise ValueError("UNSET_BRANCH_GATE:" + segment + ":" + branch + ":" + key)


def prepare(config):
    """Same population for original/expanded/rule; every excluded row is reported."""
    verify_config(config)
    items = config["inputs"]
    rows = read(items["dataset"]["path"])["records"]
    labels = read(items["labels"]["path"])["records"]
    population = read(items["population"]["path"])["records"]
    label_index = {}
    for label in labels:
        key = label["event_id"]
        if key in label_index:
            raise ValueError("DUPLICATE_EXECUTION_LABEL")
        label_index[key] = label
    def candidate_key(row):
        return str(row["code"]), int(row["E"])
    expected = {candidate_key(r) for r in population}
    if len(expected) != len(population) or {candidate_key(r) for r in rows} != expected:
        raise ValueError("POPULATION_MISMATCH")
    if len({candidate_key(r) for r in rows}) != len(rows):
        raise ValueError("DUPLICATE_CANDIDATE")
    accepted, excluded = [], []
    stats = {s: {"population": 0, "det": 0, "actions": 0, "features": 0, "labels": 0}
             for s in config["windows"]}
    for row in rows:
        reason = None
        date_start = int(row["E"]) * 1000000
        segment = next((s for s, w in config["windows"].items()
                        if w["start"] // 1000000 <= row["E"] <= w["end"] // 1000000), None)
        if segment is None or date_start < config["t0_ts"] // 1000000 * 1000000:
            reason = "OUTSIDE_FROZEN_WINDOW"
        else:
            st = stats[segment]
            st["population"] += 1
            status = row.get("status")
            if status in ("ACTION_CERTIFIED", "NO_ACTION_COMPLETE", "NO_FILL_CONFIRMED"):
                st["det"] += 1
            if status not in ("ACTION_CERTIFIED", "NO_FILL_CONFIRMED"):
                reason = status or "UNKNOWN_STATUS"
            elif row.get("branch") not in config["branch_scope"]:
                reason = "BRANCH_OUT_OF_SCOPE"
            else:
                st["actions"] += 1
                if status == "NO_FILL_CONFIRMED":
                    excluded.append({"code": row["code"], "E": row["E"], "reason": "NO_FILL_NOT_A_TRAINING_LABEL"})
                    continue
                if not all(finite(row.get(k)) for k in BASE_NUM + EXTRA_NUM):
                    reason = "MISSING_FEATURE"
                elif row.get("setup") not in ("A_SDP", "B_LVB"):
                    reason = "UNKNOWN_SETUP"
                else:
                    st["features"] += 1
                label = label_index.get(row.get("event_id"))
                label_valid = False
                if label:
                    try:
                        end = ts(label["label_end_ts"])
                        available = ts(row["decision_available_ts"])
                        start = ts(row["signal_bar_ts"])
                        cbuy, csell, net = label["C_buy"], label["C_sell"], label["net"]
                        label_valid = (label["track"] == "H3" and label["status"] == "SIM_COMPLETE"
                            and type(label["remaining_shares"]) is int and label["remaining_shares"] == 0
                            and label["missing_dependencies"] == [] and label["dependency_check_passed"] is True
                            and label["corporate_action_check_passed"] is True
                            and label["scale_check_passed"] is True
                            and label["executor_sha256"] == config["executor_sha256"]
                            and label["input_manifest_sha256"] == row["input_manifest_sha256"]
                            and label["trigger_content_hash"] == row["trigger_content_hash"]
                            and label["code"] == row["code"] and label["E"] == row["E"]
                            and finite(cbuy) and cbuy > 0 and finite(csell) and csell >= 0 and finite(net)
                            and abs(net - (csell / cbuy - 1)) <= 1e-9
                            and available == int((datetime.strptime(str(start), "%Y%m%d%H%M%S")
                                                  + timedelta(minutes=1)).strftime("%Y%m%d%H%M%S"))
                            and available <= ts(label["last_fill_ts"]) < end
                            and end == int((datetime.strptime(str(label["last_fill_ts"]), "%Y%m%d%H%M%S")
                                            + timedelta(minutes=1)).strftime("%Y%m%d%H%M%S"))
                            and end <= config["windows"][segment]["end"])
                        if end > config["windows"][segment]["end"]:
                            reason = "CROSS_WINDOW_LABEL"
                    except (KeyError, TypeError, ValueError, ZeroDivisionError):
                        label_valid = False
                if label_valid:
                    st["labels"] += 1
                elif reason is None:
                    reason = "EXECUTION_EVIDENCE_INVALID"
                if reason is None and label_valid:
                    accepted.append({**row, "segment": segment, "net": net,
                                     "opp": (csell - cbuy) / 50000.0, "target": int(net > 0)})
        if reason:
            excluded.append({"code": row["code"], "E": row["E"], "reason": reason})
    failures = []
    branch_stats = {}
    for segment, st in stats.items():
        subset = [r for r in accepted if r["segment"] == segment]
        st.update(events=len(subset), positive=sum(r["target"] for r in subset),
                  nonpositive=sum(not r["target"] for r in subset), dates=len({r["E"] for r in subset}),
                  C_det=st["det"] / st["population"] if st["population"] else 0,
                  F_int=st["features"] / st["actions"] if st["actions"] else 0,
                  L3_avl=st["labels"] / st["actions"] if st["actions"] else 0)
        if segment in ("train", "val"):
            failures.extend(segment + ":" + k for k, minimum in config["gates"][segment].items()
                            if st[k] < minimum)
            branch_stats[segment] = {}
            for branch in config["branch_scope"]:
                group = [r for r in subset if r["branch"] == branch]
                counts = dict(events=len(group), positive=sum(r["target"] for r in group),
                              nonpositive=sum(not r["target"] for r in group),
                              dates=len({r["E"] for r in group}))
                branch_stats[segment][branch] = counts
                failures.extend(segment + ":" + branch + ":" + key
                                for key, minimum in config["branch_gates"][segment][branch].items()
                                if counts[key] < minimum)
    return accepted, {"training_ready": not failures, "stats": stats, "excluded": excluded,
                      "gate_failures": failures, "population_size": len(rows),
                      "branch_stats": branch_stats,
                      "independent_prospective_validation": False,
                      "scope": "conditional evaluable subset; not a full strategy or account backtest"}


def save(path, content):
    """Only write under registered roots; account for atomic temporary-file peak."""
    path = Path(path).resolve()
    if not any(path.is_relative_to(root.resolve()) for root in ROOTS):
        raise ValueError("OUTPUT_OUTSIDE_REGISTERED_ROOTS")
    if path.exists():
        raise ValueError("OUTPUT_EXISTS_NO_OVERWRITE")
    used = sum(p.stat().st_size for root in ROOTS if root.exists()
               for p in root.rglob("*") if p.is_file())
    if used + len(content) > 128 * 1024 * 1024:
        raise ValueError("REGISTERED_BUDGET_STOP")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("xb") as stream:
        stream.write(content)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def fit(rows, config):
    import numpy as np
    import pandas as pd
    import joblib
    from sklearn.compose import ColumnTransformer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
    frame = pd.DataFrame(rows)
    train = frame[frame.segment == "train"]
    predictions, artifacts = {}, {}
    for name, numeric in (("original", BASE_NUM), ("expanded", BASE_NUM + EXTRA_NUM)):
        prep = ColumnTransformer([("num", StandardScaler(), numeric),
                                  ("cat", OneHotEncoder(handle_unknown="ignore"), BASE_CAT)])
        model = Pipeline([("prep", prep), ("lr", LogisticRegression(
            C=0.1, penalty="l2", solver="lbfgs", class_weight=None,
            max_iter=1000, tol=1e-4, random_state=20261008))])
        model.fit(train, train.target)
        if (model["lr"].n_iter_ >= 1000).any():
            raise ValueError("MODEL_NOT_CONVERGED")
        p = model.predict_proba(frame)[:, 1]
        if not np.isfinite(p).all():
            raise ValueError("NONFINITE_PREDICTION")
        predictions[name] = p
        stream = io.BytesIO()
        joblib.dump(model, stream)
        artifacts[name] = stream.getvalue()
    summaries = {}
    for segment in config["windows"]:
        sub = frame[frame.segment == segment]
        idx = frame.index[frame.segment == segment]
        if sub.empty:
            continue
        opp = sub.opp.to_numpy()
        actions = {n: predictions[n][idx] >= 0.5 for n in predictions}
        actions.update(rule=sub.contraction.to_numpy() <= 1.0, all_pass=np.ones(len(sub), dtype=bool))
        values = {n: np.where(mask, opp, 0.0) for n, mask in actions.items()}
        dates = sub.E.to_numpy()
        groups = [np.where(dates == d)[0] for d in sorted(set(dates))]
        segment_out = {}
        for n, mask in actions.items():
            selected = sub.net.to_numpy()[mask]
            segment_out[n] = {"retained": int(mask.sum()), "events": len(sub),
                              "opportunity_mean": float(values[n].mean()),
                              "own_net_mean": float(selected.mean()) if len(selected) else None,
                              "left_tail_count": int((selected <= -0.05).sum())}
        for comparator in ("original", "rule", "all_pass"):
            delta = values["expanded"] - values[comparator]
            rng = np.random.default_rng(20261008)
            draws = [delta[np.concatenate([groups[j] for j in rng.integers(0, len(groups), len(groups))])].mean()
                     for _ in range(10000)]
            segment_out["expanded_minus_" + comparator] = {
                "mean": float(delta.mean()), "ci95": list(map(float, np.quantile(draws, [0.025, 0.975]))),
                "date_clusters": len(groups)}
        summaries[segment] = segment_out
    return summaries, artifacts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("check", "train"))
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        config = read(args.config)
        rows, report = prepare(config)
        if args.command == "train":
            if not report["training_ready"]:
                raise ValueError("DATA_NOT_READY:" + ",".join(report["gate_failures"]))
            if args.output is None:
                raise ValueError("OUTPUT_DIRECTORY_REQUIRED")
            destination = args.output.resolve()
            if not any(destination.is_relative_to(root.resolve()) for root in ROOTS):
                raise ValueError("OUTPUT_OUTSIDE_REGISTERED_ROOTS")
            if destination.exists():
                raise ValueError("NEW_OUTPUT_DIRECTORY_REQUIRED")
            summaries, artifacts = fit(rows, config)
            report["evaluation"] = summaries
            report["config_sha256"] = sha(args.config)
            report["model_sha256"] = {name: hashlib.sha256(content).hexdigest()
                                      for name, content in artifacts.items()}
            report["protocol_sha256"] = config["inputs"]["protocol"]["sha256"]
            report["model_configuration"] = config["model"]
            report["evaluated_event_keys"] = [{"code": r["code"], "E": r["E"], "segment": r["segment"]}
                                              for r in rows]
            for name, content in artifacts.items():
                save(args.output / (name + ".joblib"), content)
            save(args.output / "results.json", json.dumps(report, ensure_ascii=False,
                 indent=2, allow_nan=False).encode("utf-8"))
        print(json.dumps(report, ensure_ascii=False, allow_nan=False))
        return 0 if report["training_ready"] else 2
    except (ValueError, KeyError, TypeError, FileNotFoundError) as exc:
        print(json.dumps({"training_ready": False, "blocked": str(exc)}, ensure_ascii=False))
        return 2


if __name__ == "__main__":
    sys.exit(main())
