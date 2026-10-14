"""Separate historical check/train entry. Prepare/check never fit a model."""
import argparse
from datetime import datetime
import io
import json
from pathlib import Path
import warnings

import numpy as np
from forward_training_cli import save
from historical_trigger_extension import BASE, RUN, OUT, NUMERIC, read, sha, index

CONTRACT = BASE / "contracts/historical_trigger_increment_contract_v0_1.md"
CHAIN = BASE.parent / "opportunity_v2/gate1c/feb2026_adjustment_cache_v1/adj_cache.json"
CONFIG = OUT / "fit_config_draft_v0_2.json"
FIXED = {"C": 0.1, "threshold": 0.5, "model_seed": 20260928,
         "bootstrap_seed": 20261008, "bootstrap_draws": 10000, "budget_per_event": 50000,
         "train_positive": 30, "train_nonpositive": 30, "train_dates": 40}


def prepare():
    inputs = [OUT / n for n in ("manifest.json", "fields.json", "ledger_evidence.json", "readiness.json")]
    inputs += [CONTRACT, Path(__file__), BASE / "historical_trigger_extension.py",
               BASE / "forward_training_cli.py", CHAIN,
               BASE / "contracts/dual_path_execution_contract_v0_1_1_final.md",
               BASE / "contracts/dual_path_execution_v0_1_1_manifest.json"]
    config = {"status": "DRAFT", "signed_by": None, "fit_authorized": False,
              "fit_authorized_by": None, "scope": "HISTORICAL_EXPLORATORY_124",
              "fixed": FIXED, "inputs": {str(p): sha(p) for p in inputs},
              "dataset": str(OUT / "fields.json"), "manifest": str(OUT / "manifest.json"),
              "readiness": str(OUT / "readiness.json"), "contract": str(CONTRACT)}
    save(CONFIG, json.dumps(config, ensure_ascii=False, indent=2).encode("utf-8"))


def authorize(config, signed_by, fit_authorized_by):
    """User-invoked signature and separate explicit fit approval; never fit here."""
    check(config)
    if config["status"] != "DRAFT" or not signed_by.strip() or not fit_authorized_by.strip():
        raise ValueError("EXPLICIT_USER_SIGNATURE_AND_FIT_APPROVAL_REQUIRED")
    signed = {**config, "status": "SIGNED", "signed_by": signed_by,
              "fit_authorized": True, "fit_authorized_by": fit_authorized_by,
              "signed_at": datetime.now().astimezone().isoformat(),
              "declaration": "User accepts the hash-bound fixed historical contract and separately authorizes one fit."}
    _, verdict = check(signed)
    if not verdict["training_ready"]:
        raise ValueError("SIGNED_CONFIGURATION_NOT_READY")
    path = OUT / "fit_config_signed.json"
    save(path, json.dumps(signed, ensure_ascii=False, indent=2).encode("utf-8"))
    return path


def validate_rows(rows):
    if len(rows) != 124 or len(index(rows)) != 124:
        raise ValueError("POPULATION_CHANGED")
    for r in rows:
        if not all(np.isfinite(r[f]) for f in NUMERIC + ["net", "opp", "C_buy", "C_sell", "pnl"]):
            raise ValueError("NONFINITE_INPUT")
        if r["C_buy"] <= 0 or abs(r["C_sell"] - r["C_buy"] - r["pnl"]) > 0.00021:
            raise ValueError("LEDGER_IDENTITY_FAIL")
        if abs(r["opp"] - r["pnl"] / 50000) > 1e-12 or r["label"] != int(r["net"] > 0):
            raise ValueError("TARGET_OR_OPPORTUNITY_MISMATCH")
        datetime.strptime(str(r["le"]), "%Y%m%d%H%M%S")
        e = r["E"]
        segment = "train" if e <= 20260305 else "val" if 20260309 <= e <= 20260609 else "test" if e >= 20260610 else "gap"
        if r["segment"] != segment or segment == "gap":
            raise ValueError("SEGMENT_MISMATCH")
        limit = {"train": 20260309000000, "val": 20260610000000}.get(segment)
        if limit and r["le"] >= limit:
            raise ValueError("CROSS_SEGMENT_LABEL")
    t = [r for r in rows if r["segment"] == "train"]
    if sum(r["label"] for r in t) < 30 or sum(not r["label"] for r in t) < 30 or len({r["E"] for r in t}) < 40:
        raise ValueError("TRAIN_SAMPLE_GATE_FAILED")


def verify_ledgers(rows, ledgers, admissions):
    by_key = index(ledgers)
    admitted = {(r["code"], r["E"]): r for r in admissions
                if r["track"] == "H3" and r["admitted"]}
    if set(by_key) != set(index(rows)):
        raise ValueError("LEDGER_POPULATION_MISMATCH")
    for row in rows:
        k = row["code"], row["E"]
        ledger, a = by_key[k], admitted[k]
        entry, t = ledger["entry"], ledger["H3"]
        if (t["remaining_shares"] != 0 or t["missing_dependencies"] or not t["fully_exited"]
                or t["status"] != "SIM_COMPLETE" or a["net"] != row["net"]
                or a["le"] != row["original_label_end_ts"] or a["assert_fails"]
                or a["verdict"] not in ("PASS_NO_EVENT", "PASS_EMPTY_CHAIN")):
            raise ValueError("EXECUTION_OR_COMPANY_ACTION_NOT_ADMITTED")
        buy = entry["sim_fill_price"] * entry["shares"] * 1.0013
        sell = sum(s["price"] * s["shares"] * .9982 for s in t["sells"])
        if (sum(s["shares"] for s in t["sells"]) != entry["shares"]
                or abs(buy - row["C_buy"]) > .00011 or abs(sell - row["C_sell"]) > .00011
                or abs(sell - buy - row["pnl"]) > .00011
                or row["net"] != t["net"] or row["le"] != ledger["label_end_ts"]):
            raise ValueError("FULL_CASH_LEDGER_MISMATCH")


def check(config):
    if config["fixed"] != FIXED or config["scope"] != "HISTORICAL_EXPLORATORY_124":
        raise ValueError("FIXED_CONFIGURATION_CHANGED")
    for path, value in config["inputs"].items():
        if sha(path) != value:
            raise ValueError("CONFIG_INPUT_CHANGED:" + path)
    required = {str(Path(__file__)), str(CONTRACT), str(OUT / "fields.json"),
                str(OUT / "manifest.json"), str(OUT / "readiness.json"),
                str(OUT / "ledger_evidence.json"), str(BASE / "historical_trigger_extension.py"),
                str(BASE / "forward_training_cli.py"), str(CHAIN),
                str(BASE / "contracts/dual_path_execution_contract_v0_1_1_final.md"),
                str(BASE / "contracts/dual_path_execution_v0_1_1_manifest.json")}
    if not required <= set(config["inputs"]):
        raise ValueError("INCOMPLETE_CONFIG_HASH_BINDING")
    if (Path(config["dataset"]).resolve() != (OUT / "fields.json").resolve()
            or Path(config["manifest"]).resolve() != (OUT / "manifest.json").resolve()
            or Path(config["readiness"]).resolve() != (OUT / "readiness.json").resolve()
            or Path(config["contract"]).resolve() != CONTRACT.resolve()):
        raise ValueError("FROZEN_PATH_CHANGED")
    for path, value in read(config["manifest"])["inputs"].items():
        if sha(path) != value:
            raise ValueError("FROZEN_INPUT_CHANGED:" + path)
    report = read(config["readiness"])
    if not report["data_ready"] or report["failures"]:
        raise ValueError("DATA_NOT_READY")
    rows = read(config["dataset"])
    validate_rows(rows)
    admission = read(RUN / "r1_label_admission_v0_1_2.json")
    if sha(CHAIN) != admission["frozen_input_hashes"]["adj_cache.json(feb2026)"]:
        raise ValueError("ADJUSTMENT_CHAIN_CHANGED")
    verify_ledgers(rows, read(OUT / "ledger_evidence.json"), admission["track_decisions"])
    authorized = (config["status"] == "SIGNED" and bool(config.get("signed_by"))
                  and config["fit_authorized"] is True and bool(config.get("fit_authorized_by"))
                  and "Status: FROZEN_RULES_PENDING_USER_SIGNATURE" in CONTRACT.read_text(encoding="utf-8"))
    return rows, {"data_ready": True, "training_ready": authorized, "stats": report["stats"],
                  "blocked": None if authorized else "SIGNED_CONTRACT_AND_SEPARATE_FIT_AUTHORIZATION_REQUIRED",
                  "live_source_or_T0_required": False, "new_blind_validation": False}


def clustered(values, dates):
    values, dates = np.asarray(values, dtype=float), np.asarray(dates)
    if not len(values):
        return {"mean": None, "ci95": None, "events": 0, "date_clusters": 0}
    groups = [values[dates == d] for d in sorted(set(dates))]
    sums, counts = np.array([g.sum() for g in groups]), np.array([len(g) for g in groups])
    rng = np.random.default_rng(FIXED["bootstrap_seed"])
    draws = rng.integers(0, len(groups), (FIXED["bootstrap_draws"], len(groups)))
    boot = sums[draws].sum(axis=1) / counts[draws].sum(axis=1)
    return {"mean": float(values.mean()), "ci95": list(map(float, np.quantile(boot, [.025, .975]))),
            "events": len(values), "date_clusters": len(groups)}


def fit(rows):
    # Lazy dependencies: neither prepare nor check can fit accidentally.
    import pandas as pd
    import joblib
    from sklearn.compose import ColumnTransformer
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
    frame = pd.DataFrame(rows)
    train = frame[frame.segment == "train"]
    artifacts, probabilities = {}, {}
    for name, numeric in (("original", NUMERIC[:2]), ("expanded", NUMERIC)):
        model = Pipeline([("prep", ColumnTransformer([
            ("num", StandardScaler(), numeric),
            ("cat", OneHotEncoder(handle_unknown="ignore"), ["setup", "branch"])])),
            ("lr", LogisticRegression(C=.1, penalty="l2", solver="lbfgs", class_weight=None,
                                      max_iter=1000, tol=1e-4, random_state=20260928))])
        with warnings.catch_warnings():
            warnings.simplefilter("error", ConvergenceWarning)
            model.fit(train, train.label)
        if model["lr"].n_iter_.max() >= 1000:
            raise ValueError("MODEL_NOT_CONVERGED")
        probabilities[name] = model.predict_proba(frame)[:, 1]
        if not np.isfinite(probabilities[name]).all():
            raise ValueError("NONFINITE_PREDICTION")
        buf = io.BytesIO(); joblib.dump(model, buf); artifacts[name] = buf.getvalue()
    evaluation = {}
    for s in ("train", "val", "test"):
        mask = frame.segment.to_numpy() == s
        sub = frame[mask]
        dates, opp, net = sub.E.to_numpy(), sub.opp.to_numpy(), sub.net.to_numpy()
        actions = {n: p[mask] >= .5 for n, p in probabilities.items()}
        actions.update(rule=sub.contraction.to_numpy() <= 1, all_pass=np.ones(len(sub), dtype=bool))
        values = {n: np.where(a, opp, 0.0) for n, a in actions.items()}
        arms = {n: {"opportunity": clustered(values[n], dates),
                    "own_net": clustered(net[a], dates[a]),
                    "filled_budget_return": clustered(opp[a], dates[a]),
                    "retained": int(a.sum()), "rejection_rate": float((~a).mean()),
                    "left_tail_count": int((net[a] <= -.05).sum())} for n, a in actions.items()}
        deltas = {n: clustered(values["expanded"] - values[n], dates)
                  for n in ("original", "rule", "all_pass")}
        for n in probabilities:
            arms[n]["brier"] = float(np.mean((probabilities[n][mask] - sub.label.to_numpy()) ** 2))
        evaluation[s] = {"arms": arms, "expanded_minus": deltas,
                          "constant_brier": float(np.mean((37/77 - sub.label.to_numpy()) ** 2))}
    main = evaluation["val"]["expanded_minus"]["original"]
    result = {"evaluation": evaluation, "primary_exit": "SUPPORTS_POSITIVE" if main["ci95"][0] > 0
              else "SUPPORTS_NEGATIVE" if main["ci95"][1] < 0 else "INSUFFICIENT_EVIDENCE",
              "automatic_promotion": False, "historical_exploratory": True,
              "predictions": [{"code": r["code"], "E": r["E"],
                  **{n: float(p[i]) for n, p in probabilities.items()}} for i, r in enumerate(rows)]}
    return result, artifacts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "authorize", "check", "train"))
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--signed-by")
    parser.add_argument("--fit-authorized-by")
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(); print(str(CONFIG)); return 0
    config = read(args.config)
    if args.command == "authorize":
        if not args.signed_by or not args.fit_authorized_by:
            raise ValueError("BOTH_USER_AUTHORIZATION_FLAGS_REQUIRED")
        print(str(authorize(config, args.signed_by, args.fit_authorized_by))); return 0
    rows, report = check(config)
    print(json.dumps(report, ensure_ascii=False))
    if args.command == "check" or not report["training_ready"]:
        return 0 if report["training_ready"] else 2
    if args.output is None or args.output.exists():
        raise ValueError("NEW_EXPLICIT_OUTPUT_DIRECTORY_REQUIRED")
    if not args.output.resolve().is_relative_to(RUN.resolve()):
        raise ValueError("OUTPUT_OUTSIDE_RUN")
    result, models = fit(rows)
    # Verify all evidence again before publishing fitted outputs.
    check(config)
    result["config_sha256"] = sha(args.config)
    result["input_sha256"] = config["inputs"]
    payloads = {"results.json": json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False).encode("utf-8"),
                **{n + ".joblib": b for n, b in models.items()}}
    for name, body in payloads.items():
        save(args.output / name, body)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
