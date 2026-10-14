"""Publish accepted historical reconstructions; offline ledger reconciliation, no fit."""
from collections import Counter
from datetime import datetime, timedelta
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from forward_training_cli import save
import dual_path_execution_v0_1_1 as executor

BASE = Path(__file__).resolve().parent
RUN = BASE / "gcp_round_b_small_sample_v1"
OUT = RUN / "historical_trigger_extension_v0_3"
NUMERIC = ["contraction", "vol_cv10", "pen_atr_bar", "d_atr", "width_atr"]


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def key(row):
    return row["code"], row["E"]


def index(rows):
    result = {key(row): row for row in rows}
    if len(result) != len(rows):
        raise ValueError("DUPLICATE_EVENT")
    return result


def completed(ts):
    return int((datetime.strptime(str(ts), "%Y%m%d%H%M%S")
                + timedelta(minutes=1)).strftime("%Y%m%d%H%M%S"))


def merge_fields(original, old, recovered):
    if key(original) != key(old) or original["branch"] != old["branch"]:
        raise ValueError("EVENT_BINDING_MISMATCH")
    row = {**old, **original}
    if row["branch"] == "BASE_HOLD":
        if (not recovered or key(recovered) != key(row)
                or recovered["status"] != "RECONSTRUCTION_CANDIDATE"
                or recovered["reason"] is not None):
            raise ValueError("RECONSTRUCTION_NOT_ACCEPTABLE")
        for name in ("signal_bar_ts", "decision_available_ts", "atr_ref", "signal_close"):
            if row[name] != recovered[name]:
                raise ValueError("RECONSTRUCTION_BINDING_MISMATCH:" + name)
        row.update(zone_lower=recovered["box_lower"], zone_upper=recovered["box_upper"],
                   region_provenance="RECONSTRUCTED_ORIGINAL_SWING_INPUTS",
                   historical_boundary_capture=False, old_region_comparison="NO_BASELINE")
        for name in ("pen_atr_bar", "d_atr", "width_atr"):
            row[name] = recovered[name]
        row.pop("missing_reason", None)
    else:
        row["region_provenance"] = "FROZEN_DIP_V0_2"
    if row["decision_available_ts"] != completed(row["signal_bar_ts"]):
        raise ValueError("DECISION_TIME_MISMATCH")
    if not all(np.isfinite(row[name]) for name in NUMERIC):
        raise ValueError("NONFINITE_FEATURE")
    if row["atr_ref"] <= 0 or row["zone_lower"] > row["zone_upper"]:
        raise ValueError("INVALID_REGION")
    return row


def verify_track(row, frozen, result):
    """Compare full ledger with frozen labels, allowing declared output rounding."""
    t, ent = result.get("H3", {}), result.get("entry", {})
    if (t.get("status") != "SIM_COMPLETE" or not t.get("fully_exited")
            or t.get("remaining_shares") != 0 or t.get("missing_dependencies")):
        raise ValueError("H3_NOT_CLOSED")
    if (t["net"] != row["net"] or t["label_end_ts"] != row["le"]
            or ent["sim_fill_ts"] != frozen["fill_ts"]
            or ent["sim_fill_price"] != frozen["fill_px"]
            or ent["shares"] != frozen["shares"]):
        raise ValueError("FROZEN_EXECUTION_CHANGED")
    buy, sell = t["C_buy"], t["C_sell"]
    if not np.isfinite([buy, sell, t["pnl"], t["net"]]).all() or buy <= 0:
        raise ValueError("INVALID_LEDGER")
    if abs(sell / buy - 1 - t["net"]) > 1e-6 or abs(sell - buy - t["pnl"]) > 0.00021:
        raise ValueError("LEDGER_IDENTITY_FAIL")
    if row["label"] != int(row["net"] > 0):
        raise ValueError("LABEL_SIGN_MISMATCH")
    end = t["label_end_ts"]
    correct_end = completed(t["last_fill_ts"])
    if end != t["last_fill_ts"] + 100 or correct_end <= ent["sim_fill_ts"]:
        raise ValueError("INVALID_LABEL_ENDPOINT")
    boundary = {"train": 20260309000000, "val": 20260610000000}.get(row["segment"])
    if boundary and correct_end >= boundary:
        raise ValueError("CROSS_SEGMENT_LABEL")
    if result.get("limit_ref_note"):
        raise ValueError("ENTRY_PRECLOSE_FALLBACK")
    return {"code": row["code"], "E": row["E"], "entry": ent, "H3": t,
            "original_label_end_ts": end, "label_end_ts": correct_end,
            "endpoint_correction": "DATETIME_MINUTE_CARRY" if end != correct_end else None,
            "frozen_label_match": True, "company_action_admission": "REUSED_HASH_BOUND_ADMISSION"}


def write(path, value):
    def scalar(obj):
        if isinstance(obj, np.generic):
            return obj.item()
        raise TypeError(type(obj).__name__)
    save(path, json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False, default=scalar).encode("utf-8"))


def main():
    if OUT.exists():
        raise ValueError("NEW_OUTPUT_DIRECTORY_REQUIRED")
    files = [RUN / x for x in ("trigger_fields_manifest_v0_2.json", "trigger_fields_table_v0_2.json",
             "r1_fit_manifest_v0_1_1.json", "r1_fit_features_v0_1_1.json", "r1_exec_labels_v0_1_2.json",
             "r1_label_admission_v0_1_2.json", "historical_box_provenance_v0_3/report.json")]
    files += [Path(__file__), BASE / "dual_path_execution_v0_1_1.py"]
    hashes = {str(p): sha(p) for p in files}
    manifest = read(files[0])
    evidence = {x["path"]: x["sha256"] for x in manifest["inputs"]}
    recovery = read(RUN / "historical_box_provenance_v0_3/report.json")
    if recovery["frozen_manifest_sha256"] != sha(files[0]):
        raise ValueError("RECOVERY_MANIFEST_MISMATCH")
    evidence.update(recovery["additional_input_hashes"])
    for name, value in read(RUN / "r1_fit_manifest_v0_1_1.json")["sha256"].items():
        p = RUN / name if (RUN / name).exists() else BASE / name
        evidence[str(p)] = value
    def verify_inputs():
        for path, value in evidence.items():
            if sha(path) != value:
                raise ValueError("INPUT_HASH_MISMATCH:" + path)
    verify_inputs()
    evidence.update(hashes)
    original = read(RUN / "r1_fit_features_v0_1_1.json")["H3_primary"]
    old = index(read(RUN / "trigger_fields_table_v0_2.json"))
    recovered = index(recovery["records"])
    if len(original) != 124 or set(index(original)) != set(old) or len(recovered) != 80:
        raise ValueError("FROZEN_POPULATION_MISMATCH")
    rows = [merge_fields(r, old[key(r)], recovered.get(key(r))) for r in original]
    admission = {(r["code"], r["E"]): r for r in
                 read(RUN / "r1_label_admission_v0_1_2.json")["track_decisions"]
                 if r["track"] == "H3" and r["admitted"]}
    labels = index(read(RUN / "r1_exec_labels_v0_1_2.json")["records"])
    codes = {r["code"] for r in rows}
    parts, cal = [], set()
    for x in manifest["inputs"]:
        if Path(x["path"]).parent.name != "data_daily_parts":
            continue
        part = pd.read_parquet(x["path"])
        part["code"] = part.code.astype(str).str.zfill(6)
        part["date"] = part.date.astype(int)
        cal.update(part.date.unique())
        parts.append(part[part.code.isin(codes)])
    panel = pd.concat(parts, ignore_index=True)
    daily = {c: f.sort_values("date") for c, f in panel.groupby("code")}
    calendar = sorted(cal)
    minutes, ledgers, failures = {}, [], []
    for row in rows:
        try:
            a = admission[key(row)]
            if a["net"] != row["net"] or a["le"] != row["le"] or a["assert_fails"]:
                raise ValueError("ADMISSION_BINDING_MISMATCH")
            code = row["code"]
            if code not in minutes:
                minutes[code], provenance = executor.load_minute_code(code)
                evidence.update({x["path"]: x["sha256"] for x in provenance})
            frame = daily[code]
            if frame.date.duplicated().any():
                raise ValueError("DAILY_DUPLICATE")
            result = executor.execute_event(code, row["E"], labels[key(row)]["board"], None,
                       row["setup"], row["signal_bar_ts"], minutes[code], frame, calendar)
            ledger = verify_track(row, labels[key(row)], result)
            # Bounded scale proof across each actual H3 path, not the entire market.
            dates = [d for d in calendar if row["E"] <= d <= row["le"] // 1000000]
            scale_checks = []
            for d in dates:
                bars = minutes[code].get(d)
                dr = frame[frame.date == d]
                if bars is None or len(dr) != 1:
                    raise ValueError("SCALE_DEPENDENCY_MISSING")
                close = bars[bars.date.astype("int64") % 1000000 == 150000]
                if len(close) != 1:
                    scale_checks.append({"date": d, "status": "NOT_TESTED", "reason": "NO_1500_BAR"})
                    continue
                price = float(dr.iloc[0].close)
                if abs(float(close.iloc[0].close) - price) > max(0.01, 0.005 * price):
                    raise ValueError("PATH_CLOSING_SCALE_CONFLICT")
                scale_checks.append({"date": d, "status": "PASS"})
            ledger["scale_check_days"] = dates
            ledger["scale_checks"] = scale_checks
            row.update(C_buy=result["H3"]["C_buy"], C_sell=result["H3"]["C_sell"],
                       pnl=result["H3"]["pnl"], opp=result["H3"]["pnl"] / 50000,
                       original_label_end_ts=row["le"], le=ledger["label_end_ts"],
                       executor_evidence_index=len(ledgers))
            ledgers.append(ledger)
        except (ValueError, KeyError) as exc:
            failures.append({"code": row["code"], "E": row["E"], "reason": str(exc)})
    verify_inputs()
    stats = {}
    for s in ("train", "val", "test"):
        sub = [r for r in rows if r["segment"] == s]
        stats[s] = {"events": len(sub), "positive": sum(r["label"] for r in sub),
                    "nonpositive": sum(not r["label"] for r in sub),
                    "E_dates": len({r["E"] for r in sub})}
    report = {"data_ready": not failures, "fit_authorized": False, "fitted": False,
              "network_requests": 0, "population": 124, "stats": stats, "failures": failures,
              "region_counts": dict(Counter(r["region_provenance"] for r in rows)),
              "endpoint_corrections": [dict(code=l["code"], E=l["E"], old=l["original_label_end_ts"],
                    corrected=l["label_end_ts"]) for l in ledgers if l["endpoint_correction"]],
              "scale_checks": dict(Counter(x["status"] for l in ledgers for x in l["scale_checks"])),
              "scope": "fixed historical admitted subset; no new blind validation",
              "coverage_limitation": "77/226 training execution entrants (34.1%)",
              "scale_scope": "actual H3 path daily closing prices vs 15:00 minute; not all intraday prices"}
    write(OUT / "readiness.json", report)
    if failures:
        print(json.dumps(report, ensure_ascii=False)); return
    write(OUT / "fields.json", rows)
    write(OUT / "ledger_evidence.json", ledgers)
    evidence.update({str(OUT / name): sha(OUT / name)
                     for name in ("fields.json", "ledger_evidence.json", "readiness.json")})
    write(OUT / "manifest.json", {"inputs": evidence, "population": 124, "builder_sha256": sha(__file__)})
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
