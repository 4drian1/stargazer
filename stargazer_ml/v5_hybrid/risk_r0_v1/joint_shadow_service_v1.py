"""Read-only joint shadow scoring service.

CATEGORY SPLIT (this is a historical-query wrapper + a compute entry, NOT a
forward-looking production scoring service):

  * historical_replay_lookup : read ALREADY-FROZEN tables. Safe, deterministic,
        needs no new data.
        - post_market_common_candidate(mode="lookup") -> reads rolling_lr_v2_predictions
          + full_master_table_v2 (P-close knowable).
        - intraday_trigger_status(mode="lookup")      -> reads full_intraday_replay_v2,
          filtered AS OF a query cutoff (never returns a future final state).

  * compute (new trading day) : runs the ACTUAL reconstruction pipeline and needs the
        day's raw data to exist.
        - post_market_common_candidate(mode="compute") -> daily prefix + frozen pool/golden kernels + PKL.
        - intraday_trigger_status(mode="compute")      -> prefix OHLC + existing trigger_replay_v3 kernel.
        - same_tick_risk(mode="compute")               -> legacy frozen-context reconstruction.
        - same_tick_risk(mode="prefix", snapshot=...)   -> raw new-day prefix, no historical table joins.

Design constraints (engineering acceptance, G2/G4/G5/G6):
  * No training, no network, no model writes, no notifications.
  * No side effects: never mutates production artifacts or the EXE/config.
  * Scores are read-only; isolation/quality_state is reported, never imputed to low-risk.
  * Single ENABLE switch: disabled -> every interface returns quality_state='DISABLED'.
  * Runtime fingerprint verification: if any weighted/preprocessing/key-dependency file
    is tampered (hash != frozen reference), the interface REJECTS (quality_state=
    'RUNTIME_FINGERPRINT_MISMATCH') instead of scoring -- asset verification is NOT
    merely build-time; it is re-checked at call time.
  * Strict 18-field unified schema (see joint_shadow_contract_v1.json); golden_plan is
    the 18th fixed field (always present; None when N/A). Any deviation raises.
  * ShadowEventLog actually persists append-only JSONL and de-duplicates by event key
    across restart; write failures are caught (never crash the scorer).
"""
import hashlib
import json
import os
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

import rebuild_features_v1 as v1  # frozen rebuild orchestrator (read-only)
import rebuild_features_v2 as v2  # decision-cutoff policy + trigger helpers
import joint_shadow_compute_v2 as compute

HERE = Path(__file__).resolve().parent
RISK_R0 = HERE


def project_root():
    """仓库根：环境变量 STARGAZER_PROJECT_ROOT 优先，否则按本文件位置推导
    （risk_r0_v1 → v5_hybrid → stargazer_ml → 仓库根）。支持根目录不在 D:\\stargazer。"""
    return Path(os.environ.get("STARGAZER_PROJECT_ROOT",
                               Path(__file__).resolve().parents[3]))


def _fp_path(key):
    """指纹清单键 → 实际文件路径。相对键按 project_root() 解析（可移植），
    绝对键保留兼容旧清单。"""
    p = Path(key)
    return p if p.is_absolute() else project_root() / p

# ----- kill switch / enable flag (env-overridable, default OFF-safe for shadow) -----
ENABLED = os.environ.get("JOINT_SHADOW_ENABLED", "0") == "1"

# ----- unified schema (EXACTLY 18 fields; golden_plan is the 18th, always present) -----
SCHEMA_FIELDS = [
    "code", "P", "E", "feature_cutoff", "available_at", "scored_at",
    "selection_model_version", "selection_score",
    "trigger_time", "trigger_status", "trigger_basis",
    "risk_model_version", "risk_horizon", "risk_amplitude",
    "quality_state", "missing_reasons", "evidence_refs", "golden_plan",
]
assert len(SCHEMA_FIELDS) == 18, "unified schema must be exactly 18 fields"

RISK_HORIZON_MIN = 120
SELECTION_MODEL_VERSION = "green_common_profile_round_a_v0_1_impl_fix_FINAL_B2"
PKL_PATH = RISK_R0.parent / "green_common_profile" / "round_a_v0_1_impl_fix" / "fold_model_FINAL_B2.pkl"

# ----- frozen expected fingerprints（项目相对路径键；由 release_audit_v1/regen_fingerprints.py 生成） -----
_FP_FILE = HERE / "joint_shadow_fingerprints_rel_v1.json"
try:
    EXPECTED_FP = json.loads(_FP_FILE.read_text(encoding="utf-8")) if _FP_FILE.exists() else None
except Exception:
    EXPECTED_FP = None


def _sha256(path):
    p = Path(path)
    if not p.exists():
        return None
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_fingerprints(expected=None):
    """Return (mismatches, ok). Re-hashes the live files and compares to the frozen
    reference. Empty/unreadable reference is a hard refusal."""
    if not expected:
        return ["EMPTY_FINGERPRINT_REFERENCE"], False
    if not isinstance(expected, dict):
        return ["INVALID_FINGERPRINT_REFERENCE"], False
    mism = []
    for path, exp in expected.items():
        try:
            live = _sha256(_fp_path(path))
        except OSError:
            live = None
        if live is None:
            mism.append(f"{path}:MISSING")
        elif live != exp:
            mism.append(path)
    return mism, (len(mism) == 0)


def _norm_ts(value, day):
    """Validate either HHMMSS or a same-day full timestamp; never compare units."""
    if isinstance(value, bool) or not np.isfinite(float(value)) or int(value) != float(value):
        raise ValueError("INVALID_TIMESTAMP")
    value, day = int(value), int(day)
    full = day * 1000000 + value if 0 <= value < 1000000 else value
    if len(str(full)) != 14 or full // 1000000 != day:
        raise ValueError("TIMESTAMP_DAY_MISMATCH")
    datetime.strptime(str(full), "%Y%m%d%H%M%S")
    return full


def _empty_record(code=None, P=None, E=None):
    return {k: None for k in SCHEMA_FIELDS} | {
        "code": code, "P": P, "E": E, "risk_horizon": RISK_HORIZON_MIN, "golden_plan": None}


def _golden_from_row(row):
    golden = {}
    for c in ("setup", "pool", "F618", "atr_ref", "plan_dist_atr", "golden_dist_618_atr", "golden_wave_pos"):
        if c not in row.index:
            golden[c] = None
            continue
        v = row[c]
        if pd.isna(v):
            golden[c] = None
        else:
            golden[c] = (float(v) if isinstance(v, (int, float, np.floating, np.integer)) else str(v))
    return golden


class JointShadowService:
    def __init__(self, enabled=None, fp_expected=None, daily_loader=None,
                 minute_loader=None, calendar=None):
        self.enabled = ENABLED if enabled is None else bool(enabled)
        self.fp_expected = EXPECTED_FP if fp_expected is None else fp_expected
        self._boot = None      # (codes, members, native, days)
        self._spec = None      # (mapping, fields232, fields266, extra266) from load_spec()
        self._boost = None     # (booster232, booster266)
        self._pb2 = None
        self._master = None
        self._replay_ref = None
        self._diff = None
        self._pkl = None       # loaded fold_model pkl for compute-mode selection
        self._daily_loader = daily_loader or compute.read_daily_prefix
        self._minute_loader = minute_loader or compute.read_minute_prefix
        self._calendar = calendar

    def _days(self):
        return self._calendar if self._calendar is not None else compute.calendar_dates()

    # ---------------- read-only frozen artifact loading ----------------
    def _load_static(self):
        if self._pb2 is not None:
            return
        gp = RISK_R0.parent / "green_common_profile" / "round_a_v0_1_impl_fix"
        self._pb2 = pd.read_parquet(gp / "rolling_lr_v2_predictions.parquet")
        self._pb2["code"] = self._pb2.code.astype(str).str.zfill(6)
        self._master = pd.read_parquet(gp / "full_master_table_v2.parquet")
        self._master["code"] = self._master.code.astype(str).str.zfill(6)
        self._replay_ref = pd.read_parquet(gp / "full_intraday_replay_v2.parquet")
        self._replay_ref["code"] = self._replay_ref.code.astype(str).str.zfill(6)
        self._diff = pd.read_parquet(v1.OVERLAY)
        self._spec = v1.load_spec()

    def _load_pkl(self):
        if self._pkl is not None:
            return self._pkl
        import pickle
        with open(PKL_PATH, "rb") as f:
            self._pkl = pickle.load(f)
        return self._pkl

    def bootstrap(self, days):
        """Replay frozen minutes for `days` once and cache. days: iterable[int] (YYYYMMDD)."""
        self._load_static()
        days = sorted(set(int(d) for d in days))
        mapping, fields232, fields266, extra266 = self._spec
        import pyarrow.parquet as pq
        import pyarrow.parquet as _pq
        sm_clock = pq.read_table(v1.SHARED_MARKET, columns=[v1.TIME, "cutoff_index"]).to_pandas()
        v1.TS_TO_CI = dict(zip(sm_clock[v1.TIME].astype(np.int64), sm_clock.cutoff_index))
        SM_ALL = _pq.read_table(v1.SHARED_MARKET).column_names
        mapping_sources = set(mapping.values())
        base_sources = {s for s in mapping_sources if not s.startswith("ar_")}
        derived = {"time_node_nearest_abs_distance", "session_is_pm",
                    "context_sector_minus_sse_return_raw", "context_stock_minus_sector_return_raw",
                    "context_stock_minus_leader_return_raw"}
        mom_all = set(pd.read_parquet(v1.MOMENTUM).columns)
        v1.SM_ALL = SM_ALL
        v1.SM_COLS = [c for c in SM_ALL if c in base_sources]
        v1.MOM_COLS = sorted((base_sources & mom_all) - set(v1.SM_COLS) - derived)
        v1.JP_COLS = sorted({s for s in base_sources if s.startswith("stock_")})
        v1.CTX_COLS = sorted({s for s in base_sources if s.startswith("context_")} - derived)
        v1.NH_OWN_COLS = [c for c in _pq.read_table(v1.REPAIR / "own.parquet").column_names
                           if c.startswith("nh_own")]
        codes, members, native = v1.replay(days)
        booster232 = v1.load_booster(v1.MODEL_232)
        booster266 = v1.load_booster(v1.MODEL_266)
        self._boot = (codes, members, native, days)
        self._boost = (booster232, booster266)
        self._spec = (mapping, fields232, fields266, extra266)

    # ---------------- unified record assembly (strict 18-field schema) ----------------
    def _record(self, code, P, E, **kw):
        bad = set(kw) - set(SCHEMA_FIELDS)
        if bad:
            raise ValueError(f"non-schema fields in record: {bad}")
        r = _empty_record(code, P, E)
        r.update(kw)
        return r

    # ---------------- INTERFACE A: post-market common candidate ----------------
    def post_market_common_candidate(self, code, P, E, mode="lookup"):
        """Selection score (pB2) + golden plan.
        mode='lookup' -> frozen tables; mode='compute' -> daily prefix + frozen kernels.
        Compute never looks up the requested date in the historical master table."""
        code = str(code).zfill(6); P = int(P); E = int(E)
        if not self.enabled:
            return self._record(code, P, E, quality_state="DISABLED",
                                missing_reasons=["service_not_enabled"],
                                evidence_refs=["joint_shadow:UNENABLED"])
        mism, ok = verify_fingerprints(self.fp_expected)
        if not ok:
            return self._record(code, P, E, quality_state="RUNTIME_FINGERPRINT_MISMATCH",
                                missing_reasons=["fingerprint_mismatch:" + m for m in mism],
                                evidence_refs=["joint_shadow:fingerprint_check"])
        try:
            if mode == "compute":
                return self._post_market_compute(code, P, E)
            if mode != "lookup":
                raise ValueError("INVALID_MODE")
            self._load_static()
            rec = self._record(code, P, E, feature_cutoff=int(P) * 1000000 + 150000,
                               available_at="P_close", scored_at=int(time.time()),
                               evidence_refs=["lookup:rolling_lr_v2_predictions.parquet"])
            pb = self._pb2[(self._pb2.fold == "FINAL") & (self._pb2.code == code) & (self._pb2.P == P)]
            if len(pb):
                rec["selection_model_version"] = SELECTION_MODEL_VERSION
                rec["selection_score"] = float(pb.iloc[0]["pB2"])
            else:
                rec["missing_reasons"] = ["NOT_IN_SCORING_POPULATION(CORP_ACTION_ISOLATED)"]
            m = self._master[(self._master.code == code) & (self._master.P == P)]
            if len(m):
                rec["golden_plan"] = _golden_from_row(m.iloc[0])
                rec["evidence_refs"].append("lookup:full_master_table_v2.parquet")
            else:
                rec.setdefault("missing_reasons", []).append("golden_plan_missing")
            if rec["quality_state"] is None:
                rec["quality_state"] = "OK" if rec["selection_score"] is not None else "PARTIAL"
            return rec
        except compute.ComputeUnavailable as e:
            return self._record(code, P, E, quality_state="NOT_COMPUTABLE",
                                missing_reasons=[str(e)], evidence_refs=["compute:daily_prefix"])
        except Exception as e:  # never crash the shell; report isolation
            return self._record(code, P, E, quality_state="RUNTIME_ERROR",
                                missing_reasons=[f"{type(e).__name__}: {e}"],
                                evidence_refs=["joint_shadow:post_market"])

    def _post_market_compute(self, code, P, E):
        pkl = self._load_pkl()
        days = self._days()
        if P not in days or days.index(P) + 1 >= len(days) or days[days.index(P) + 1] != E:
            raise compute.ComputeUnavailable("P_E_CALENDAR_MISMATCH")
        row, prefix = compute.selection_row(self._daily_loader(code, P), code, P, days)
        feats = pkl["features"]
        med = pkl["median"]
        values = row.reindex(feats)
        if not np.isfinite(values.dropna().to_numpy(dtype=float)).all():
            raise compute.ComputeUnavailable("SELECTION_FEATURE_INVALID")
        x = pkl["scaler"].transform(values.fillna(med).fillna(0.0).to_frame().T.astype(float))
        pB2 = float(np.clip(pkl["clf"].predict_proba(x)[:, 1][0], 1e-6, 1 - 1e-6))
        rec = self._record(code, P, E, feature_cutoff=int(P) * 1000000 + 150000,
                           available_at=None, scored_at=int(time.time()),
                           selection_model_version=SELECTION_MODEL_VERSION,
                           selection_score=pB2, golden_plan=_golden_from_row(row),
                           quality_state="OK", missing_reasons=[
                               "STRUCTURAL_POOL_IMPUTATION:" + c for c in values.index[values.isna()]],
                           evidence_refs=["compute:fold_model_FINAL_B2.pkl", "compute:process_code",
                                          "compute:full_master_table_v2.build_anchor_features",
                                          "daily_prefix_sha256:" + hashlib.sha256(
                                              pd.util.hash_pandas_object(prefix, index=False).values.tobytes()).hexdigest(),
                                          "RESEARCH_SOURCE_SCALE_AND_ACTUAL_ARRIVAL_NOT_CERTIFIED"])
        return rec

    # ---------------- INTERFACE B: intraday trigger status ----------------
    def intraday_trigger_status(self, code, E, as_of_ts=None, mode="lookup"):
        """Individual trigger state AS OF `as_of_ts` (a query cutoff, HHMMSS or ts int).
        If as_of_ts is None, returns the full-day final state (NOT a forward-looking query).
        mode='compute' runs trigger_replay_v3 on day E (requires raw minute data)."""
        code = str(code).zfill(6); E = int(E)
        if not self.enabled:
            return self._record(code, None, E, quality_state="DISABLED",
                                missing_reasons=["service_not_enabled"],
                                evidence_refs=["joint_shadow:UNENABLED"])
        mism, ok = verify_fingerprints(self.fp_expected)
        if not ok:
            return self._record(code, None, E, quality_state="RUNTIME_FINGERPRINT_MISMATCH",
                                missing_reasons=["fingerprint_mismatch:" + m for m in mism],
                                evidence_refs=["joint_shadow:fingerprint_check"])
        try:
            if mode == "compute":
                return self._trigger_compute(code, E, as_of_ts)
            if mode != "lookup":
                raise ValueError("INVALID_MODE")
            self._load_static()
            rec = self._record(code, None, E, available_at="E_completed", scored_at=int(time.time()),
                               evidence_refs=["lookup:full_intraday_replay_v2.parquet"])
            r = self._replay_ref[(self._replay_ref.code == code) & (self._replay_ref.E == E)]
            if len(r) == 0:
                rec["quality_state"] = "TRIGGER_RECORD_MISSING"
                rec["missing_reasons"] = ["no_frozen_intraday_replay_row"]
                return rec
            row = r.iloc[0]
            final_status = str(row.get("status", "")) or None
            trigger_time = (float(row["trigger_t"]) if pd.notna(row.get("trigger_t")) else None)
            basis = row.get("replay_reason", row.get("reason"))
            basis = str(basis) if pd.notna(basis) else None
            status, ttime = self._as_of(final_status, trigger_time, as_of_ts, E)
            rec["trigger_status"] = status
            rec["trigger_time"] = ttime
            rec["trigger_basis"] = basis if status == "TRIGGERED" else None
            rec["quality_state"] = "OK" if status == "TRIGGERED" else "HISTORICAL_VIEW"
            rec["evidence_refs"].append("HISTORICAL_LOOKUP_NOT_FORWARD_COMPUTATION")
            return rec
        except Exception as e:
            return self._record(code, None, E, quality_state="RUNTIME_ERROR",
                                missing_reasons=[f"{type(e).__name__}: {e}"],
                                evidence_refs=["joint_shadow:trigger"])

    @staticmethod
    def _as_of(final_status, trigger_time, as_of_ts, E=None):
        """Compute the trigger state AS OF as_of_ts. A triggered row carries trigger_time
        (the ts it actually fired). We must never return a future final state."""
        if as_of_ts is None:
            if trigger_time is not None and pd.notna(trigger_time):
                if E is None:
                    E = int(trigger_time) // 1000000
                trigger_time = _norm_ts(trigger_time, E)
            return final_status, trigger_time  # explicit full-day view
        if E is None:
            stamp = as_of_ts if int(as_of_ts) >= 1000000 else trigger_time
            if stamp is None or int(stamp) < 1000000:
                raise ValueError("DAY_REQUIRED_FOR_HHMMSS")
            E = int(stamp) // 1000000
        cutoff = _norm_ts(as_of_ts, E)
        if trigger_time is not None and pd.notna(trigger_time):
            trigger = _norm_ts(trigger_time, E)
            if final_status == "TRIGGERED" and trigger <= cutoff:
                return "TRIGGERED", trigger
        return "PENDING_AS_OF", None           # fired later -> not yet triggered at as_of

    def _trigger_compute(self, code, E, as_of_ts):
        # No final-day lookup: only the supplied prefix reaches the existing pure kernel.
        try:
            if as_of_ts is None:
                raise compute.ComputeUnavailable("COMPUTE_REQUIRES_AS_OF")
            cutoff = _norm_ts(as_of_ts, E)
            days = self._days()
            if E not in days or days.index(E) == 0:
                raise compute.ComputeUnavailable("E_NOT_IN_CALENDAR")
            P = days[days.index(E) - 1]
            daily = compute.daily_prefix(self._daily_loader(code, P), code, P)
            plan = compute.golden_features(daily, P, days)
            frame = self._minute_loader(code, E, cutoff).copy()
            # The frozen trigger kernel consumes start-labelled one-minute bars.
            # A bar starting at the query minute is not completed yet.
            minute_boundary = cutoff - cutoff % 100
            frame = frame[(frame.date >= E * 1000000) & (frame.date < minute_boundary)]
            if 'code' in frame and not frame.code.astype(str).str.zfill(6).eq(code).all():
                raise compute.ComputeUnavailable("MINUTE_CODE_MISMATCH")
            if frame.empty:
                raise compute.ComputeUnavailable("MINUTE_PREFIX_MISSING")
            for _, duplicate in frame[frame.duplicated('date', keep=False)].groupby('date'):
                if len(duplicate[['open', 'high', 'low', 'close']].drop_duplicates()) != 1:
                    raise compute.ComputeUnavailable("MINUTE_DUP_CONFLICT")
            frame = frame.drop_duplicates('date').sort_values('date')
            prices = frame[['open', 'high', 'low', 'close']].to_numpy(dtype=float)
            if (not np.isfinite(prices).all() or (prices <= 0).any()
                    or (frame.low > frame[['open', 'close']].min(axis=1)).any()
                    or (frame.high < frame[['open', 'close']].max(axis=1)).any()):
                raise compute.ComputeUnavailable("MINUTE_PRICE_INVALID")
            if E * 1000000 + 93000 not in set(frame.date):
                raise compute.ComputeUnavailable("OPENING_BAR_MISSING")
            mod = compute.source_module(compute.TRIGGER_SOURCE)
            bars = mod.to_bars(frame.to_dict('records'))
            five, segments, _ = mod.aggregate_5min(bars, E)
            # Source timestamps follow the frozen trigger kernel's start-label convention.
            completed = [(bar, seg) for bar, seg in zip(five, segments)
                         if _norm_ts(bar['t'], E) <= cutoff]
            five, segments = [x[0] for x in completed], [x[1] for x in completed]
            if not five:
                raise compute.ComputeUnavailable("NO_COMPLETED_5MIN_BAR")
            if frame.iloc[0]['open'] < plan['F618']:
                status, ttime, reason = "NON_DOWNSIDE_AS_OF", None, "OPEN_BELOW_F"
            else:
                result = mod.replay(five, segments, plan['F618'], plan['atr_ref'])
                fired = result.get('trigger') is not None
                status = "TRIGGERED" if fired else "PENDING_AS_OF"
                ttime = _norm_ts(result['trigger_t'], E) if fired else None
                reason = result['reason']
            return self._record(code, P, E, feature_cutoff=cutoff, available_at=None,
                                scored_at=int(time.time()), trigger_status=status,
                                trigger_time=ttime, trigger_basis=reason, golden_plan=plan,
                                quality_state="OK", missing_reasons=[],
                                evidence_refs=["compute:trigger_replay_v3.prefix",
                                               "FROZEN_KERNEL_START_LABEL_CONVENTION",
                                               "ACTUAL_ARRIVAL_AND_CROSS_SOURCE_SCALE_NOT_CERTIFIED"])
        except compute.ComputeUnavailable as e:
            return self._record(code, None, E, quality_state="NOT_COMPUTABLE",
                                missing_reasons=[str(e)], evidence_refs=["compute:trigger_prefix"])
        except Exception as e:
            return self._record(code, None, E, quality_state="RUNTIME_ERROR",
                                missing_reasons=[f"{type(e).__name__}: {e}"],
                                evidence_refs=["compute:trigger_replay_v3"])

    # ---------------- INTERFACE C: same-tick risk (232/266/ATR) ----------------
    def same_tick_risk(self, candidates, days=None, mode="compute", snapshot=None):
        """candidates: list of dict(code, P, E, cutoff_ts). Heavy frozen-minute replay
        is cached per `days`. Returns unified records with risk output."""
        if not self.enabled:
            return [self._record(str(c["code"]).zfill(6), c.get("P"), c.get("E"),
                                feature_cutoff=int(c.get("cutoff_ts")) if c.get("cutoff_ts") else None,
                                quality_state="DISABLED", missing_reasons=["service_not_enabled"],
                                evidence_refs=["joint_shadow:UNENABLED"]) for c in candidates]
        if not candidates:
            return []  # empty input -> empty output, no crash
        mism, ok = verify_fingerprints(self.fp_expected)
        if not ok:
            return [self._record(str(c["code"]).zfill(6), c.get("P"), c.get("E"),
                                feature_cutoff=int(c.get("cutoff_ts")) if c.get("cutoff_ts") else None,
                                quality_state="RUNTIME_FINGERPRINT_MISMATCH",
                                missing_reasons=["fingerprint_mismatch:" + m for m in mism],
                                evidence_refs=["joint_shadow:fingerprint_check"]) for c in candidates]
        try:
            if mode == "prefix":
                return self._risk_prefix(candidates, snapshot)
            if mode != "compute":
                raise ValueError("INVALID_MODE")
            # FIX (cold start): ensure spec is loaded before any unpack, regardless of whether
            # _load_static was previously called by another interface.
            self._load_static()
            mapping, fields232, fields266, extra266 = self._spec
            if days is None:
                days = sorted({int(c["E"]) for c in candidates})  # actual E days, not month-first
            if self._boot is None or set(self._boot[3]) != set(days):
                self.bootstrap(days)
            codes, members, native, _ = self._boot
            booster232, booster266 = self._boost

            keys = pd.DataFrame([{
                "code": str(c["code"]).zfill(6), "P": int(c["P"]), "E": int(c["E"]),
                v1.TIME: int(c["cutoff_ts"])} for c in candidates])
            keys["code"] = keys.code.astype(str).str.zfill(6)
            keys[v1.TIME] = keys[v1.TIME].astype(np.int64)
            if keys.duplicated(["code", v1.TIME]).any():
                raise ValueError("DUPLICATE_CANDIDATE_KEY")

            frame = v1.build_candidate_frame(keys, days, codes, members, native, mapping, self._diff)
            cand232 = v1.resolve_logical(frame, mapping, fields232)
            frame266 = v1.build_266(frame, keys, days, codes, members, native, extra266)
            raw232 = cand232[fields232].to_numpy(dtype=float)
            raw266 = frame266[fields266].to_numpy(dtype=float)
            inf_rows = sorted(set(np.flatnonzero(np.isinf(raw232).any(axis=1)))
                              | set(np.flatnonzero(np.isinf(raw266).any(axis=1))))
            for fr, fields in ((cand232, fields232), (frame266, fields266)):
                vals = fr[fields].to_numpy(dtype=float).copy()
                vals[np.isinf(vals)] = np.nan
                fr[fields] = vals
            amp232 = booster232.predict(v1.native_missing_view(cand232, fields232))
            amp266 = booster266.predict(v1.native_missing_view(frame266, fields266))

            records = []
            self._load_static()
            for i, c in enumerate(candidates):
                code = str(c["code"]).zfill(6); P = int(c["P"]); E = int(c["E"])
                cutoff_ts = int(c["cutoff_ts"])
                reasons = []
                if i in set(inf_rows):
                    reasons.append("infinite_feature_value")
                if np.isnan(raw232[i]).all():
                    reasons.append("all_232_missing")
                for col, label in (("stock_day_return_raw", "joint_parts"),
                                   ("context_sector_mean_return_loo_raw", "context_loo")):
                    if col in frame and pd.isna(frame.loc[i, col]):
                        reasons.append("source_join_missing:" + label)
                iso = bool(reasons)
                m = self._master[(self._master.code == code) & (self._master.P == P)]
                atr_ref = (float(m.iloc[0]["atr_ref"]) if len(m) and pd.notna(m.iloc[0]["atr_ref"]) else None)
                risk_amp = {"amp_232_raw": float(amp232[i]), "amp_266_raw": float(amp266[i]),
                            "amp_232": float(np.clip(amp232[i], 0, 1)),
                            "amp_266": float(np.clip(amp266[i], 0, 1)),
                            "atr_ref": atr_ref}
                rec = self._record(code, P, E, feature_cutoff=cutoff_ts, available_at="E_cutoff",
                                   scored_at=int(time.time()),
                                   risk_model_version="auxiliary_120m_fixed_fit_20261018_v1|common_candidate_fixed_fit_20261019_v1",
                                   risk_horizon=RISK_HORIZON_MIN, risk_amplitude=risk_amp,
                                   quality_state="ISOLATED" if iso else "OK",
                                   missing_reasons=reasons,
                                   evidence_refs=["compute:rebuild_features_v1.replay", "joint_shadow_service_v1"])
                records.append(rec)
            return records
        except compute.ComputeUnavailable as e:
            return [self._record(str(c["code"]).zfill(6), c.get("P"), c.get("E"),
                                 feature_cutoff=c.get("cutoff_ts"), quality_state="NOT_COMPUTABLE",
                                 missing_reasons=[str(e)], evidence_refs=["compute:raw_risk_prefix"])
                    for c in candidates]
        except Exception as e:
            return [self._record(str(c["code"]).zfill(6), c.get("P"), c.get("E"),
                                feature_cutoff=int(c.get("cutoff_ts")) if c.get("cutoff_ts") else None,
                                quality_state="RUNTIME_ERROR",
                                missing_reasons=[f"{type(e).__name__}: {e}"],
                                evidence_refs=["joint_shadow:same_tick_risk"]) for c in candidates]

    def _risk_prefix(self, candidates, snapshot):
        from joint_shadow_context_v1 import build_features
        if snapshot is None:
            raise compute.ComputeUnavailable("RISK_SNAPSHOT_REQUIRED_NO_HISTORICAL_FALLBACK")
        frame, x232, x266, audit = build_features(snapshot, candidates)
        if self._boost is None:
            self._boost = (v1.load_booster(v1.MODEL_232), v1.load_booster(v1.MODEL_266))
        predictions = [model.predict(v1.native_missing_view(x, list(x)))
                       for model, x in zip(self._boost, (x232, x266))]
        source_digest = hashlib.sha256(json.dumps(audit['evidence_refs'], sort_keys=True,
            ensure_ascii=False).encode('utf-8')).hexdigest()
        records = []
        for i, c in enumerate(candidates):
            required = ['stock_day_return_raw', 'context_sector_mean_return_loo_raw',
                        'sse_15m_day_return_prev_close', 'cyb_15m_day_return_prev_close',
                        'new_sse_golden_position']
            missing = [name for name in required if not np.isfinite(float(frame.loc[i, name]))]
            if any(not np.isfinite(p[i]) for p in predictions):
                missing.append('MODEL_NONFINITE_OUTPUT')
            amplitude = None if missing else {
                'amp_232_raw': float(predictions[0][i]), 'amp_266_raw': float(predictions[1][i]),
                'amp_232': float(np.clip(predictions[0][i], 0, 1)),
                'amp_266': float(np.clip(predictions[1][i], 0, 1)),
                'semantics': 'FUTURE_120_TRADING_MINUTES_ADVERSE_AMPLITUDE_NOT_PROBABILITY',
                'membership_scope': audit['membership_scope'],
                'observed_at': audit['observed_at'],
                'comparison_board_count': audit['boards'],
                'comparison_stock_count': audit['stocks'],
                'missing_stock_source_count': len(audit['missing_stock_sources']),
                'feature_missing_232': int(x232.iloc[i].isna().sum()),
                'feature_missing_266': int(x266.iloc[i].isna().sum())}
            reasons = ['required_feature_missing:' + name for name in missing]
            if audit['membership_scope'] == 'SNAPSHOT_RECONSTRUCTION_RESEARCH':
                reasons.append('SNAPSHOT_MEMBERSHIP_NOT_HISTORICAL_PIT')
            if audit['missing_stock_sources']:
                reasons.append('PARTIAL_COMPARISON_UNIVERSE')
            records.append(self._record(str(c['code']).zfill(6), int(c['P']), int(c['E']),
                feature_cutoff=snapshot.cutoff, available_at=snapshot.available_at,
                scored_at=int(time.time()), risk_horizon=RISK_HORIZON_MIN,
                risk_model_version='auxiliary_120m_fixed_fit_20261018_v1|common_candidate_fixed_fit_20261019_v1',
                risk_amplitude=amplitude,
                quality_state='NOT_COMPUTABLE' if missing else ('PARTIAL' if reasons else 'OK'),
                missing_reasons=reasons, evidence_refs=['compute:raw_risk_prefix',
                    'risk_snapshot_sources_sha256:' + source_digest,
                    *[snapshot.receipts[k]['reference'] for k in
                      (str(c['code']).zfill(6), '000001.SH', '399006.SZ') if k in snapshot.receipts]]))
        return records


def event_key(interface, code, cutoff_ts, P, E):
    return f"{interface}|{str(code).zfill(6)}|{int(cutoff_ts)}|{int(P)}|{int(E)}"


class ShadowEventLog:
    """Append-only, disk-persistent shadow event log (G6). De-duplicates by unique event
    key; on restart it reloads existing keys from the backing JSONL so the same key is
    never re-written. Write failures are caught (the scorer never crashes) and recorded
    in `last_write_error`."""

    SECRET_HINTS = ("sk-", "api_key", "apikey", "token", "authorization", "secret", "password")

    def __init__(self, out_dir=None):
        self.out_dir = Path(out_dir) if out_dir else None
        self._keys = set()
        self._records = []
        self.last_write_error = None
        if self.out_dir:
            self.out_dir.mkdir(parents=True, exist_ok=True)
            self._load_existing()

    def _load_existing(self):
        import json as _json
        for f in sorted(self.out_dir.glob("shadow_events_*.jsonl")):
            for line in f.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = _json.loads(line)
                except Exception:
                    continue
                k = rec.get("_event_key")
                if k:
                    self._keys.add(k)

    def append(self, record, key=None, interface="?"):
        if key is None:
            key = event_key(interface, record.get("code"),
                            record.get("feature_cutoff") or 0, record.get("P") or 0, record.get("E") or 0)
        if key in self._keys:
            return False  # already persisted (this session or a prior restart)
        if self.out_dir:
            try:
                self._write_line(key, record)
            except Exception as e:  # never crash the scorer on a log write failure
                self.last_write_error = f"{type(e).__name__}: {e}"
                return False
        self._keys.add(key)
        self._records.append(record)
        self.last_write_error = None
        return True

    def _write_line(self, key, record):
        import json as _json
        payload = dict(record)
        payload["_event_key"] = key
        payload["_written_at"] = int(time.time())
        line = _json.dumps(payload, ensure_ascii=False)
        if any(h in line.lower() for h in self.SECRET_HINTS):
            raise ValueError("shadow record must not contain secrets")
        day = time.strftime("%Y%m%d", time.localtime())
        path = self.out_dir / f"shadow_events_{day}.jsonl"
        with open(path, "a", encoding="utf-8") as f:
            f.write(line + "\n")

    def count(self):
        return len(self._records)

    def disk_count(self):
        if not self.out_dir:
            return 0
        n = 0
        for f in self.out_dir.glob("shadow_events_*.jsonl"):
            n += sum(1 for _ in f.read_text(encoding="utf-8").splitlines() if _.strip())
        return n

    def render(self, record):
        import json as _json
        txt = _json.dumps(record, ensure_ascii=False)
        if any(h in txt.lower() for h in self.SECRET_HINTS):
            raise ValueError("shadow record must not contain secrets")
        return txt


def info():
    return {
        "module": "joint_shadow_service_v1",
        "enabled": ENABLED,
        "status": "UNENABLED (本轮未启用以记录影子观察)" if not ENABLED else "ENABLED",
    "side_effects": "scoring read-only; explicit ShadowEventLog may append local records",
        "schema_version": "joint_shadow_v1 (strict 18 fields)",
        "risk_horizon_min": RISK_HORIZON_MIN,
        "categories": {
            "historical_replay_lookup": ["post_market_common_candidate(lookup)", "intraday_trigger_status(lookup)"],
            "compute_from_daily_and_minute_prefix": ["post_market_common_candidate(compute)",
                                                    "intraday_trigger_status(compute)"],
            "risk_frozen_context_reconstruction": ["same_tick_risk(compute)"],
            "risk_from_raw_snapshot_prefix": ["same_tick_risk(prefix, snapshot=RiskSnapshot)"],
        },
        "runtime_fingerprint_check": "ON (rejects on tampered weight/preprocessing/key-dependency)",
        "interfaces": ["post_market_common_candidate", "intraday_trigger_status", "same_tick_risk"],
    }
