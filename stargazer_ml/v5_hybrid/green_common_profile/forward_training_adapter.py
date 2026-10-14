"""Offline two-branch adapter. No networking, no monitoring, no simulated fills.

Input labels are minute starts. Missing slots are never padded. Vendor finality
must be certified outside this adapter before activation; wall time alone is not
proof of a final bar. Existing daily algorithms are reused, not redefined here.
"""
from datetime import datetime, timedelta
import math

import gcp_round_b_engine as engine
import gcp_dual_path_coverage_probe as probe
from gcp_car_v0_2_extractor import detect_signal_v02


def bar_end(ts):
    if len(str(ts)) != 14 or not str(ts).isdigit():
        raise ValueError("INVALID_MINUTE_TIMESTAMP")
    dt = datetime.strptime(str(int(ts)), "%Y%m%d%H%M%S")
    return int((dt + timedelta(minutes=1)).strftime("%Y%m%d%H%M%S"))


def extract(day, daily_upto_p, minute, observed_ts, grid):
    """Return certified prefix action or explicit unclassifiable/no-action state.

    Caller supplies P-only daily history, frozen candidate qualification and
    validated start-labelled canonical grid. Later minute rows are excluded.
    This function does not authorize any source, collection, fitting or T0.
    """
    unavailable = lambda reason: {"status": "DATA_INSUFFICIENT", "reason": reason}
    if int(observed_ts) // 1000000 != int(day):
        return unavailable("OBSERVATION_DATE_MISMATCH")
    if daily_upto_p.empty or minute.empty:
        return unavailable("NO_INPUT")
    if daily_upto_p["date"].duplicated().any():
        return unavailable("DAILY_DUPLICATE")
    if int(daily_upto_p["date"].max()) >= day:
        return unavailable("DAILY_PIT_VIOLATION")
    if (minute["date"].astype("int64") // 1000000 != day).any():
        return unavailable("MINUTE_DATE_MISMATCH")
    try:
        completed = minute["date"].map(bar_end)
    except (ValueError, TypeError):
        return unavailable("INVALID_MINUTE_TIMESTAMP")
    frame = minute[completed <= observed_ts].sort_values("date").copy()
    if frame.empty:
        return unavailable("NO_COMPLETED_BAR")
    if frame["date"].duplicated().any():
        return unavailable("DUPLICATE_SLOT")
    for _, row in frame.iterrows():
        v = [float(row[k]) for k in ("open", "high", "low", "close", "volume", "amount")]
        o, h, lo, c, vol, amt = v
        if (not all(math.isfinite(x) for x in v) or min(o, h, lo, c) <= 0
                or lo > min(o, c) or h < max(o, c) or vol < 0 or amt < 0
                or (vol == 0) != (amt == 0)):
            return unavailable("INVALID_OHLCV")
    daily = daily_upto_p.sort_values("date")
    atr = engine.build_atr_ref(daily)
    if not math.isfinite(atr) or atr <= 0:
        return unavailable("INVALID_ATR")
    zones = engine.build_zones(daily, atr)
    box = probe.evaluate_calibers(daily, atr).get("main", {})
    dip = detect_signal_v02(frame, zones, atr, float(daily["close"].iloc[-1]))
    choices = []
    if dip and dip["signal_type"] in ("BREAK", "ALREADY_BELOW"):
        choices.append((dip["signal_ts"], 1, "DIP_REPAIR", dip))
    # Exactly the existing S1 predicate; do not execute the unresolved S2 volume rule.
    if box.get("valid"):
        for _, row in frame.iterrows():
            ts = int(row["date"])
            if not engine.MONITOR_START <= engine.hhmm(ts) <= engine.MONITOR_END:
                continue
            if (row["low"] <= box["lower"] + probe.LOWER_TOL * atr
                    and row["close"] >= box["lower"]
                    and not row["open"] == row["high"] == row["low"] == row["close"]):
                choices.append((ts, 0, "BASE_HOLD", {
                    "zone_lower": box["lower"], "zone_upper": box["upper"],
                    "signal_type": "S1", "zone_sources": ("SWING_BOX",),
                }))
                break
    winner = min(choices, key=lambda x: (bar_end(x[0]), x[1])) if choices else None
    end = winner[0] if winner else day * 1000000 + engine.MONITOR_END * 100
    slots = engine.expected_grid(grid, day)
    if not slots or len(set(slots)) != len(slots):
        return unavailable("INVALID_FROZEN_GRID")
    expected = {day * 1000000 + sl for sl in slots
                if sl <= end % 1000000}
    present = set(frame["date"].astype(int))
    if any(ts % 1000000 not in slots for ts in present):
        return unavailable("UNEXPECTED_SLOT")
    missing = sorted(expected - present)
    if winner and missing:
        return {**unavailable("BLOCKS_FIRST_BRANCH"), "missing_slots": missing}
    if winner is None:
        if observed_ts < bar_end(end):
            return {"status": "OBSERVING", "reason": "WINDOW_NOT_COMPLETE"}
        if missing:
            return {**unavailable("NO_SIGNAL_UNVERIFIED"), "missing_slots": missing}
        return {"status": "NO_ACTION_COMPLETE"}
    ts, _, branch, zone = winner
    bar = frame.loc[frame["date"].astype(int) == ts].iloc[0]
    lower, upper = float(zone["zone_lower"]), float(zone["zone_upper"])
    return {"status": "ACTION_CERTIFIED", "branch": branch,
            "signal_bar_ts": int(ts), "decision_available_ts": bar_end(ts),
            "sig_type_native": zone["signal_type"], "atr_ref": atr,
            "signal_close": float(bar["close"]), "zone_lower": lower, "zone_upper": upper,
            "anchor_sources": list(zone["zone_sources"]),
            "width_atr": (upper - lower) / atr,
            "pen_atr_bar": (lower - float(bar["low"])) / atr,
            "d_atr": (float(bar["close"]) - upper) / atr,
            "box_lower": lower if branch == "BASE_HOLD" else None,
            "box_upper": upper if branch == "BASE_HOLD" else None,
            "scope": ["DIP_REPAIR", "BASE_HOLD"]}
