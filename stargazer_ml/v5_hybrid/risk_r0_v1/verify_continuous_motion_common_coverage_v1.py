"""Real-candidate acceptance for the NEW common-member (3-bar stable) local-momentum
caliber. Reads-only on already-bound population (NO new fetch, NO re-inventory): it replays
the bound raw-minute ledger OFFLINE via build_continuous_motion_population_v1.arrays, which
is exactly the source the fixed-fit pipeline consumes.

What makes this a REAL acceptance (not the synthetic coverage discussion it replaces):
 * Candidate keys are the ACTUAL (code, decision_ts, segment) rows from the bound population
   parts -- not an artificial all-board x all-bar sweep.
 * Cutoff is honored per candidate: a candidate's features are read at its OWN decision_ts
   bar. The caliber is causal (return_/deceleration look back; leader capture accumulates
   forward only over bars <= decision_ts), so reading the full-series result at that bar is
   identical to truncating inputs at the cutoff -- no future bar leaks in.
 * LOO exclusion is correct AND fast: each candidate is judged on its OWN board with ITSELF
   excluded (exclude = global code, matched as a member VALUE, not its position -- the
   verifier bug fixed earlier). Because real concept boards average ~80 members (~80k
   memberships), we use common_loo_derive_v1: compute the board once, then derive every
   member's LOO output algebraically (proven to equal board_motion_common exactly, with and
   without exclude, in the unit test).

Outputs (saved, audit-side only -- NOT fed into any model):
 * fingerprints.json : input/file hashes (bindings) + code hashes (this verifier + new module
   + frozen matrix + population builder + LOO-derive helper) + config hash.
 * report.json       : per-frequency, per-segment, per-view finite rates, common-member
   support distribution, and a missing-reason breakdown for every non-finite field.
Coverage indices stay on the audit side; nothing is promoted or trained here.
"""
import hashlib
import json
import os
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

import continuous_motion_matrix_v1 as cmm_orig
import continuous_motion_fixed_fit_v1 as fit
from build_continuous_motion_population_v1 import arrays, bindings
from build_multiboard_full_features_v1 import market_arrays
from materialize_relative_stall_pilot_v1 import HERE
import common_loo_derive_v1 as derv

FIELDS = ('return_', 'deceleration', 'current_deceleration', 'captured_deceleration')
MIN_COMMON = 1
# Output dir is overridable so a re-run (e.g. to obtain the DECOMPOSED missing-reason
# categories) writes to a NEW directory and never clobbers previously registered
# fingerprints -- the old report/fingerprints are part of the audit trail.
OUT_DIR = Path(os.environ.get('ACCEPTANCE_COMMON_OUT_DIR')
               or (Path(__file__).resolve().parent / 'acceptance_common_20261018'))


def _file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _classify_reason(field, cc_view, cc_ref, is_dead, min_common):
    """Split missingness into categories that are structurally distinct.

    IMPORTANT (audit note): the earlier run merged every common_count==0 into one
    `no_common_set` bucket. That is too coarse -- the caliber's session-open dead-zone
    masking (a deliberate, disclosed design choice) must NOT be silently attributed to the
    same cause as a genuine data gap, because only the latter is remediable by data.
    """
    if cc_view == 0:
        if is_dead:
            # The 3-bar stable window deliberately withholds at the AM-open / lunch-open
            # dead bars. NOTE: masking the first 3 bars of each half-session is the current
            # CONSERVATIVE implementation choice; it is not a mathematical necessity of
            # computing a 3-bar level. Kept as-is and disclosed -- not tuned by effect.
            return 'dead_zone_masked'
        if cc_ref > 0:
            # common existed before excluding this candidate: it was the sole common member.
            return 'loo_removed_only_common_member'
        # Not a dead bar, and no member has a valid 3-bar window -> genuine data gap.
        return 'gap_no_stable_window'
    if cc_view < min_common:
        return 'under_supported'
    if field in ('return_', 'deceleration'):
        return 'level_or_decel_window_unmet'
    return 'leader_unavailable_or_decel_nan'


def _summarize(field_acc, support_acc, reason_acc):
    out = {}
    for f in FIELDS:
        arr = np.asarray(field_acc[f], dtype=bool)
        n = len(arr)
        out[f] = dict(
            n=n,
            finite=int(arr.sum()),
            finite_rate=float(arr.mean()) if n else 0.0,
            missing_reasons={k: int(v) for k, v in reason_acc[f].items()},
        )
    sup = np.asarray(support_acc, dtype=float)
    valid = sup[sup >= 0]
    if len(valid):
        hist = defaultdict(int)
        for v in valid.astype(int).tolist():
            hist[v] += 1
        out['_support'] = dict(
            n=int(len(valid)),
            mean=float(valid.mean()),
            pct_at_least_1=float((valid >= 1).mean()),
            pct_at_least_2=float((valid >= 2).mean()),
            histogram={str(k): int(v) for k, v in sorted(hist.items())},
        )
    else:
        out['_support'] = dict(n=0)
    return out


def run():
    # Reserve exclusively BEFORE any expensive replay. A directory from a completed
    # or interrupted attempt is evidence, never a resumable/overwritable destination.
    OUT_DIR.mkdir(exist_ok=False)
    import threading, time
    # Keep stdout alive during the (silent) bound-data replay so the harness idle timer
    # does not cancel a long but healthy run.
    def _heartbeat():
        while True:
            time.sleep(5)
            print('.', end='', flush=True)
    threading.Thread(target=_heartbeat, daemon=True).start()

    base = json.loads((fit.previous.prep.OUT / 'config.json').read_bytes())
    parts = fit.previous.prep.load_parts(base)
    TIME = fit.previous.prep.TIME
    population = pd.concat(
        [f[['code', TIME]].assign(segment=s) for s, f in parts.items()],
        ignore_index=True)
    days = sorted(set(population[TIME] // 1000000))
    codes, members, returns, report = arrays(days)
    ci = {c: i for i, c in enumerate(codes)}
    code_to_boards = defaultdict(list)
    for m in members:
        for idx in m:
            code_to_boards.setdefault(codes[idx], []).append(m)

    market = pd.read_parquet(HERE / 'local_market_context_20261017_v1/market_price_features.parquet')
    di = {d: i for i, d in enumerate(days)}
    segments = sorted(set(population.segment))

    acc, support, reason = ({}, {}, {})
    for view in ('loo', 'board'):
        acc[view] = {fr: {s: {f: [] for f in FIELDS} for s in segments} for fr in (5, 15)}
        support[view] = {fr: {s: [] for s in segments} for fr in (5, 15)}
        reason[view] = {fr: {s: {f: defaultdict(int) for f in FIELDS} for s in segments} for fr in (5, 15)}

    for freq in (5, 15):
        width = 240 // freq
        half = width // 2
        # Session-open dead bars masked by the 3-bar stable window (see _classify_reason).
        dead_set = {0, 1, 2, half, half + 1, half + 2}
        r = returns[freq]
        for month in sorted({d // 100 for d in days}):
            md = [d for d in days if d // 100 == month]
            a, b = di[md[0]], di[md[-1]] + 1
            rslice = r[:, a * width:b * width]
            stocks = cmm_orig.path_arrays(1 + rslice, np.isfinite(rslice), freq)
            clock, markets = market_arrays(market, md, freq)
            pos = {int(t): i for i, t in enumerate(clock)}

            # Group this month's candidates by their board(s), recording the member's ROW
            # index within that board (used by the LOO derivation) and the decision_ts bar.
            board_cand = defaultdict(list)
            sub = population[population[TIME].floordiv(100000000).eq(month)]
            n_miss = 0
            for row in sub.itertuples(index=False):
                code = row.code
                t = int(getattr(row, TIME))
                seg = row.segment
                boards = code_to_boards.get(code)
                if not boards or t not in pos:
                    n_miss += 1
                    for view in ('loo', 'board'):
                        for f in FIELDS:
                            acc[view][freq][seg][f].append(False)
                            reason[view][freq][seg][f]['no_board_or_clock_miss'] += 1
                        support[view][freq][seg].append(-1)
                    continue
                bar = pos[t]
                gi = ci[code]
                for m in boards:
                    row_i = m.index(gi)
                    board_cand[tuple(m)].append((row_i, bar, seg))

            # One board compute, then derive per-member LOO algebraically.
            for m in members:
                cand = board_cand.get(tuple(m))
                if not cand:
                    continue
                inter = derv.compute_board(rslice, m, freq, stocks, min_common=MIN_COMMON)
                bf = derv.board_fields(inter)
                n_bars_arr = inter['n']                       # (n_bars,) common_count (board)
                # Precompute EVERY member's LOO output at once (vectorized), then index the
                # candidate's own row at its decision_ts bar. Calling loo_field per candidate
                # would allocate a full-length array ~5M times -- pathologically slow.
                loo_all = {f: derv.loo_all_members(inter, f) for f in FIELDS}
                for row_i, bar, seg in cand:
                    is_dead = (bar % width) in dead_set
                    cc_board = float(n_bars_arr[bar])
                    for f in FIELDS:
                        # board view
                        v = bf[f][bar]
                        ok = bool(np.isfinite(v))
                        acc['board'][freq][seg][f].append(ok)
                        if not ok:
                            reason['board'][freq][seg][f][
                                _classify_reason(f, cc_board, cc_board, is_dead, MIN_COMMON)] += 1
                    support['board'][freq][seg].append(cc_board)
                    # LOO view (candidate excluded from its own board)
                    cm_i = inter['common'][row_i, bar]
                    cc_loo = float(n_bars_arr[bar] - (1 if cm_i else 0))
                    for f in FIELDS:
                        v = loo_all[f][row_i, bar]
                        ok = bool(np.isfinite(v))
                        acc['loo'][freq][seg][f].append(ok)
                        if not ok:
                            reason['loo'][freq][seg][f][
                                _classify_reason(f, cc_loo, cc_board, is_dead, MIN_COMMON)] += 1
                    support['loo'][freq][seg].append(cc_loo)
            print(json.dumps(dict(stage='acceptance_month', frequency=freq, month=month,
                                  candidates=int(len(sub)), missed=n_miss)), flush=True)

    acceptance = {}
    verdict_pass = True
    for freq in (5, 15):
        acceptance[freq] = {}
        for view in ('loo', 'board'):
            acceptance[freq][view] = {
                seg: _summarize(acc[view][freq][seg], support[view][freq][seg], reason[view][freq][seg])
                for seg in segments}
            if view == 'loo':
                for seg in segments:
                    for f in FIELDS:
                        if acceptance[freq][view][seg][f]['finite'] == 0:
                            verdict_pass = False

    fps = dict(bindings())
    fps[str(fit.previous.prep.OUT / 'config.json')] = _file_sha(fit.previous.prep.OUT / 'config.json')
    code_shas = {p: _file_sha(HERE / p) for p in (
        'continuous_motion_common_v1.py',
        'continuous_motion_matrix_v1.py',
        'build_continuous_motion_population_v1.py',
        'common_loo_derive_v1.py')}
    code_shas[str(Path(__file__).resolve())] = _file_sha(Path(__file__).resolve())

    verdict = 'PASS' if verdict_pass else 'BLOCKED_ALL_NULL_FIELD'
    report = dict(
        meta=dict(generated_at=datetime.now(timezone.utc).isoformat(),
                  days=days, n_days=len(days), n_codes=len(codes), n_boards=len(members),
                  n_candidates=int(len(population)), segments=segments,
                  frequencies=[5, 15], min_common=MIN_COMMON,
                  cutoff_honored='per-candidate decision_ts (causal caliber)',
                  loo_exclusion='candidate global code excluded from own board',
                  loo_method='common_loo_derive_v1 (exact, unit-tested)'),
        input_fingerprints=fps,
        code_fingerprints=code_shas,
        arrays_report=report,
        acceptance=acceptance,
        verdict=verdict,
        note='Coverage indices are AUDIT-SIDE only; not promoted into any model or training.',
    )
    with (OUT_DIR / 'report.json').open('x', encoding='utf-8') as stream:
        stream.write(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))
    with (OUT_DIR / 'fingerprints.json').open('x', encoding='utf-8') as stream:
        stream.write(json.dumps(
            dict(input_fingerprints=fps, code_fingerprints=code_shas), ensure_ascii=False, indent=2))
    print(json.dumps(dict(stage='acceptance_done', verdict=verdict,
                          loo_5_return=acceptance[5]['loo'][segments[0]]['return_']['finite_rate'],
                          loo_15_return=acceptance[15]['loo'][segments[0]]['return_']['finite_rate'])),
          flush=True)
    return report


if __name__ == '__main__':
    run()
