"""Phase-6 corrections on v4 artifacts only (no refit, no 20% change, no feature rebuild, zero fetch).

1. ATR arm normalized: score = atr_ref / ref_close (ref = cutoff 15m close, same clock as v4
   label reference). Random expectation unchanged; 232/266-ATR recomputed.
2. 232/266 wording: keep-set mean 120m drops are equal across the seven sections; this does NOT
   establish equal ranking or equal opportunity loss. Per-member exclusion-probability
   differences are emitted.
3. Tie bounds corrected: exclusion count fixed at m; best/worst tie handling only within the
   boundary tie group (m slots in-group to best/worst outcomes). Probability-weighted value
   necessarily lies within the corrected bounds (verified on 600819).
"""
import argparse
import hashlib
import io
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
OUT = HERE / 'rebuild_features_v1'
RETENTION = 0.2
writer = {}


def put(name, body):
    path = OUT / name
    if path.exists():
        raise ValueError('OUTPUT_EXISTS:' + name)
    with path.open('xb') as stream:
        stream.write(body)
    writer[name] = dict(bytes=len(body), sha256=hashlib.sha256(body).hexdigest())


def exclusion_weights(scores, m):
    order = np.argsort(-scores, kind='stable')
    s = scores[order]
    n = len(s)
    p = np.zeros(n)
    if m <= 0:
        return p, 0
    cut_score = s[m - 1]
    boundary = np.flatnonzero(s == cut_score)
    above = boundary[0]
    k = len(boundary)
    certain = above
    remaining = m - certain
    p[order[:certain]] = 1.0
    if remaining > 0:
        p[order[boundary]] = remaining / k
    return p, int(k)


def fixed_count_bounds(outcome, scores, m):
    """Best/worst tie handling with exclusion count FIXED at m (retained-set per-name bounds).
    Best = the r in-group exclusion slots go to the r HIGHEST realized adverse members
    (retained set keeps the low-adverse ones); worst = to the r LOWEST. Returns (best, worst)."""
    n = len(scores)
    kept = n - m
    order = np.argsort(-scores, kind='stable')
    s = scores[order]
    cut_score = s[m - 1]
    boundary = np.flatnonzero(s == cut_score)
    above = boundary[0]
    r = m - above
    o = outcome[order]
    sum_all = float(o.sum())
    certain_total = float(o[:above].sum())
    group_out = o[boundary]
    if r > 0:
        excluded_best = certain_total + float(np.sort(group_out)[-r:].sum())   # exclude high-adverse
        excluded_worst = certain_total + float(np.sort(group_out)[:r].sum())   # exclude low-adverse
    else:
        excluded_best = excluded_worst = certain_total
    return (sum_all - excluded_best) / kept, (sum_all - excluded_worst) / kept


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', action='store_true', required=True)
    ap.parse_args()

    put('plan_v5.json', json.dumps(dict(
        status='PHASE6_CORRECTIONS_ON_V4_ARTIFACTS_ONLY',
        generated=time.strftime('%Y-%m-%d %H:%M:%S'),
        corrections=['ATR arm normalized by cutoff 15m close (same clock as v4 labels)',
                     '232/266 wording: equal keep-set means do not establish equal ranking/opportunity-loss; per-member p_excl diffs emitted',
                     'tie bounds: fixed exclusion count m; best/worst only within boundary tie group; prob-weighted value verified inside bounds (600819)'],
        unchanged=['models', 'features', 'labels', '20% rule', 'random expectation', 'events'],
        http_requests=0, fit=False), indent=1).encode())

    members = pd.read_parquet(OUT / 'control_members_v4.parquet')
    labels = pd.read_parquet(OUT / 'labels_v4.parquet')
    ref = labels[['section_trigger', 'code', 'ref_close']]
    events_v4 = pd.read_parquet(OUT / 'control_events_v4.parquet')

    # ---------------- 1) normalized ATR arm ----------------
    sec3 = pd.read_parquet(OUT / 'same_cutoff_sections_v3.parquet')
    sec3['code'] = sec3.code.astype(str).str.zfill(6)
    norm = members[members.arm == 'ATR'][['section_trigger', 'code', 'E', 'cutoff_ts', 'adverse120',
                                          'ret120_signed', 'score']].copy()
    norm = norm.merge(sec3[['section_trigger', 'code', 'atr_ref']],
                      on=['section_trigger', 'code'], how='left', validate='one_to_one')
    norm = norm.merge(ref, on=['section_trigger', 'code'], how='left', validate='one_to_one')
    norm['score_norm'] = norm.atr_ref / norm.ref_close
    norm['ref_choice'] = 'cutoff_15m_close (same clock as v4 120m label reference)'

    event_rows, member_v5, tie_rows = [], [], []
    for sec_id, g in norm.groupby('section_trigger'):
        g = g.reset_index(drop=True)
        n = len(g)
        m = int(np.floor(n * RETENTION))
        kept = n - m
        out = g.adverse120.to_numpy(dtype=float)

        def arm_stats(scores):
            p, k = exclusion_weights(scores, m)
            total = float(np.sum((1 - p) * out))
            per = total / kept
            best, worst = fixed_count_bounds(out, scores, m)
            return p, k, total, per, best, worst

        p_atr, k_atr, total_atr, per_atr, best_atr, worst_atr = arm_stats(g.score_norm.to_numpy(dtype=float))
        # 232 / 266 unchanged arms recomputed from v4 member scores for consistency
        g232 = members[(members.section_trigger == sec_id) & (members.arm == 'M232')].reset_index(drop=True)
        g266 = members[(members.section_trigger == sec_id) & (members.arm == 'M266')].reset_index(drop=True)
        p232, k232, total232, per232, best232, worst232 = arm_stats(g232.score.to_numpy(dtype=float))
        p266, k266, total266, per266, best266, worst266 = arm_stats(g266.score.to_numpy(dtype=float))

        rand = float(out.mean())
        e4 = events_v4[events_v4.section_trigger == sec_id].iloc[0]
        event_rows.append(dict(section_trigger=sec_id, E=int(e4.E), cutoff_ts=int(e4.cutoff_ts),
                               n=n, m=m, retained=kept,
                               M232_expect=per232, M266_expect=per266,
                               ATR_norm_expect=per_atr, random_expect=rand,
                               diff_232_rand=per232 - rand, diff_266_rand=per266 - rand,
                               diff_232_atr_norm=per232 - per_atr, diff_266_atr_norm=per266 - per_atr,
                               tie_k_232=k232, tie_k_266=k266, tie_k_atr_norm=k_atr))
        for i, row in g.iterrows():
            member_v5.append(dict(section_trigger=sec_id, code=row.code, n=n, m=m,
                                  score_atr_norm=float(row.score_norm), atr_ref=float(row.atr_ref),
                                  ref_close=float(row.ref_close), p_excl_atr_norm=float(p_atr[i]),
                                  p_excl_232=float(p232[i]), p_excl_266=float(p266[i]),
                                  adverse120=float(row.adverse120),
                                  certain_set_differs_232_266=bool(abs(p232[i] - p266[i]) > 1e-12)))
        for arm, per, best, worst, k in (('M232', per232, best232, worst232, k232),
                                         ('M266', per266, best266, worst266, k266),
                                         ('ATR_norm', per_atr, best_atr, worst_atr, k_atr)):
            inside = min(best, worst) - 1e-15 <= per <= max(best, worst) + 1e-15
            tie_rows.append(dict(section_trigger=sec_id, arm=arm, m=m, boundary_k=k,
                                 prob_weighted_per_name=per, bound_best=best, bound_worst=worst,
                                 prob_inside_bounds=bool(inside),
                                 note='exclusion count fixed at m; best/worst assign the r in-group slots '
                                      'to lowest/highest realized adverse within the boundary tie group'))
    events = pd.DataFrame(event_rows)
    members_v5 = pd.DataFrame(member_v5)
    ties = pd.DataFrame(tie_rows)

    # verification table for 600819 (rule: prob-weighted must lie inside corrected bounds)
    v600819 = ties[ties.section_trigger == '600819']
    assert bool(v600819.prob_inside_bounds.all()), 'BOUNDS_VERIFICATION_FAILED_600819'

    # 232 vs 266 exclusion-set differences (may be empty per member; collect all rows with any diff)
    diffs_232_266 = members_v5[members_v5.certain_set_differs_232_266][
        ['section_trigger', 'code', 'p_excl_232', 'p_excl_266', 'adverse120']]

    sink = io.BytesIO(); events.to_parquet(sink, index=False, compression='zstd'); put('control_events_v5.parquet', sink.getvalue())
    sink = io.BytesIO(); members_v5.to_parquet(sink, index=False, compression='zstd'); put('control_members_v5.parquet', sink.getvalue())
    sink = io.BytesIO(); ties.to_parquet(sink, index=False, compression='zstd'); put('tie_bounds_v5.parquet', sink.getvalue())
    sink = io.BytesIO(); diffs_232_266.to_parquet(sink, index=False, compression='zstd'); put('arm232_vs_266_diffs_v5.parquet', sink.getvalue())
    put('stop_record_v5.json', json.dumps(dict(status='COMPLETE_NOT_FIT', http_requests=0,
        verification_600819=v600819.to_dict('records'),
        manifest={k: v['sha256'] for k, v in writer.items()}), indent=1).encode())
    print(events.to_string())
    print()
    print(ties.to_string())
    print()
    print('232/266 p_excl diffs:', len(diffs_232_266))
    print(diffs_232_266.to_string())


if __name__ == '__main__':
    main()
