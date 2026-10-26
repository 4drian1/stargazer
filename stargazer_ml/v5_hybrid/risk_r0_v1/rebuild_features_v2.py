"""Phase-3 orchestrator: pB2 binding + decision-cutoff realignment + recovery + v2 outputs.

New file; consumes frozen artifacts read-only. Zero network, zero fits.
Reuses rebuild_features_v1 (imported; its frozen-builder imports are read-only).
"""
import argparse
import hashlib
import io
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

import rebuild_features_v1 as v1
from rebuild_features_v1 import (TIME, sha, log, load_spec, load_booster, native_missing_view,
                                 resolve_logical, OUT as OUT_V1)
from intraday_relative_stall_v1 import slots as grid_slots

HERE = Path(__file__).resolve().parent
OUT = HERE / 'rebuild_features_v1'          # same product directory, v2-suffixed file names
PHASE1 = HERE / '_tmp_coverage_assess' / 'per_record_695.csv'
PB2_TABLE = HERE.parent / 'green_common_profile' / 'round_a_v0_1_impl_fix' / 'rolling_lr_v2_predictions.parquet'
CANDIDATE_REPLAY = HERE.parent / 'green_common_profile' / 'round_a_v0_1_impl_fix' / 'full_intraday_replay_v2.parquet'
MASTER_TABLE = HERE.parent / 'green_common_profile' / 'round_a_v0_1_impl_fix' / 'full_master_table_v2.parquet'
LEDGER = json.loads((HERE / 'relative_stall_5m_20261018_v1/raw_input_ledger.json').read_bytes())
LEDGER_BY_CODE = {p['code']: p for p in LEDGER}
QUARANTINE_CSV = HERE.parents[1] / 'pullback_train' / 'quarantine_2026.csv'
SEG_EXTENSION = 'EXTENSION_RESEARCH'
T0 = time.time()

writer = {}


def put(name, body, terminal=False):
    path = OUT / name
    if path.exists():
        raise ValueError('OUTPUT_EXISTS:' + name)
    with path.open('xb') as stream:
        stream.write(body)
    writer[name] = dict(bytes=len(body), sha256=hashlib.sha256(body).hexdigest())


def slot_times():
    """240 1m slot index -> hhmmss bar-end time."""
    out = []
    for i in range(240):
        if i < 120:
            m = 9 * 60 + 31 + i
        else:
            m = 13 * 60 + 1 + (i - 120)
        out.append((m // 60) * 10000 + (m % 60) * 100)
    return out


SLOT_T = slot_times()


def decision_cutoff(trigger_t):
    """Latest completed 15m grid cutoff <= trigger_t (same-day hhmmss). Returns hhmmss int or None."""
    day = 20260401  # any trading day for grid shape
    grid = [int(t % 1000000) for t in grid_slots(day, 15)]
    cand = [g for g in grid if g <= int(trigger_t)]
    return max(cand) if cand else None


def slot_profile(path, day, quarantined):
    """Valid 1m bar slots for `day` replicating minute_returns bar rules. Returns bool[240]."""
    frame = pq.read_table(path, columns=['date', 'open', 'high', 'low', 'close', 'volume', 'amount']).to_pandas()
    ts = frame.date.to_numpy(dtype=np.int64)
    d = ts // 1000000
    minute = ts // 10000 % 100 * 60 + ts // 100 % 100
    am = (minute > 570) & (minute <= 690)
    pm = (minute > 780) & (minute <= 900)
    num = frame[['open', 'high', 'low', 'close', 'volume', 'amount']].to_numpy(dtype=float)
    o, h, l, c, vol, amt = num.T
    valid = (np.isfinite(num).all(axis=1) & (np.minimum.reduce([o, h, l, c]) > 0)
             & (l <= np.minimum(o, c)) & (h >= np.maximum(o, c)) & (vol >= 0) & (amt >= 0)
             & (ts % 100 == 0) & (ts // 100 % 100 < 60) & (am | pm))
    dup = frame.duplicated('date', keep=False)
    if dup.any():
        bad = dup.groupby(frame.date).apply(lambda g: g[['open', 'high', 'low', 'close', 'volume', 'amount']]
                                            .nunique(dropna=False).max(axis=1).max() > 1, include_groups=False)
        conflicts = set(bad[bad].index)
        valid &= ~np.isin(ts, list(conflicts))
    slot = np.where(am, minute - 571, 120 + minute - 781)
    ok = np.zeros(240, dtype=bool)
    m = (d == day) & valid
    ok[slot[m]] = True
    if day in quarantined:
        ok[:] = False
    return ok


def missing_intervals(missing_idx):
    """Compress missing slot indexes into [start_time, end_time] intervals (hhmmss)."""
    if not len(missing_idx):
        return []
    idx = sorted(missing_idx)
    spans, start, prev = [], idx[0], idx[0]
    for i in idx[1:]:
        if i == prev + 1:
            prev = i
        else:
            spans.append((SLOT_T[start], SLOT_T[prev]))
            start = prev = i
    spans.append((SLOT_T[start], SLOT_T[prev]))
    return spans


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', action='store_true', required=True)
    ap.parse_args()

    mapping, fields232, fields266, extra266 = load_spec()
    put('plan_v2.json', json.dumps(dict(
        status='PHASE3_DECISION_ALIGNED_REBUILD_NOT_FIT', generated=time.strftime('%Y-%m-%d %H:%M:%S'),
        parent='rebuild_features_v1.py', inputs={str(p): sha(p) for p in [
            PB2_TABLE, PHASE1, v1.MODEL_232 / 'lightgbm_model.txt.gz', v1.MODEL_266 / 'lightgbm_model.txt.gz',
            v1.PREPROC_CONFIG, v1.OVERLAY, v1.LEDGER, QUARANTINE_CSV]},
        decision_policy=dict(triggered='latest completed 15m grid cutoff <= trigger_t (grid=slots(day,15), 16 cutoffs 09:45..15:00)',
                             non_triggered='E-day 15:00', before_first_cutoff='NO_RISK_SNAPSHOT_AT_DECISION isolated'),
        recovery_policy='bars required only through decision cutoff; missing slots at/after cutoff recoverable; missing before cutoff still rejected; prev full session + joint_parts reference still mandatory',
        http_requests=0, fit=False), indent=1).encode())

    # ---------------- pB2 ----------------
    pb = pd.read_parquet(PB2_TABLE)
    pb['code'] = pb.code.astype(str).str.zfill(6)
    pb_final = pb[pb.fold == 'FINAL'][['code', 'P', 'pB2']]
    if pb_final.duplicated(['code', 'P']).any():
        raise ValueError('PB2_KEY_NOT_UNIQUE')

    res = pd.read_csv(PHASE1, dtype={'code': str})
    res['code'] = res.code.str.zfill(6)
    replay = pd.read_parquet(CANDIDATE_REPLAY)
    replay['code'] = replay.code.astype(str).str.zfill(6)
    res = res.merge(replay[['code', 'P', 'E', 'trigger_t', 'trigger_close']],
                    on=['code', 'E'], how='left', validate='many_to_one')
    res = res.merge(pb_final, on=['code', 'P'], how='left', validate='many_to_one')
    res['pb2_reason'] = np.where(res.pB2.notna(), '', 'NOT_IN_SCORING_POPULATION(CORP_ACTION_ISOLATED)')
    log('inputs_loaded', records=len(res), pb2_matched=int(res.pB2.notna().sum()))

    # ---------------- decision cutoff + re-verification ----------------
    calendar = json.loads((PHASE1.parent / 'sse_calendar.json').read_bytes()) if (PHASE1.parent / 'sse_calendar.json').exists() else None
    # rebuild calendar from shared clock (authoritative)
    clk = pq.read_table(v1.SHARED_MARKET, columns=[TIME]).to_pandas()
    cal_days = sorted(set((clk[TIME] // 1000000).astype(int)))
    cal_index = {d: i for i, d in enumerate(cal_days)}
    quarantine = {}
    with QUARANTINE_CSV.open(encoding='utf-8-sig', newline='') as stream:
        for row in __import__('csv').DictReader(stream):
            quarantine.setdefault(row['code'], set()).add(int(row['day']))

    verdicts = []
    for r in res.itertuples():
        code, E = r.code, int(r.E)
        triggered = r.status == 'TRIGGERED'
        if triggered:
            cut = decision_cutoff(r.trigger_t)
            if cut is None:
                verdicts.append(dict(code=code, P=int(r.P), E=E, status=r.status, trigger_t=float(r.trigger_t),
                                     decision_cutoff_ts=None, verdict='NO_RISK_SNAPSHOT_AT_DECISION',
                                     missing_slots=[], missing_intervals=[], detail='trigger_t before first completed 15m cutoff'))
                continue
            cutoff_ts = E * 1000000 + cut
            grid = [int(t % 1000000) for t in grid_slots(20260401, 15)]
            bars_needed = (grid.index(cut) + 1) * 15
        else:
            cutoff_ts = E * 1000000 + 150000
            bars_needed = 240
        item = dict(code=code, P=int(r.P), E=E, status=r.status, trigger_t=float(r.trigger_t) if triggered else np.nan,
                    decision_cutoff_ts=cutoff_ts, verdict=None, missing_slots=[], missing_intervals=[], detail='')
        # non-triggered keep phase-1 verdict (computed at identical 15:00 semantics)
        if not triggered:
            item['verdict'] = 'OK' if r.reason == 'OK' else r.reason
            if item['verdict'] != 'OK':
                entry = LEDGER_BY_CODE.get(code)
                if entry is not None:
                    ok = slot_profile(entry['path'], E, quarantine.get(code, set()))
                    miss = np.flatnonzero(~ok)
                    item['missing_slots'] = [int(i) for i in miss]
                    item['missing_intervals'] = missing_intervals(miss)
                    item['detail'] = 'at 15:00 cutoff need 240 bars, missing %d' % len(miss)
            verdicts.append(item)
            continue
        # triggered: full re-verification at decision cutoff
        entry = LEDGER_BY_CODE.get(code)
        if entry is None:
            item['verdict'] = 'no_minute_file'
            verdicts.append(item)
            continue
        ok = slot_profile(entry['path'], E, quarantine.get(code, set()))
        need = np.zeros(240, dtype=bool)
        need[:bars_needed] = True
        missing = np.flatnonzero(need & ~ok)
        if len(missing):
            item['verdict'] = 'E_day_bars_missing_before_cutoff'
            item['missing_slots'] = [int(i) for i in missing]
            item['missing_intervals'] = missing_intervals(missing)
            item['detail'] = 'need %d bars through cutoff, missing %d' % (bars_needed, len(missing))
            verdicts.append(item)
            continue
        prev_day = cal_days[cal_index[E] - 1] if cal_index.get(E, 0) > 0 else None
        if prev_day is None:
            item['verdict'] = 'no_prior_trading_day'
            verdicts.append(item)
            continue
        prev_ok = slot_profile(entry['path'], prev_day, quarantine.get(code, set()))
        if not prev_ok.all():
            miss = np.flatnonzero(~prev_ok)
            item['verdict'] = 'prev_day_missing_or_incomplete'
            item['missing_slots'] = [int(i) for i in miss]
            item['missing_intervals'] = missing_intervals(miss)
            item['detail'] = 'prev trading day %d incomplete (missing %d bars)' % (prev_day, len(miss))
            verdicts.append(item)
            continue
        item['verdict'] = 'OK'
        verdicts.append(item)

    ver = pd.DataFrame(verdicts)
    trig_verdicts = ver[ver.status == 'TRIGGERED'].verdict.value_counts().to_dict()
    log('reverification', triggered=trig_verdicts)
    scored = ver[ver.verdict == 'OK'].copy()
    log('scored_population', records=len(scored), triggered=int((scored.status == 'TRIGGERED').sum()))

    # ---------------- replay + acceptance (same 24 samples as v1) ----------------
    pred232 = pd.read_parquet(v1.MODEL_232 / 'predictions.parquet')
    test = pred232[pred232.segment == 'test'].reset_index(drop=True)
    pick = np.linspace(0, len(test) - 1, v1.N_REPLAY_SAMPLES).astype(int)
    sample = test.iloc[pick].reset_index(drop=True)
    sample_keys = sample[['code', TIME]].copy()
    sample_keys['code'] = sample_keys.code.astype(str).str.zfill(6)

    days = sorted(set((scored.decision_cutoff_ts // 1000000).astype(int))
                  | set((sample_keys[TIME] // 1000000).astype(int)))
    log('replay_days', days=len(days))

    codes, members, native = v1.replay(days)
    booster232 = load_booster(v1.MODEL_232)
    booster266 = load_booster(v1.MODEL_266)

    # configure v1 module globals used by its builders
    sm_clock = pq.read_table(v1.SHARED_MARKET, columns=[TIME, 'cutoff_index']).to_pandas()
    v1.TS_TO_CI = dict(zip(sm_clock[TIME].astype(np.int64), sm_clock.cutoff_index))
    SM_ALL = pq.read_table(v1.SHARED_MARKET).column_names
    mapping_sources = set(mapping.values())
    base_sources = {s for s in mapping_sources if not s.startswith('ar_')}
    derived = {'time_node_nearest_abs_distance', 'session_is_pm', 'context_sector_minus_sse_return_raw',
               'context_stock_minus_sector_return_raw', 'context_stock_minus_leader_return_raw'}
    mom_all = set(pd.read_parquet(v1.MOMENTUM).columns)
    v1.SM_ALL = SM_ALL
    v1.SM_COLS = [c for c in SM_ALL if c in base_sources]
    v1.MOM_COLS = sorted((base_sources & mom_all) - set(v1.SM_COLS) - derived)
    v1.JP_COLS = sorted({s for s in base_sources if s.startswith('stock_')})
    v1.CTX_COLS = sorted({s for s in base_sources if s.startswith('context_')} - derived)
    v1.NH_OWN_COLS = [c for c in pq.read_table(v1.REPAIR / 'own.parquet').column_names if c.startswith('nh_own')]
    diff = pd.read_parquet(v1.OVERLAY)

    # acceptance on frozen samples (bit-exact expectation)
    sample_frame = v1.build_candidate_frame(sample_keys, days, codes, members, native, mapping, diff)
    sample232 = resolve_logical(sample_frame, mapping, fields232)
    got232 = booster232.predict(native_missing_view(sample232, fields232))
    frozen232 = sample.lightgbm_raw.to_numpy(dtype=float)
    d232 = np.abs(got232 - frozen232)
    frame266_s = v1.build_266(sample_frame, sample_keys, days, codes, members, native, extra266)
    got266 = booster266.predict(native_missing_view(frame266_s, fields266))
    pred266 = pd.read_parquet(v1.MODEL_266 / 'predictions.parquet')
    f266 = {(str(c).zfill(6), int(t)): float(v) for c, t, v in zip(pred266.code, pred266[TIME], pred266.lightgbm_raw)}
    frozen266 = np.array([f266[(c, int(t))] for c, t in zip(sample_keys.code, sample_keys[TIME])])
    d266 = np.abs(got266 - frozen266)
    from prepare_direct_warning_comparison_v1 import load_arm
    arm = load_arm('test', 'C1')
    arm_map = {(str(c).zfill(6), int(t)): row for c, t, row in
               zip(arm.code, arm[TIME], arm[fields232].itertuples(index=False, name=None))}
    arm_rows = np.array([arm_map[(c, int(t))] for c, t in zip(sample_keys.code, sample_keys[TIME])], dtype=float)
    s232 = sample232[fields232].to_numpy(dtype=float)
    fd = np.abs(s232 - arm_rows)
    fd[np.isnan(s232) & np.isnan(arm_rows)] = 0.0
    feat_max = float(np.nanmax(fd))
    acceptance = dict(
        samples=v1.N_REPLAY_SAMPLES,
        model232=dict(max_abs_delta=float(d232.max()), all_pass=bool((d232 <= 0).all())),
        model266=dict(max_abs_delta=float(d266.max()), all_pass=bool((d266 <= 0).all())),
        feature_identity=dict(max_abs_delta=feat_max, pass_=bool(feat_max == 0)),
        passed=bool(d232.max() == 0 and d266.max() == 0 and feat_max == 0))
    put('replay_acceptance_v2.json', json.dumps(acceptance, indent=1).encode())
    log('replay_acceptance_v2', **{k: v for k, v in acceptance.items() if k != 'samples'})
    if not acceptance['passed']:
        put('stop_record_v2.json', json.dumps(dict(status='FAILED_REPLAY_ACCEPTANCE')).encode(), terminal=True)
        raise ValueError('REPLAY_ACCEPTANCE_FAILED')

    # ---------------- candidate frame at actual decision cutoffs ----------------
    keys = scored[['code', 'decision_cutoff_ts']].rename(columns={'decision_cutoff_ts': TIME}).reset_index(drop=True)
    cand_frame = v1.build_candidate_frame(keys, days, codes, members, native, mapping, diff)
    cand232 = resolve_logical(cand_frame, mapping, fields232)
    frame266 = v1.build_266(cand_frame, keys, days, codes, members, native, extra266)
    raw232 = cand232[fields232].to_numpy(dtype=float)
    raw266 = frame266[fields266].to_numpy(dtype=float)
    inf_rows = sorted(set(np.flatnonzero(np.isinf(raw232).any(axis=1)).tolist())
                      | set(np.flatnonzero(np.isinf(raw266).any(axis=1)).tolist()))
    for frame, fields in ((cand232, fields232), (frame266, fields266)):
        values = frame[fields].to_numpy(dtype=float).copy()
        values[np.isinf(values)] = np.nan
        frame[fields] = values
    amp232 = booster232.predict(native_missing_view(cand232, fields232))
    amp266 = booster266.predict(native_missing_view(frame266, fields266))
    log('candidate_inference_done', rows=len(scored), inf_rows=len(inf_rows))

    # ---------------- new-cutoff validation on frozen-overlap keys ----------------
    # for scored keys that exist in the frozen own_<month> table, direct recompute (repaired refs)
    # must match production final values (same argument as v1; float32 storage tolerance).
    from build_multiboard_full_features_v1 import market_arrays
    from conditional_board_patch_v1 import with_new_high
    from multiboard_risk_features_v1 import matrix_features, summarize
    report = json.loads((v1.MULTIBOARD / 'report.json').read_bytes())
    spec = json.loads((v1.MULTIBOARD / 'feature_spec.json').read_bytes())
    own_cols = spec['own']
    frozen_own = {}
    key_months = set((keys[TIME] // 100000000).astype(int))
    for part in report['parts']:
        if int(part['month']) not in key_months:
            continue
        t = pq.read_table(v1.MULTIBOARD / part['own'], columns=['code', TIME] + own_cols).to_pandas()
        t['code'] = t.code.astype(str)
        for r in t.itertuples():
            frozen_own[(r.code, int(getattr(r, TIME)))] = None
    ci = {c: i for i, c in enumerate(codes)}
    di = {d: i for i, d in enumerate(days)}
    market = pd.read_parquet(v1.MARKET_FEATURES)
    prod = pd.concat([cand232.reset_index(drop=True), frame266[fields266[232:]].reset_index(drop=True)], axis=1)
    overlap = [(c, int(t)) for c, t in zip(keys.code, keys[TIME]) if (c, int(t)) in frozen_own]
    worst = 0.0
    for code, ts in overlap:
        month = ts // 100000000
        md = [d for d in days if d // 100 == month]
        a, b = di[md[0]], di[md[-1]] + 1
        for frequency in (5, 15):
            width = 240 // frequency
            grid, markets = market_arrays(market, md, frequency)
            pos = {t: i for i, t in enumerate(grid)}
            returns = native[frequency][:, a * width:b * width]
            idx = ci[code]
            related = [m for m in members if idx in m]
            small = matrix_features(returns, related, frequency, exclude_index=idx)
            own = with_new_high(summarize(small, markets, frequency, f'own{frequency}_', own=True),
                                markets, frequency, f'own{frequency}_')
            at = pos[ts]
            row = prod[(keys.code == code) & (keys[TIME] == ts)].iloc[0]
            for f in own_cols:
                if f.startswith(f'own{frequency}_'):
                    vv, pp = own[f][at], float(row[f])
                    if np.isnan(vv) and np.isnan(pp):
                        continue
                    worst = max(worst, abs(vv - pp))
    new_cutoff_validation = dict(overlap_keys=len(overlap), max_abs_diff_recompute_vs_production=worst,
                                 pass_=bool(worst <= 5e-5),
                                 note='production own = frozen baseline + frozen overlay patches == direct repaired-replay recompute')
    put('new_cutoff_validation_v2.json', json.dumps(new_cutoff_validation, indent=1).encode())
    log('new_cutoff_validation', **new_cutoff_validation)

    # ---------------- isolation + outputs ----------------
    scored = scored.reset_index(drop=True)
    iso = []
    for i in range(len(scored)):
        reasons = []
        if i in set(inf_rows):
            reasons.append('infinite_feature_value')
        if np.isnan(raw232[i]).all():
            reasons.append('all_232_missing')
        for col, label in [('stock_day_return_raw', 'joint_parts'), ('context_sector_mean_return_loo_raw', 'context_loo')]:
            if col in cand_frame and pd.isna(cand_frame.loc[i, col]):
                reasons.append('source_join_missing:' + label)
        iso.append(dict(code=scored.loc[i, 'code'], ts=int(scored.loc[i, 'decision_cutoff_ts']),
                        isolated=bool(reasons), reasons=reasons,
                        n_missing_232=int(np.isnan(raw232[i]).sum()),
                        n_missing_266=int(np.isnan(raw266[i]).sum())))
    iso_frame = pd.DataFrame(iso)
    log('isolation', isolated=int(iso_frame.isolated.sum()), of=len(scored))

    info = scored.merge(res[['code', 'P', 'E', 'pB2', 'pb2_reason', 'trigger_close']],
                        on=['code', 'P', 'E'], how='left')
    scores = pd.DataFrame({
        'code': info.code, 'P': info.P, 'E': info.E,
        TIME: info.decision_cutoff_ts.astype(np.int64),
        'trigger_t': info.trigger_t, 'trigger_close': info.trigger_close,
        'status': info.status, 'pB2': info.pB2, 'pb2_reason': info.pb2_reason,
        'amp_232_raw': amp232, 'amp_266_raw': amp266,
        'amp_232': np.clip(amp232, 0, 1), 'amp_266': np.clip(amp266, 0, 1),
        'n_missing_232': iso_frame.n_missing_232, 'n_missing_266': iso_frame.n_missing_266,
        'isolated': iso_frame.isolated,
        'isolation_reason': iso_frame.reasons.map(lambda r: ';'.join(r) if r else ''),
        'scope': SEG_EXTENSION, 'data_cutoff': 20260908})
    sel = pd.read_parquet(MASTER_TABLE)
    sel['code'] = sel.code.astype(str).str.zfill(6)
    scores = scores.merge(sel[['code', 'P', 'setup', 'pool', 'F618', 'atr_ref', 'plan_dist_atr',
                               'golden_dist_618_atr']], on=['code', 'P'], how='left', validate='many_to_one')
    scores = scores.rename(columns={'plan_dist_atr': 'plan_dist'})
    replay_src = replay[['code', 'E', 'reason']].rename(columns={'reason': 'star_status_reason'})
    scores = scores.merge(replay_src, on=['code', 'E'], how='left', validate='many_to_one')

    p_snapshot = scores.drop(columns=['status', 'trigger_t', 'trigger_close', 'star_status_reason',
                                      'amp_232_raw', 'amp_266_raw', 'amp_232', 'amp_266'])
    p_snapshot['asof'] = 'P_close'
    p_snapshot['trigger_status'] = np.nan
    p_snapshot['trigger_basis'] = np.nan
    e_update = scores.copy()
    e_update['asof'] = 'E_decision_cutoff'
    e_update['trigger_status'] = e_update.status
    e_update['trigger_basis'] = e_update.apply(
        lambda r: 'trigger_t=%s trigger_close=%s cutoff=grid<=%s reason=%s' % (
            r.trigger_t, r.trigger_close,
            str(int(r.decision_ts_assumed_end % 1000000)).zfill(6), r.star_status_reason), axis=1)
    sink = io.BytesIO(); p_snapshot.to_parquet(sink, index=False, compression='zstd'); put('pday_snapshot_v2.parquet', sink.getvalue())
    sink = io.BytesIO(); e_update.to_parquet(sink, index=False, compression='zstd'); put('eday_intraday_update_v2.parquet', sink.getvalue())
    feats = pd.concat([scores[['code', TIME, 'E', 'status', 'pB2']].reset_index(drop=True),
                       cand232.reset_index(drop=True),
                       frame266[fields266[232:]].reset_index(drop=True)], axis=1)
    for column in list(fields232) + list(fields266[232:]):
        feats[column] = feats[column].astype('float32')
    sink = io.BytesIO(); feats.to_parquet(sink, index=False, compression='zstd'); put('candidate_features_v2.parquet', sink.getvalue())
    put('candidate_isolation_v2.json', json.dumps(iso, indent=1).encode())

    # ---------------- population stats / cross-section / true missing ----------------
    valid = scores[~scores.isolated].copy()
    valid['day'] = valid[TIME] // 1000000
    per_day = valid.groupby('day').size()
    per_cutoff = valid.groupby(TIME).size()
    stats = dict(
        records_scored=int(len(valid)), records_total_candidates=695,
        triggered=int((valid.status == 'TRIGGERED').sum()),
        distinct_E_days=int(valid.day.nunique()),
        cross_section_per_E_day=dict(mean=float(per_day.mean()), min=int(per_day.min()), max=int(per_day.max()),
                                     ge5=int((per_day >= 5).sum()), ge10=int((per_day >= 10).sum()), days=int(len(per_day))),
        distinct_cutoffs=int(len(per_cutoff)),
        amp232=dict(mean=float(valid.amp_232.mean()), min=float(valid.amp_232.min()), max=float(valid.amp_232.max())),
        pB2_matched=int(valid.pB2.notna().sum()), pB2_missing=int(valid.pB2.isna().sum()))
    put('population_stats_v2.json', json.dumps(stats, indent=1).encode())

    missing_records = ver[ver.verdict != 'OK'].copy()
    missing_records['missing_slots'] = missing_records.missing_slots.map(list)
    missing_records['missing_intervals'] = missing_records.missing_intervals.map(
        lambda s: [[int(a), int(b)] for a, b in s])
    true_missing = missing_records[['code', 'P', 'E', 'status', 'trigger_t', 'decision_cutoff_ts',
                                    'verdict', 'missing_slots', 'missing_intervals', 'detail']].to_dict('records')
    put('true_missing_v2.json', json.dumps(true_missing, indent=1).encode())

    evidence = dict(model232_manifest_sha256=sha(v1.MODEL_232 / 'manifest.json'),
                    model266_manifest_sha256=sha(v1.MODEL_266 / 'manifest.json'),
                    coverage_assess_sha256=sha(HERE / 'coverage_assess_v1.json'),
                    pb2_table_sha256=sha(PB2_TABLE))
    meta = dict(scope=SEG_EXTENSION, asof_semantics=dict(
                    pday_snapshot_v2='P-close knowable columns; E-day amplitudes/triggers removed to avoid look-ahead',
                    eday_intraday_update_v2='each row scored at its actual decision cutoff (triggered: latest completed 15m grid <= trigger_t; non-triggered: 15:00)'),
                pb2_policy='pB2 from rolling_lr_v2_predictions.parquet FINAL fold by (code,P); 8 unmatched codes left NaN with reason NOT_IN_SCORING_POPULATION(CORP_ACTION_ISOLATED); no zero fill',
                decision_grid='slots(day,15): 16 cutoffs 09:45..11:30,13:15..15:00 (bar-completion semantics)',
                evidence=evidence, data_cutoff='frozen minute ledger last timestamp 20260908',
                replay_acceptance='see replay_acceptance_v2.json (bit-exact rerun)',
                new_cutoff_validation='see new_cutoff_validation_v2.json',
                true_missing='see true_missing_v2.json (minimal fetch budget list)')
    put('candidate_scores_meta_v2.json', json.dumps(meta, ensure_ascii=False, indent=1).encode())
    put('stop_record_v2.json', json.dumps(dict(status='COMPLETE_NOT_FIT', http_requests=0,
        manifest={k: v['sha256'] for k, v in writer.items()}), indent=1).encode(), terminal=True)
    log('done', bytes=sum(v['bytes'] for v in writer.values()))
    print(json.dumps(stats, indent=1))


if __name__ == '__main__':
    main()
