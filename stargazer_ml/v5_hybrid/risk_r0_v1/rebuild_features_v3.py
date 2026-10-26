"""Phase-4: same-cutoff cross-section completion for the 9 trigger events.

For each trigger, score ALL admission-OK same-day candidates at the trigger's decision cutoff.
Zero network, no frozen modification, no refit, no 237/240 relaxation (missing bars BEFORE the
section cutoff still reject; gaps at/after the cutoff do not block).
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
                                 resolve_logical)
from rebuild_features_v2 import slot_profile, missing_intervals, SLOT_T, QUARANTINE_CSV
from intraday_relative_stall_v1 import slots as grid_slots

HERE = Path(__file__).resolve().parent
OUT = HERE / 'rebuild_features_v1'
PHASE1_DIR = HERE / '_tmp_coverage_assess'
SEG_EXTENSION = 'EXTENSION_RESEARCH'
T0 = time.time()
LEDGER = json.loads((HERE / 'relative_stall_5m_20261018_v1/raw_input_ledger.json').read_bytes())
LEDGER_BY_CODE = {p['code']: p for p in LEDGER}

writer = {}


def put(name, body):
    path = OUT / name
    if path.exists():
        raise ValueError('OUTPUT_EXISTS:' + name)
    with path.open('xb') as stream:
        stream.write(body)
    writer[name] = dict(bytes=len(body), sha256=hashlib.sha256(body).hexdigest())


def cutoff_bars(cut_hhmmss):
    grid = [int(t % 1000000) for t in grid_slots(20260401, 15)]
    return (grid.index(int(cut_hhmmss)) + 1) * 15


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', action='store_true', required=True)
    ap.parse_args()

    mapping, fields232, fields266, extra266 = load_spec()
    put('plan_v3.json', json.dumps(dict(
        status='PHASE4_SAME_CUTOFF_SECTIONS_NOT_FIT', generated=time.strftime('%Y-%m-%d %H:%M:%S'),
        parent='rebuild_features_v2.py', inputs={str(p): sha(p) for p in [
            PHASE1_DIR / 'per_record_787.csv', v1.MODEL_232 / 'lightgbm_model.txt.gz',
            v1.MODEL_266 / 'lightgbm_model.txt.gz', OUT / 'eday_intraday_update_v2.parquet',
            v1.PREPROC_CONFIG, v1.LEDGER, QUARANTINE_CSV]},
        section_policy=('members = all admission-OK status-filtered same-E-day candidates (787 population口径), '
                        'verified at the trigger decision cutoff: bars required only through cutoff; '
                        'gaps at/after cutoff do not block (e.g. an 11:00 cutoff is not blocked by an 11:30 gap); '
                        'missing bars before cutoff still reject; prev full session + joint_parts reference mandatory; '
                        'later-triggering members are NOT excluded (only-past-information scoring)'),
        http_requests=0, fit=False), indent=1).encode())

    # ---------------- inputs ----------------
    r787 = pd.read_csv(PHASE1_DIR / 'per_record_787.csv', dtype={'code': str})
    r787['code'] = r787.code.str.zfill(6)
    replay = pd.read_parquet(v1.CANDIDATE_REPLAY)
    replay['code'] = replay.code.astype(str).str.zfill(6)
    r787 = r787.merge(replay[['code', 'P', 'E', 'trigger_t', 'trigger_close']],
                      on=['code', 'E'], how='left', validate='many_to_one')
    pb = pd.read_parquet(HERE.parent / 'green_common_profile/round_a_v0_1_impl_fix/rolling_lr_v2_predictions.parquet')
    pb['code'] = pb.code.astype(str).str.zfill(6)
    r787 = r787.merge(pb[pb.fold == 'FINAL'][['code', 'P', 'pB2']], on=['code', 'P'], how='left', validate='many_to_one')
    sel = pd.read_parquet(v1.CANDIDATE_REPLAY.parent / 'full_master_table_v2.parquet')
    sel['code'] = sel.code.astype(str).str.zfill(6)
    r787 = r787.merge(sel[['code', 'P', 'F618', 'atr_ref', 'plan_dist_atr']],
                      on=['code', 'P'], how='left', validate='many_to_one')

    e_v2 = pd.read_parquet(OUT / 'eday_intraday_update_v2.parquet')
    e_v2['code'] = e_v2.code.astype(str).str.zfill(6)
    triggers = e_v2[e_v2.status == 'TRIGGERED'].copy()
    if len(triggers) != 9:
        raise ValueError('EXPECTED_9_TRIGGERS:' + str(len(triggers)))
    sections = []
    for r in triggers.itertuples():
        cut_ts = int(getattr(r, TIME))
        sections.append(dict(trigger_code=r.code, E=int(r.E), cutoff_ts=cut_ts,
                             cut_hhmmss=cut_ts % 1000000, trigger_t=float(r.trigger_t)))
    log('sections', n=len(sections), days=sorted({s['E'] for s in sections}))

    # ---------------- per-section member verification ----------------
    quarantine = {}
    import csv as csvmod
    with QUARANTINE_CSV.open(encoding='utf-8-sig', newline='') as stream:
        for row in csvmod.DictReader(stream):
            quarantine.setdefault(row['code'], set()).add(int(row['day']))
    sm_clock = pq.read_table(v1.SHARED_MARKET, columns=[TIME, 'cutoff_index']).to_pandas()
    cal_days = sorted(set((sm_clock[TIME] // 1000000).astype(int)))
    cal_index = {d: i for i, d in enumerate(cal_days)}
    ts_to_ci = dict(zip(sm_clock[TIME].astype(np.int64), sm_clock.cutoff_index))

    # joint_parts reference pre-pass for all section cutoffs
    needed_ci = {ts_to_ci[s['cutoff_ts']] for s in sections}
    jmanifest = json.loads((v1.JOINT / 'manifest.json').read_bytes())
    ref_keys = set()
    for item in jmanifest['joint_parts']:
        t = pq.read_table(v1.JOINT / item['path'], columns=['code', 'cutoff_index', 'stock_reference_close_raw'],
                          filters=[('cutoff_index', 'in', sorted(needed_ci))]).to_pandas()
        t = t[t.stock_reference_close_raw.notna()]
        ref_keys |= set(zip(t.code.astype(str), t.cutoff_index))

    member_rows = []
    for s in sections:
        need_bars = cutoff_bars(s['cut_hhmmss'])
        next_day = cal_days[cal_index[s['E']] + 1]
        for m in r787[r787.E == s['E']].itertuples():
            code = m.code
            entry = LEDGER_BY_CODE.get(code)
            reason = ''
            miss_intervals = []
            if entry is None:
                reason = 'no_minute_file'
            else:
                ok = slot_profile(entry['path'], s['E'], quarantine.get(code, set()))
                miss = np.flatnonzero(~ok[:need_bars])
                if len(miss):
                    reason = 'E_day_bars_missing_before_cutoff'
                    miss_intervals = missing_intervals(miss)
                else:
                    prev_day = cal_days[cal_index[s['E']] - 1]
                    prev_ok = slot_profile(entry['path'], prev_day, quarantine.get(code, set()))
                    if not prev_ok.all():
                        reason = 'prev_day_missing_or_incomplete'
                        miss_intervals = missing_intervals(np.flatnonzero(~prev_ok))
                    elif (code, ts_to_ci[s['cutoff_ts']]) not in ref_keys:
                        reason = 'joint_reference_missing'
            # label window availability (diagnostic for control feasibility):
            # future slots t+1..t+120 -> E-day bars after cutoff + next-day bars
            label_ok = False
            label_detail = ''
            if not reason:
                e_after = 240 - need_bars
                need_next = 120 - e_after
                ok_e = slot_profile(entry['path'], s['E'], quarantine.get(code, set()))
                ok_n = slot_profile(entry['path'], next_day, quarantine.get(code, set()))
                bad_e = np.flatnonzero(~ok_e[need_bars:])
                bad_n = np.flatnonzero(~ok_n[:need_next]) if need_next > 0 else np.array([])
                if len(bad_e) or len(bad_n):
                    label_detail = 'label window gaps: E_after=%s next=%s' % (
                        missing_intervals(bad_e), missing_intervals(bad_n))
                else:
                    label_ok = True
            member_rows.append(dict(section_trigger=s['trigger_code'], E=s['E'], cutoff_ts=s['cutoff_ts'],
                                    code=code, P=int(m.P), status=m.status, trigger_t=m.trigger_t,
                                    is_trigger=(code == s['trigger_code'] and m.status == 'TRIGGERED'),
                                    verdict=reason if reason else 'OK', missing_intervals=miss_intervals,
                                    label_ok=label_ok, label_detail=label_detail,
                                    atr_ref=m.atr_ref, F618=m.F618, pB2=m.pB2))
    members = pd.DataFrame(member_rows)
    log('members_built', rows=len(members),
        by_section=members.groupby('section_trigger').size().to_dict())

    scored = members[members.verdict == 'OK'].copy().reset_index(drop=True)
    keys = scored[['code', 'cutoff_ts']].rename(columns={'cutoff_ts': TIME}).reset_index(drop=True)
    if keys.duplicated().any():
        # same code in two sections of the same day at different cutoffs is fine (different TIME);
        # same (code,TIME) cannot repeat by construction
        raise ValueError('DUPLICATE_SECTION_KEY')

    # ---------------- replay + acceptance ----------------
    pred232 = pd.read_parquet(v1.MODEL_232 / 'predictions.parquet')
    test = pred232[pred232.segment == 'test'].reset_index(drop=True)
    pick = np.linspace(0, len(test) - 1, v1.N_REPLAY_SAMPLES).astype(int)
    sample_keys = test.iloc[pick][['code', TIME]].copy()
    sample_keys['code'] = sample_keys.code.astype(str).str.zfill(6)
    days = sorted(set((keys[TIME] // 1000000).astype(int)) | set((sample_keys[TIME] // 1000000).astype(int)))
    log('replay_days', days=len(days))
    codes, members_mtx, native = v1.replay(days)
    booster232 = load_booster(v1.MODEL_232)
    booster266 = load_booster(v1.MODEL_266)
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
    v1.TS_TO_CI = ts_to_ci
    diff = pd.read_parquet(v1.OVERLAY)

    sample_frame = v1.build_candidate_frame(sample_keys, days, codes, members_mtx, native, mapping, diff)
    sample232 = resolve_logical(sample_frame, mapping, fields232)
    d232 = np.abs(booster232.predict(native_missing_view(sample232, fields232)) -
                  test.iloc[pick].lightgbm_raw.to_numpy(dtype=float))
    frame266_s = v1.build_266(sample_frame, sample_keys, days, codes, members_mtx, native, extra266)
    pred266 = pd.read_parquet(v1.MODEL_266 / 'predictions.parquet')
    f266 = {(str(c).zfill(6), int(t)): float(v) for c, t, v in zip(pred266.code, pred266[TIME], pred266.lightgbm_raw)}
    frozen266 = np.array([f266[(c, int(t))] for c, t in zip(sample_keys.code, sample_keys[TIME])])
    d266 = np.abs(booster266.predict(native_missing_view(frame266_s, fields266)) - frozen266)
    acceptance = dict(samples=v1.N_REPLAY_SAMPLES,
                      model232_max_abs_delta=float(d232.max()), model266_max_abs_delta=float(d266.max()),
                      passed=bool(d232.max() == 0 and d266.max() == 0))
    put('replay_acceptance_v3.json', json.dumps(acceptance, indent=1).encode())
    log('replay_acceptance_v3', **acceptance)
    if not acceptance['passed']:
        put('stop_record_v3.json', json.dumps(dict(status='FAILED_REPLAY_ACCEPTANCE')).encode())
        raise ValueError('REPLAY_ACCEPTANCE_FAILED')

    # ---------------- section inference ----------------
    cand_frame = v1.build_candidate_frame(keys, days, codes, members_mtx, native, mapping, diff)
    cand232 = resolve_logical(cand_frame, mapping, fields232)
    frame266 = v1.build_266(cand_frame, keys, days, codes, members_mtx, native, extra266)
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
    log('section_inference_done', rows=len(scored), inf_rows=len(inf_rows))

    iso_reasons = []
    for i in range(len(scored)):
        reasons = []
        if i in set(inf_rows):
            reasons.append('infinite_feature_value')
        if np.isnan(raw232[i]).all():
            reasons.append('all_232_missing')
        for col, label in [('stock_day_return_raw', 'joint_parts'), ('context_sector_mean_return_loo_raw', 'context_loo')]:
            if pd.isna(cand_frame.loc[i, col]):
                reasons.append('source_join_missing:' + label)
        iso_reasons.append(';'.join(reasons))
    scored['isolated'] = [bool(r) for r in iso_reasons]
    scored['isolation_reason'] = iso_reasons
    scored['amp_232_raw'] = amp232
    scored['amp_266_raw'] = amp266
    scored['amp_232'] = np.clip(amp232, 0, 1)
    scored['amp_266'] = np.clip(amp266, 0, 1)
    scored['n_missing_232'] = [int(np.isnan(raw232[i]).sum()) for i in range(len(scored))]
    scored['scope'] = SEG_EXTENSION

    # ranks within each section (same cutoff_ts), descending amplitude / atr_ref
    out_frames = []
    for cut_ts, group in scored[~scored.isolated].groupby('cutoff_ts'):
        g = group.copy()
        g['rank_amp232'] = g.amp_232.rank(ascending=False, method='min')
        g['rank_amp266'] = g.amp_266.rank(ascending=False, method='min')
        g['rank_atr'] = g.atr_ref.rank(ascending=False, method='min')
        out_frames.append(g)
    valid = pd.concat(out_frames, ignore_index=True) if out_frames else scored.iloc[0:0]
    trig_rows = valid[valid.is_trigger]
    summary = []
    for s in sections:
        sec_all = members[members.section_trigger == s['trigger_code']]
        sec_val = valid[valid.cutoff_ts == s['cutoff_ts']]
        trow = sec_val[sec_val.is_trigger]
        summary.append(dict(trigger=s['trigger_code'], E=s['E'], cutoff=s['cutoff_ts'],
                            cut_hhmmss=s['cut_hhmmss'], trigger_t=s['trigger_t'],
                            n_members=int(len(sec_all)), n_qualified=int(len(sec_val)),
                            n_label_ok=int(sec_val.label_ok.sum()) if len(sec_val) else 0,
                            trigger_amp_232=float(trow.amp_232.iloc[0]) if len(trow) else None,
                            trigger_amp_266=float(trow.amp_266.iloc[0]) if len(trow) else None,
                            rank_amp232=int(trow.rank_amp232.iloc[0]) if len(trow) else None,
                            rank_amp266=int(trow.rank_amp266.iloc[0]) if len(trow) else None,
                            rank_atr=int(trow.rank_atr.iloc[0]) if len(trow) else None))
    summary_frame = pd.DataFrame(summary)
    put('cross_section_summary_v3.json', json.dumps(summary, indent=1).encode())

    # ---------------- outputs ----------------
    table = valid[['section_trigger', 'E', 'cutoff_ts', 'code', 'P', 'status', 'is_trigger',
                   'trigger_t', 'amp_232', 'amp_266', 'amp_232_raw', 'amp_266_raw',
                   'rank_amp232', 'rank_amp266', 'rank_atr', 'atr_ref', 'F618', 'pB2',
                   'label_ok', 'label_detail', 'n_missing_232', 'isolated', 'isolation_reason',
                   'scope']].copy()
    sink = io.BytesIO(); table.to_parquet(sink, index=False, compression='zstd')
    put('same_cutoff_sections_v3.parquet', sink.getvalue())
    excluded = members[members.verdict != 'OK'].copy()
    excluded['missing_intervals'] = excluded.missing_intervals.map(lambda s: [[int(a), int(b)] for a, b in s])
    put('section_excluded_v3.json', json.dumps(excluded.to_dict('records'), indent=1).encode())
    feats = pd.concat([scored[['code', 'cutoff_ts', 'E', 'section_trigger', 'is_trigger']]
                       .rename(columns={'cutoff_ts': TIME}).reset_index(drop=True),
                       cand232.reset_index(drop=True),
                       frame266[fields266[232:]].reset_index(drop=True)], axis=1)
    for column in list(fields232) + list(fields266[232:]):
        feats[column] = feats[column].astype('float32')
    sink = io.BytesIO(); feats.to_parquet(sink, index=False, compression='zstd')
    put('section_features_v3.parquet', sink.getvalue())
    meta = dict(scope=SEG_EXTENSION,
                population='787口径: status in {TRIGGERED,NOT_TRIGGERED} & E window & master_admission==OK, ledger-intersected for scoring',
                cutoff_rule='section cutoff = trigger decision cutoff; bars required only through cutoff; 11:00 cutoff NOT blocked by an 11:30 gap; gaps before cutoff still reject',
                later_trigger_policy='members that trigger later in the day are kept (scored with information available at the section cutoff)',
                evidence=dict(model232_manifest_sha256=sha(v1.MODEL_232 / 'manifest.json'),
                              model266_manifest_sha256=sha(v1.MODEL_266 / 'manifest.json')),
                data_cutoff='frozen minute ledger last timestamp 20260908')
    put('candidate_scores_meta_v3.json', json.dumps(meta, ensure_ascii=False, indent=1).encode())
    put('stop_record_v3.json', json.dumps(dict(status='COMPLETE_NOT_FIT', http_requests=0,
        manifest={k: v['sha256'] for k, v in writer.items()}), indent=1).encode())
    log('done', bytes=sum(v['bytes'] for v in writer.values()))
    print(summary_frame.to_string())


if __name__ == '__main__':
    main()
