"""Phase-5: fixed offline control on the 7 usable events (small-sample exploration only).

User rules implemented verbatim:
1. exclusion m = floor(0.2*n), same retained count across arms; random = exact expectation
   (universe mean); a single Monte Carlo draw only as sanity.
2. boundary ties resolved by probability, never by code: at rank floor(0.2n), if k tied and m
   slots remain, each tied candidate excluded with prob m/k; retained-set expectation =
   sum P(keep_i)*outcome_i; identical rule for 232/266/ATR; tie counts and
   keep-all/exclude-all bounds disclosed.
3. 120m label strictly AFTER the risk cutoff: reference = 15m close AT the cutoff; outcome =
   min low over the next 120 trading minutes / reference - 1 (contract
   INTRADAY_AUXILIARY_120M_OBSERVATION_V1, zero-fetch deterministic). Any trigger-time
   evaluation uses trigger_close reference on a SEPARATE clock, never mixed.
4. 7 events over 6 trading days with overlapping windows: per-event per-day results are the
   ONLY primary report; block bootstrap is exploratory appendix; no capability certification
   from "interval excludes zero".
5. label-available universe is a post-hoc evaluation intersection: the original qualified
   scoring universe (115 rows) is preserved alongside per-record label-missing reasons;
   label availability must NOT become a real-time eligibility condition.

Universe = the replayable downside-path subset of the common candidates (787 population);
it is NOT the full candidate set nor any post-close selection set.
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
from rebuild_features_v1 import TIME, sha, log
from rebuild_features_v2 import slot_profile, missing_intervals, QUARANTINE_CSV, SLOT_T
from intraday_relative_stall_v1 import slots as grid_slots

HERE = Path(__file__).resolve().parent
OUT = HERE / 'rebuild_features_v1'
SEG_EXTENSION = 'EXTENSION_RESEARCH'
EVENTS = ['300639', '600819', '603191', '600230', '300788', '300895', '301179']
LEDGER = json.loads((HERE / 'relative_stall_5m_20261018_v1/raw_input_ledger.json').read_bytes())
LEDGER_BY_CODE = {p['code']: p for p in LEDGER}
RETENTION = 0.2
T0 = time.time()
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


def load_day(path, day):
    frame = pq.read_table(path, columns=['date', 'low', 'close'],
                          filters=[('date', '>=', day * 1000000), ('date', '<=', day * 1000000 + 160000)]).to_pandas()
    ts = frame.date.to_numpy(dtype=np.int64)
    d = ts // 1000000
    minute = ts // 10000 % 100 * 60 + ts // 100 % 100
    am = (minute > 570) & (minute <= 690)
    pm = (minute > 780) & (minute <= 900)
    valid = (np.isfinite(frame[['low', 'close']].to_numpy(dtype=float)).all(axis=1)
             & (ts % 100 == 0) & (ts // 100 % 100 < 60) & (am | pm) & (d == day))
    frame = frame.assign(_slot=np.where(am, minute - 571, 120 + minute - 781))
    out_low = np.full(240, np.nan)
    out_close = np.full(240, np.nan)
    m = valid
    out_low[frame._slot[m]] = frame.low.to_numpy(dtype=float)[m]
    out_close[frame._slot[m]] = frame.close.to_numpy(dtype=float)[m]
    return out_low, out_close


def build_label(path, E, cut_hhmmss, next_day, t7_day, quarantined):
    """Returns dict with ref close, 120m signed min return, adverse drop, T7 changes or reasons."""
    need = cutoff_bars(cut_hhmmss)
    if E in quarantined:
        return dict(reason='E_day_quarantined')
    low_e, close_e = load_day(path, E)
    if not np.isfinite(close_e[:need]).all():
        return dict(reason='bars_missing_before_cutoff')
    ref = close_e[need - 1]
    take_e = min(120, 240 - need)  # window may end mid-afternoon for morning cutoffs
    after_low = low_e[need:need + take_e]
    need_next = 120 - take_e
    lows_next = np.array([])
    if need_next > 0:
        if next_day in quarantined:
            return dict(reason='next_day_quarantined')
        low_n, _ = load_day(path, next_day)
        lows_next = low_n[:need_next]
    window = np.concatenate([after_low[np.isfinite(after_low)], lows_next[np.isfinite(lows_next)]])
    if len(window) != 120:
        return dict(reason='label_window_incomplete', have=int(len(window)))
    signed = float(window.min() / ref - 1)
    t7_close = np.nan
    t7_reason = ''
    if t7_day in quarantined:
        t7_reason = 't7_quarantined'
    else:
        _, close_t7 = load_day(path, t7_day)
        t7_close = close_t7[239]
        if not np.isfinite(t7_close):
            t7_reason = 't7_close_missing'
    return dict(ref_close=float(ref), ret120_signed=signed, adverse120=max(0.0, -signed),
                t7_close=(float(t7_close) if np.isfinite(t7_close) else np.nan), t7_reason=t7_reason)


def exclusion_weights(scores, m):
    """Rule-2 tie-weighted exclusion probabilities. Returns (P_excl, tie_info)."""
    order = np.argsort(-scores, kind='stable')
    s = scores[order]
    n = len(s)
    p = np.zeros(n)
    if m <= 0:
        return p, dict(boundary_k=0, boundary_start=None, m=m)
    if m >= n:
        return np.ones(n), dict(boundary_k=n, boundary_start=0, m=m)
    # boundary group: all positions whose score equals the score at rank m (0-based m-1)
    cut_score = s[m - 1]
    boundary = np.flatnonzero(s == cut_score)
    above = boundary[0]
    k = len(boundary)
    certain = above  # excluded with probability 1
    remaining = m - certain
    p[order[:certain]] = 1.0
    if remaining > 0:
        p[order[boundary]] = remaining / k
    tie = dict(boundary_k=int(k), boundary_start=int(above), m=int(m),
               boundary_codes=None, remaining=int(remaining))
    return p, tie


def arm_expectation(outcome, p_excl):
    keep = 1.0 - p_excl
    return float(np.sum(keep * outcome))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', action='store_true', required=True)
    ap.parse_args()

    put('plan_v4.json', json.dumps(dict(
        status='PHASE5_FIXED_OFFLINE_CONTROL_SMALL_SAMPLE_EXPLORATION_NOT_DEPLOYMENT',
        generated=time.strftime('%Y-%m-%d %H:%M:%S'),
        rules={'1': 'm=floor(0.2n); identical retained counts; random=exact expectation (universe mean); single MC sanity only',
               '2': 'boundary ties by probability m/k, never by code; expectation=sum P(keep)*outcome; same for 232/266/ATR; tie counts + keep-all/exclude-all bounds disclosed',
               '3': '120m label strictly after cutoff: ref=15m close at cutoff; outcome=min low over next 120 trading minutes/ref-1; trigger-time variant uses trigger_close on a separate clock',
               '4': '7 events / 6 days, overlapping windows: per-event per-day table is the ONLY primary report; block bootstrap appendix only; no capability claims',
               '5': 'label-available universe is a post-hoc evaluation intersection; original 115-row scoring universe preserved with per-record label-missing reasons'},
        universe='787 population replayable downside-path subset; NOT the full candidate set, NOT a post-close selection set',
        events=EVENTS, retention=RETENTION, http_requests=0, fit=False), indent=1).encode())

    sec = pd.read_parquet(OUT / 'same_cutoff_sections_v3.parquet')
    sec['code'] = sec.code.astype(str).str.zfill(6)
    replay = pd.read_parquet(v1.CANDIDATE_REPLAY)
    replay['code'] = replay.code.astype(str).str.zfill(6)
    trig_info = replay[replay.status == 'TRIGGERED'][['code', 'E', 'trigger_close']].drop_duplicates(['code', 'E'])

    sm_clock = pq.read_table(v1.SHARED_MARKET, columns=[TIME]).to_pandas()
    cal_days = sorted(set((sm_clock[TIME] // 1000000).astype(int)))
    cal_index = {d: i for i, d in enumerate(cal_days)}
    import csv as csvmod
    quarantine = {}
    with QUARANTINE_CSV.open(encoding='utf-8-sig', newline='') as stream:
        for row in csvmod.DictReader(stream):
            quarantine.setdefault(row['code'], set()).add(int(row['day']))

    # ---------------- labels ----------------
    label_rows = []
    for r in sec.itertuples():
        code = r.code
        E = int(r.E)
        cut_ts = int(r.cutoff_ts)
        cut_hhmmss = cut_ts % 1000000
        entry = LEDGER_BY_CODE[code]
        next_day = cal_days[cal_index[E] + 1]
        t7_day = cal_days[cal_index[E] + 7] if cal_index[E] + 7 < len(cal_days) else None
        if t7_day is None:
            lab = dict(reason='t7_beyond_calendar')
        else:
            lab = build_label(entry['path'], E, cut_hhmmss, next_day, t7_day, quarantine.get(code, set()))
        label_rows.append(dict(section_trigger=r.section_trigger, code=code, E=E, cutoff_ts=cut_ts,
                               is_trigger=bool(r.is_trigger), t7_day=t7_day, **lab))
    labels = pd.DataFrame(label_rows)
    labels = labels.merge(trig_info.rename(columns={'code': 'code', 'E': 'E'}), on=['code', 'E'], how='left')
    labels.loc[~labels.is_trigger, 'trigger_close'] = np.nan
    labels = labels.merge(sec[['section_trigger', 'code', 'E', 'cutoff_ts', 'amp_232', 'amp_266', 'atr_ref']],
                          on=['section_trigger', 'code', 'E', 'cutoff_ts'], how='left', validate='one_to_one')
    log('labels_built', rows=len(labels), ok=int(labels.adverse120.notna().sum()),
        t7_ok=int(labels.t7_close.notna().sum()))

    # ---------------- arms per event ----------------
    event_rows, member_rows = [], []
    for sec_id, g in labels.groupby('section_trigger'):
        if sec_id not in EVENTS:
            continue
        g = g.reset_index(drop=True)
        uni = g[g.adverse120.notna()].copy().reset_index(drop=True)  # label-available universe
        n_all = len(g)
        n = len(uni)
        m = int(np.floor(n * RETENTION))
        out = uni.adverse120.to_numpy(dtype=float)
        arms = {}
        kept_n = n - m
        for arm, col in (('M232', 'amp_232'), ('M266', 'amp_266'), ('ATR', 'atr_ref')):
            scores = uni[col].to_numpy(dtype=float)
            p, tie = exclusion_weights(scores, m)
            total = arm_expectation(out, p)  # rule-2 literal: sum_i P(keep_i) * outcome_i
            per_name = total / kept_n       # per-retained-name expectation (comparable across arms & vs random mean)
            # bounds: all boundary kept vs all boundary excluded
            s = np.sort(-scores)
            cut_score = -s[m - 1]
            boundary_mask = scores == cut_score
            keep_all = ~boundary_mask
            excl_all = np.zeros(n, dtype=bool)
            excl_all[np.argsort(-scores, kind='stable')[:m]] = True
            bound_keep = float(out[keep_all].sum() / keep_all.sum())
            bound_excl = float(out[~excl_all].sum() / (~excl_all).sum())
            arms[arm] = dict(total=total, expect=per_name, tie_k=tie['boundary_k'], bound_keep=bound_keep,
                             bound_excl=bound_excl, p_excl=p)
            for i, row in uni.iterrows():
                member_rows.append(dict(section_trigger=sec_id, E=int(row.E), code=row.code,
                                        cutoff_ts=int(row.cutoff_ts), arm=arm,
                                        score=float(row[col]), p_excl=float(p[i]),
                                        adverse120=float(row.adverse120),
                                        ret120_signed=float(row.ret120_signed),
                                        t7_chg_cutoff=(float(row.t7_close / row.ref_close - 1)
                                                       if pd.notna(row.t7_close) else np.nan),
                                        t7_chg_trigger=(float(row.t7_close / row.trigger_close - 1)
                                                        if pd.notna(row.t7_close) and pd.notna(row.trigger_close)
                                                        and row.trigger_close > 0 else np.nan),
                                        is_trigger=bool(row.is_trigger)))
        rand_exp = float(out.mean())  # exact random expectation, per retained name (rule 1)
        trig = uni[uni.is_trigger]
        row = dict(section_trigger=sec_id, E=int(g.E.iloc[0]), cutoff_ts=int(g.cutoff_ts.iloc[0]),
                   n_scored=int(n_all), n_label_universe=n, m=m, retained=kept_n,
                   M232_total=arms['M232']['total'], M266_total=arms['M266']['total'], ATR_total=arms['ATR']['total'],
                   M232_expect=arms['M232']['expect'], M266_expect=arms['M266']['expect'],
                   ATR_expect=arms['ATR']['expect'], random_expect=rand_exp,
                   random_total=rand_exp * kept_n,
                   diff_232_rand=arms['M232']['expect'] - rand_exp,
                   diff_266_rand=arms['M266']['expect'] - rand_exp,
                   diff_232_atr=arms['M232']['expect'] - arms['ATR']['expect'],
                   diff_266_atr=arms['M266']['expect'] - arms['ATR']['expect'],
                   tie_k_232=arms['M232']['tie_k'], tie_k_266=arms['M266']['tie_k'], tie_k_atr=arms['ATR']['tie_k'],
                   bound_keep_232=arms['M232']['bound_keep'], bound_excl_232=arms['M232']['bound_excl'],
                   bound_keep_266=arms['M266']['bound_keep'], bound_excl_266=arms['M266']['bound_excl'],
                   trigger_adverse=(float(trig.adverse120.iloc[0]) if len(trig) else np.nan))
        event_rows.append(row)
    events = pd.DataFrame(event_rows).sort_values(['E', 'cutoff_ts']).reset_index(drop=True)
    members = pd.DataFrame(member_rows)
    log('arms_done', events=len(events))

    # Monte Carlo sanity (single)
    rng = np.random.RandomState(42)
    mc = {}
    for sec_id, g in labels.groupby('section_trigger'):
        if sec_id not in EVENTS:
            continue
        uni = g[g.adverse120.notna()]
        n, m = len(uni), int(np.floor(len(uni) * RETENTION))
        idx = rng.choice(n, size=max(1, n - m), replace=False)
        mc[sec_id] = float(uni.adverse120.to_numpy()[idx].mean())
    events['mc_sanity'] = events.section_trigger.map(mc)

    # seven-day opportunity of excluded candidates (cutoff clock; primary = certain exclusions of M232)
    excl_rows = []
    for sec_id in EVENTS:
        g = labels[(labels.section_trigger == sec_id) & labels.adverse120.notna()].copy()
        for arm, col in (('M232', 'amp_232'), ('M266', 'amp_266'), ('ATR', 'atr_ref')):
            p, _ = exclusion_weights(g[col].to_numpy(dtype=float), int(np.floor(len(g) * RETENTION)))
            g['p_excl'] = p
            for tier, mask in (('certain', g.p_excl == 1.0), ('boundary', (g.p_excl > 0) & (g.p_excl < 1))):
                sub = g[mask]
                chg = sub.t7_close / sub.ref_close - 1
                chg = chg[np.isfinite(chg)]
                excl_rows.append(dict(section_trigger=sec_id, arm=arm, tier=tier, n=len(sub), n_chg=len(chg),
                                      chg_mean=float(chg.mean()) if len(chg) else np.nan,
                                      chg_median=float(chg.median()) if len(chg) else np.nan,
                                      up_share=float((chg > 0).mean()) if len(chg) else np.nan))
    excluded7 = pd.DataFrame(excl_rows)
    # trigger-clock separate table (rule 3): the 7 triggers under trigger_close reference
    trig_clock = []
    for sec_id in EVENTS:
        t = labels[(labels.section_trigger == sec_id) & labels.is_trigger]
        if len(t):
            t = t.iloc[0]
            chg = (t.t7_close / t.trigger_close - 1) if (pd.notna(t.t7_close) and pd.notna(t.trigger_close)
                                                          and t.trigger_close > 0) else np.nan
            trig_clock.append(dict(section_trigger=sec_id, E=int(t.E), trigger_close=float(t.trigger_close),
                                   ref_close=float(t.ref_close), t7_close=(float(t.t7_close)
                                            if pd.notna(t.t7_close) else np.nan),
                                   chg7_trigger_clock=(float(chg) if np.isfinite(chg) else np.nan),
                                   clock='trigger_time_reference_separate_from_cutoff_clock'))
    trig_clock = pd.DataFrame(trig_clock)

    # ---------------- bootstrap (exploratory appendix only) ----------------
    B = 20000
    diffs = events[['diff_232_rand', 'diff_266_rand', 'diff_232_atr', 'diff_266_atr']].to_numpy(dtype=float)
    boots = np.empty((B, diffs.shape[1]))
    for b in range(B):
        idx = rng.randint(0, len(events), size=len(events))
        boots[b] = diffs[idx].mean(axis=0)
    boot = {name: dict(mean=float(diffs[:, j].mean()),
                       ci2_5=float(np.percentile(boots[:, j], 2.5)),
                       ci97_5=float(np.percentile(boots[:, j], 97.5)),
                       event_mean=float(diffs[:, j].mean()))
            for j, name in enumerate(['M232_minus_random', 'M266_minus_random', 'M232_minus_ATR', 'M266_minus_ATR'])}
    boot_note = ('EXPLORATORY ONLY: 7 events / 6 days with overlapping windows; block bootstrap over events '
                 'must NOT be used to certify capability (no "interval excludes zero" claims).')

    # ---------------- outputs ----------------
    sink = io.BytesIO(); events.to_parquet(sink, index=False, compression='zstd'); put('control_events_v4.parquet', sink.getvalue())
    sink = io.BytesIO(); members.to_parquet(sink, index=False, compression='zstd'); put('control_members_v4.parquet', sink.getvalue())
    sink = io.BytesIO(); excluded7.to_parquet(sink, index=False, compression='zstd'); put('excluded_seven_day_v4.parquet', sink.getvalue())
    sink = io.BytesIO(); trig_clock.to_parquet(sink, index=False, compression='zstd'); put('trigger_clock_seven_day_v4.parquet', sink.getvalue())
    lab_keep = labels.copy()
    lab_keep['ret120_signed'] = lab_keep.ret120_signed.astype('float64')
    sink = io.BytesIO(); lab_keep.to_parquet(sink, index=False, compression='zstd'); put('labels_v4.parquet', sink.getvalue())
    # label-missing reasons for the original 115-row scoring universe (rule 5)
    missing = labels[labels.adverse120.isna()][['section_trigger', 'code', 'E', 'cutoff_ts', 'reason']]
    missing['detail'] = ''
    put('label_missing_reasons_v4.json', json.dumps(missing.to_dict('records'), indent=1).encode())
    put('bootstrap_v4.json', json.dumps(dict(note=boot_note, n_events=len(events), reps=B, arms=boot), indent=1).encode())

    summary = dict(events=events.to_dict('records'),
                   seven_day=excluded7.to_dict('records'),
                   trigger_clock=trig_clock.to_dict('records'),
                   tie_counts=events[['section_trigger', 'tie_k_232', 'tie_k_266', 'tie_k_atr']].to_dict('records'))
    put('control_summary_v4.json', json.dumps(summary, indent=1).encode())
    put('stop_record_v4.json', json.dumps(dict(status='COMPLETE_NOT_FIT', http_requests=0,
        manifest={k: v['sha256'] for k, v in writer.items()}), indent=1).encode())
    log('done', bytes=sum(v['bytes'] for v in writer.values()))
    print(events[['section_trigger', 'E', 'cutoff_ts', 'n_label_universe', 'm', 'M232_expect', 'M266_expect',
                  'ATR_expect', 'random_expect', 'diff_232_rand', 'diff_266_rand', 'diff_232_atr', 'diff_266_atr',
                  'tie_k_232', 'mc_sanity']].to_string())


if __name__ == '__main__':
    main()
