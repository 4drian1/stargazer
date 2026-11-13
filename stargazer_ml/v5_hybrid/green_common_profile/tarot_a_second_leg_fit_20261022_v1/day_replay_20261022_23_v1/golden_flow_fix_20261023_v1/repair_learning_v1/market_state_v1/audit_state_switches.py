"""Frozen-rule transition diagnostics, not smoothing, training or human labels."""
import gzip
import io
import json
from collections import Counter
import numpy as np
import pandas as pd
from run_acceptance import HERE, STUDY, RISK, TIME, put, sha
from state_rules import describe
from market_features import observe, build
from repair_features import _rsi

GROUPS = dict(
    map=['break_map_known'],
    position=['break_two_close_reclaim', 'break_stop_above'],
    rsi=['repair_rsi6', 'repair_rsi15'],
    volume=['repair_volume_after_low_ratio'],
    lows=['repair_last_two_lows_rising'],
    price_clock=['repair_bars_since_low', 'repair_return_3bars',
                 'repair_last_two_closes_rising', 'repair_low_rebound_return'],
    break_context=['break_broken_no_repair'])
REPAIR = {'STRONG_REPAIR', 'WEAK_REPAIR'}
DOWN = {'BREAK_UNREPAIRED', 'DECLINE_ENHANCING'}


def sufficient_groups(index, old, new, target):
    """One-group substitutions into old input; program sensitivity, not causality."""
    result = []
    for group, suffixes in GROUPS.items():
        hybrid = dict(old)
        for suffix in suffixes:
            key = index + '_' + suffix
            hybrid[key] = new[key]
        if describe(index, int(new[TIME]), hybrid)['state'] == target:
            result.append(group)
    return result


def node_at(bars, bands, atr, observation):
    """Identity of the same node selected by day_observations; prefix only."""
    closes = bars.close.to_numpy(); lows = bars.low.to_numpy()
    last, low = float(closes[-1]), float(lows.min())
    i = len(bars) - 1; low_i = int(np.argmin(lows))
    if observation['reclaim'] == 1:
        eligible = [s for s in bands if i >= max(1, low_i + 1)
                    and (lows[:-1] < s['node']).any()
                    and min(closes[-2:]) >= s['node']]
        assert eligible
        return min(eligible, key=lambda s: abs(last-s['node']))['node'], 'RECLAIM'
    if observation['stop'] == 1:
        eligible = [s for s in bands if s['node'] <= low and low-s['node'] <= .2*atr]
        assert eligible
        return max(eligible, key=lambda s: s['node'])['node'], 'STOP_ABOVE'
    return None, 'NONE'


def write_frame(name, frame):
    buffer = io.BytesIO(); frame.to_parquet(buffer, index=False, compression='zstd')
    put(name, buffer.getvalue())


def main():
    v2 = STUDY/'repair_effectiveness_v2'
    check_path = HERE/'multiday_review_checks.json'
    feature_path = v2/'full_market_features.parquet'
    reference_path = HERE/'reference_states.parquet'
    checks = json.loads(check_path.read_bytes())
    fixed = {(index, day) for index in ['sse', 'cyb'] for day in checks['dates']}
    worst = {(r['index'], r['day']) for r in checks['worst_switching']}
    selected = sorted(fixed | worst)
    paths = [check_path, feature_path, reference_path, HERE/'state_rules.py',
             HERE/'audit_state_switches.py', STUDY/'repair_features.py',
             STUDY/'market_break_repair_v1/market_features.py']
    for index in ['sse', 'cyb']:
        paths += [RISK/f'index_minutes_v1/{index}_5min.parquet',
                  STUDY/f'market_break_repair_v1/{index}_source_receipts.json.gz',
                  v2/f'phase_missing_{index}_maps.json.gz']
    inputs = {str(p): sha(p) for p in paths}
    put('switch_audit_resume_plan.json', dict(inputs=inputs, index_days=selected,
        previous_failure='switch_audit_failure_01.json: missing source receipt; no audit results written',
        missing_receipts='Rebuild only missing selected days from same native source and frozen wave rules',
        selection='previously frozen 12 calendar days plus six recorded worst index-days',
        groups=GROUPS, substitutions='One group from current row into preceding row; not causal attribution',
        roundtrip='A-B-A uses next row only in retrospective audit, never state inputs',
        fits=0, rule_changed=False, paid_calls=0, independent_accuracy=False))
    panel = pd.read_parquet(feature_path); reference = pd.read_parquet(reference_path)
    frames = []; transitions = []; prefix_checks = []; summaries = []; completed_maps = {}
    for index in ['sse', 'cyb']:
        native = pd.read_parquet(RISK/f'index_minutes_v1/{index}_5min.parquet').sort_values('ts').reset_index(drop=True)
        fast = _rsi(native.close.to_numpy(), 6); slow = _rsi(native.close.to_numpy(), 15)
        receipts = []
        for p in [STUDY/f'market_break_repair_v1/{index}_source_receipts.json.gz', v2/f'phase_missing_{index}_maps.json.gz']:
            receipts += json.loads(gzip.decompress(p.read_bytes()))
        maps = {int(r['day']): r for r in receipts}
        assert len(maps) == len(receipts)
        missing = sorted(d for n,d in selected if n == index and d not in maps)
        if missing:
            _, extra = build(native, index, missing)
            maps.update({int(r['day']): r for r in extra})
            put('switch_missing_'+index+'_receipts.json.gz', gzip.compress(json.dumps(extra,allow_nan=False).encode(),mtime=0))
        completed_maps[index] = missing
        for _, day in [p for p in selected if p[0] == index]:
            f = panel[(panel['index'] == index) & (panel[TIME]//1000000 == day)].sort_values(TIME).reset_index(drop=True)
            slots = np.flatnonzero(native.day.to_numpy() == day)
            bars = native.iloc[slots].rename(columns={'raw_volume': 'volume'}).reset_index(drop=True)
            assert len(f) == len(bars) == 48
            np.testing.assert_array_equal(f[TIME], bars.ts)
            receipt = maps[day]; bands = receipt['bands']; atr = receipt['atr_prior']
            assert all(s['anchors_known_at'] < day and s['first_selected_day'] < day
                       for band in bands for s in band['sources'])
            prior = (fast[slots[0]-1], slow[slots[0]-1])
            rebuilt = observe(bars, fast[slots], slow[slots], prior, bands, atr)
            for col in rebuilt:
                np.testing.assert_allclose(rebuilt[col], f[index+'_break_'+col], rtol=0, atol=0, equal_nan=True)
            states = [describe(index, int(r[TIME]), r.to_dict()) for _, r in f.iterrows()]
            expected = reference[(reference['index'] == index) & (reference['asof']//1000000 == day)].sort_values('asof')
            assert [s['state'] for s in states] == expected.state.tolist()
            rows = []
            for i, state in enumerate(states):
                o = state['observed']; node, path = node_at(bars.iloc[:i+1], bands, atr, o)
                row = dict(index=index, day=day, asof=int(f[TIME].iloc[i]), state=state['state'],
                    selection='fixed_calendar' if (index, day) in fixed else 'worst_switching',
                    close=float(bars.close.iloc[i]), low=float(bars.low.iloc[i]),
                    node=node, path=path, missing='|'.join(state['missing']), **o)
                row['rsi_gap'] = None if o['fast'] is None or o['slow'] is None else o['fast']-o['slow']
                rows.append(row)
                if not i or states[i-1]['state'] == state['state']:
                    continue
                prev = states[i-1]; a, b = prev['state'], state['state']; old = f.iloc[i-1].to_dict(); new = f.iloc[i].to_dict()
                causes = sufficient_groups(index, old, new, b)
                kind = ('UNKNOWN_BOUNDARY' if 'UNKNOWN' in [a,b] else
                        'REPAIR_STRENGTH' if a in REPAIR and b in REPAIR else
                        'REPAIR_DOWN_CROSS' if (a in REPAIR and b in DOWN) or (a in DOWN and b in REPAIR) else 'OTHER')
                transition = dict(index=index, day=day, asof=row['asof'], selection=row['selection'],
                    previous=a, current=b, kind=kind, sufficient_groups='|'.join(causes),
                    no_single_group_sufficient=not causes, roundtrip_middle=i+1 < len(states) and states[i+1]['state'] == a,
                    previous_node=rows[-2]['node'], current_node=node,
                    previous_path=rows[-2]['path'], current_path=path,
                    current_price_below_previous_node=rows[-2]['node'] is not None and row['close'] < rows[-2]['node'],
                    new_session_low=o['bars_since_low'] == 0,
                    previous_rsi_gap=rows[-2]['rsi_gap'], current_rsi_gap=row['rsi_gap'],
                    previous_volume_ratio=prev['observed']['volume_ratio'], current_volume_ratio=o['volume_ratio'],
                    previous_lows_rising=prev['observed']['lows_rising'], current_lows_rising=o['lows_rising'],
                    previous_reclaim=prev['observed']['reclaim'], current_reclaim=o['reclaim'],
                    previous_stop=prev['observed']['stop'], current_stop=o['stop'],
                    missing=row['missing'])
                transitions.append(transition)
            frames.extend(rows)
            summaries.append(dict(index=index, day=day, selection=rows[0]['selection'],
                transitions=sum(states[i]['state'] != states[i-1]['state'] for i in range(1,48))))
            # A non-endpoint real prefix in each selection class; no later bars in observe/node_at.
            if day in [checks['dates'][2], next(d for n,d in sorted(worst) if n == index)]:
                cut = 24; prefix = observe(bars.iloc[:cut], fast[slots[:cut]], slow[slots[:cut]], prior, bands, atr)
                pd.testing.assert_frame_equal(prefix, rebuilt.iloc[:cut])
                assert node_at(bars.iloc[:cut], bands, atr, states[cut-1]['observed']) == (rows[cut-1]['node'], rows[cut-1]['path'])
                prefix_checks.append(dict(index=index, day=day, cutoff=rows[cut-1]['asof'], rows=cut))
    frame = pd.DataFrame(frames); changes = pd.DataFrame(transitions)
    results = {}
    for selection in ['fixed_calendar', 'worst_switching']:
        t = changes[changes.selection == selection]; r = frame[frame.selection == selection]
        strength = t[t.kind == 'REPAIR_STRENGTH']
        results[selection] = dict(index_days=int(r[['index','day']].drop_duplicates().shape[0]),
            observations=len(r), transitions=len(t), kinds=t.kind.value_counts().to_dict(),
            roundtrip_middle=int(t.roundtrip_middle.sum()),
            sufficient_group_counts=dict(Counter(g for value in t.sufficient_groups for g in value.split('|') if g)),
            no_single_group_sufficient=int(t.no_single_group_sufficient.sum()),
            repair_strength_changes=len(strength),
            repair_strength_lows_sufficient=int(strength.sufficient_groups.str.split('|').apply(lambda v:'lows' in v).sum()),
            strong_to_weak=int(((t.previous=='STRONG_REPAIR') & (t.current=='WEAK_REPAIR')).sum()),
            repair_down_crosses=int((t.kind=='REPAIR_DOWN_CROSS').sum()),
            new_low_crosses=int(t.new_session_low.sum()))
    protected = json.loads((RISK/'live_monitor_fingerprints_20261023_v4_flow.json').read_bytes())
    for manifest in [inputs, protected]:
        for p,h in manifest.items(): assert sha(p)==h,p
    write_frame('switch_audit_observations.parquet', frame)
    put('switch_audit_transitions.csv', changes.to_csv(index=False).encode('utf-8-sig'))
    put('switch_audit_report.json', dict(groups=results, index_days=summaries, prefix_checks=prefix_checks,
        full_day_replay_rows=len(frame), completed_maps=completed_maps, production_fingerprints_verified=len(protected),
        group_counts_overlap=True, review_status='PROGRAM_LOGIC_AUDIT_NOT_INDEPENDENT_HUMAN_REVIEW',
        fits=0, paid_calls=0, rule_changed=False))
    print(json.dumps(dict(groups=results, prefix_checks=prefix_checks),ensure_ascii=False))


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        put('switch_audit_failure_02.json', dict(error=repr(exc), status='STOP_NO_AUTORETRY'))
        raise
