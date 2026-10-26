"""Fixed-six local price-path diagnostics. No fetch, labels, fit or live facts."""
from collections import Counter
from copy import deepcopy
import json
from pathlib import Path

import pyarrow.parquet as pq

from minute_risk_features_v1 import aggregate, features, legacy_panel_clock, parse_ts, source_end
from price_verification_guarded_run import sha
from research_budget_v2 import bytes_used, TOTAL_CAP

HERE = Path(__file__).resolve().parent
BASE = HERE.parent / 'green_common_profile/gcp_round_b_small_sample_v1'
PROJECT = HERE.parents[2]
OUT = HERE / 'intraday_cache_probe_20261016_v1'
CAP = 2 * 1024**2
EXPECTED = [('605358', 20250103), ('001339', 20250106), ('300713', 20250106),
            ('300726', 20250106), ('300919', 20250106), ('300964', 20250106)]


def encoded(value):
    return (json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2) + '\n').encode('utf8')


def run():
    if OUT.exists():
        raise ValueError('OUTPUT_EXISTS_NO_RESTART')
    if bytes_used() + CAP > TOTAL_CAP:
        raise ValueError('REGISTERED_BUDGET')
    refs = {}

    def track(path):
        path = Path(path).resolve()
        refs[str(path)] = sha(path)
        return path

    for name in ('risk_intraday_cache_probe_v1.py', 'minute_risk_features_v1.py',
                 'test_minute_risk_features_v1.py', 'research_budget_v2.py'):
        track(HERE / name)
    events = json.loads(track(BASE / 'r1_branch_at_decision_v0_1_2.json').read_bytes())['events']
    cases = sorted((r for r in events if r['branch_at_decision'] == 'DIP_REPAIR'
                    and r['cert'] == 'CERTIFIED' and 20250102 <= r['E'] <= 20260908),
                   key=lambda r: (r['E'], r['code']))[:6]
    if [(r['code'], r['E']) for r in cases] != EXPECTED:
        raise ValueError('FIXED_SELECTION_DRIFT')
    dip = json.loads(track(BASE / 'r1_dip_scan_and_gap_freeze.json').read_bytes())['per_event']
    by_key = {(r['code'], r['E']): r for r in dip}
    if len(by_key) != len(dip):
        raise ValueError('DUPLICATE_DIP_EVENT')
    calendar = json.loads(track(HERE / 'historical_stock_day_preparation_v1/manifest.json').read_bytes())['calendar_dates']
    results = []
    for case in cases:
        code, day = case['code'], case['E']
        origin = by_key[code, day]
        previous = origin['P']
        if calendar[calendar.index(previous) + 1] != day:
            raise ValueError('P_E_RELATION')
        primary = track(BASE / f'minute_ext/{code}.parquet')
        rows = [r for r in pq.read_table(primary).to_pylist() if r['date'] // 1000000 in (previous, day)]
        alternatives = []
        for path in (BASE / f'minute/{code}.parquet', BASE / f'minute_car/{code}.parquet',
                     HERE.parent / f'../pullback_train/minute_cache_clean/{code}.parquet'):
            if not path.exists():
                continue
            alt = [r for r in pq.read_table(track(path)).to_pylist()
                   if r['date'] // 1000000 in (previous, day)]
            # Preserve all duplicate values; do not silently overwrite either source.
            left, right = {}, {}
            for source, target in ((rows, left), (alt, right)):
                for r in source:
                    target.setdefault(r['date'], set()).add(tuple(r[k] for k in ('open', 'high', 'low', 'close', 'volume')))
            shared = sorted(left.keys() & right.keys())
            alternatives.append(dict(path=str(path.resolve()), shared_timestamps=len(shared),
                primary_only=len(left.keys() - right.keys()), alternative_only=len(right.keys() - left.keys()),
                ohlc_different=sum({v[:4] for v in left[t]} != {v[:4] for v in right[t]} for t in shared),
                volume_different=sum({v[4] for v in left[t]} != {v[4] for v in right[t]} for t in shared),
                source_duplicates=sum(len(v) > 1 for v in right.values()), used_for_construction=False))
        scenarios = []
        for semantics in ('START', 'END'):
            bars, issues = aggregate(rows, code, [previous, day], semantics)
            derived = features(bars)
            checks = []
            for bar in bars[16:]:
                cutoff = parse_ts(bar['bar_end'])
                prefix = [r for r in rows if source_end(r, semantics) <= cutoff]
                pb, _ = aggregate(prefix, code, [previous, day], semantics)
                pf = features(pb)
                n = sum(b['bar_end'] <= bar['bar_end'] for b in bars)
                if pb[:n] != bars[:n] or pf[:n] != derived[:n]:
                    raise ValueError('PREFIX_MISMATCH')
                perturbed = deepcopy(rows)
                future = 0
                for r in perturbed:
                    if source_end(r, semantics) > cutoff:
                        future += 1
                        for field in ('open', 'high', 'low', 'close'):
                            r[field] *= 1.5
                        r['volume'] *= 10
                fb, _ = aggregate(perturbed, code, [previous, day], semantics)
                if fb[:n] != bars[:n] or features(fb)[:n] != derived[:n]:
                    raise ValueError('FUTURE_PERTURBATION_MISMATCH')
                checks.append(dict(cutoff=bar['bar_end'], prefix_equal=True,
                    future_perturbation='PASS' if future else 'NOT_TESTED_NO_FUTURE_ROWS',
                    mutated_rows=future))
            scenarios.append(dict(semantics_assumption=semantics, certified=False, issues=issues,
                completeness=dict(Counter(b['status'] for b in bars[16:])),
                e_day=[dict(bar=b, fields=f) for b, f in zip(bars[16:], derived[16:])], checks=checks))
        results.append(dict(code=code, P=previous, E=day, branch='DIP_REPAIR',
            frozen_signal_bar_ts=case['dip_ts'], zone=origin['dip_zone_info'],
            reconstruction_scope='HISTORICAL_PRICE_PATH_NOT_AS_RECORDED_PLAN_OR_QUOTE',
            raw_rows=len(rows), alternatives=alternatives, scenarios=scenarios))
    journal = []
    journal_root = PROJECT / 'stargazer_desktop/dist/data/pentacles_demo/journal'
    for path in sorted((journal_root / 'days').glob('*.json')):
        value = json.loads(track(path).read_bytes())
        members = list(value['candidates'].values())
        journal.append(dict(day=value['day'], candidates=len(members),
            quote_samples=sum(r.get('samples', 0) for r in members),
            recorded_touches=sum(r.get('first_observed_touch') is not None for r in members),
            attempts=value.get('attempts', {})))
    panels = []
    for path in sorted((HERE.parent / 'opportunity_v2/gate1c/panels').glob('sector_sametime_proxy_*.parquet')):
        table = pq.read_table(track(path))
        data = table.to_pylist()
        converted = [legacy_panel_clock(r['trade_date'], r['hhmm'], r['minute_ts']) for r in data]
        panels.append(dict(path=str(path), rows=len(data), valid_flags=dict(Counter(str(r['sector_proxy_valid']) for r in data)),
            benchmark_scopes=sorted({r['benchmark_scope'] for r in data}),
            clock_conversion_checked=len(converted), converted_min=min(converted), converted_max=max(converted),
            source_arrival_verified=False, admitted_to_r0=False))
    report = dict(scope='LOCAL_SIX_CASE_PRICE_DIAGNOSTICS', cases=6, results=results,
        actual_journal=journal, legacy_sector_panels=panels, full_two_stage_replay_ready=False,
        live_gate_result=None, training_ready=False,
        blockers=['MINUTE_START_END_SEMANTICS_UNCERTIFIED', 'ACTUAL_SOURCE_ARRIVAL_UNKNOWN',
                  'NO_ACTUAL_QUOTES_IN_CHECKED_JOURNAL', 'SSE_15M_NOT_BOUND',
                  'SECTOR_KNOWN_AT_AND_FULL_MEMBER_PRICES_NOT_BOUND', 'PRIOR_LEADER_LIST_NOT_BOUND'],
        exclusions='No divergence, Fibonacci edge, sector stalling, leader resealing or trading efficacy certification.')
    for path, digest in refs.items():
        if sha(Path(path)) != digest:
            raise ValueError('INPUT_CHANGED_DURING_RUN')
    payloads = {'diagnostics.json': encoded(report)}
    manifest = dict(inputs=[dict(path=p, sha256=h) for p, h in sorted(refs.items())],
        outputs=[dict(path=name, bytes=len(body)) for name, body in payloads.items()],
        source_mutations=False, fetch=False, fit=False)
    payloads['manifest.json'] = encoded(manifest)
    total = sum(map(len, payloads.values()))
    if total > CAP or bytes_used() + total > TOTAL_CAP:
        raise ValueError('WRITE_BUDGET')
    OUT.mkdir(exist_ok=False)
    for name, body in payloads.items():
        with (OUT / name).open('xb') as stream:
            stream.write(body)
    print(json.dumps(dict(output=str(OUT), bytes=total, cases=len(results),
        scenarios=12, prefix_cutoffs=192, training_ready=False)))


if __name__ == '__main__':
    run()
