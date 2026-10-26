"""Offline frozen-label association test, NOT an executable profit backtest or fit."""
from collections import Counter, defaultdict
import gzip
import hashlib
import io
import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from candidate_resonance_rules_v1 import VERSION, assess
from historical_risk_core_v1 import valid_bar
from market_index_source_v1 import bind_from_experiment_config
from research_budget_v2 import bytes_used, TOTAL_CAP
from stockdb_adjustment_v2 import rebuild_mult_factors

HERE = Path(__file__).resolve().parent
OUT = HERE / 'candidate_resonance_test_20261016_v1'
CAP = 16 * 1024**2


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def market_records(payload):
    node = payload['data']['sh000300']
    if 'day' not in node or any(k in node for k in ('qfqday', 'hfqday')):
        raise ValueError('INDEX_NOT_RAW_DAY')
    identity = node.get('qt', {}).get('sh000300', [])
    if len(identity) < 3 or identity[2] != '000300' or '沪深300' not in identity[1]:
        raise ValueError('INDEX_IDENTITY')
    records = {}
    for r in node['day']:
        date = int(r[0].replace('-', ''))
        bar = dict(zip(('open', 'close', 'high', 'low'), map(float, r[1:5])))
        if not valid_bar(bar):
            raise ValueError('INDEX_OHLC_INVALID')
        if date in records and records[date] != bar:
            raise ValueError('INDEX_DATE_CONFLICT')
        records[date] = bar
    return records


def source_state(evidence, calendar, i):
    p = calendar[i]
    start, volume_start = calendar[max(0, i - 68)], calendar[max(0, i - 18)]
    if any(r['date'] is None or start <= r['date'] <= p for r in evidence['issues']):
        return 'SOURCE_EVENT_PENDING'
    if any(volume_start <= r['date'] <= p for r in evidence['events']):
        return 'VOLUME_ACTION_WINDOW_PENDING'
    return None


def evidence_factors(evidence, calendar):
    for event in evidence['events']:
        if event['multiplier'] is None and not any(
                r['date'] is None or r['date'] == event['date'] for r in evidence['issues']):
            raise ValueError('UNRESOLVED_EVENT_WITHOUT_ISOLATION')
    return rebuild_mult_factors([dict(date=r['date'], mult=r['multiplier'])
                                 for r in evidence['events'] if r['multiplier'] is not None], calendar)


def run():
    if OUT.exists():
        raise ValueError('OUTPUT_EXISTS_NO_RESTART')
    if bytes_used() + CAP > TOTAL_CAP:
        raise ValueError('REGISTERED_BUDGET')
    refs = {}

    def track(p, expected=None):
        p = Path(p).resolve()
        h = sha(p)
        if expected is not None and h != expected:
            raise ValueError('INPUT_HASH_DRIFT:' + str(p))
        refs[str(p)] = h
        return p

    for n in ('candidate_resonance_rules_v1.py', 'test_candidate_resonance_rules_v1.py',
              'test_candidate_resonance_history_v1.py', 'test_candidate_resonance_history_parser_v1.py',
              'contracts/R0_CANDIDATE_RULE_TEST_V1.md', 'volume_exhaustion_exploration_v1.py',
              'historical_risk_core_v1.py', 'stockdb_adjustment_v2.py', 'research_budget_v2.py',
              'market_index_source_v1.py', 'contracts/historical_input_overlay_20261015.json'):
        track(HERE / n)
    acceptance = json.loads(track(HERE / 'archived_corpus_v4_acceptance_20261015.json').read_bytes())
    for r in acceptance['controls']:
        track(r['path'], r['sha256'])
    base = []
    for ref in acceptance['matrix_files']:
        base.extend(pq.read_table(track(ref['path'], ref['sha256'])).to_pylist())
    if len(base) != acceptance['persisted_rows'] or len({(r['code'], r['P']) for r in base}) != len(base):
        raise ValueError('BASE_POPULATION_DRIFT')
    calendar = json.loads(track(HERE / 'historical_stock_day_preparation_v1/manifest.json').read_bytes())['calendar_dates']
    positions = {d: i for i, d in enumerate(calendar)}
    bound_closes = bind_from_experiment_config()
    overlay = json.loads((HERE / 'contracts/historical_input_overlay_20261015.json').read_bytes())
    index_csv = track(overlay['market_source']['bound_input'], overlay['market_source']['sha256'])
    track(index_csv.with_name(index_csv.name + '.binding.json'), overlay['market_source']['binding_sha256'])
    index = {}
    for blob in sorted(index_csv.parent.rglob('*.bin')):
        sidecar = track(blob.with_suffix('.json'))
        meta = json.loads(sidecar.read_bytes())
        track(blob, meta['sha256'])
        if blob.stat().st_size != meta['bytes'] or meta['symbol'] != 'sh000300' or meta['outcome'] != 'ok':
            raise ValueError('INDEX_ARCHIVE_IDENTITY')
        for day, bar in market_records(json.loads(blob.read_bytes())).items():
            if day in index and index[day] != bar:
                raise ValueError('INDEX_OVERLAP_CONFLICT')
            index[day] = bar
    if not index or any(day not in index or index[day]['close'] != close for day, close in bound_closes.items()):
        raise ValueError('INDEX_CLOSE_PARITY_FAIL')
    # Fifth column is a non-input placeholder; index volume is NOT used or claimed observed.
    market = np.array([[index.get(d, {}).get(k, np.nan) for k in ('open', 'high', 'low', 'close')] + [1.]
                       for d in calendar])
    corpus = HERE / 'historical_archived_corpus_v4_20261015'
    old = HERE / 'historical_archived_corpus_v3_20261015'
    stops = {p: json.loads(track(p / 'stop_record.json').read_bytes()) for p in (corpus, old)}
    evidence = json.loads(track(corpus / 'evidence_and_gate.json').read_bytes())['audit']
    by_code = defaultdict(list)
    for row in base:
        by_code[row['code']].append(row)
    states, candidates, features, cohorts, perturbations = Counter(), [], [], defaultdict(list), []
    prefix_count = 0
    for code, rows in sorted(by_code.items()):
        path = corpus / f'{code}_daily.json.gz'
        if not path.exists():
            path = old / path.name
        raw = gzip.decompress(track(path).read_bytes())
        entry = next(r for r in stops[path.parent]['request_ledger'] if r['code'] == code and r['table'] == '日k')
        if len(raw) != entry['bytes'] or hashlib.sha256(raw).hexdigest() != entry['sha256']:
            raise ValueError('RAW_DAILY_HASH_DRIFT')
        bars = {}
        for key, value in json.loads(raw):
            d = int(key.rsplit(':', 1)[1])
            if value is None:
                continue
            if value.get('code') != code or value.get('date') != d or not key.startswith(f'日k:{code}:'):
                raise ValueError('DAILY_IDENTITY')
            v = [value.get(k, np.nan) for k in ('open', 'high', 'low', 'close', 'volume')]
            if d in bars and bars[d] != v:
                raise ValueError('DAILY_CONFLICT')
            bars[d] = v
        ev = evidence[code]
        # evidence.events contains all audit outcomes, not just admitted factor events.
        # Unresolved rows remain in issues and block their dependency windows below.
        factor = evidence_factors(ev, calendar)
        stock = np.array([bars.get(d, [np.nan] * 5) for d in calendar], dtype=float)
        stock[:, :4] *= np.array([factor[d] for d in calendar])[:, None]
        source_counts = Counter()
        for row in sorted(rows, key=lambda r: r['P']):
            p, i = row['P'], positions[row['P']]
            blocked = source_state(ev, calendar, i)
            if blocked is not None:
                decision = dict(state=blocked, candidate=None)
            else:
                decision = assess(stock, market, i)
                if decision != assess(stock[:i + 1], market[:i + 1], i):
                    raise ValueError('FULL_PREFIX_MISMATCH')
                prefix_count += 1
            source_counts[decision['state']] += 1
            states[decision['state']] += 1
            component = decision.get('components', {})
            features.append(dict(code=code, P=p, state=decision['state'],
                candidate=decision['candidate'], market_ret20=decision.get('market_ret20'),
                market_nearest_distance=(decision.get('market_location') or {}).get('distance'),
                stock_preceding_rise=decision.get('stock_context', {}).get('preceding_rise'),
                stock_retracement=decision.get('stock_context', {}).get('retracement'),
                shock_volume_ratio=decision.get('shock_volume_ratio'),
                shrink_volume_ratio=decision.get('shrink_volume_ratio'),
                market_environment_pass=component.get('market_environment'),
                market_position_pass=component.get('market_position'),
                stock_position_pass=component.get('stock_position'),
                volume_pass=component.get('volume'), confirmation_pass=component.get('confirmation')))
            if row['training_eligible'] is True and decision['candidate'] is not None:
                cohorts[p].append((row, decision))
                if decision['candidate']:
                    candidates.append(dict(code=code, P=p, segment=row['segment'],
                        original_mae_5d=row['mae_5d'], original_y_tail_5d=row['y_tail_5d'], decision=decision))
        # Fixed date; no substitution if source/price missing. Pure price-prefix perturbation only.
        i = positions[20250102]
        before = assess(stock, market, i)
        if before['state'] == 'DATA_INSUFFICIENT':
            perturbations.append(dict(code=code, P=20250102, state='NOT_TESTED_DATA_MISSING'))
        else:
            x, m = stock.copy(), market.copy()
            x[i + 1:] *= 3; m[i + 1:] *= .2
            if before != assess(x, m, i):
                raise ValueError('FUTURE_PERTURBATION_MISMATCH')
            perturbations.append(dict(code=code, P=20250102, state='PASS_PRICE_PREFIX_ONLY'))
        print(f'{code}: {len(rows)} rows, states={dict(source_counts)}', flush=True)
    n, positives, atr_positives, random_expected = 0, 0, 0, 0.
    all_cohort = []
    for date, group in sorted(cohorts.items()):
        all_cohort.extend(r for r, _ in group)
        selected = [r for r, d in group if d['candidate']]
        k = len(selected)
        n += k
        positives += sum(r['y_tail_5d'] for r in selected)
        atr = sorted((r for r, _ in group), key=lambda r: (r['features']['atr14_pct'], r['code']))[:k]
        atr_positives += sum(r['y_tail_5d'] for r in atr)
        random_expected += k * sum(r['y_tail_5d'] for r, _ in group) / len(group)
    result = dict(status='RULE_ENGINE_TEST_COMPLETE_NOT_FINANCIAL_GATE', version=VERSION,
        market_symbol='SH000300', intraday_sse_case_replayed=False, market_volume_used=False,
        matrix_population=len(base), original_eligible=acceptance['eligible'],
        shared_source_evaluable=len(all_cohort), states=dict(states),
        actual_prefix_comparisons=prefix_count,
        source_pending_not_numeric_tested=len(base) - prefix_count,
        future_perturbations=perturbations, index_close_parity_dates=len(bound_closes),
        candidate_count=n, candidate_dates=len({r['P'] for r in candidates}),
        candidate_unique_codes=len({r['code'] for r in candidates}),
        historical_label_association=dict(rule_tail_count=positives,
            rule_tail_rate=positives / n if n else None,
            same_day_same_count_atr_tail_count=atr_positives,
            same_day_same_count_atr_tail_rate=atr_positives / n if n else None,
            same_day_same_count_random_expected_tail_rate=random_expected / n if n else None),
        by_year={str(y): dict(candidates=sum(r['P'] // 10000 == y for r in candidates),
                              tail=sum(r['original_y_tail_5d'] for r in candidates if r['P'] // 10000 == y))
                 for y in range(2019, 2027)},
        risk_population_not_filtered=True, candidates=candidates,
        formal_pnl_generated=False, labels_rebuilt=False, fits=0, new_requests=0,
        financial_gate_passed=False, fit_ready=False,
        limitations=['43-stock archive selection/survivorship and historical-availability assumptions',
            'no sector/leader or 15-minute context constructed',
            'P-close risk labels are NOT next-open entry strategy returns',
            'no causal or significance inference and no grid placebo/ablation yet',
            'PASS means code tests only; no parameter optimization or deployment'])
    arrays = [('code', pa.string()), ('P', pa.int64()), ('state', pa.string()), ('candidate', pa.bool_())]
    arrays += [(k, pa.float64()) for k in ('market_ret20', 'market_nearest_distance', 'stock_preceding_rise',
                                          'stock_retracement', 'shock_volume_ratio', 'shrink_volume_ratio')]
    arrays += [(k, pa.bool_()) for k in ('market_environment_pass', 'market_position_pass',
                                       'stock_position_pass', 'volume_pass', 'confirmation_pass')]
    sink = io.BytesIO()
    pq.write_table(pa.Table.from_pylist(features, schema=pa.schema(arrays)), sink, compression='zstd')
    manifest = dict(version=VERSION, inputs=[dict(path=p, sha256=h) for p, h in sorted(refs.items())],
                    note='Technical snapshot of already fixed local inputs/code; not FIT authorization')
    blobs = {'rule_states.parquet': sink.getvalue(),
             'report.json': json.dumps(result, ensure_ascii=False, allow_nan=False, indent=1).encode('utf8'),
             'manifest.json': json.dumps(manifest, ensure_ascii=False, indent=1).encode('utf8')}
    for p, h in refs.items():
        if sha(p) != h:
            raise ValueError('INPUT_CHANGED_DURING_BUILD')
    if sum(map(len, blobs.values())) + 65536 > CAP or bytes_used() + sum(map(len, blobs.values())) + 65536 > TOTAL_CAP:
        raise ValueError('RULE_TEST_OUTPUT_BUDGET')
    OUT.mkdir(exist_ok=False)
    for name, data in blobs.items():
        with (OUT / name).open('xb') as f:
            f.write(data)
    print(json.dumps({k: v for k, v in result.items() if k not in ('candidates', 'future_perturbations')}, ensure_ascii=False))


if __name__ == '__main__':
    run()
