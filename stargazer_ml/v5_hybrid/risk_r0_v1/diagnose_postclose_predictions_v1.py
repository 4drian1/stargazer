"""Frozen-prediction diagnostics only: no model or preprocessing fitting."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from audit_postclose_first_fit_v1 import loss, verify_rows
from inventory_ths_stockdb_cache_v1 import HERE, digest
from postclose_training_v1 import CAP, check
from research_budget_v2 import TOTAL_CAP, bytes_used

FIT = HERE / 'postclose_first_research_fit_20261017_v1'
SPEC = HERE / 'contracts/POSTCLOSE_FIXED_PREDICTION_DIAGNOSTICS_V1.json'
OUT = FIT / 'diagnostics_v1'
TARGET = 'next_session_adverse_excursion'


def summary(frame, constant):
    if frame.empty:
        return dict(rows=0, dates=0, stocks=0, mean_adverse=None, tail_rate=None,
                    coverage=None, pinball=None, constant_pinball=None)
    y = frame[TARGET].to_numpy()
    return dict(rows=len(frame), dates=int(frame.day.nunique()), stocks=int(frame.code.nunique()),
                mean_adverse=float(y.mean()), tail_rate=float((y > constant).mean()),
                coverage=float((y <= frame.q90_adverse).mean()),
                pinball=float(loss(y, frame.q90_adverse).mean()),
                constant_pinball=float(loss(y, constant).mean()))


def strata(frame, edges, constant):
    bins = np.searchsorted(np.asarray(edges), frame.q90_adverse.to_numpy(), side='left')
    return [dict(bin=i + 1, **summary(frame.iloc[np.flatnonzero(bins == i)], constant))
            for i in range(5)]


def filter_by_day(frame, constant):
    columns = ['q90_adverse', 'q90_linear', 'stock_return1d']
    valid = np.isfinite(frame[columns].to_numpy(dtype=float)).all(axis=1)
    candidates = frame.loc[valid].copy()
    counts = candidates.groupby('day').size()
    candidates = candidates[candidates.day.isin(counts[counts >= 5].index)]
    daily = []
    for day, group in candidates.groupby('day', sort=True):
        group = group.copy()
        group['tie'] = [digest(f'{code}|{day}|fixed-diagnostic-tie-v1'.encode()) for code in group.code]
        n = len(group)
        keep = n - int(np.floor(.2 * n))
        row = dict(day=int(day), available=n, kept=keep)
        rankings = dict(model=('q90_adverse', False), linear=('q90_linear', False),
                        simple_weakness=('stock_return1d', True))
        for method, (column, ascending) in rankings.items():
            ordered = group.sort_values([column, 'tie'], ascending=[ascending, True])
            retained = ordered.iloc[n - keep:]
            row[method] = dict(adverse_sum=float(retained[TARGET].sum()),
                               tail_sum=float((retained[TARGET] > constant).sum()))
        row['random_expectation'] = dict(adverse_sum=float(group[TARGET].mean() * keep),
                                         tail_sum=float((group[TARGET] > constant).mean() * keep))
        row['unfiltered'] = dict(adverse_sum=float(group[TARGET].sum()),
                                 tail_sum=float((group[TARGET] > constant).sum()))
        daily.append(row)
    return daily, dict(total_test_rows=len(frame), finite_intersection_rows=int(valid.sum()),
                       comparable_rows=len(candidates), comparable_dates=len(daily),
                       comparable_stocks=int(candidates.code.nunique()),
                       coverage_vs_test=float(len(candidates) / len(frame)) if len(frame) else None)


def paired_interval(daily, comparator, metric):
    if not daily:
        return None
    values = np.array([[r['model'][metric] - r[comparator][metric], r['kept']] for r in daily])
    blocks = np.array([values[i:i + 10].sum(axis=0) for i in range(0, len(values), 10)])
    rng = np.random.default_rng(20261017)
    draws = blocks[rng.integers(0, len(blocks), size=(1000, len(blocks)))].sum(axis=1)
    return dict(difference=float(values[:, 0].sum() / values[:, 1].sum()),
                interval_95=np.quantile(draws[:, 0] / draws[:, 1], [.025, .975]).tolist(),
                date_blocks=len(blocks), draws=1000,
                interpretation='DESCRIPTIVE_POST_HOC_NOT_FINANCIAL_GATE')


def filter_summary(daily):
    if not daily:
        return dict(status='NOT_EVALUABLE', methods={}, differences={})
    kept = sum(r['kept'] for r in daily)
    available = sum(r['available'] for r in daily)
    methods = {}
    for method in ['model', 'linear', 'simple_weakness', 'random_expectation', 'unfiltered']:
        denominator = available if method == 'unfiltered' else kept
        methods[method] = dict(denominator=denominator,
            mean_adverse=sum(r[method]['adverse_sum'] for r in daily) / denominator,
            tail_rate=sum(r[method]['tail_sum'] for r in daily) / denominator)
    differences = {method: {metric: paired_interval(daily, method, metric)
                           for metric in ['adverse_sum', 'tail_sum']}
                   for method in ['linear', 'simple_weakness', 'random_expectation']}
    return dict(status='DESCRIPTIVE_COMPARISON', actual_retained_fraction=kept / available,
                methods=methods, differences=differences)


def run():
    if OUT.exists():
        raise ValueError('OUTPUT_EXISTS_NO_OVERWRITE')
    spec = json.loads(SPEC.read_bytes())
    if spec['remove_fraction'] != .2 or spec['remove_rounding'] != 'floor(0.2*n) per day; identical retained counts for every method':
        raise ValueError('METHOD_SPEC_DRIFT')
    prior = json.loads((FIT / 'independent_readback_audit.json').read_bytes())
    refs = {str(FIT / name): sha for name, sha in prior['audited_original_sha256'].items()}
    refs.update({str(SPEC): digest(SPEC.read_bytes()), str(Path(__file__)): digest(Path(__file__).read_bytes()),
                 str(FIT / 'independent_readback_audit.json'): digest((FIT / 'independent_readback_audit.json').read_bytes())})
    for path, sha in refs.items():
        if digest(Path(path).read_bytes()) != sha:
            raise ValueError('INPUT_DRIFT:' + path)
    _, _, _, parts = check()
    predictions = pd.read_parquet(FIT / 'predictions.parquet')
    population = pd.concat([f.assign(segment=s) for s, f in parts.items()], ignore_index=True)
    verify_rows(predictions, population)
    data = predictions.merge(population[['code', 'day', 'stock_return1d', 'sse_lag1_price_return']],
                             on=['code', 'day'], how='left', validate='one_to_one')
    if not np.isfinite(data[[TARGET, 'q90_adverse', 'q90_linear']].to_numpy()).all():
        raise ValueError('INVALID_PREDICTIONS')
    OUT.mkdir(exist_ok=False)
    def write(name, value, terminal=False):
        body = json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2).encode('utf-8')
        reserve = 0 if terminal else 65536
        current = sum(p.stat().st_size for p in FIT.rglob('*') if p.is_file())
        if current + len(body) + reserve > CAP or bytes_used() + len(body) + reserve > TOTAL_CAP:
            raise ValueError('DIAGNOSTIC_BUDGET_STOP')
        with (OUT / name).open('xb') as stream:
            stream.write(body)
    write('plan.json', dict(inputs=refs, method=spec))
    try:
        train = data[data.segment.eq('train')]
        test = data[data.segment.eq('test')].copy()
        constant = float(np.quantile(train[TARGET], .9))
        edges = np.quantile(train.q90_adverse, [.2, .4, .6, .8]).tolist()
        test['month'] = test.day.astype(str).str[:6]
        r = test.sse_lag1_price_return
        test['market_context'] = np.where(r.isna(), 'unknown', np.where(r < 0, 'negative', 'nonnegative'))
        monthly = {str(k): summary(g, constant) for k, g in test.groupby('month', sort=True)}
        contexts = {key: summary(test[test.market_context.eq(key)], constant)
                    for key in ['negative', 'nonnegative', 'unknown']}
        bins = {s: strata(data[data.segment.eq(s)], edges, constant) for s in parts}
        daily, coverage = filter_by_day(test, constant)
        report = dict(status='COMPLETE_FIXED_PREDICTION_DIAGNOSTICS_NOT_DEPLOYMENT',
            scope='POST_HOC_RESTRICTED_HISTORICAL_RECONSTRUCTION', train_tail_threshold=constant,
            train_score_quintile_edges=edges, test_monthly=monthly, test_market_background=contexts,
            fixed_strata=bins, same_day_coverage=coverage, same_day_filter=filter_summary(daily),
            cohort_composition={s: dict(stocks=int(f.code.nunique()),
                absent_from_training=sorted(set(f.code) - set(parts['train'].code))) for s, f in parts.items()},
            fit_run=False, test_tuning=False, deployment_authorized=False)
        for path, sha in refs.items():
            if digest(Path(path).read_bytes()) != sha:
                raise ValueError('INPUT_CHANGED_DURING_DIAGNOSTICS:' + path)
        write('daily_comparison.json', daily)
        write('report.json', report)
        write('stop_record.json', dict(status='COMPLETE', original_inputs_unchanged=True), terminal=True)
        return report
    except Exception as exc:
        write('stop_record.json', dict(status='FAILED', error=f'{type(exc).__name__}:{exc}'), terminal=True)
        raise


if __name__ == '__main__':
    print(json.dumps(run(), ensure_ascii=False, allow_nan=False, indent=2))
