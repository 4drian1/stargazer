"""Independent arithmetic readback and missing-month attribution; no model fit."""
import json
import numpy as np
import pandas as pd
from diagnose_postclose_predictions_v1 import FIT, OUT, TARGET
from finalize_postclose_cohort_v3 import read_population, corrected_masks
from inventory_ths_stockdb_cache_v1 import digest
from postclose_training_v1 import CAP
from research_budget_v2 import TOTAL_CAP, bytes_used


def run():
    plan = json.loads((OUT / 'plan.json').read_bytes())
    for path, sha in plan['inputs'].items():
        from pathlib import Path
        if digest(Path(path).read_bytes()) != sha:
            raise ValueError('DIAGNOSTIC_INPUT_DRIFT:' + path)
    report = json.loads((OUT / 'report.json').read_bytes())
    daily = json.loads((OUT / 'daily_comparison.json').read_bytes())
    predictions = pd.read_parquet(FIT / 'predictions.parquet')
    population, _ = read_population()
    test = predictions[predictions.segment.eq('test')].merge(
        population[['code', 'day', 'stock_return1d']], on=['code', 'day'], validate='one_to_one')
    threshold = float(np.quantile(predictions.loc[predictions.segment.eq('train'), TARGET], .9))
    np.testing.assert_allclose(report['train_tail_threshold'], threshold, rtol=0, atol=1e-14)
    expected = test[np.isfinite(test[['q90_adverse', 'q90_linear', 'stock_return1d']]).all(axis=1)]
    sizes = expected.groupby('day').size()
    expected = expected[expected.day.isin(sizes[sizes >= 5].index)]
    if len(expected) != report['same_day_coverage']['comparable_rows']:
        raise ValueError('COVERAGE_DRIFT')
    if set(expected.day) != {row['day'] for row in daily}:
        raise ValueError('DATE_DRIFT')
    for row in daily:
        group = expected[expected.day.eq(row['day'])].copy()
        group['tie'] = [digest(f'{c}|{row["day"]}|fixed-diagnostic-tie-v1'.encode()) for c in group.code]
        count = len(group) - len(group) // 5
        if row['kept'] != count or row['available'] != len(group):
            raise ValueError('KEEP_COUNT_DRIFT')
        for method, column, ascending in [('model', 'q90_adverse', False),
                                          ('linear', 'q90_linear', False),
                                          ('simple_weakness', 'stock_return1d', True)]:
            kept = group.sort_values([column, 'tie'], ascending=[ascending, True]).tail(count)
            np.testing.assert_allclose(row[method]['adverse_sum'], kept[TARGET].sum(), rtol=0, atol=1e-12)
            np.testing.assert_allclose(row[method]['tail_sum'], (kept[TARGET] > threshold).sum(), rtol=0, atol=1e-12)
        for metric, values in [('adverse_sum', group[TARGET]), ('tail_sum', group[TARGET] > threshold)]:
            np.testing.assert_allclose(row['random_expectation'][metric], values.sum() * count / len(group), rtol=0, atol=1e-12)
    for method, facts in report['same_day_filter']['methods'].items():
        denominator = sum(row['available' if method == 'unfiltered' else 'kept'] for row in daily)
        for metric, field in [('adverse_sum', 'mean_adverse'), ('tail_sum', 'tail_rate')]:
            np.testing.assert_allclose(facts[field], sum(row[method][metric] for row in daily) / denominator, rtol=0, atol=1e-12)
    nominal = population[population.proposed_segment_name.eq('test')].copy()
    nominal['month'] = nominal.day.astype(str).str[:6]
    months = {}
    for month, group in nominal.groupby('month', sort=True):
        masks = corrected_masks(group)
        months[str(month)] = dict(population_rows=len(group), population_dates=int(group.day.nunique()),
            cohort_rows=int(group.research_cohort_eligible.sum()),
            complete_label_rows=int(group.label_state.eq('CONDITIONAL_NEXT_SESSION_COMPLETE').sum()),
            family_available={key: int(value.sum()) for key, value in masks.items()},
            reference_close_alignment_rows=int(group.context_close_aligned.sum()))
    hashes = {name: digest((OUT / name).read_bytes())
              for name in ['plan.json', 'report.json', 'daily_comparison.json', 'stop_record.json']}
    result = dict(status='PASS_SAME_DAY_MEMBERSHIP_COUNTS_AND_ARITHMETIC',
        original_fit_inputs_unchanged=True, audited_sha256=hashes,
        nominal_test_month_coverage=months, fit_run=False,
        limitations='Monthly missing-family counts can overlap; they are not disjoint exclusion counts.')
    body = json.dumps(result, ensure_ascii=False, allow_nan=False, indent=2).encode('utf-8')
    if sum(p.stat().st_size for p in FIT.rglob('*') if p.is_file()) + len(body) > CAP or bytes_used() + len(body) > TOTAL_CAP:
        raise ValueError('AUDIT_BUDGET_STOP')
    with (OUT / 'readback_audit.json').open('xb') as stream:
        stream.write(body)
    return result


if __name__ == '__main__':
    print(json.dumps(run(), ensure_ascii=False, indent=2))
