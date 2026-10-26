"""Frozen-prediction diagnostics, not feature selection, new labels, or a fit."""
import argparse
import csv
import io
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score

import prepare_multiboard_comparison_v1 as prep
from build_decision_after_labels_v1 import SOURCE, LEGACY
from build_joint_risk_features_v1 import QUARANTINE
from decision_after_labels_v1 import minute_clock, price_grid
from research_budget_v2 import bytes_used, TOTAL_CAP

HERE = prep.HERE
FIT = HERE / 'multiboard_three_arm_fit_20261018_v1'
OUT = HERE / 'multiboard_transfer_diagnosis_20261018_v1'
CAP, RESERVE = 4 * 1024**2, 64 * 1024
MODELS = ['B1_lr', 'M1_lr', 'B1_lightgbm', 'M1_lightgbm']


def auc(y, score, weight=None):
    return float(roc_auc_score(y, score, sample_weight=weight)) if len(np.unique(y)) == 2 else None


def rank_summary(frame, score):
    y = frame.y_tail_1200m.to_numpy(dtype=int)
    score = np.asarray(score, dtype=float)
    if len(y) != len(score) or not np.isfinite(score).all() or not np.isin(y, [0, 1]).all():
        raise ValueError('INVALID_RANK_DIAGNOSTIC')
    if not len(y):
        return dict(rows=0, status='NOT_TESTED_EMPTY_POPULATION', pooled_auc=None,
                    date_equal_weight_auc=None, stock_equal_weight_auc=None,
                    same_cutoff_pair_weighted_auc=None)
    day = frame[prep.TIME] // 1000000
    day_weight = 1 / day.map(day.value_counts()).to_numpy(dtype=float)
    stock_weight = 1 / frame.code.map(frame.code.value_counts()).to_numpy(dtype=float)
    pair_sum, pairs, mixed = 0., 0, 0
    for indices in frame.reset_index(drop=True).groupby(prep.TIME, sort=False).indices.values():
        labels = y[indices]
        pos = int(labels.sum())
        neg = len(labels) - pos
        if pos and neg:
            numerator = rankdata(score[indices])[labels == 1].sum() - pos * (pos + 1) / 2
            pair_sum += numerator
            pairs += pos * neg
            mixed += 1
    return dict(rows=len(frame), dates=int(day.nunique()), stocks=int(frame.code.nunique()),
        actual_tail_rate=float(y.mean()), pooled_auc=auc(y, score),
        date_equal_weight_auc=auc(y, score, day_weight), stock_equal_weight_auc=auc(y, score, stock_weight),
        same_cutoff_pair_weighted_auc=float(pair_sum / pairs) if pairs else None,
        comparable_positive_negative_pairs=pairs, mixed_label_cutoffs=mixed,
        score_mean_positive=float(score[y == 1].mean()) if (y == 1).any() else None,
        score_mean_negative=float(score[y == 0].mean()) if (y == 0).any() else None)


def shift(train, other, fields):
    result = {}
    for name in fields:
        a, b = train[name].to_numpy(dtype=float), other[name].to_numpy(dtype=float)
        if np.isinf(a).any() or np.isinf(b).any():
            raise ValueError('INFINITE_SHIFT_INPUT')
        av, bv = a[np.isfinite(a)], b[np.isfinite(b)]
        std = float(av.std()) if len(av) else 0.
        result[name] = dict(train_finite=len(av), segment_finite=len(bv),
            train_missing=float(np.isnan(a).mean()), segment_missing=float(np.isnan(b).mean()),
            mean_shift_train_sd=float((bv.mean() - av.mean()) / std) if len(av) and len(bv) and std > 0 else None,
            outside_train_range=float(((bv < av.min()) | (bv > av.max())).mean()) if len(av) and len(bv) else None,
            train_min=float(av.min()) if len(av) else None, train_max=float(av.max()) if len(av) else None,
            segment_quantiles=np.quantile(bv, [.1, .5, .9]).tolist() if len(bv) else None)
    return result


def logit_terms(frame, preprocessing, linear):
    fields = preprocessing['columns']
    values = frame[fields].to_numpy(dtype=float)
    if np.isinf(values).any():
        raise ValueError('INFINITE_EXPLANATION_INPUT')
    median, mean, scale = (np.asarray(preprocessing[k], dtype=float) for k in ['median', 'mean', 'scale'])
    coefficient = np.asarray(linear['coef'], dtype=float)
    if len(coefficient) != len(fields) * 2 or (scale <= 0).any():
        raise ValueError('EXPLANATION_SHAPE_OR_SCALE')
    missing = np.isnan(values)
    numeric = ((np.where(missing, median, values) - mean) / scale) * coefficient[:len(fields)]
    absent = missing * coefficient[len(fields):]
    score = numeric.sum(1) + absent.sum(1) + linear['intercept']
    terms = {f: dict(numeric_mean_logit=float(numeric[:, i].mean()),
                     missing_mean_logit=float(absent[:, i].mean()),
                     mean_absolute_logit=float(np.abs(numeric[:, i] + absent[:, i]).mean()))
             for i, f in enumerate(fields)}
    return score, terms


def signal_unit(frame, field):
    if field.startswith('mb'):
        # A shared signal copied into twenty candidate rows is one observation.
        values = frame.groupby(prep.TIME, sort=False)[field].nunique(dropna=False)
        if (values > 1).any():
            raise ValueError('SHARED_SIGNAL_CONFLICT')
        return frame.drop_duplicates(prep.TIME), 'unique_decision_cutoff'
    return frame, 'candidate_by_cutoff'


def signal_summary(frame, field):
    unit, identity = signal_unit(frame, field)
    values = unit[field].to_numpy(dtype=float)
    if np.isinf(values).any() or (values[np.isfinite(values)] < 0).any() or (values[np.isfinite(values)] > 1).any():
        raise ValueError('SIGNAL_RATE_OUTSIDE_RANGE')
    masks = {'unknown': ~np.isfinite(values), 'known_zero': values == 0, 'active': values > 0}
    # Conditional event rates below are candidate-weighted, not duplicated global-signal counts.
    raw = frame[field].to_numpy(dtype=float)
    candidate_masks = {'unknown': ~np.isfinite(raw), 'known_zero': raw == 0, 'active': raw > 0}
    states = {}
    for state, mask in masks.items():
        candidates = frame.loc[candidate_masks[state]]
        states[state] = dict(observations=int(mask.sum()), candidate_rows=len(candidates),
            cutoffs=int(candidates[prep.TIME].nunique()), dates=int((candidates[prep.TIME] // 1000000).nunique()),
            tail_events=int(candidates.y_tail_1200m.sum()),
            tail_rate=float(candidates.y_tail_1200m.mean()) if len(candidates) else None)
    return dict(unit=identity, observations=len(unit), states=states,
                conditional_rates_are_descriptive_not_independent_or_causal=True)


def positive_offsets(low, reference, indices, expected_mae, horizon=1200):
    """Outcome-only; no new threshold/horizon labels and never a model feature."""
    if not (len(indices) == len(expected_mae)) or horizon < 1:
        raise ValueError('INVALID_OUTCOME_REQUEST')
    first, minima = [], []
    for t, expected in zip(indices, expected_mae):
        if t < 0 or t + horizon >= len(low):
            raise ValueError('OUTCOME_WINDOW_TRUNCATED')
        anchor = reference[t]
        future = low[t+1:t+horizon+1]
        if not np.isfinite(future).all() or not np.isfinite(anchor) or anchor <= 0 or (future <= 0).any():
            raise ValueError('MISSING_OR_INVALID_OUTCOME_PRICE')
        returns = future / anchor - 1
        if not np.isfinite(expected) or not np.isclose(returns.min(), expected, atol=1e-12, rtol=0):
            raise ValueError('EXISTING_MAE_NOT_REPRODUCED')
        hit = np.flatnonzero(returns <= -.08 + 1e-12)
        if not len(hit):
            raise ValueError('EXISTING_POSITIVE_NOT_REPRODUCED')
        first.append(int(hit[0]) + 1)
        minima.append(int(np.argmin(returns)) + 1)
    return np.asarray(first, dtype=np.int32), np.asarray(minima, dtype=np.int32)


def duration_summary(frame):
    values = frame.first_touch_trading_minute.to_numpy()
    return dict(existing_positive_records=len(frame), dates=int((frame[prep.TIME] // 1000000).nunique()),
        stocks=int(frame.code.nunique()), first_touch_quantiles=np.quantile(values, [.1, .5, .9]).tolist() if len(values) else None,
        cumulative={str(t): dict(records=int((values <= t).sum()),
                    share=float((values <= t).mean()) if len(values) else None) for t in [15, 60, 240, 480, 1200]},
        overlapping_positive_windows_not_independent_events=True)


def inputs():
    accepted_path = HERE / 'multiboard_three_arm_fit_acceptance_20261018_v1.json'
    accepted = json.loads(accepted_path.read_bytes())
    if accepted['status'] != 'PASS_FIXED_RESEARCH_FIT_SERIALIZATION_AND_RISK_READBACK':
        raise ValueError('ACCEPTED_FIXED_FIT_REQUIRED')
    plan = json.loads((FIT / 'plan.json').read_bytes())
    terminal = json.loads((FIT / 'stop_record.json').read_bytes())
    manifest = json.loads((FIT / 'artifact_manifest.json').read_bytes())
    if terminal['status'] != 'COMPLETE_RESEARCH_NOT_DEPLOYMENT' or prep.sha(FIT / 'artifact_manifest.json') != terminal['manifest_sha256']:
        raise ValueError('FIT_TERMINAL_OR_MANIFEST_DRIFT')
    refs = dict(plan['inputs'])
    for path, expected in refs.items():
        if prep.sha(path) != expected:
            raise ValueError('ORIGINAL_FIT_INPUT_DRIFT:' + path)
    for name, item in manifest['outputs'].items():
        refs[str(FIT / name)] = item['sha256']
    for name in ['stop_record.json', 'artifact_manifest.json']:
        refs[str(FIT / name)] = prep.sha(FIT / name)
    refs[str(accepted_path)] = prep.sha(accepted_path)
    if prep.sha(FIT / 'report.json') != accepted['report_sha256'] or prep.sha(FIT / 'plan.json') != accepted['plan_sha256']:
        raise ValueError('FIT_ACCEPTANCE_BINDING_DRIFT')
    label_manifest = HERE / 'joint_decision_after_labels_20261017_v2/manifest.json'
    label = json.loads(label_manifest.read_bytes())
    refs[str(label_manifest)] = prep.sha(label_manifest)
    for path in [QUARANTINE, LEGACY / 'evidence_and_gate.json']:
        refs[str(path)] = label['inputs'][str(path)]
    calendar = SOURCE / 'shared_market_and_leaders.parquet'
    source_manifest = json.loads((SOURCE / 'manifest.json').read_bytes())
    refs[str(calendar)] = source_manifest['output_hashes'][calendar.name]['sha256']
    refs[str(SOURCE / 'manifest.json')] = prep.sha(SOURCE / 'manifest.json')
    for name in ['diagnose_multiboard_transfer_v1.py', 'test_multiboard_transfer_diagnosis_v1.py',
                 'audit_multiboard_transfer_diagnosis_v1.py', 'research_budget_v2.py', 'risk_fit_eval_v1.py',
                 'decision_after_labels_v1.py', 'build_decision_after_labels_v1.py',
                 'build_joint_risk_features_v1.py', 'read_multiboard_full_features_v1.py']:
        refs[str(HERE / name)] = prep.sha(HERE / name)
    for path, expected in refs.items():
        if prep.sha(path) != expected:
            raise ValueError('FROZEN_INPUT_DRIFT:' + path)
    return plan['config'], label, refs


def outcome_positions(cohort, label, refs):
    calendar = pd.read_parquet(SOURCE / 'shared_market_and_leaders.parquet', columns=[prep.TIME])[prep.TIME].to_numpy()
    days = np.unique(calendar // 1000000)
    clock = minute_clock(days)
    audit = json.loads((LEGACY / 'evidence_and_gate.json').read_bytes())['audit']
    quarantine = {}
    with QUARANTINE.open(encoding='utf-8-sig', newline='') as source:
        for row in csv.DictReader(source):
            quarantine.setdefault(row['code'], set()).add(int(row['day']))
    sources = {row['code']: row for row in label['source_files']}
    positions = []
    for code, group in cohort[cohort.y_tail_1200m.eq(1)].groupby('code', sort=True):
        source = sources[code]
        refs[source['path']] = source['sha256']
        if prep.sha(source['path']) != source['sha256']:
            raise ValueError('FROZEN_CANDIDATE_MINUTES_DRIFT:' + code)
        native = pd.read_parquet(source['path'])
        grid = price_grid(native, code, clock, quarantine.get(code, set()))
        factors, issues = np.ones(len(days)), np.zeros(len(days), bool)
        for event in audit[code]['events']:
            mult = event.get('multiplier')
            if mult is not None:
                if event['status'] != 'ORDINARY_PARAMETERS_CORROBORATED' or not np.isfinite(mult) or mult <= 0:
                    raise ValueError('INVALID_PRIOR_NORMALIZATION_EVIDENCE')
                factors[days >= event['date']] *= mult
        for issue in audit[code]['issues']:
            issues |= True if issue.get('date') is None else days == issue['date']
        t = group.cutoff_index.to_numpy(dtype=np.int64) * 15 + 14
        np.testing.assert_array_equal(clock[t], group[prep.TIME])
        np.testing.assert_array_equal(clock[t+1200], group.label_end_time)
        for start, end in zip(t // 240, (t + 1200) // 240):
            if issues[start:end+1].any():
                raise ValueError('PREVIOUSLY_ADMITTED_OUTCOME_NOW_HAS_ACTION_ISSUE')
        normal = np.repeat(factors, 240)
        first, lowest = positive_offsets(grid[:, 2] * normal, grid[:, 3] * normal,
                                         t, group.mae_1200m.to_numpy(dtype=float))
        item = group[['code', prep.TIME, 'proposed_segment_name']].copy()
        item['first_touch_trading_minute'], item['minimum_trading_minute'] = first, lowest
        positions.append(item)
        print(json.dumps(dict(stage='existing_positive_outcome_position', code=code, records=len(item))), flush=True)
    return pd.concat(positions, ignore_index=True)


def run():
    if OUT.exists():
        raise ValueError('OUTPUT_EXISTS_NO_RESTART')
    config, label, refs = inputs()
    predictions = pd.read_parquet(FIT / 'predictions.parquet')
    population = pd.read_parquet(prep.OUT / 'population_audit.parquet')
    report = json.loads((FIT / 'report.json').read_bytes())
    used = 0
    outputs = {}

    def write(name, body, terminal=False):
        nonlocal used
        extra = 0 if terminal else RESERVE
        if used + len(body) + extra > CAP or bytes_used() + len(body) + extra > TOTAL_CAP:
            raise ValueError('DIAGNOSTIC_BUDGET_STOP')
        if not OUT.exists():
            OUT.mkdir(exist_ok=False)
        with (OUT / name).open('xb') as target:
            target.write(body)
        used += len(body)
        outputs[name] = dict(sha256=prep.sha(OUT / name), bytes=len(body))

    write('plan.json', prep.encoded(dict(inputs=refs, purpose='FIXED_PREDICTION_AND_EXISTING_POSITIVE_OUTCOME_DIAGNOSTICS',
        cap=CAP, fit=False, relabel=False, thresholds_changed=False, units='shared=unique cutoff; own=candidate cutoff',
        future_data_scope='ONLY_OUTCOME_POSITION_NOT_X', full_original_rows=len(population))))
    try:
        parts, all_coverage = {}, {}
        for segment in prep.SEGMENTS:
            full = prep.read_segment(segment)
            selected = population[population.proposed_segment_name.eq(segment)].reset_index(drop=True)
            common, _ = prep.common_population(full, config['core_observation_groups'])
            np.testing.assert_array_equal(common, selected.common_main_eligible)
            pd.testing.assert_frame_equal(full[['code', prep.TIME]], selected[['code', prep.TIME]])
            parts[segment] = full.loc[common].reset_index(drop=True)
            for fields in [config['arms']['M1']]:
                all_coverage[segment] = {f: int(full[f].notna().sum()) for f in fields}
            stored = predictions[predictions.segment.eq(segment)].reset_index(drop=True)
            pd.testing.assert_frame_equal(stored[['code', prep.TIME, 'y_tail_1200m', 'mae_1200m', 'label_end_time']],
                parts[segment][['code', prep.TIME, 'y_tail_1200m', 'mae_1200m', 'label_end_time']])
            parts[segment] = parts[segment].assign(**{m + '_raw': stored[m + '_raw'].to_numpy() for m in MODELS})
        common_codes = sorted(set.intersection(*(set(frame.code) for frame in parts.values())))
        data = dict(status='FIXED_DIAGNOSTICS_NOT_CAUSAL_OR_NEW_FIT', common_stock_codes=common_codes,
                    ranking={}, feature_shift={}, lr_contributions={}, signals={}, timing={},
                    full_population_feature_finite=all_coverage,
                    fit_performed=False, recalibrated=False, labels_changed=False, http_requests=0,
                    new_model_or_target_selected=False, independent_blind_test=False)
        fields = config['arms']['M1']
        rates = [f for f in fields if f.startswith(('mb5_', 'mb15_', 'own5_', 'own15_')) and f.endswith('_rate')]
        for segment, frame in parts.items():
            data['feature_shift'][segment] = shift(parts['train'], frame, fields)
            matched = frame[frame.code.isin(common_codes)].reset_index(drop=True)
            data['ranking'][segment] = {m: dict(all=rank_summary(frame, frame[m + '_raw']),
                same_stock_intersection=rank_summary(matched, matched[m + '_raw'])) for m in MODELS}
            data['signals'][segment] = {f: signal_summary(frame, f) for f in rates}
        for model in ['B1_lr', 'M1_lr']:
            state = json.loads((FIT / (model + '_preprocessing.json')).read_bytes())
            linear = json.loads((FIT / (model + '_model.json')).read_bytes())
            baseline = None
            data['lr_contributions'][model] = {}
            for segment, frame in parts.items():
                logits, terms = logit_terms(frame, state, linear)
                np.testing.assert_allclose(expit(logits), frame[model + '_raw'], atol=1e-11, rtol=0)
                if baseline is None:
                    baseline = terms
                for f, record in terms.items():
                    record['numeric_shift_from_train'] = record['numeric_mean_logit'] - baseline[f]['numeric_mean_logit']
                    record['missing_shift_from_train'] = record['missing_mean_logit'] - baseline[f]['missing_mean_logit']
                mean_from_terms = linear['intercept'] + sum(t['numeric_mean_logit'] + t['missing_mean_logit'] for t in terms.values())
                np.testing.assert_allclose(mean_from_terms, logits.mean(), atol=1e-10, rtol=0)
                data['lr_contributions'][model][segment] = dict(mean_logit=float(logits.mean()), fields=terms,
                    calibration_slope=report['fitted'][model]['calibration_slope'])
        cohort = pd.concat(list(parts.values()), ignore_index=True)
        offsets = outcome_positions(cohort, label, refs)
        if len(offsets) != int(cohort.y_tail_1200m.sum()) or offsets.duplicated(['code', prep.TIME]).any():
            raise ValueError('POSITIVE_POPULATION_CHANGED')
        for segment in prep.SEGMENTS:
            selected = offsets[offsets.proposed_segment_name.eq(segment)]
            data['timing'][segment] = duration_summary(selected)
        sink = io.BytesIO()
        offsets.to_parquet(sink, index=False, compression='zstd')
        write('OUTCOME_ONLY_positive_positions.parquet', sink.getvalue())
        write('diagnostics.json', prep.encoded(data))
        for path, expected in refs.items():
            if prep.sha(path) != expected:
                raise ValueError('INPUT_CHANGED_DURING_DIAGNOSIS:' + path)
        write('manifest.json', prep.encoded(dict(inputs=refs, outputs=outputs.copy(),
            original_population_rows=len(population), common_rows=len(cohort), existing_positive_records=len(offsets),
            same_stock_intersection=common_codes, new_labels=False, fit=False, http_requests=0)))
        write('stop_record.json', prep.encoded(dict(status='COMPLETE_DIAGNOSTICS_NOT_NEW_FIT',
            manifest_sha256=prep.sha(OUT / 'manifest.json'), bytes_before_stop=used)), terminal=True)
    except Exception as exc:
        write('stop_record.json', prep.encoded(dict(status='FAILED_NO_RESTART', error=f'{type(exc).__name__}:{exc}')), terminal=True)
        raise
    return dict(status='COMPLETE_DIAGNOSTICS_NOT_NEW_FIT', bytes=used, positive_records=len(offsets))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', action='store_true', required=True)
    parser.parse_args()
    print(json.dumps(run(), ensure_ascii=False))
