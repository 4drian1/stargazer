"""Freeze a keyed three-arm research specification; never fit a real estimator."""
import argparse
import hashlib
from importlib.metadata import version
import json
from pathlib import Path

import numpy as np
import pandas as pd

from joint_fit_preparation_v1 import whitelist
from read_multiboard_full_features_v1 import metadata, read_segment
from research_budget_v2 import bytes_used, TOTAL_CAP

HERE = Path(__file__).resolve().parent
OUT = HERE / 'multiboard_comparison_preparation_20261018_v1'
CAP, RESERVE, FIT_CAP = 2 * 1024**2, 64 * 1024, 16 * 1024**2
TIME = 'decision_ts_assumed_end'
SEGMENTS = ('train', 'select', 'calibration', 'test')
CORE_GROUPS = {
    'old_golden': ['sse_15m_anchor_position'],
    'new_golden': ['new_sse_golden_position'],
    'golden_time': ['time_node_nearest_abs_distance'],
    'sse5': ['sse_5m_rsi14', 'sse_5m_return_adjacent_bar'],
    'sse15': ['sse_15m_rsi14', 'sse_15m_day_return_prev_close'],
    'cyb5': ['cyb_5m_rsi14', 'cyb_5m_macd_hist_over_close'],
    'cyb15': ['cyb_15m_rsi14', 'cyb_15m_day_return_prev_close'],
    'original_sector': ['context_sector_mean_return_loo_raw', 'context_sector_breadth_loo'],
    'original_leader': ['context_current_leader_return_loo_raw', 'context_captured_leader_mean_return_loo_raw'],
    'stock': ['stock_day_return_raw', 'stock_rsi14_5m_raw', 'stock_rsi14_15m_raw'],
    'global5': ['mb5_mean'],
    'global15': ['mb15_mean'],
    'own5': ['own5_mean'],
    'own15': ['own15_mean'],
    'own_leader5': ['own5_current_leader_change', 'own5_captured_leader_change'],
    'own_leader15': ['own15_current_leader_change', 'own15_captured_leader_change'],
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def encoded(value):
    return json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2).encode('utf-8')


def make_arms(old, spec):
    original = whitelist(old)
    old_gold = old['feature_groups']['golden_position']
    new_gold = [f for f in spec['shared'] if f.startswith('new_sse_golden_')]
    suffixes = ['', '_distance_0_236', '_distance_0_382', '_distance_0_5', '_distance_0_618', '_distance_0_786']
    expected_new = ['new_sse_golden_position' if not s else 'new_sse_golden' + s for s in suffixes]
    expected_old = ['sse_15m_anchor_position' if not s else 'sse_15m_golden' + s for s in suffixes]
    if old_gold != expected_old or new_gold != expected_new or len(original) != 100:
        raise ValueError('GOLDEN_MAPPING_OR_ORIGINAL_WHITELIST_CHANGED')
    replacements = dict(zip(old_gold, new_gold))
    revised = [replacements.get(f, f) for f in original]
    extra = [f for f in spec['shared'] + spec['own'] if f not in new_gold]
    if len(extra) != 100 or set(extra) & set(original):
        raise ValueError('NEW_GROUP_FEATURE_LIST_CHANGED')
    arms = {'B0': original, 'B1': revised, 'M1': revised + extra}
    for fields in arms.values():
        if len(fields) != len(set(fields)):
            raise ValueError('DUPLICATE_MODEL_FIELD')
        whitelist({'feature_groups': {'explicit': fields}})
    return arms


def common_population(frame, groups=CORE_GROUPS):
    masks = {}
    for name, fields in groups.items():
        values = frame[fields].to_numpy(dtype=float)
        if np.isinf(values).any():
            raise ValueError('INFINITE_CORE_OBSERVATION')
        masks[name] = np.isfinite(values).any(axis=1)
    return np.logical_and.reduce(list(masks.values())), masks


def validate_times_labels(frame, segment, boundaries):
    if frame.empty or frame.duplicated(['code', TIME]).any():
        raise ValueError('EMPTY_OR_DUPLICATE_COHORT')
    for name in (TIME, 'label_end_time'):
        text = frame[name].astype(str)
        if not text.str.fullmatch(r'\d{14}').all():
            raise ValueError('INVALID_COMPLETE_TIMESTAMP')
        pd.to_datetime(text, format='%Y%m%d%H%M%S', errors='raise')
    days = frame[TIME].to_numpy(dtype=np.int64) // 1000000
    entry = boundaries[segment]
    if not ((days >= entry['start']) & (days <= entry['end'])).all():
        raise ValueError('DECISION_OUTSIDE_FROZEN_SEGMENT')
    if not frame.proposed_segment_name.eq(segment).all() or not (frame.label_end_time > frame[TIME]).all():
        raise ValueError('LABEL_NOT_STRICTLY_FUTURE_OR_SEGMENT_MISMATCH')
    index = SEGMENTS.index(segment)
    if index < len(SEGMENTS) - 1:
        next_start = boundaries[SEGMENTS[index + 1]]['start'] * 1000000
        if not (frame.label_end_time < next_start).all():
            raise ValueError('LABEL_CROSSES_NEXT_SEGMENT')
    y, mae = frame.y_tail_1200m.to_numpy(), frame.mae_1200m.to_numpy(dtype=float)
    if not np.isin(y, [0, 1]).all() or not np.isfinite(mae).all():
        raise ValueError('INVALID_TARGET')
    if not np.array_equal(y, (mae <= -0.08 + 1e-12).astype(int)):
        raise ValueError('TARGET_THRESHOLD_CONFLICT')


def neutral_keep(codes, times, scores, fraction=0.2, minimum=5):
    """Filtering research only; invalid/tiny cross sections are not filled."""
    codes, times, scores = np.asarray(codes), np.asarray(times), np.asarray(scores, dtype=float)
    if not (len(codes) == len(times) == len(scores)) or not np.isfinite(scores).all() or not 0 <= fraction < 1:
        raise ValueError('INVALID_RANKING_INPUT')
    if len(set(zip(codes, times))) != len(codes):
        raise ValueError('DUPLICATE_RANKING_KEY')
    keep, supported = np.ones(len(codes), bool), np.zeros(len(codes), bool)
    for cutoff in np.unique(times):
        indices = np.flatnonzero(times == cutoff)
        if len(indices) < minimum:
            continue
        supported[indices] = True
        ordered = sorted(indices, key=lambda i: (-scores[i], hashlib.sha256(
            f'R0_MULTIBOARD_V1|{cutoff}|{codes[i]}'.encode()).hexdigest()))
        keep[ordered[:int(np.floor(fraction * len(indices)))]] = False
    return keep, supported


def profile(frame, fields):
    values = frame[fields].to_numpy(dtype=float)
    if np.isinf(values).any():
        raise ValueError('INFINITE_MODEL_FIELD')
    return {f: {'finite': int(np.isfinite(values[:, i]).sum()),
                'missing': int(np.isnan(values[:, i]).sum()),
                'distinct_nonmissing': int(frame[f].nunique(dropna=True))} for i, f in enumerate(fields)}


def summarize(frame, mask):
    subset = frame.loc[mask]
    return dict(rows=len(subset), positive=int(subset.y_tail_1200m.sum()),
                negative=int(len(subset) - subset.y_tail_1200m.sum()),
                dates=int((subset[TIME] // 1000000).nunique()), stocks=int(subset.code.nunique()))


def prepare():
    if OUT.exists():
        raise ValueError('OUTPUT_EXISTS_NO_RESTART')
    report, spec = metadata()
    full = HERE / 'multiboard_full_features_20261018_v1'
    old_path = HERE / 'joint_momentum_review_20261017_v1/fit_review_config_v2.json'
    old = json.loads(old_path.read_bytes())
    arms = make_arms(old, spec)
    label_path = HERE / 'contracts/INTRADAY_DECISION_AFTER_LABEL_V1.json'
    label = json.loads(label_path.read_bytes())
    boundaries = {r['name']: r for r in label['proposed_segments']}
    refs = dict(report['inputs'])
    for name in ('report.json', 'feature_spec.json', 'stop_record.json'):
        refs[str(full / name)] = sha(full / name)
    refs.update({str(full / name): v['sha256'] for name, v in report['outputs'].items()})
    for name, key in [('multiboard_full_features_acceptance_20261018_v1.json', 'code_refs'),
                      ('multiboard_group_counts_acceptance_20261018_v1.json', 'refs')]:
        path = HERE / name
        receipt = json.loads(path.read_bytes())
        if not receipt['status'].startswith('PASS_'):
            raise ValueError('FEATURE_ACCEPTANCE_REQUIRED')
        refs.update(receipt[key])
        refs[str(path)] = sha(path)
    for path in [old_path, label_path, Path(__file__), HERE / 'test_prepare_multiboard_comparison_v1.py',
                 HERE / 'joint_fit_preparation_v1.py', HERE / 'join_momentum_review_v1.py',
                 HERE / 'joint_momentum_review_20261017_v1/manifest.json',
                 HERE / 'joint_momentum_review_20261017_v1/market_features_by_cutoff.parquet']:
        refs[str(path)] = sha(path)
    for path, expected in refs.items():
        if sha(path) != expected:
            raise ValueError('FROZEN_INPUT_DRIFT:' + path)

    audits, profiles, masks_out, blockers = {}, {}, [], []
    all_fields = list(dict.fromkeys(sum(arms.values(), [])))
    for segment in SEGMENTS:
        frame = read_segment(segment)
        validate_times_labels(frame, segment, boundaries)
        mask, groups = common_population(frame)
        profile_all = profile(frame, all_fields)
        selected = frame.loc[mask]
        profile_main = profile(selected, all_fields)
        all_rows = summarize(frame, np.ones(len(frame), bool))
        main = summarize(frame, mask)
        support = selected.groupby(TIME).size()
        audits[segment] = dict(full_cohort=all_rows, common_main=main,
            excluded_rows=len(frame) - len(selected), retained_share=len(selected) / len(frame),
            absent_by_core_group={g: int((~m).sum()) for g, m in groups.items()},
            comparable_cutoffs=int((support >= 5).sum()),
            comparable_rows=int(support[support >= 5].sum()),
            tiny_cross_section_rows=int(support[support < 5].sum()))
        profiles[segment] = dict(full=profile_all, common_main=profile_main)
        if not main['rows'] or not main['positive'] or not main['negative']:
            blockers.append(segment + ':COMMON_POPULATION_NOT_TWO_CLASS')
        if segment == 'train':
            for arm, fields in arms.items():
                empty = [f for f in fields if profile_main[f]['finite'] == 0]
                if empty:
                    blockers.append(arm + ':ALL_NULL_TRAIN_COLUMNS:' + ','.join(empty))
        audit = frame[['code', TIME, 'proposed_segment_name']].copy()
        audit['common_main_eligible'] = mask
        audit['missing_core_groups'] = ['|'.join(g for g, m in groups.items() if not m[i]) for i in range(len(frame))]
        masks_out.append(audit)
    population = pd.concat(masks_out, ignore_index=True)
    if len(population) != 100704 or population.duplicated(['code', TIME]).any():
        raise ValueError('FULL_POPULATION_CHANGED')
    sink = __import__('io').BytesIO()
    population.to_parquet(sink, index=False, compression='zstd')
    missing_policy = dict(common_population='same decision-observable core mask for every arm and model; all original rows retained in audit',
        within_family='NULL permitted; coverage alone never proves a family observed',
        lr='train-only median then mean/std numeric; fixed indicator for every field; constant scale=1',
        lightgbm='native NaN plus the same fixed per-field missing indicators',
        all_null_train='STOP_NO_AUTOMATIC_COLUMN_REMOVAL', infinite='REJECT',
        constant_nonmissing='KEEP_AND_REPORT; missingness may vary', real_preprocessing_fitted=False)
    config = dict(version='MULTIBOARD_THREE_ARM_RESEARCH_V1', status='FIXED_RESEARCH_SPEC_NOT_FIT_AUTHORIZATION',
        fit_authorized=False, http_authorized=False, deployment_authorized=False,
        logical_reader='read_multiboard_full_features_v1.read_segment', arms=arms,
        core_observation_groups=CORE_GROUPS, label_contract=str(label_path),
        label='strictly future1200tradingminutes, adverse low/reference<=-8%; not simulated fill or PnL',
        segments=label['proposed_segments'], missing_policy=missing_policy,
        models=old['models_proposed'],
        selection=dict(lr='one fixed configuration; no hyperparameter search',
            lightgbm_early_stopping_rounds=50, lightgbm_metric='binary_logloss', select_only=True),
        calibration=dict(method='sigmoid_on_clipped_raw_probability_logit', epsilon=1e-12,
            estimator=dict(C=1.0, solver='lbfgs', max_iter=1000, tol=0.0001, class_weight=None, random_state=42),
            fit_segment='calibration', model_training_on_calibration=False,
            report_uncalibrated_and_calibrated=True),
        evaluation=dict(primary='paired B1_vs_M1 Brier and LogLoss; B0_vs_B1 separates golden update',
            secondary=['ROC_AUC', 'PR_AUC', 'calibration_bins', 'monthly_underestimation', 'tail_rate_at_equal_retention'],
            constant='train common-population event frequency only',
            simple_rule='negative stock_return15m_raw; evaluate only same finite-score subset across all arms; not full user rule engine',
            same_cutoff_random='analytical expected tail count under same floor(0.2*n) rejects',
            rejection_fraction=0.2, minimum_cross_section=5,
            ties='SHA256(R0_MULTIBOARD_V1|cutoff|code), independent of scores/labels/input order',
            uncertainty=dict(method='paired moving-block bootstrap by observed trading date', block_dates=10, reps=1000, seed=42),
            alert_threshold=None, financial_promotion_gate=None, pnl_gate=False,
            independent_blind_test=False, test_role='historical research reused; never selects arms, parameters or threshold'),
        technical_gates='both classes and no all-NULL training X, NOT statistical power or financial effectiveness',
        later_fit_output_cap=FIT_CAP, package_versions={p: version(p) for p in ['numpy', 'pandas', 'scikit-learn', 'lightgbm', 'pyarrow']})
    payloads = {'config.json': encoded(config), 'population_audit.parquet': sink.getvalue(),
                'field_profiles.json': encoded(profiles)}
    result = dict(status='PASS_FIXED_COMPARISON_PREPARATION_NOT_FIT' if not blockers else 'BLOCKED_COMPARISON_PREPARATION',
        inputs=refs, segments=audits, technical_blockers=blockers, original_rows=len(population),
        arm_numeric_fields={k: len(v) for k, v in arms.items()},
        arm_numeric_plus_missing_indicator_fields={k: 2 * len(v) for k, v in arms.items()},
        fit_authorized=False, fit_performed=False, real_preprocessing_fitted=False,
        full_rule_completeness_certified=False, http_requests=0, new_blind_test=False,
        outputs={n: dict(bytes=len(b), sha256=hashlib.sha256(b).hexdigest()) for n, b in payloads.items()})
    payloads['report.json'] = encoded(result)
    payloads['stop_record.json'] = encoded(dict(status='COMPLETE', fit_performed=False))
    size = sum(map(len, payloads.values()))
    if size + RESERVE > CAP or bytes_used() + size + RESERVE + FIT_CAP > TOTAL_CAP:
        raise ValueError('PREPARATION_OR_PROPOSED_FIT_BUDGET_STOP')
    for path, expected in refs.items():
        if sha(path) != expected:
            raise ValueError('INPUT_CHANGED_DURING_PREPARATION:' + path)
    OUT.mkdir(exist_ok=False)
    for name, body in payloads.items():
        with (OUT / name).open('xb') as stream:
            stream.write(body)
    return check()


def check():
    terminal = json.loads((OUT / 'stop_record.json').read_bytes())
    report = json.loads((OUT / 'report.json').read_bytes())
    if terminal['status'] != 'COMPLETE':
        raise ValueError('PREPARATION_NOT_COMPLETE')
    for path, expected in report['inputs'].items():
        if sha(path) != expected:
            raise ValueError('FROZEN_INPUT_DRIFT:' + path)
    for name, item in report['outputs'].items():
        if sha(OUT / name) != item['sha256']:
            raise ValueError('PREPARATION_ARTIFACT_DRIFT:' + name)
    return dict(status=report['status'], preparation_ready=not report['technical_blockers'],
        fit_authorized=False, fit_performed=False, training_ready=False,
        next_gate='SEPARATE_FIT_AUTHORIZATION_ON_CONFIG_SHA256', config_sha256=sha(OUT / 'config.json'),
        technical_blockers=report['technical_blockers'], segments=report['segments'],
        registered_bytes=bytes_used(), total_cap=TOTAL_CAP)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--prepare', action='store_true', help='create immutable preparation; no fit')
    parser.add_argument('--check', action='store_true', help='read-only validation; no fit')
    args = parser.parse_args()
    if args.prepare == args.check:
        parser.error('choose exactly one of --prepare or --check')
    answer = prepare() if args.prepare else check()
    print(json.dumps(answer, ensure_ascii=False, allow_nan=False))
    raise SystemExit(0 if answer['preparation_ready'] else 2)
