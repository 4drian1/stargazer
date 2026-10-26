"""Read-only validation of fixed predictions; no scoring, fit, collection or promotion."""
import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from postclose_research_scorer_v1 import CANDIDATE, HERE, strict_json, sha
from diagnose_postclose_predictions_v1 import filter_by_day, filter_summary

CONTRACT = HERE / 'contracts/POSTCLOSE_NEXT_PERIOD_EVALUATION_V1.json'
PACKAGE = HERE / 'postclose_scoring_handoff_20261017_v1/manifest.json'
TARGET = 'next_session_adverse_excursion'


def timestamp(value):
    # Nullable parquet columns may be represented as exact integral float64.
    if isinstance(value, (float, np.floating)) and np.isfinite(value) and value.is_integer():
        value = int(value)
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError('TIMESTAMP_INTEGER_REQUIRED')
    text = str(value)
    if len(text) != 14:
        raise ValueError('TIMESTAMP_PRECISION')
    parsed = datetime.strptime(text, '%Y%m%d%H%M%S')
    if parsed.strftime('%Y%m%d%H%M%S') != text:
        raise ValueError('INVALID_TIMESTAMP')
    return int(value)


def preparation_check():
    spec = strict_json(CONTRACT.read_bytes())
    manifest = strict_json(PACKAGE.read_bytes())
    if spec['candidate_id'] != CANDIDATE or manifest['candidate_id'] != CANDIDATE:
        raise ValueError('CANDIDATE_DRIFT')
    for item in manifest['artifacts'].values():
        if sha(Path(item['path']).read_bytes()) != item['sha256']:
            raise ValueError('FROZEN_CANDIDATE_DRIFT')
    blockers = []
    if not spec['window_proposal']['approved']:
        blockers.append('EVALUATION_WINDOW_NOT_APPROVED')
    acceptance = spec['financial_acceptance']
    if not acceptance['approved_before_new_labels'] or any(v is None for v in acceptance.values()):
        blockers.append('FINANCIAL_ACCEPTANCE_NOT_PREREGISTERED')
    for name, value in spec['binding'].items():
        if value is None:
            blockers.append('UNBOUND:' + name)
    return dict(status='PREPARATION_ONLY', candidate_id=CANDIDATE, evaluation_ready=not blockers,
                blockers=blockers, independent_validation_passed=False,
                deployment_authorized=False, next_period_metrics=None)


def validate_population(frame, codes, exposure_end, evaluation_ts, freeze_ts, next_sessions):
    """Rows without a mature target stay in the denominator, not in risk metrics.

    All timestamps are declared exchange-local Asia/Shanghai times. This checks
    their relations; it cannot authenticate original availability or source data.
    """
    required = {'code', 'day', 'feature_max_available_at', 'scored_at', 'next_open_ts',
                'label_end_time', 'data_complete_time', 'label_state', TARGET,
                'q90_adverse', 'q90_linear', 'stock_return1d', 'score_state'}
    if not required.issubset(frame.columns) or frame.empty:
        raise ValueError('INCOMPLETE_OR_EMPTY_POPULATION')
    if frame[['code', 'day']].duplicated().any():
        raise ValueError('DUPLICATE_POPULATION_KEY')
    if not set(frame.code).issubset(codes):
        raise ValueError('OUTSIDE_FROZEN_CODE_SCOPE')
    evaluation_ts, freeze_ts = timestamp(evaluation_ts), timestamp(freeze_ts)
    if evaluation_ts < freeze_ts:
        raise ValueError('EVALUATION_BEFORE_FREEZE')
    for row in frame.to_dict('records'):
        if isinstance(row['day'], bool) or not isinstance(row['day'], (int, np.integer)):
            raise ValueError('DAY_INTEGER_REQUIRED')
        day = int(row['day'])
        timestamp(day * 1000000 + 150000)
        if day <= exposure_end or day <= freeze_ts // 1000000:
            raise ValueError('DEVELOPED_OR_PREFREEZE_DATE')
        end, next_open = timestamp(row['label_end_time']), timestamp(row['next_open_ts'])
        if day not in next_sessions or next_open // 1000000 != next_sessions[day]:
            raise ValueError('NEXT_SESSION_CALENDAR_MISMATCH')
        if next_open // 1000000 <= day or next_open % 1000000 != 93000:
            raise ValueError('INVALID_NEXT_SESSION_OPEN')
        if end != (next_open // 1000000) * 1000000 + 150000:
            raise ValueError('WRONG_NEXT_SESSION_LABEL_END')
        if row['score_state'] == 'SCORED':
            available = timestamp(row['feature_max_available_at'])
            scored = timestamp(row['scored_at'])
            if not day * 1000000 + 150000 <= available <= scored < next_open or scored > evaluation_ts:
                raise ValueError('LOOKAHEAD_OR_LATE_PREDICTION')
            for name in ['q90_adverse', 'q90_linear']:
                value = row[name]
                if pd.isna(value) or not np.isfinite(value) or not 0 <= value <= 1:
                    raise ValueError('INVALID_FIXED_PREDICTION')
        elif row['score_state'] == 'UNSCORABLE':
            if not pd.isna(row['q90_adverse']) or not pd.isna(row['q90_linear']):
                raise ValueError('UNSCORABLE_MUST_NOT_HAVE_SCORE')
        else:
            raise ValueError('UNKNOWN_SCORE_STATE')
        if row['label_state'] == 'COMPLETE':
            complete = timestamp(row['data_complete_time'])
            value = row[TARGET]
            if not end <= complete <= evaluation_ts:
                raise ValueError('LABEL_NOT_MATURE_AT_EVALUATION')
            if pd.isna(value) or not np.isfinite(value) or not 0 <= value < 1:
                raise ValueError('INVALID_ADVERSE_TARGET')
        elif row['label_state'] in {'PENDING', 'DATA_INSUFFICIENT', 'ACTION_UNRESOLVED'}:
            if not pd.isna(row[TARGET]):
                raise ValueError('INCOMPLETE_LABEL_MUST_BE_NULL')
        else:
            raise ValueError('UNKNOWN_LABEL_STATE')


def descriptive_evaluation(frame, constant, train_edges):
    """Pure metrics on prevalidated rows; not a financial acceptance decision."""
    if not np.isfinite(constant) or not 0 <= constant < 1:
        raise ValueError('INVALID_FROZEN_BASELINE')
    if len(train_edges) != 4 or not np.isfinite(train_edges).all() or sorted(train_edges) != list(train_edges):
        raise ValueError('INVALID_FIXED_TRAIN_BINS')
    scored = frame.score_state.eq('SCORED')
    labeled = frame.label_state.eq('COMPLETE')
    population = dict(rows=len(frame), dates=int(frame.day.nunique()),
                      scored_rows=int(scored.sum()), complete_label_rows=int(labeled.sum()),
                      status_counts=frame.label_state.value_counts().to_dict())
    valid = frame[scored & labeled].sort_values(['day', 'code']).copy()
    def metrics(group):
        if group.empty:
            return dict(rows=0, pinball=None, constant_pinball=None, linear_pinball=None,
                        coverage=None, tail_rate=None)
        y = group[TARGET].to_numpy(float)
        def pinball(pred):
            residual = y - pred
            return float(np.maximum(.9 * residual, -.1 * residual).mean())
        return dict(rows=len(group), pinball=pinball(group.q90_adverse.to_numpy()),
                    constant_pinball=pinball(constant), linear_pinball=pinball(group.q90_linear.to_numpy()),
                    coverage=float((y <= group.q90_adverse).mean()), tail_rate=float((y > constant).mean()))
    valid['month'] = valid.day.astype(str).str[:6]
    bins = np.searchsorted(train_edges, valid.q90_adverse.to_numpy(), side='left')
    daily, coverage = filter_by_day(valid, constant)
    comparison = filter_summary(daily)
    # Existing helper labels its intervals post-hoc; never upgrade that label here.
    return dict(status='DESCRIPTIVE_ONLY_NOT_FINANCIAL_GATE', population=population,
                metrics=metrics(valid), evaluated_fraction=len(valid) / len(frame),
                same_day_fraction_of_population=coverage['comparable_rows'] / len(frame),
                monthly={str(k): metrics(g) for k,g in valid.groupby('month')},
                fixed_train_quintiles=[dict(bin=i+1, **metrics(valid.iloc[np.flatnonzero(bins == i)])) for i in range(5)],
                same_day_coverage=coverage, same_day_comparison=comparison,
                independent_validation_passed=False, deployment_authorized=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true', required=True)
    parser.parse_args()
    result = preparation_check()
    print(json.dumps(result, ensure_ascii=False, allow_nan=False))
    return 0 if result['evaluation_ready'] else 2


if __name__ == '__main__':
    raise SystemExit(main())
