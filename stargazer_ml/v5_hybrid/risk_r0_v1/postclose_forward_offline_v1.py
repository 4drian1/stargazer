"""OFFLINE ACCEPTANCE ONLY: synthetic clock, immutable forecasts, explicit label joins.

No source collector or live scoring endpoint is provided. Historical feature values
may exercise the adapter; their synthetic timestamps are never real PIT evidence.
"""
from pathlib import Path

import numpy as np
import pandas as pd

from postclose_research_scorer_v1 import ResearchScorer, encode, sha, strict_json, CANDIDATE
from postclose_next_period_evaluation_v1 import timestamp, validate_population
from research_budget_v2 import bytes_used, TOTAL_CAP

MODE = 'OFFLINE_ACCEPTANCE_ONLY'


class OfflineAdapter:
    def __init__(self):
        self.scorer = ResearchScorer()
        refs = self.scorer.manifest['artifacts']
        fit = Path(refs['model']['path']).parent
        audit = strict_json((fit / 'comparison_and_readback.json').read_bytes())
        path = fit / 'linear_model.json'
        if sha(path.read_bytes()) != audit['new_fit_sha256'][path.name]:
            raise ValueError('LINEAR_MODEL_HASH_DRIFT')
        self.linear = strict_json(path.read_bytes())
        self.linear_path, self.linear_sha = path, sha(path.read_bytes())
        self.state = strict_json(Path(refs['preprocessing']['path']).read_bytes())

    def forecast(self, day, inputs, next_day, recorded_at, *, mode):
        if mode != MODE:
            raise PermissionError('LIVE_COLLECTION_AND_SCORING_NOT_ENABLED')
        recorded_at = timestamp(recorded_at)
        timestamp(day*1000000+150000)
        next_open = timestamp(next_day*1000000+93000)
        if not day < next_day or not day*1000000+150000 <= recorded_at < next_open:
            raise ValueError('INVALID_RECORDING_WINDOW')
        if sha(self.linear_path.read_bytes()) != self.linear_sha:
            raise ValueError('LINEAR_MODEL_HASH_DRIFT')
        scope = self.scorer.manifest['research_codes']
        supplied = {}
        for row in inputs:
            if row['code'] in supplied or row['code'] not in scope:
                raise ValueError('DUPLICATE_OR_OUTSIDE_POPULATION')
            supplied[row['code']] = row
        result = []
        for code in scope:
            base = dict(code=code, day=day, next_open_ts=next_open,
                        label_end_time=next_day*1000000+150000,
                        feature_max_available_at=None, scored_at=None,
                        eligibility_known_ts=None, data_arrival_ts=None,
                        recorded_at=recorded_at, score_state='UNSCORABLE',
                        q90_adverse=None, q90_linear=None, stock_return1d=None,
                        reasons=['MISSING_INPUT'], feature_content_sha256=None)
            row = supplied.get(code)
            if row is not None:
                time_fields = ['eligibility_known_ts','data_arrival_ts','feature_max_available_at','scored_at']
                if any(row.get(f) is None for f in time_fields):
                    base['reasons'] = ['UNKNOWN_TIME']
                    result.append(base)
                    continue
                known, arrival, available, scored = [timestamp(row[f]) for f in
                    time_fields]
                if not (known <= available and day*1000000+150000 <= arrival <= available <= scored <= recorded_at < next_open):
                    raise ValueError('INVALID_AVAILABILITY_OR_SCORE_ORDER')
                base.update(eligibility_known_ts=known, data_arrival_ts=arrival,
                            feature_max_available_at=available, scored_at=scored,
                            feature_content_sha256=sha(encode(row['features'])))
                # The frozen historical interface supplies numerical validation only.
                # Its null arrival is not overwritten and is not a forward-PIT claim.
                payload = dict(candidate_id=CANDIDATE,
                    feature_schema_sha256=self.scorer.manifest['feature_schema_sha256'],
                    usage_mode='HISTORICAL_RECONSTRUCTION', rows=[dict(code=code, day=day,
                        feature_asof_ts=day*1000000+150000, decision_available_ts=None,
                        features=row['features'])])
                scored_row = self.scorer.score(payload)['rows'][0]
                base['reasons'] = scored_row['reasons']
                if scored_row['status'] == 'RESEARCH_SCORE':
                    x = np.array([np.nan if row['features'][f] is None else row['features'][f]
                                  for f in self.scorer.fields], dtype=float)
                    linear_x = np.r_[(np.where(np.isnan(x),self.state['median'],x)-self.state['mean'])/self.state['scale'],
                                     np.isnan(x).astype(float)]
                    linear = float(np.clip(linear_x @ self.linear['coef'] + self.linear['intercept'],0,1))
                    base.update(score_state='SCORED', q90_adverse=scored_row['q90_next_session_adverse_excursion'],
                                q90_linear=linear, stock_return1d=row['features']['stock_return1d'])
            result.append(base)
        return dict(mode=MODE, candidate_id=CANDIDATE, day=day, rows=result,
                    population_codes=scope, provenance_verified=False,
                    independent_validation_passed=False, synthetic_clock=True)


class OfflineStore:
    """Single-writer test store. Exclusive writes; partial/crashed files fail readback.

    Not a production concurrency/atomic-commit implementation or collector.
    """
    def __init__(self, root, cap=2*1024**2):
        self.root, self.cap = Path(root), cap

    def freeze(self, kind, key, content):
        if kind not in {'forecast','labels'} or not str(key).isalnum():
            raise ValueError('INVALID_RECORD_ID')
        if content.get('mode') != MODE:
            raise PermissionError('OFFLINE_CONTENT_REQUIRED')
        body = encode(dict(content=content, content_sha256=sha(encode(content))))
        path = self.root / f'{kind}_{key}.json'
        if path.exists():
            if self.read(path.name) != content:
                raise ValueError('IMMUTABLE_RECORD_CONFLICT')
            return path.name
        used = sum(p.stat().st_size for p in self.root.rglob('*') if p.is_file()) if self.root.exists() else 0
        if used+len(body) > self.cap or bytes_used()+len(body) > TOTAL_CAP:
            raise ValueError('WRITE_BUDGET_STOP')
        self.root.mkdir(parents=True, exist_ok=True)
        with path.open('xb') as stream:
            stream.write(body)
        return path.name

    def read(self, name):
        if Path(name).name != name:
            raise ValueError('INVALID_RECORD_PATH')
        envelope = strict_json((self.root/name).read_bytes())
        if sha(encode(envelope['content'])) != envelope['content_sha256']:
            raise ValueError('STORAGE_INTEGRITY_FAIL')
        return envelope['content']


def join_labels(forecast, labels, evaluation_ts, *, exposure_end, freeze_ts, next_sessions, expected_codes):
    if forecast.get('mode') != MODE or labels.get('mode') != MODE:
        raise ValueError('OFFLINE_ACCEPTANCE_ONLY')
    if labels.get('forecast_sha256') != sha(encode(forecast)):
        raise ValueError('FORECAST_REFERENCE_MISMATCH')
    if (forecast['candidate_id'] != CANDIDATE or sorted(forecast['population_codes']) != sorted(expected_codes)
            or any(r['day'] != forecast['day'] for r in forecast['rows'])):
        raise ValueError('FROZEN_POPULATION_OR_CANDIDATE_MISMATCH')
    expected = {(r['code'],r['day']) for r in forecast['rows']}
    actual = [(r['code'],r['day']) for r in labels['rows']]
    if len(actual) != len(set(actual)) or set(actual) != expected:
        raise ValueError('LABEL_POPULATION_MISMATCH')
    if len(expected) != len(forecast['rows']) or sorted(forecast['population_codes']) != sorted(r['code'] for r in forecast['rows']):
        raise ValueError('FORECAST_POPULATION_MISMATCH')
    outcomes = {(r['code'],r['day']):r for r in labels['rows']}
    rows = []
    for record in forecast['rows']:
        outcome = outcomes[(record['code'],record['day'])]
        if set(outcome) != {'code','day','label_state','data_complete_time','next_session_adverse_excursion'}:
            raise ValueError('OUTCOME_CANNOT_OVERWRITE_FORECAST')
        rows.append(dict(record, **{k:v for k,v in outcome.items() if k not in {'code','day'}}))
    frame = pd.DataFrame(rows)
    validate_population(frame, set(forecast['population_codes']),exposure_end,evaluation_ts,freeze_ts,next_sessions)
    return frame
