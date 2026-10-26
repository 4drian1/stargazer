"""Real frozen local feature inputs -> research scores, not live/PIT observation.

Only whitelisted features are projected from parquet. Never loads a label column.
Today's file-observation clock cannot repair unknown historical arrival times.
"""
import io
from datetime import datetime, timezone, timedelta
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

from postclose_research_scorer_v1 import HERE, ResearchScorer, encode, sha, strict_json, CANDIDATE


def clock_now():
    return datetime.now(timezone(timedelta(hours=8))).isoformat(timespec='microseconds')


def bound_read(path, expected, receipts):
    started = clock_now()
    body = Path(path).read_bytes()
    if sha(body) != expected:
        raise ValueError('SOURCE_HASH_DRIFT:' + str(path))
    receipts.append(dict(path=str(path),sha256=expected,bytes=len(body),
                         observation_started_at=started,observation_finished_at=clock_now(),
                         original_data_arrival_ts=None))
    return body


def overlay_features(base, overlay, fields):
    if base[['code','day']].duplicated().any() or overlay[['code','day']].duplicated().any():
        raise ValueError('DUPLICATE_SOURCE_KEYS')
    if not base[['code','day']].equals(overlay[['code','day']]):
        raise ValueError('OVERLAY_KEY_OR_ORDER_DRIFT')
    result = base[['code','day',*fields]].copy()
    for field in fields:
        if field.startswith('context_') and field in overlay:
            result[field] = overlay[field].to_numpy()
    result['context_stock_minus_sector_daily_return'] = result.stock_return1d-result.context_sector_mean_return_loo_raw
    result['context_stock_minus_leader_daily_return'] = result.stock_return1d-result.context_current_leader_return_loo_raw
    return result


def load_local_features(scorer):
    refs = scorer.manifest['artifacts']
    plan = strict_json(Path(refs['plan']['path']).read_bytes())
    receipts = []
    repair_path = HERE/'postclose_daily_context_repair_20261017_v1/manifest.json'
    repair = strict_json(bound_read(repair_path,plan['inputs'][str(repair_path)],receipts))
    base_path = HERE/'postclose_research_20261017_v2/stock_day_population.parquet'
    base_body = bound_read(base_path,repair['inputs'][str(base_path)],receipts)
    # This projection excludes all labels/outcomes, split masks and cohort eligibility.
    base = pq.read_table(io.BytesIO(base_body),columns=['code','day',*scorer.fields]).to_pandas()
    overlay_path = repair_path.parent/'daily_context_overlay.parquet'
    body = bound_read(overlay_path,repair['outputs'][overlay_path.name]['sha256'],receipts)
    schema = pq.read_schema(io.BytesIO(body))
    columns = ['code','day',*[f for f in scorer.fields if f.startswith('context_') and f in schema.names]]
    overlay = pq.read_table(io.BytesIO(body),columns=columns).to_pandas()
    return overlay_features(base,overlay,scorer.fields), receipts


def latest_day_payload(data, scorer):
    day = int(data.day.max())
    selected = data[data.day.eq(day)].set_index('code')
    if not selected.index.is_unique or not set(selected.index).issubset(scorer.manifest['research_codes']):
        raise ValueError('INVALID_DAY_POPULATION')
    rows = []
    absent = []
    for code in scorer.manifest['research_codes']:
        if code not in selected.index:
            features = {f:None for f in scorer.fields}
            absent.append(code)
        else:
            row = selected.loc[code]
            features = {f:None if pd.isna(row[f]) else float(row[f]) for f in scorer.fields}
        rows.append(dict(code=code,day=day,feature_asof_ts=day*1000000+150000,
                         decision_available_ts=None,features=features))
    return dict(candidate_id=CANDIDATE,feature_schema_sha256=scorer.manifest['feature_schema_sha256'],
                usage_mode='HISTORICAL_RECONSTRUCTION',rows=rows), absent


def observe_existing_cache():
    scorer = ResearchScorer()
    data, receipts = load_local_features(scorer)
    payload, absent = latest_day_payload(data,scorer)
    features_ready_at = clock_now()
    scores = scorer.score(payload)
    scored_at = clock_now()
    for row in scores['rows']:
        row.update(forward_validation_eligible=False,eligibility_known_ts=None,
                   original_data_arrival_ts=None,
                   exclusion_reason='PREVIOUSLY_DEVELOPED_HISTORICAL_CACHE_NOT_FORWARD_RECORD')
        if row['code'] in absent:
            row['reasons'].append('MISSING_SOURCE_ROW')
    return dict(mode='REAL_LOCAL_CACHE_REPLAY_NOT_PROSPECTIVE',receipts=receipts,
                feature_input=payload,scores=scores,features_ready_at=features_ready_at,
                scored_at=scored_at,record_prepared_at=clock_now(),
                source_day=int(data.day.max()),source_population_rows=len(data),
                missing_source_codes=absent,real_forward_records=0,
                labels_read=False,network_requests=0,fit_run=False,
                warning='Observed now is not original arrival; cached features are not live recomputation.')
