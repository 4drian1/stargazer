"""Independent readback of review artifacts; never fit preprocessing or models."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
from joint_fit_preparation_v1 import whitelist, cutoff_support, group_observation_masks
from research_budget_v2 import bytes_used, TOTAL_CAP

HERE = Path(__file__).parent
OUT = HERE / 'joint_fit_review_20261017_v1'


def run():
    manifest = json.loads((OUT / 'manifest.json').read_bytes())
    spec = json.loads((HERE / 'contracts/JOINT_RISK_FIT_REVIEW_V1.json').read_bytes())
    fields = whitelist(spec)
    assert fields == manifest['fields'] and len(fields) == 46
    assert json.loads((OUT / 'stop_record.json').read_bytes())['status'] == 'COMPLETE'
    for name, sha in manifest['inputs'].items():
        assert hashlib.sha256(Path(name).read_bytes()).hexdigest() == sha, name
    for name, entry in manifest['outputs'].items():
        body = (OUT / name).read_bytes()
        assert len(body) == entry['bytes']
        assert hashlib.sha256(body).hexdigest() == entry['sha256'], name
    frames = []
    boundaries = dict(train=20251001000000, select=20260101000000, calibration=20260401000000)
    metadata = {'code', 'cutoff_index', 'decision_ts_assumed_end', 'label_end_time',
                'y_tail_1200m', 'mae_1200m', 'proposed_segment_name',
                'has_any_value_in_each_candidate_group', 'source_part_index', 'source_row_index'}
    for name, expected in manifest['summary'].items():
        data = pd.read_parquet(OUT / (name + '.parquet'))
        assert set(data.columns) == set(fields) | metadata
        assert not data.duplicated(['code', 'cutoff_index']).any()
        assert len(data) == expected['rows'] and int(data.y_tail_1200m.sum()) == expected['positive']
        assert (data.proposed_segment_name == name).all()
        assert (data.label_end_time > data.decision_ts_assumed_end).all()
        if name in boundaries:
            assert (data.label_end_time < boundaries[name]).all()
        assert np.isfinite(data.mae_1200m).all()
        assert set(data.y_tail_1200m.unique()) <= {0, 1}
        assert np.array_equal((data.mae_1200m <= -.08 + 1e-12).astype(int), data.y_tail_1200m)
        assert not np.isinf(data[fields].to_numpy(dtype=float)).any()
        for field in fields:
            actual = dict(missing=int(data[field].isna().sum()), finite=int(data[field].notna().sum()),
                          distinct=int(data[field].nunique(dropna=True)))
            assert actual == manifest['field_audit'][name][field]
        masks = group_observation_masks(data, spec['feature_groups'])
        assert np.array_equal(np.logical_and.reduce(list(masks.values())), data.has_any_value_in_each_candidate_group)
        assert cutoff_support(data.decision_ts_assumed_end, 5) == manifest['ranking_cross_sections'][name]
        assert np.array_equal(data.session_is_pm, ((data.decision_ts_assumed_end // 10000) % 100 >= 13).astype(int))
        for field, left, right in [
            ('context_sector_minus_sse_return_raw', 'context_sector_mean_return_loo_raw', 'sse_15m_day_return_prev_close'),
            ('context_stock_minus_sector_return_raw', 'stock_day_return_raw', 'context_sector_mean_return_loo_raw'),
            ('context_stock_minus_leader_return_raw', 'stock_day_return_raw', 'context_current_leader_return_loo_raw')]:
            assert np.allclose(data[field], data[left] - data[right], equal_nan=True, rtol=0, atol=1e-14)
        distance = data.time_node_nearest_abs_distance.dropna()
        assert distance.between(0, 1).all()
        frames.append(data[['code', 'cutoff_index', 'decision_ts_assumed_end', 'proposed_segment_name']])
    all_keys = pd.concat(frames, ignore_index=True)
    assert len(all_keys) == 100704 and not all_keys.duplicated(['code', 'cutoff_index']).any()
    assert all_keys.groupby(all_keys.decision_ts_assumed_end // 1000000).proposed_segment_name.nunique().max() == 1
    result = dict(status='PASS_STORAGE_KEYS_SPLITS_AND_FEATURE_REVIEW_ONLY', rows=len(all_keys),
                  fields=len(fields), inputs_and_outputs_hash_verified=True, no_infinite_features=True,
                  feature_missing_rows_dropped=0, real_preprocessing_fitted=False, fit_run=False,
                  training_ready=False, full_rule_inputs_ready=False,
                  manifest_sha256=hashlib.sha256((OUT / 'manifest.json').read_bytes()).hexdigest())
    body = json.dumps(result, ensure_ascii=False, allow_nan=False, indent=2).encode()
    if bytes_used() + len(body) > TOTAL_CAP:
        raise ValueError('RESEARCH_BUDGET_STOP')
    with (OUT / 'readback_acceptance.json').open('xb') as stream:
        stream.write(body)
    print(json.dumps(result))


if __name__ == '__main__':
    run()
