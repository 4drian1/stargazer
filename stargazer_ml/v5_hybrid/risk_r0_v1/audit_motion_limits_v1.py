"""Independent static-set and persisted-score arithmetic audit; no model FIT."""
import gzip
import json

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

import diagnose_motion_limits_v1 as diagnostic
from materialize_relative_stall_pilot_v1 import sha
from research_budget_v2 import bytes_used,TOTAL_CAP


def run():
    out=diagnostic.OUT;plan=json.loads((out/'plan.json').read_bytes())
    stop=json.loads((out/'stop_record.json').read_bytes());manifest=json.loads((out/'manifest.json').read_bytes())
    assert stop['status']=='COMPLETE_NO_FIT' and stop['manifest_sha256']==sha(out/'manifest.json')
    for name,item in manifest['outputs'].items():assert sha(out/name)==item['sha256']
    for path,h in plan['inputs'].items():assert sha(path)==h,path
    report=json.loads(gzip.decompress((out/'report.json.gz').read_bytes()))
    scope=json.loads((diagnostic.HERE/'ths_full_concept_scope_20261017_v1.json').read_bytes())
    boards=[b for b in pq.read_table(scope['source']).to_pylist() if b['source']=='ths' and b['type']=='concept']
    bound={r['code'] for r in json.loads((diagnostic.HERE/'relative_stall_5m_20261018_v1/raw_input_ledger.json').read_bytes())}
    target=pd.read_parquet(diagnostic.fit.previous.prep.auxiliary.OUT/'labels.parquet')
    pred=pd.read_parquet(diagnostic.fit.OUT/'predictions.parquet');keys=['segment','code',diagnostic.fit.previous.prep.TIME]
    pd.testing.assert_frame_equal(pred[keys],target[keys])
    blocked=set()
    for code,r in report['static_membership_by_code'].items():
        related=[b for b in boards if code in b['symbols']]
        possible=[b['code'] for b in related if all(c in bound or c==code for c in b['symbols'])]
        assert possible==r['possible_board_codes']
        assert r['rows']==int(target.code.eq(code).sum())
        if related and not possible:blocked.add(code)
    assert target.code.isin(blocked).sum()==report['structurally_blocked_rows']
    for segment,r in report['ridge_blocks'].items():
        p=pred[pred.segment.eq(segment)];y=target[target.segment.eq(segment)]
        np.testing.assert_allclose(r['raw_predicted_mean'],p.ridge_raw.mean(),atol=1e-12,rtol=0)
        np.testing.assert_allclose(sum(f['mean_score_contribution'] for f in r['original_feature_drift']),
                                   r['original_232_mean_contribution'],atol=1e-12,rtol=0)
        np.testing.assert_allclose(r['intercept']+r['original_232_mean_contribution']+r['added_118_mean_residual'],
                                   p.ridge_raw.mean(),atol=1e-12,rtol=0)
        for month,m in r['monthly'].items():
            select=p[diagnostic.fit.previous.prep.TIME].floordiv(100000000).eq(int(month))
            np.testing.assert_allclose(m['raw_predicted_mean'],p.loc[select,'ridge_raw'].mean(),atol=1e-12,rtol=0)
    may=report['lgbm_monthly_errors']['test_202605'];mix=may['train_code_mixture']
    np.testing.assert_allclose(mix['observed_mean']-mix['train_mean'],
        mix['population_mix_component']+mix['within_code_outcome_shift'],atol=1e-14,rtol=0)
    np.testing.assert_allclose(sum(x['bias']*x['rows'] for x in may['per_code'])/may['rows'],
                               may['row_weighted_bias'],atol=1e-14,rtol=0)
    raw=[diagnostic.Path(k) for k in plan['inputs'] if diagnostic.Path(k).suffix=='.parquet'
         and diagnostic.Path(k).parent.name in ('minute_cache_clean','minute','minute_ext','minute_car')]
    answer=dict(status='PASS_STATIC_SET_AND_SCORE_ARITHMETIC_AUDIT_NO_FIT',report_sha256=sha(out/'report.json.gz'),
        structurally_blocked_codes=len(blocked),structurally_blocked_rows=report['structurally_blocked_rows'],
        model_or_admission_changed=False,fit=False,http_requests=0,
        io_wording_correction=dict(original_raw_minute_reads_field_is_imprecise=True,
            actual_bar_tables_parsed=0,explicit_hash_bound_minute_files=len(raw),
            bytes_per_explicit_hash_pass=sum(p.stat().st_size for p in raw),
            successful_diagnostic_explicit_hash_passes=2,
            note='Raw parquet bytes were read for integrity hashes, not parsed. This audit adds another hash pass. No full 4196-file minute replay.'),
        added_118_block_is_algebraic_residual=True,individual_new_field_cause_not_identified=True)
    body=json.dumps(answer,ensure_ascii=False,allow_nan=False,indent=2).encode('utf-8')
    if bytes_used()+len(body)+32*1024>TOTAL_CAP:raise ValueError('REGISTERED_BUDGET_STOP')
    with (diagnostic.HERE/'continuous_motion_limits_acceptance_20261018_v1.json').open('xb') as stream:stream.write(body)
    return answer


if __name__=='__main__':print(json.dumps(run()))
