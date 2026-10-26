"""Read back conditional returns and reconstruct coverage without the build helper."""
from decimal import Decimal
import json

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

import bind_prior_anchor_evidence_v1 as build


def run():
    out = build.OUT
    destination = build.HERE/'prior_anchor_evidence_acceptance_20261018_v1.json'
    if destination.exists():
        raise ValueError('ACCEPTANCE_EXISTS_NO_OVERWRITE')
    report = json.loads((out/'report.json').read_bytes())
    manifest = json.loads((out/'manifest.json').read_bytes())
    terminal = json.loads((out/'stop_record.json').read_bytes())
    if terminal['status'] != 'COMPLETE_NOT_FIT' or build.prior.old.prep.sha(out/'manifest.json') != terminal['manifest_sha256']:
        raise ValueError('INVALID_TERMINAL')
    for path, expected in report['inputs'].items():
        if build.prior.old.prep.sha(path) != expected:
            raise ValueError('INPUT_DRIFT:' + path)
    for name, item in manifest['outputs'].items():
        if build.prior.old.prep.sha(out/name) != item['sha256']:
            raise ValueError('OUTPUT_DRIFT:' + name)
    candidate = pd.read_parquet(out/'anchor_admission.parquet')
    parent = pd.read_parquet(build.prior.OUT/'anchor_potential.parquet')
    pd.testing.assert_frame_equal(candidate[parent.columns], parent)
    if len(candidate) != 65293 or candidate.coverage_verified.any() or candidate.formal_admitted.any():
        raise ValueError('CANDIDATE_POPULATION_OR_CERTIFICATION_CHANGED')
    if candidate.status.value_counts().to_dict() != report['status_counts']:
        raise ValueError('STATUS_COUNT_MISMATCH')
    values = pd.read_parquet(out/'conditional_reference_returns.parquet')
    if values.duplicated(['code', build.TIME]).any():
        raise ValueError('DUPLICATE_RETURN_KEY')
    clock = pd.read_parquet(build.prior.old.SOURCE/'shared_market_and_leaders.parquet',
                            columns=['cutoff_index', build.TIME]).sort_values('cutoff_index')
    day = clock[build.TIME].to_numpy()[values.cutoff_index.to_numpy()] // 1000000
    values['current_session_day'] = day
    joined = values.merge(candidate[['code','current_session_day','status',
        'reference_close_potential','ordinary_factor_ratio']], on=['code','current_session_day'],
        how='left', validate='many_to_one', suffixes=('', '_admission'))
    if not joined.status.eq('CONDITIONAL_RESEARCH').all():
        raise ValueError('RETURN_OUTSIDE_CONDITIONAL_SCOPE')
    for col in ['reference_close_potential', 'ordinary_factor_ratio']:
        np.testing.assert_allclose(joined[col], joined[col+'_admission'], atol=0, rtol=0)
    independent = np.array([float(Decimal(str(c))/Decimal(str(a))*Decimal(str(m))-1)
        for c,a,m in zip(values.stock_close, values.reference_close_potential, values.ordinary_factor_ratio)])
    np.testing.assert_allclose(values.conditional_day_return, independent, rtol=0, atol=1e-12)
    # Verify source rows were genuinely missing the old reference and price unchanged.
    old_manifest = json.loads((build.prior.old.SOURCE/'manifest.json').read_bytes())
    checked = 0
    for part in old_manifest['joint_parts']:
        f = pd.read_parquet(build.prior.old.SOURCE/part['path'],
            columns=['code','cutoff_index','stock_close','stock_reference_close_raw'])
        choose = values[values.code.isin(f.code.unique())]
        actual = choose.merge(f, on=['code','cutoff_index'], how='left', validate='one_to_one',
                              suffixes=('', '_old'), indicator=True)
        if not actual['_merge'].eq('both').all() or actual.stock_reference_close_raw.notna().any():
            raise ValueError('SOURCE_REFERENCE_NOT_MISSING_OR_KEY_ABSENT')
        np.testing.assert_allclose(actual.stock_close, actual.stock_close_old, rtol=0, atol=0)
        checked += len(actual)
    if checked != len(values) or checked != report['research_return_rows']:
        raise ValueError('SOURCE_REPLAY_COUNT_MISMATCH')
    spec = json.loads((build.HERE/'ths_full_concept_scope_20261017_v1.json').read_bytes())
    boards = [b for b in pq.read_table(spec['source']).to_pylist() if b['source']=='ths' and b['type']=='concept']
    # Independent expansion of inverse member counts, without sector_weights().
    weight = {}
    for board in boards:
        for code in board['symbols']:
            weight[code] = weight.get(code, 0.) + 1 / len(board['symbols']) / len(boards)
    coverage = pd.read_parquet(out/'coverage_increment.parquet')
    recomputed = values.assign(weight=values.code.map(weight)).groupby(build.TIME).weight.sum()
    recomputed = recomputed.reindex(coverage[build.TIME], fill_value=0.).to_numpy()
    np.testing.assert_allclose(coverage.conditional_research_gain, recomputed, rtol=0, atol=1e-12)
    fitted = build.HERE/'multiboard_three_arm_fit_20261018_v1'
    artifacts = json.loads((fitted/'artifact_manifest.json').read_bytes())['outputs']
    for name, item in artifacts.items():
        if build.prior.old.prep.sha(fitted/name) != item['sha256']:
            raise ValueError('OLD_MODEL_ARTIFACT_CHANGED')
    d2 = build.HERE/'postclose_repaired_research_fit_20261017_v1/model.txt'
    if build.prior.old.prep.sha(d2) != '5d8402cd530cb83778d901e33eb87788f21ebc8619414f335b6d56905a81e005':
        raise ValueError('D2_CHANGED')
    result = dict(status='PASS_CONDITIONAL_REFERENCE_SIDECAR_NOT_FIT', tests=report['tests'],
        candidates_preserved=len(candidate), independent_decimal_returns=checked,
        independently_reconstructed_coverage_cutoffs=len(coverage), input_hashes_verified=len(report['inputs']),
        old_fit_artifacts_unchanged=len(artifacts), d2_unchanged=True,
        formal_coverage_verified=False, http_requests=0, fit=False,
        report_sha256=build.prior.old.prep.sha(out/'report.json'),
        auditor_sha256=build.prior.old.prep.sha(build.HERE/'audit_prior_anchor_evidence_v1.py'))
    body = build.prior.old.prep.encoded(result)
    size = sum(p.stat().st_size for p in out.iterdir() if p.is_file())
    if size+len(body)+build.RESERVE > build.CAP or build.prior.old.bytes_used()+len(body)+build.RESERVE > build.prior.old.TOTAL_CAP:
        raise ValueError('ACCEPTANCE_BUDGET_STOP')
    with destination.open('xb') as stream:
        stream.write(body)
    print(json.dumps(result))


if __name__ == '__main__':
    run()
