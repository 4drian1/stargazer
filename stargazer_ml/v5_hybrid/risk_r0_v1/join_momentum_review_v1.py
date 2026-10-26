"""Factorized exact-cutoff join; immutable review population and existing X remain intact."""
from pathlib import Path
import hashlib
import io
import json
import numpy as np
import pandas as pd
from research_budget_v2 import bytes_used, TOTAL_CAP

HERE = Path(__file__).parent
BASE = HERE / 'joint_fit_review_20261017_v1'
MOMENTUM = HERE / 'market_momentum_proposal_20261017_v1'
OUT = HERE / 'joint_momentum_review_20261017_v1'
TIME = 'decision_ts_assumed_end'
CAP, RESERVE = 16 * 1024**2, 1024**2
METRICS = ['rsi14', 'macd_dif_over_close', 'macd_dea_over_close', 'macd_hist_over_close',
           'bbi_slope', 'close_over_bbi_minus_one', 'price_above_bbi']


def digest(body):
    return hashlib.sha256(body).hexdigest()


def project_market(source, existing):
    """Return new columns plus repeated RSI aliases for independent dedup checks."""
    keys = ['symbol', 'frequency_minutes', TIME]
    if source.duplicated(keys).any():
        raise ValueError('DUPLICATE_MARKET_KEY')
    if set(source.symbol) != {'000001.SH', '399006.SZ'} or set(source.frequency_minutes) != {5, 15}:
        raise ValueError('EXPECTED_TWO_INDICES_AND_FREQUENCIES')
    blocks, aliases = [], []
    for symbol, name in [('000001.SH', 'sse'), ('399006.SZ', 'cyb')]:
        for step in [5, 15]:
            data = source[(source.symbol == symbol) & (source.frequency_minutes == step)].copy()
            # Decisions are on the original completed-15m grid; never nearest/asof join.
            minute = data[TIME] // 100 % 100
            data = data[minute % 15 == 0].set_index(TIME)
            columns = {}
            for metric in METRICS:
                target = f'{name}_{step}m_{metric}'
                if target in existing:
                    aliases.append(data[['intraday_' + metric]].rename(columns={'intraday_' + metric: target}))
                else:
                    columns[target] = data['intraday_' + metric]
            if step == 15:
                for layer in ['prior_daily', 'provisional_daily']:
                    for metric in METRICS:
                        column = f'{layer}_{metric}'
                        target = f'{name}_{column}'
                        if name == 'sse' and layer == 'prior_daily' and metric == 'rsi14':
                            aliases.append(data[[column]].rename(columns={column: 'sse_15m_daily_rsi14_asof_prior_day'}))
                        else:
                            columns[target] = data[column]
                columns[f'{name}_provisional_daily_rsi14_change_from_prior'] = data.provisional_daily_rsi14_change_from_prior
            blocks.append(pd.DataFrame(columns, index=data.index))
    result = pd.concat(blocks, axis=1, join='outer').sort_index()
    if result.columns.duplicated().any() or set(result) & set(existing):
        raise ValueError('REPEATED_FEATURE_ALIAS')
    return result.reset_index(), pd.concat(aliases, axis=1).sort_index().reset_index()


def attach(base, context):
    if base.duplicated(['code', 'cutoff_index']).any():
        raise ValueError('DUPLICATE_COHORT_KEY')
    if context.duplicated(TIME).any():
        raise ValueError('DUPLICATE_CONTEXT_CUTOFF')
    if (set(base) & set(context)) != {TIME}:
        raise ValueError('EXISTING_COLUMN_COLLISION')
    result = base.merge(context, on=TIME, how='left', sort=False, validate='many_to_one')
    pd.testing.assert_frame_equal(result[list(base)], base.reset_index(drop=True))
    return result


def read_segment(segment, directory=OUT):
    """Public logical reader, not a training entrypoint."""
    if segment not in ('train', 'select', 'calibration', 'test'):
        raise ValueError('UNKNOWN_SEGMENT')
    manifest = json.loads((directory / 'manifest.json').read_bytes())
    if manifest['status'] != 'COMPLETE_RESEARCH_CANDIDATE_JOIN_NOT_FIT_READY':
        raise ValueError('JOIN_NOT_COMPLETE')
    refs = manifest['inputs']
    base_path = Path(manifest['base_directory']) / (segment + '.parquet')
    context_path = directory / 'market_features_by_cutoff.parquet'
    for path, expected in [(base_path, refs[str(base_path)]),
                           (context_path, manifest['outputs'][context_path.name]['sha256'])]:
        if digest(path.read_bytes()) != expected:
            raise ValueError('FROZEN_ARTIFACT_DRIFT:' + str(path))
    return attach(pd.read_parquet(base_path), pd.read_parquet(context_path))


def run():
    if OUT.exists():
        raise ValueError('OUTPUT_EXISTS_NO_OVERWRITE')
    refs = {}
    for root in [BASE, MOMENTUM]:
        for name in ['manifest.json', 'readback_acceptance.json']:
            path = root / name
            refs[str(path)] = digest(path.read_bytes())
        if not json.loads((root / 'readback_acceptance.json').read_bytes())['status'].startswith('PASS'):
            raise ValueError('INPUT_NOT_ACCEPTED')
    base_manifest = json.loads((BASE / 'manifest.json').read_bytes())
    momentum_manifest = json.loads((MOMENTUM / 'manifest.json').read_bytes())
    for name, sha in momentum_manifest['inputs'].items():
        if digest(Path(name).read_bytes()) != sha:
            raise ValueError('MOMENTUM_INPUT_DRIFT')
    for name in ['join_momentum_review_v1.py', 'test_join_momentum_review_v1.py',
                 'contracts/JOINT_RISK_FIT_REVIEW_V1.json', 'contracts/MULTITIMEFRAME_MOMENTUM_SPEC_V1.json']:
        path = HERE / name
        refs[str(path)] = digest(path.read_bytes())
    momentum_path = MOMENTUM / momentum_manifest['output']['path']
    refs[str(momentum_path)] = digest(momentum_path.read_bytes())
    if refs[str(momentum_path)] != momentum_manifest['output']['sha256']:
        raise ValueError('MOMENTUM_OUTPUT_DRIFT')
    context, aliases = project_market(pd.read_parquet(momentum_path), base_manifest['fields'])
    new_fields = [c for c in context if c != TIME]
    if len(new_fields) != 54:
        raise ValueError('UNEXPECTED_CANDIDATE_FIELD_COUNT')
    summary, audits = {}, {}
    for segment, expected in base_manifest['summary'].items():
        path = BASE / (segment + '.parquet')
        body = path.read_bytes()
        refs[str(path)] = digest(body)
        if refs[str(path)] != base_manifest['outputs'][path.name]['sha256']:
            raise ValueError('BASE_SEGMENT_DRIFT')
        base = pd.read_parquet(io.BytesIO(body))
        if not base[TIME].isin(context[TIME]).all():
            raise ValueError('COHORT_CUTOFF_NOT_IN_SIDE_TABLE')
        result = attach(base, context)
        repeated = base[[TIME]].merge(aliases, on=TIME, how='left', sort=False, validate='many_to_one')
        for field in aliases.columns.drop(TIME):
            np.testing.assert_allclose(base[field], repeated[field], equal_nan=True, rtol=0, atol=1e-10)
        if np.isinf(result[new_fields].to_numpy(dtype=float)).any():
            raise ValueError('INFINITE_FEATURE')
        if len(result) != expected['rows'] or int(result.y_tail_1200m.sum()) != expected['positive']:
            raise ValueError('COHORT_CHANGED')
        summary[segment] = dict(rows=len(result), positive=int(result.y_tail_1200m.sum()),
            dates=int((result[TIME]//1000000).nunique()), stocks=int(result.code.nunique()),
            new_fields_all_present_rows=int(result[new_fields].notna().all(axis=1).sum()),
            missing_entire_new_family_rows=int(result[new_fields].isna().all(axis=1).sum()))
        audits[segment] = {f:dict(missing=int(result[f].isna().sum()), distinct=int(result[f].nunique())) for f in new_fields}
    config = json.loads((HERE / 'contracts/JOINT_RISK_FIT_REVIEW_V1.json').read_bytes())
    config.update(version='JOINT_RISK_FIT_REVIEW_MOMENTUM_V2',
        parent_config='contracts/JOINT_RISK_FIT_REVIEW_V1.json',
        logical_reader='join_momentum_review_v1.read_segment',
        joined_market_parameter_status='PROPOSED_NOT_FROZEN',
        new_features_materialized=True)
    config['feature_groups']['market'].extend(new_fields)
    config['not_materialized_as_confirmed_rules']['divergence_confirmed'] = 'Explicit-parameter kernel tested; real confirmed events still NULL pending parameters'
    sink = io.BytesIO()
    context.to_parquet(sink, index=False, compression='zstd')
    payloads = {'market_features_by_cutoff.parquet': sink.getvalue(),
                'fit_review_config_v2.json': json.dumps(config, ensure_ascii=False, allow_nan=False, indent=2).encode()}
    report = dict(status='COMPLETE_RESEARCH_CANDIDATE_JOIN_NOT_FIT_READY', inputs=refs,
        base_directory=str(BASE), fields=base_manifest['fields'] + new_fields, field_count=100,
        new_fields=new_fields, market_cutoffs=len(context), matrix_rows=sum(s['rows'] for s in summary.values()),
        summary=summary, field_audit=audits, original_columns_unchanged=True,
        repeated_rsi_alias_checks='ALL_COHORT_ROWS_PASS', feature_missing_rows_dropped=0,
        fit_run=False, real_preprocessing_fitted=False, training_ready=False, http_requests=0,
        outputs={name:dict(bytes=len(body), sha256=digest(body)) for name,body in payloads.items()})
    payloads['manifest.json'] = json.dumps(report, ensure_ascii=False, allow_nan=False, indent=2).encode()
    if sum(map(len,payloads.values())) + RESERVE > CAP or bytes_used() + sum(map(len,payloads.values())) + RESERVE > TOTAL_CAP:
        raise ValueError('RESEARCH_BUDGET_STOP')
    if any(digest(Path(name).read_bytes()) != sha for name,sha in refs.items()):
        raise ValueError('INPUT_CHANGED_DURING_JOIN')
    OUT.mkdir(exist_ok=False)
    for name, body in payloads.items():
        with (OUT/name).open('xb') as stream:
            stream.write(body)
    for segment in summary:
        actual = read_segment(segment)
        original = pd.read_parquet(BASE/(segment+'.parquet'))
        pd.testing.assert_frame_equal(actual[list(original)], original)
        assert len(actual) == summary[segment]['rows']
    acceptance = json.dumps(dict(status='PASS_EXACT_CUTOFF_ALL_COHORT_ROWS',
        manifest_sha256=digest(payloads['manifest.json']), rows=report['matrix_rows'],
        fields=100, missing_rows_dropped=0, old_columns_unchanged=True, training_ready=False), indent=2).encode()
    if bytes_used()+len(acceptance)>TOTAL_CAP:
        raise ValueError('RESEARCH_BUDGET_STOP')
    with (OUT/'readback_acceptance.json').open('xb') as stream:
        stream.write(acceptance)
    print(json.dumps(dict(status=report['status'], fields=100, rows=report['matrix_rows'],
                         market_cutoffs=len(context), bytes=sum(map(len,payloads.values()))+len(acceptance),summary=summary)))


if __name__ == '__main__':
    run()
