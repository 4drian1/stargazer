"""Sparse acceptance correction: preserve v1 files, normalize only text dtypes."""
from pathlib import Path
import hashlib
import json
import pandas as pd
from materialize_left_side_candidates_v1 import build, PRICE, MOMENTUM, TIME
from research_budget_v2 import bytes_used, TOTAL_CAP

HERE = Path(__file__).parent
BASE = HERE / 'left_side_market_20261017_v1'
OUT = HERE / 'left_side_market_20261017_v2'


def digest(body):
    return hashlib.sha256(body).hexdigest()


def normalize_text(frame):
    result = frame.copy()
    for name in ['symbol', 'stage', 'signal_status']:
        result[name] = result[name].astype(object)
    return result


def read_stage(stage, directory=OUT):
    if stage not in ('intraday', 'postclose_daily'):
        raise ValueError('UNKNOWN_STAGE')
    manifest = json.loads((directory/'manifest.json').read_bytes())
    if manifest['status'] != 'ACCEPTED_LEFT_SIDE_MARKET_EVIDENCE_NOT_FIT_READY':
        raise ValueError('UNACCEPTED_SIDE_TABLE')
    entry = manifest['artifacts'][stage]
    path = Path(entry['path'])
    if digest(path.read_bytes()) != entry['sha256']:
        raise ValueError('SIDE_TABLE_DRIFT')
    return normalize_text(pd.read_parquet(path))


def run():
    if OUT.exists():
        raise ValueError('OUTPUT_EXISTS_NO_OVERWRITE')
    manifest_path = BASE/'manifest.json'
    old = json.loads(manifest_path.read_bytes())
    for path, sha in old['inputs'].items():
        if digest(Path(path).read_bytes()) != sha:
            raise ValueError('FROZEN_INPUT_DRIFT')
    price = pd.read_parquet(PRICE/'market_price_features.parquet')
    momentum = pd.read_parquet(MOMENTUM/'market_momentum.parquet')
    expected = build(price, momentum)
    artifacts = {}
    for stage, frame in zip(['intraday', 'postclose_daily'], expected):
        path = BASE/(stage+'.parquet')
        if digest(path.read_bytes()) != old['outputs'][path.name]['sha256']:
            raise ValueError('BASE_OUTPUT_DRIFT')
        actual = normalize_text(pd.read_parquet(path))
        pd.testing.assert_frame_equal(actual, normalize_text(frame))
        if actual.duplicated(['symbol', 'frequency_minutes', TIME]).any():
            raise ValueError('DUPLICATE_STAGE_KEY')
        assert (actual.right_confirmation_bars == 0).all() and not actual.training_admitted.any()
        artifacts[stage] = dict(path=str(path), sha256=digest(path.read_bytes()), rows=len(actual))
    old_review = HERE/'joint_momentum_review_20261017_v1/manifest.json'
    frozen = json.loads(old_review.read_bytes())
    assert all(digest(Path(p).read_bytes()) == h for p,h in frozen['inputs'].items())
    for name, entry in frozen['outputs'].items():
        assert digest((old_review.parent/name).read_bytes()) == entry['sha256']
    refs = {str(manifest_path):digest(manifest_path.read_bytes()),str(old_review):digest(old_review.read_bytes())}
    for name in ['accept_left_side_market_v2.py','test_accept_left_side_market_v2.py']:
        path = HERE/name
        refs[str(path)] = digest(path.read_bytes())
    manifest = dict(status='ACCEPTED_LEFT_SIDE_MARKET_EVIDENCE_NOT_FIT_READY', inputs=refs,
        artifacts=artifacts, summary=old['summary'], acceptance_cases=old['acceptance_cases'],
        correction='V1_READBACK_STRING_DTYPE_ONLY; BUSINESS_AND_NUMERIC_VALUES_UNCHANGED',
        text_normalization_columns=['symbol','stage','signal_status'],
        v1_preserved=True, right_confirmation_bars=0, training_ready=False, fit_run=False,
        postclose_stock_population_built=False, http_requests=0)
    report = dict(status='PASS_ALL_ROWS_VALUE_AND_SCHEMA_NORMALIZED_READBACK',
        rows={s:a['rows'] for s,a in artifacts.items()}, source_recomputed_all_rows=True,
        numeric_checks_relaxed=False, old_100_column_review_unchanged=True,
        training_ready=False)
    bodies = {'manifest.json':json.dumps(manifest,ensure_ascii=False,allow_nan=False,indent=2).encode()}
    report['manifest_sha256'] = digest(bodies['manifest.json'])
    bodies['readback_acceptance.json'] = json.dumps(report,ensure_ascii=False,indent=2).encode()
    if sum(map(len,bodies.values()))>1024**2 or bytes_used()+sum(map(len,bodies.values()))>TOTAL_CAP:
        raise ValueError('RESEARCH_BUDGET_STOP')
    OUT.mkdir(exist_ok=False)
    for name,body in bodies.items():
        with (OUT/name).open('xb') as stream:stream.write(body)
    for stage in artifacts:
        assert len(read_stage(stage)) == artifacts[stage]['rows']
    print(json.dumps(dict(status=manifest['status'],rows=report['rows'],bytes=sum(map(len,bodies.values())),
                         research_bytes=bytes_used(),cap=TOTAL_CAP)))


if __name__=='__main__':run()
