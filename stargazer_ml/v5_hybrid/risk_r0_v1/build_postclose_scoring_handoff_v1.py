"""Package existing research weights and replay saved predictions; never fit."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from postclose_research_scorer_v1 import HERE, CANDIDATE, ResearchScorer, encode, sha
from postclose_repaired_training_v1 import prepare, OUT as FIT
from read_postclose_daily_context_repair_v1 import read_repaired_population
from research_budget_v2 import bytes_used, TOTAL_CAP

OUT = HERE / 'postclose_scoring_handoff_20261017_v1'
CAP = 2 * 1024**2


def make_manifest(fields, codes):
    audit = json.loads((FIT / 'comparison_and_readback.json').read_bytes())
    artifacts = {}
    for name, filename in [('model', 'model.txt'), ('preprocessing', 'preprocessing.json'),
                           ('plan', 'plan.json'), ('report', 'report.json'),
                           ('predictions', 'predictions.parquet')]:
        path = FIT / filename
        fingerprint = sha(path.read_bytes())
        if fingerprint != audit['new_fit_sha256'][filename]:
            raise ValueError('AUDITED_CANDIDATE_DRIFT:' + filename)
        artifacts[name] = dict(path=str(path), sha256=fingerprint)
    for name, path in [('runtime', HERE / 'postclose_research_scorer_v1.py'),
                       ('contract', HERE / 'contracts/POSTCLOSE_READONLY_SCORING_V1.json'),
                       ('builder', Path(__file__)),
                       ('tests', HERE / 'test_postclose_research_scorer_v1.py')]:
        artifacts[name] = dict(path=str(path), sha256=sha(path.read_bytes()))
    return dict(candidate_id=CANDIDATE, status='RESEARCH_ONLY_NOT_DEPLOYED', fields=fields,
                feature_schema_sha256=sha(encode(fields)), research_codes=sorted(codes), artifacts=artifacts,
                candidate_selection='Existing D2 fixed baseline retained; D1/D3 not promoted by test selection',
                independent_validation_passed=False, deployment_authorized=False,
                limitations=['Historical arrival times unknown; endpoint assumed, not measured',
                             'Snapshot members and raw partial context carry disclosed reconstruction bias',
                             'Unknown resealing remains missing; Q90 calibration unstable across months',
                             '43-code evidence domain is not an all-market validated model',
                             'Manifest hashes prove integrity, not authenticity or PIT validity'])


def input_for(frame, manifest):
    rows = []
    fields = manifest['fields']
    for _, row in frame.iterrows():
        features = {f: None if pd.isna(row[f]) else float(row[f]) for f in fields}
        rows.append(dict(code=str(row.code), day=int(row.day),
                         feature_asof_ts=int(row.day) * 1000000 + 150000,
                         decision_available_ts=None, features=features))
    return dict(candidate_id=CANDIDATE, feature_schema_sha256=manifest['feature_schema_sha256'],
                usage_mode='HISTORICAL_RECONSTRUCTION', rows=rows)


def run():
    if OUT.exists():
        raise ValueError('OUTPUT_EXISTS_NO_OVERWRITE')
    if bytes_used() + CAP > TOTAL_CAP:
        raise ValueError('HANDOFF_BUDGET_RESERVATION_FAILED')
    fields, _, parts, _ = prepare()
    data, _ = read_repaired_population()
    manifest = make_manifest(fields, data.code.unique().tolist())
    # All known code scope comes from the preserved full population, not test survivors.
    if len(manifest['research_codes']) != 43:
        raise ValueError('SCOPE_DRIFT')
    OUT.mkdir(exist_ok=False)

    def write(name, value):
        body = encode(value)
        reserve = 16384 if name != 'stop_record.json' else 0
        current = sum(p.stat().st_size for p in OUT.rglob('*') if p.is_file())
        if current + len(body) + reserve > CAP or bytes_used() + len(body) + reserve > TOTAL_CAP:
            raise ValueError('HANDOFF_BUDGET_STOP')
        with (OUT / name).open('xb') as stream:
            stream.write(body)

    try:
        write('manifest.json', manifest)
        scorer = ResearchScorer(OUT / 'manifest.json')
        actual, keys = [], []
        frame = parts['test'].reset_index(drop=True)
        for start in range(0, len(frame), 128):
            result = scorer.score(input_for(frame.iloc[start:start + 128], manifest))
            for row in result['rows']:
                if row['status'] != 'RESEARCH_SCORE':
                    raise ValueError('SAVED_ACCEPTED_ROW_REJECTED:' + str(row['reasons']))
                actual.append(row['q90_next_session_adverse_excursion'])
                keys.append((row['code'], row['day']))
        saved = pd.read_parquet(FIT / 'predictions.parquet')
        saved = saved[saved.segment.eq('test')].set_index(['code', 'day'])
        expected = saved.loc[keys, 'q90_adverse'].to_numpy()
        error = float(np.max(np.abs(np.asarray(actual) - expected)))
        if error > 1e-12:
            raise ValueError('PREDICTION_REPLAY_MISMATCH')
        write('example_input.json', input_for(frame.iloc[:2], manifest))
        acceptance = dict(status='PASS_ENGINEERING_REPLAY_NOT_INDEPENDENT_VALIDATION',
                          rows=len(frame), dates=int(frame.day.nunique()),
                          max_absolute_prediction_error=error, schema_fields=len(fields),
                          research_scope_codes=len(manifest['research_codes']),
                          candidate_id=CANDIDATE, new_fit=False, network_requests=0,
                          independent_validation_passed=False, deployment_authorized=False,
                          original_artifacts_unchanged=True)
        scorer.verify_artifacts()
        write('acceptance.json', acceptance)
        write('stop_record.json', dict(status='COMPLETE_RESEARCH_HANDOFF_NOT_DEPLOYMENT'))
        return acceptance
    except Exception as exc:
        write('stop_record.json', dict(status='FAILED', error=str(exc)))
        raise


if __name__ == '__main__':
    print(json.dumps(run(), ensure_ascii=False))
