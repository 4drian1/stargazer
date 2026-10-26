"""Freeze evaluation preparation references without accessing new labels or sources."""
import json
from pathlib import Path

import pandas as pd

from postclose_next_period_evaluation_v1 import CONTRACT, PACKAGE, preparation_check
from postclose_research_scorer_v1 import HERE, encode, sha, strict_json
from research_budget_v2 import TOTAL_CAP, bytes_used

OUT = HERE / 'postclose_next_period_preparation_20261017_v1'
EXPOSURE = HERE / 'postclose_temporal_validation_plan_20261017_v1/exposure_ledger.parquet'
CAP = 2 * 1024**2


def run():
    if OUT.exists():
        raise ValueError('OUTPUT_EXISTS_NO_OVERWRITE')
    if bytes_used() + CAP > TOTAL_CAP:
        raise ValueError('PREPARATION_BUDGET_RESERVATION_FAILED')
    spec = strict_json(CONTRACT.read_bytes())
    exposure = pd.read_parquet(EXPOSURE)
    if int(exposure.day.max()) != spec['previously_accessed_through']:
        raise ValueError('EXPOSURE_CUTOFF_DRIFT')
    files = [CONTRACT, PACKAGE, EXPOSURE, Path(__file__),
             HERE / 'postclose_next_period_evaluation_v1.py',
             HERE / 'test_postclose_next_period_evaluation_v1.py',
             HERE / 'diagnose_postclose_predictions_v1.py']
    refs = {str(p):sha(p.read_bytes()) for p in files}
    report = dict(**preparation_check(), frozen_references=refs,
                  exposure=dict(rows=len(exposure), dates=int(exposure.day.nunique()),
                                stocks=int(exposure.code.nunique()),
                                latest_accessed_day=int(exposure.day.max())),
                  validation_method_implemented=True,
                  prospective_scoring_implemented=False,
                  actual_next_period_population_rows=None,
                  window=spec['window_proposal'],
                  fit_run=False, calibration_run=False, collection_started=False,
                  network_requests=0)
    body = encode(report)
    if len(body) > CAP or bytes_used() + len(body) > TOTAL_CAP:
        raise ValueError('PREPARATION_OUTPUT_BUDGET_STOP')
    OUT.mkdir(exist_ok=False)
    with (OUT / 'preparation.json').open('xb') as stream:
        stream.write(body)
    if any(sha(Path(p).read_bytes()) != h for p,h in refs.items()):
        raise ValueError('FROZEN_INPUT_CHANGED_DURING_PREPARATION')
    return report


if __name__ == '__main__':
    print(json.dumps(run(), ensure_ascii=False, allow_nan=False))
