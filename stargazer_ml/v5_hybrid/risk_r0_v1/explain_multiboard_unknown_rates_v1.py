"""Separate empty positive groups from nonempty groups with untestable signals."""
import json
import numpy as np

import diagnose_multiboard_transfer_v1 as diag
from research_budget_v2 import bytes_used, TOTAL_CAP


def reasons(frame, field):
    prefix = field.split('_', 1)[0] + '_'
    signal, positive, coverage = (frame[name].to_numpy(dtype=float) for name in
        [field, prefix + 'positive_share', field[:-5] + '_coverage'])
    unknown = ~np.isfinite(signal)
    masks = dict(known=~unknown,
                 no_current_positive_group=unknown & (positive == 0),
                 nonempty_group_no_evaluable_signal=unknown & (positive > 0) & (coverage == 0))
    masks['unresolved_other'] = ~(masks['known'] | masks['no_current_positive_group'] | masks['nonempty_group_no_evaluable_signal'])
    return {name: int(mask.sum()) for name, mask in masks.items()}


def run():
    import pandas as pd
    target = diag.HERE / 'multiboard_transfer_unknown_reasons_20261018_v1.json'
    if target.exists():
        raise ValueError('SUPPLEMENT_EXISTS_NO_OVERWRITE')
    accepted_path = diag.HERE / 'multiboard_transfer_diagnosis_acceptance_20261018_v1.json'
    accepted = json.loads(accepted_path.read_bytes())
    if diag.prep.sha(diag.OUT / 'diagnostics.json') != accepted['report_sha256']:
        raise ValueError('DIAGNOSTIC_REPORT_DRIFT')
    core = json.loads((diag.OUT / 'diagnostics.json').read_bytes())
    population_path = diag.prep.OUT / 'population_audit.parquet'
    population = pd.read_parquet(population_path)
    records = {}
    for segment in diag.prep.SEGMENTS:
        full = diag.prep.read_segment(segment)
        stored = population[population.proposed_segment_name.eq(segment)].reset_index(drop=True)
        pd.testing.assert_frame_equal(full[['code', diag.prep.TIME]], stored[['code', diag.prep.TIME]])
        frame = full.loc[stored.common_main_eligible.to_numpy()].reset_index(drop=True)
        records[segment] = {}
        for field, summary in core['signals'][segment].items():
            unit, _ = diag.signal_unit(frame, field)
            counts = reasons(unit, field)
            if sum(counts.values()) != summary['observations'] or counts['known'] != sum(
                    summary['states'][state]['observations'] for state in ['known_zero', 'active']):
                raise ValueError('UNKNOWN_REASONS_DO_NOT_RECONCILE')
            records[segment][field] = counts
    record = dict(status='PASS_UNKNOWN_REASON_PARTITION_NOT_NEW_X', segments=records,
        refs={str(p): diag.prep.sha(p) for p in [accepted_path, diag.OUT / 'diagnostics.json',
             population_path, diag.HERE / 'explain_multiboard_unknown_rates_v1.py',
             diag.HERE / 'test_explain_multiboard_unknown_rates_v1.py']},
        new_fit=False, labels_changed=False, http_requests=0)
    body = diag.prep.encoded(record)
    current = sum(p.stat().st_size for p in diag.OUT.rglob('*') if p.is_file())
    if current + len(body) + diag.RESERVE > diag.CAP or bytes_used() + len(body) + diag.RESERVE > TOTAL_CAP:
        raise ValueError('SUPPLEMENT_BUDGET_STOP')
    with target.open('xb') as stream:
        stream.write(body)
    print(json.dumps(dict(status=record['status'], bytes=len(body))))


if __name__ == '__main__':
    run()
