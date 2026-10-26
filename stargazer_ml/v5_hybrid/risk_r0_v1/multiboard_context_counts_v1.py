"""Expose exact group denominators as audit metadata, never extra model X."""
import numpy as np
import pandas as pd


def integer_counts(values):
    values = np.asarray(values, dtype=float)
    valid = np.isfinite(values)
    if (values[valid] < 0).any() or (np.abs(values[valid] - np.rint(values[valid])) > 1e-6).any():
        raise ValueError('RATIO_DOES_NOT_RECONSTRUCT_INTEGER_COUNT')
    return pd.array(np.rint(values), dtype='Int64')


def reconstruct(frame, prefix, catalog_count):
    """Rates remain untouched. Zero *known* outcomes is not zero actual risk."""
    size = np.broadcast_to(np.asarray(catalog_count, dtype=float), (len(frame),))
    integer_counts(size)
    def ratio(name):
        value = frame[prefix + name].to_numpy(dtype=float)
        finite = np.isfinite(value)
        if ((value[finite] < -1e-12) | (value[finite] > 1 + 1e-12)).any():
            raise ValueError('RATIO_OUTSIDE_UNIT_INTERVAL')
        return value
    observed = np.where(size == 0, 0., ratio('observed_share') * size)
    positive = np.where(observed == 0, 0., ratio('positive_share') * observed)
    result = {prefix + 'catalog_count': integer_counts(size),
              prefix + 'observed_count': integer_counts(observed),
              prefix + 'positive_count': integer_counts(positive)}
    names = [f'{benchmark}_positive_{flag}' for benchmark in ('sse', 'cyb')
             for flag in ('nonpositive', 'falling', 'not_new_high')]
    names += ['current_change_negative', 'captured_change_negative']
    for name in names:
        known = np.where(positive == 0, 0., ratio(name + '_coverage') * positive)
        true_known = np.where(known == 0, 0., ratio(name + '_rate') * known)
        result[prefix + name + '_known_count'] = integer_counts(known)
        result[prefix + name + '_true_known_count'] = integer_counts(true_known)
        result[prefix + name + '_unknown_count'] = integer_counts(positive - known)
    return pd.DataFrame(result, index=frame.index)


def read_counts(segment):
    """Separate keyed audit view; original feature values and X lists unchanged."""
    import json
    from pathlib import Path
    import pyarrow.parquet as pq
    from build_multiboard_full_features_v1 import HERE, sha
    from intraday_relative_stall_v1 import TIME
    from read_multiboard_full_features_v1 import metadata, read_segment
    report, _ = metadata()
    frame = read_segment(segment)
    scope_path = HERE / 'ths_full_concept_scope_20261017_v1.json'
    if sha(scope_path) != report['inputs'][str(scope_path.resolve())]:
        raise ValueError('CATALOG_SCOPE_CHANGED')
    scope = json.loads(scope_path.read_bytes())
    path = Path(scope['source'])
    if sha(path) != scope['source_sha256']:
        raise ValueError('CATALOG_MEMBERSHIP_CHANGED')
    boards = [b for b in pq.read_table(path).to_pylist() if b['source'] == 'ths' and b['type'] == 'concept']
    affiliations = {}
    for board in boards:
        for code in board['symbols']:
            affiliations[code] = affiliations.get(code, 0) + 1
    own_size = frame.code.map(affiliations).to_numpy(dtype=float)
    blocks = [frame[['code', TIME]]]
    for frequency in (5, 15):
        blocks.append(reconstruct(frame, f'mb{frequency}_', len(boards)))
        blocks.append(reconstruct(frame, f'own{frequency}_', own_size))
    return pd.concat(blocks, axis=1)


def accept():
    import io
    import json
    from pathlib import Path
    import unittest
    from build_multiboard_full_features_v1 import HERE, sha
    from research_budget_v2 import bytes_used, TOTAL_CAP
    target = HERE / 'multiboard_group_counts_acceptance_20261018_v1.json'
    if target.exists():
        raise ValueError('ACCEPTANCE_EXISTS_NO_OVERWRITE')
    suite = unittest.defaultTestLoader.loadTestsFromName('test_multiboard_context_counts_v1')
    log = io.StringIO()
    tests = unittest.TextTestRunner(stream=log, verbosity=2).run(suite)
    if not tests.wasSuccessful() or tests.skipped:
        raise ValueError('GROUP_COUNTS_TESTS_FAILED_OR_SKIPPED')
    result = dict(status='PASS_AUDIT_DENOMINATORS_NOT_NEW_X_OR_FIT', tests=tests.testsRun,
        test_output=log.getvalue(), http_requests=0, fit=False,
        refs={str(p): sha(p) for p in [Path(__file__), HERE / 'test_multiboard_context_counts_v1.py',
            HERE / 'multiboard_full_features_20261018_v1/report.json',
            HERE / 'read_multiboard_full_features_v1.py']})
    body = json.dumps(result, ensure_ascii=False, allow_nan=False, indent=2).encode('utf-8')
    if len(body) > 64 * 1024 or bytes_used() + len(body) > TOTAL_CAP:
        raise ValueError('COUNTS_ACCEPTANCE_BUDGET_STOP')
    with target.open('xb') as stream:
        stream.write(body)
    print(json.dumps(dict(status=result['status'], tests=tests.testsRun)))


if __name__ == '__main__':
    accept()
