"""Bind existing conditional evidence to anchor candidates; immutable research sidecar."""
import gzip
import io
import json
from pathlib import Path
import unittest

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

import verify_prior_close_anchor_v1 as prior
from prior_anchor_evidence_v1 import classify, normalized_return

HERE, TIME = prior.HERE, prior.TIME
OUT = HERE/'prior_anchor_evidence_binding_20261018_v1'
CAP, RESERVE = 3 * 1024**2, 64 * 1024


def run():
    if OUT.exists():
        raise ValueError('OUTPUT_EXISTS_NO_RESTART')
    if prior.old.bytes_used() + CAP > prior.old.TOTAL_CAP:
        raise ValueError('REGISTERED_BUDGET_RESERVATION_FAILED')
    log = io.StringIO()
    suite = unittest.defaultTestLoader.loadTestsFromNames([
        'test_prior_anchor_evidence_v1', 'test_prior_close_anchor_v1',
        'test_multiboard_observation_repair_v1'])
    tests = unittest.TextTestRunner(stream=log).run(suite)
    if not tests.wasSuccessful() or tests.skipped:
        raise ValueError('EVIDENCE_ENTRY_TEST_FAILED:' + log.getvalue())
    refs = {}

    def bind(path, expected=None):
        path = Path(path)
        actual = prior.old.prep.sha(path)
        if expected is not None and expected != actual:
            raise ValueError('FROZEN_INPUT_DRIFT:' + str(path))
        refs[str(path)] = actual
        return path

    for name in ['bind_prior_anchor_evidence_v1.py', 'prior_anchor_evidence_v1.py',
                 'test_prior_anchor_evidence_v1.py', 'research_budget_v2.py',
                 'test_prior_close_anchor_v1.py', 'test_multiboard_observation_repair_v1.py']:
        bind(HERE/name)
    accepted = json.loads(bind(HERE/'prior_close_anchor_acceptance_20261018_v1.json').read_bytes())
    report = json.loads(bind(prior.OUT/'report.json', accepted['report_sha256']).read_bytes())
    for path, expected in report['inputs'].items():
        bind(path, expected)
    old_manifest = json.loads((prior.old.SOURCE/'manifest.json').read_bytes())
    parent = json.loads(bind(prior.OUT/'manifest.json').read_bytes())
    for name, item in parent['outputs'].items():
        bind(prior.OUT/name, item['sha256'])
    labels = json.loads(bind(HERE/'joint_decision_after_labels_20261017_v2/manifest.json').read_bytes())
    ordinary_path = HERE/'historical_archived_corpus_v4_20261015/evidence_and_gate.json'
    ordinary = json.loads(bind(ordinary_path, labels['inputs'][str(ordinary_path)]).read_bytes())
    scope = bind(HERE/'contracts/RESTRICTED_HISTORICAL_RESEARCH_V3.md')
    old_acceptance = json.loads(bind(HERE/'archived_corpus_v4_acceptance_20261015.json').read_bytes())
    for entry in old_acceptance['controls']:
        bind(entry['path'], entry['sha256'])
    clock = pd.read_parquet(prior.old.SOURCE/'shared_market_and_leaders.parquet', columns=['cutoff_index', TIME])
    clock = clock.sort_values('cutoff_index')
    dates = clock[TIME].to_numpy() // 1000000
    candidate = pd.read_parquet(prior.OUT/'anchor_potential.parquet')
    candidate['previous_day'] = dates[candidate.prior_cutoff_index.to_numpy()]
    candidate['status'] = 'PENDING'
    candidate['reason'] = 'NO_BOUND_ORDINARY_ACTION_EVIDENCE'
    candidate['ordinary_factor_ratio'] = np.nan
    audit = ordinary['audit']
    ledger = json.loads(bind(HERE/'relative_stall_5m_20261018_v1/raw_input_ledger.json').read_bytes())
    sources = {s['code']: s for s in ledger}
    label_sources = {s['code']: s for s in labels['source_files']}
    q = pd.read_csv(prior.QUARANTINE, dtype={'code': str})
    quarantine = q.groupby('code').day.apply(set).to_dict()
    used, outputs = 0, {}

    def write(name, body, terminal=False):
        nonlocal used
        reserve = 0 if terminal else RESERVE
        if used + len(body) + reserve > CAP or prior.old.bytes_used() + len(body) + reserve > prior.old.TOTAL_CAP:
            raise ValueError('SERIALIZED_BUDGET_STOP')
        if not OUT.exists():
            OUT.mkdir(exist_ok=False)
        with (OUT/name).open('xb') as stream:
            stream.write(body)
        used += len(body)
        outputs[name] = dict(bytes=len(body), sha256=prior.old.prep.sha(OUT/name))

    write('plan.json', prior.old.prep.encoded(dict(inputs=refs.copy(), tests=tests.testsRun,
        test_output=log.getvalue(), cap=CAP, scope=str(scope),
        no_current_or_future_daily_price_admission=True, formal_coverage_verified=False,
        fit=False, http_requests=0, old_labels_unchanged=True)))
    try:
        selected_codes = sorted(set(candidate.code) & set(audit))
        for code in selected_codes:
            daily_ref = [p for p in labels['inputs'] if p.endswith(f'{code}_daily.json.gz')]
            if len(daily_ref) != 1:
                raise ValueError('DAILY_EVIDENCE_NOT_UNIQUELY_BOUND:' + code)
            daily = bind(daily_ref[0], labels['inputs'][daily_ref[0]])
            bars = {}
            for key, row in json.loads(gzip.decompress(daily.read_bytes())):
                if row is None:
                    continue
                day = int(key.rsplit(':', 1)[1])
                if key != f'日k:{code}:{day}' or row.get('code') != code or row.get('date') != day:
                    raise ValueError('DAILY_IDENTITY_MISMATCH')
                price = {k: row.get(k) for k in ('open','high','low','close')}
                if day in bars and bars[day] != price:
                    raise ValueError('DAILY_PRICE_CONFLICT')
                bars[day] = price
            source, original = sources.get(code), label_sources.get(code)
            source_bound = (source is not None and original is not None
                and Path(source['path']).resolve() == Path(original['path']).resolve()
                and source['sha256'] == original['sha256'])
            if source_bound:
                bind(source['path'], source['sha256'])
            for i, row in candidate[candidate.code.eq(code)].iterrows():
                status, reason, factor = classify(row.reference_close_potential,
                    int(row.previous_day), int(row.current_session_day), bars, audit[code],
                    source_bound, quarantine.get(code, set()))
                candidate.loc[i, ['status', 'reason', 'ordinary_factor_ratio']] = [status, reason, factor]
        eligible = candidate[candidate.status.eq('CONDITIONAL_RESEARCH')].copy()
        if eligible.duplicated(['code', 'current_session_day']).any():
            raise ValueError('DUPLICATE_ADMISSION_KEY')
        candidate['research_eligible'] = candidate.status.eq('CONDITIONAL_RESEARCH')
        candidate['coverage_verified'] = False
        # Formal full-certification field inherited from prior evidence stays false.
        side = pd.read_parquet(prior.OUT/'coverage_potential.parquet')[[TIME]]
        gain = np.zeros(len(side))
        scope_spec = json.loads((HERE/'ths_full_concept_scope_20261017_v1.json').read_bytes())
        boards = [b for b in pq.read_table(scope_spec['source']).to_pylist()
                  if b['source']=='ths' and b['type']=='concept']
        codes = sorted({c for b in boards for c in b['symbols']})
        weights = dict(zip(codes, prior.old.sector_weights(codes, [b['symbols'] for b in boards])))
        current_parts = []
        for part in old_manifest['joint_parts']:
            path = prior.old.SOURCE/part['path']
            # Already pinned and verified through the prior report's inputs.
            f = pd.read_parquet(path, columns=['code','cutoff_index','stock_close','stock_reference_close_raw'])
            f = f[f.code.isin(eligible.code.unique())].copy()
            if f.empty:
                continue
            f['current_session_day'] = dates[f.cutoff_index.to_numpy()]
            f[TIME] = clock[TIME].to_numpy()[f.cutoff_index.to_numpy()]
            f = f[f[TIME].isin(side[TIME]) & f.stock_reference_close_raw.isna()]
            f = f.merge(eligible[['code','current_session_day','reference_close_potential','ordinary_factor_ratio']],
                        on=['code','current_session_day'], how='inner', validate='many_to_one')
            if f.empty:
                continue
            f['conditional_day_return'] = [normalized_return(c,a,m) for c,a,m in zip(
                f.stock_close, f.reference_close_potential, f.ordinary_factor_ratio)]
            current_parts.append(f[['code', TIME, 'cutoff_index', 'stock_close',
                'reference_close_potential', 'ordinary_factor_ratio', 'conditional_day_return']])
            positions = np.searchsorted(side[TIME].to_numpy(), f[TIME])
            np.add.at(gain, positions, f.code.map(weights).to_numpy())
        returns = pd.concat(current_parts, ignore_index=True) if current_parts else pd.DataFrame(columns=[
            'code',TIME,'cutoff_index','stock_close','reference_close_potential','ordinary_factor_ratio','conditional_day_return'])
        if returns.duplicated(['code', TIME]).any():
            raise ValueError('DUPLICATE_CONTEXT_KEY')
        old_cov = pd.read_parquet(prior.old.OUT/'coverage_attribution.parquet')
        np.testing.assert_array_equal(side[TIME], old_cov[TIME])
        if (gain > old_cov.prior_close_unavailable + 1e-12).any():
            raise ValueError('CONDITIONAL_GAIN_EXCEEDS_MISSING_REFERENCE')
        side['old_coverage'] = old_cov.valid_return
        side['conditional_research_gain'] = gain
        side['conditional_research_coverage'] = side.old_coverage + gain
        population = pd.read_parquet(prior.old.prep.OUT/'population_audit.parquet')
        joined = population.merge(side, on=TIME, validate='many_to_one')
        summary = {s: {c: float(joined.loc[joined.proposed_segment_name.eq(s)
            & joined.common_main_eligible, c].mean()) for c in side if c != TIME}
            for s in prior.old.prep.SEGMENTS}
        write('anchor_admission.parquet', prior.old.parquet(candidate))
        write('conditional_reference_returns.parquet', prior.old.parquet(returns))
        write('coverage_increment.parquet', prior.old.parquet(side))
        for path, expected in refs.items():
            if prior.old.prep.sha(path) != expected:
                raise ValueError('INPUT_CHANGED_DURING_RUN:' + path)
        write('report.json', prior.old.prep.encoded(dict(status='COMPLETE_CONDITIONAL_REFERENCE_SIDECAR_NOT_FIT',
            inputs=refs, outputs=outputs.copy(), tests=tests.testsRun, test_output=log.getvalue(),
            candidate_code_days=len(candidate), evidence_intersection_code_days=int(candidate.code.isin(audit).sum()),
            evidence_intersection_codes=len(selected_codes), status_counts=candidate.status.value_counts().to_dict(),
            reason_counts=candidate.reason.value_counts().to_dict(), research_return_rows=len(returns),
            common_population_coverage=summary, formal_complete_certification_rows=0,
            coverage_verified=False, old_fields_changed=False, labels_changed=False,
            current_or_future_daily_price_used_for_admission=False,
            assumptions=['Same bound minute source is on archived HTTP raw-price basis',
                'Recorded ordinary actions are sufficient within declared restricted research scope',
                'Historical true arrival timestamps remain unknown'],
            next_step='Rebuild versioned board context only for admitted references; retain pending members and old labels',
            fit=False, http_requests=0)))
        write('manifest.json', prior.old.prep.encoded(dict(outputs=outputs.copy())))
        write('stop_record.json', prior.old.prep.encoded(dict(status='COMPLETE_NOT_FIT',
            manifest_sha256=prior.old.prep.sha(OUT/'manifest.json'))), terminal=True)
        print(json.dumps(dict(bytes=used, status_counts=candidate.status.value_counts().to_dict(),
            return_rows=len(returns), summary=summary), ensure_ascii=False))
    except Exception as exc:
        write('stop_record.json', prior.old.prep.encoded(dict(status='FAILED_NO_RESTART',
            error=f'{type(exc).__name__}:{exc}')), terminal=True)
        raise


if __name__ == '__main__':
    run()
