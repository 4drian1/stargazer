"""Actual-data prefix/perturbation and overlay readback acceptance; no fitting."""
import io
import json
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from build_joint_risk_features_v1 import wilder_rsi
from finalize_postclose_cohort_v3 import read_population
from inventory_ths_stockdb_cache_v1 import HERE, digest
from repair_postclose_daily_context_v1 import OUT, SOURCE, CAP, closing_panel, daily_returns, board_panel, unique_leaders
from read_postclose_daily_context_repair_v1 import read_repaired_population
from research_budget_v2 import TOTAL_CAP, bytes_used


def run():
    manifest = json.loads((OUT / 'manifest.json').read_bytes())
    for path, sha in manifest['inputs'].items():
        if digest(Path(path).read_bytes()) != sha:
            raise ValueError('FROZEN_INPUT_DRIFT:' + path)
    original, frozen = read_population()
    repaired, _ = read_repaired_population()
    overlay = pd.read_parquet(OUT / 'daily_context_overlay.parquet')
    changed = [c for c in overlay if c.startswith('context_')] + [
        'context_stock_minus_sector_daily_return', 'context_stock_minus_leader_daily_return', 'research_cohort_eligible']
    pd.testing.assert_frame_equal(repaired.drop(columns=changed), original.drop(columns=changed))
    oldfit = HERE / 'postclose_first_research_fit_20261017_v1'
    audit = json.loads((oldfit / 'independent_readback_audit.json').read_bytes())
    for name, sha in audit['audited_original_sha256'].items():
        if digest((oldfit / name).read_bytes()) != sha:
            raise ValueError('OLD_FIT_CHANGED')
    source = json.loads((SOURCE / 'manifest.json').read_bytes())
    catalog = json.loads((HERE / 'ths_full_concept_scope_20261017_v1.json').read_bytes())
    membership = pq.read_table(catalog['source']).to_pylist()
    membership = sorted([b for b in membership if b['source']=='ths' and b['type']=='concept'], key=lambda b:b['code'])
    codes = sorted({c for b in membership for c in b['symbols']}); code_index={c:i for i,c in enumerate(codes)}
    days = np.sort(original.day.unique())
    closes = np.full((len(codes),len(days)), np.nan)
    for index, part in enumerate(source['joint_parts']):
        # These exact physical shards were hash checked above, not regenerated from today's sources.
        body = (SOURCE / part['path']).read_bytes()
        frame = pq.read_table(io.BytesIO(body),columns=['code','cutoff_index','stock_close']).to_pandas()
        for code, values in closing_panel(frame,len(days)).items():
            closes[code_index[code]] = values
        if index % 20 == 0:
            print(json.dumps(dict(stage='acceptance_daily_close',parts=index+1)),flush=True)
    members = [np.asarray([code_index[c] for c in b['symbols']],dtype=int) for b in membership]
    returns = daily_returns(closes)
    means, counts, _ = board_panel(returns, members)
    boards, leaders = unique_leaders(returns, means, members, codes)
    checks = []
    for end in [100,300,400]:
        prefix = daily_returns(closes[:,:end+1])
        changed_closes = closes.copy(); changed_closes[:,end+1:] *= 1.5
        mutated = daily_returns(changed_closes)
        np.testing.assert_equal(prefix, returns[:,:end+1])
        np.testing.assert_equal(mutated[:,:end+1], prefix)
        prefix_mean, prefix_count, _ = board_panel(prefix,members)
        future_mean, _, _ = board_panel(mutated,members)
        np.testing.assert_allclose(prefix_mean,means[:,:end+1],rtol=0,atol=1e-14,equal_nan=True)
        np.testing.assert_allclose(future_mean[:,:end+1],prefix_mean,rtol=0,atol=1e-14,equal_nan=True)
        pb, ps = unique_leaders(prefix,prefix_mean,members,codes)
        np.testing.assert_equal(pb,boards[:end+1]); np.testing.assert_equal(ps,leaders[:end+1])
        actual_leaders = np.unique(leaders[:end+1][leaders[:end+1] >= 0])
        for s in actual_leaders:
            np.testing.assert_allclose(wilder_rsi(closes[s])[:end+1],wilder_rsi(closes[s,:end+1]),rtol=0,atol=1e-12,equal_nan=True)
        if not np.isfinite(prefix).any():
            raise ValueError('EMPTY_PREFIX_VACUOUS_PASS')
        checks.append(dict(end_day=int(days[end]),stocks=len(codes),boards=len(members),observed_leader_rsi_series=len(actual_leaders)))
    for row in overlay.itertuples(index=False):
        d = int(np.searchsorted(days,row.day))
        expected = codes[leaders[d]] if leaders[d] >= 0 else None
        observed = None if pd.isna(row.observed_leader_stock) else row.observed_leader_stock
        if observed != expected:
            raise ValueError('OBSERVED_LEADER_READBACK_DRIFT')
        if (not row.daily_reference_pair_aligned or row.code == expected) and pd.notna(row.context_current_leader_return_loo_raw):
            raise ValueError('OWN_OR_UNALIGNED_LEADER_ADMITTED')
    segments = {}
    for segment, group in repaired[repaired.research_cohort_eligible].groupby('proposed_segment_name'):
        segments[str(segment)] = dict(rows=len(group),dates=int(group.day.nunique()),stocks=int(group.code.nunique()))
    result = dict(status='PASS_DATA_REPLAY_NOT_FIT_AUTHORIZATION',actual_prefix_and_future_checks=checks,
        exact_population_keys=True,old_target_and_unrelated_fields_unchanged=True,
        original_fit_hashes_unchanged=True,field_count=len(frozen['fields']),proposed_segments=segments,
        artifact_sha256={name:digest((OUT/name).read_bytes()) for name in ['manifest.json','daily_context_overlay.parquet']},
        acceptance_code_sha256=digest(Path(__file__).read_bytes()),
        reader_sha256=digest((HERE/'read_postclose_daily_context_repair_v1.py').read_bytes()),fit_run=False)
    body=json.dumps(result,ensure_ascii=False,allow_nan=False,indent=2).encode('utf-8')
    if sum(p.stat().st_size for p in OUT.rglob('*') if p.is_file())+len(body)>CAP or bytes_used()+len(body)>TOTAL_CAP:
        raise ValueError('ACCEPTANCE_BUDGET_STOP')
    with (OUT/'readback_acceptance.json').open('xb') as stream:stream.write(body)
    print(json.dumps(result,ensure_ascii=False))
    return result


if __name__=='__main__':run()
