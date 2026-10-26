"""Rebuild only affected dates; sparse conditional board context, no fit."""
import csv
import hashlib
import io
import json
from pathlib import Path
import unittest

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

import bind_prior_anchor_evidence_v1 as binding
import repair_multiboard_observation_v1 as old
from build_multiboard_full_features_v1 import daily_references, market_arrays
from conditional_board_patch_v1 import patch_missing, with_new_high, differences
from intraday_relative_stall_v1 import slots
from intraday_relative_stall_v2 import minute_returns
from multiboard_risk_features_v1 import matrix_features, summarize
from read_multiboard_full_features_v1 import metadata, read_segment

HERE, TIME = old.HERE, old.TIME
OUT = HERE/'conditional_board_observations_20261018_v1'
CAP, RESERVE = 6*1024**2, 128*1024


def run():
    if OUT.exists():
        raise ValueError('OUTPUT_EXISTS_NO_RESTART')
    if old.bytes_used()+CAP > old.TOTAL_CAP:
        raise ValueError('REGISTERED_BUDGET_RESERVATION_FAILED')
    log = io.StringIO()
    tests = unittest.TextTestRunner(stream=log).run(unittest.defaultTestLoader.loadTestsFromNames([
        'test_conditional_board_patch_v1', 'test_multiboard_observation_repair_v1',
        'test_multiboard_risk_features_v1', 'test_prior_anchor_evidence_v1']))
    if not tests.wasSuccessful() or tests.skipped:
        raise ValueError('ENTRY_TESTS_FAILED:'+log.getvalue())
    refs, outputs, used = {}, {}, 0

    def bind(path, expected=None):
        path = Path(path)
        value = old.prep.sha(path)
        if expected is not None and value != expected:
            raise ValueError('INPUT_DRIFT:'+str(path))
        refs[str(path)] = value
        return path

    for name in ['build_conditional_board_observations_v1.py', 'conditional_board_patch_v1.py',
        'test_conditional_board_patch_v1.py', 'multiboard_risk_features_v1.py',
        'intraday_relative_stall_v2.py', 'newhigh_common_motion_v1.py', 'research_budget_v2.py']:
        bind(HERE/name)
    acceptance = json.loads(bind(HERE/'prior_anchor_evidence_acceptance_20261018_v1.json').read_bytes())
    accepted = json.loads(bind(binding.OUT/'report.json', acceptance['report_sha256']).read_bytes())
    for path, expected in accepted['inputs'].items():
        bind(path, expected)
    for name, item in accepted['outputs'].items():
        bind(binding.OUT/name, item['sha256'])
    # report.outputs precedes its own serialization; consume pinned manifest too.
    bound_manifest = json.loads(bind(binding.OUT/'manifest.json').read_bytes())
    for name, item in bound_manifest['outputs'].items():
        bind(binding.OUT/name, item['sha256'])
    full, _ = metadata()
    bind(HERE/'multiboard_full_features_20261018_v1/report.json')
    for name, item in full['outputs'].items():
        bind(HERE/'multiboard_full_features_20261018_v1'/name, item['sha256'])
    ordinary = pd.read_parquet(binding.OUT/'anchor_admission.parquet')
    ordinary = ordinary[ordinary.research_eligible].copy()
    if len(ordinary) != 400 or ordinary.formal_admitted.any() or ordinary.coverage_verified.any():
        raise ValueError('CONDITIONAL_SCOPE_CHANGED')
    # Non-unit action factors need a separately tested mixed-basis board policy.
    if not ordinary.ordinary_factor_ratio.eq(1.).all():
        raise ValueError('NONUNIT_FACTOR_NOT_IN_THIS_BOARD_PATCH_SCOPE')
    days = sorted(ordinary.current_session_day.unique().tolist())
    clock = pd.read_parquet(old.SOURCE/'shared_market_and_leaders.parquet', columns=['cutoff_index', TIME])
    selected = clock[clock[TIME].floordiv(1000000).isin(days)]
    if len(selected) != len(days)*16:
        raise ValueError('AFFECTED_CLOCK_INCOMPLETE')
    di = {day:i for i,day in enumerate(days)}
    clock_day = {r.cutoff_index: di[r.decision_ts_assumed_end//1000000] for r in selected.itertuples()}
    clock_pos = {k:i for i,k in enumerate(selected.cutoff_index)}
    spec = json.loads(bind(HERE/'ths_full_concept_scope_20261017_v1.json').read_bytes())
    boards = sorted([b for b in pq.read_table(bind(spec['source'],spec['source_sha256'])).to_pylist()
                     if b['source']=='ths' and b['type']=='concept'],key=lambda b:b['code'])
    codes = sorted({c for b in boards for c in b['symbols']})
    ci = {c:i for i,c in enumerate(codes)}
    members = [[ci[c] for c in b['symbols']] for b in boards]
    reference = np.full((len(codes),len(days)),np.nan)
    before15 = np.full((len(codes),len(days)*16),np.nan)
    manifest = json.loads((old.SOURCE/'manifest.json').read_bytes())
    for part in manifest['joint_parts']:
        path = bind(old.SOURCE/part['path'],manifest['output_hashes'][part['path']]['sha256'])
        frame = pd.read_parquet(path,columns=['code','cutoff_index','stock_reference_close_raw','stock_day_return_raw'])
        daily_references(frame,ci,clock_day,reference,dict(values=before15,_map=clock_pos))
    repaired_reference = reference.copy()
    allowed = np.zeros_like(reference,dtype=bool)
    for row in ordinary.itertuples():
        i,j = ci[row.code],di[row.current_session_day]
        if np.isfinite(reference[i,j]):
            raise ValueError('ADMISSION_WOULD_REPLACE_OLD_REFERENCE')
        repaired_reference[i,j] = row.reference_close_potential
        allowed[i,j] = True
    quarantine = bind(binding.prior.QUARANTINE)
    excluded = {}
    with quarantine.open(encoding='utf-8-sig',newline='') as stream:
        for row in csv.DictReader(stream):
            excluded.setdefault(row['code'],set()).add(int(row['day']))
    ledger = json.loads(bind(HERE/'relative_stall_5m_20261018_v1/raw_input_ledger.json').read_bytes())
    if len(ledger)!=4196 or len(boards)!=978 or len(codes)!=5540 or len(days)!=84:
        raise ValueError('FROZEN_SCOPE_CHANGED')

    def write(name,body,terminal=False):
        nonlocal used
        reserve=0 if terminal else RESERVE
        if used+len(body)+reserve>CAP or old.bytes_used()+len(body)+reserve>old.TOTAL_CAP:
            raise ValueError('SERIALIZED_BUDGET_STOP')
        if not OUT.exists():
            OUT.mkdir(exist_ok=False)
        with (OUT/name).open('xb') as stream:
            stream.write(body)
        used+=len(body)
        outputs[name]=dict(bytes=len(body),sha256=old.prep.sha(OUT/name))

    write('plan.json',old.prep.encoded(dict(inputs=refs.copy(),tests=tests.testsRun,
        test_output=log.getvalue(),affected_days=days,conditional_code_days=len(ordinary),
        cap=CAP,fit=False,http_requests=0,raw_files=4196,complete_history_rebuild=False)))
    try:
        before5=np.full((len(codes),len(days)*48),np.nan)
        after5=before5.copy();after15=before15.copy()
        counts={5:0,15:0};compared=0
        for number,entry in enumerate(ledger):
            code=entry['code'];i=ci[code];path=Path(entry['path']);body=path.read_bytes()
            if hashlib.sha256(body).hexdigest()!=entry['sha256']:
                raise ValueError('RAW_LEDGER_DRIFT:'+code)
            refs[str(path)]=entry['sha256']
            frame=pq.read_table(pa.BufferReader(body),filters=[('date','>=',days[0]*1000000),
                ('date','<=',days[-1]*1000000+150000)],columns=[
                'code','date','open','high','low','close','volume','amount']).to_pandas()
            frame=frame[frame.date.floordiv(1000000).isin(days)]
            baseline,_=minute_returns(frame,code,days,dict(zip(days,reference[i])),excluded.get(code,set()))
            valid=np.isfinite(before15[i])
            np.testing.assert_allclose(baseline[15][valid],before15[i,valid],atol=1e-12,rtol=0)
            compared+=int(valid.sum());before5[i]=baseline[5];after5[i]=baseline[5]
            if allowed[i].any():
                fresh,_=minute_returns(frame,code,days,dict(zip(days,repaired_reference[i])),excluded.get(code,set()))
                for frequency,target in [(5,after5),(15,after15)]:
                    permitted=np.repeat(allowed[i],240//frequency)
                    target[i],count=patch_missing(target[i],fresh[frequency],permitted)
                    counts[frequency]+=count
            if (number+1)%400==0:
                print(json.dumps(dict(stage='affected_dates_native_paths',files=number+1,total=4196)),flush=True)
        if counts[15]!=accepted['research_return_rows']:
            raise ValueError('ADMITTED_15M_COUNT_NOT_REPRODUCED')
        parent_returns=pd.read_parquet(binding.OUT/'conditional_reference_returns.parquet')
        at=parent_returns.cutoff_index.map(clock_pos).to_numpy(dtype=int)
        ix=parent_returns.code.map(ci).to_numpy(dtype=int)
        np.testing.assert_allclose(after15[ix,at],parent_returns.conditional_day_return,atol=1e-12,rtol=0)
        cohort=[]
        for segment in old.prep.SEGMENTS:
            f=read_segment(segment).assign(segment=segment)
            side=old.read_context(segment).drop(columns='segment')
            f=f.merge(side,on=['code',TIME],validate='one_to_one',sort=False)
            cohort.append(f)
        cohort=pd.concat(cohort,ignore_index=True)
        if len(cohort)!=100704 or cohort.duplicated(['code',TIME]).any():
            raise ValueError('ORIGINAL_POPULATION_CHANGED')
        relevant=cohort[cohort[TIME].floordiv(1000000).isin(days)].copy()
        market=pd.read_parquet(bind(HERE/'local_market_context_20261017_v1/market_price_features.parquet'))
        pieces=[];baseline_cells=0;real_prefix=0
        for month in sorted({d//100 for d in days}):
            subset=[d for d in days if d//100==month]
            start,stop=di[subset[0]],di[subset[-1]]+1
            keys=relevant[relevant[TIME].floordiv(100000000).eq(month)].copy()
            shared_keys=pd.DataFrame({TIME:[t for d in subset for t in slots(d,15)]})
            shared_keys['code']='__SHARED__'
            for frequency,original,repaired in [(5,before5,after5),(15,before15,after15)]:
                n=240//frequency
                raw=original[:,start*n:stop*n];new=repaired[:,start*n:stop*n]
                grid,markets=market_arrays(market,subset,frequency)
                positions={t:i for i,t in enumerate(grid)}
                take=[positions[t] for t in shared_keys[TIME]]
                for candidate,indexes in [('__SHARED__',None),*keys.groupby('code',sort=False).groups.items()]:
                    own=candidate!='__SHARED__';prefix=f'own{frequency}_' if own else f'mb{frequency}_'
                    index=ci[candidate] if own else None
                    related=[m for m in members if index in m] if own else members
                    if not related:
                        continue
                    identity=keys.loc[indexes,['code',TIME]] if own else shared_keys[['code',TIME]]
                    at=[positions[t] for t in identity[TIME]] if own else take
                    results=[]
                    for array in (raw,new):
                        b=matrix_features(array,related,frequency,exclude_index=index)
                        s=summarize(b,markets,frequency,prefix,own=own)
                        s=with_new_high(s,markets,frequency,prefix)
                        results.append({k:v[at] for k,v in s.items()})
                    expected=(keys.loc[indexes] if own else keys.drop_duplicates(TIME).set_index(TIME).loc[identity[TIME]])
                    for field,value in results[0].items():
                        np.testing.assert_allclose(value,expected[field].to_numpy(dtype=float),atol=1e-12,rtol=0)
                        baseline_cells+=len(value)
                    pieces.append(differences(identity.reset_index(drop=True),*results))
                # Real-data prefix replay on one affected day, not an empty synthetic proof.
                if real_prefix<2:
                    cut=n//2; sample=new[:,:n]; mk={k:v[:n] for k,v in markets.items()}
                    observed=summarize(matrix_features(sample,members,frequency),mk,frequency,'p_')
                    if not np.isfinite(sample).any():
                        raise ValueError('EMPTY_REAL_PREFIX')
                    for fill in (np.nan,.75):
                        changed=sample.copy();changed[:,cut+1:]=fill
                        replay=summarize(matrix_features(changed,members,frequency),mk,frequency,'p_')
                        for field,value in observed.items():
                            np.testing.assert_allclose(value[:cut+1],replay[field][:cut+1],atol=1e-12,rtol=0,equal_nan=True)
                    real_prefix+=1
            print(json.dumps(dict(stage='monthly_sparse_observations',month=month)),flush=True)
        diff=pd.concat(pieces,ignore_index=True)
        if diff.duplicated(['code',TIME,'feature']).any():
            raise ValueError('DUPLICATE_SPARSE_PATCH')
        write('observation_differences.parquet',old.parquet(diff))
        stats=[]
        population=pd.read_parquet(old.prep.OUT/'population_audit.parquet')
        for segment in old.prep.SEGMENTS:
            keys=population[population.common_main_eligible & population.proposed_segment_name.eq(segment)][['code',TIME]]
            shared=diff[diff.code.eq('__SHARED__')].drop(columns='code')
            own=diff[~diff.code.eq('__SHARED__')]
            changed=pd.concat([keys.merge(shared,on=TIME),keys.merge(own,on=['code',TIME])],ignore_index=True)
            for field,g in changed.groupby('feature'):
                stats.append(dict(segment=segment,feature=field,changed_rows=len(g),
                    gained_known=int((g.old_value.isna() & g.new_value.notna()).sum()),
                    lost_known=int((g.old_value.notna() & g.new_value.isna()).sum()),
                    changed_finite=int((g.old_value.notna() & g.new_value.notna()).sum()),
                    from_zero_to_positive=int((g.old_value.eq(0) & g.new_value.gt(0)).sum()),
                    from_positive_to_zero=int((g.old_value.gt(0) & g.new_value.eq(0)).sum())))
        write('common_population_changes.parquet',old.parquet(pd.DataFrame(stats)))
        # Verify all inputs again; includes every consumed frozen minute source.
        for path,expected in refs.items():
            if old.prep.sha(path)!=expected:
                raise ValueError('INPUT_CHANGED_DURING_BUILD:'+path)
        write('report.json',old.prep.encoded(dict(status='COMPLETE_CONDITIONAL_BOARD_PATCH_NOT_FIT',
            inputs=refs,outputs=outputs.copy(),tests=tests.testsRun,test_output=log.getvalue(),
            conditional_code_days=len(ordinary),affected_days=len(days),raw_files=len(ledger),
            old_valid_15m_reproduced=compared,new_5m_returns=counts[5],new_15m_returns=counts[15],
            original_cohort_rows=len(cohort),affected_cohort_rows=len(relevant),
            baseline_observation_cells_reproduced=baseline_cells,real_prefix_frequencies=real_prefix,
            changed_cells=len(diff),changed_shared_cells=int(diff.code.eq('__SHARED__').sum()),
            changed_own_cells=int((~diff.code.eq('__SHARED__')).sum()),
            scope='CONDITIONAL_RESEARCH_ONLY_NOT_FULL_SOURCE_CERTIFICATION',
            coverage_verified=False,old_fields_changed=False,labels_changed=False,
            fit=False,http_requests=0)))
        write('manifest.json',old.prep.encoded(dict(outputs=outputs.copy())))
        write('stop_record.json',old.prep.encoded(dict(status='COMPLETE_NOT_FIT',
            manifest_sha256=old.prep.sha(OUT/'manifest.json'))),terminal=True)
        print(json.dumps(dict(bytes=used,changed_cells=len(diff),new_returns=counts,fit=False)))
    except Exception as exc:
        write('stop_record.json',old.prep.encoded(dict(status='FAILED_NO_RESTART',error=f'{type(exc).__name__}:{exc}')),terminal=True)
        raise


if __name__=='__main__':
    run()
