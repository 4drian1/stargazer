"""Append an independent 120-slot price target on the unchanged common cohort."""
import argparse
from collections import defaultdict
import csv
import gzip
import io
import json
from pathlib import Path
import unittest

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

import build_decision_after_labels_v1 as parent
from decision_after_labels_v1 import minute_clock,price_grid,future_minimum,label_rows
from decision_after_basis_v2 import native_aligned_days
from read_decision_after_labels_v2 import OUT as PARENT_LABELS
import direct_warning_fixed_fit_v1 as fit

HERE,TIME=fit.HERE,fit.TIME
CONTRACT=HERE/'contracts/INTRADAY_AUXILIARY_120M_OBSERVATION_V1.json'
OUT=HERE/'auxiliary_120m_observations_20261018_v1'
CAP,RESERVE,HORIZON=2*1024**2,64*1024,120


def auxiliary_values(grid,positions,factors,horizon=HORIZON):
    positions=np.asarray(positions,dtype=int)
    factors=np.asarray(factors,dtype=float)
    if type(horizon) is not int or horizon<1 or len(grid)!=len(factors)*240:
        raise ValueError('INVALID_AUXILIARY_GRID')
    if (positions<0).any() or (positions>=len(grid)).any():
        raise ValueError('REFERENCE_OUTSIDE_CLOCK')
    if not (np.isfinite(factors)&(factors>0)).all():
        raise ValueError('INVALID_FACTOR')
    norm=np.repeat(factors,240)
    normalized=grid[:,2]*norm
    if not np.isfinite(normalized[np.isfinite(grid[:,2])]).all():
        raise ValueError('NORMALIZATION_OVERFLOW')
    reference=grid[positions,3]*norm[positions]
    minimum=future_minimum(normalized,horizon)[positions]
    ends=positions+horizon
    valid=(ends<len(grid))&np.isfinite(reference)&(reference>0)&np.isfinite(minimum)&(minimum>0)
    signed=np.full(len(positions),np.nan)
    signed[valid]=minimum[valid]/reference[valid]-1
    drop=np.maximum(0.,-signed)
    if np.isinf(signed).any() or np.isinf(drop).any():
        raise ValueError('NONFINITE_AUXILIARY_LABEL')
    return signed,drop,ends,valid


def metadata():
    spec=json.loads(CONTRACT.read_bytes())
    if (spec['status']!='CHAT_APPROVED_AUXILIARY_LABEL_PREPARATION_ONLY'
        or spec['horizon_trading_minutes']!=120 or spec['threshold'] is not None
        or any(spec[k] is not False for k in ['fit_authorized','new_http_authorized','deployment_authorized','training_eligible'])):
        raise ValueError('AUXILIARY_SCOPE_NOT_APPROVED')
    fit.prep.check()
    accepted=HERE/'direct_warning_fixed_fit_acceptance_20261018_v1.json'
    if json.loads(accepted.read_bytes())['status']!='PASS_FIXED_DIRECT_WARNING_RESEARCH_REPLAY_NOT_PROMOTION':
        raise ValueError('PARENT_COHORT_ACCEPTANCE_REQUIRED')
    if json.loads((PARENT_LABELS/'stop_record.json').read_bytes())['status']!='COMPLETE':
        raise ValueError('PARENT_ADMISSION_NOT_COMPLETE')
    refs=dict(json.loads((fit.OUT/'plan.json').read_bytes())['inputs'])
    source_manifest=json.loads((parent.SOURCE/'manifest.json').read_bytes())
    labels=json.loads((PARENT_LABELS/'manifest.json').read_bytes())
    path=parent.SOURCE/'shared_market_and_leaders.parquet'
    expected=source_manifest['output_hashes']['shared_market_and_leaders.parquet']['sha256']
    if fit.previous.sha(path)!=expected:
        raise ValueError('FROZEN_CALENDAR_DRIFT')
    times=pq.read_table(path,columns=[TIME]).column(0).to_numpy()
    days=np.unique(times//1000000);clock=minute_clock(days)
    if not np.array_equal(clock[14::15],times):
        raise ValueError('PARENT_CLOCK_MISMATCH')
    cohort=pd.read_parquet(fit.OUT/'predictions.parquet',columns=[
        'code',TIME,'label_end_time','y_tail_1200m','mae_1200m','segment'])
    if len(cohort)!=86319 or cohort.duplicated(['code',TIME]).any():
        raise ValueError('PARENT_POPULATION_DRIFT')
    sources={s['code']:s for s in labels['source_files']}
    selected={code:sources[code] for code in sorted(cohort.code.unique())}
    daily={}
    for code in selected:
        matches=[p for p in labels['inputs'] if p.endswith(f'{code}_daily.json.gz')]
        if len(matches)!=1:
            raise ValueError('DAILY_EVIDENCE_REFERENCE_NOT_UNIQUE:'+code)
        daily[code]=matches[0]
        refs[matches[0]]=labels['inputs'][matches[0]]
        refs[selected[code]['path']]=selected[code]['sha256']
    audit,ordinary_acceptance=parent.admitted_evidence()
    if not set(selected)<=set(audit):
        raise ValueError('PARENT_CODE_NOT_ORDINARY_ADMITTED')
    for path in [CONTRACT,accepted,fit.OUT/'predictions.parquet',PARENT_LABELS/'manifest.json',
        PARENT_LABELS/'stop_record.json',parent.SOURCE/'manifest.json',parent.SOURCE/'shared_market_and_leaders.parquet',
        parent.QUARANTINE,ordinary_acceptance,parent.LEGACY/'evidence_and_gate.json',Path(__file__),
        HERE/'decision_after_labels_v1.py',HERE/'decision_after_basis_v2.py',
        HERE/'build_decision_after_labels_v1.py',HERE/'test_auxiliary_120m_v1.py',HERE/'audit_auxiliary_120m_v1.py']:
        refs[str(path)]=fit.previous.sha(path)
    for path,expected in refs.items():
        if fit.previous.sha(path)!=expected:
            raise ValueError('FROZEN_INPUT_DRIFT:'+path)
    quarantine=defaultdict(set)
    with parent.QUARANTINE.open(encoding='utf-8-sig',newline='') as stream:
        for row in csv.DictReader(stream):quarantine[row['code']].add(int(row['day']))
    return dict(spec=spec,refs=refs,clock=clock,days=days,cohort=cohort,sources=selected,
                daily=daily,audit=audit,quarantine=quarantine)


def load_code(code,meta):
    item=meta['sources'][code]
    raw=Path(item['path']).read_bytes()
    if parent.digest(raw)!=item['sha256']:
        raise ValueError('MINUTE_INPUT_DRIFT:'+code)
    native=pq.read_table(io.BytesIO(raw)).to_pandas()
    grid=price_grid(native,code,meta['clock'],meta['quarantine'].get(code,set()))
    # Reuse the precise native-boundary correction that admitted the original labels.
    used={}
    evidence,_=parent.evidence_for(code,meta['audit'],grid,meta['days'],used)
    if used!={meta['daily'][code]:meta['refs'][meta['daily'][code]]}:
        raise ValueError('DAILY_SOURCE_SELECTION_CHANGED:'+code)
    path=Path(meta['daily'][code]);body=path.read_bytes()
    if parent.digest(body)!=meta['refs'][str(path)]:
        raise ValueError('DAILY_EVIDENCE_DRIFT:'+code)
    bars={int(k.rsplit(':',1)[1]):v for k,v in json.loads(gzip.decompress(body)) if v is not None}
    evidence['aligned']=native_aligned_days(grid,meta['days'],bars,native)
    return grid,evidence


def run():
    log=io.StringIO()
    tests=unittest.TextTestRunner(stream=log,verbosity=2).run(
        unittest.defaultTestLoader.loadTestsFromName('test_auxiliary_120m_v1'))
    if not tests.wasSuccessful():raise ValueError(log.getvalue())
    meta=metadata()
    writer=fit.base.EvidenceWriter(OUT,CAP,RESERVE)
    writer.json('plan.json',dict(inputs=meta['refs'],contract=meta['spec'],output_cap=CAP,
        source_files=meta['sources'],source_bytes=sum(s['bytes'] for s in meta['sources'].values()),
        parent_population_rows=86319,population_expanded=False,fit=False,http_requests=0))
    try:
        cohort=meta['cohort'].copy();cohort['parent_position']=np.arange(len(cohort))
        tables=[];parent_error=0.
        for code,frame in cohort.groupby('code',sort=True):
            grid,evidence=load_code(code,meta)
            positions=np.searchsorted(meta['clock'],frame[TIME].to_numpy())
            if (positions>=len(grid)).any() or not np.array_equal(meta['clock'][positions],frame[TIME]):
                raise ValueError('DECISION_NOT_IN_FROZEN_CLOCK')
            if (positions%15!=14).any():raise ValueError('REFERENCE_NOT_COMPLETED_15M')
            parent_keys=pd.DataFrame(dict(code=code,cutoff_index=positions//15,stock_close=grid[positions,3]))
            replay,ends=label_rows(parent_keys,grid,meta['days'],evidence,horizon=1200)
            if not replay.label_state.eq(0).all():raise ValueError('PARENT_ADMISSION_REPLAY_FAILED:'+code)
            np.testing.assert_array_equal(meta['clock'][ends],frame.label_end_time.to_numpy(dtype=np.int64))
            np.testing.assert_array_equal(replay.y_tail_1200m.to_numpy(),frame.y_tail_1200m.to_numpy())
            np.testing.assert_allclose(replay.mae_1200m,frame.mae_1200m,atol=1e-12,rtol=0)
            parent_error=max(parent_error,float(np.max(np.abs(replay.mae_1200m.to_numpy()-frame.mae_1200m.to_numpy()))))
            signed,drop,ends,valid=auxiliary_values(grid,positions,evidence['factors'])
            if not valid.all():raise ValueError('AUXILIARY_INCOMPLETE_ON_ADMITTED_PARENT:'+code)
            if (signed<frame.mae_1200m.to_numpy()-1e-12).any():raise ValueError('SUBWINDOW_MINIMUM_INCONSISTENT')
            table=frame[['parent_position','code',TIME,'segment']].copy()
            table['auxiliary_end_time']=meta['clock'][ends]
            table['future_min_return_120m']=signed
            table['max_adverse_drop_120m']=drop
            table['auxiliary_state']='CONDITIONAL_AUXILIARY_COMPLETE'
            table['data_complete_time']=None
            table['training_eligible']=False
            tables.append(table)
            print(json.dumps(dict(stage='parent_replay_and_auxiliary',code=code,rows=len(frame))),flush=True)
        labels=pd.concat(tables,ignore_index=True).sort_values('parent_position').reset_index(drop=True)
        pd.testing.assert_frame_equal(labels[['code',TIME,'segment']],cohort[['code',TIME,'segment']])
        summaries={}
        for segment,frame in labels.groupby('segment',sort=False):
            x=frame.max_adverse_drop_120m.to_numpy()
            summaries[segment]=dict(rows=len(frame),dates=int((frame[TIME]//1000000).nunique()),stocks=int(frame.code.nunique()),
                zero_downside_rows=int((x==0).sum()),mean_max_adverse_drop=float(x.mean()),
                downside_quantiles={str(q):float(np.quantile(x,q)) for q in [0,.25,.5,.75,.9,.95,1]})
        sink=io.BytesIO();labels.to_parquet(sink,index=False,compression='zstd')
        writer.write('labels.parquet',sink.getvalue())
        writer.json('report.json',dict(status='COMPLETE_AUXILIARY_LABELS_NOT_FIT',tests=tests.testsRun,test_output=log.getvalue(),
            rows=len(labels),codes=len(meta['sources']),segments=summaries,parent_mae_replay_max_error=parent_error,
            parent_1200m_admission_and_binary_replay='PASS',horizon_trading_minutes=120,
            threshold=None,binary_label=None,training_eligible=False,source_scope_expanded=False,
            data_complete_time=None,clock='INHERITED_ASSUMED_END',coverage_verified=False,
            normalization='INHERITED_RESTRICTED_ORDINARY_EVENT_EVIDENCE',
            warning_efficacy_proven=False,fit=False,http_requests=0,deployment=False))
        for path,expected in meta['refs'].items():
            if fit.previous.sha(path)!=expected:raise ValueError('INPUT_CHANGED:'+path)
        writer.json('manifest.json',dict(outputs=writer.outputs.copy()))
        writer.json('stop_record.json',dict(status='COMPLETE_NOT_FIT',manifest_sha256=fit.previous.sha(OUT/'manifest.json')),terminal=True)
        return dict(status='COMPLETE_NOT_FIT',rows=len(labels),bytes=writer.used,parent_replay_max_error=parent_error)
    except Exception as exc:
        writer.json('stop_record.json',dict(status='FAILED_NO_RESTART',error=f'{type(exc).__name__}:{exc}'),terminal=True)
        raise


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--build',action='store_true',required=True)
    parser.parse_args();print(json.dumps(run()))
