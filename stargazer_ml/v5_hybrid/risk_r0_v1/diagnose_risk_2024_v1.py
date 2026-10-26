"""Read frozen F2 models and data; diagnose calibration and selection, never refit."""
from collections import defaultdict
import json
from pathlib import Path

import joblib
import numpy as np
import pyarrow.parquet as pq

from price_verification_guarded_run import encoded, sha
from research_budget_v2 import bytes_used, TOTAL_CAP
from risk_crossyear_v1 import HERE, MANIFEST, OUTPUT, load_data, population_fingerprint, require, verify_refs
from risk_fit_v1 import arrays, logits, probabilities
from risk_fit_eval_v1 import block_indices, board, interval, keep_mask, probability_metrics
from risk_failure_diagnostics_v1 import feature_drift, market_thresholds, regime, tail_stats

DEST = HERE/'risk_2024_diagnosis_v1_20261015'
CAP = 2*1024**2


def calibration_delta(rows, raw, calibrated, evaluation):
    y=np.array([r['y_tail_5d'] for r in rows])
    raw,calibrated=np.asarray(raw),np.asarray(calibrated)
    require(len(rows)==len(raw)==len(calibrated) and np.isfinite(raw).all()
        and np.isfinite(calibrated).all(),'INVALID_PROBABILITY_INPUT')
    dates=sorted({r['P'] for r in rows})
    lookup={d:i for i,d in enumerate(dates)}
    ids=np.array([lookup[r['P']] for r in rows])
    counts=np.bincount(ids)
    delta=(calibrated-y)**2-(raw-y)**2
    sums=np.bincount(ids,weights=delta)
    draws=block_indices(len(dates),evaluation['bootstrap_reps'],evaluation['block_dates'],evaluation['seed'])
    return interval(delta.mean(),sums[draws].sum(1)/counts[draws].sum(1))


def mapping_check(rows, raw, calibrated, fraction):
    raw,calibrated=np.asarray(raw),np.asarray(calibrated)
    require(np.isfinite(raw).all() and np.isfinite(calibrated).all()
        and ((raw>=0)&(raw<=1)).all() and ((calibrated>=0)&(calibrated<=1)).all(),'PROBABILITY_DOMAIN')
    order=np.argsort(raw,kind='stable')
    unchanged=np.array_equal(keep_mask(rows,raw,fraction),keep_mask(rows,calibrated,fraction))
    return dict(monotone_nondecreasing=bool(np.all(np.diff(calibrated[order])>=0)),
        primary_daily_masks_identical=unchanged, distinct_raw=len(np.unique(raw)),
        distinct_calibrated=len(np.unique(calibrated)),
        note='Numerical ties may affect masks even for a mathematically increasing transform; explicitly checked.')


def swap_groups(rows):
    return {'both':[r for r in rows if r['lightgbm_retained_primary'] and r['lr_retained_primary']],
        'gb_only':[r for r in rows if r['lightgbm_retained_primary'] and not r['lr_retained_primary']],
        'lr_only':[r for r in rows if r['lr_retained_primary'] and not r['lightgbm_retained_primary']],
        'neither':[r for r in rows if not r['lr_retained_primary'] and not r['lightgbm_retained_primary']]}


def overlap_clusters(rows):
    """Mechanical union of same-code overlapping P..label_end intervals; not independent incidents."""
    by_code=defaultdict(list)
    for row in rows:
        by_code[row['code']].append(row)
    clusters=[]
    for code,values in sorted(by_code.items()):
        current=None
        for r in sorted(values,key=lambda r:(r['P'],r['label_end_date'])):
            if current is None or r['P']>current['end']:
                current=dict(code=code,start=r['P'],end=r['label_end_date'],rows=0,worst_mae=r['mae_5d'])
                clusters.append(current)
            current['end']=max(current['end'],r['label_end_date'])
            current['rows']+=1
            current['worst_mae']=min(current['worst_mae'],r['mae_5d'])
    return clusters


def swap_summary(rows):
    subsets=swap_groups(rows)
    info={name:dict(tail_stats(values),dates=len({r['P'] for r in values}),
        codes=len({r['code'] for r in values})) for name,values in subsets.items()}
    gb=sum(r['lightgbm_retained_primary'] for r in rows)
    lr=sum(r['lr_retained_primary'] for r in rows)
    counts={name:sum(r['y_tail_5d'] for r in subsets[name]) for name in ('gb_only','lr_only')}
    days=defaultdict(list)
    for r in rows:
        days[r['P']].append(r)
    paired=all(sum(r['lightgbm_retained_primary'] for r in day)==sum(r['lr_retained_primary'] for r in day)
               for day in days.values())
    return dict(population=len(rows),subsets=info,retained_gb=gb,retained_lr=lr,
        excess_tail_count_gb_minus_lr=counts['gb_only']-counts['lr_only'],
        equal_total_retained=gb==lr,daily_retained_counts_paired=paired,
        paired_rate_delta=(counts['gb_only']-counts['lr_only'])/gb if paired and gb else None,
        note='Original full-day masks inherited; subgroup counts may differ; NULL paired rate is not zero.')


def run():
    require(not DEST.exists(),'DIAGNOSIS_EXISTS_NO_OVERWRITE')
    manifest=json.loads(MANIFEST.read_bytes())
    verify_refs(manifest['inputs'])
    stop=json.loads((OUTPUT/'stop_record.json').read_bytes())
    require(stop['status']=='completed','CROSSYEAR_FIT_NOT_COMPLETE')
    run_refs=[dict(path=str(OUTPUT/r['name']),sha256=r['sha256']) for r in stop['files']]
    verify_refs(run_refs)
    _,base,folds,_,_,fingerprints=load_data()
    require(fingerprints==manifest['fold_populations'],'POPULATION_DRIFT')
    groups=folds['F2']
    require(population_fingerprint(groups)==manifest['fold_populations']['F2'],'F2_DRIFT')
    model=joblib.load(OUTPUT/'F2_models.joblib')
    test=pq.read_table(OUTPUT/'F2_test_predictions.parquet').to_pylist()
    require(len(test)==len(groups['test']),'TEST_COUNT_DRIFT')
    for r,p in zip(groups['test'],test):
        require(all(r[k]==p[k] for k in ('code','P','label_end_date','mae_5d','y_tail_5d')),'TEST_IDENTITY_DRIFT')
        p['features']=r['features']
    roles={}
    prediction_arrays={}
    fraction=base['evaluation']['primary_rejection_fraction']
    for role,rows in groups.items():
        x,y=arrays(rows,base['fields'])
        roles[role]=dict(n=len(rows),dates=len({r['P'] for r in rows}),observed_tail_rate=float(y.mean()),models={})
        prediction_arrays[role]={}
        for name in ('lr','lightgbm'):
            raw=probabilities(model[name],x)
            calibrated=probabilities(model['calibrators'][name],logits(raw))
            prediction_arrays[role][name]=dict(raw=raw,calibrated=calibrated)
            if role=='test':
                np.testing.assert_allclose(raw,[r[name+'_raw'] for r in test],rtol=0,atol=0)
                np.testing.assert_allclose(calibrated,[r[name] for r in test],rtol=0,atol=0)
            check=mapping_check(rows,raw,calibrated,fraction)
            require(check['monotone_nondecreasing'] and check['primary_daily_masks_identical'],
                    'CALIBRATION_CHANGED_RANKING_OR_PRIMARY_MASK')
            roles[role]['models'][name]=dict(raw=probability_metrics(y,raw),
                calibrated=probability_metrics(y,calibrated),raw_mean=float(raw.mean()),
                calibrated_mean=float(calibrated.mean()),mapping=check,
                calibration_brier_minus_raw=calibration_delta(rows,raw,calibrated,base['evaluation']))
        roles[role]['interpretation']={
            'train':'RETROSPECTIVE_IN_SAMPLE_MODEL_AND_FUTURE_CALIBRATOR_NOT_OOT',
            'select':'EARLY_STOP_SELECTION_AND_FUTURE_CALIBRATOR_NOT_OOT',
            'calibration':'CALIBRATOR_FIT_IN_SAMPLE_NOT_INDEPENDENT',
            'test':'HISTORICAL_TEST_ALREADY_INSPECTED_NOT_NEW_BLIND_TEST'}[role]
    coefficients={}
    for name in ('lr','lightgbm'):
        c=model['calibrators'][name]
        require(c.coef_[0,0]>0,'NONPOSITIVE_CALIBRATION_SLOPE')
        support=prediction_arrays['calibration'][name]['raw']
        target=prediction_arrays['test'][name]['raw']
        coefficients[name]=dict(slope=float(c.coef_[0,0]),intercept=float(c.intercept_[0]),
            formula='sigmoid(slope * clipped_logit(raw_p) + intercept)',
            calibration_raw_min=float(support.min()),calibration_raw_max=float(support.max()),
            test_raw_outside_calibration_range_fraction=float(np.mean((target<support.min())|(target>support.max()))))
    thresholds=market_thresholds([dict(r,segment='train') for r in groups['train']])
    monthly,boards,regimes,codes=defaultdict(list),defaultdict(list),defaultdict(list),defaultdict(list)
    for r in test:
        monthly[str(r['P']//100)].append(r)
        boards[board(r['code'])].append(r)
        regimes[regime(r['features']['market_vol_20d'],thresholds)].append(r)
        codes[r['code']].append(r)
    summary=swap_summary(test)
    expected=json.loads((OUTPUT/'F2_report.json').read_bytes())['paired_additions']['lightgbm_minus_lr_tail_rate']['mean']
    np.testing.assert_allclose(summary['paired_rate_delta'],expected,rtol=0,atol=1e-15)
    case_groups=swap_groups(test)
    cases=[]
    for name in ('gb_only','lr_only'):
        for r in sorted(case_groups[name],key=lambda r:(r['mae_5d'],r['P'],r['code'])):
            cases.append(dict(subset=name,**{k:r[k] for k in ('code','P','label_end_date','mae_5d','y_tail_5d',
                'lr_raw','lr','lightgbm_raw','lightgbm')},board=board(r['code'])))
    clusters={name:overlap_clusters([r for r in case_groups[name] if r['y_tail_5d']==1])
              for name in ('gb_only','lr_only')}
    result=dict(status='READONLY_2024_DIAGNOSIS_COMPLETED',fit_refit_performed=False,network_requests=0,
        exploratory_after_label_inspection=True,roles=roles,calibrators=coefficients,
        swaps=summary,by_month={k:swap_summary(v) for k,v in sorted(monthly.items())},
        by_board={k:swap_summary(v) for k,v in sorted(boards.items())},
        by_market_regime={k:swap_summary(v) for k,v in sorted(regimes.items())},
        by_code={k:swap_summary(v) for k,v in sorted(codes.items())},train_market_vol_terciles=thresholds,
        overlapping_tail_window_clusters=clusters,
        feature_drift_calibration_to_test=feature_drift(groups['calibration'],groups['test']),
        note='No root-cause proof or new model selection; relative subset tail means are not additive whole-arm deep-tail differences.')
    paths={Path(r['path']) for r in manifest['inputs']}
    paths.update(Path(r['path']) for r in run_refs)
    paths.update((MANIFEST,OUTPUT/'stop_record.json',HERE/'crossyear_fit_result_audit_v1.json',
                  Path(__file__),HERE/'test_diagnose_risk_2024_v1.py'))
    refs=[dict(path=str(p),sha256=sha(p)) for p in sorted(paths)]
    contents={'diagnostics.json':encoded(result),'swap_cases.json':encoded(cases),
        'input_manifest.json':encoded(dict(inputs=refs,scope='READONLY_NO_FIT_2024_DIAGNOSIS'))}
    import hashlib
    terminal=encoded(dict(status='completed',fit_performed=False,network_requests=0,
        files=[dict(name=n,bytes=len(data),sha256=hashlib.sha256(data).hexdigest()) for n,data in contents.items()]))
    total=sum(map(len,contents.values()))+len(terminal)
    require(total<=CAP and bytes_used()+total<=TOTAL_CAP,'DIAGNOSTIC_BUDGET_STOP')
    verify_refs(refs)
    DEST.mkdir(exist_ok=False)
    for name,data in contents.items():
        with (DEST/name).open('xb') as f:
            f.write(data)
    with (DEST/'stop_record.json').open('xb') as f:
        f.write(terminal)
    print(json.dumps(dict(output=str(DEST),total_bytes=total,fit_performed=False,swap_summary=summary),ensure_ascii=False))


if __name__=='__main__':
    run()
