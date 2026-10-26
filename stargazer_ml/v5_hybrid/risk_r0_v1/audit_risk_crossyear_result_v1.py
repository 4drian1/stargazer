"""Read-only model replay and independent weighted-MAE bootstrap; NEVER calls fit."""
import json
import math
from pathlib import Path

import joblib
import numpy as np
import pyarrow.parquet as pq

from price_verification_guarded_run import encoded, sha
from research_budget_v2 import bytes_used, TOTAL_CAP
from risk_crossyear_v1 import HERE, MANIFEST, OUTPUT, load_data, require, verify_refs


def weighted_tail(sorted_values, integer_weights):
    """Equivalent to linear np.quantile on repeated observations, without expansion."""
    total = int(integer_weights.sum())
    require(total > 0, 'EMPTY_WEIGHTED_TAIL')
    rank = .05*(total-1)
    cumulative = np.cumsum(integer_weights)
    lower = sorted_values[np.searchsorted(cumulative, math.floor(rank), side='right')]
    upper = sorted_values[np.searchsorted(cumulative, math.ceil(rank), side='right')]
    fraction = rank-math.floor(rank)
    # Difference interpolation preserves exact ties; weighted-sum interpolation can
    # round one ulp below a repeated cutoff and wrongly exclude the entire tie.
    difference = upper-lower
    threshold = lower+difference*fraction if fraction < .5 else upper-difference*(1-fraction)
    selected = sorted_values <= threshold
    return float(np.sum(sorted_values[selected]*integer_weights[selected])/integer_weights[selected].sum())


def close(got, expected):
    np.testing.assert_allclose(got, expected, rtol=1e-11, atol=1e-12)


def audit_evaluation(rows, predictions, report, evaluation):
    dates = sorted({r['P'] for r in rows})
    ids = np.array([dates.index(r['P']) for r in rows])
    counts = np.bincount(ids)
    y = np.array([r['y_tail_5d'] for r in rows])
    mae = np.array([r['mae_5d'] for r in rows])
    n, block, reps, seed = len(dates), evaluation['block_dates'], evaluation['bootstrap_reps'], evaluation['seed']
    rng = np.random.default_rng(seed)
    starts = rng.integers(0,n-block+1,size=(reps,math.ceil(n/block)))
    draws = np.array([[s+j for s in row for j in range(block)][:n] for row in starts])
    constant = np.array([r['p_constant'] for r in predictions])
    for model in ('constant','lr_raw','lr','lightgbm_raw','lightgbm'):
        p = constant if model == 'constant' else np.array([r[model] for r in predictions])
        q = np.clip(p,1e-12,1-1e-12)
        close(np.mean((q-y)**2),report['probability'][model]['brier'])
        close(-np.mean(y*np.log(q)+(1-y)*np.log1p(-q)),report['probability'][model]['log_loss'])
        if model != 'constant':
            diff = (q-y)**2-(constant-y)**2
            daily = np.bincount(ids,weights=diff)
            close([diff.mean(),*np.quantile(daily[draws].sum(1)/counts[draws].sum(1),[.025,.975])],
                  [report['probability'][model]['brier_minus_constant']['mean'],
                   *report['probability'][model]['brier_minus_constant']['ci95']])
    primary_masks = None
    for summary in report['filtering']:
        masks = {}
        for model in ('atr_rule','lr','lightgbm'):
            score = np.array([r['features']['atr14_pct'] for r in rows]) if model == 'atr_rule' else np.array(
                [r[model] for r in predictions])
            mask = np.ones(len(rows),dtype=bool)
            for d in range(n):
                ix = np.flatnonzero(ids==d)
                ranked = sorted(ix,key=lambda i:(-score[i],rows[i]['code']))
                mask[ranked[:math.floor(len(ix)*summary['rejection_fraction'])]]=False
            masks[model]=mask
            if summary['role']=='primary':
                require(np.array_equal(mask,[r[model+'_retained_primary'] for r in predictions]),'MASK_DRIFT')
            arm=summary['arms'][model]
            require(int(mask.sum())==arm['retained'],'RETAINED_COUNT_DRIFT')
            close(y[mask].mean(),arm['tail_rate'])
            close(np.mean(mae[mask]),arm['mean_mae'])
            v=np.sort(mae[mask])
            close(weighted_tail(v,np.ones(len(v),dtype=int)),arm['worst5pct_mae_mean'])
        kept = np.bincount(ids,weights=masks['atr_rule'])
        require(all(np.array_equal(kept,np.bincount(ids,weights=m)) for m in masks.values()),'UNPAIRED_COUNTS')
        tails = {model:np.bincount(ids,weights=y*mask) for model,mask in masks.items()}
        tails['random_expected']=np.bincount(ids,weights=y)*kept/counts
        close(tails['random_expected'].sum()/kept.sum(),summary['random_expected_tail_rate'])
        pairs = [('lr','atr_rule'),('lr','random_expected'),('lightgbm','atr_rule'),('lightgbm','random_expected')]
        if summary['role']=='primary':
            primary_masks=masks
            pairs += [('lightgbm','lr')]
        for a,b in pairs:
            delta=tails[a]-tails[b]
            point=delta.sum()/kept.sum()
            ci=np.quantile(delta[draws].sum(1)/kept[draws].sum(1),[.025,.975])
            expected=(report['paired_additions']['lightgbm_minus_lr_tail_rate'] if b=='lr'
                      else summary['paired_tail_rate_differences'][a+'_minus_'+b])
            close([point,*ci],[expected['mean'],*expected['ci95']])
    ordered = {}
    for model,mask in primary_masks.items():
        ix=np.flatnonzero(mask)
        ix=ix[np.argsort(mae[ix],kind='stable')]
        ordered[model]=(mae[ix],ids[ix])
    deep_draws={model:[] for model in ('atr_rule','lr')}
    for draw in draws:
        multiplicities=np.bincount(draw,minlength=n)
        tails={m:weighted_tail(v,multiplicities[date_ids]) for m,(v,date_ids) in ordered.items()}
        for baseline in deep_draws:
            deep_draws[baseline].append(tails['lightgbm']-tails[baseline])
    for baseline,values in deep_draws.items():
        expected=report['paired_additions']['worst5pct_mae_differences']['lightgbm_minus_'+baseline]
        point=weighted_tail(ordered['lightgbm'][0],np.ones(len(ordered['lightgbm'][0]),dtype=int))-weighted_tail(
            ordered[baseline][0],np.ones(len(ordered[baseline][0]),dtype=int))
        close(point,expected['mean'])
        close(np.quantile(values,[.025,.975]),expected['ci95'])
    return dict(n=len(rows),dates=n,all_three_filter_proportions_replayed=True,
        probabilities_brier_logloss_verified=True,paired_tail_cis_recomputed=True,
        deep_mae_cis_independently_weighted_recomputed=True)


def run():
    destination=HERE/'crossyear_fit_result_audit_v1.json'
    require(not destination.exists(),'AUDIT_EXISTS_NO_OVERWRITE')
    stop=json.loads((OUTPUT/'stop_record.json').read_bytes())
    require(stop['status']=='completed' and stop['fit_completed_folds']==['F1','F2']
            and stop['reused_folds']==['F3'],'BATCH_NOT_COMPLETE')
    verify_refs([dict(path=str(OUTPUT/r['name']),sha256=r['sha256']) for r in stop['files']])
    manifest=json.loads(MANIFEST.read_bytes())
    verify_refs(manifest['inputs'])
    config,base,groups,reused,_,fingerprints=load_data()
    require(fingerprints==manifest['fold_populations'],'POPULATION_DRIFT')
    saved=json.loads((OUTPUT/'input_and_authorization.json').read_bytes())
    require(sha(saved['authorization_path'])==saved['authorization_sha256'],'AUTHORIZATION_DRIFT')
    require(sha(MANIFEST)==saved['check']['manifest_sha256'],'MANIFEST_DRIFT')
    result={}
    for fold in ('F1','F2','F3'):
        rows=groups[fold]['test']
        predictions=pq.read_table(OUTPUT/(fold+'_test_predictions.parquet')).to_pylist()
        require(len(rows)==len(predictions),'PREDICTION_COUNT')
        require(len({(r['code'],r['P']) for r in predictions})==len(predictions),'DUPLICATE_PREDICTIONS')
        for row,prediction in zip(rows,predictions):
            require(all(row[k]==prediction[k] for k in ('code','P','label_end_date','y_tail_5d','mae_5d')),
                    'LABEL_IDENTITY_DRIFT')
        if fold != 'F3':
            models=joblib.load(OUTPUT/(fold+'_models.joblib'))
            x=np.array([[r['features'][f] for f in base['fields']] for r in rows])
            for name in ('lr','lightgbm'):
                raw=models[name].predict_proba(x)[:,1]
                close(raw,[r[name+'_raw'] for r in predictions])
                q=np.clip(raw,1e-12,1-1e-12)
                p=models['calibrators'][name].predict_proba(np.log(q/(1-q)).reshape(-1,1))[:,1]
                close(p,[r[name] for r in predictions])
        else:
            require(all(all(r[k]==old[k] for k in old) for r,old in zip(predictions,reused)),'F3_REUSE_DRIFT')
            require(not (OUTPUT/'F3_models.joblib').exists(),'F3_UNEXPECTED_MODEL')
        close([r['p_constant'] for r in predictions],np.mean([r['y_tail_5d'] for r in groups[fold]['train']]))
        report=json.loads((OUTPUT/(fold+'_report.json')).read_bytes())
        result[fold]=audit_evaluation(rows,predictions,report,base['evaluation'])
    verify_refs(manifest['inputs'])
    # Independent research-flag check, without calling the frozen consistency function.
    reports=[json.loads((OUTPUT/(f+'_report.json')).read_bytes()) for f in ('F1','F2','F3')]
    pair_keys=('lightgbm_minus_atr_rule','lightgbm_minus_random_expected')
    pairs=[r['filtering'][0]['paired_tail_rate_differences'] for r in reports]
    tail=(sum(all(p[k]['ci95'][1]<0 for k in pair_keys) for p in pairs)>=2
          and all(p[k]['mean']<=0 for p in pairs for k in pair_keys))
    probability=sum(r['probability']['lightgbm']['brier_minus_constant']['ci95'][1]<0
        and r['probability']['lightgbm']['log_loss']<=r['probability']['constant']['log_loss'] for r in reports)>=2
    red=[i+1 for i,r in enumerate(reports) if any(v['ci95'][1]<0
        for v in r['paired_additions']['worst5pct_mae_differences'].values())]
    flags=json.loads((OUTPUT/'summary.json').read_bytes())['flags']
    require(flags['tail_consistency']==tail and flags['probability_consistency']==probability
        and flags['deep_tail_red_flag_folds']==red and flags['research_consistency']==(tail and probability and not red),
        'SUMMARY_FLAG_DRIFT')
    audit=dict(status='CROSSYEAR_RESULT_AUDIT_PASS',fit_refit_performed=False,network_requests=0,
        folds=result,predictions_verified=sum(r['n'] for r in result.values()),summary_flags_verified=flags,
        frozen_inputs_unchanged=len(manifest['inputs']),run_actual_bytes=sum(p.stat().st_size for p in OUTPUT.iterdir()),
        registered_actual_bytes=bytes_used(),audit_code=dict(path=str(Path(__file__).resolve()),sha256=sha(__file__)),
        files=[dict(path=str(p),sha256=sha(p)) for p in sorted(OUTPUT.iterdir()) if p.is_file()])
    data=encoded(audit)
    require(len(data)<2*1024**2 and bytes_used()+len(data)<=TOTAL_CAP,'AUDIT_BUDGET')
    with destination.open('xb') as f:
        f.write(data)
    print(json.dumps(dict(output=str(destination),status=audit['status'],predictions_verified=audit['predictions_verified'],
        run_actual_bytes=audit['run_actual_bytes'],flags=flags),ensure_ascii=False))


if __name__=='__main__':
    run()
