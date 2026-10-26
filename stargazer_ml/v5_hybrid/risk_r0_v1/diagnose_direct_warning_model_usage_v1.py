"""Post-fit descriptive feature-use audit; not selection, tuning or refitting."""
import json

import lightgbm as lgb
import numpy as np

import direct_warning_fixed_fit_v1 as fit
from audit_direct_warning_fixed_fit_v1 import write_exclusive


def run():
    acceptance=fit.HERE/'direct_warning_fixed_fit_acceptance_20261018_v1.json'
    if json.loads(acceptance.read_bytes())['status']!='PASS_FIXED_DIRECT_WARNING_RESEARCH_REPLAY_NOT_PROMOTION':
        raise ValueError('FIT_READBACK_REQUIRED')
    config=json.loads((fit.prep.OUT/'config.json').read_bytes())
    fields=config['arms']['C1']; n=len(fields)
    train=fit.prep.load_arm('train','C1')
    model=fit.OUT/'C1_lightgbm_model.txt'
    booster=lgb.Booster(model_file=str(model))
    splits=booster.feature_importance(importance_type='split')
    coefficients=json.loads((fit.OUT/'C1_lr_model.json').read_bytes())['coef']
    if len(splits)!=2*n or len(coefficients)!=2*n:
        raise ValueError('FEATURE_DIMENSION_DRIFT')
    rows={}
    for f in config['added_fields']:
        i=fields.index(f); values=train[f].to_numpy(dtype=float)
        rows[f]=dict(train_known=int(np.isfinite(values).sum()),train_missing=int(np.isnan(values).sum()),
            train_nonzero_known=int((np.isfinite(values)&(values!=0)).sum()),
            train_distinct=int(train[f].nunique(dropna=True)),
            tree_numeric_splits=int(splits[i]),tree_missing_indicator_splits=int(splits[n+i]),
            lr_standardized_numeric_coefficient=float(coefficients[i]),
            lr_missing_indicator_coefficient=float(coefficients[n+i]))
    report=dict(status='POSTHOC_DESCRIPTIVE_MODEL_USAGE_NOT_SELECTION',
        inputs={str(p):fit.previous.sha(p) for p in [model,fit.OUT/'C1_lr_model.json',fit.prep.OUT/'config.json',acceptance]},
        new_field_numeric_splits=sum(r['tree_numeric_splits'] for r in rows.values()),
        new_field_missing_splits=sum(r['tree_missing_indicator_splits'] for r in rows.values()),
        total_tree_splits=int(splits.sum()),fields=rows,
        main_fits=0,calibration_fits=0,http_requests=0,
        interpretation='Zero new-field tree splits means no direct dependence on the added fields. Appending duplicate columns can change colsample/regularization; prediction differences do not identify new warning signal effects. Nonzero LR coefficients alone are not evidence of useful risk prediction.')
    write_exclusive(fit.HERE/'direct_warning_model_usage_diagnostic_20261018_v1.json',report)
    return dict(new_field_tree_splits=report['new_field_numeric_splits']+report['new_field_missing_splits'],
                total_tree_splits=report['total_tree_splits'],fields=len(rows))


if __name__=='__main__':
    print(json.dumps(run()))
