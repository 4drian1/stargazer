"""Only frozen whitelist replacement and paired probability diagnostics; no fit."""
import numpy as np

from multiboard_fit_evaluation_v1 import probabilities, paired_loss, filtering


def mapped_inputs(frame,fields):
    if not fields or len(fields)!=len(set(fields)):
        raise ValueError('INVALID_FROZEN_WHITELIST')
    if any(f.startswith(('ar_','nh_')) or f.startswith(('y_tail','mae_','label_')) for f in fields):
        raise ValueError('WHITELIST_NOT_EXISTING_MODEL_INPUTS')
    old=frame[fields].copy()
    new=old.copy()
    mapping={f:'ar_'+f for f in fields if 'ar_'+f in frame}
    for field,source in mapping.items():
        new[field]=frame[source].to_numpy()
    if np.isinf(old.to_numpy(dtype=float)).any() or np.isinf(new.to_numpy(dtype=float)).any():
        raise ValueError('INFINITE_MODEL_INPUT')
    changed=~np.isclose(old.to_numpy(dtype=float),new.to_numpy(dtype=float),equal_nan=True,rtol=0,atol=1e-12)
    return old,new,mapping,changed


def compare(frame,original,conditional,policy):
    y=frame.y_tail_1200m.to_numpy()
    from prepare_multiboard_comparison_v1 import TIME
    dates=frame[TIME].to_numpy()//1000000
    old,new=probabilities(y,original),probabilities(y,conditional)
    return dict(original=old,conditional=new,
        delta_conditional_minus_original={k:new[k]-old[k] if new[k] is not None and old[k] is not None else None
            for k in ['brier','log_loss','roc_auc','pr_auc_average_precision','predicted_risk','actual_minus_predicted']},
        paired_loss=paired_loss(y,conditional,original,dates,policy['uncertainty']),
        same_cutoff_filter=filtering(frame,dict(original=original,conditional=conditional),policy),
        filter_difference_is_point_estimate_not_a_paired_confidence_interval=True,
        changed_probability_rows=int((np.abs(conditional-original)>1e-12).sum()),
        maximum_absolute_probability_change=float(np.max(np.abs(conditional-original))))
