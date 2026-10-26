"""Review-only feature operations; no estimator, real-data preprocessing fit or I/O."""
import numpy as np
import pandas as pd

DERIVED=['time_node_nearest_abs_distance','session_is_pm','context_sector_minus_sse_return_raw',
         'context_stock_minus_sector_return_raw','context_stock_minus_leader_return_raw']


def whitelist(config):
    fields=[f for group in config['feature_groups'].values() for f in group]
    forbidden={'code','stock_close','stock_reference_close_raw','training_eligible','joint_training_admitted',
               'stock_corporate_action_scale_verified','proposed_segment','label_state','source_cache'}
    if not fields or len(fields)!=len(set(fields)):raise ValueError('EMPTY_OR_DUPLICATE_WHITELIST')
    if any(f in forbidden or f.startswith(('diagnostic_','y_tail','mae_','label_','leader_observed_','own_concepts_'))
           or 'golden_level_' in f or f.endswith(('anchor_low','anchor_high')) for f in fields):
        raise ValueError('FORBIDDEN_FUTURE_AUDIT_OR_ABSOLUTE_FIELD')
    return fields


def derive(frame):
    result=frame.copy()
    ratios=['0_236','0_382','0_5','0_618','0_786']
    distances=result[[f'sse_15m_time_fraction_distance_{r}' for r in ratios]].to_numpy(dtype=float)
    valid=np.isfinite(distances).any(axis=1)
    closest=np.min(np.where(np.isfinite(distances),np.abs(distances),np.inf),axis=1)
    closest[~valid]=np.nan
    result['time_node_nearest_abs_distance']=closest
    result['session_is_pm']=((result.decision_ts_assumed_end//10000)%100>=13).astype('int8')
    result['context_sector_minus_sse_return_raw']=result.context_sector_mean_return_loo_raw-result.sse_15m_day_return_prev_close
    result['context_stock_minus_sector_return_raw']=result.stock_day_return_raw-result.context_sector_mean_return_loo_raw
    result['context_stock_minus_leader_return_raw']=result.stock_day_return_raw-result.context_current_leader_return_loo_raw
    return result


def group_observation_masks(frame,groups):
    return {name:np.isfinite(frame[fields].to_numpy(dtype=float)).any(axis=1) for name,fields in groups.items()}


def numeric(frame):
    data=frame.to_numpy(dtype=float)
    if np.isinf(data).any():raise ValueError('INFINITE_FEATURE_NOT_MISSING')
    return data


def fit_lr_preprocessing(train,phase='train'):
    """Only synthetic tests call this in preparation; never fit on select/cal/test."""
    if phase!='train':raise ValueError('PREPROCESSING_TRAIN_ONLY')
    x=numeric(train)
    if len(x)==0 or np.isnan(x).all(axis=0).any():raise ValueError('EMPTY_OR_ALL_NULL_TRAIN_COLUMN')
    median=np.nanmedian(x,axis=0);filled=np.where(np.isnan(x),median,x)
    mean=filled.mean(axis=0);scale=filled.std(axis=0)
    return dict(columns=list(train.columns),median=median,mean=mean,
                scale=np.where(scale>0,scale,1.))


def transform_lr(frame,state):
    if list(frame.columns)!=state['columns']:raise ValueError('FEATURE_ORDER_CHANGED')
    x=numeric(frame);missing=np.isnan(x)
    values=(np.where(missing,state['median'],x)-state['mean'])/state['scale']
    return np.c_[values,missing.astype(float)]


def native_missing_view(frame):
    x=numeric(frame)
    return np.c_[x,np.isnan(x).astype(float)]


def cutoff_support(times,minimum=5):
    counts=pd.Series(times).value_counts()
    return dict(cutoffs=int(len(counts)),rows=int(counts.sum()),
        comparable_cutoffs=int((counts>=minimum).sum()),comparable_rows=int(counts[counts>=minimum].sum()),
        tiny_cutoffs=int((counts<minimum).sum()),minimum_cross_section=minimum,
        max_stocks=int(counts.max()) if len(counts) else 0)
