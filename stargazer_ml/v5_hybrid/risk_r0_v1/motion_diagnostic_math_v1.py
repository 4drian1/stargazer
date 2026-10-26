"""Read-only descriptive arithmetic, not feature admission or a new predictor."""
import numpy as np


def static_membership(boards,bound,codes):
    result={}
    for code in codes:
        related=[b for b in boards if code in b['symbols']]
        possible=[b for b in related if (set(b['symbols'])-{code})<=bound]
        result[code]=dict(related=len(related),potentially_complete=len(possible),
            possible_board_codes=[b['code'] for b in possible],
            possible_board_names=[b.get('name') for b in possible])
    return result


def old_linear_contributions(frame,fields,state,coefficient):
    n=len(fields);total=len(state['columns']);beta=np.asarray(coefficient,dtype=float)
    if state['columns'][:n]!=fields or beta.shape!=(2*total,):raise ValueError('COEFFICIENT_SCHEMA_MISMATCH')
    x=frame[fields].to_numpy(dtype=float);missing=np.isnan(x)
    median=np.asarray(state['median'])[:n];mean=np.asarray(state['mean'])[:n];scale=np.asarray(state['scale'])[:n]
    if np.isinf(x).any() or not (np.isfinite(scale)&(scale>0)).all():raise ValueError('INVALID_STANDARDIZATION')
    z=(np.where(missing,median,x)-mean)/scale
    contribution=z*beta[:n]+missing*beta[total:total+n]
    return contribution,z


def outcome_mix(train_code,train_y,code,y):
    """Descriptive code-mixture identity. Unknown codes use disclosed train mean.

    This is not fitted or deployed as a risk model and never alters labels.
    """
    tc=np.asarray(train_code);ty=np.asarray(train_y,dtype=float)
    c=np.asarray(code);actual=np.asarray(y,dtype=float)
    if not len(actual) or not len(ty) or not np.isfinite(actual).all() or not np.isfinite(ty).all():
        raise ValueError('INVALID_DESCRIPTIVE_GROUP')
    overall=float(ty.mean());means={v:float(ty[tc==v].mean()) for v in np.unique(tc)}
    reference=np.array([means.get(v,overall) for v in c]);expected=float(reference.mean())
    return dict(train_mean=overall,observed_mean=float(actual.mean()),
        fixed_train_code_mean_reference=expected,population_mix_component=expected-overall,
        within_code_outcome_shift=float(actual.mean())-expected,
        unseen_code_rows=int(sum(v not in means for v in c)),
        unknown_code_policy='train global mean for diagnostic identity only',causal=False)
