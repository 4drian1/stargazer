"""Daily-basis evidence uses native boundary records, not just label slots.

Does not move timestamps, insert prices or change the 1200-slot target clock.
"""
import numpy as np

from decision_after_labels_v1 import aligned_days


def native_aligned_days(grid,days,bars,native,tolerance=.01):
    out=aligned_days(grid,days,bars,tolerance)
    prices=['open','high','low','close']
    extras=native[native.date.isin(np.r_[days*1000000+93000,days*1000000+130000])]
    shaped=grid.reshape(len(days),240,4)
    for day,group in extras.groupby(extras.date//1000000,sort=False):
        index=np.searchsorted(days,day)
        if index>=len(days) or days[index]!=day:continue
        out[index]=False
        if not np.isfinite(shaped[index]).all() or int(day) not in bars:continue
        if (group.groupby('date')[prices].nunique(dropna=False).max(axis=1)>1).any():continue
        group=group.drop_duplicates('date').sort_values('date')
        values=group[prices].to_numpy(dtype=float);o,h,l,c=values.T
        if not (np.isfinite(values).all() and (values>0).all() and (l<=np.minimum(o,c)).all() and (h>=np.maximum(o,c)).all()):continue
        opening=group[group.date==int(day)*1000000+93000]
        actual=np.array([float(opening.open.iloc[0]) if len(opening) else shaped[index,0,0],
            max(shaped[index,:,1].max(),h.max()),min(shaped[index,:,2].min(),l.min()),shaped[index,-1,3]])
        expected=np.array([bars[int(day)].get(p,np.nan) for p in prices],dtype=float)
        valid=(np.isfinite(expected).all() and 0<expected[2]<=min(expected[0],expected[3])<=max(expected[0],expected[3])<=expected[1])
        out[index]=valid and np.all(np.abs(actual-expected)<=tolerance+1e-12)
    return out
