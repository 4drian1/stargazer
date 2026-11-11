"""Isolated memory schedule for the unchanged momentum kernel.

The frozen kernel already computes each symbol independently. Execute it once
per symbol, then retain only its 15-minute decision keys before projection.
project_market discards exactly these other 5-minute keys itself. No indicator,
seed, or projected value is changed; production files are untouched.
"""
import gc
import pandas as pd
import materialize_multitimeframe_momentum_v1 as module

ORIGINAL = module.build

def bounded(frame):
    parts=[]
    for _,part in frame.groupby('symbol',sort=True):
        result=ORIGINAL(part)
        t=result['decision_ts_assumed_end']
        parts.append(result[(t//100%100)%15==0].copy())
        del result
        gc.collect()
    return pd.concat(parts,ignore_index=True).sort_values(module.KEYS).reset_index(drop=True)

def enable():
    module.build=bounded
