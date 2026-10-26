"""Logical new-version population; never alters the first fit or old cohort."""
import json
import numpy as np
import pandas as pd
from finalize_postclose_cohort_v3 import read_population, corrected_masks
from inventory_ths_stockdb_cache_v1 import digest
from repair_postclose_daily_context_v1 import OUT


def read_repaired_population():
    original, frozen = read_population()
    manifest = json.loads((OUT / 'manifest.json').read_bytes())
    path = OUT / 'daily_context_overlay.parquet'
    if digest(path.read_bytes()) != manifest['outputs'][path.name]['sha256']:
        raise ValueError('OVERLAY_HASH_DRIFT')
    overlay = pd.read_parquet(path)
    if not original[['code','day']].equals(overlay[['code','day']]):
        raise ValueError('OVERLAY_KEY_DRIFT')
    if manifest['fields_preserved'] != frozen['fields']:
        raise ValueError('OLD_FIELD_CONTRACT_DRIFT')
    result = original.copy()
    for column in overlay:
        if column.startswith('context_'):
            result[column] = overlay[column].to_numpy()
    result['context_stock_minus_sector_daily_return'] = result.stock_return1d - result.context_sector_mean_return_loo_raw
    result['context_stock_minus_leader_daily_return'] = result.stock_return1d - result.context_current_leader_return_loo_raw
    masks = corrected_masks(result)
    eligible = result.label_state.eq('CONDITIONAL_NEXT_SESSION_COMPLETE') & ~result.proposed_purged & np.logical_and.reduce(list(masks.values()))
    if not np.array_equal(eligible, overlay.proposed_repaired_cohort_eligible):
        raise ValueError('PROPOSED_COHORT_DRIFT')
    result['research_cohort_eligible'] = eligible
    return result, manifest
