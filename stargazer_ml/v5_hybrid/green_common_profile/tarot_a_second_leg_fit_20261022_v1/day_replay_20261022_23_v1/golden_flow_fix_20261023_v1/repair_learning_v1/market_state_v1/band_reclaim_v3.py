"""User-confirmed band-upper qualification; preserves the selected v2 node."""
from copy import deepcopy


def qualify_band(parts, band, previous_close, current_close):
    result=deepcopy(parts);position=result['position'];path=position['current_path']
    upper=None if band is None else band['upper']
    two_above=(previous_close is not None and upper is not None
               and previous_close>=upper and current_close>=upper)
    position.update(band_upper=upper,
        two_closes_above_band_upper=bool(two_above) if path=='RECLAIM' else None,
        strong_position_qualified=path=='STOP_ABOVE' or (path=='RECLAIM' and bool(two_above)))
    if path=='RECLAIM' and not two_above:
        result.update(state='BAND_RECLAIM_PENDING',label='节点已收回，整带尚未连续两根收回')
    result['reference_version']='V3_BAND_UPPER_TWO_COMPLETED_CLOSES'
    return result
