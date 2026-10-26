"""Read-only three-arm A setup replay on fixed P-close daily prefixes."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.dataset as ds

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(ROOT/'stargazer_ml/v5_hybrid/green_common_profile'),
                str(ROOT/'stargazer_desktop')]
from gcp_demo_pools import process_latest_code
from golden_rules import build_rule, cycle_anchors

P_DATES = (20261019, 20261020, 20261021)
DAILY = Path('D:/a股数据/data_daily_parts')
OLD = HERE/'stage_v2_replay_20261016_19_20.json'
RATIOS = (.236, .382, .5, .618, .786)


def new_conditions(g, div_index, atr):
    lows = g.low.to_numpy(dtype=float)
    highs = g.high.to_numpy(dtype=float)
    volume = g.volume.to_numpy(dtype=float)
    pivot = next((i for i in range(div_index-3, max(1, div_index-60)-1, -1)
                  if i >= 2 and lows[i] <= min(lows[i-2:i+3])), None)
    wave = {'status': 'NO_CONFIRMED_SWING_LOW'}
    if pivot is not None:
        peak = pivot + int(np.argmax(highs[pivot:div_index+1]))
        a, b = float(lows[pivot]), float(highs[peak])
        rise = b/a-1 if a > 0 else float('nan')
        retrace = (b-float(g.close.iloc[-1]))/(b-a) if b > a else float('nan')
        levels = {str(r): round(b-r*(b-a), 4) for r in RATIOS}
        below = [(r, p) for r, p in ((r, b-r*(b-a)) for r in RATIOS)
                 if p <= float(g.close.iloc[-1])+1e-10]
        node = max(below, key=lambda item: item[1]) if below else None
        near = min(((abs(float(g.close.iloc[-1])-p)/p, r)
                    for r, p in ((r, b-r*(b-a)) for r in RATIOS) if p > 0),
                   default=(float('inf'), None))
        valid = bool(peak > pivot and rise > .15 and .236 <= retrace <= .786 and node)
        wave = {'status': 'OK' if valid else 'WAVE_GATE_FAIL',
                'pivot_date': int(g.date.iloc[pivot]), 'pivot_low': a,
                'peak_date': int(g.date.iloc[peak]), 'peak_high': b,
                'rise': round(rise, 6), 'retrace': round(retrace, 6),
                'levels': levels, 'near_level': near[1],
                'near_level_within_1pct': bool(near[0] <= .01),
                'observation_node': round(node[1], 4) if node else None,
                'observation_zone': [round(node[1], 4),
                                     round(min(node[1]+.2*atr, float(g.close.iloc[-1])), 4)]
                                    if valid else None,
                'deep_pullback_needs_pyxis': bool(retrace >= .618) if valid else None}
    panic = None
    for i in range(div_index+1, len(g)):
        if i < 8:
            continue
        mean8 = float(np.mean(volume[i-8:i]))
        if (float(g.close.iloc[i]) < float(g.open.iloc[i])
                and volume[i] > 1.25*mean8):
            panic = {'date': int(g.date.iloc[i]), 'volume': float(volume[i]),
                     'prior8_mean': mean8}
            break
    last3_shrinking = bool(len(g) >= 3 and volume[-3] > volume[-2] > volume[-1])
    end_vs_panic = float(volume[-1]/panic['volume']) if panic else None
    volume_ok = bool(panic and end_vs_panic < .70 and last3_shrinking)
    return {'panic': panic, 'end_volume_to_panic': end_vs_panic,
            'last3_volume_shrinking': last3_shrinking, 'volume_gate': volume_ok,
            'wave': wave, 'new_gate': bool(volume_ok and wave['status'] == 'OK')}


def replay():
    prior = json.loads(OLD.read_text(encoding='utf-8'))
    by_date = {p: {r['code'] for r in prior['days'][str(p)]['old_common']
                   if r['setup'] == 'A_SDP'} for p in P_DATES}
    codes = sorted(set.union(*by_date.values()))
    paths = sorted(DAILY.glob('part_*.parquet'))
    dataset = ds.dataset([str(p) for p in paths], format='parquet')
    columns = ['date','code','name','open','high','low','close','pre_close','volume',
               'amount','is_st','float_mv']
    days = {str(p): {'old_a': len(by_date[p]), 'rows': [], 'arms': {
        'OLD_A': [], 'WAVE_A': [], 'BOTH_A': []}} for p in P_DATES}
    for code in codes:
        g = dataset.to_table(columns=columns, filter=(ds.field('code') == code)
                             & (ds.field('date') <= max(P_DATES))).to_pandas()
        g = g.drop_duplicates(['code','date'], keep='last').sort_values('date')
        for p in P_DATES:
            if code not in by_date[p]:
                continue
            prefix = g[g.date <= p].reset_index(drop=True)
            if prefix.empty or int(prefix.date.iloc[-1]) != p:
                raise ValueError('P_PREFIX_MISSING:'+code+':'+str(p))
            a, _ = process_latest_code(prefix)
            if len(a) != 1 or str(a.iloc[0].setup_type) != 'A_SDP':
                raise ValueError('OLD_A_REPLAY_MISMATCH:'+code+':'+str(p))
            div_date = int(a.iloc[0].div_date)
            div_index = int(prefix.index[prefix.date == div_date][0])
            daily = {int(r.date): {k: float(getattr(r,k)) for k in ('open','high','low','close')}
                     for r in prefix.itertuples()}
            tr = pd.concat([prefix.high-prefix.low,
                            (prefix.high-prefix.close.shift(1)).abs(),
                            (prefix.low-prefix.close.shift(1)).abs()],axis=1).max(axis=1)
            atr = float(tr.rolling(14).mean().tail(5).median())
            anchor = cycle_anchors(daily)
            last = prefix.iloc[-1]
            old = build_rule(float(last.close),float(last.high),float(last.low),
                             atr,anchor['high'],anchor['low'])
            new = new_conditions(prefix,div_index,atr)
            old_valid = bool(old['eligible'] and old['f618'] <= float(last.close))
            wave_valid = bool(new['new_gate'])
            row = {'code': code, 'name': str(last['name']), 'P': p,
                   'close': float(last.close), 'div_date': div_date,
                   'atr_ref': round(atr, 6),
                   'old_cycle_pass': bool(old['eligible']),
                   'old_cycle_distance_atr': round(old['cycle_distance_atr'], 6),
                   'old_f618': old['f618'],
                   'old_zone_valid': bool(old['f618'] <= float(last.close)),
                   'old_zone': [old['f618'],round(min(old['f618']+.2*atr,
                       float(last.close)),4)] if old_valid else None,
                   **new}
            days[str(p)]['rows'].append(row)
            if old_valid:
                days[str(p)]['arms']['OLD_A'].append(code)
            if wave_valid:
                days[str(p)]['arms']['WAVE_A'].append(code)
            if wave_valid and old['eligible']:
                days[str(p)]['arms']['BOTH_A'].append(code)
    return {'version': 'A_COMBINED_V1_RESEARCH_20261022',
            'source': 'LOCAL_DAILY_PREFIX_NO_FETCH', 'dates': list(P_DATES),
            'daily_file_count': len(paths), 'days': days}


if __name__ == '__main__':
    result = replay()
    output = HERE/'a_combined_v1_replay_20261019_20_21.json'
    with output.open('x', encoding='utf-8') as stream:
        json.dump(result,stream,ensure_ascii=False,indent=2,allow_nan=False)
    print(json.dumps({p:{'old_a':v['old_a'],
                         'arms':{k:len(x) for k,x in v['arms'].items()}}
                      for p,v in result['days'].items()},ensure_ascii=False))
