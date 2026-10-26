"""Read-only P-close trial of the 2026-10-22 candidate-stage contract."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.dataset as ds


def down_days(g):
    down = g.close.to_numpy(dtype=float) < g.pre_close.to_numpy(dtype=float)
    count = 0
    for item in down[::-1]:
        if not item:
            break
        count += 1
    return count


def mixed_days(g):
    change = g.close.to_numpy(dtype=float) / g.pre_close.to_numpy(dtype=float) - 1
    count = 0
    for item in change[::-1]:
        if not (item < 0 or abs(item) <= .005):
            break
        count += 1
    return count


def short_trial(g):
    n, age = down_days(g), mixed_days(g)
    volume = g.volume.to_numpy(dtype=float)
    shrinking = bool(n and all(volume[i] < volume[i-1]
                               for i in range(len(g)-n, len(g))))
    return dict(eligible=3 <= n <= 5 and age <= 5 and shrinking,
                strict_down_days=n, mixed_decline_age=age,
                daily_volume_shrinking=shrinking)


def long_base_trial(g):
    if len(g) < 80:
        return dict(eligible=False, reason='WARMUP_LT_80')
    c = g.close.astype(float).reset_index(drop=True)
    v = g.volume.astype(float).reset_index(drop=True)
    prior = g.pre_close.astype(float).reset_index(drop=True)
    base = g.tail(5)
    before = g.iloc[-12:-5]
    prior_down7 = bool(len(before) == 7 and (before.close < before.pre_close).all())
    tr = pd.concat([(g.high-g.low).astype(float).reset_index(drop=True),
                    (g.high.astype(float).reset_index(drop=True)-prior).abs(),
                    (g.low.astype(float).reset_index(drop=True)-prior).abs()], axis=1).max(axis=1)
    atr = float(tr.rolling(14).mean().tail(5).median())
    width = float(base.close.max()-base.close.min())
    no_new_low = bool((base.low.iloc[1:] >= base.low.iloc[0]-1e-10).all())
    median3 = float(v.tail(3).median())
    old_median = float(v.iloc[-63:-3].median())
    ratio = median3 / old_median if old_median > 0 else float('nan')
    mad = float((v.tail(3)-median3).abs().median()) / median3 if median3 > 0 else float('nan')
    last = g.iloc[-1]
    outer = bool(last.is_st == 0 and last.float_mv > 0 and last.amount >= 5e7 and last.volume > 0)
    ret20 = float(c.iloc[-1]/c.iloc[-21]-1)
    ret60 = float(c.iloc[-1]/c.iloc[-61]-1)
    narrow = bool(np.isfinite(atr) and atr > 0 and width <= atr+1e-10)
    passed = bool(prior_down7 and narrow and no_new_low and outer
                  and ret60 <= -.10 and ret20 <= 0 and ratio <= .70 and mad <= .30)
    return dict(eligible=passed, prior_strict_down7=prior_down7,
                close_width_atr=round(width/atr, 4) if atr > 0 else None,
                no_new_low=no_new_low, ret20=round(ret20, 4), ret60=round(ret60, 4),
                volume_level_ratio=round(ratio, 4), volume_stability=round(mad, 4),
                outer=outer)


def run(parent_path, golden_path, daily_dir, p):
    parent = json.loads(parent_path.read_text(encoding='utf-8'))
    golden = json.loads(golden_path.read_text(encoding='utf-8'))
    if parent['data_date'] != p or golden['data_date'] != p:
        raise ValueError('SNAPSHOT_DATE_MISMATCH')
    paths = sorted(daily_dir.glob('part_*.parquet'))
    if not paths:
        raise ValueError('DAILY_SOURCE_MISSING')
    columns = ['date','code','open','high','low','close','pre_close','volume',
               'amount','is_st','float_mv']
    table = ds.dataset([str(x) for x in paths], format='parquet').to_table(
        columns=columns, filter=(ds.field('date') >= 20260101) & (ds.field('date') <= p))
    data = table.to_pandas()
    data['code'] = data.code.astype(str).str.zfill(6)
    data = data[data.code.str.match(r'^(60|68|00|30)\d{4}$')]
    data = data.drop_duplicates(['code','date'], keep='last').sort_values(['code','date'])
    original = {r['code']: r for r in parent['rows']}
    accepted_golden = {r['code']: r for r in golden['rows']}
    a, b, c, audit = [], [], [], []
    for code, g in data.groupby('code', sort=False):
        if int(g.date.iloc[-1]) != p:
            continue
        g = g.reset_index(drop=True)
        setup = None
        if code in original:
            old = original[code]
            if abs(float(g.close.iloc[-1])-float(old['reference_price'])) > 1e-8:
                raise ValueError('P_CLOSE_CHANGED:'+code)
            setup = old['setup']
            if old['setup'] == 'A_SDP':
                a.append(code)
        if setup != 'A_SDP':
            result = short_trial(g)
            last = g.iloc[-1]
            outer = bool(last.is_st == 0 and last.float_mv > 0
                         and last.amount >= 5e7 and last.volume > 0)
            if result['eligible'] or setup == 'B_LVB':
                audit.append(dict(code=code, branch='B_SHORT', outer=outer, **result))
            if result['eligible'] and outer:
                b.append(code)
        result = long_base_trial(g)
        if result['eligible']:
            c.append(code)
            audit.append(dict(code=code, branch='B_BASE_LONG', **result))
    def describe(code, branch):
        row = accepted_golden.get(code)
        return dict(code=code, branch=branch,
                    old_common_member=code in original,
                    old_golden_eligible=row is not None,
                    old_model_score_applicable=code in original and branch != 'B_BASE_LONG',
                    old_model_score_bound=code in accepted_golden,
                    reference_price=(row or original.get(code) or {}).get('reference_price'),
                    f618=(row or {}).get('entry_rule', {}).get('f618'),
                    f618_below_close=bool(row and row['reference_price'] < row['entry_rule']['f618']))
    selected = [describe(code, branch) for branch, codes in
                [('A_SDP_UNCHANGED', a), ('B_SHORT', b), ('B_BASE_LONG', c)] for code in sorted(codes)]
    return dict(version='CANDIDATE_STAGE_V2_TRIAL_20261022', p=p,
                source='LOCAL_PARQUET_NO_FETCH', daily_file_count=len(paths),
                parent_sha256=hashlib.sha256(parent_path.read_bytes()).hexdigest(),
                golden_sha256=hashlib.sha256(golden_path.read_bytes()).hexdigest(),
                old_common_count=len(original), a_count=len(a), b_short_count=len(b),
                b_base_long_count=len(c), selected=selected, b_short_audit=audit,
                scoring_note='Only old common members are within the frozen head domain; new branch membership is not model trained.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--parent', type=Path, required=True)
    parser.add_argument('--golden', type=Path, required=True)
    parser.add_argument('--daily-dir', type=Path, required=True)
    parser.add_argument('--p', type=int, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    result = run(args.parent, args.golden, args.daily_dir, args.p)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
    print(json.dumps({k: result[k] for k in ('p','old_common_count','a_count','b_short_count','b_base_long_count')},
                     ensure_ascii=False))
