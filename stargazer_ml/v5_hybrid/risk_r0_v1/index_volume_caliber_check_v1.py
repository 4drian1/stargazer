"""Index lc5 volume/amount caliber cross-check (量额对拍), offline, read-only.

Prior frozen finding (TDX_INDEX_ACCEPTANCE_RESULT_20261016.md): the lc5 'volume'
unit is NOT verified (volume_verified=false); amount/raw_volume was observed ~=100
and 'volume == 手 (lots of 100 shares)' was NOT accepted. Volume rules are blocked
until the caliber is settled.

This script does NOT invent a unit. It produces the amount-vs-volume pairing
evidence for the SAME historical bars:

 1. absolute magnitude of the daily summed amount -> what unit 'amount' can be
    (yuan vs 10k-yuan), judged against known A-share daily turnover scale;
 2. implied volume-weighted average price (VWAP) under each volume-unit
    hypothesis (shares vs lots) -> which one is economically plausible;
 3. a unit-invariant cross check between the two indices (创业板 avg price should
    exceed 主板 avg price under the SAME unit);
 4. caliber CONSISTENCY across the whole history (level jumps), which is what
    actually matters for a relative/shrink-volume (缩量) comparison.

No network, no fit, no model, no rule change, no packing.
"""
import json
import pathlib
import statistics

import pyarrow.parquet as pq

MIRROR = pathlib.Path(r'D:\stargazer\stargazer_ml\v5_hybrid\risk_r0_v1\index_minutes_v1')
FILES = {'000001.SH': 'sse_5min.parquet', '399006.SZ': 'cyb_5min.parquet'}

# Plausible A-share volume-weighted average price band (yuan per share).
VWAP_LO, VWAP_HI = 3.0, 80.0


def load(symbol):
    cols = pq.read_table(MIRROR / FILES[symbol]).to_pydict()
    n = len(cols['ts'])
    bars = [{k: cols[k][i] for k in cols} for i in range(n)]
    return bars


def per_day(bars):
    days = {}
    for b in bars:
        d = days.setdefault(b['day'], {'amount': 0.0, 'volume': 0.0, 'bars': 0,
                                       'ratios': [], 'close': None})
        d['amount'] += float(b['amount'])
        d['volume'] += float(b['raw_volume'])
        d['bars'] += 1
        if b['raw_volume']:
            d['ratios'].append(float(b['amount']) / float(b['raw_volume']))
        d['close'] = b['close']
    return dict(sorted(days.items()))


def main():
    report = {'mirror': str(MIRROR), 'symbols': {}}
    for symbol in FILES:
        bars = load(symbol)
        days = per_day(bars)
        ratios = [d['amount'] / d['volume'] for d in days.values() if d['volume']]
        med = statistics.median(ratios)
        # unit-invariant cross check: within-day bar-ratio consistency
        spread = []
        for d in days.values():
            if len(d['ratios']) >= 10:
                m = statistics.mean(d['ratios'])
                if m:
                    spread.append(statistics.pstdev(d['ratios']) / m)
        # caliber consistency: day-over-day jumps of the daily ratio
        keys = list(days)
        jumps = []
        for a, b in zip(keys, keys[1:]):
            ra = days[a]['amount'] / days[a]['volume'] if days[a]['volume'] else None
            rb = days[b]['amount'] / days[b]['volume'] if days[b]['volume'] else None
            if ra and rb:
                ch = abs(rb / ra - 1.0)
                if ch >= 0.20:
                    jumps.append({'from_day': a, 'to_day': b, 'change': round(rb / ra - 1.0, 4)})
        last = keys[-1]
        report['symbols'][symbol] = {
            'bars': len(bars),
            'days': len(days),
            'first_day': keys[0],
            'last_day': last,
            'last_day_amount_sum': days[last]['amount'],
            'last_day_volume_sum': days[last]['volume'],
            'last_day_close': days[last]['close'],
            'daily_ratio_median': med,
            'daily_ratio_min': min(ratios),
            'daily_ratio_max': max(ratios),
            'vwap_if_volume_is_shares': med,
            'vwap_if_volume_is_lots': med / 100.0,
            'within_day_ratio_cv_median': statistics.median(spread) if spread else None,
            'ratio_jumps_ge_20pct': jumps[:20],
            'ratio_jumps_count': len(jumps),
            'last_10_days': [
                {'day': k, 'amount_sum': days[k]['amount'], 'volume_sum': days[k]['volume'],
                 'ratio': (days[k]['amount'] / days[k]['volume']) if days[k]['volume'] else None,
                 'bars': days[k]['bars'], 'close': days[k]['close']}
                for k in keys[-10:]
            ],
        }

    a = report['symbols'].get('000001.SH', {})
    b = report['symbols'].get('399006.SZ', {})
    if a.get('daily_ratio_median') and b.get('daily_ratio_median'):
        report['unit_invariant_cross_check'] = {
            'cyb_over_sse_ratio': b['daily_ratio_median'] / a['daily_ratio_median'],
            'note': ('same under either unit hypothesis (a constant factor cancels); '
                     '创业板 average price should exceed 主板 average price'),
        }

    out = MIRROR.parent / 'index_volume_caliber_check_v1.json'
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')

    for symbol, r in report['symbols'].items():
        print('=' * 70)
        print(symbol, 'bars', r['bars'], 'days', r['days'], r['first_day'], '->', r['last_day'])
        print('  last day: amount_sum=%.6g volume_sum=%.6g close=%.4f'
              % (r['last_day_amount_sum'], r['last_day_volume_sum'], r['last_day_close']))
        print('  daily ratio (amount/volume): median=%.6g min=%.6g max=%.6g'
              % (r['daily_ratio_median'], r['daily_ratio_min'], r['daily_ratio_max']))
        print('  implied VWAP if volume is SHARES: %.6g' % r['vwap_if_volume_is_shares'])
        print('  implied VWAP if volume is LOTS(100sh): %.6g' % r['vwap_if_volume_is_lots'])
        print('  within-day bar ratio CV median: %s' % r['within_day_ratio_cv_median'])
        print('  day-over-day ratio jumps >=20%%: %d' % r['ratio_jumps_count'])
        for j in r['ratio_jumps_ge_20pct'][:10]:
            print('    ', j['from_day'], '->', j['to_day'], j['change'])
    print('=' * 70)
    print('cross check:', report.get('unit_invariant_cross_check'))
    print('plausible VWAP band used:', VWAP_LO, '-', VWAP_HI, 'yuan/share')
    print('written:', out)


if __name__ == '__main__':
    main()
