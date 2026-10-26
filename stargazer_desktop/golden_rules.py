"""Frozen rule wiring, separate from fitted weights and the legacy price zones."""
import copy
import hashlib
import json
import math

VERSION = 'TAROT_GOLDEN_RULES_1_0'
BAND_ATR = .2
CYCLE_RATIOS = (.236, .382, .5, .618, .786)


def cycle_anchors(daily):
    """Seven-day initialization is not a rolling re-draw. Close updates only."""
    dates = sorted(daily)
    hi = lo = None
    for i, day in enumerate(dates):
        c = daily[day]['close']
        if i >= 6 and hi is None:
            hi = max(daily[d]['high'] for d in dates[:7])
            lo = min(daily[d]['low'] for d in dates[:7])
        if hi is None:
            continue
        if i >= 3:
            start = daily[dates[i-3]]
            closes = [daily[d]['close'] for d in dates[i-3:i+1]]
            if (start['close'] > start['open'] and all(closes[j+1] < closes[j] for j in range(3))
                    and any((closes[j]-closes[j+1])/closes[j] >= .01-1e-12 for j in range(3))):
                hi = min(hi, start['close'])
        if i >= 7:
            closes = [daily[d]['close'] for d in dates[i-7:i+1]]
            if all(closes[j+1] > closes[j] for j in range(7)):
                lo = max(lo, closes[0])
        if c > hi:
            hi = c
        if c < lo:
            lo = c
    if hi is None or lo is None or hi <= lo:
        raise ValueError('CYCLE_ANCHORS_NOT_READY')
    return dict(high=hi, low=lo, source='CLOSE_ONLY_7DAY_INIT_3DOWN_7UP_V1')


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    allow_nan=False).encode()).hexdigest()


def build_rule(close, high, low, atr, anchor_hi, anchor_lo):
    values = (close, high, low, atr, anchor_hi, anchor_lo)
    if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v <= 0 for v in values):
        raise ValueError('GOLDEN_INPUT_INVALID')
    if low > close or close > high or anchor_lo >= anchor_hi:
        raise ValueError('GOLDEN_RANGE_INVALID')
    levels = [{'ratio': r, 'price': anchor_hi-r*(anchor_hi-anchor_lo)} for r in CYCLE_RATIOS]
    nearest = min(levels, key=lambda r: (abs(close-r['price']), r['ratio']))
    distance = abs(close-nearest['price']) / atr
    f = round(high-.618*(high-low), 4)  # Same precision as the frozen research trigger.
    return dict(version=VERSION, band_atr=BAND_ATR, atr_ref=atr, f618=f,
                anchor_hi=anchor_hi, anchor_lo=anchor_lo, cycle_levels=levels,
                nearest_cycle=nearest, cycle_distance_atr=distance,
                eligible=distance <= BAND_ATR+1e-12,
                condition='DOWNSIDE_AND_STOP_AND_ACTIVE_RSI_DIVERGENCE_COMPLETED_5M',
                scope='研究规则观察，非买入或收益保证；执行只看前日0.618，未知不通过')


def project(parent, rules):
    """Keep the complete disposition, but only admitted rows enter the new list."""
    value = copy.deepcopy(parent)
    admitted, excluded = [], []
    for old in parent['rows']:
        row = copy.deepcopy(old)
        rule = rules.get(old['code'])
        if not rule or not rule.get('eligible'):
            excluded.append(dict(code=old['code'], reason='OUTSIDE_CYCLE_GOLDEN_BAND' if rule else 'GOLDEN_INPUT_UNAVAILABLE',
                                 entry_rule=rule))
            continue
        if rule['f618'] > old['reference_price']:
            excluded.append(dict(code=old['code'], reason='PREVIOUS_F618_ABOVE_CLOSE', entry_rule=rule))
            continue
        row.update(legacy_zone=row.get('zone'), entry_rule=rule,
                   zone=dict(zone_lower=rule['f618'], zone_upper=min(old['reference_price'], rule['f618']+BAND_ATR*rule['atr_ref']),
                             zone_sources=['PREVIOUS_DAY_F618']),
                   atr_ref=rule['atr_ref'], distance_atr=(row['reference_price']-rule['f618'])/rule['atr_ref'],
                   status='GOLDEN_PLAN_READY', signal='待次日完成5分K线核验下杀、止跌、背离；非报价触价即买入')
        admitted.append(row)
    admitted.sort(key=lambda r: (r['entry_rule']['cycle_distance_atr'], r['code']))
    for rank, row in enumerate(admitted, 1):
        row['rank'] = rank
    value.update(rows=admitted, rule_version=VERSION, projection_parent_hash=digest(parent),
                 common_candidate_count=len(parent['rows']), excluded_rows=excluded,
                 ranking='原共性候选→距周期黄金节点≤0.2ATR；次日只观察F618下杀止跌背离',
                 model_status='冻结评分不重训；新规则观察，不自动买卖')
    return value


def is_projection_of(value, parent):
    """Risk reuse: no new names, dates, prices or source provenance may sneak in."""
    if value.get('rule_version') != VERSION or value.get('projection_parent_hash') != digest(parent):
        return False
    for key in ('data_date', 'monitor_date', 'inputs'):
        if value.get(key) != parent.get(key):
            return False
    original = {r['code']: r for r in parent['rows']}
    rows = value.get('rows', [])
    if not rows or len({r['code'] for r in rows}) != len(rows):
        return False
    fields = ('code', 'name', 'setup', 'date', 'reference_price', 'rsi14')
    return all(r['code'] in original and all(r.get(k)==original[r['code']].get(k) for k in fields) for r in rows)
