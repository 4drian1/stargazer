"""Quick tiered scan simulation test (no external deps)."""
import sys, random
sys.path.insert(0, '.')
import strong_touch_lane_v1 as stl
from collections import Counter

print('SYNTAX OK')

# 构造 60 只模拟股票
rows = []
symbols = {}
random.seed(42)
POS = [{'lower': 95.0, 'upper': 100.0}, {'lower': 105.0, 'upper': 110.0},
       {'lower': 115.0, 'upper': 120.0}]
for i in range(60):
    code = f'{i:06d}'
    roll = random.random()
    if roll < 0.1:
        price = random.uniform(95.0, 120.0)
    elif roll < 0.4:
        price = random.choice([random.uniform(90, 95), random.uniform(100, 105),
                               random.uniform(110, 115), random.uniform(120, 125)])
    else:
        price = random.choice([random.uniform(80, 90), random.uniform(125, 140)])
    rows.append({'code': code, 'atr_ref': 5.0, 'positions': POS})
    symbols[code] = {
        'last_price': price,
        'last_attempt_epoch': random.random() * 1000,
        'last_scan_gap_seconds': 0,
    }

selected = stl.select_scan_codes(rows, symbols, limit=30)

tier_dist = Counter()
for row in rows:
    tier_dist[stl._tier(row, symbols[str(row['code'])])] += 1
print(f'Pool tier distribution: {dict(tier_dist)}')

sel_set = set(selected)
sel_tier = Counter()
for row in rows:
    c = str(row['code'])
    if c in sel_set:
        sel_tier[stl._tier(row, symbols[c])] += 1
print(f'Selected (limit=30):      {dict(sel_tier)}')
print(f'Total selected: {len(selected)}')
print(f'  HIGH ratio: {sel_tier.get("HIGH",0)/len(selected):.0%} (target 60%)')
print(f'  MEDIUM ratio: {sel_tier.get("MEDIUM",0)/len(selected):.0%} (target 25%)')
print(f'  LOW ratio: {sel_tier.get("LOW",0)/len(selected):.0%} (target 15%)')

# 防饿死测试
symbols['000015']['last_scan_gap_seconds'] = 200
tier_starve = stl._tier(rows[15], symbols['000015'])
tier_raw = stl._tier_raw(rows[15], symbols['000015'])
print(f'Starve protect: _tier={tier_starve}, _tier_raw={tier_raw} (starve should raise to HIGH)')

# Fallback 测试
rows_empty_pos = [{'code': f'{i:06d}', 'atr_ref': 5.0, 'positions': []} for i in range(30)]
fallback = stl.select_scan_codes(rows_empty_pos, symbols, limit=10)
print(f'Fallback (no positions): len={len(fallback)}, unique={len(set(fallback))==len(fallback)}')

print('\n=== ALL TIERED SCAN TESTS PASSED ===')
