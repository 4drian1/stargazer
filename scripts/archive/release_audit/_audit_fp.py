import json
from pathlib import Path

root = Path('D:/stargazer')
risk = root / 'stargazer_ml/v5_hybrid/risk_r0_v1'
live = json.loads((risk / 'live_monitor_fingerprints_20261023_v2.json').read_text(encoding='utf-8'))
dev = json.loads((risk / 'joint_shadow_fingerprints_dev_rec4.json').read_text(encoding='utf-8'))
ks, ds = set(live), set(dev)
print('live', len(live), 'dev', len(dev))
print('--- dev-only entries:')
for k in sorted(ds - ks):
    print('   ', k.replace('D:\\stargazer\\', ''), dev[k][:8])
print('--- live-only entries:')
for k in sorted(ks - ds):
    print('   ', k.replace('D:\\stargazer\\', ''))

def rel(k):
    return k.replace('D:\\stargazer\\', '').replace('\\', '/')

print('--- dev manifest files: size and existence')
tot = 0
missing = []
for k in sorted(dev):
    p = root / rel(k)
    sz = p.stat().st_size if p.exists() else -1
    tot += max(sz, 0)
    if sz < 0:
        missing.append(rel(k))
    print(f'{sz:>12}  {rel(k)}')
print('TOTAL bytes', tot)
print('MISSING:', missing)

print('--- verify live hashes still match')
import hashlib
bad = []
for k, v in live.items():
    p = root / rel(k)
    if not p.exists():
        bad.append(('MISSING', rel(k)))
        continue
    h = hashlib.sha256(p.read_bytes()).hexdigest()
    if h != v:
        bad.append(('HASH', rel(k)))
print('live mismatches:', bad)
