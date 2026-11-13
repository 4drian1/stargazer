"""对照 v158 发布清单与当前工作区实际 SHA256，列出全部失配项。

只读：不写清单，不改任何源文件。
"""
import gzip
import hashlib
import json
from pathlib import Path

ROOT = Path(r'D:\stargazer')
REF = ROOT / 'stargazer_ml/v5_hybrid/risk_r0_v1/live_monitor_fingerprints_20261105_v158_flow.json.gz'

expected = json.loads(gzip.open(REF, 'rt', encoding='utf8').read())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


missing, mismatch, ok = [], [], []
for rel, want in sorted(expected.items()):
    path = ROOT / rel
    if not path.is_file():
        missing.append(rel)
        continue
    got = sha(path)
    (ok if got == want else mismatch).append((rel, want, got))

print('manifest entries :', len(expected))
print('match            :', len(ok))
print('missing on disk  :', len(missing))
print('MISMATCH         :', len(mismatch))
print()
if missing:
    print('--- missing ---')
    for rel in missing[:40]:
        print('  ', rel)
    print()
if mismatch:
    print('--- mismatch ---')
    for rel, want, got in mismatch:
        print('  %-60s want=%s got=%s' % (rel, want[:12], got[:12]))
