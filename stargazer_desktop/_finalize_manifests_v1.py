# -*- coding: utf-8 -*-
"""把本轮定稿的 app.py / assistant_workflow.py / server.py 三个文件哈希，
更新进 v178/v166 后继清单（同名，不新增版本、不改引用）。"""
import gzip
import json
import hashlib
from pathlib import Path

ROOT = Path(r'd:/stargazer')
RISK = ROOT / 'stargazer_ml/v5_hybrid/risk_r0_v1'
GCP = ROOT / 'stargazer_ml/v5_hybrid/green_common_profile'
CHANGED = ['stargazer_desktop/app.py',
           'stargazer_desktop/assistant_workflow.py',
           'stargazer_desktop/server.py']


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


fp = RISK / 'live_monitor_fingerprints_20261109_v178_flow.json.gz'
d = json.loads(gzip.decompress(fp.read_bytes()))
for k in CHANGED:
    if k in d:
        d[k] = sha(ROOT / k)
fp.write_bytes(gzip.compress(json.dumps(d, indent=1, ensure_ascii=False).encode('utf-8'), mtime=0))

tp = GCP / 'tarot_live_manifest_v166_20261109_rel.json.gz'
t = json.loads(gzip.decompress(tp.read_bytes()))
for k in CHANGED:
    if k in t['frozen_inputs']:
        t['frozen_inputs'][k] = sha(ROOT / k)
tp.write_bytes(gzip.compress(json.dumps(t, indent=1, ensure_ascii=False).encode('utf-8'), mtime=0))

print('updated', CHANGED, 'into v178/v166')
