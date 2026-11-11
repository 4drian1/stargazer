# -*- coding: utf-8 -*-
"""把 v178/v166 清单中被新 UI 改写的条目恢复为“已装 EXE 54825f57”对应的生产哈希。
这是“以安装包和原始回执为准恢复匹配生产”，不是把差异洗成绿灯。
只改这 3 个被污染的键；其余键保持原值。"""
import gzip
import json
from pathlib import Path

RISK = Path(r'd:/stargazer/stargazer_ml/v5_hybrid/risk_r0_v1')
GCP = Path(r'd:/stargazer/stargazer_ml/v5_hybrid/green_common_profile')

PROD = {
    'stargazer_desktop/server.py': '13d283487db0336f27ce7887d0b7e311b5d9359d9b8b3da56df99b87f795bdc9',
    'stargazer_desktop/assistant_workflow.py': 'eb46f467afae4c9657c38d9596f89f3a7507b7ad0e7dae282a335257dd80bc02',
    'stargazer_desktop/app.py': 'e20e4745b0265cba87c1a73b9625840c6ce8a372165adec3cf3832057ca8d1a4',
}

# v178 flow
fp = RISK / 'live_monitor_fingerprints_20261109_v178_flow.json.gz'
d = json.loads(gzip.decompress(fp.read_bytes()))
for k, v in PROD.items():
    if k in d:
        print('flow v178', k, 'was', d[k][:10], '->', v[:10])
        d[k] = v
fp.write_bytes(gzip.compress(json.dumps(d, indent=1, ensure_ascii=False).encode('utf-8'), mtime=0))

# v166 tarot (frozen_inputs)
tp = GCP / 'tarot_live_manifest_v166_20261109_rel.json.gz'
t = json.loads(gzip.decompress(tp.read_bytes()))
for k, v in PROD.items():
    if k in t['frozen_inputs']:
        print('tarot v166', k, 'was', t['frozen_inputs'][k][:10], '->', v[:10])
        t['frozen_inputs'][k] = v
tp.write_bytes(gzip.compress(json.dumps(t, indent=1, ensure_ascii=False).encode('utf-8'), mtime=0))

print('manifests restored to production (EXE-matching) hashes for the 3 contaminated files.')
