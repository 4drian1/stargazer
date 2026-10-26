"""生成 #18.46 精确补提文件清单：三份相对指纹清单的并集 + 运行链补充文件。"""
import json
import gzip
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RISK = ROOT / 'stargazer_ml/v5_hybrid/risk_r0_v1'
GCP = ROOT / 'stargazer_ml/v5_hybrid/green_common_profile'

files = set()
for name in ('live_monitor_fingerprints_20261026_v27_rel.json',
             'joint_shadow_fingerprints_rel_v1.json'):
    for k in json.loads((RISK / name).read_text(encoding='utf-8')):
        files.add(k)
for k in json.loads(gzip.decompress(
        (GCP / 'tarot_live_manifest_v18_20261026_rel.json.gz').read_bytes()))['frozen_inputs']:
    files.add(k)

# 运行链补充（spec hiddenimports / 监控机指数源 / 回归测试），不在冻结清单内但属发布代码
EXTRA = [
    'stargazer_desktop/public_index_feed_v1.py',      # 监控机公开指数分钟源
    'stargazer_desktop/risk_outlook_v1.py',           # spec hiddenimport
    'stargazer_desktop/tarot_event_v1.py',            # spec hiddenimport
    'stargazer_desktop/test_ui_framework.py',
    'stargazer_desktop/test_shadow_ui.py',
    'stargazer_desktop/test_shadow_journal.py',
    'stargazer_desktop/test_joint_risk_bridge.py',
    'stargazer_desktop/test_oracles_v1.py',
    'stargazer_desktop/test_decision_layer.py',
    'stargazer_desktop/test_conversation_intent.py',
    'stargazer_desktop/test_golden_rules.py',
    'stargazer_desktop/test_tarot_monitor_v2.py',
    'stargazer_desktop/test_postclose_risk_runtime.py',
    'stargazer_desktop/test_public_index_feed_v1.py',
    'stargazer_desktop/test_tarot_selector_binding_v1.py',
    'stargazer_desktop/test_index_direct_source.py',
]
for rel in EXTRA:
    if (ROOT / rel).exists():
        files.add(rel)
    else:
        print('MISSING extra:', rel)

missing = [f for f in sorted(files) if not (ROOT / f).exists()]
print('MISSING from disk:', missing)
out = ROOT / 'release_audit_v1/add_files_1846.txt'
out.write_text('\n'.join(sorted(files)) + '\n', encoding='utf-8')
print('total:', len(files), '->', out)
