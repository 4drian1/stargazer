# -*- coding: utf-8 -*-
"""候选包内容核验：新 UI 资源 + 4 个 ui_* 接口模块确实入包；旧 web 文件/config.yaml 不入包。"""
import os
from PyInstaller.archive.readers import CArchiveReader, ZlibArchiveReader

EXE = r'd:\stargazer\stargazer_desktop\dist_candidate\Stargazer.exe'
TMP = r'd:\stargazer\stargazer_desktop\build\_pyz_check.tmp'

ca = CArchiveReader(EXE)
ca_names = list(ca.toc.keys())

def norm(n):
    return n.replace('\\', '/')

ca_norm = [norm(n) for n in ca_names]

print('=== CArchive entries:', len(ca_names), '===')

# --- web / datas 检查 ---
required_web = [
    'web/index.html',
    'web/assets/index-PJ_Huq9D.css',
    'web/assets/index-CIOVEJrB.js',
]
must_absent = [
    'web/app.js',
    'web/pentacles.js',
    'web/local_storage.html',
    'config.yaml',
]
# config.example.yaml 是安全构建模板，spec 显式捆绑，预期存在（非私有配置）。
print('config.example.yaml present (expected template):', 'config.example.yaml' in ca_norm)
# 实际 web 资源清单（前缀匹配）
web_entries = [n for n in ca_norm if n.startswith('web/')]
print('web/ entries in CArchive:', web_entries)

for r in required_web:
    print(('OK  ' if r in ca_norm else 'MISS') + ' required web: ' + r)
for a in must_absent:
    print(('OK  absent ' if a not in ca_norm else 'FAIL present ') + a)

# --- PYZ 模块检查 ---
pyz_names = [n for n in ca_names if n.endswith('.pyz')]
print('PYZ entries:', pyz_names)
assert pyz_names, 'no PYZ found'
os.makedirs(os.path.dirname(TMP), exist_ok=True)
pyz_blob = ca.extract(pyz_names[0])
with open(TMP, 'wb') as f:
    f.write(pyz_blob)
zr = ZlibArchiveReader(TMP)
mod_names = list(zr.toc.keys())
mod_norm = [norm(m) for m in mod_names]
print('PYZ module count:', len(mod_names))

ui_mods = ['ui_backend_config', 'ui_calendar_api', 'ui_plan_versions_api', 'ui_global_overview_api']
print('--- ui_* modules in PYZ ---')
for u in ui_mods:
    hit = [m for m in mod_norm if u in m]
    print(('OK  ' if hit else 'MISS') + ' ' + u + ('  -> ' + str(hit) if hit else ''))

# 顺带确认关键业务模块仍在
key_mods = ['server', 'assistant_workflow', 'trade_plan_v1', 'tdx_daily_source_v2',
            'pentacles_review_data', 'corvus_info_link_v1', 'i34_bridge']
print('--- key business modules in PYZ ---')
for k in key_mods:
    hit = [m for m in mod_norm if m == k or m.endswith('/' + k) or ('/' + k + '.') in m or m.startswith(k + '.')]
    print(('OK  ' if hit else 'MISS') + ' ' + k)

ok = (all(r in ca_norm for r in required_web)
      and all(a not in ca_norm for a in must_absent)
      and all(any(u in m for m in mod_norm) for u in ui_mods))
print('RESULT:', 'PASS' if ok else 'FAIL')
for _p in (TMP,):
    if os.path.exists(_p):
        os.remove(_p)
