# -*- coding: utf-8 -*-
"""把本轮新 UI 成果（当前磁盘版本）完整保存到独立副本，不影响生产共享源码树。"""
import hashlib
import zipfile
from pathlib import Path

DESK = Path(r'd:/stargazer/stargazer_desktop')
OUT = Path(r'd:/stargazer/ui_next/artifacts/newui_sources_independent_v1.zip')

files = [
    'server.py', 'assistant_workflow.py', 'app.py', 'ui_backend_config.py',
    'ui_calendar_api.py', 'ui_plan_versions_api.py', 'ui_global_overview_api.py',
    'acceptance_ui_integration_v1.py',
    'UI_INTEGRATION_RELEASE_REPORT_v1.md',
    'web/index.html', 'web/assets/index-CIOVEJrB.js', 'web/assets/index-PJ_Huq9D.css',
]

OUT.parent.mkdir(parents=True, exist_ok=True)
with zipfile.ZipFile(OUT, 'w', zipfile.ZIP_DEFLATED) as z:
    for f in files:
        p = DESK / f
        if p.exists():
            data = p.read_bytes()
            z.writestr(f, data)
            print(f'{f:42s} {hashlib.sha256(data).hexdigest()[:12]}')
        else:
            print(f'{f:42s} MISSING')
print('saved ->', OUT)
