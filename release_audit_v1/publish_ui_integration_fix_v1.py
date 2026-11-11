#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""发布后继冻结清单（本轮：新 UI 接入 → 替换 web/ + server.py/assistant_workflow.py 改 + 8 引用）。

变更 vs v176/v164：
  产品/运行文件改动（进入 EXPECTED_CHANGED）：
    stargazer_desktop/server.py                     （新增 3 个合同接口路由 + 执行状态富集）
    stargazer_desktop/assistant_workflow.py          （state() 回读持久化确认状态）
    stargazer_desktop/web/index.html                 （替换为新 React 前端入口）
    8 个运行时引用文件（版本串 v176->v177 / v164->v165 改动）
  已删除文件（从新清单摘除，不再追踪）：
    stargazer_desktop/web/app.js / web/pentacles.js / web/local_storage.html
  新增文件（ui_backend_config.py / ui_calendar_api.py / ui_plan_versions_api.py /
          ui_global_overview_api.py + web/assets/*）不属风险指纹清单范畴，不纳入，
  由候选包提取核验代替（见发布后验收）。

纪律：不得复用/覆盖旧清单；发布后继清单 v177 / v165（旧 v176 / v164 原样保留）。
"""
import gzip
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RISK = ROOT / 'stargazer_ml/v5_hybrid/risk_r0_v1'
GCP = ROOT / 'stargazer_ml/v5_hybrid/green_common_profile'

OLD_FLAT = RISK / 'live_monitor_fingerprints_20261108_v176_flow.json.gz'
NEW_FLAT = RISK / 'live_monitor_fingerprints_20261108_v177_flow.json.gz'
OLD_TAROT = GCP / 'tarot_live_manifest_v164_20261108_rel.json.gz'
NEW_TAROT = GCP / 'tarot_live_manifest_v165_20261108_rel.json.gz'

EXPECTED_CHANGED = {
    'stargazer_desktop/server.py',
    'stargazer_desktop/assistant_workflow.py',
    'stargazer_desktop/web/index.html',
    'stargazer_desktop/Stargazer_demo_safe.spec',
    'stargazer_desktop/golden_stock_analysis.py',
    'stargazer_desktop/sector_rule_scan.py',
    'stargazer_desktop/on_demand_model_worker.py',
    'stargazer_ml/v5_hybrid/green_common_profile/market_price_evidence_v1.py',
    'stargazer_ml/v5_hybrid/green_common_profile/tarot_live_v2.py',
    'stargazer_ml/v5_hybrid/risk_r0_v1/tarot_intraday_risk_v2.py',
    'stargazer_ml/v5_hybrid/risk_r0_v1/accept_live_postclose_v1.py',
    'stargazer_ml/v5_hybrid/risk_r0_v1/live_postclose_job_v1.py',
}


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def build_flat(old_file: Path, new_file: Path):
    old = json.loads(gzip.decompress(old_file.read_bytes()))
    assert not new_file.exists(), f'{new_file.name} 已存在，禁止覆盖'
    out, dropped, changed = {}, [], []
    for k, prev in old.items():
        p = ROOT / k
        if not p.exists():
            dropped.append(k)
            continue
        cur = sha256(p)
        if cur != prev:
            changed.append(k)
        out[k] = cur
    print('flat dropped (missing on disk):', dropped)
    unexpected = {c.replace('\\', '/') for c in changed} - EXPECTED_CHANGED
    assert not unexpected, f'意外变更（超出本轮范围）: {unexpected}'
    new_file.write_bytes(gzip.compress(
        json.dumps(out, indent=1, ensure_ascii=False).encode('utf-8'), mtime=0))
    return out, changed, dropped


def build_tarot(old_file: Path, new_file: Path):
    old = json.loads(gzip.decompress(old_file.read_bytes()))
    assert not new_file.exists(), f'{new_file.name} 已存在，禁止覆盖'
    frozen, dropped, changed = {}, [], []
    for k, prev in old['frozen_inputs'].items():
        p = ROOT / k
        if not p.exists():
            dropped.append(k)
            continue
        cur = sha256(p)
        if cur != prev:
            changed.append(k)
        frozen[k] = cur
    print('tarot dropped (missing on disk):', dropped)
    unexpected = {c.replace('\\', '/') for c in changed} - EXPECTED_CHANGED
    assert not unexpected, f'意外变更（超出本轮范围）: {unexpected}'
    out = dict(old)
    out['frozen_inputs'] = frozen
    out['supersedes'] = old_file.name
    out['supersedes_sha256'] = sha256(old_file)
    out['version'] = 'v165_20261108_rel'
    out['release_note'] = (old.get('release_note', '') +
                           ' | v165(后继清单): 同 v164 内容 + 新UI接入(server.py/'
                           'assistant_workflow.py/web 替换/index.html, 8 引用改指 v177/v165)；'
                           '不覆盖 v164 及更早。')
    new_file.write_bytes(gzip.compress(
        json.dumps(out, indent=1, ensure_ascii=False).encode('utf-8'), mtime=0))
    return frozen, changed, dropped


def verify_with_real_verifier(flat: dict):
    sys.path.insert(0, str(RISK))
    import joint_shadow_service_v1 as svc
    mism, ok = svc.verify_fingerprints(flat)
    return mism, ok


def verify_tarot(frozen: dict):
    return [k for k, v in frozen.items() if sha256(ROOT / k) != v]


if __name__ == '__main__':
    flat, flat_changed, flat_dropped = build_flat(OLD_FLAT, NEW_FLAT)
    frozen, tarot_changed, tarot_dropped = build_tarot(OLD_TAROT, NEW_TAROT)
    print('v177 entries:', len(flat), '| changed vs v176:', len(flat_changed),
          '| dropped:', len(flat_dropped))
    for k in flat_changed:
        print('   ~', k)
    print('v165 frozen_inputs:', len(frozen), '| changed vs v164:', len(tarot_changed),
          '| dropped:', len(tarot_dropped))
    for k in tarot_changed:
        print('   ~', k)
    mism, ok = verify_with_real_verifier(flat)
    tb = verify_tarot(frozen)
    print('verify v177 (joint_shadow_service_v1.verify_fingerprints):',
          'OK' if ok else 'FAIL', mism[:5])
    print('verify v165 (tarot 同口径):', 'OK' if not tb else 'FAIL', tb[:5])
    if not (ok and not tb):
        raise SystemExit('后继清单校验未通过')
    print('published:', NEW_FLAT.name, NEW_TAROT.name)
