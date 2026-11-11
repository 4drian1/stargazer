#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""发布后继冻结清单（2026-11-10）：强势池自动准备链（FIX_HANDOFF 五点落盘）。

变更（vs v183/v171）：
  新增 stargazer_desktop/strong_sector_pool_prep_v1.py（预约生命周期+显式准备入口）；
  strong_sector_watch_v1.py（schedule 池块接入 prep 入口，旧 failed 永久跳过修复）、
  server.py（GET/POST /api/strong-pool/* + 板块确认自动提交准备）、
  postclose_prepare_v1.py（after_selection 同一入口）、assistant_workflow.py（confirm_plan 挂钩）、
  web/index.html + web/pentacles.js（"监控板块强势股"区 + "更新板块强势池"按钮）、
  Stargazer_demo_safe.spec（hiddenimports 增列）；
  golden_stock_analysis / on_demand_model_worker / sector_rule_scan / market_price_evidence_v1 /
  accept_live_postclose_v1 / live_postclose_job_v1 / tarot_intraday_risk_v2 引用改指 v184，
  tarot_live_v2 引用改指 v172。
按纪律：不覆盖旧清单；tarot 同口径发布后继；今日已 READY 10 板块池复用不重算。

输出（新文件，旧清单原样保留）：
  stargazer_ml/v5_hybrid/risk_r0_v1/live_monitor_fingerprints_20261110_v184_flow.json.gz
  stargazer_ml/v5_hybrid/green_common_profile/tarot_live_manifest_v172_20261110_rel.json.gz
"""
import gzip
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RISK = ROOT / 'stargazer_ml/v5_hybrid/risk_r0_v1'
GCP = ROOT / 'stargazer_ml/v5_hybrid/green_common_profile'

OLD_FLAT = RISK / 'live_monitor_fingerprints_20261109_v183_flow.json.gz'
NEW_FLAT = RISK / 'live_monitor_fingerprints_20261110_v184_flow.json.gz'
OLD_TAROT = GCP / 'tarot_live_manifest_v171_20261109_rel.json.gz'
NEW_TAROT = GCP / 'tarot_live_manifest_v172_20261110_rel.json.gz'

NEW_KEY = 'stargazer_desktop/strong_sector_pool_prep_v1.py'

EXPECTED_CHANGED = {
    NEW_KEY,
    'stargazer_desktop/strong_sector_watch_v1.py',
    'stargazer_desktop/server.py',
    'stargazer_desktop/postclose_prepare_v1.py',
    'stargazer_desktop/assistant_workflow.py',
    'stargazer_desktop/web/index.html',
    'stargazer_desktop/web/pentacles.js',
    'stargazer_desktop/Stargazer_demo_safe.spec',
    'stargazer_desktop/golden_stock_analysis.py',
    'stargazer_desktop/on_demand_model_worker.py',
    'stargazer_desktop/sector_rule_scan.py',
    'stargazer_ml/v5_hybrid/green_common_profile/market_price_evidence_v1.py',
    'stargazer_ml/v5_hybrid/green_common_profile/tarot_live_v2.py',
    'stargazer_ml/v5_hybrid/risk_r0_v1/accept_live_postclose_v1.py',
    'stargazer_ml/v5_hybrid/risk_r0_v1/live_postclose_job_v1.py',
    'stargazer_ml/v5_hybrid/risk_r0_v1/tarot_intraday_risk_v2.py',
}

# 关键文件必须被 flow 清单绑定；tarot 清单沿用其既有绑定口径
# （v171 绑定 server.py/plan_board_change/ths_board_identity/assistant_workflow/pentacles.js，
#   不绑定 strong_sector_watch_v1.py），新增模块随 server.py 一并入 tarot 清单。
MUST_BIND_FLOW = {
    'stargazer_desktop/strong_sector_pool_prep_v1.py',
    'stargazer_desktop/strong_sector_watch_v1.py',
    'stargazer_desktop/server.py',
}
MUST_BIND_TAROT = {
    'stargazer_desktop/strong_sector_pool_prep_v1.py',
    'stargazer_desktop/server.py',
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
    assert NEW_KEY not in old, '新文件不应已存在于旧清单'
    out, missing, changed = {}, [], []
    for k, prev in old.items():
        p = ROOT / k
        if not p.exists():
            missing.append(k)
            continue
        cur = sha256(p)
        if cur != prev:
            changed.append(k)
        out[k] = cur
    assert not missing, f'{old_file.name}: 缺文件 {missing[:5]}'
    unexpected = {c.replace('\\', '/') for c in changed} - EXPECTED_CHANGED
    assert not unexpected, f'意外变更（超出本轮范围）: {unexpected}'
    out[NEW_KEY] = sha256(ROOT / NEW_KEY)
    new_file.write_bytes(gzip.compress(
        json.dumps(out, indent=1, ensure_ascii=False).encode('utf-8'), mtime=0))
    return out, changed + [NEW_KEY]


def build_tarot(old_file: Path, new_file: Path):
    old = json.loads(gzip.decompress(old_file.read_bytes()))
    assert not new_file.exists(), f'{new_file.name} 已存在，禁止覆盖'
    frozen, missing, changed = {}, [], []
    for k, prev in old['frozen_inputs'].items():
        p = ROOT / k
        if not p.exists():
            missing.append(k)
            continue
        cur = sha256(p)
        if cur != prev:
            changed.append(k)
        frozen[k] = cur
    assert not missing, f'{old_file.name}: 缺文件 {missing[:5]}'
    unexpected = {c.replace('\\', '/') for c in changed} - EXPECTED_CHANGED
    assert not unexpected, f'意外变更（超出本轮范围）: {unexpected}'
    frozen[NEW_KEY] = sha256(ROOT / NEW_KEY)
    out = dict(old)
    out['frozen_inputs'] = frozen
    out['supersedes'] = old_file.name
    out['supersedes_sha256'] = sha256(old_file)
    out['version'] = 'v172_20261110_rel'
    out['release_note'] = (old.get('release_note', '') +
                           ' | v172(后继清单): 强势池自动准备链（板块确认/选股后/计划确认/手动按钮'
                           '同一 prep 入口；预约生命周期落盘可恢复，同版READY复用不重算，'
                           '0只/失败/缺源/日线缺失四分，字节SHA回读；不覆盖 v171，禁止沿用旧清单')
    new_file.write_bytes(gzip.compress(
        json.dumps(out, indent=1, ensure_ascii=False).encode('utf-8'), mtime=0))
    return frozen, changed + [NEW_KEY]


def verify_with_real_verifier(flat: dict):
    sys.path.insert(0, str(RISK))
    import joint_shadow_service_v1 as svc
    mism, ok = svc.verify_fingerprints(flat)
    return mism, ok


def verify_tarot(frozen: dict):
    return [k for k, v in frozen.items() if sha256(ROOT / k) != v]


if __name__ == '__main__':
    flat, flat_changed = build_flat(OLD_FLAT, NEW_FLAT)
    frozen, tarot_changed = build_tarot(OLD_TAROT, NEW_TAROT)
    for k in MUST_BIND_FLOW:
        assert k in flat, f'flow 清单缺关键绑定: {k}'
    for k in MUST_BIND_TAROT:
        assert k in frozen, f'tarot 清单缺关键绑定: {k}'
    print('v184 entries:', len(flat), '| changed vs v183:', len(flat_changed))
    for k in flat_changed:
        print('   ~', k)
    print('v172 frozen_inputs:', len(frozen), '| changed vs v171:', len(tarot_changed))
    for k in tarot_changed:
        print('   ~', k)
    mism, ok = verify_with_real_verifier(flat)
    tb = verify_tarot(frozen)
    print('verify v184 (joint_shadow_service_v1.verify_fingerprints):',
          'OK' if ok else 'FAIL', mism[:5])
    print('verify v172 (tarot 同口径):', 'OK' if not tb else 'FAIL', tb[:5])
    if not (ok and not tb):
        raise SystemExit('后继清单校验未通过')
    print('published:', NEW_FLAT.name, NEW_TAROT.name)
