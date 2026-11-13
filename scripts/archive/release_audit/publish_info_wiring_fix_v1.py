#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""发布后继冻结清单（本轮：4 项接线层修复，Corvus 信息接线后继）。

变更（vs v174/v162）：
  仅 4 个产品文件被本次修复改动（其余文件未动）：
    stargazer_desktop/corvus_info_link_v1.py
    stargazer_desktop/tarot_monitor_v2.py
    stargazer_desktop/assistant_workflow.py
    stargazer_desktop/i34_bridge.py
  本轮不新增文件、不改任何其它文件；运行时引用改指 v175 / v163。

按纪律：不得复用/覆盖旧清单；发布后继清单 v175 / v163（旧 v174 / v162 原样保留）。

输出（新文件）：
  stargazer_ml/v5_hybrid/risk_r0_v1/live_monitor_fingerprints_20261108_v175_flow.json.gz
  stargazer_ml/v5_hybrid/green_common_profile/tarot_live_manifest_v163_20261108_rel.json.gz
"""
import gzip
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RISK = ROOT / 'stargazer_ml/v5_hybrid/risk_r0_v1'
GCP = ROOT / 'stargazer_ml/v5_hybrid/green_common_profile'

OLD_FLAT = RISK / 'live_monitor_fingerprints_20261108_v174_flow.json.gz'
NEW_FLAT = RISK / 'live_monitor_fingerprints_20261108_v175_flow.json.gz'
OLD_TAROT = GCP / 'tarot_live_manifest_v162_20261108_rel.json.gz'
NEW_TAROT = GCP / 'tarot_live_manifest_v163_20261108_rel.json.gz'

EXPECTED_CHANGED = {
    # 4 个产品文件（本轮接线修复改动）
    'stargazer_desktop/corvus_info_link_v1.py',
    'stargazer_desktop/tarot_monitor_v2.py',
    'stargazer_desktop/assistant_workflow.py',
    'stargazer_desktop/i34_bridge.py',
    # 8 个运行时引用文件（版本字符串 v174->v175 / v162->v163 改动，属发布一部分）
    'stargazer_desktop/golden_stock_analysis.py',
    'stargazer_desktop/sector_rule_scan.py',
    'stargazer_desktop/on_demand_model_worker.py',
    'stargazer_ml/v5_hybrid/green_common_profile/market_price_evidence_v1.py',
    'stargazer_ml/v5_hybrid/green_common_profile/tarot_live_v2.py',
    'stargazer_ml/v5_hybrid/risk_r0_v1/accept_live_postclose_v1.py',
    'stargazer_ml/v5_hybrid/risk_r0_v1/live_postclose_job_v1.py',
    'stargazer_ml/v5_hybrid/risk_r0_v1/tarot_intraday_risk_v2.py',
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
    new_file.write_bytes(gzip.compress(
        json.dumps(out, indent=1, ensure_ascii=False).encode('utf-8'), mtime=0))
    return out, changed


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
    out = dict(old)
    out['frozen_inputs'] = frozen
    out['supersedes'] = old_file.name
    out['supersedes_sha256'] = sha256(old_file)
    out['version'] = 'v163_20261108_rel'
    out['release_note'] = (old.get('release_note', '') +
                           ' | v163(后继清单): 同 v162 内容 + 4 项接线层修复'
                           '（可得性完整时刻核对 / 更正版本现行性 / D2核心三链同消费 / '
                           '盘中重审键含信息版本且推送键不变）；不覆盖 v162 及更早。')
    new_file.write_bytes(gzip.compress(
        json.dumps(out, indent=1, ensure_ascii=False).encode('utf-8'), mtime=0))
    return frozen, changed


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
    print('v175 entries:', len(flat), '| changed vs v174:', len(flat_changed))
    for k in flat_changed:
        print('   ~', k)
    print('v163 frozen_inputs:', len(frozen), '| changed vs v162:', len(tarot_changed))
    for k in tarot_changed:
        print('   ~', k)
    mism, ok = verify_with_real_verifier(flat)
    tb = verify_tarot(frozen)
    print('verify v175 (joint_shadow_service_v1.verify_fingerprints):',
          'OK' if ok else 'FAIL', mism[:5])
    print('verify v163 (tarot 同口径):', 'OK' if not tb else 'FAIL', tb[:5])
    if not (ok and not tb):
        raise SystemExit('后继清单校验未通过')
    print('published:', NEW_FLAT.name, NEW_TAROT.name)
