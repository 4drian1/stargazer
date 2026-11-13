#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""发布后继冻结清单（2026-11-08）：风险证据双入口修复 + I34 GUI 导入入口。

背景：plan_evidence.py（计划投影绑定回退）、postclose_risk_runtime.py（冻结原始
快照绑定 + 逐股投影校验）、live_plan_lane_v1.py（盘中风险通道同一回退）、
i34_bridge.py（/i34 面板页导入表单 + D2 状态显示）变更；golden_stock_analysis /
on_demand_model_worker / sector_rule_scan 三处运行时引用改指 v172。
按纪律：不得复用/覆盖旧清单，发布后继清单；tarot 同口径发布 v160 后继。

输出（新文件，旧清单原样保留）：
  stargazer_ml/v5_hybrid/risk_r0_v1/live_monitor_fingerprints_20261108_v172_flow.json.gz
  stargazer_ml/v5_hybrid/green_common_profile/tarot_live_manifest_v160_20261108_rel.json.gz

校验：flat 用真实 joint_shadow_service_v1.verify_fingerprints 零失配；tarot 逐项重算。
"""
import gzip
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RISK = ROOT / 'stargazer_ml/v5_hybrid/risk_r0_v1'
GCP = ROOT / 'stargazer_ml/v5_hybrid/green_common_profile'

OLD_FLAT = RISK / 'live_monitor_fingerprints_20261108_v172_flow.json.gz'
NEW_FLAT = RISK / 'live_monitor_fingerprints_20261108_v173_flow.json.gz'
OLD_TAROT = GCP / 'tarot_live_manifest_v160_20261108_rel.json.gz'
NEW_TAROT = GCP / 'tarot_live_manifest_v161_20261108_rel.json.gz'

EXPECTED_CHANGED = {
    'stargazer_desktop/plan_evidence.py',
    'stargazer_desktop/postclose_risk_runtime.py',
    'stargazer_desktop/live_plan_lane_v1.py',
    'stargazer_desktop/i34_bridge.py',
    'stargazer_desktop/golden_stock_analysis.py',
    'stargazer_desktop/sector_rule_scan.py',
    'stargazer_desktop/on_demand_model_worker.py',
    'stargazer_ml/v5_hybrid/green_common_profile/market_price_evidence_v1.py',
    'stargazer_ml/v5_hybrid/risk_r0_v1/tarot_intraday_risk_v2.py',
    'stargazer_ml/v5_hybrid/risk_r0_v1/live_postclose_job_v1.py',
    'stargazer_ml/v5_hybrid/risk_r0_v1/accept_live_postclose_v1.py',
    'stargazer_ml/v5_hybrid/green_common_profile/tarot_live_v2.py',
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
    out['version'] = 'v161_20261108_rel'
    out['release_note'] = (old.get('release_note', '') +
                           ' | v161(后继清单): 风险证据双入口修复（冻结原始选股快照绑定+'
                           '确认计划逐股投影校验，人工新增/输入变化/日期不符显式缺失）+ '
                           'I34 GUI 导入入口（/i34 面板页表单，不自动付费）；'
                           '桌面与 ml 侧运行时引用统一改指 v173/v161 被重算收纳；'
                           '不覆盖 v159/v160，禁止沿用旧清单')
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
    print('v172 entries:', len(flat), '| changed vs v171:', len(flat_changed))
    for k in flat_changed:
        print('   ~', k)
    print('v160 frozen_inputs:', len(frozen), '| changed vs v159:', len(tarot_changed))
    for k in tarot_changed:
        print('   ~', k)
    mism, ok = verify_with_real_verifier(flat)
    tb = verify_tarot(frozen)
    print('verify v172 (joint_shadow_service_v1.verify_fingerprints):',
          'OK' if ok else 'FAIL', mism[:5])
    print('verify v160 (tarot 同口径):', 'OK' if not tb else 'FAIL', tb[:5])
    if not (ok and not tb):
        raise SystemExit('后继清单校验未通过')
    print('published:', NEW_FLAT.name, NEW_TAROT.name)
