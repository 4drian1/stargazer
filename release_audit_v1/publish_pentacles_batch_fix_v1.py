#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""发布后继冻结清单（本轮：pentacles_monitor.py 报价通道 batch_size=30 修复）。

变更（vs v175/v163）：
  仅 1 个产品文件被本次修复改动：
    stargazer_desktop/pentacles_monitor.py
  8 个运行时引用文件（版本串 v175->v176 / v163->v164 改动，属发布一部分）：
    stargazer_desktop/golden_stock_analysis.py
    stargazer_desktop/sector_rule_scan.py
    stargazer_desktop/on_demand_model_worker.py
    stargazer_ml/v5_hybrid/green_common_profile/market_price_evidence_v1.py
    stargazer_ml/v5_hybrid/green_common_profile/tarot_live_v2.py
    stargazer_ml/v5_hybrid/risk_r0_v1/accept_live_postclose_v1.py
    stargazer_ml/v5_hybrid/risk_r0_v1/live_postclose_job_v1.py
    stargazer_ml/v5_hybrid/risk_r0_v1/tarot_intraday_risk_v2.py

按纪律：不得复用/覆盖旧清单；发布后继清单 v176 / v164（旧 v175 / v163 原样保留）。
"""
import gzip
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RISK = ROOT / 'stargazer_ml/v5_hybrid/risk_r0_v1'
GCP = ROOT / 'stargazer_ml/v5_hybrid/green_common_profile'

OLD_FLAT = RISK / 'live_monitor_fingerprints_20261108_v175_flow.json.gz'
NEW_FLAT = RISK / 'live_monitor_fingerprints_20261108_v176_flow.json.gz'
OLD_TAROT = GCP / 'tarot_live_manifest_v163_20261108_rel.json.gz'
NEW_TAROT = GCP / 'tarot_live_manifest_v164_20261108_rel.json.gz'

EXPECTED_CHANGED = {
    'stargazer_desktop/pentacles_monitor.py',
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
    out['version'] = 'v164_20261108_rel'
    out['release_note'] = (old.get('release_note', '') +
                           ' | v164(后继清单): 同 v163 内容 + pentacles_monitor.py '
                           '报价通道 batch_size=30 修复（7 只计划票合并为 1 次请求）；'
                           '不覆盖 v163 及更早。')
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
    print('v176 entries:', len(flat), '| changed vs v175:', len(flat_changed))
    for k in flat_changed:
        print('   ~', k)
    print('v164 frozen_inputs:', len(frozen), '| changed vs v163:', len(tarot_changed))
    for k in tarot_changed:
        print('   ~', k)
    mism, ok = verify_with_real_verifier(flat)
    tb = verify_tarot(frozen)
    print('verify v176 (joint_shadow_service_v1.verify_fingerprints):',
          'OK' if ok else 'FAIL', mism[:5])
    print('verify v164 (tarot 同口径):', 'OK' if not tb else 'FAIL', tb[:5])
    if not (ok and not tb):
        raise SystemExit('后继清单校验未通过')
    print('published:', NEW_FLAT.name, NEW_TAROT.name)
