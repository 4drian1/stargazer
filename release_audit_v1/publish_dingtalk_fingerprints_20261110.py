#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""发布后继冻结清单（2026-11-10 第三批）：钉钉推送通道修复。

根因：config provider=dingtalk 且 dingtalk_webhook/secret 均已配置，但 notifier.send_wechat
丢失了 dingtalk 分支（仅 pushplus/serverchan），降级走到 serverchan 分支报
"未配置 serverchan_sendkey"；server /api/state 缺 has_dingtalk、update_config 缺
dingtalk_webhook/secret 保存键。本轮（vs v185/v173）：
  notifier.py（恢复 dingtalk 分支：markdown 段落升级 + 加签）、server.py（has_dingtalk + 保存键）；
  golden_stock_analysis / on_demand_model_worker / sector_rule_scan / market_price_evidence_v1 /
  accept_live_postclose_v1 / live_postclose_job_v1 / tarot_intraday_risk_v2 引用改指 v186，
  tarot_live_v2 引用改指 v174。
按纪律：不覆盖旧清单；test_dingtalk_paragraphs_v1.py 通过 + 全回归后发布。

输出（新文件，旧清单原样保留）：
  stargazer_ml/v5_hybrid/risk_r0_v1/live_monitor_fingerprints_20261110_v186_flow.json.gz
  stargazer_ml/v5_hybrid/green_common_profile/tarot_live_manifest_v174_20261110_rel.json.gz
"""
import gzip
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RISK = ROOT / 'stargazer_ml/v5_hybrid/risk_r0_v1'
GCP = ROOT / 'stargazer_ml/v5_hybrid/green_common_profile'

OLD_FLAT = RISK / 'live_monitor_fingerprints_20261110_v185_flow.json.gz'
NEW_FLAT = RISK / 'live_monitor_fingerprints_20261110_v186_flow.json.gz'
OLD_TAROT = GCP / 'tarot_live_manifest_v173_20261110_rel.json.gz'
NEW_TAROT = GCP / 'tarot_live_manifest_v174_20261110_rel.json.gz'

NEW_KEYS = set()

EXPECTED_CHANGED = {
    'stargazer_desktop/notifier.py',
    'stargazer_desktop/server.py',
    'stargazer_desktop/golden_stock_analysis.py',
    'stargazer_desktop/on_demand_model_worker.py',
    'stargazer_desktop/sector_rule_scan.py',
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
    assert not (NEW_KEYS & set(old)), '新绑定不应已存在于旧清单'
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
    out['version'] = 'v174_20261110_rel'
    out['release_note'] = (old.get('release_note', '') +
                           ' | v174(后继清单): 钉钉推送通道修复（notifier.send_wechat 恢复 dingtalk '
                           '分支：markdown 段落升级+加签；/api/state has_dingtalk；配置保存键补齐）；'
                           '不覆盖 v173，禁止沿用旧清单')
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
    for k in ('stargazer_desktop/notifier.py', 'stargazer_desktop/server.py'):
        assert k in flat, f'flow 清单缺关键绑定: {k}'
        assert k in frozen, f'tarot 清单缺关键绑定: {k}'
    print('v186 entries:', len(flat), '| changed vs v185:', len(flat_changed))
    for k in flat_changed:
        print('   ~', k)
    print('v174 frozen_inputs:', len(frozen), '| changed vs v173:', len(tarot_changed))
    for k in tarot_changed:
        print('   ~', k)
    mism, ok = verify_with_real_verifier(flat)
    tb = verify_tarot(frozen)
    print('verify v186 (joint_shadow_service_v1.verify_fingerprints):',
          'OK' if ok else 'FAIL', mism[:5])
    print('verify v174 (tarot 同口径):', 'OK' if not tb else 'FAIL', tb[:5])
    if not (ok and not tb):
        raise SystemExit('后继清单校验未通过')
    print('published:', NEW_FLAT.name, NEW_TAROT.name)
