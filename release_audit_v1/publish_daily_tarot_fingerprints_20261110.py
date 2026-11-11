#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""发布后继冻结清单（2026-11-10 第二批）：每日塔罗接线修复。

根因：daily_tarot_v1.py 文案层就绪，但 notifier._maybe_daily_tarot 调度方法从未合入，
test_daily_tarot_v1.py 一直 AttributeError 失败，工作日 09:00 塔罗推送从未发出。
本轮（vs v184/v172）：
  notifier.py（新增 _maybe_daily_tarot + 主循环调用）、config.example.yaml（push.daily_tarot 段）、
  Stargazer_demo_safe.spec（hiddenimports 增列 daily_tarot_v1）；
  新绑定 stargazer_desktop/daily_tarot_v1.py 入双清单（此前两份清单均未绑定该文件）；
  golden_stock_analysis / on_demand_model_worker / sector_rule_scan / market_price_evidence_v1 /
  accept_live_postclose_v1 / live_postclose_job_v1 / tarot_intraday_risk_v2 引用改指 v185，
  tarot_live_v2 引用改指 v173。
按纪律：不覆盖旧清单；测试 test_daily_tarot_v1.py 56/56 通过后发布。

输出（新文件，旧清单原样保留）：
  stargazer_ml/v5_hybrid/risk_r0_v1/live_monitor_fingerprints_20261110_v185_flow.json.gz
  stargazer_ml/v5_hybrid/green_common_profile/tarot_live_manifest_v173_20261110_rel.json.gz
"""
import gzip
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RISK = ROOT / 'stargazer_ml/v5_hybrid/risk_r0_v1'
GCP = ROOT / 'stargazer_ml/v5_hybrid/green_common_profile'

OLD_FLAT = RISK / 'live_monitor_fingerprints_20261110_v184_flow.json.gz'
NEW_FLAT = RISK / 'live_monitor_fingerprints_20261110_v185_flow.json.gz'
OLD_TAROT = GCP / 'tarot_live_manifest_v172_20261110_rel.json.gz'
NEW_TAROT = GCP / 'tarot_live_manifest_v173_20261110_rel.json.gz'

NEW_KEYS = {'stargazer_desktop/daily_tarot_v1.py'}

EXPECTED_CHANGED = {
    'stargazer_desktop/notifier.py',
    'stargazer_desktop/config.example.yaml',
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
    out['stargazer_desktop/daily_tarot_v1.py'] = sha256(ROOT / 'stargazer_desktop/daily_tarot_v1.py')
    out['stargazer_desktop/config.example.yaml'] = sha256(ROOT / 'stargazer_desktop/config.example.yaml')
    new_file.write_bytes(gzip.compress(
        json.dumps(out, indent=1, ensure_ascii=False).encode('utf-8'), mtime=0))
    return out, changed + ['stargazer_desktop/daily_tarot_v1.py', 'stargazer_desktop/config.example.yaml']


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
    frozen['stargazer_desktop/daily_tarot_v1.py'] = sha256(ROOT / 'stargazer_desktop/daily_tarot_v1.py')
    out = dict(old)
    out['frozen_inputs'] = frozen
    out['supersedes'] = old_file.name
    out['supersedes_sha256'] = sha256(old_file)
    out['version'] = 'v173_20261110_rel'
    out['release_note'] = (old.get('release_note', '') +
                           ' | v173(后继清单): 每日塔罗调度接线（notifier._maybe_daily_tarot 工作日'
                           '09:00一次/周末不发/LLM失败静默回退事实核；daily_tarot_v1 新绑定入清单）；'
                           '不覆盖 v172，禁止沿用旧清单')
    new_file.write_bytes(gzip.compress(
        json.dumps(out, indent=1, ensure_ascii=False).encode('utf-8'), mtime=0))
    return frozen, changed + ['stargazer_desktop/daily_tarot_v1.py']


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
    for k in ('stargazer_desktop/daily_tarot_v1.py', 'stargazer_desktop/notifier.py'):
        assert k in flat, f'flow 清单缺关键绑定: {k}'
        assert k in frozen, f'tarot 清单缺关键绑定: {k}'
    print('v185 entries:', len(flat), '| changed vs v184:', len(flat_changed))
    for k in flat_changed:
        print('   ~', k)
    print('v173 frozen_inputs:', len(frozen), '| changed vs v172:', len(tarot_changed))
    for k in tarot_changed:
        print('   ~', k)
    mism, ok = verify_with_real_verifier(flat)
    tb = verify_tarot(frozen)
    print('verify v185 (joint_shadow_service_v1.verify_fingerprints):',
          'OK' if ok else 'FAIL', mism[:5])
    print('verify v173 (tarot 同口径):', 'OK' if not tb else 'FAIL', tb[:5])
    if not (ok and not tb):
        raise SystemExit('后继清单校验未通过')
    print('published:', NEW_FLAT.name, NEW_TAROT.name)
