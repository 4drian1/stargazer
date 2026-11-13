#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""发布后继冻结清单（2026-11-10 第四批）：强势池准备模块第三轮收口（v187/v175）。

变更（vs v186/v174）：strong_sector_pool_prep_v1.py 单文件——
  frozen 复用走统一绑定校验；成员版本创建时冻结且"任务=池=当前"三态一致，缺绑定不得宣称有效；
  state() 任务/目录两条路径统一年代判定（历史=READY_STALE）；_decide 持锁递归改锁外重判
  （FileLock 非重入死锁修复）；_retry_or_stop 锁内遇不可抢占 RUNNING 不覆盖；临时文件按
  attempt 隔离（.part.<attempt_id>，各自只清自己的）。
golden_stock_analysis / on_demand_model_worker / sector_rule_scan / market_price_evidence_v1 /
accept_live_postclose_v1 / live_postclose_job_v1 / tarot_intraday_risk_v2 引用改指 v187，
tarot_live_v2 引用改指 v175。
按纪律：不覆盖旧清单；test_strong_pool_prep_v1.py 29/29 通过后发布。

输出（新文件，旧清单原样保留）：
  stargazer_ml/v5_hybrid/risk_r0_v1/live_monitor_fingerprints_20261110_v187_flow.json.gz
  stargazer_ml/v5_hybrid/green_common_profile/tarot_live_manifest_v175_20261110_rel.json.gz
"""
import gzip
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RISK = ROOT / 'stargazer_ml/v5_hybrid/risk_r0_v1'
GCP = ROOT / 'stargazer_ml/v5_hybrid/green_common_profile'

OLD_FLAT = RISK / 'live_monitor_fingerprints_20261110_v186_flow.json.gz'
NEW_FLAT = RISK / 'live_monitor_fingerprints_20261110_v187_flow.json.gz'
OLD_TAROT = GCP / 'tarot_live_manifest_v174_20261110_rel.json.gz'
NEW_TAROT = GCP / 'tarot_live_manifest_v175_20261110_rel.json.gz'

NEW_KEYS = set()

EXPECTED_CHANGED = {
    'stargazer_desktop/strong_sector_pool_prep_v1.py',
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
    out['version'] = 'v175_20261110_rel'
    out['release_note'] = (old.get('release_note', '') +
                           ' | v175(后继清单): 强势池准备模块第三轮收口（绑定复用/版本三态一致/'
                           '统一年代判定/锁外重判/按 attempt 隔离临时文件，29项测试）；'
                           '不覆盖 v174，禁止沿用旧清单')
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
    print('v187 entries:', len(flat), '| changed vs v186:', len(flat_changed))
    for k in flat_changed:
        print('   ~', k)
    print('v175 frozen_inputs:', len(frozen), '| changed vs v174:', len(tarot_changed))
    for k in tarot_changed:
        print('   ~', k)
    mism, ok = verify_with_real_verifier(flat)
    tb = verify_tarot(frozen)
    print('verify v187 (joint_shadow_service_v1.verify_fingerprints):',
          'OK' if ok else 'FAIL', mism[:5])
    print('verify v175 (tarot 同口径):', 'OK' if not tb else 'FAIL', tb[:5])
    if not (ok and not tb):
        raise SystemExit('后继清单校验未通过')
    print('published:', NEW_FLAT.name, NEW_TAROT.name)
