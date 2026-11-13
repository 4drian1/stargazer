#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""发布后继冻结清单（2026-11-10 第七批）：强势池发布兜底 + WAIT_DAILY 合同 + 去重资格修复。

变更（vs v189/v177）：
  strong_sector_pool_prep_v1.py：_run_attempt 异常处理覆盖计算→提交全程（写池/替换/记账失败
  按当前 attempt_id 安全结束预约，绝不永久 RUNNING，不虚报 READY）；_wait_daily 与读取同一
  合同（state 字段统一，等待日线原因重启可见）。
  notifier.py：run_once 推送关闭时只窗口展示，不消耗外部发送去重资格（重开同内容可发送；
  已发送/送达未知不重发规则保留）。
  web/pentacles.js：强势池区显示"等待日线+原因/历史结果(过期)"。
golden_stock_analysis / on_demand_model_worker / sector_rule_scan / market_price_evidence_v1 /
accept_live_postclose_v1 / live_postclose_job_v1 / tarot_intraday_risk_v2 引用改指 v190，
tarot_live_v2 引用改指 v178。
按纪律：不覆盖旧清单；test_strong_pool_prep_v1.py 33/33、test_push_disabled_dedup_v1.py 2/2
通过后发布。

输出（新文件，旧清单原样保留）：
  stargazer_ml/v5_hybrid/risk_r0_v1/live_monitor_fingerprints_20261110_v190_flow.json.gz
  stargazer_ml/v5_hybrid/green_common_profile/tarot_live_manifest_v178_20261110_rel.json.gz
"""
import gzip
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RISK = ROOT / 'stargazer_ml/v5_hybrid/risk_r0_v1'
GCP = ROOT / 'stargazer_ml/v5_hybrid/green_common_profile'

OLD_FLAT = RISK / 'live_monitor_fingerprints_20261110_v189_flow.json.gz'
NEW_FLAT = RISK / 'live_monitor_fingerprints_20261110_v190_flow.json.gz'
OLD_TAROT = GCP / 'tarot_live_manifest_v177_20261110_rel.json.gz'
NEW_TAROT = GCP / 'tarot_live_manifest_v178_20261110_rel.json.gz'

EXPECTED_CHANGED = {
    'stargazer_desktop/strong_sector_pool_prep_v1.py',
    'stargazer_desktop/notifier.py',
    'stargazer_desktop/web/pentacles.js',
    'stargazer_desktop/sector_rule_scan.py',
    'stargazer_desktop/golden_stock_analysis.py',
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
    out['version'] = 'v178_20261110_rel'
    out['release_note'] = (old.get('release_note', '') +
                           ' | v178(后继清单): 强势池发布路径兜底（写盘/替换失败不永久占用）、'
                           'WAIT_DAILY 合同统一、关闭推送不消耗外部去重资格；不覆盖 v177，禁止沿用旧清单')
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
    print('v190 entries:', len(flat), '| changed vs v189:', len(flat_changed))
    for k in flat_changed:
        print('   ~', k)
    print('v178 frozen_inputs:', len(frozen), '| changed vs v177:', len(tarot_changed))
    for k in tarot_changed:
        print('   ~', k)
    mism, ok = verify_with_real_verifier(flat)
    tb = verify_tarot(frozen)
    print('verify v190 (joint_shadow_service_v1.verify_fingerprints):',
          'OK' if ok else 'FAIL', mism[:5])
    print('verify v178 (tarot 同口径):', 'OK' if not tb else 'FAIL', tb[:5])
    if not (ok and not tb):
        raise SystemExit('后继清单校验未通过')
    print('published:', NEW_FLAT.name, NEW_TAROT.name)
