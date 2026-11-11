#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""发布后继冻结清单（2026-11-10 第九批）：强势池回读修复 + 全球市场收集器 + 持仓富化 + 页面修正。

变更（vs v198/v186）：
  strong_sector_pool_prep_v1.py：回读核验改内容级（不再用 ssw.encoded 比对生产 save 字节，
  修复 POOL_READBACK_MISMATCH 全部失败）；该错误不再永久不可重试（旧失败任务可恢复）；
  server.py：新增 POST /api/market/global_overview/collect；/api/state 持仓富化
  （数量/加权成本/可卖T+1/盈亏）+ 关联公告（Corvus）；
  services.py：position_details；global_market_collect_v1.py（新）：腾讯免费源快照收集器
  （美股/黄金/白银/WTI/布伦特/恒生真实值；日经/中美10Y 如实 MISSING，不拼造）；
  web/：交易计划盘后选股按钮、强势池有界轮询、板块展开、Corvus 近期/历史分离、
  全球市场上新按钮、持仓数量/可卖/盈亏/关联公告、复盘字段中文名。
golden_stock_analysis / on_demand_model_worker / sector_rule_scan / market_price_evidence_v1 /
accept_live_postclose_v1 / live_postclose_job_v1 / tarot_intraday_risk_v2 引用改指 v199，
tarot_live_v2 引用改指 v187。
按纪律：不覆盖旧清单；回归 33/33+24/24+2/2+ALL_PASS 后发布。

输出（新文件，旧清单原样保留）：
  stargazer_ml/v5_hybrid/risk_r0_v1/live_monitor_fingerprints_20261110_v205_flow.json.gz
  stargazer_ml/v5_hybrid/green_common_profile/tarot_live_manifest_v193_20261110_rel.json.gz
"""
import gzip
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RISK = ROOT / 'stargazer_ml/v5_hybrid/risk_r0_v1'
GCP = ROOT / 'stargazer_ml/v5_hybrid/green_common_profile'

OLD_FLAT = RISK / 'live_monitor_fingerprints_20261110_v204_flow.json.gz'
NEW_FLAT = RISK / 'live_monitor_fingerprints_20261110_v205_flow.json.gz'
OLD_TAROT = GCP / 'tarot_live_manifest_v192_20261110_rel.json.gz'
NEW_TAROT = GCP / 'tarot_live_manifest_v193_20261110_rel.json.gz'

EXPECTED_CHANGED = {
    'stargazer_desktop/i34_bridge.py',
    'stargazer_desktop/global_market_collect_v1.py',
    'stargazer_desktop/strong_sector_pool_prep_v1.py',
    'stargazer_desktop/server.py',
    'stargazer_desktop/services.py',
    'stargazer_desktop/web/index.html',
    'stargazer_desktop/strong_sector_pool_prep_v1.py',
    'stargazer_desktop/server.py',
    'stargazer_desktop/services.py',
    'stargazer_desktop/web/index.html',
    'stargazer_desktop/golden_stock_analysis.py',
    'stargazer_desktop/on_demand_model_worker.py',
    'stargazer_desktop/sector_rule_scan.py',
    'stargazer_ml/v5_hybrid/green_common_profile/market_price_evidence_v1.py',
    'stargazer_ml/v5_hybrid/green_common_profile/tarot_live_v2.py',
    'stargazer_ml/v5_hybrid/risk_r0_v1/accept_live_postclose_v1.py',
    'stargazer_ml/v5_hybrid/risk_r0_v1/live_postclose_job_v1.py',
    'stargazer_ml/v5_hybrid/risk_r0_v1/tarot_intraday_risk_v2.py',
}
RETIRED = set()
NEW_KEYS = set()


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def build_flat(old_file: Path, new_file: Path):
    old = json.loads(gzip.decompress(old_file.read_bytes()))
    assert not new_file.exists(), f'{new_file.name} 已存在，禁止覆盖'
    out, missing, changed, retired = {}, [], [], []
    for k, prev in old.items():
        p = ROOT / k
        if not p.exists():
            if k in RETIRED:
                retired.append(k)
                continue
            missing.append(k)
            continue
        cur = sha256(p)
        if cur != prev:
            changed.append(k)
        out[k] = cur
    assert not missing, f'{old_file.name}: 缺文件 {missing[:5]}'
    unexpected = {c.replace('\\', '/') for c in changed} - EXPECTED_CHANGED
    assert not unexpected, f'意外变更（超出本轮范围）: {unexpected}'
    assert set(retired) == RETIRED, f'退役集合不一致: {set(retired) ^ RETIRED}'
    for k in sorted(NEW_KEYS):
        assert (ROOT / k).is_file(), f'新文件缺失: {k}'
        out[k] = sha256(ROOT / k)
    new_file.write_bytes(gzip.compress(
        json.dumps(out, indent=1, ensure_ascii=False).encode('utf-8'), mtime=0))
    return out, changed + sorted(NEW_KEYS) + sorted(RETIRED)


def build_tarot(old_file: Path, new_file: Path):
    old = json.loads(gzip.decompress(old_file.read_bytes()))
    assert not new_file.exists(), f'{new_file.name} 已存在，禁止覆盖'
    frozen, missing, changed, retired = {}, [], [], []
    for k, prev in old['frozen_inputs'].items():
        p = ROOT / k
        if not p.exists():
            if k in RETIRED:
                retired.append(k)
                continue
            missing.append(k)
            continue
        cur = sha256(p)
        if cur != prev:
            changed.append(k)
        frozen[k] = cur
    assert not missing, f'{old_file.name}: 缺文件 {missing[:5]}'
    unexpected = {c.replace('\\', '/') for c in changed} - EXPECTED_CHANGED
    assert not unexpected, f'意外变更（超出本轮范围）: {unexpected}'
    assert set(retired) == RETIRED, f'退役集合不一致: {set(retired) ^ RETIRED}'
    for k in sorted(NEW_KEYS):
        frozen[k] = sha256(ROOT / k)
    out = dict(old)
    out['frozen_inputs'] = frozen
    out['supersedes'] = old_file.name
    out['supersedes_sha256'] = sha256(old_file)
    out['version'] = 'v193_20261110_rel'
    out['release_note'] = (old.get('release_note', '') +
                           ' | v187(后继清单): 强势池回读内容级核验（POOL_READBACK_MISMATCH 修复+可重试）、'
                           '全球行情收集器（腾讯免费源，缺项如实MISSING）、持仓富化、页面修正；'
                           '不覆盖 v186，禁止沿用旧清单')
    new_file.write_bytes(gzip.compress(
        json.dumps(out, indent=1, ensure_ascii=False).encode('utf-8'), mtime=0))
    return frozen, changed + sorted(NEW_KEYS) + sorted(RETIRED)


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
    print('v199 entries:', len(flat), '| changed vs v198:', len(flat_changed))
    for k in flat_changed:
        print('   ~', k)
    print('v187 frozen_inputs:', len(frozen), '| changed vs v186:', len(tarot_changed))
    mism, ok = verify_with_real_verifier(flat)
    tb = verify_tarot(frozen)
    print('verify v205 (joint_shadow_service_v1.verify_fingerprints):',
          'OK' if ok else 'FAIL', mism[:5])
    print('verify v193 (tarot 同口径):', 'OK' if not tb else 'FAIL', tb[:5])
    if not (ok and not tb):
        raise SystemExit('后继清单校验未通过')
    print('published:', NEW_FLAT.name, NEW_TAROT.name)
