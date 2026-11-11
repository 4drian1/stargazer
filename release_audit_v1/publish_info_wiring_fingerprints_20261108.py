#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""发布后继冻结清单（2026-11-08 第二批）：Corvus 信息接线（主工作台入口 + 计划请求 + 盘中请求）。

变更（vs v173/v161）：
  新增 stargazer_desktop/corvus_info_link_v1.py（信息证据接线层）；
  tarot_monitor_v2.py（盘中送审附 stock 信息事实 + info.* 入合法引用表 + 事件键纳入信息版本）、
  trade_facts_v1.py（提示词信息引用规则）、assistant_workflow.py（towers 动态 + 计划上下文
  information_evidence + 提示词）、plan_evidence.py（information_status/model_view 动态）、
  i34_bridge.py（/i34 统一信息页：连接横幅 + 重要优先 + 其他等级折叠）、
  web/index.html + web/app.js（Corvus 按钮打开统一信息页 + 信息位动态状态）；
  golden_stock_analysis / on_demand_model_worker / sector_rule_scan / market_price_evidence_v1 /
  tarot_intraday_risk_v2 / live_postclose_job_v1 / accept_live_postclose_v1 引用改指 v174，
  tarot_live_v2 引用改指 v162。
按纪律：不得复用/覆盖旧清单，发布后继清单；tarot 同口径发布 v162 后继。

输出（新文件，旧清单原样保留）：
  stargazer_ml/v5_hybrid/risk_r0_v1/live_monitor_fingerprints_20261108_v174_flow.json.gz
  stargazer_ml/v5_hybrid/green_common_profile/tarot_live_manifest_v162_20261108_rel.json.gz
"""
import gzip
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RISK = ROOT / 'stargazer_ml/v5_hybrid/risk_r0_v1'
GCP = ROOT / 'stargazer_ml/v5_hybrid/green_common_profile'

OLD_FLAT = RISK / 'live_monitor_fingerprints_20261108_v173_flow.json.gz'
NEW_FLAT = RISK / 'live_monitor_fingerprints_20261108_v174_flow.json.gz'
OLD_TAROT = GCP / 'tarot_live_manifest_v161_20261108_rel.json.gz'
NEW_TAROT = GCP / 'tarot_live_manifest_v162_20261108_rel.json.gz'

NEW_KEY = 'stargazer_desktop/corvus_info_link_v1.py'

EXPECTED_CHANGED = {
    NEW_KEY,
    'stargazer_desktop/tarot_monitor_v2.py',
    'stargazer_desktop/trade_facts_v1.py',
    'stargazer_desktop/assistant_workflow.py',
    'stargazer_desktop/plan_evidence.py',
    'stargazer_desktop/i34_bridge.py',
    'stargazer_desktop/web/index.html',
    'stargazer_desktop/web/app.js',
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
    # 本轮新增文件（信息证据接线层）纳入清单
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
    out = dict(old)
    out['frozen_inputs'] = frozen
    out['supersedes'] = old_file.name
    out['supersedes_sha256'] = sha256(old_file)
    out['version'] = 'v162_20261108_rel'
    out['release_note'] = (old.get('release_note', '') +
                           ' | v162(后继清单): Corvus 信息接线（主工作台统一信息页入口 + '
                           '计划请求 information_evidence + 盘中请求 info.* 入合法引用表与'
                           '回复校验，D2等待/日期不符/旧版本一律 UNKNOWN 不当已知事实，'
                           '信息不等于涨跌预测）；运行时引用统一改指 v174/v162 被重算收纳；'
                           '不覆盖 v159/v160/v161，禁止沿用旧清单')
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
    print('v174 entries:', len(flat), '| changed vs v173:', len(flat_changed))
    for k in flat_changed:
        print('   ~', k)
    print('v162 frozen_inputs:', len(frozen), '| changed vs v161:', len(tarot_changed))
    for k in tarot_changed:
        print('   ~', k)
    mism, ok = verify_with_real_verifier(flat)
    tb = verify_tarot(frozen)
    print('verify v174 (joint_shadow_service_v1.verify_fingerprints):',
          'OK' if ok else 'FAIL', mism[:5])
    print('verify v162 (tarot 同口径):', 'OK' if not tb else 'FAIL', tb[:5])
    if not (ok and not tb):
        raise SystemExit('后继清单校验未通过')
    print('published:', NEW_FLAT.name, NEW_TAROT.name)
