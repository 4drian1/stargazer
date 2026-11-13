#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""发布后继冻结清单（2026-11-10 第八批）：新 UI 日常操作链统一接线。

变更（vs v192/v180）：
  server.py：GET /api/strong-pool/board（板块真实股票只读）、/api/assistant/plan/reject（拒绝语义）；
  assistant_workflow.py：拒绝标记（state 不展示已拒绝草案为待确认；confirm 拒绝已拒绝草案）；
  strong_sector_watch_v1.py：invoke 保留退出码/受限 stderr/超时与超限原因；spec 补 worker 依赖 datas；
  app.py：桌面窗口 text_select；i34_bridge.py：panel 展示字段 enrich（原文URL/影响说明/披露时间，缺失置 null）；
  web/：新 UI 构建产物（旧 app.js/pentacles.js/local_storage.html 与旧 assets 退役，备份于 _archive/web_backup_20261110_v192）。
golden_stock_analysis / on_demand_model_worker / sector_rule_scan / market_price_evidence_v1 /
accept_live_postclose_v1 / live_postclose_job_v1 / tarot_intraday_risk_v2 引用改指 v193，
tarot_live_v2 引用改指 v181。
按纪律：不覆盖旧清单；回归全绿 + 前端构建通过后发布。

输出（新文件，旧清单原样保留）：
  stargazer_ml/v5_hybrid/risk_r0_v1/live_monitor_fingerprints_20261110_v195_flow.json.gz
  stargazer_ml/v5_hybrid/green_common_profile/tarot_live_manifest_v183_20261110_rel.json.gz
"""
import gzip
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RISK = ROOT / 'stargazer_ml/v5_hybrid/risk_r0_v1'
GCP = ROOT / 'stargazer_ml/v5_hybrid/green_common_profile'

OLD_FLAT = RISK / 'live_monitor_fingerprints_20261110_v194_flow.json.gz'
NEW_FLAT = RISK / 'live_monitor_fingerprints_20261110_v195_flow.json.gz'
OLD_TAROT = GCP / 'tarot_live_manifest_v182_20261110_rel.json.gz'
NEW_TAROT = GCP / 'tarot_live_manifest_v183_20261110_rel.json.gz'

EXPECTED_CHANGED = {
    'stargazer_desktop/web/index.html',
    'stargazer_desktop/assistant_workflow.py',
    'stargazer_desktop/strong_sector_watch_v1.py',
    'stargazer_desktop/app.py',
    'stargazer_desktop/i34_bridge.py',
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
# 退役的旧 UI 文件（已备份 _archive/web_backup_20261110_v192，不再入清单）
RETIRED = {
    'stargazer_desktop/web/assets/index-QPs-mPSB.js',
}
NEW_KEYS = {
    'stargazer_desktop/web/assets/index-mEyVFVuZ.js',
    'stargazer_desktop/web/assets/index-Bs_MEfWr.css',
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
    out['version'] = 'v183_20261110_rel'
    out['release_note'] = (old.get('release_note', '') +
                           ' | v181(后继清单): 新 UI 日常操作链（worker 依赖/text_select/拒绝语义/'
                           '板块股票只读/强势池有界轮询/i34 展示字段/页面简化）；退役旧 UI 文件已备份；'
                           '刷新按钮显式反馈（刷新中…/完成时间）；不覆盖 v182，禁止沿用旧清单')
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
    print('v193 entries:', len(flat), '| changed vs v192:', len(flat_changed))
    for k in flat_changed:
        print('   ~', k)
    print('v182 frozen_inputs:', len(frozen), '| changed vs v182:', len(tarot_changed))
    mism, ok = verify_with_real_verifier(flat)
    tb = verify_tarot(frozen)
    print('verify v195 (joint_shadow_service_v1.verify_fingerprints):',
          'OK' if ok else 'FAIL', mism[:5])
    print('verify v183 (tarot 同口径):', 'OK' if not tb else 'FAIL', tb[:5])
    if not (ok and not tb):
        raise SystemExit('后继清单校验未通过')
    print('published:', NEW_FLAT.name, NEW_TAROT.name)
