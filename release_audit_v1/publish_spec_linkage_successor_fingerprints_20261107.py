#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""发布后继冻结清单（2026-11-07，I34 信息影子源部署 + 股票联动模块）。

背景：Stargazer_demo_safe.spec 白名单加入 c3/corvus/impact_v1_1/i34_demo/i34_stock_linkage_v1.py
（影子源正式注册进桌面入口后的新增随 EXE 发布源码）→ v167/v155 预检检测到 spec 失配。
按纪律：不得复用/覆盖旧清单，必须发布后继清单并让运行时指向它，再构建候选 EXE。

输出（新文件，旧清单原样保留）：
  stargazer_ml/v5_hybrid/risk_r0_v1/live_monitor_fingerprints_20261107_v168_flow.json.gz
  stargazer_ml/v5_hybrid/green_common_profile/tarot_live_manifest_v156_20261107_rel.json.gz

变更：
  + c3/corvus/impact_v1_1/i34_demo/i34_stock_linkage_v1.py（新增随 EXE 发布的桌面源码）
  ~ 全部既有键 SHA 重算（spec 变更被正确收纳，不再失配）
校验：写出后用真实 joint_shadow_service_v1.verify_fingerprints 与 tarot 同口径逐项重算，
零失配才算成功。
"""
import gzip
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RISK = ROOT / 'stargazer_ml/v5_hybrid/risk_r0_v1'
GCP = ROOT / 'stargazer_ml/v5_hybrid/green_common_profile'

OLD_FLAT = RISK / 'live_monitor_fingerprints_20261107_v170_flow.json.gz'
NEW_FLAT = RISK / 'live_monitor_fingerprints_20261107_v171_flow.json.gz'
OLD_TAROT = GCP / 'tarot_live_manifest_v158_20261107_rel.json.gz'
NEW_TAROT = GCP / 'tarot_live_manifest_v159_20261107_rel.json.gz'

NEW_KEYS = ['c3/corvus/impact_v1_1/i34_demo/i34_stock_linkage_v1.py']


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
    for k in NEW_KEYS:
        p = ROOT / k
        assert p.exists(), f'新键不存在: {k}'
        out[k] = sha256(p)
    assert not missing, f'{old_file.name}: 缺文件 {missing[:5]}'
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
    for k in NEW_KEYS:
        p = ROOT / k
        assert p.exists(), f'新键不存在: {k}'
        frozen[k] = sha256(p)
    assert not missing, f'{old_file.name}: 缺文件 {missing[:5]}'

    out = dict(old)
    out['frozen_inputs'] = frozen
    out['supersedes'] = old_file.name
    out['supersedes_sha256'] = sha256(old_file)
    out['version'] = 'v159_20261107_rel'
    out['approved_runtime_updates'] = sorted(set(
        list(old.get('approved_runtime_updates', [])) +
        ['stargazer_desktop/Stargazer_demo_safe.spec']))
    out['release_note'] = (old.get('release_note', '') +
                           ' | v159(后继清单): 第二轮复核三缺口收口——联动默认并集(信息代码∪计划股票)、'
                           '生产数据目录注入(dist/data 直读)、D2 待处理队列+信息层自动合并；'
                           '9 运行时引用文件改指 v171/v159 被重算收纳；不覆盖 v158，禁止沿用旧清单')
    new_file.write_bytes(gzip.compress(
        json.dumps(out, indent=1, ensure_ascii=False).encode('utf-8'), mtime=0))
    return frozen, changed


def verify_with_real_verifier(flat: dict):
    sys.path.insert(0, str(RISK))
    import joint_shadow_service_v1 as svc
    mism, ok = svc.verify_fingerprints(flat)
    return mism, ok


def verify_tarot(frozen: dict):
    bad = [k for k, v in frozen.items() if sha256(ROOT / k) != v]
    return bad


if __name__ == '__main__':
    flat, flat_changed = build_flat(OLD_FLAT, NEW_FLAT)
    frozen, tarot_changed = build_tarot(OLD_TAROT, NEW_TAROT)
    print('v171 entries:', len(flat), '| changed vs v170:', len(flat_changed))
    for k in flat_changed:
        print('   ~', k)
    print('v159 frozen_inputs:', len(frozen), '| changed vs v158:', len(tarot_changed))
    for k in tarot_changed:
        print('   ~', k)
    print('added:', NEW_KEYS)

    mism, ok = verify_with_real_verifier(flat)
    tb = verify_tarot(frozen)
    print('verify v171 (joint_shadow_service_v1.verify_fingerprints):',
          'OK' if ok else 'FAIL', mism[:5])
    print('verify v159 (tarot 同口径):', 'OK' if not tb else 'FAIL', tb[:5])
    if not (ok and not tb):
        raise SystemExit('后继清单校验未通过')
    print('published:', NEW_FLAT.name, NEW_TAROT.name)
