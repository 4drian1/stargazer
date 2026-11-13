#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""发布后继冻结清单（I34 桌面集成定稿后，路线 18.183）。

背景：server.py 加入 i34 路由注册、新增 stargazer_desktop/i34_bridge.py、
Stargazer_demo_safe.spec 补 I34 白名单 → v89 运行时与 Tarot80 均检测到 2 项失配。
按纪律：不得复用/覆盖旧清单，必须发布后继清单并让运行时指向它，再构建候选 EXE。

输出（新文件，旧清单原样保留）：
  risk_r0_v1/live_monitor_fingerprints_20261101_v90_flow.json.gz   (flat: path->sha256)
  green_common_profile/tarot_live_manifest_v81_20261101_rel.json.gz (frozen_inputs + 元信息)

每条键按项目相对 POSIX 路径解析（project_root = 仓库根），SHA256 现算。
变更：
  + stargazer_desktop/i34_bridge.py（新增随 EXE 发布的桌面源码）
  ~ 重新计算所有既有键的 SHA（故 server.py / Stargazer_demo_safe.spec / 7 处清单引用
    文件内容变化被正确收纳，不再失配）
校验：写出后用真实 verify_fingerprints 与 tarot 同口径逐项重算，零失配才算成功。
"""
import gzip
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RISK = ROOT / 'stargazer_ml/v5_hybrid/risk_r0_v1'
GCP = ROOT / 'stargazer_ml/v5_hybrid/green_common_profile'

OLD_V89 = RISK / 'live_monitor_fingerprints_20261101_v90_flow.json.gz'
NEW_V90 = RISK / 'live_monitor_fingerprints_20261101_v91_flow.json.gz'
OLD_TAROT = GCP / 'tarot_live_manifest_v81_20261101_rel.json.gz'
NEW_TAROT = GCP / 'tarot_live_manifest_v82_20261101_rel.json.gz'

NEW_KEYS = ['stargazer_desktop/i34_bridge.py']
CHANGED_NOTE = ('stargazer_desktop/server.py',
                'stargazer_desktop/Stargazer_demo_safe.spec')


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
    out['version'] = 'v82_20261101_rel'
    out['approved_runtime_updates'] = sorted(set(
        list(old.get('approved_runtime_updates', [])) + list(CHANGED_NOTE)))
    out['release_note'] = (old.get('release_note', '') +
                           ' | v81(后继清单): I34 桌面集成定稿(server.py i34 路由注册 + 新增 '
                           'stargazer_desktop/i34_bridge.py + 安全 spec 白名单)导致 server.py/'
                           'Stargazer_demo_safe.spec 变更；全部既有键 SHA 重算，新增 i34_bridge.py 为冻结输入；'
                           '不覆盖 v80，禁止沿用旧清单')
    new_file.write_bytes(gzip.compress(
        json.dumps(out, indent=1, ensure_ascii=False).encode('utf-8'), mtime=0))
    return frozen, changed


def verify_with_real_verifier(flat: dict):
    """用真实 joint_shadow_service_v1.verify_fingerprints 校验，而不是自己比字符串。"""
    sys.path.insert(0, str(RISK))
    import joint_shadow_service_v1 as svc
    mism, ok = svc.verify_fingerprints(flat)
    return mism, ok


def verify_tarot(frozen: dict):
    """与 tarot_live_v2.verified_manifest 完全同口径逐项重算。"""
    bad = [k for k, v in frozen.items() if sha256(ROOT / k) != v]
    return bad


if __name__ == '__main__':
    flat, flat_changed = build_flat(OLD_V89, NEW_V90)
    frozen, tarot_changed = build_tarot(OLD_TAROT, NEW_TAROT)
    print('v90  entries:', len(flat), '| changed vs v89:', len(flat_changed))
    for k in flat_changed:
        print('   ~', k)
    print('v81  frozen_inputs:', len(frozen), '| changed vs v80:', len(tarot_changed))
    for k in tarot_changed:
        print('   ~', k)
    print('added:', NEW_KEYS)

    mism, ok = verify_with_real_verifier(flat)
    tb = verify_tarot(frozen)
    print('verify v90 (joint_shadow_service_v1.verify_fingerprints):',
          'OK' if ok else 'FAIL', mism[:5])
    print('verify v81 (tarot 同口径):', 'OK' if not tb else 'FAIL', tb[:5])
    if not (ok and not tb):
        raise SystemExit('后继清单校验未通过')
    print('published:', NEW_V90.name, NEW_TAROT.name)
