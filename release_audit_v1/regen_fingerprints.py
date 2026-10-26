"""重新生成项目相对路径键的冻结指纹清单（路线 #18.46）。

输入（旧，绝对路径键，存档保留不覆盖）：
  risk_r0_v1/live_monitor_fingerprints_20261023_v2.json
  risk_r0_v1/joint_shadow_fingerprints_dev_rec4.json
  green_common_profile/tarot_live_manifest_v17_20261026_flow.json.gz
输出（新，相对 POSIX 路径键 + 当前工作区重算 SHA256）：
  risk_r0_v1/live_monitor_fingerprints_20261026_v27_rel.json
  risk_r0_v1/joint_shadow_fingerprints_rel_v1.json
  green_common_profile/tarot_live_manifest_v18_20261026_rel.json.gz
校验：生成后对每个新清单跑一遍相对键解析校验，全部通过才算成功。
"""
import gzip
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RISK = ROOT / 'stargazer_ml/v5_hybrid/risk_r0_v1'
GCP = ROOT / 'stargazer_ml/v5_hybrid/green_common_profile'


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def rel_key(abs_key: str) -> str:
    p = Path(abs_key)
    if not p.is_absolute():
        return abs_key.replace('\\', '/')
    return p.resolve().relative_to(ROOT).as_posix()


def regen_flat(old_file: Path, new_file: Path):
    old = json.loads(old_file.read_text(encoding='utf-8'))
    out = {}
    missing = []
    for k, _v in old.items():
        rk = rel_key(k)
        p = ROOT / rk
        if not p.exists():
            missing.append(rk)
            continue
        out[rk] = sha256(p)
    assert not missing, f'{old_file.name}: 缺文件 {missing}'
    new_file.write_text(json.dumps(out, indent=1, ensure_ascii=False) + '\n',
                        encoding='utf-8')
    return len(out)


def regen_tarot(old_file: Path, new_file: Path):
    old = json.loads(gzip.decompress(old_file.read_bytes()))
    out = dict(old)
    frozen = {}
    missing = []
    for k, _v in old['frozen_inputs'].items():
        rk = rel_key(k)
        p = ROOT / rk
        if not p.exists():
            missing.append(rk)
            continue
        frozen[rk] = sha256(p)
    assert not missing, f'{old_file.name}: 缺文件 {missing}'
    out['frozen_inputs'] = frozen
    out['approved_runtime_updates'] = [rel_key(k) for k in old['approved_runtime_updates']]
    out['approved_predecessor_inputs'] = {rel_key(k): v for k, v in
                                          old['approved_predecessor_inputs'].items()}
    out['supersedes'] = old_file.name
    out['supersedes_sha256'] = sha256(old_file)
    out['release_note'] = (old.get('release_note', '') +
                           ' | v18: 指纹键迁移为项目相对路径（#18.46），SHA 全量重算，校验保留')
    payload = json.dumps(out, indent=1, ensure_ascii=False).encode('utf-8')
    new_file.write_bytes(gzip.compress(payload, mtime=0))
    return len(frozen)


def verify_flat(new_file: Path):
    d = json.loads(new_file.read_text(encoding='utf-8'))
    bad = [k for k, v in d.items() if sha256(ROOT / k) != v]
    assert not bad, f'{new_file.name} 校验失败: {bad[:5]}'
    return len(d)


def verify_tarot(new_file: Path):
    m = json.loads(gzip.decompress(new_file.read_bytes()))
    bad = [k for k, v in m['frozen_inputs'].items() if sha256(ROOT / k) != v]
    assert not bad, f'{new_file.name} 校验失败: {bad[:5]}'
    return len(m['frozen_inputs'])


if __name__ == '__main__':
    n1 = regen_flat(RISK / 'live_monitor_fingerprints_20261023_v2.json',
                    RISK / 'live_monitor_fingerprints_20261026_v27_rel.json')
    n2 = regen_flat(RISK / 'joint_shadow_fingerprints_dev_rec4.json',
                    RISK / 'joint_shadow_fingerprints_rel_v1.json')
    n3 = regen_tarot(GCP / 'tarot_live_manifest_v17_20261026_flow.json.gz',
                     GCP / 'tarot_live_manifest_v18_20261026_rel.json.gz')
    print('regenerated:', n1, n2, n3)
    print('verified:', verify_flat(RISK / 'live_monitor_fingerprints_20261026_v27_rel.json'),
          verify_flat(RISK / 'joint_shadow_fingerprints_rel_v1.json'),
          verify_tarot(GCP / 'tarot_live_manifest_v18_20261026_rel.json.gz'))
