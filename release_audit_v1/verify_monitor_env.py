"""监控机部署自检脚本（路线 #18.46）。

用法（监控机 venv 解释器）：
  python verify_monitor_env.py [项目根]
检查：
  1. Python >= 3.12
  2. 关键依赖可导入且版本与锁定一致（numpy/scipy/pandas/lightgbm/pyarrow/msgpack/Flask/requests/akshare）
  3. 三份相对键冻结指纹清单全量 SHA 校验通过
退出码 0 = 全部通过；非 0 = 有缺口（缺口明细打印到 stdout）。
"""
import json
import gzip
import hashlib
import sys
from pathlib import Path

ROOT = Path(sys.argv[1] if len(sys.argv) > 1 else Path(__file__).resolve().parents[1]).resolve()
failures = []

# 1. Python 版本
if sys.version_info < (3, 12):
    failures.append(f'Python {sys.version_info.major}.{sys.version_info.minor} 低于 3.12（NumPy/SciPy 锁定要求）')
print(f'[1] Python {sys.version}')

# 2. 依赖
import importlib.metadata as md
NEED = {'numpy': '2.5.2', 'scipy': '1.18.1', 'pandas': '3.0.5', 'lightgbm': '4.7.0',
        'pyarrow': '25.0.1', 'msgpack': '1.2.2', 'Flask': '3.1.3', 'requests': '2.34.2',
        'akshare': '1.18.94'}
for pkg, want in NEED.items():
    try:
        got = md.version(pkg)
        flag = 'OK' if got == want else f'VERSION_MISMATCH(want {want})'
        if got != want:
            failures.append(f'{pkg}=={got}，锁定要求 {want}')
    except md.PackageNotFoundError:
        flag = 'MISSING'
        failures.append(f'{pkg} 未安装（锁定 {want}）')
    print(f'[2] {pkg}: {flag}')

# 3. 冻结指纹
def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()

RISK = ROOT / 'stargazer_ml/v5_hybrid/risk_r0_v1'
GCP = ROOT / 'stargazer_ml/v5_hybrid/green_common_profile'
CHECKS = [
    (RISK / 'live_monitor_fingerprints_20261026_v27_rel.json', 'json'),
    (RISK / 'joint_shadow_fingerprints_rel_v1.json', 'json'),
    (GCP / 'tarot_live_manifest_v18_20261026_rel.json.gz', 'tarot'),
]
for file, kind in CHECKS:
    if not file.exists():
        failures.append(f'指纹清单缺失: {file.relative_to(ROOT)}')
        print(f'[3] {file.name}: MISSING')
        continue
    if kind == 'tarot':
        entries = json.loads(gzip.decompress(file.read_bytes()))['frozen_inputs']
    else:
        entries = json.loads(file.read_text(encoding='utf-8'))
    bad = [k for k, v in entries.items()
           if not (ROOT / k).exists() or sha256(ROOT / k) != v]
    if bad:
        failures.append(f'{file.name} 校验失败 {len(bad)}/{len(entries)} 项: {bad[:5]}')
    print(f'[3] {file.name}: {"OK" if not bad else "MISMATCH"} ({len(entries)-len(bad)}/{len(entries)})')

print()
if failures:
    print('FAIL：')
    for f in failures:
        print('  -', f)
    sys.exit(1)
print('PASS：监控机环境校验全部通过')
