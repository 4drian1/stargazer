"""Read-only package gate; inspect bytes, never launch the EXE or read credentials."""
import hashlib
import json
from pathlib import Path
import re
import sys

from PyInstaller.archive.readers import CArchiveReader
from verify_live_release import REQUIRED_WORKERS


def verify_package(exe, root, reader_factory=CArchiveReader):
    root, exe = Path(root), Path(exe)
    archive = reader_factory(str(exe))
    errors, workers = [], []
    for name in REQUIRED_WORKERS:
        if name not in archive.toc:
            errors.append('PACKAGE_WORKER_MISSING:'+name)
            continue
        raw = archive.extract(name)
        equal = raw == (root/'stargazer_desktop'/name).read_bytes()
        workers.append(dict(name=name, source_equal=equal, sha256=hashlib.sha256(raw).hexdigest()))
        if not equal: errors.append('PACKAGE_WORKER_CHANGED:'+name)
    for name in archive.toc:
        clean = name.replace('\\', '/').lower()
        if clean == 'config.yaml' or clean.startswith(('data/', 'torch/', 'torch\\')):
            errors.append('PACKAGE_PRIVATE_OR_HEAVY_INPUT:'+name)
    # The bundled script must point at the same published manifest as disk.
    name = 'golden_stock_analysis.py'
    if name in archive.toc:
        packaged = set(re.findall(rb'live_monitor_fingerprints_\d+_v\d+_flow\.json\.gz', archive.extract(name)))
        current = set(re.findall(rb'live_monitor_fingerprints_\d+_v\d+_flow\.json\.gz',
                                 (root/'stargazer_desktop'/name).read_bytes()))
        if len(packaged) != 1 or packaged != current: errors.append('PACKAGE_MANIFEST_REFERENCE_MISMATCH')
    return dict(ok=not errors, exe=str(exe), bytes=exe.stat().st_size,
                sha256=hashlib.sha256(exe.read_bytes()).hexdigest(), workers=workers, errors=errors)


if __name__ == '__main__':
    value = verify_package(sys.argv[1], Path(__file__).resolve().parents[1])
    print(json.dumps(value, ensure_ascii=False, indent=2))
    raise SystemExit(0 if value['ok'] else 1)
