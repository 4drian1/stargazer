"""Run one test suite with the desktop code and resources from the candidate EXE."""
import importlib.abc
import importlib.util
import os
from pathlib import Path
import re
import runpy
import sys
import tempfile

from PyInstaller.archive.readers import CArchiveReader


class CandidateModules(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    def __init__(self, archive, desktop, bundle):
        self.archive = archive
        self.desktop = desktop
        self.bundle = bundle
        self.loaded = set()

    def find_spec(self, fullname, path=None, target=None):
        name = Path(*fullname.split('.'))
        script = name.with_suffix('.py')
        package = name / '__init__.py'
        if fullname not in self.archive.toc:
            return None
        risk_module = self.desktop.parent / 'stargazer_ml/v5_hybrid/risk_r0_v1' / script
        if (self.desktop / script).is_file() or fullname == 'runtime_disk_limit_v1' and risk_module.is_file():
            return importlib.util.spec_from_file_location(fullname, self.bundle / script, loader=self)
        if (self.desktop / package).is_file():
            return importlib.util.spec_from_file_location(fullname, self.bundle / package, loader=self,
                                                          submodule_search_locations=[str(self.bundle / name)])
        return None

    def create_module(self, spec):
        return None

    def exec_module(self, module):
        self.loaded.add(module.__name__)
        exec(self.archive.extract(module.__name__), module.__dict__)


def main(exe, suite):
    root = Path(__file__).resolve().parents[1]
    desktop = root / 'stargazer_desktop'
    archive = CArchiveReader(str(exe))
    pyz = archive.open_embedded_archive(next(k for k in archive.toc if k.endswith('.pyz')))
    files = {k.replace(chr(92), '/'): k for k in archive.toc}
    homepage = archive.extract(files['web/index.html'])
    referenced = [b.decode() for b in re.findall(rb'["\x27]\./(assets/[^"\x27]+)["\x27]', homepage)]
    assert len(referenced) >= 2
    for path in ('web/index.html', *('web/' + name for name in referenced)):
        assert archive.extract(files[path]) == (desktop / path).read_bytes(), path
    # Windows' secure mkdir mode can remove inherited ACLs in the sandbox.
    mkdir = os.mkdir
    os.mkdir = lambda path, mode=0o777, **kw: mkdir(path, 0o777 if mode == 0o700 else mode, **kw)
    tempfile.tempdir = str(root)
    with tempfile.TemporaryDirectory(prefix='pyz_replay_') as temp:
        bundle = Path(temp)
        for path in ('ths_board_frozen_members_v1.json', 'ths_board_verified_catalog_v1.json',
                     'ths_industry_source_seed_v1.json', 'ths_industry_source_seed_v1.json.gz',
                     'assistant_graph_worker.py', 'golden_stock_analysis.py', 'config.example.yaml', 'web/index.html'):
            if path in files:
                destination = bundle / path
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(archive.extract(files[path]))
        finder = CandidateModules(pyz, desktop, bundle)
        sys.meta_path.insert(0, finder)
        sys.path[:0] = [str(root / 'patrol_repair_20261111_v1'), str(desktop),
                       str(root / 'stargazer_ml/v5_hybrid/risk_r0_v1'),
                       str(root / 'stargazer_ml/v5_hybrid/green_common_profile')]
        sys.argv = [str(suite)]
        try:
            runpy.run_path(str(suite), run_name='__main__')
        finally:
            assert finder.loaded, 'candidate bytecode did not load'
            print('Candidate PYZ application modules loaded:', len(finder.loaded))


if __name__ == '__main__':
    main(Path(sys.argv[1]), Path(sys.argv[2]))
