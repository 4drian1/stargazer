# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import collect_submodules

hiddenimports = ['yaml', 'services', 'notifier', 'paths', 'ml_bridge', 'intraday_monitor', 'sector_pullback']
hiddenimports += collect_submodules('agent')
hiddenimports += collect_submodules('tools')
hiddenimports += collect_submodules('strategy')


a = Analysis(
    ['app.py'],
    pathex=[],
    binaries=[],
    datas=[('web', 'web'), ('STRATEGY.md', '.'), ('config.yaml', '.'), ('data', 'data')],
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='Stargazer_v5_9_1',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=['app.ico'],
)
