# -*- mode: python ; coding: utf-8 -*-


a = Analysis(
    ['app.py'],
    pathex=[],
    binaries=[],
    datas=[('web', 'web'), ('STRATEGY.md', '.'), ('config.yaml', '.'), ('data', 'data')],
    hiddenimports=['yaml', 'services', 'notifier', 'paths', 'ml_bridge', 'intraday_monitor', 'agent.brain', 'tools.financials', 'tools.lhb', 'tools.market', 'tools.news', 'tools.research', 'strategy.exits', 'strategy.funnel', 'strategy.oversold', 'strategy.scorer'],
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
    name='Stargazer',
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
