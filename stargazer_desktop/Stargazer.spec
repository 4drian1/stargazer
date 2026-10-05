# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import collect_all

datas = [('web', 'web'), ('STRATEGY.md', '.'), ('config.yaml', '.'), ('data', 'data'), ('C:/Users/22107/AppData/Local/Python/pythoncore-3.14-64/Lib/site-packages/akshare/file_fold', 'akshare/file_fold')]
binaries = []
hiddenimports = ['yaml', 'services', 'notifier', 'paths', 'ml_bridge', 'plan_bridge', 'intraday_monitor', 'agent.brain', 'tools.financials', 'tools.lhb', 'tools.market', 'tools.news', 'tools.research', 'strategy.exits', 'strategy.funnel', 'strategy.oversold', 'strategy.scorer', 'info_panel', 'psycopg', 'pyarrow', 'akshare.stock_feature.stock_info', 'akshare.index.index_research_sw', 'akshare.index.index_stock_us_sina', 'akshare.futures.futures_foreign']
tmp_ret = collect_all('py_mini_racer')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]


a = Analysis(
    ['app.py'],
    pathex=[],
    binaries=binaries,
    datas=datas,
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
