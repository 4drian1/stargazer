# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for Stargazer desktop app.

关键点：
- shadow_log / shadow_batch 动态导入，需要 pathex + hiddenimports + data 三重覆盖
- openai 延迟导入，PyInstaller 静态分析可能漏检，显式声明
- stargazer_ml/pullback_train 作为 data 打入，运行时由 _shadow_log() 从 BUNDLE_DIR 查找
"""
from PyInstaller.utils.hooks import collect_submodules
import os

_ROOT = os.path.dirname(os.path.abspath(SPEC))            # stargazer_desktop
_PROJECT = os.path.dirname(_ROOT)                          # d:\stargazer
_ML_PULLBACK = os.path.join(_PROJECT, "stargazer_ml", "pullback_train")

hiddenimports = [
    'yaml', 'services', 'notifier', 'paths', 'ml_bridge', 'plan_bridge',
    'intraday_monitor', 'pentacles_demo', 'fmt', 'openai', 'pywebview',
]
hiddenimports += collect_submodules('agent')
hiddenimports += collect_submodules('tools')
hiddenimports += collect_submodules('strategy')
# shadow_log 在 stargazer_ml/pullback_train 下，动态 sys.path 导入
hiddenimports.append('shadow_log')

a = Analysis(
    ['app.py'],
    pathex=[_ML_PULLBACK, _ROOT],
    binaries=[],
    datas=[
        ('web', 'web'),
        ('STRATEGY.md', '.'),
        ('config.yaml', '.'),
        ('data', 'data'),
        # 注意：不把 pullback_train 打入包（该目录 15GB，且
        # _score_live_signals 从磁盘相对路径 <exe>\..\stargazer_ml 读脚本，
        # 不走包内资源；打入会让单文件超 4GB 直接打包失败）。
    ],
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=['torch', 'torchvision', 'transformers', 'tensorflow',
              'matplotlib', 'IPython', 'jupyter', 'notebook'],
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
