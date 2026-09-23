"""路径解析：兼容源码运行和 PyInstaller 打包(--onefile)两种情况。
打包后：只读资源(前端/STRATEGY.md)在 sys._MEIPASS 临时目录；
可写数据(config.yaml、watchlist)放在 exe 同级目录，保证升级/迁移不丢。"""
import os
import shutil
import sys

FROZEN = getattr(sys, "frozen", False)

if FROZEN:
    BUNDLE_DIR = sys._MEIPASS                 # 打包进来的只读资源
    APP_DIR = os.path.dirname(sys.executable)  # exe 所在目录（可写）
else:
    BUNDLE_DIR = os.path.dirname(os.path.abspath(__file__))
    APP_DIR = BUNDLE_DIR


def resource(name):
    """只读资源路径（web/、STRATEGY.md 等）。"""
    return os.path.join(BUNDLE_DIR, name)


def data_file(name, seed_from_bundle=True):
    """可写数据路径。打包模式下若 exe 旁没有，则从包内复制一份出来。"""
    p = os.path.join(APP_DIR, name)
    if FROZEN and seed_from_bundle and not os.path.exists(p):
        src = os.path.join(BUNDLE_DIR, name)
        if os.path.exists(src):
            os.makedirs(os.path.dirname(p), exist_ok=True)
            shutil.copy(src, p)
    return p
