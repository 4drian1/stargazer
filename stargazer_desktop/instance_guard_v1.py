"""单实例保护（Step2）。

语义（按本轮稳定性要求）：
- 固定锁字节 + 生产实例身份：``instance.lock`` 首字节区域写入固定 magic；
  实例身份（pid/exe/argv/frozen/时间）写在独立的 ``instance_identity.json``，
  诊断始终可读。注意 Windows 实测：文件上存在任何区域锁时，其他句柄连
  未锁定区域也无法读取，因此身份绝不能放在被锁文件里。
- 跨进程排他：OS 级字节锁（Windows msvcrt.locking / POSIX flock），
  第二个实例 acquire 立即失败（不等待、不重试抢占）。
- 原实例退出后可重新启动：锁随进程句柄释放（含异常退出），残留文件不影响下次获取。
- PyInstaller 父子进程兼容：只允许主程序入口（server.py start）调用 acquire；
  invoke()/子进程从不调用；子进程默认不继承句柄（close_fds=True），不会误持锁。
- 旧 EXE 尚无新锁：旧版本从不创建/持有本锁，因此它既不会被本锁拦截，
  也不能依赖它来拦截开发实例——跨进程状态锁（cross_process_lock_v1）才是数据防线。
- 端口记录：note_port/running_port 记录当前实例的 HTTP 端口；
  server 侧“所有可用端口绑定失败 → 不启动监控”的接线使用这两个函数。

本模块只依赖标准库（延迟导入 paths/notifier），可被测试独立加载。
"""
import json
import os
import sys
import threading
import time

LOCK_FILENAME = "instance.lock"
IDENTITY_FILENAME = "instance_identity.json"
PORT_FILENAME = "instance_port.json"
LOCK_MAGIC = b"STARGAZER_INSTANCE_LOCK_V1"

_INSTANCE_FH = None          # 持锁句柄（保持打开 = 持续持锁）
_GUARD_LOCK = threading.Lock()


def _base_dir():
    """锁/身份/端口文件目录：优先应用数据目录（paths.data_file 的父目录），
    失败回退到本模块所在目录。测试可整体替换本函数。"""
    try:
        import paths
        p = paths.data_file(LOCK_FILENAME)
        return os.path.dirname(p)
    except Exception:
        return os.path.dirname(os.path.abspath(__file__))


def _lock_path():
    return os.path.join(_base_dir(), LOCK_FILENAME)


def _identity_path():
    return os.path.join(_base_dir(), IDENTITY_FILENAME)


def _port_path():
    return os.path.join(_base_dir(), PORT_FILENAME)


def _instance_identity():
    """生产实例身份：可读 JSON（不含随机混淆）。"""
    exe = sys.executable or ""
    frozen = bool(getattr(sys, "frozen", False))
    try:
        entry = sys.argv[0] if sys.argv else ""
    except Exception:
        entry = ""
    return {
        "pid": os.getpid(),
        "exe": exe,
        "entry": entry,
        "frozen": frozen,
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
    }


def _try_lock_fd(fd):
    """非阻塞锁住锁文件首字节；成功 True。Windows/POSIX 双实现。"""
    if os.name == "nt":
        import msvcrt
        os.lseek(fd, 0, os.SEEK_SET)
        msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
        return True
    import fcntl
    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    return True


def _write_identity_file():
    """把实例身份写入独立 JSON（tmp + replace 原子落盘）。"""
    payload = _instance_identity()
    tmp = _identity_path() + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False)
    os.replace(tmp, _identity_path())
    return payload


def read_holder_identity():
    """读取当前锁持有者身份（诊断用；不存在返回 None）。"""
    try:
        with open(_identity_path(), "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def acquire_instance_lock():
    """取得单实例锁：成功 True 并保持持有；已有实例 → False。

    非阻塞：绝不等待已有实例退出（由用户决定先退出旧实例）。
    """
    global _INSTANCE_FH
    with _GUARD_LOCK:
        if _INSTANCE_FH is not None:
            return True  # 本进程已持有
        try:
            os.makedirs(_base_dir(), exist_ok=True)
        except OSError:
            pass
        fd = None
        try:
            fd = os.open(_lock_path(), os.O_RDWR | os.O_CREAT, 0o644)
            _try_lock_fd(fd)
        except (OSError, ValueError):
            # 已有实例持有（或锁不可用）：读取持有者身份便于诊断后放弃。
            holder = read_holder_identity()
            if fd is not None:
                try:
                    os.close(fd)
                except OSError:
                    pass
            try:
                import notifier
                notifier.log(f"单实例锁获取失败（已有实例运行）: {holder}")
            except Exception:
                pass
            return False
        # 固定锁字节：锁文件内容固定 magic（持锁期间他人读不到，诊断走身份 JSON）。
        try:
            os.ftruncate(fd, 0)
            os.lseek(fd, 0, os.SEEK_SET)
            os.write(fd, LOCK_MAGIC)
        except OSError:
            pass
        try:
            _write_identity_file()
        except OSError:
            try:
                import notifier
                notifier.log("单实例锁已获取，但身份写入失败（风险留痕）")
            except Exception:
                pass
        _INSTANCE_FH = fd
        return True


def release_instance_lock():
    """释放单实例锁（正常退出路径调用；异常退出由 OS 兜底释放）。"""
    global _INSTANCE_FH
    with _GUARD_LOCK:
        fd = _INSTANCE_FH
        _INSTANCE_FH = None
        if fd is None:
            return
        try:
            if os.name == "nt":
                import msvcrt
                os.lseek(fd, 0, os.SEEK_SET)
                msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
            else:
                import fcntl
                fcntl.flock(fd, fcntl.LOCK_UN)
        except OSError:
            pass
        try:
            os.close(fd)
        except OSError:
            pass
        try:
            os.remove(_identity_path())  # 干净退出清掉身份；崩溃残留可作事后诊断
        except OSError:
            pass


def note_port(port):
    """记录当前实例绑定成功的 HTTP 端口（供诊断/二次启动提示）。"""
    payload = dict(_instance_identity())
    payload["port"] = int(port)
    tmp = _port_path() + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False)
    os.replace(tmp, _port_path())


def running_port():
    """返回记录中的实例端口（int 或 None）。"""
    try:
        with open(_port_path(), "r", encoding="utf-8") as f:
            data = json.load(f)
        port = data.get("port")
        return int(port) if port is not None else None
    except Exception:
        return None
