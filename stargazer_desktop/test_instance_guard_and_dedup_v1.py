"""Step2 稳定性修复的纯逻辑测试（不启动监控、不触发指纹校验）。

覆盖：
1. instance_guard_v1：端口记录往返 + 跨进程单实例锁（父持锁 → 子被拦截）。
2. notifier 去重/回执持久化：load 修剪、save 原子、回执追加、Notifier 初始化加载。
"""
import importlib
import json
import os
import subprocess
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import instance_guard_v1
import notifier


def test_note_and_running_port_roundtrip():
    d = tempfile.mkdtemp()
    instance_guard_v1._base_dir = lambda: d
    try:
        instance_guard_v1.note_port(17890)
        assert instance_guard_v1.running_port() == 17890
        instance_guard_v1.note_port(17905)
        assert instance_guard_v1.running_port() == 17905
    finally:
        instance_guard_v1._base_dir = lambda: os.path.dirname(os.path.abspath(__file__))


def test_cross_process_instance_lock():
    """父进程持有排他锁时，子进程必须被拦截（BLOCKED）。"""
    d = tempfile.mkdtemp()
    instance_guard_v1._base_dir = lambda: d
    try:
        assert instance_guard_v1.acquire_instance_lock() is True  # 父拿到锁
        child = subprocess.run(
            [sys.executable, "-c",
             "import sys; sys.path.insert(0, sys.argv[1]); "
             "import instance_guard_v1; "
             "instance_guard_v1._base_dir = lambda: sys.argv[2]; "
             "print('HELD' if instance_guard_v1.acquire_instance_lock() else 'BLOCKED')",
             HERE, d],
            capture_output=True, text=True, timeout=30)
        assert child.returncode == 0, child.stderr
        assert "BLOCKED" in child.stdout, child.stdout + child.stderr
    finally:
        fh = instance_guard_v1._INSTANCE_FH
        if fh is not None:
            try:
                fh.close()
            except OSError:
                pass
        instance_guard_v1._INSTANCE_FH = None
        instance_guard_v1._base_dir = lambda: os.path.dirname(os.path.abspath(__file__))


def test_load_dedup_state_prunes():
    d = tempfile.mkdtemp()
    p = os.path.join(d, "push_state_v1.json")
    today = __import__("datetime").datetime.now().strftime("%Y-%m-%d")
    state = {
        "pushed": {"k1": time.time() - 10, "old": time.time() - 40 * 86400},
        "stop_fired": [f"{today}|600001|stop", "2000-01-01|600002|stop"],
        "fired": [f"{today} 09:05", "2000-01-01 09:05"],
    }
    with open(p, "w", encoding="utf-8") as f:
        json.dump(state, f)
    loaded = notifier.load_dedup_state(p)
    assert "k1" in loaded["pushed"] and "old" not in loaded["pushed"]
    assert f"{today}|600001|stop" in loaded["stop_fired"]
    assert "2000-01-01|600002|stop" not in loaded["stop_fired"]
    assert f"{today} 09:05" in loaded["fired"]
    assert "2000-01-01 09:05" not in loaded["fired"]


def test_save_dedup_state_atomic_and_roundtrip():
    d = tempfile.mkdtemp()
    p = os.path.join(d, "push_state_v1.json")
    state = {"pushed": {"k": 1.0}, "stop_fired": ["a"], "fired": ["b"]}
    notifier.save_dedup_state(p, state)
    assert os.path.exists(p) and not os.path.exists(p + ".tmp")
    with open(p, "r", encoding="utf-8") as f:
        assert json.load(f) == state


def test_record_push_receipt_appends():
    d = tempfile.mkdtemp()
    p = os.path.join(d, "push_state_v1.json")
    notifier.push_state_path = lambda: p
    log_dir = os.path.dirname(p)
    rp = os.path.join(log_dir, "push_receipts_v1.jsonl")
    if os.path.exists(rp):
        os.remove(rp)
    notifier.record_push_receipt("morning", "标题", True, "ok", True)
    notifier.record_push_receipt("stop_watch", "标题2", False, "disabled", False)
    with open(rp, "r", encoding="utf-8") as f:
        lines = [json.loads(x) for x in f.read().splitlines() if x.strip()]
    assert len(lines) == 2
    assert lines[0]["key"] == "morning" and lines[0]["ok"] is True
    assert lines[1]["key"] == "stop_watch" and lines[1]["ok"] is False
    notifier.push_state_path = importlib.import_module("notifier").push_state_path


def test_notifier_init_loads_state():
    d = tempfile.mkdtemp()
    p = os.path.join(d, "push_state_v1.json")
    today = __import__("datetime").datetime.now().strftime("%Y-%m-%d")
    with open(p, "w", encoding="utf-8") as f:
        json.dump({"pushed": {"k": time.time()}, "stop_fired": [f"{today}|600001|stop"],
                   "fired": [f"{today} 09:05"]}, f)
    notifier.push_state_path = lambda: p
    try:
        n = notifier.Notifier(get_cfg=lambda: {})
        assert n._pushed.get("k") is not None
        assert f"{today}|600001|stop" in n._stop_fired
        assert f"{today} 09:05" in n._fired
    finally:
        notifier.push_state_path = importlib.import_module("notifier").push_state_path


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for fn in fns:
        try:
            fn()
            print("PASS", fn.__name__)
        except Exception as e:
            failed += 1
            print("FAIL", fn.__name__, "->", repr(e))
    print("FAILED" if failed else "ALL_PASS", failed)
    raise SystemExit(1 if failed else 0)
