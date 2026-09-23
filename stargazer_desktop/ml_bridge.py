"""Local subprocess bridge between the desktop app and v5.9 models."""
from __future__ import annotations

import glob
import json
import os
import subprocess
import threading
from datetime import datetime

import paths


_lock = threading.Lock()
_running = False
_last_error = None
_refresh_running = False
_refresh_error = None
_refresh_last_at = None


def _settings(cfg):
    model = cfg.get("model") or {}
    # 脚本路径：优先用 config.yaml 里的值，否则从 paths.APP_DIR 推导同级 stargazer_ml
    _ml_root = os.path.join(os.path.dirname(paths.APP_DIR), "stargazer_ml")
    _pullback = os.path.join(_ml_root, "pullback_train")
    import sys as _sys
    script = model.get("batch_script") or os.path.join(_ml_root, "v5_9_live_batch.py")
    refresh_script = model.get("refresh_script")
    if not refresh_script and script:
        # 未配置时按环境自动选择刷新脚本：
        # 有 data_daily_parts 用增量刷新（本地），没有就用 portable 全量刷新
        d = os.path.dirname(script)
        data_root = os.environ.get("STOCKDB_DATA_ROOT", "D:/a股数据")
        parts_dir = os.path.join(data_root, "data_daily_parts")
        has_parts = os.path.isdir(parts_dir) and glob.glob(os.path.join(parts_dir, "part_*.parquet"))
        refresh_script = os.path.join(d,
            "v5_9_refresh_live_data.py" if has_parts else "v5_9_portable_refresh.py")
    return {
        "enabled": model.get("enabled", True),
        "python": model.get("python") or _sys.executable,
        "script": script,
        "refresh_script": refresh_script,
        # T5 影子打分脚本（stargazer_ml/pullback_train），盘后随主模型一起刷新
        "shadow_script": model.get(
            "shadow_script",
            os.path.join(_pullback, "shadow_batch.py")),
        "device": model.get("device", "auto"),
        "top_n": int(model.get("top_n", 50)),
        "cache": paths.data_file(model.get("cache_dir", "data/model"), seed_from_bundle=False),
        "home": model.get("home"),
        "stockdb_root": model.get("stockdb_root"),
    }


def _read_json(path, default):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return default


def _decode_output(raw: bytes) -> str:
    """子进程输出可能是 UTF-8（PYTHONIOENCODING）或 GBK（Windows 默认控制台）。
    依次尝试解码，避免中文路径/错误信息变成乱码。"""
    for enc in ("utf-8", "gbk"):
        try:
            return raw.decode(enc)
        except (UnicodeDecodeError, LookupError):
            continue
    return raw.decode("utf-8", errors="replace")


def status(cfg):
    settings = _settings(cfg)
    summary = _read_json(os.path.join(settings["cache"], "current_summary.json"), {})
    result = {
        "enabled": bool(settings["enabled"]),
        "running": _running,
        "last_error": _last_error,
        "cache_ready": bool(summary),
        "uses_kimi": False,
        "refresh_running": _refresh_running,
        "refresh_error": _refresh_error,
        "refresh_last_at": _refresh_last_at,
    }
    result.update(summary)
    return result


def _norm_sig(v):
    """信号日归一化为纯数字字符串（历史行 sig_date 存在 TEXT/INT 两种存储）。"""
    s = str(v).strip()
    return str(int(s)) if s.isdigit() else s


def _annotate_t5_actions(cfg, rows):
    """把决策表已录入的人工操作挂到建议行（sig_date+code+dp 精确匹配，评审第2点）。

    同一只票同一天有多个决策点，绝不能只按 code 关联；建议行没有对应
    决策行时保持 None（按钮可点），已录入的行展示 buy/skip 状态。
    """
    if not rows:
        return
    import sqlite3
    try:
        conn = sqlite3.connect(_decisions_db(cfg), timeout=20)
        tabs = {r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'")}
        if "shadow_decisions" not in tabs:
            conn.close(); return
        cols = {r[1] for r in conn.execute("PRAGMA table_info(shadow_decisions)")}
        extra = ["fill_price"] + (["exit_price"] if "exit_price" in cols else [])
        cur = conn.execute(
            f"SELECT sig_date, code, dp, action, {','.join(extra)} "
            "FROM shadow_decisions WHERE action IS NOT NULL")
        acted = {}
        for sig_date, code, dp, action, *rest in cur.fetchall():
            info = {"action": action}
            for k, v in zip(extra, rest):
                info[k] = v
            acted[f"{_norm_sig(sig_date)}|{str(code)}|{int(dp)}"] = info
        conn.close()
    except Exception:
        return
    for r in rows:
        try:
            key = f"{_norm_sig(r.get('date'))}|{str(r.get('code'))}|{int(r.get('dp'))}"
        except (TypeError, ValueError):
            continue
        info = acted.get(key)
        if info:
            r["action"] = info["action"]
            if info.get("fill_price") is not None:
                r["fill_price"] = info["fill_price"]
            if info.get("exit_price") is not None:
                r["exit_price"] = info["exit_price"]


def _json_safe(v):
    """递归把 NaN/Infinity 清洗为 None。

    模型落盘的 advice JSON 由 Python json.dump 写出，默认 allow_nan=True，
    会出现裸 NaN；浏览器 JSON.parse 对 NaN 直接抛错，会让整个
    /api/model/advice 响应作废。出口统一转成合法 JSON（null）。
    """
    import math
    if isinstance(v, float):
        return v if math.isfinite(v) else None
    if isinstance(v, dict):
        return {k: _json_safe(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_json_safe(x) for x in v]
    return v


def advice(cfg, limit=20):
    settings = _settings(cfg)
    rows = _read_json(os.path.join(settings["cache"], "horizontal_advice.json"), [])
    g0_rows = _read_json(os.path.join(settings["cache"], "g0_advice.json"), [])
    t5_payload = _read_json(os.path.join(settings["cache"], "t5_shadow_advice.json"), {})
    market = _read_json(os.path.join(settings["cache"], "market_advice.json"), [])
    n = max(1, int(limit))
    t5_rows = (t5_payload.get("rows") or [])[:n]
    _annotate_t5_actions(cfg, t5_rows)
    return _json_safe({"horizontal": rows[:n], "g0_shadow": g0_rows[:n],
            "t5_shadow": t5_rows,
            "t5_model_version": t5_payload.get("model_version"),
            "t5_as_of": t5_payload.get("as_of"),
            "market": market, "status": status(cfg)})


def _decisions_db(cfg):
    """影子决策留痕库，与 intraday_monitor 的分钟快照同库。

    惰性导入：intraday_monitor 顶层 import ml_bridge，反向顶层导入会循环。
    """
    import intraday_monitor
    return intraday_monitor._db_path(cfg)


def _run_shadow(settings, env, cfg):
    """盘后用冻结四头（AM-R1/PM-R2/AM-O4/PM-O6）刷新 t5_shadow_advice.json。

    影子评分独立于 v5.9 主模型：主模型失败不阻塞影子，影子失败也不覆盖
    主模型错误；错误写入 last_shadow.log 并返回错误摘要。
    """
    script = settings.get("shadow_script")
    if not script or not os.path.isfile(script):
        return "影子脚本不存在"
    command = [settings["python"], script, "--output-dir", settings["cache"],
               "--decisions-db", _decisions_db(cfg)]
    try:
        completed = subprocess.run(
            command,
            cwd=os.path.dirname(script), env=env, capture_output=True,
            timeout=1800,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0), check=False,
        )
        log_path = os.path.join(settings["cache"], "last_shadow.log")
        with open(log_path, "w", encoding="utf-8") as f:
            f.write(_decode_output(completed.stdout or b""))
            if completed.stderr:
                f.write("\n[stderr]\n")
                f.write(_decode_output(completed.stderr))
        if completed.returncode:
            return (_decode_output(completed.stderr)
                    or _decode_output(completed.stdout)
                    or "shadow process failed")[-2000:]
        return None
    except Exception as exc:
        return f"{type(exc).__name__}: {exc}"


def _run(cfg, as_of):
    global _running, _last_error
    settings = _settings(cfg)
    os.makedirs(settings["cache"], exist_ok=True)
    command = [settings["python"], settings["script"], "--output-dir", settings["cache"],
               "--top-n", str(settings["top_n"]), "--device", settings["device"]]
    if as_of:
        command.extend(["--as-of", str(as_of)])
    try:
        env = os.environ.copy()
        if settings.get("home"):
            env["STARGAZER_HOME"] = settings["home"]
        if settings.get("stockdb_root"):
            env["STOCKDB_ROOT"] = settings["stockdb_root"]
        completed = subprocess.run(
            command, cwd=os.path.dirname(settings["script"]), env=env, capture_output=True,
            timeout=3600,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0), check=False,
        )
        with open(os.path.join(settings["cache"], "last_run.log"), "w", encoding="utf-8") as f:
            f.write(_decode_output(completed.stdout or b""))
            if completed.stderr:
                f.write("\n[stderr]\n")
                f.write(_decode_output(completed.stderr))
        if completed.returncode:
            _last_error = (_decode_output(completed.stderr) or _decode_output(completed.stdout)
                           or "model process failed")[-2000:]
        else:
            _last_error = None
        # 主模型结束后刷新 T5 影子建议（失败不覆盖主模型状态）
        shadow_err = _run_shadow(settings, env, cfg)
        if shadow_err:
            _last_error = (_last_error + " | shadow: " + shadow_err)[:2000] \
                if _last_error else f"shadow: {shadow_err}"
    except Exception as exc:
        _last_error = f"{type(exc).__name__}: {exc}"
    finally:
        with _lock:
            _running = False


def start(cfg, as_of=None):
    global _running, _last_error
    settings = _settings(cfg)
    if not settings["enabled"]:
        return {"ok": False, "error": "本地模型已禁用"}
    if not os.path.isfile(settings["python"]):
        return {"ok": False, "error": f"模型 Python 不存在: {settings['python']}"}
    if not os.path.isfile(settings["script"]):
        return {"ok": False, "error": f"模型入口不存在: {settings['script']}"}
    with _lock:
        if _running:
            return {"ok": True, "running": True, "message": "模型正在生成建议，请勿重复启动"}
        _running = True
        _last_error = None
    threading.Thread(target=_run, args=(cfg, as_of), daemon=True,
                     name="v5.9-local-model").start()
    return {"ok": True, "running": True, "started_at": datetime.now().isoformat(timespec="seconds"),
            "uses_kimi": False}


def _run_refresh(cfg):
    global _refresh_running, _refresh_error, _refresh_last_at
    settings = _settings(cfg)
    script = settings.get("refresh_script")
    if not script:
        _refresh_error = "未配置 model.refresh_script"
        _refresh_running = False
        return
    if not os.path.isfile(script):
        _refresh_error = f"刷新脚本不存在: {script}"
        _refresh_running = False
        return
    env = os.environ.copy()
    if settings.get("home"):
        env["STARGAZER_HOME"] = settings["home"]
    if settings.get("stockdb_root"):
        env["STOCKDB_ROOT"] = settings["stockdb_root"]
    try:
        completed = subprocess.run(
            [settings["python"], script], cwd=os.path.dirname(script), env=env,
            capture_output=True,
            timeout=7200, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0), check=False,
        )
        log_path = os.path.join(settings["cache"], "last_refresh.log")
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        with open(log_path, "w", encoding="utf-8") as f:
            f.write(_decode_output(completed.stdout or b""))
            if completed.stderr:
                f.write("\n[stderr]\n")
                f.write(_decode_output(completed.stderr))
        if completed.returncode:
            tail = (_decode_output(completed.stderr) or _decode_output(completed.stdout)
                    or "refresh process failed")[-2000:]
            _refresh_error = tail
        else:
            _refresh_error = None
    except Exception as exc:
        _refresh_error = f"{type(exc).__name__}: {exc}"
    finally:
        _refresh_last_at = datetime.now().isoformat(timespec="seconds")
        _refresh_running = False


def refresh(cfg):
    global _refresh_running, _refresh_error
    settings = _settings(cfg)
    if not settings.get("refresh_script"):
        return {"ok": False, "error": "未配置 model.refresh_script"}
    if not os.path.isfile(settings["refresh_script"]):
        return {"ok": False, "error": f"刷新脚本不存在: {settings['refresh_script']}"}
    with _lock:
        if _refresh_running:
            return {"ok": True, "running": True, "message": "数据刷新已在进行中"}
        _refresh_running = True
        _refresh_error = None
    threading.Thread(target=_run_refresh, args=(cfg,), daemon=True,
                     name="v5.9-refresh-live").start()
    return {"ok": True, "running": True, "started_at": datetime.now().isoformat(timespec="seconds")}
