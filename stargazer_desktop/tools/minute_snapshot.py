"""盘中分钟快照采集（绿盘低吸 ML 训练数据）。

目的：盘中规则触发时保存分钟级现场，供后续训练"盘中时机模型"。
四组样本全存，避免训练集只有正向样本的选择偏差：
- trigger   通过完整规则筛选的触发候选
- near_miss 差一点触发（过了预筛但复验未过，或跌幅 -2.5%~-3%）
- control   同板块同时刻随机对照股（1~2 只）
- rejected  规则通过但模型期望值 EV<=0 / 被风险否决的候选（extra 记原因）

数据源：stockdb 本地服务（分钟k 表，date 为 14 位 int YYYYMMDDHHMMSS）。
落盘：OUT_ROOT/{yyyymmdd}/{code}_{HHMMSS}_{trigger}_{group}.json + index.csv。
T+1 路径由 minute_snapshot_backfill.py 每日回填（t1_filled 列标记）。

硬性约束：任何失败静默吞掉，绝不阻塞扫描/推送主流程。
"""
from __future__ import annotations

import csv
import json
import os
import random
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

import requests

BASE_URL = os.environ.get("STOCKDB_URL", "http://127.0.0.1:7899")
OUT_ROOT = os.environ.get(
    "MINUTE_SNAPSHOT_DIR",
    "D:/stargazer/stargazer_ml/pullback_train/minute_snapshots",
)

_GROUPS = ("trigger", "near_miss", "control", "rejected")
_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="minsnap")
_local = threading.local()

# 每日采集健康计数（评审要求：程序曾存在但一直没落盘，必须有健康日志可发现）。
# health_{yyyymmdd}.json: {"attempt","saved","failed","empty_bars","dedup","async_submit"}
_health_lock = threading.Lock()


def _bump(day: str, key: str, by: int = 1) -> None:
    """原子式累计当日健康计数；任何失败静默（与全模块约束一致）。"""
    try:
        os.makedirs(OUT_ROOT, exist_ok=True)
        path = os.path.join(OUT_ROOT, f"health_{day}.json")
        with _health_lock:
            stat = {}
            if os.path.exists(path):
                try:
                    with open(path, encoding="utf-8") as f:
                        stat = json.load(f)
                except Exception:
                    stat = {}
            stat[key] = int(stat.get(key, 0)) + by
            tmp = path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(stat, f, ensure_ascii=False)
            os.replace(tmp, path)
    except Exception:
        pass


def health_report(day: str | None = None) -> dict:
    """读取某日健康计数（默认今天）；文件缺失返回 {}（=当日零采集）。"""
    day = day or datetime.now().strftime("%Y%m%d")
    path = os.path.join(OUT_ROOT, f"health_{day}.json")
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}

# 同一(代码,分钟)去重，避免每个检查点重复存
_seen: set[str] = set()
_seen_lock = threading.Lock()


def _index_path(day: str) -> str:
    return os.path.join(OUT_ROOT, day, "index.csv")


def _append_index(day: str, row: dict) -> None:
    path = _index_path(day)
    new = not os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["datetime", "date", "code", "sector",
                                          "trigger", "group", "n_bars", "file",
                                          "t1_filled"])
        if new:
            w.writeheader()
        w.writerow(row)


def fetch_minute_bars(code: str, yyyymmdd: str, timeout: int = 10) -> list[dict]:
    """从 stockdb 拉某日 1 分钟K，返回按时间排序的 bar 列表（失败返回 []）。"""
    try:
        r = requests.get(f"{BASE_URL}/?cmd=get&t=分钟k:{code}:{yyyymmdd}*",
                         timeout=timeout)
        if r.status_code != 200:
            return []
        data = r.json()
    except Exception:
        return []
    out = []
    for item in data or []:
        rec = item[1] if isinstance(item, list) and len(item) >= 2 else (
            item if isinstance(item, dict) else None)
        if not isinstance(rec, dict) or rec.get("date") is None:
            continue
        out.append({
            "dt": str(rec["date"])[8:12],          # HHMM
            "open": rec.get("open"), "high": rec.get("high"),
            "low": rec.get("low"), "close": rec.get("close"),
            "volume": rec.get("volume"), "amount": rec.get("amount"),
        })
    out.sort(key=lambda b: b["dt"])
    return out


def snapshot(code: str, sector: str, trigger: str, group: str,
             context: dict | None = None, extra: dict | None = None,
             bars: list | None = None) -> str | None:
    """同步保存一次快照。返回文件路径；任何异常返回 None（静默）。"""
    if group not in _GROUPS:
        group = "trigger"
    now = datetime.now()
    day = now.strftime("%Y%m%d")
    hm = now.strftime("%H%M")
    dedup_key = f"{day}:{code}:{hm}:{group}"
    with _seen_lock:
        if dedup_key in _seen:
            _bump(day, "dedup")
            return None
        _seen.add(dedup_key)
    _bump(day, "attempt")
    try:
        if bars is None:
            bars = fetch_minute_bars(code, day)
        payload = {
            "code": code, "date": day, "time": hm,
            "sector": sector, "trigger": trigger, "group": group,
            "saved_at": now.isoformat(timespec="seconds"),
            "n_bars": len(bars),
            "context": context or {},       # 当时大盘/板块/龙头/中军状态（调用方传入）
            "extra": extra or {},           # rejected 原因、EV、规则命中细节等
            "bars": bars,                   # 截至保存时当日全部可见分钟线（触发前>=60根）
            "t1_bars": None,                # 由 backfill 脚本回填 T+1 路径
        }
        os.makedirs(os.path.join(OUT_ROOT, day), exist_ok=True)
        fname = f"{code}_{hm}_{trigger}_{group}.json"
        fpath = os.path.join(OUT_ROOT, day, fname)
        with open(fpath, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False)
        _append_index(day, {
            "datetime": now.strftime("%Y-%m-%d %H:%M:%S"),
            "date": day, "code": code, "sector": sector,
            "trigger": trigger, "group": group,
            "n_bars": len(bars), "file": fname, "t1_filled": 0,
        })
        _bump(day, "saved")
        if not bars:
            _bump(day, "empty_bars")
        return fpath
    except Exception:
        _bump(day, "failed")
        return None


def snapshot_async(code: str, sector: str, trigger: str, group: str,
                   context: dict | None = None, extra: dict | None = None,
                   bars: list | None = None) -> None:
    """异步保存（线程池），绝不抛异常到调用方。"""
    try:
        _bump(datetime.now().strftime("%Y%m%d"), "async_submit")
        _executor.submit(snapshot, code, sector, trigger, group,
                         context, extra, bars)
    except Exception:
        pass


def pick_controls(codes: list[str], exclude: set[str], n: int = 2,
                  seed: str = "") -> list[str]:
    """同板块同时刻随机对照：从候选池剔除已触发/预筛股后随机取 n 只。"""
    pool = [c for c in codes if c not in exclude]
    if not pool:
        return []
    rnd = random.Random(f"{seed}:{datetime.now():%Y%m%d%H%M}")
    rnd.shuffle(pool)
    return pool[:n]


def flush() -> None:
    """等待线程池排空（供进程退出前调用，可选）。"""
    try:
        _executor.shutdown(wait=False)
    except Exception:
        pass
