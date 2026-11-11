"""Batch-1 后端配置（按 interface_contract/backend_api_inventory_20261108.md v2 对齐）。

纪律：
- 安装目录必须明确指定（STARGAZER_DESKTOP_DIR，默认 d:/stargazer/stargazer_desktop）；
  现行计划按生产实际读取规则（trade_plan_v1.load_plan）读取，绝不自行猜路径。
- 旧 frozen_*.json 仅作为独立历史来源；不充当现行计划，不用当前文件充当多版本。
- 全球总览快照文件默认落在隔离目录；生产文件一律只读。
- PG 旧档案口令/查询一律来自环境变量，代码中不硬编码口令。

注意：本模块为 ui_next 新接口的独立命名配置桥接（避免与生产已有模块重名），
仅从环境变量读取，默认指向生产桌面目录。
"""
import os
from pathlib import Path

import paths  # 统一资源/安装目录解析（dev 与冻结 EXE 均正确；绝不指向 _MEI 临时解包目录）


def desktop_dir() -> Path:
    """明确指定的安装目录（生产数据根）。"""
    env = os.environ.get("STARGAZER_DESKTOP_DIR")
    return Path(env) if env else Path(paths.APP_DIR)


def plan_archive_dir(root=None) -> "Path | None":
    """真实历史归档目录（frozen_*.json，独立历史来源）。"""
    env = os.environ.get("STARGAZER_PLAN_ARCHIVE_DIR")
    if env:
        return Path(env)
    root = root or desktop_dir()
    for cand in (root / ".." / "stargazer_plan", root / "stargazer_plan"):
        if cand.is_dir():
            return cand
    return None


def global_overview_snapshot_path() -> Path:
    """全球总览 GET 读取的隔离行情快照（缺省=隔离目录，绝不指向生产文件）。"""
    env = os.environ.get("STARGAZER_GLOBAL_OVERVIEW_SNAPSHOT")
    if env:
        return Path(env)
    # 默认落到安装目录（或沙箱）下的 data/isolated/，绝不指向包内只读或临时解包(_MEI)目录。
    return Path(paths.APP_DIR) / "data" / "isolated" / "global_overview_snapshot.json"


def global_overview_max_age_hours() -> float:
    """快照超过此时长（小时）视为过期（STALE），决策不可用。"""
    try:
        return float(os.environ.get("STARGAZER_GLOBAL_OVERVIEW_MAX_AGE_HOURS", "24"))
    except ValueError:
        return 24.0


def is_demo_mode() -> bool:
    """样本值仅在显式演示模式（STARGAZER_DEMO_MODE=1）提供；默认全 MISSING。"""
    return os.environ.get("STARGAZER_DEMO_MODE", "").strip() == "1"


def sandbox_dir() -> "Path | None":
    """写入沙箱目录（STARGAZER_SANDBOX_DIR）。设置后宿主进入沙箱模式：
    paths.data_file 全量重定向到沙箱镜像，真实确认/保存逻辑只操作副本。"""
    env = os.environ.get("STARGAZER_SANDBOX_DIR")
    return Path(env) if env else None


def pg_dsn() -> "str | None":
    dsn = os.environ.get("STARGAZER_PG_DSN")
    if dsn:
        return dsn
    host = os.environ.get("STARGAZER_PG_HOST")
    if not host:
        return None
    port = os.environ.get("STARGAZER_PG_PORT", "5432")
    db = os.environ.get("STARGAZER_PG_DB", "corvus")
    user = os.environ.get("STARGAZER_PG_USER", "corvus")
    pw = os.environ.get("STARGAZER_PG_PASSWORD", "")
    return f"host={host} port={port} dbname={db} user={user} password={pw}"


def pg_plan_version_query() -> "str | None":
    return os.environ.get("STARGAZER_PG_PLAN_VERSION_QUERY")
