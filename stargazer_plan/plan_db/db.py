"""数据库连接（plan_db/db.py）。

复用 corvus 的 pg.yaml 连接配置（localhost:5432 / 库 corvus），
所有新表建在独立 schema `stargazer_plan`（见 ddl.py）。
"""
from pathlib import Path

import psycopg
import yaml

# corvus/config/pg.yaml 为第 3 步启用的系统记录层连接配置
_CFG_PATH = Path(__file__).resolve().parent.parent.parent / "corvus" / "config" / "pg.yaml"


def pg_dsn(cfg_path=None):
    """读取 pg.yaml 拼 DSN（与 corvus.pgstore.pg_dsn 同口径）。"""
    cfg = yaml.safe_load(open(str(cfg_path or _CFG_PATH), encoding="utf-8"))
    return ("host={host} port={port} dbname={dbname} user={user} "
            "password={password}").format(**cfg)


def get_pg_conn(cfg_path=None, autocommit=False):
    conn = psycopg.connect(pg_dsn(cfg_path), connect_timeout=15)
    conn.autocommit = autocommit
    return conn
