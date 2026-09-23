"""PostgreSQL 记录层（pgstore.py，第 3 步最小闭环）。

定位（规格 minimal_event_price_loop_v1）：
- SQLite（data/corvus.sqlite）= 采集缓冲（新闻原文/候选/行情观测），继续原样运行；
- PostgreSQL（config/pg.yaml）= 系统记录层：正式事件、实体映射、价格反应标签、
  重叠归因、标签运行账本。可重复审计、幂等重跑、多进程并发写。

铁律：
- 标签只追加不覆盖：同一 (event_id, entity_id, label_version) 重复生成时
  用 INSERT ... ON CONFLICT 更新 generated_at/字段，不留重复行；
- CENSORED_TAIL 永不写成 0 收益；
- 所有表 DDL 幂等（IF NOT EXISTS），迁移只加不减。
"""
from pathlib import Path

import psycopg
import yaml

_CFG_PATH = Path(__file__).resolve().parent.parent / "config" / "pg.yaml"


def pg_dsn(cfg_path=None):
    cfg = yaml.safe_load(open(str(cfg_path or _CFG_PATH), encoding="utf-8"))
    return ("host={host} port={port} dbname={dbname} user={user} "
            "password={password}").format(**cfg)


def get_pg_conn(cfg_path=None, autocommit=False):
    conn = psycopg.connect(pg_dsn(cfg_path), connect_timeout=15)
    conn.autocommit = autocommit
    return conn


# ---------------------------------------------------------------------------
# DDL（幂等）。字段语义见 spec/minimal_event_price_loop_v1.md。
# ---------------------------------------------------------------------------
_SCHEMA = """
CREATE TABLE IF NOT EXISTS information_event (
    event_id            BIGSERIAL PRIMARY KEY,
    candidate_id        INTEGER NOT NULL,             -- SQLite 候选表溯源
    raw_id              INTEGER NOT NULL,             -- SQLite raw_news 溯源
    title               TEXT,
    canonical_event_type TEXT NOT NULL,
    event_type_raw      TEXT,
    materiality_tier    TEXT NOT NULL CHECK (materiality_tier IN ('P0','P1','P2','UNKNOWN')),
    assertion_status    TEXT CHECK (assertion_status IN ('AFFIRMED','NEGATED','UNCERTAIN','HISTORICAL_REFERENCE')),
    source_id           TEXT NOT NULL,
    source_level        TEXT,
    source_url          TEXT,
    body_hash           TEXT,                          -- 镜像/原文哈希，双来源证据链
    evidence_span       TEXT,                          -- 原文证据片段（≤200字）
    entities            JSONB,                         -- Qwen 实体原样保留
    action              TEXT,
    amount_value        DOUBLE PRECISION,
    amount_unit         TEXT,
    amount_type         TEXT,
    amount_status       TEXT CHECK (amount_status IN ('EXPLICIT','CALCULATED','AMBIGUOUS','MISSING')),
    amount_evidence     TEXT,
    official_scope      TEXT,
    effective_date_raw  TEXT,
    effective_date_normalized DATE,
    effective_date_status TEXT CHECK (effective_date_status IN ('EXPLICIT','INFERRED','CONFLICT','MISSING')),
    effective_date_evidence TEXT,
    published_at        TIMESTAMPTZ,
    available_at        TIMESTAMPTZ NOT NULL,          -- 市场何时知道（标签起点唯一依据）
    effective_at        TIMESTAMPTZ,                   -- 事件何时生效（不得替代 available_at）
    first_seen_at       TIMESTAMPTZ,
    event_version       INTEGER NOT NULL DEFAULT 1,    -- 修订/撤回版本链
    supersedes_event_id BIGINT REFERENCES information_event(event_id),
    model_eligible      BOOLEAN NOT NULL DEFAULT FALSE,
    data_status         TEXT NOT NULL DEFAULT 'COMPLETE'
                        CHECK (data_status IN ('COMPLETE','PARTIAL','STALE','FAILED','OOD')),
    created_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (raw_id, event_version)                     -- 幂等锚：同一原文同一版本只正式化一次
);
CREATE INDEX IF NOT EXISTS idx_event_available_at ON information_event(available_at);
CREATE INDEX IF NOT EXISTS idx_event_type ON information_event(canonical_event_type);
CREATE INDEX IF NOT EXISTS idx_event_tier ON information_event(materiality_tier);

CREATE TABLE IF NOT EXISTS event_entity_link (
    link_id             BIGSERIAL PRIMARY KEY,
    event_id            BIGINT NOT NULL REFERENCES information_event(event_id) ON DELETE CASCADE,
    entity_type         TEXT NOT NULL CHECK (entity_type IN
                        ('DIRECT_STOCK','SECTOR','MARKET','GLOBAL','MULTI_ENTITY','UNKNOWN_SCOPE')),
    entity_id           TEXT NOT NULL,                 -- 股票代码 / 板块名 / 'A_SHARE_MARKET' 等
    relation_type       TEXT NOT NULL DEFAULT 'PRIMARY'
                        CHECK (relation_type IN ('PRIMARY','AFFECTED','SECONDARY')),
    mapping_method      TEXT NOT NULL,                 -- RULE_CODE_OFFICIAL / RULE_NAME_MATCH /
                                                       -- BOARD_COMPONENT / LLM_CANDIDATE / MANUAL
    mapping_confidence  DOUBLE PRECISION,
    evidence            TEXT,                          -- 映射证据（原文片段/规则说明）
    created_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (event_id, entity_type, entity_id, mapping_method)
);
CREATE INDEX IF NOT EXISTS idx_link_entity ON event_entity_link(entity_type, entity_id);

CREATE TABLE IF NOT EXISTS event_price_reaction (
    reaction_id         BIGSERIAL PRIMARY KEY,
    event_id            BIGINT NOT NULL REFERENCES information_event(event_id) ON DELETE CASCADE,
    entity_type         TEXT NOT NULL,
    entity_id           TEXT NOT NULL,                 -- 该标签挂在哪个实体上（个股/板块/市场）
    label_start         TIMESTAMPTZ NOT NULL,          -- 价格起点：≥ available_at 的第一个可交易时点
    label_start_basis   TEXT NOT NULL CHECK (label_start_basis IN
                        ('PREMARKET_OPEN','INTRADAY_NEXT_MINUTE','POST_CLOSE_NEXT_DAY',
                         'WEEKEND_NEXT_DAY','RESUME_FIRST_TRADE')),
    entry_price         DOUBLE PRECISION,
    benchmark_market    TEXT,
    benchmark_sector    TEXT,
    -- 收益标签
    ret_30m             DOUBLE PRECISION,
    ret_to_close        DOUBLE PRECISION,
    ret_1d              DOUBLE PRECISION,
    ret_3d              DOUBLE PRECISION,
    ret_5d              DOUBLE PRECISION,
    -- 相对收益（异常收益 = 个股 - 基准）
    abnormal_ret_market_30m  DOUBLE PRECISION,
    abnormal_ret_market_1d   DOUBLE PRECISION,
    abnormal_ret_market_3d   DOUBLE PRECISION,
    abnormal_ret_market_5d   DOUBLE PRECISION,
    abnormal_ret_sector_30m  DOUBLE PRECISION,
    abnormal_ret_sector_1d   DOUBLE PRECISION,
    abnormal_ret_sector_3d   DOUBLE PRECISION,
    abnormal_ret_sector_5d   DOUBLE PRECISION,
    -- 路径标签
    mfe_30m  DOUBLE PRECISION, mfe_1d DOUBLE PRECISION, mfe_3d DOUBLE PRECISION, mfe_5d DOUBLE PRECISION,
    mae_30m  DOUBLE PRECISION, mae_1d DOUBLE PRECISION, mae_3d DOUBLE PRECISION, mae_5d DOUBLE PRECISION,
    gap_return            DOUBLE PRECISION,
    volatility_expansion  DOUBLE PRECISION,
    volume_shock          DOUBLE PRECISION,
    reversal_30m          DOUBLE PRECISION,
    reversal_1d           DOUBLE PRECISION,
    -- 反身性研究字段（只存现象，不贴结论）
    initial_direction     TEXT CHECK (initial_direction IN ('UP','DOWN','FLAT')),
    initial_shock_atr     DOUBLE PRECISION,
    liquidity_sweep_depth DOUBLE PRECISION,
    reclaim_vwap          BOOLEAN,
    reclaim_pre_event_price BOOLEAN,
    reversal_strength     DOUBLE PRECISION,
    reversal_speed        DOUBLE PRECISION,
    post_reversal_ret_3d  DOUBLE PRECISION,
    post_reversal_ret_5d  DOUBLE PRECISION,
    -- 状态与审计
    limit_locked   BOOLEAN,
    suspended      BOOLEAN,
    corp_action_during_horizon BOOLEAN,
    horizon_complete BOOLEAN NOT NULL DEFAULT FALSE,
    label_status   TEXT NOT NULL DEFAULT 'OK' CHECK (label_status IN ('OK','CENSORED_TAIL')),
    label_version  TEXT NOT NULL,
    price_data_version TEXT,
    source_hash    TEXT,
    code_version   TEXT,
    generated_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (event_id, entity_id, label_version)        -- 幂等锚
);
CREATE INDEX IF NOT EXISTS idx_reaction_event ON event_price_reaction(event_id);
CREATE INDEX IF NOT EXISTS idx_reaction_entity ON event_price_reaction(entity_type, entity_id);

CREATE TABLE IF NOT EXISTS event_overlap (
    overlap_id          BIGSERIAL PRIMARY KEY,
    event_id            BIGINT NOT NULL REFERENCES information_event(event_id) ON DELETE CASCADE,
    other_event_id      BIGINT NOT NULL REFERENCES information_event(event_id) ON DELETE CASCADE,
    entity_id           TEXT NOT NULL,                 -- 重叠发生的实体（个股代码等）
    overlap_window      TEXT,                          -- 描述重叠的时间窗
    overlap_event_count INTEGER NOT NULL,
    overlap_event_ids   JSONB NOT NULL,
    macro_overlap       BOOLEAN NOT NULL DEFAULT FALSE,
    sector_overlap      BOOLEAN NOT NULL DEFAULT FALSE,
    attribution_quality TEXT NOT NULL CHECK (attribution_quality IN ('CLEAN','MODERATE','HEAVY')),
    label_version       TEXT NOT NULL,
    generated_at        TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (event_id, other_event_id, entity_id, label_version)
);

CREATE TABLE IF NOT EXISTS event_label_run (
    run_id              BIGSERIAL PRIMARY KEY,
    label_version       TEXT NOT NULL,
    started_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
    finished_at         TIMESTAMPTZ,
    status              TEXT NOT NULL DEFAULT 'RUNNING'
                        CHECK (status IN ('RUNNING','SUCCESS','FAILED')),
    params              JSONB,                         -- 本次运行的冻结参数
    stats               JSONB,                         -- 各状态计数
    price_data_version  TEXT,
    code_version        TEXT,
    error               TEXT
);
"""


def init_pg_schema(conn):
    """建表（幂等）。只加不减；改结构走新迁移函数，禁止在 DDL 里 DROP。"""
    with conn.cursor() as cur:
        cur.execute(_SCHEMA)
    conn.commit()
