"""SQLite 持久层（先 SQLite，字段按 PostgreSQL 设计）。

铁律：
- 一切记录带 published_at / collected_at / available_at，分开存，统一带时区 ISO8601。
- 缺失数据存 NULL，绝不填 0。
- 原始数据只追加、不覆盖、不修改。
"""
import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "corvus.sqlite"

_SCHEMA = """
-- 原始新闻：正文不进 SQLite，完整原文按 raw_path 落盘（只追加，文件名含 raw_id）
CREATE TABLE IF NOT EXISTS raw_news (
    raw_id            INTEGER PRIMARY KEY AUTOINCREMENT,
    source_id         TEXT NOT NULL,
    title             TEXT,
    url               TEXT,
    raw_text_hash     TEXT NOT NULL,   -- 落盘原文的 sha256，verify_raw_news() 校验用
    published_at      TEXT,            -- 解析后带时区 ISO8601；解析失败存 NULL
    collected_at      TEXT NOT NULL,   -- 本系统采集时间（带时区 ISO8601）
    raw_path          TEXT,            -- 原文路径 data/raw/news/{source_id}/{YYYYMMDD}/{raw_id}.json
    content_license   TEXT,
    mime              TEXT,            -- 原文类型 application/json / text/html
    encoding          TEXT,            -- 原文编码 utf-8
    source_time_raw   TEXT,            -- 来源原始时间文本（未解析，留审计）
    tz_inference      TEXT,            -- 时区推断方式，如 naive->+08:00 按北京时间解释
    time_parse_status TEXT CHECK (time_parse_status IN ('OK','PARTIAL','LEGACY_MISSING')),
    detail_status     TEXT CHECK (detail_status IN ('OK','FAILED','PENDING','TODO')),
    detail_mime       TEXT,            -- 详情原文类型 application/pdf / text/html
    data_status       TEXT NOT NULL DEFAULT 'COMPLETE'
                      CHECK (data_status IN ('COMPLETE','PARTIAL','STALE','FAILED','OOD'))
);

-- 采集运行账本：发现"程序正常退出但某源三天零产出"
CREATE TABLE IF NOT EXISTS collection_run (
    run_id              INTEGER PRIMARY KEY AUTOINCREMENT,
    source_id           TEXT NOT NULL,
    started_at          TEXT NOT NULL,
    finished_at         TEXT,
    status              TEXT CHECK (status IN ('SUCCESS','PARTIAL','FAILED')),
    fetched_count       INTEGER,
    inserted_count      INTEGER,
    duplicate_count     INTEGER,
    failed_count        INTEGER,
    latest_published_at TEXT,
    error_summary       TEXT,
    collector_version   TEXT NOT NULL DEFAULT '1'
);

-- 规格 4.0 schema（含版本与撤回字段）
-- 事件候选：Qwen 事实抽取 + 规则校验产物，只有人工接受才晋升 information_event
CREATE TABLE IF NOT EXISTS information_event_candidate (
    candidate_id   INTEGER PRIMARY KEY AUTOINCREMENT,
    raw_id         INTEGER NOT NULL REFERENCES raw_news(raw_id),
    source_id      TEXT NOT NULL,
    event_type     TEXT,               -- Qwen 原始描述（event_type_raw，保留原值）
    canonical_event_type TEXT,         -- 标准类别（event_taxonomy）
    materiality_tier     TEXT CHECK (materiality_tier IN ('P0','P1','P2','UNKNOWN') OR materiality_tier IS NULL),
    mapping_status       TEXT CHECK (mapping_status IN ('MAPPED','UNMAPPED_CONTEXT') OR mapping_status IS NULL),
    model_eligible       INTEGER NOT NULL DEFAULT 0,  -- 1=可进事件影响模型（证据约束全过）
    is_container_document INTEGER NOT NULL DEFAULT 0,
    underlying_event_types TEXT,       -- JSON 数组：容器文档提示性底层事件族
    entities       TEXT,               -- JSON 数组
    action         TEXT,
    amount         REAL,
    amount_raw_text TEXT,
    amount_unit    TEXT,
    amount_type    TEXT,
    amount_status  TEXT CHECK (amount_status IN ('EXPLICIT','CALCULATED','AMBIGUOUS','MISSING') OR amount_status IS NULL),
    amount_evidence TEXT,
    official_scope TEXT,
    effective_date TEXT,
    effective_date_raw       TEXT,
    effective_date_normalized TEXT,
    effective_date_status    TEXT CHECK (effective_date_status IN ('EXPLICIT','INFERRED','CONFLICT','MISSING') OR effective_date_status IS NULL),
    effective_date_evidence  TEXT,
    assertion_status TEXT CHECK (assertion_status IN ('AFFIRMED','NEGATED','UNCERTAIN','HISTORICAL_REFERENCE') OR assertion_status IS NULL),
    extract_raw    TEXT,               -- Qwen 原始输出
    parse_ok       INTEGER NOT NULL DEFAULT 0,
    latency_s      REAL,
    model_version  TEXT NOT NULL DEFAULT 'qwen3-4b-q4_k_m',
    rule_check     TEXT,               -- JSON：每条规则的通过情况
    candidate_status TEXT NOT NULL DEFAULT 'PENDING'
                   CHECK (candidate_status IN ('PENDING','SCHEMA_VALIDATED','HUMAN_ACCEPTED','HUMAN_REJECTED','NEEDS_REVIEW','P2_ROUTED')),
    reviewed_by    TEXT,
    reviewed_at    TEXT,
    review_note    TEXT,
    created_at     TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS information_event (
    event_id            INTEGER PRIMARY KEY AUTOINCREMENT,
    canonical_event_id  TEXT NOT NULL,       -- 同一事件的唯一标识（跨版本共享）
    event_version       INTEGER NOT NULL DEFAULT 1,
    supersedes_event_id TEXT,                -- 修订/撤回时指向旧版本 event_id
    retracted_at        TEXT,                -- 撤回时间，NULL 表示未撤回
    effective_at        TEXT,                -- 政策生效时间（≠发布时间）
    first_seen_at       TEXT NOT NULL,       -- 本系统首次见到该事件的时间
    source_id           TEXT NOT NULL,
    source_level        TEXT NOT NULL CHECK (source_level IN ('OFFICIAL','AGGREGATOR','RUMOR')),
    source_url          TEXT,
    published_at        TEXT,
    collected_at        TEXT NOT NULL,
    available_at        TEXT,                -- 通过可见性校验后可被决策使用的时间
    language            TEXT,
    content_license     TEXT,
    mapping_version     TEXT NOT NULL DEFAULT 'v1',
    raw_id              INTEGER REFERENCES raw_news(raw_id),
    raw_text_hash       TEXT,
    evidence_ids        TEXT,                -- JSON 数组文本
    data_class          TEXT NOT NULL DEFAULT 'prospective_gold'
                        CHECK (data_class IN ('prospective_gold','historical_silver','historical_untrusted')),
    data_status         TEXT NOT NULL DEFAULT 'COMPLETE'
                        CHECK (data_status IN ('COMPLETE','PARTIAL','STALE','FAILED','OOD')),
    -- 【事实抽取层 —— Qwen 只出这些】
    event_type          TEXT,
    entities            TEXT,                -- JSON
    action              TEXT,
    amount              REAL,                -- 金额/比例等数值，无单位语义歧义时存原文由上层处理
    official_scope      TEXT,
    effective_date      TEXT,
    -- 【影响推断层 —— 由映射表/历史统计模型出，不由 LLM 拍脑袋】
    affected_sectors    TEXT,                -- JSON
    affected_stocks     TEXT,                -- JSON
    impact_direction    TEXT CHECK (impact_direction IN ('POSITIVE','NEGATIVE','NEUTRAL') OR impact_direction IS NULL),
    impact_magnitude    INTEGER CHECK (impact_magnitude IS NULL OR (impact_magnitude BETWEEN 1 AND 5)),
    impact_horizon      TEXT,
    confidence          REAL
);

CREATE TABLE IF NOT EXISTS market_observation (
    obs_id          INTEGER PRIMARY KEY AUTOINCREMENT,
    instrument_id   TEXT NOT NULL,
    event_time      TEXT NOT NULL,   -- 行情对应的时间（带时区 ISO8601）
    available_at    TEXT NOT NULL,   -- 按可见性规则计算出的可用时间
    open            REAL,            -- 缺失存 NULL，绝不填 0
    high            REAL,
    low             REAL,
    close           REAL,
    volume          REAL,
    source          TEXT NOT NULL,   -- fred / yfinance / akshare
    raw_hash        TEXT,
    quality_status  TEXT NOT NULL DEFAULT 'RAW'
                    CHECK (quality_status IN ('RAW','SUSPECT','EXCLUDED'))
);

CREATE TABLE IF NOT EXISTS global_context_snapshot (
    snapshot_id        TEXT PRIMARY KEY,   -- 如 SNAPSHOT_20260910_0930
    as_of_time         TEXT NOT NULL,      -- 北京时间
    us_zone            TEXT,               -- 各分区状态 JSON
    hk_zone            TEXT,
    jp_kr_zone         TEXT,
    fx_zone            TEXT,
    rates_zone         TEXT,
    commodity_zone     TEXT,
    domestic_zone      TEXT,
    regime_id          INTEGER,
    change_point_flag  INTEGER NOT NULL DEFAULT 0,  -- 0/1
    feature_version    TEXT NOT NULL DEFAULT 'v1'
);

CREATE TABLE IF NOT EXISTS information_risk_output (
    output_id    INTEGER PRIMARY KEY AUTOINCREMENT,
    snapshot_id  TEXT NOT NULL REFERENCES global_context_snapshot(snapshot_id),
    candidate_id TEXT,
    p_stop       REAL,
    p_loss_3d    REAL,
    p_loss_5d    REAL,
    risk_level   TEXT CHECK (risk_level IN ('R0','R1','R2','R3','R4','UNKNOWN') OR risk_level IS NULL),
    data_status  TEXT NOT NULL DEFAULT 'COMPLETE'
                 CHECK (data_status IN ('COMPLETE','PARTIAL','STALE','FAILED','OOD')),
    evidence_ids TEXT,   -- JSON 数组文本
    model_version TEXT NOT NULL DEFAULT 'v1'
);

-- Tarot 日级风险标签（冻结口径，规格：一日一行，不是几千只候选当几千个样本）
CREATE TABLE IF NOT EXISTS tarot_day_labels (
    trade_date                 TEXT NOT NULL,   -- 信号日 t
    snapshot_type              TEXT NOT NULL DEFAULT 'PREMARKET',
    strategy_id                TEXT NOT NULL,   -- tarot.hermit.a2@v1 / tarot.hermit.legacy@v1 / tarot.b0@v1
    candidate_generator_version TEXT NOT NULL,  -- 池文件+行数+日期范围指纹
    candidate_universe_hash    TEXT NOT NULL,   -- 当日 membership (code,date) 排序后 sha256
    candidate_count            INTEGER NOT NULL,
    entry_valid_rate           REAL,            -- t+1 开盘价可得比例（买不到/停牌识别）
    stop_rate_3d               REAL,            -- 池内 entry*(1-3%) 止损触碰率
    loss_rate_3d               REAL,
    loss_rate_5d               REAL,
    mean_net_return_3d         REAL,
    mean_net_return_5d         REAL,
    left_tail_return_5d        REAL,            -- 池 net_ret_5d 5% 分位
    n_valid_3d                 INTEGER,
    n_valid_5d                 INTEGER,
    cost_version               TEXT NOT NULL,   -- costs_v2_frozen
    label_version              TEXT NOT NULL,   -- tarot_day_labels@v1
    label_status               TEXT NOT NULL DEFAULT 'OK'
                               CHECK (label_status IN ('OK','CENSORED_TAIL')),
    label_frozen_at            TEXT NOT NULL,
    UNIQUE (trade_date, strategy_id, label_version)
);
"""

_INDEXES = """
CREATE INDEX IF NOT EXISTS idx_raw_news_published_at   ON raw_news(published_at);
CREATE INDEX IF NOT EXISTS idx_raw_news_collected_at   ON raw_news(collected_at);
CREATE INDEX IF NOT EXISTS idx_raw_news_source         ON raw_news(source_id);
CREATE INDEX IF NOT EXISTS idx_event_published_at      ON information_event(published_at);
CREATE INDEX IF NOT EXISTS idx_event_available_at      ON information_event(available_at);
CREATE INDEX IF NOT EXISTS idx_event_first_seen        ON information_event(first_seen_at);
CREATE INDEX IF NOT EXISTS idx_mkt_event_time          ON market_observation(event_time);
CREATE INDEX IF NOT EXISTS idx_mkt_available_at        ON market_observation(available_at);
CREATE INDEX IF NOT EXISTS idx_mkt_instrument_time     ON market_observation(instrument_id, event_time);
CREATE INDEX IF NOT EXISTS idx_run_source_started      ON collection_run(source_id, started_at);
CREATE UNIQUE INDEX IF NOT EXISTS ux_mkt_inst_time     ON market_observation(instrument_id, event_time);
"""

# 旧库增量迁移：ALTER TABLE ADD COLUMN（不 DROP 重建，不丢数据）
_COLUMN_MIGRATIONS = [
    ("raw_news", "mime", "ALTER TABLE raw_news ADD COLUMN mime TEXT"),
    ("raw_news", "encoding", "ALTER TABLE raw_news ADD COLUMN encoding TEXT"),
    ("raw_news", "source_time_raw", "ALTER TABLE raw_news ADD COLUMN source_time_raw TEXT"),
    ("raw_news", "tz_inference", "ALTER TABLE raw_news ADD COLUMN tz_inference TEXT"),
    ("raw_news", "time_parse_status",
     "ALTER TABLE raw_news ADD COLUMN time_parse_status TEXT CHECK (time_parse_status IN ('OK','PARTIAL','LEGACY_MISSING'))"),
    ("raw_news", "detail_status",
     "ALTER TABLE raw_news ADD COLUMN detail_status TEXT CHECK (detail_status IN ('OK','FAILED','PENDING','TODO'))"),
    ("raw_news", "detail_mime", "ALTER TABLE raw_news ADD COLUMN detail_mime TEXT"),
    ("raw_news", "data_status",
     "ALTER TABLE raw_news ADD COLUMN data_status TEXT NOT NULL DEFAULT 'COMPLETE' "
     "CHECK (data_status IN ('COMPLETE','PARTIAL','STALE','FAILED','OOD'))"),
    ("information_event_candidate", "canonical_event_type",
     "ALTER TABLE information_event_candidate ADD COLUMN canonical_event_type TEXT"),
    ("information_event_candidate", "materiality_tier",
     "ALTER TABLE information_event_candidate ADD COLUMN materiality_tier TEXT "
     "CHECK (materiality_tier IN ('P0','P1','P2') OR materiality_tier IS NULL)"),
    ("information_event_candidate", "is_container_document",
     "ALTER TABLE information_event_candidate ADD COLUMN is_container_document INTEGER NOT NULL DEFAULT 0"),
    ("information_event_candidate", "underlying_event_types",
     "ALTER TABLE information_event_candidate ADD COLUMN underlying_event_types TEXT"),
    ("raw_news", "body_source",
     "ALTER TABLE raw_news ADD COLUMN body_source TEXT"),
    ("raw_news", "body_status",
     "ALTER TABLE raw_news ADD COLUMN body_status TEXT "
     "CHECK (body_status IN ('FULL_TEXT','MIRROR_MATCHED','METADATA_ONLY','FAILED') OR body_status IS NULL)"),
    ("raw_news", "body_hash",
     "ALTER TABLE raw_news ADD COLUMN body_hash TEXT"),
    ("raw_news", "mirror_match_method",
     "ALTER TABLE raw_news ADD COLUMN mirror_match_method TEXT"),
    ("raw_news", "mirror_match_confidence",
     "ALTER TABLE raw_news ADD COLUMN mirror_match_confidence REAL"),
]

# 候选表两次历史重建的判定标记：
# 第 1 次：AUTO_VALIDATED → SCHEMA_VALIDATED（2026-09-12）
# 第 2 次：materiality_tier 增加 UNKNOWN + 证据约束四层字段 + P0 安全门字段（规格 v1.2，2026-09-13）
_REBUILD2_MARKER = "effective_date_status"


def _candidate_new_schema(name):
    """候选表 v1.2 完整建表语句（重建用）。"""
    return f"""
        CREATE TABLE {name} (
            candidate_id   INTEGER PRIMARY KEY AUTOINCREMENT,
            raw_id         INTEGER NOT NULL REFERENCES raw_news(raw_id),
            source_id      TEXT NOT NULL,
            event_type     TEXT,               -- Qwen 原始描述（event_type_raw，保留原值）
            canonical_event_type TEXT,         -- 标准类别（event_taxonomy）
            materiality_tier     TEXT CHECK (materiality_tier IN ('P0','P1','P2','UNKNOWN') OR materiality_tier IS NULL),
            mapping_status       TEXT CHECK (mapping_status IN ('MAPPED','UNMAPPED_CONTEXT') OR mapping_status IS NULL),
            model_eligible       INTEGER NOT NULL DEFAULT 0,  -- 1=可进事件影响模型（证据约束全过）
            is_container_document INTEGER NOT NULL DEFAULT 0,
            underlying_event_types TEXT,       -- JSON 数组：容器文档提示性底层事件族
            entities       TEXT,               -- JSON 数组
            action         TEXT,
            amount         REAL,
            amount_raw_text TEXT,
            amount_unit    TEXT,
            amount_type    TEXT,
            amount_status  TEXT CHECK (amount_status IN ('EXPLICIT','CALCULATED','AMBIGUOUS','MISSING') OR amount_status IS NULL),
            amount_evidence TEXT,
            official_scope TEXT,
            effective_date TEXT,
            effective_date_raw       TEXT,
            effective_date_normalized TEXT,
            effective_date_status    TEXT CHECK (effective_date_status IN ('EXPLICIT','INFERRED','CONFLICT','MISSING') OR effective_date_status IS NULL),
            effective_date_evidence  TEXT,
            assertion_status TEXT CHECK (assertion_status IN ('AFFIRMED','NEGATED','UNCERTAIN','HISTORICAL_REFERENCE') OR assertion_status IS NULL),
            extract_raw    TEXT,               -- Qwen 原始输出
            parse_ok       INTEGER NOT NULL DEFAULT 0,
            latency_s      REAL,
            model_version  TEXT NOT NULL DEFAULT 'qwen3-4b-q4_k_m',
            rule_check     TEXT,               -- JSON：每条规则的通过情况
            candidate_status TEXT NOT NULL DEFAULT 'PENDING'
                CHECK (candidate_status IN ('PENDING','SCHEMA_VALIDATED','HUMAN_ACCEPTED','HUMAN_REJECTED','NEEDS_REVIEW','P2_ROUTED')),
            reviewed_by    TEXT,
            reviewed_at    TEXT,
            review_note    TEXT,
            created_at     TEXT NOT NULL
        );"""


def _migrate_candidate_status(conn):
    """候选表两步重建迁移（幂等）：
    ① 旧 CHECK 含 AUTO_VALIDATED → 重建为 SCHEMA_VALIDATED 版；
    ② 缺 v1.2 列 → 以新 schema 重建（第一步的产物也走这步，列在 INSERT 中补默认值）。
    数据全部保留；HUMAN_ACCEPTED/HUMAN_REJECTED 不改动。"""
    row = conn.execute(
        "SELECT sql FROM sqlite_master WHERE type='table' AND name='information_event_candidate'"
    ).fetchone()
    if not row:
        return
    sql = row["sql"] or ""
    if "AUTO_VALIDATED" in sql:
        conn.executescript(f"""
            BEGIN;
            {_candidate_new_schema("information_event_candidate_new")}
            INSERT INTO information_event_candidate_new
                (candidate_id, raw_id, source_id, event_type, entities, action, amount,
                 official_scope, effective_date, extract_raw, parse_ok, latency_s,
                 model_version, rule_check, candidate_status, reviewed_by, reviewed_at,
                 review_note, created_at)
            SELECT candidate_id, raw_id, source_id, event_type, entities, action, amount,
                   official_scope, effective_date, extract_raw, parse_ok, latency_s,
                   model_version, rule_check,
                   CASE candidate_status WHEN 'AUTO_VALIDATED' THEN 'SCHEMA_VALIDATED'
                        ELSE candidate_status END,
                   reviewed_by, reviewed_at, review_note, created_at
            FROM information_event_candidate;
            DROP TABLE information_event_candidate;
            ALTER TABLE information_event_candidate_new RENAME TO information_event_candidate;
            COMMIT;
        """)
        return
    if _REBUILD2_MARKER not in {r[1] for r in conn.execute("PRAGMA table_info(information_event_candidate)")}:
        old_cols = [r[1] for r in conn.execute("PRAGMA table_info(information_event_candidate)")]
        col_list = ", ".join(old_cols)
        conn.executescript(f"""
            BEGIN;
            {_candidate_new_schema("information_event_candidate_new")}
            INSERT INTO information_event_candidate_new ({col_list})
            SELECT {col_list} FROM information_event_candidate;
            DROP TABLE information_event_candidate;
            ALTER TABLE information_event_candidate_new RENAME TO information_event_candidate;
            COMMIT;
        """)


def get_conn(db_path=None):
    """打开连接：WAL 模式、外键开。"""
    path = Path(db_path) if db_path else DB_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), timeout=60)  # 并发采集/抽取时等待锁而非立即报错
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def _migrate(conn):
    """对已有表按列增量迁移（幂等）；候选表状态值改名需重建表，单独处理。"""
    _migrate_candidate_status(conn)
    existing = {t: {r[1] for r in conn.execute(f"PRAGMA table_info({t})")}
                for (t, _, _) in _COLUMN_MIGRATIONS}
    for table, column, ddl in _COLUMN_MIGRATIONS:
        if column not in existing[table]:
            conn.execute(ddl)


def init_db(db_path=None):
    """建表（IF NOT EXISTS）+ 旧库列迁移，全部幂等。"""
    conn = get_conn(db_path)
    try:
        conn.executescript(_SCHEMA)
        _migrate(conn)
        conn.executescript(_INDEXES)
        conn.commit()
    finally:
        conn.close()
    return db_path or str(DB_PATH)
