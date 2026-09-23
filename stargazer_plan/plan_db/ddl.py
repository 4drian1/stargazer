"""幂等 DDL（plan_db/ddl.py，规格 plan_db_v1.md 第 1 节）。

- 9 张业务表 + 状态机落库表 trigger_event，全部 CREATE TABLE IF NOT EXISTS；
- 金额/价格一律 nullable 且 CHECK 禁止 0 占位；
- plan_version 冻结不可变：BEFORE UPDATE/DELETE 触发器（FROZEN/SUPERSEDED 行
  只允许 FROZEN→SUPERSEDED 一次合法流转，且连 frozen_at 也不许动）；
  DRAFT→FROZEN 瞬间由触发器用系统时钟写 frozen_at，杜绝人工回填；
- 其余表的 *_at 列一律"插入即定"，禁止 UPDATE 回填（无人工回填时间戳通道）；
- 触发器幂等：DROP TRIGGER IF EXISTS + CREATE TRIGGER；
- 重复执行 ensure_schema() 不产生任何重复对象。
"""

SCHEMA = "stargazer_plan"

_DDL = """
CREATE SCHEMA IF NOT EXISTS stargazer_plan;

-- ---------------------------------------------------------------------------
-- 1. plan_session：一次"明日计划"讨论 = 一个 session
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS stargazer_plan.plan_session (
    session_id    BIGSERIAL PRIMARY KEY,
    trading_date  DATE NOT NULL,                -- 计划监控的交易日（"明日"）
    session_tag   TEXT NOT NULL,                -- 讨论批次标签
    dialogue_ref  TEXT,                         -- 指向对话来源（聊天记录位置等）
    is_demo       BOOLEAN NOT NULL DEFAULT FALSE,  -- 演示数据标记，不污染正式口径
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (trading_date, session_tag)
);

-- ---------------------------------------------------------------------------
-- 2. plan_version：DRAFT→FROZEN→SUPERSEDED；冻结后业务字段不可 UPDATE（触发器）
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS stargazer_plan.plan_version (
    version_id    BIGSERIAL PRIMARY KEY,
    session_id    BIGINT NOT NULL REFERENCES stargazer_plan.plan_session(session_id),
    version_no    INTEGER NOT NULL CHECK (version_no >= 1),
    status        TEXT NOT NULL DEFAULT 'DRAFT'
                  CHECK (status IN ('DRAFT','FROZEN','SUPERSEDED')),
    frozen_at     TIMESTAMPTZ,                  -- 冻结时刻：仅 DRAFT→FROZEN 瞬间由触发器写
    content_hash  TEXT NOT NULL,                -- 版本内容 sha256，防篡改审计
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (session_id, version_no)
);

-- ---------------------------------------------------------------------------
-- 3. plan_candidate：每条含 trigger/invalid/observe 三条件（JSON 给人看）
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS stargazer_plan.plan_candidate (
    candidate_id    BIGSERIAL PRIMARY KEY,
    version_id      BIGINT NOT NULL REFERENCES stargazer_plan.plan_version(version_id),
    sort_order      INTEGER NOT NULL,
    code            TEXT NOT NULL,
    name            TEXT,
    direction       TEXT NOT NULL DEFAULT 'BUY' CHECK (direction IN ('BUY','SELL')),
    trigger_json    JSONB NOT NULL,             -- 触发条件（结构化，给人看）
    invalidate_json JSONB NOT NULL,             -- 失效条件
    observe_json    JSONB NOT NULL,             -- 观察条件
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (version_id, sort_order, code),
    CHECK (trigger_json IS NOT NULL AND invalidate_json IS NOT NULL AND observe_json IS NOT NULL)
);

-- ---------------------------------------------------------------------------
-- 4. trigger_rule：condition_json 给人看，rule_dsl 给状态机执行
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS stargazer_plan.trigger_rule (
    rule_id        BIGSERIAL PRIMARY KEY,
    candidate_id   BIGINT NOT NULL REFERENCES stargazer_plan.plan_candidate(candidate_id),
    rule_type      TEXT NOT NULL CHECK (rule_type IN ('TRIGGER','INVALIDATE','OBSERVE')),
    condition_json JSONB,                       -- 给人看的结构化条件
    rule_dsl       TEXT NOT NULL,               -- 确定性 DSL 字符串（状态机执行）
    created_at     TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (candidate_id, rule_type, rule_dsl)
);

-- ---------------------------------------------------------------------------
-- 5. model_recommendation：影子模型输出，只记录、不推荐、不影响状态机
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS stargazer_plan.model_recommendation (
    rec_id       BIGSERIAL PRIMARY KEY,
    candidate_id BIGINT NOT NULL REFERENCES stargazer_plan.plan_candidate(candidate_id),
    as_of_time   TIMESTAMPTZ NOT NULL,
    model_name   TEXT NOT NULL,
    recommend    BOOLEAN NOT NULL DEFAULT FALSE,
    rationale    TEXT,
    created_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (candidate_id, as_of_time, model_name)
);

-- ---------------------------------------------------------------------------
-- 6. human_decision：人工决策，冻结之外的第二个人工关口
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS stargazer_plan.human_decision (
    decision_id  BIGSERIAL PRIMARY KEY,
    candidate_id BIGINT NOT NULL REFERENCES stargazer_plan.plan_candidate(candidate_id),
    decided_at   TIMESTAMPTZ NOT NULL,          -- 系统时钟写入（插入即定，禁 UPDATE）
    source       TEXT NOT NULL CHECK (source IN ('CHAT','PUSH','DESK')),
    action       TEXT NOT NULL CHECK (action IN ('APPROVE','REJECT','MODIFY','WATCH')),
    reason       TEXT,
    risk_type    TEXT,                          -- 风险类型（冻结枚举，spec 第 7 节）；可空
    evidence_ref TEXT,                          -- 证据引用（MANUAL_OTHER 时必填说明）
    created_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (candidate_id, decided_at, source)
);

-- ---------------------------------------------------------------------------
-- 7. execution_log：人工成交记录；action 允许 NONE（决定不买也是记录）
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS stargazer_plan.execution_log (
    exec_id      BIGSERIAL PRIMARY KEY,
    decision_id  BIGINT NOT NULL REFERENCES stargazer_plan.human_decision(decision_id),
    source       TEXT NOT NULL,
    source_ref   TEXT NOT NULL,
    action       TEXT NOT NULL CHECK (action IN ('BUY','SELL','NONE')),
    risk_type    TEXT,                          -- 风险类型（冻结枚举，spec 第 7 节）；可空
    price        DOUBLE PRECISION CHECK (price IS NULL OR price <> 0),   -- 禁 0 占位
    volume       BIGINT CHECK (volume IS NULL OR volume > 0),
    executed_at  TIMESTAMPTZ NOT NULL,          -- 成交时刻（插入即定，禁 UPDATE）
    created_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (source, source_ref)
);

-- ---------------------------------------------------------------------------
-- 8. outcome_attribution：五种结局 + 未执行反事实；CENSORED_TAIL 不填 0
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS stargazer_plan.outcome_attribution (
    outcome_id   BIGSERIAL PRIMARY KEY,
    candidate_id BIGINT NOT NULL REFERENCES stargazer_plan.plan_candidate(candidate_id),
    label_version TEXT NOT NULL,
    outcome      TEXT CHECK (outcome IN ('EXECUTED','NOT_TRIGGERED','DECLINED',
                                         'NO_DECISION','DATA_MISSING')),
    triggered_at TIMESTAMPTZ,                   -- 首次触发时刻（状态机输出）
    entry_price  DOUBLE PRECISION CHECK (entry_price IS NULL OR entry_price <> 0),
    return_1d    DOUBLE PRECISION,              -- 反事实收益（比例），CENSORED_TAIL 时为 NULL
    return_3d    DOUBLE PRECISION,
    return_5d    DOUBLE PRECISION,
    mfe          DOUBLE PRECISION,              -- 窗口内最大有利变动（反事实）
    mae          DOUBLE PRECISION,              -- 窗口内最大不利变动（反事实）
    label_status TEXT NOT NULL DEFAULT 'OK' CHECK (label_status IN ('OK','CENSORED_TAIL')),
    counterfactual_basis TEXT,                  -- 反事实假设显式记录，不得伪装成真实成交
    overlap_note TEXT,                          -- 重叠事件口径：只标"事件后反应"不声称因果
    computed_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (candidate_id, label_version)
);

-- ---------------------------------------------------------------------------
-- 9. dialogue_evidence：原始对话与结构化字段互相追溯的桥
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS stargazer_plan.dialogue_evidence (
    evidence_id     BIGSERIAL PRIMARY KEY,
    session_id      BIGINT NOT NULL REFERENCES stargazer_plan.plan_session(session_id),
    message_ref     TEXT NOT NULL,              -- 对话消息定位符
    message_ts      TIMESTAMPTZ,                -- 消息原始时间（插入即定，禁 UPDATE）
    speaker         TEXT,
    content         TEXT,
    structured_refs JSONB,                      -- 指向 plan_version/plan_candidate 等
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (session_id, message_ref)
);

-- ---------------------------------------------------------------------------
-- 状态机落库表（规格第 2 节）：触发事件，inputs_hash 支撑分钟重放校验
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS stargazer_plan.trigger_event (
    event_id     BIGSERIAL PRIMARY KEY,
    candidate_id BIGINT NOT NULL REFERENCES stargazer_plan.plan_candidate(candidate_id),
    fired_at     TIMESTAMPTZ NOT NULL,          -- 触发分钟（来自分钟线数据本身）
    rule_type    TEXT NOT NULL CHECK (rule_type IN ('TRIGGER','INVALIDATE','OBSERVE')),
    inputs_hash  TEXT NOT NULL,                 -- sha256(分钟区间+规则dsl)
    label_version TEXT NOT NULL,
    created_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (candidate_id, fired_at, rule_type, inputs_hash)
);

-- ---------------------------------------------------------------------------
-- 触发器 1：plan_version 冻结不可变
--   FROZEN：仅允许 FROZEN→SUPERSEDED 流转，且其余业务列与 frozen_at 一律不许动；
--   SUPERSEDED：完全不可动；FROZEN/SUPERSEDED 禁止 DELETE（审计保留）；
--   DRAFT→FROZEN 瞬间由触发器写 frozen_at=now()，人工回填的 past time 被覆盖。
-- ---------------------------------------------------------------------------
CREATE OR REPLACE FUNCTION stargazer_plan.plan_version_guard() RETURNS trigger AS $f$
BEGIN
    IF TG_OP = 'DELETE' THEN
        IF OLD.status IN ('FROZEN','SUPERSEDED') THEN
            RAISE EXCEPTION 'plan_version % 已 %，禁止删除（审计保留）',
                OLD.version_id, OLD.status;
        END IF;
        RETURN OLD;
    END IF;
    -- 以下为 UPDATE
    IF OLD.status IN ('FROZEN','SUPERSEDED') THEN
        IF OLD.status = 'FROZEN' AND NEW.status = 'SUPERSEDED'
           AND NEW.session_id IS NOT DISTINCT FROM OLD.session_id
           AND NEW.version_no IS NOT DISTINCT FROM OLD.version_no
           AND NEW.content_hash IS NOT DISTINCT FROM OLD.content_hash THEN
            NEW.frozen_at := OLD.frozen_at;  -- 合法流转也不许动 frozen_at
            RETURN NEW;
        END IF;
        RAISE EXCEPTION 'plan_version % 状态为 %，业务字段不可修改；盘中变更只能以新 version_no 追加新版本',
            OLD.version_id, OLD.status;
    END IF;
    -- DRAFT：业务字段可改，但 frozen_at 只能由系统在 DRAFT→FROZEN 瞬间写入
    IF OLD.status = 'DRAFT' AND NEW.status = 'FROZEN' THEN
        NEW.frozen_at := now();
    ELSE
        NEW.frozen_at := OLD.frozen_at;
    END IF;
    RETURN NEW;
END;
$f$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trg_plan_version_guard ON stargazer_plan.plan_version;
CREATE TRIGGER trg_plan_version_guard
    BEFORE UPDATE OR DELETE ON stargazer_plan.plan_version
    FOR EACH ROW EXECUTE FUNCTION stargazer_plan.plan_version_guard();

-- ---------------------------------------------------------------------------
-- 触发器 2：通用 *_at 列"插入即定"，禁止 UPDATE 回填（无人工回填时间戳通道）
--   列名以 TG_ARGV 传入；plan_version 的 frozen_at 由上面的专用触发器管辖，不重复挂。
-- ---------------------------------------------------------------------------
CREATE OR REPLACE FUNCTION stargazer_plan.no_at_backfill() RETURNS trigger AS $f$
DECLARE
    col  TEXT;
    same BOOLEAN;
BEGIN
    FOREACH col IN ARRAY TG_ARGV LOOP
        EXECUTE format('SELECT ($1).%I IS NOT DISTINCT FROM ($2).%I', col, col)
            USING NEW, OLD INTO same;
        IF NOT same THEN
            RAISE EXCEPTION '表 % 的 % 列由系统生成，禁止 UPDATE（无人工回填时间戳通道）',
                TG_TABLE_NAME, col;
        END IF;
    END LOOP;
    RETURN NEW;
END;
$f$ LANGUAGE plpgsql;
"""

# 各表需要保护"插入即定"的 *_at 列（created_at/computed_at 有 now() 默认值，
# fired_at/executed_at/decided_at/as_of_time/message_ts/triggered_at 为插入时系统或数据时钟写入）
_AT_COLUMNS = {
    "plan_session": ["created_at"],
    "plan_candidate": ["created_at"],
    "trigger_rule": ["created_at"],
    "model_recommendation": ["created_at", "as_of_time"],
    "human_decision": ["created_at", "decided_at"],
    "execution_log": ["created_at", "executed_at"],
    "outcome_attribution": ["triggered_at"],  # computed_at 由账本 DO UPDATE 时系统刷新
    "dialogue_evidence": ["created_at", "message_ts"],
    "trigger_event": ["created_at", "fired_at"],
}


def _trigger_ddl():
    parts = []
    for table, cols in _AT_COLUMNS.items():
        args = ", ".join("'{}'".format(c) for c in cols)
        parts.append(
            "DROP TRIGGER IF EXISTS trg_{t}_no_at_backfill ON stargazer_plan.{t};\n"
            "CREATE TRIGGER trg_{t}_no_at_backfill\n"
            "    BEFORE UPDATE ON stargazer_plan.{t}\n"
            "    FOR EACH ROW EXECUTE FUNCTION stargazer_plan.no_at_backfill({args});".format(
                t=table, args=args)
        )
    return "\n\n".join(parts)


def ensure_columns(conn):
    """规则层 v1 迁移（2026-09-17）：给已存在的库补可空列，幂等可重复调用。

    human_decision + risk_type/evidence_ref；execution_log + risk_type。
    """
    with conn.cursor() as cur:
        cur.execute("""
        ALTER TABLE stargazer_plan.human_decision
            ADD COLUMN IF NOT EXISTS risk_type TEXT;
        ALTER TABLE stargazer_plan.human_decision
            ADD COLUMN IF NOT EXISTS evidence_ref TEXT;
        ALTER TABLE stargazer_plan.execution_log
            ADD COLUMN IF NOT EXISTS risk_type TEXT;
        """)
    conn.commit()


def ensure_schema(conn):
    """幂等建 schema、9 张表、trigger_event 与全部触发器；可重复调用。"""
    with conn.cursor() as cur:
        cur.execute(_DDL)
        cur.execute(_trigger_ddl())
    conn.commit()
    ensure_columns(conn)
