"""幂等写入层（plan_db/repo.py）。

所有写入走规格幂等锚 ON CONFLICT，重复运行不产生重复行；
*_at 时间戳一律由系统生成（now() 默认值或数据自带时刻），函数签名不接受
任何 *_at 入参——无人工回填通道（规格第 0 节红线、第 5 节门槛 9）。
"""
import hashlib
import json

from psycopg.types.json import Jsonb


def _j(obj):
    """dict/list → psycopg Jsonb 适配器（None 原样保留）。"""
    return Jsonb(obj) if isinstance(obj, (dict, list)) else obj


# ---------------------------------------------------------------------------
# plan_session / plan_version
# ---------------------------------------------------------------------------

def upsert_session(conn, trading_date, session_tag, dialogue_ref=None, is_demo=False):
    """幂等锚 UNIQUE(trading_date, session_tag)；返回 session_id。"""
    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO stargazer_plan.plan_session "
            "(trading_date, session_tag, dialogue_ref, is_demo) "
            "VALUES (%s, %s, %s, %s) "
            "ON CONFLICT (trading_date, session_tag) DO NOTHING",
            (trading_date, session_tag, dialogue_ref, is_demo),
        )
        cur.execute(
            "SELECT session_id FROM stargazer_plan.plan_session "
            "WHERE trading_date = %s AND session_tag = %s",
            (trading_date, session_tag),
        )
        session_id = cur.fetchone()[0]
    conn.commit()
    return session_id


def compute_content_hash(session_id, version_no, candidates):
    """版本内容 sha256（候选三条件 JSON 规范化后参与哈希），防篡改审计。"""
    payload = {
        "session_id": session_id,
        "version_no": version_no,
        "candidates": [
            {k: c[k] for k in ("sort_order", "code", "trigger_json",
                               "invalidate_json", "observe_json")}
            for c in sorted(candidates, key=lambda x: x["sort_order"])
        ],
    }
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True,
                     default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def ensure_version(conn, session_id, version_no, content_hash):
    """幂等锚 UNIQUE(session_id, version_no)；已存在则原样返回 version_id。"""
    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO stargazer_plan.plan_version "
            "(session_id, version_no, content_hash) VALUES (%s, %s, %s) "
            "ON CONFLICT (session_id, version_no) DO NOTHING",
            (session_id, version_no, content_hash),
        )
        cur.execute(
            "SELECT version_id, status FROM stargazer_plan.plan_version "
            "WHERE session_id = %s AND version_no = %s",
            (session_id, version_no),
        )
        version_id, status = cur.fetchone()
    conn.commit()
    return version_id, status


def freeze_version(conn, version_id):
    """人工冻结：DRAFT→FROZEN（frozen_at 由触发器用系统时钟写入）。

    只允许人工调用本函数完成冻结，不存在任何 LLM 自动冻结通道。
    重复调用安全：仅当状态仍为 DRAFT 时才执行流转。
    """
    with conn.cursor() as cur:
        cur.execute(
            "UPDATE stargazer_plan.plan_version SET status = 'FROZEN' "
            "WHERE version_id = %s AND status = 'DRAFT'",
            (version_id,),
        )
    conn.commit()


def get_version(conn, version_id):
    with conn.cursor() as cur:
        cur.execute(
            "SELECT version_id, session_id, version_no, status, frozen_at, content_hash "
            "FROM stargazer_plan.plan_version WHERE version_id = %s",
            (version_id,),
        )
        return cur.fetchone()


# ---------------------------------------------------------------------------
# plan_candidate / trigger_rule
# ---------------------------------------------------------------------------

def ensure_candidate(conn, version_id, sort_order, code, name=None,
                     direction="BUY", trigger_json=None, invalidate_json=None,
                     observe_json=None):
    """幂等锚 UNIQUE(version_id, sort_order, code)；返回 candidate_id。

    三条件 JSON 均非空（DB CHECK 兜底）：每条候选必须 trigger/invalid/observe 齐全。
    """
    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO stargazer_plan.plan_candidate "
            "(version_id, sort_order, code, name, direction, "
            " trigger_json, invalidate_json, observe_json) "
            "VALUES (%s, %s, %s, %s, %s, %s, %s, %s) "
            "ON CONFLICT (version_id, sort_order, code) DO NOTHING",
            (version_id, sort_order, code, name, direction,
             _j(trigger_json), _j(invalidate_json), _j(observe_json)),
        )
        cur.execute(
            "SELECT candidate_id FROM stargazer_plan.plan_candidate "
            "WHERE version_id = %s AND sort_order = %s AND code = %s",
            (version_id, sort_order, code),
        )
        candidate_id = cur.fetchone()[0]
    conn.commit()
    return candidate_id


def ensure_trigger_rule(conn, candidate_id, rule_type, rule_dsl, condition_json=None):
    """幂等锚 UNIQUE(candidate_id, rule_type, rule_dsl)。"""
    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO stargazer_plan.trigger_rule "
            "(candidate_id, rule_type, condition_json, rule_dsl) "
            "VALUES (%s, %s, %s, %s) "
            "ON CONFLICT (candidate_id, rule_type, rule_dsl) DO NOTHING",
            (candidate_id, rule_type, _j(condition_json), rule_dsl),
        )
    conn.commit()


# ---------------------------------------------------------------------------
# model_recommendation / human_decision / execution_log / outcome / evidence
# ---------------------------------------------------------------------------

def ensure_model_recommendation(conn, candidate_id, model_name, recommend=False,
                                rationale=None, as_of_time=None):
    """影子模型输出；as_of_time 缺省由系统时钟生成，只记录不推荐。"""
    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO stargazer_plan.model_recommendation "
            "(candidate_id, as_of_time, model_name, recommend, rationale) "
            "VALUES (%s, COALESCE(%s, now()), %s, %s, %s) "
            "ON CONFLICT (candidate_id, as_of_time, model_name) DO NOTHING",
            (candidate_id, as_of_time, model_name, recommend, rationale),
        )
    conn.commit()


def insert_human_decision(conn, candidate_id, source, action, reason=None,
                          decided_at=None):
    """人工决策；decided_at 缺省由系统时钟生成（演示/回放可传确定性时刻）。

    返回 decision_id。幂等锚 UNIQUE(candidate_id, decided_at, source)。
    """
    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO stargazer_plan.human_decision "
            "(candidate_id, decided_at, source, action, reason) "
            "VALUES (%s, COALESCE(%s, now()), %s, %s, %s) "
            "ON CONFLICT (candidate_id, decided_at, source) DO NOTHING "
            "RETURNING decision_id",
            (candidate_id, decided_at, source, action, reason),
        )
        row = cur.fetchone()
        if row is None:
            cur.execute(
                "SELECT decision_id FROM stargazer_plan.human_decision "
                "WHERE candidate_id = %s AND source = %s ORDER BY decided_at DESC LIMIT 1",
                (candidate_id, source),
            )
            row = cur.fetchone()
    conn.commit()
    return row[0]


def insert_execution(conn, decision_id, source, source_ref, action,
                     price=None, volume=None, executed_at=None):
    """人工成交记录；executed_at 缺省由系统时钟生成（实盘应传券商回报时刻）。

    action 允许 NONE——决定不买也是记录。幂等锚 UNIQUE(source, source_ref)。
    """
    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO stargazer_plan.execution_log "
            "(decision_id, source, source_ref, action, price, volume, executed_at) "
            "VALUES (%s, %s, %s, %s, %s, %s, COALESCE(%s, now())) "
            "ON CONFLICT (source, source_ref) DO NOTHING",
            (decision_id, source, source_ref, action, price, volume, executed_at),
        )
    conn.commit()


def ensure_dialogue_evidence(conn, session_id, message_ref, speaker=None,
                             content=None, structured_refs=None, message_ts=None):
    """对话证据桥；幂等锚 UNIQUE(session_id, message_ref)。

    structured_refs 记录与之互相关联的结构化记录（version_id/candidate_id 等），
    支撑 dialogue_evidence ↔ plan_version ↔ plan_candidate 双向追溯。
    """
    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO stargazer_plan.dialogue_evidence "
            "(session_id, message_ref, speaker, content, structured_refs, message_ts) "
            "VALUES (%s, %s, %s, %s, %s, %s) "
            "ON CONFLICT (session_id, message_ref) DO NOTHING",
            (session_id, message_ref, speaker, content, _j(structured_refs), message_ts),
        )
    conn.commit()
