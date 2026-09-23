"""第 4 步验收脚本（stargazer_plan/verify_step4.py）。

逐条测试规格 plan_db_v1.md 第 5 节 9 项门槛，全部通过打印总判定 ✓。
脚本幂等：先幂等重建演示链路（同幂等锚重复写入不产生重复行），再逐项验收；
重复运行不会报错、不会污染正式口径（演示 session 均 is_demo=True）。
"""
import sys
from pathlib import Path

import pandas as pd
import psycopg

sys.path.insert(0, str(Path(__file__).resolve().parent))

from plan_db import demo, machine, outcome
from plan_db.db import get_pg_conn
from plan_db.demo import CODE, LABEL_VERSION, SESSION_TAG, load_demo_minute


# ---------------------------------------------------------------------------
# 工具
# ---------------------------------------------------------------------------

def _expect_error(conn, sql, params, keyword=None):
    """执行一条应当被拒绝的 SQL；被拒绝返回 (True, 报错信息)，成功执行返回 (False, ...)。"""
    c = get_pg_conn(autocommit=True)   # 独立 autocommit 连接，报错不污染主事务
    try:
        with c.cursor() as cur:
            cur.execute(sql, params)
        return False, "未被拒绝（违反预期）"
    except psycopg.Error as e:
        msg = str(e).strip().splitlines()[0]
        if keyword and keyword not in msg:
            return False, "被拒绝但原因不符: " + msg
        return True, msg
    finally:
        c.close()


def _q(conn, sql, params=None):
    with conn.cursor() as cur:
        cur.execute(sql, params)
        return cur.fetchall()


def _demo_scope_counts(session_id):
    """演示 session 涉及的全部表行数（幂等验收用）。

    用独立的 autocommit 短连接：SELECT 的 AccessShareLock 随语句结束即释放，
    避免长事务阻塞 check6 内嵌 ensure_demo() 的幂等 DDL（DROP/CREATE TRIGGER）。
    """
    conn = get_pg_conn(autocommit=True)
    try:
        return {

        "plan_session": _q(conn,
            "SELECT count(*) FROM stargazer_plan.plan_session WHERE session_id=%s",
            (session_id,))[0][0],
        "plan_version": _q(conn,
            "SELECT count(*) FROM stargazer_plan.plan_version WHERE session_id=%s",
            (session_id,))[0][0],
        "plan_candidate": _q(conn,
            "SELECT count(*) FROM stargazer_plan.plan_candidate c "
            "JOIN stargazer_plan.plan_version v ON v.version_id=c.version_id "
            "WHERE v.session_id=%s", (session_id,))[0][0],
        "trigger_rule": _q(conn,
            "SELECT count(*) FROM stargazer_plan.trigger_rule r "
            "JOIN stargazer_plan.plan_candidate c ON c.candidate_id=r.candidate_id "
            "JOIN stargazer_plan.plan_version v ON v.version_id=c.version_id "
            "WHERE v.session_id=%s", (session_id,))[0][0],
        "model_recommendation": _q(conn,
            "SELECT count(*) FROM stargazer_plan.model_recommendation m "
            "JOIN stargazer_plan.plan_candidate c ON c.candidate_id=m.candidate_id "
            "JOIN stargazer_plan.plan_version v ON v.version_id=c.version_id "
            "WHERE v.session_id=%s", (session_id,))[0][0],
        "human_decision": _q(conn,
            "SELECT count(*) FROM stargazer_plan.human_decision d "
            "JOIN stargazer_plan.plan_candidate c ON c.candidate_id=d.candidate_id "
            "JOIN stargazer_plan.plan_version v ON v.version_id=c.version_id "
            "WHERE v.session_id=%s", (session_id,))[0][0],
        "execution_log": _q(conn,
            "SELECT count(*) FROM stargazer_plan.execution_log e "
            "JOIN stargazer_plan.human_decision d ON d.decision_id=e.decision_id "
            "JOIN stargazer_plan.plan_candidate c ON c.candidate_id=d.candidate_id "
            "JOIN stargazer_plan.plan_version v ON v.version_id=c.version_id "
            "WHERE v.session_id=%s", (session_id,))[0][0],
        "outcome_attribution": _q(conn,
            "SELECT count(*) FROM stargazer_plan.outcome_attribution o "
            "JOIN stargazer_plan.plan_candidate c ON c.candidate_id=o.candidate_id "
            "JOIN stargazer_plan.plan_version v ON v.version_id=c.version_id "
            "WHERE v.session_id=%s", (session_id,))[0][0],
        "dialogue_evidence": _q(conn,
            "SELECT count(*) FROM stargazer_plan.dialogue_evidence WHERE session_id=%s",
            (session_id,))[0][0],
        "trigger_event": _q(conn,
            "SELECT count(*) FROM stargazer_plan.trigger_event t "
            "JOIN stargazer_plan.plan_candidate c ON c.candidate_id=t.candidate_id "
            "JOIN stargazer_plan.plan_version v ON v.version_id=c.version_id "
            "WHERE v.session_id=%s", (session_id,))[0][0],
        }
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# 9 项门槛
# ---------------------------------------------------------------------------

def check1_frozen_update_rejected(conn, info):
    """门槛 1：FROZEN 版本 UPDATE 任何业务字段都被拒绝（触发器实现）。"""
    version_id = info["version_id"]
    ok, msg = _expect_error(
        conn,
        "UPDATE stargazer_plan.plan_version SET content_hash='tampered' "
        "WHERE version_id=%s", (version_id,))
    return ok, "篡改 content_hash 被拒绝: {}".format(msg)


def check2_append_only_new_version(conn, info):
    """门槛 2：盘中变更只能以新 version_no 追加（FROZEN→SUPERSEDED 合法流转 + v2 追加）。"""
    session_id, version_id = info["session_id"], info["version_id"]
    with conn.cursor() as cur:
        cur.execute(
            "UPDATE stargazer_plan.plan_version SET status='SUPERSEDED' "
            "WHERE version_id=%s AND status='FROZEN'", (version_id,))
        cur.execute(
            "INSERT INTO stargazer_plan.plan_version "
            "(session_id, version_no, content_hash) VALUES (%s, 2, %s) "
            "ON CONFLICT (session_id, version_no) DO NOTHING",
            (session_id, "v2-" + info["trading_date"]))
    conn.commit()
    row = _q(conn,
             "SELECT count(*), count(*) FILTER (WHERE status='DRAFT') "
             "FROM stargazer_plan.plan_version "
             "WHERE session_id=%s AND version_no=2", (session_id,))[0]
    ok = row[0] == 1 and row[1]
    return ok, "v1 已 SUPERSEDED、v2(DRAFT) 追加{}".format("成功" if ok else "失败")


def check3_candidate_three_conditions(conn, info):
    """门槛 3：每条 candidate 三条件齐全（CHECK 非空 + trigger_rule 三行 + 价格禁 0）。"""
    session_id = info["session_id"]
    ok_null, msg_null = _expect_error(
        conn,
        "INSERT INTO stargazer_plan.plan_candidate "
        "(version_id, sort_order, code, trigger_json, invalidate_json, observe_json) "
        "VALUES (%s, 99, '000000', '{}', '{}', NULL)",
        (info["version_id"],))
    ok_zero, msg_zero = _expect_error(
        conn,
        "INSERT INTO stargazer_plan.execution_log "
        "(decision_id, source, source_ref, action, price, executed_at) "
        "SELECT decision_id, 'X', %s, 'BUY', 0, now() "
        "FROM stargazer_plan.human_decision LIMIT 1", ("X-zero-price-check",))
    rows = _q(conn,
        "SELECT count(*) FILTER (WHERE c.trigger_json IS NOT NULL "
        "                     AND c.invalidate_json IS NOT NULL "
        "                     AND c.observe_json IS NOT NULL), "
        "       count(*) FILTER (WHERE n_rules = 3) "
        "FROM stargazer_plan.plan_candidate c "
        "JOIN stargazer_plan.plan_version v ON v.version_id=c.version_id "
        "LEFT JOIN (SELECT candidate_id, count(*) AS n_rules "
        "           FROM stargazer_plan.trigger_rule GROUP BY candidate_id) r "
        "ON r.candidate_id=c.candidate_id WHERE v.session_id=%s", (session_id,))
    n_ok, n_rules3 = rows[0]
    ok = ok_null and ok_zero and n_ok == 5 and n_rules3 == 5
    return ok, ("三条件 CHECK 拦截: {}；价格禁 0 拦截: {}；"
                "演示候选三条件齐全 {}/5、规则齐全 {}/5".format(
                    msg_null[:60], msg_zero[:60], n_ok, n_rules3))


def check4_bidirectional_trace(conn, info):
    """门槛 4：dialogue_evidence ↔ plan_version ↔ plan_candidate 双向追溯。"""
    session_id, version_id = info["session_id"], info["version_id"]
    fwd = _q(conn,
        "SELECT count(DISTINCT e.evidence_id), count(DISTINCT c.candidate_id) "
        "FROM stargazer_plan.dialogue_evidence e "
        "JOIN stargazer_plan.plan_version v "
        "  ON v.version_id = (e.structured_refs->>'version_id')::bigint "
        "JOIN stargazer_plan.plan_candidate c ON c.version_id = v.version_id "
        "WHERE e.session_id=%s", (session_id,))[0]
    rev = _q(conn,
        "SELECT count(DISTINCT e.evidence_id) "
        "FROM stargazer_plan.plan_candidate c "
        "JOIN stargazer_plan.plan_version v ON v.version_id=c.version_id "
        "JOIN stargazer_plan.plan_session s ON s.session_id=v.session_id "
        "JOIN stargazer_plan.dialogue_evidence e ON e.session_id=s.session_id "
        "WHERE v.version_id=%s", (version_id,))[0][0]
    ok = fwd[0] >= 3 and fwd[1] == 5 and rev >= 3
    return ok, ("正向: 证据{}条→候选{}个；反向: 候选→会话→证据{}条".format(fwd[0], fwd[1], rev))


def check5_five_link_chain(conn, info):
    """门槛 5：计划→提醒→人工决策→成交→结果五环外键关联完整。"""
    cid = info["candidates"][1]   # 演示A：触发→批准→成交
    row = _q(conn,
        "SELECT t.event_id, d.decision_id, e.exec_id, o.outcome_id, o.outcome "
        "FROM stargazer_plan.plan_candidate c "
        "LEFT JOIN stargazer_plan.trigger_event t "
        "  ON t.candidate_id=c.candidate_id AND t.rule_type='TRIGGER' "
        "LEFT JOIN stargazer_plan.human_decision d "
        "  ON d.candidate_id=c.candidate_id AND d.action='APPROVE' "
        "LEFT JOIN stargazer_plan.execution_log e "
        "  ON e.decision_id=d.decision_id AND e.action<>'NONE' "
        "LEFT JOIN stargazer_plan.outcome_attribution o "
        "  ON o.candidate_id=c.candidate_id "
        "WHERE c.candidate_id=%s LIMIT 1", (cid,))[0]
    ok = all(x is not None for x in row[:5]) and row[4] == "EXECUTED"
    return ok, ("触发事件/决策/成交/结果/结局 = {}（{}）".format(row[:4], row[4]))


def check6_idempotent_replay_write(conn, info):
    """门槛 6：重复写入（同幂等锚）不产生重复行——重跑整条演示链路后行数不变。"""
    session_id = info["session_id"]
    conn.commit()   # 释放外层读事务的 AccessShareLock，避免阻塞内嵌 ensure_demo 的幂等 DDL
    before = _demo_scope_counts(session_id)
    info2 = demo.ensure_demo()
    after = _demo_scope_counts(session_id)
    diffs = {k: (before[k], after[k]) for k in before if before[k] != after[k]}
    ok = not diffs
    return ok, "重跑前后各表行数一致: {}".format(
        before if ok else diffs)


def check7_minute_replay_hash(conn, info):
    """门槛 7：分钟重放复现触发时间（inputs_hash 与 fired_at 序列完全一致）。"""
    cid = info["candidates"][1]
    day_str = info["trading_date"]
    day_df, pre_close = load_demo_minute(day_str)
    rules = machine.load_rules(conn, cid)
    db_rows = _q(conn,
        "SELECT fired_at, rule_type, inputs_hash FROM stargazer_plan.trigger_event "
        "WHERE candidate_id=%s ORDER BY fired_at, rule_type", (cid,))
    n_before = len(db_rows)
    events, ih = machine.run_minute_machine(
        conn, cid, CODE, day_str, day_df, rules, pre_close=pre_close,
        label_version=LABEL_VERSION)
    n_after = _q(conn,
        "SELECT count(*) FROM stargazer_plan.trigger_event WHERE candidate_id=%s",
        (cid,))[0][0]
    def _norm(t):
        t = pd.Timestamp(t)
        return t.tz_convert("Asia/Shanghai").tz_localize(None) if t.tzinfo else t

    replay = [(_norm(t), rt) for t, rt in events]
    stored = [(_norm(t), rt) for t, rt, _h in db_rows]
    hash_ok = all(r[2] == ih for r in db_rows)
    ok = replay == stored and hash_ok and n_before == n_after
    return ok, ("重放事件 {} 个 == 库内 {} 个；inputs_hash 一致: {}；"
                "行数 {}→{} 无重复".format(
                    len(replay), len(stored), hash_ok, n_before, n_after))


def check8_five_outcomes(conn, info):
    """门槛 8：五种结局分类正确区分（库内五值齐全 + 纯函数单测）。"""
    session_id = info["session_id"]
    rows = _q(conn,
        "SELECT DISTINCT o.outcome FROM stargazer_plan.outcome_attribution o "
        "JOIN stargazer_plan.plan_candidate c ON c.candidate_id=o.candidate_id "
        "JOIN stargazer_plan.plan_version v ON v.version_id=c.version_id "
        "WHERE v.session_id=%s", (session_id,))
    got = {r[0] for r in rows}
    want = set(outcome.OUTCOMES)
    unit = {
        "EXECUTED": outcome.classify_outcome(True, True, {"APPROVE"}, True),
        "NOT_TRIGGERED": outcome.classify_outcome(True, False, set(), False),
        "DECLINED": outcome.classify_outcome(True, True, {"REJECT"}, False),
        "NO_DECISION": outcome.classify_outcome(True, True, set(), False),
        "DATA_MISSING": outcome.classify_outcome(False, False, set(), False),
    }
    unit_ok = all(unit[k] == k for k in want)
    ok = got == want and unit_ok
    return ok, "库内结局集合 {}；纯函数单测通过: {}".format(sorted(got), unit_ok)


def check9_no_manual_backfill(conn, info):
    """门槛 9：无人工回填时间戳通道（所有 *_at 系统生成 + UPDATE 触发器拦截）。"""
    cols = _q(conn,
        "SELECT table_name, column_name, column_default FROM information_schema.columns "
        "WHERE table_schema='stargazer_plan' AND right(column_name, 3) = '_at' "
        "ORDER BY table_name, column_name")
    event_at = {"fired_at", "frozen_at", "executed_at", "decided_at",
                "as_of_time", "message_ts", "triggered_at"}
    sys_generated, violations = [], []
    for table, col, default in cols:
        has_now = default and "now()" in default
        if has_now:
            sys_generated.append("{}.{}(default now())".format(table, col))
        elif col in event_at:
            sys_generated.append("{}.{}(插入即定)".format(table, col))
        else:
            violations.append("{}.{} 无默认值且非事件列".format(table, col))
    ok_created, msg_created = _expect_error(
        conn,
        "UPDATE stargazer_plan.plan_session SET created_at=now()-interval '1 day' "
        "WHERE session_id=%s", (info["session_id"],))
    ok_fired, msg_fired = _expect_error(
        conn,
        "UPDATE stargazer_plan.trigger_event SET fired_at=fired_at-interval '1 minute' "
        "WHERE candidate_id=%s",
        (info["candidates"][1],))
    ok = not violations and ok_created and ok_fired
    return ok, ("*_at 列全部系统生成（{} 列）；篡改 created_at 拦截: {}；"
                "篡改 fired_at 拦截: {}；违规: {}".format(
                    len(sys_generated), ok_created, ok_fired, violations or "无"))


# ---------------------------------------------------------------------------

def main():
    print("== 第 4 步验收：最小交易计划数据库（规格 plan_db_v1.md 第 5 节）==")
    print("-- 幂等重建演示链路（akshare 新浪分钟线，is_demo=True）...")
    info = demo.ensure_demo()
    print("   session_id={} trading_date={} pre_close={} 日线最新={}".format(
        info["session_id"], info["trading_date"], info["pre_close"],
        info["daily_last_date"]))

    conn = get_pg_conn()
    checks = [
        ("1. FROZEN 版本 UPDATE 业务字段被拒绝", check1_frozen_update_rejected),
        ("2. 盘中变更只能以新 version_no 追加", check2_append_only_new_version),
        ("3. 每条 candidate 三条件 CHECK（含价格禁 0）", check3_candidate_three_conditions),
        ("4. dialogue ↔ version ↔ candidate 双向追溯", check4_bidirectional_trace),
        ("5. 计划→提醒→决策→成交→结果五环关联", check5_five_link_chain),
        ("6. 重复写入（同幂等锚）不产生重复行", check6_idempotent_replay_write),
        ("7. 分钟重放 inputs_hash 与触发序列一致", check7_minute_replay_hash),
        ("8. 五种结局分类正确区分", check8_five_outcomes),
        ("9. 无人工回填时间戳通道", check9_no_manual_backfill),
    ]
    results = []
    for name, fn in checks:
        try:
            ok, detail = fn(conn, info)
        except Exception as e:
            conn.rollback()
            ok, detail = False, "异常: {}".format(e)
        results.append(ok)
        print("{} {}: {}".format("✓" if ok else "✗", name, detail))

    counts = _demo_scope_counts(info["session_id"])
    print("\n演示数据落库行数（session_id={}）:".format(info["session_id"]))
    for k, v in counts.items():
        print("   {:<22} {}".format(k, v))

    passed = sum(results)
    print("\n总判定: {}/9 项通过 {}".format(
        passed, "✓ 全部通过" if passed == 9 else "✗ 存在未通过项"))
    conn.close()
    return 0 if passed == 9 else 1


if __name__ == "__main__":
    sys.exit(main())
