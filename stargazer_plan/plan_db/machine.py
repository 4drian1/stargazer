"""确定性状态机（plan_db/machine.py，规格第 2 节）。

输入：FROZEN 版本的候选 + trigger_rule 表中的规则 DSL + 当日分钟线。
输出：trigger_event 落库（candidate_id, fired_at, rule_type, inputs_hash, label_version）。

回放纪律：状态机为纯函数，同一份分钟输入 + 同一组规则必然产生完全相同的
fired_at 序列；inputs_hash = sha256(分钟区间指纹 + 规则 DSL 规范化串)，
回放时校验一致。写入用 ON CONFLICT DO NOTHING，重复运行不产生重复行。

触发语义：每条规则按"边缘触发"（False→True 跳变）产生事件；
首个 INVALIDATE 事件之后的 TRIGGER 事件被失效抑制丢弃（计划已失效，不再触发）。
"""
import hashlib
import json

import pandas as pd

from . import dsl

DEFAULT_LABEL_VERSION = "minute_replay_v1"


def load_rules(conn, candidate_id):
    """读取候选的全部规则（rule_type, rule_dsl），按规则表幂等锚排序保证确定性。"""
    with conn.cursor() as cur:
        cur.execute(
            "SELECT rule_type, rule_dsl FROM stargazer_plan.trigger_rule "
            "WHERE candidate_id = %s ORDER BY rule_type, rule_dsl",
            (candidate_id,),
        )
        return cur.fetchall()


def compute_inputs_hash(code, trade_day, minute_df, pre_close, rules):
    """分钟区间指纹 + 规则 DSL（规范化后排序拼接）的 sha256。"""
    fp = {
        "code": str(code),
        "trade_day": str(trade_day),
        "n_bars": int(len(minute_df)),
        "dt_first": str(minute_df["dt"].iloc[0]),
        "dt_last": str(minute_df["dt"].iloc[-1]),
        "pre_close": None if pre_close is None else float(pre_close),
        "rules": sorted(
            {"{}|{}".format(rt, dsl.canonical(dsl.parse(text))) for rt, text in rules}
        ),
    }
    raw = json.dumps(fp, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def run_rules(minute_df, rules, pre_close=None):
    """纯函数核心：求触发事件序列（不落库）。

    参数
    ----
    minute_df : 当日分钟 DataFrame（dt/open/close/high/low/volume，按 dt 升序）
    rules     : [(rule_type, rule_dsl), ...]
    pre_close : 昨收（仅 pct_chg 规则需要）

    返回 [(fired_at: pd.Timestamp, rule_type), ...]，已按时间排序并做失效抑制。
    """
    events = []
    for rule_type, text in rules:
        ast = dsl.parse(text)
        cond = dsl.evaluate(ast, minute_df, pre_close)
        for t in dsl.edge_fired_times(cond, minute_df):
            events.append((t, rule_type))
    # 失效抑制：首个 INVALIDATE 触发时刻起，之后的 TRIGGER 事件作废
    inv_times = [t for t, rt in events if rt == "INVALIDATE"]
    if inv_times:
        cutoff = min(inv_times)
        events = [e for e in events if not (e[1] == "TRIGGER" and e[0] >= cutoff)]
    events.sort(key=lambda e: (e[0], e[1]))
    return events


def run_minute_machine(conn, candidate_id, code, trade_day, minute_df, rules,
                       pre_close=None, label_version=DEFAULT_LABEL_VERSION):
    """对单候选跑状态机并幂等落库 trigger_event。

    返回 (events, inputs_hash)；重复运行 fired_at 序列与 inputs_hash 完全一致，
    且 ON CONFLICT DO NOTHING 不产生重复行。
    """
    if minute_df is None or len(minute_df) == 0:
        raise ValueError("分钟数据为空（应归类 DATA_MISSING，状态机不运行）")
    minute_df = minute_df.sort_values("dt").reset_index(drop=True)
    events = run_rules(minute_df, rules, pre_close)
    inputs_hash = compute_inputs_hash(code, trade_day, minute_df, pre_close, rules)
    with conn.cursor() as cur:
        for fired_at, rule_type in events:
            cur.execute(
                "INSERT INTO stargazer_plan.trigger_event "
                "(candidate_id, fired_at, rule_type, inputs_hash, label_version) "
                "VALUES (%s, %s, %s, %s, %s) "
                "ON CONFLICT (candidate_id, fired_at, rule_type, inputs_hash) DO NOTHING",
                (candidate_id, fired_at.to_pydatetime(), rule_type,
                 inputs_hash, label_version),
            )
    conn.commit()
    return events, inputs_hash
