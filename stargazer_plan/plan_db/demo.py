"""端到端演示（plan_db/demo.py）。

用 akshare 新浪分钟线（ak.stock_zh_a_minute，北交所不支持）取一只近期票的
真实分钟数据，对历史交易日 D 构造一个"明日计划"演示 session（is_demo=True，
不污染正式口径）：

    session → draft version → 人工 freeze → 次日状态机跑历史分钟线
            → 触发/未触发 → 人工决策/无响应 → 成交记录 → 盘后五结局分类
            → 未执行反事实（日线窗口不足则 CENSORED_TAIL，绝不填 0）

五种结局各造一个候选，全部落库；幂等可重跑（同 trading_date+session_tag
不产生重复 session）。
"""
import datetime as dt
import json
from pathlib import Path

import akshare as ak
import pandas as pd

from . import daily as daily_mod
from . import machine, outcome, repo
from .ddl import ensure_schema
from .db import get_pg_conn

SESSION_TAG = "demo_step4"
SYMBOL = "sh600019"          # 宝钢股份（沪主板，新浪分钟线支持）
CODE = "600019"
MISSING_CODE = "000001"      # 演示 DATA_MISSING：不取分钟数据的票
LABEL_VERSION = machine.DEFAULT_LABEL_VERSION

_DEMO_DIR = Path(__file__).resolve().parent.parent / "data" / "minute_demo"


def fetch_minute_akshare(symbol=SYMBOL):
    """akshare 新浪分钟线 → 规范 DataFrame（dt/open/close/high/low/volume）。"""
    df = ak.stock_zh_a_minute(symbol=symbol, period="1", adjust="")
    df = df.rename(columns={"day": "dt"})
    df["dt"] = pd.to_datetime(df["dt"])
    for col in ("open", "close", "high", "low", "volume"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.sort_values("dt").reset_index(drop=True)
    return df[["dt", "open", "close", "high", "low", "volume"]]


def _pick_day(minute_df):
    """选演示监控日 D：倒数第二个有完整数据的交易日（避免取到当天未完成数据）。"""
    days = sorted(pd.unique(minute_df["dt"].dt.normalize()))
    d = pd.Timestamp(days[-2])
    prev = pd.Timestamp(days[-3])
    pre_close = float(minute_df.loc[minute_df["dt"].dt.normalize() == prev, "close"].iloc[-1])
    return d, pre_close


def _save_demo_minute(day, minute_df, pre_close):
    """演示分钟数据落本地 parquet（verify 重放用，保证与演示时输入完全一致）。"""
    _DEMO_DIR.mkdir(parents=True, exist_ok=True)
    day_df = minute_df[minute_df["dt"].dt.normalize() == day].reset_index(drop=True)
    parquet_path = _DEMO_DIR / "{}_{}.parquet".format(CODE, day.strftime("%Y%m%d"))
    meta_path = _DEMO_DIR / "{}_{}.meta.json".format(CODE, day.strftime("%Y%m%d"))
    day_df.to_parquet(parquet_path, index=False)
    meta_path.write_text(json.dumps(
        {"code": CODE, "trade_day": day.strftime("%Y-%m-%d"), "pre_close": pre_close},
        ensure_ascii=False), encoding="utf-8")
    return parquet_path, meta_path, day_df


def load_demo_minute(day_str):
    """verify 重放入口：读取演示 parquet 与 meta，返回 (day_df, pre_close)。"""
    parquet_path = _DEMO_DIR / "{}_{}.parquet".format(CODE, day_str.replace("-", ""))
    meta_path = _DEMO_DIR / "{}_{}.meta.json".format(CODE, day_str.replace("-", ""))
    day_df = pd.read_parquet(parquet_path)
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    return day_df, meta["pre_close"]


# ---------------------------------------------------------------------------

def ensure_demo(cfg_path=None):
    """幂等构建整个演示链路；返回各环节主键与统计信息（verify 复用）。"""
    conn = get_pg_conn(cfg_path)
    ensure_schema(conn)

    # 1) 分钟数据：akshare 新浪分钟线，D = 倒数第二个完整交易日
    minute_all = fetch_minute_akshare()
    day, pre_close = _pick_day(minute_all)
    _, _, day_df = _save_demo_minute(day, minute_all, pre_close)

    close = day_df["close"]
    am = day_df[day_df["dt"].dt.hour < 13]
    x1 = round(float(am["close"].min()), 2)                      # 早盘最低收价（保证上午触发）
    x2 = round(float(close.min() + (close.max() - close.min()) * 0.5), 2)
    x_huge = round(float(close.max()) * 1.2, 2)                  # 永不触发的假设价

    # 2) session → draft → 人工 freeze
    session_id = repo.upsert_session(
        conn, day.date(), SESSION_TAG,
        dialogue_ref="akshare 新浪分钟线端到端演示（人工构造，demo=True）",
        is_demo=True)

    candidates_spec = [
        # (sort_order, code, name, 触发DSL, 触发条件JSON)
        (1, CODE, "演示A-批准成交", "price >= {}".format(x1),
         {"price_gte": x1, "含义": "价格上穿早盘低位"}),
        (2, CODE, "演示B-人工否决",
         "price <= {} or pct_chg <= -9.9".format(x2),
         {"price_lte": x2, "or": {"pct_chg_lte": "-9.9%"}}),
        (3, CODE, "演示C-无人响应",
         "price >= {} and volume_ratio >= 1.0".format(x1),
         {"price_gte": x1, "and": {"volume_ratio_gte": 1.0}}),
        (4, CODE, "演示D-全天未触发", "price >= {}".format(x_huge),
         {"price_gte": x_huge, "含义": "假设触发价（演示反事实假设）"}),
        (5, MISSING_CODE, "演示E-数据缺失", "price >= 9999",
         {"price_gte": 9999}),
    ]
    invalidate_dsl = "time >= 14:55"
    observe_dsl = "volume_ratio >= 2"

    content_hash = repo.compute_content_hash(
        session_id, 1,
        [{"sort_order": s, "code": c, "trigger_json": tj,
          "invalidate_json": {"time_gte": "14:55"},
          "observe_json": {"volume_ratio_gte": 2}}
         for s, c, _, _, tj in candidates_spec])
    version_id, status = repo.ensure_version(conn, session_id, 1, content_hash)
    if status == "DRAFT":
        repo.freeze_version(conn, version_id)   # 人工冻结动作的唯一入口

    cand = {}
    for sort_order, code, name, trig_dsl, trig_json in candidates_spec:
        cid = repo.ensure_candidate(
            conn, version_id, sort_order, code, name=name,
            trigger_json=trig_json,
            invalidate_json={"time_gte": "14:55", "含义": "尾盘失效"},
            observe_json={"volume_ratio_gte": 2, "含义": "放量观察"})
        cand[sort_order] = cid
        # 每条候选三条件齐全：TRIGGER / INVALIDATE / OBSERVE 各一条规则
        repo.ensure_trigger_rule(conn, cid, "TRIGGER", trig_dsl, condition_json=trig_json)
        repo.ensure_trigger_rule(conn, cid, "INVALIDATE", invalidate_dsl,
                                 condition_json={"time_gte": "14:55"})
        repo.ensure_trigger_rule(conn, cid, "OBSERVE", observe_dsl,
                                 condition_json={"volume_ratio_gte": 2})

    # 3) 次日状态机：有分钟数据的候选跑确定性监控（候选5无数据 → DATA_MISSING）
    fired = {}
    inputs_hash = {}
    for sort_order in (1, 2, 3, 4):
        cid = cand[sort_order]
        rules = machine.load_rules(conn, cid)
        events, ih = machine.run_minute_machine(
            conn, cid, CODE, day.date(), day_df, rules, pre_close=pre_close,
            label_version=LABEL_VERSION)
        fired[cid] = [t for t, rt in events if rt == "TRIGGER"]
        inputs_hash[cid] = ih
    # 演示要求：1/2/3 号候选必须触发（分别走向 EXECUTED/DECLINED/NO_DECISION），
    # 4 号必须全天未触发（NOT_TRIGGERED），否则说明构造与真实行情不符
    for cid in (cand[1], cand[2], cand[3]):
        if not fired[cid]:
            raise RuntimeError("演示候选 {} 未触发，演示构造与数据不符，请人工检查".format(cid))
    if fired[cand[4]]:
        raise RuntimeError("演示候选 4 意外触发，x_huge 构造失效，请人工检查")

    # 4) 人工决策与成交（候选1 批准+成交；候选2 否决；候选3/4 无人响应）
    #    演示用确定性时刻（同一交易日同一时刻），保证重复运行幂等；
    #    正式场景 decided_at 缺省由系统时钟生成
    dec1_at = dt.datetime.combine(day.date(), dt.time(9, 35))
    dec2_at = dt.datetime.combine(day.date(), dt.time(9, 36))
    d1 = repo.insert_human_decision(conn, cand[1], "CHAT", "APPROVE",
                                    reason="演示：批准买入", decided_at=dec1_at)
    entry1 = float(day_df.loc[day_df["dt"] == fired[cand[1]][0], "close"].iloc[0])
    repo.insert_execution(conn, d1, "DESK", "DEMO-EXEC-{}-{}".format(CODE, cand[1]),
                          action="BUY", price=entry1, volume=100,
                          executed_at=fired[cand[1]][0] + pd.Timedelta(minutes=3))
    repo.insert_human_decision(conn, cand[2], "PUSH", "REJECT",
                               reason="演示：人工否决", decided_at=dec2_at)

    # 影子模型记录（只记录不推荐，不影响状态机；演示用确定性 as_of_time 保证幂等）
    repo.ensure_model_recommendation(conn, cand[1], "shadow-demo-v0",
                                     recommend=False,
                                     rationale="影子表只记录，不推荐、不影响状态机",
                                     as_of_time=dec1_at)

    # 5) 对话证据桥（双向追溯）
    for i, (ref, speaker, content) in enumerate([
            ("msg-{:04d}".format(session_id) + "-01", "human",
             "明天关注 600019，早盘站上 {} 就准备买".format(x1)),
            ("msg-{:04d}".format(session_id) + "-02", "agent",
             "已生成计划草稿：5 条候选，三条件齐全，请人工确认后冻结"),
            ("msg-{:04d}".format(session_id) + "-03", "human",
             "确认冻结，明日按状态机监控，触发后我再决定")], start=1):
        repo.ensure_dialogue_evidence(
            conn, session_id, ref, speaker=speaker, content=content,
            structured_refs={"version_id": version_id,
                             "candidate_ids": sorted(cand.values())},
            message_ts=dt.datetime.combine(
                (day - pd.Timedelta(days=1)).date(),
                dt.time(20, 0 + i * 5), tzinfo=dt.timezone.utc))

    # 6) 盘后：五结局分类 + 未执行反事实（日线窗口不足 → CENSORED_TAIL，不填 0）
    daily_df = daily_mod.load_daily(CODE)
    overlap = "同日多候选重叠，只标'事件后反应'，不声称因果"
    specs = {}
    specs[cand[1]] = dict(outcome="EXECUTED", entry=entry1,
                          basis="真实人工成交，无反事实（EXECUTED）")
    for cid, trig_times in fired.items():
        if cid == cand[1] or not trig_times:
            continue
        entry = float(day_df.loc[day_df["dt"] == trig_times[0], "close"].iloc[0])
        specs[cid] = dict(outcome=None, entry=entry,
                          basis="would_fill_at=trigger_price_if_fired={:.2f}"
                                "（触发分钟收盘价，反事实假设）".format(entry))
    specs[cand[4]] = dict(outcome=None, entry=x_huge,
                          basis="would_fill_at=assumed_trigger_price={:.2f}"
                                "（计划全天未触发，假设价，非真实成交）".format(x_huge))
    specs[cand[5]] = dict(outcome="DATA_MISSING", entry=None,
                          basis="分钟数据缺失，无法判定（绝不猜测）")

    decisions = {cand[1]: {"APPROVE"}, cand[2]: {"REJECT"}}
    for sort_order in (1, 2, 3, 4, 5):
        cid = cand[sort_order]
        sp = specs[cid]
        if sp["outcome"] is None:
            trig = fired.get(cid, [])
            oc = outcome.classify_outcome(
                minute_present=(sort_order != 5),
                triggered=bool(trig),
                decision_actions=decisions.get(cid, set()),
                has_execution=False)
            sp["outcome"] = oc
        trig_times = fired.get(cid, [])
        trig_at = trig_times[0].to_pydatetime() if trig_times else None
        cf = outcome.counterfactual_from_daily(daily_df, day, sp["entry"]) \
            if sp["outcome"] in ("NOT_TRIGGERED", "DECLINED", "NO_DECISION") \
            else {"return_1d": None, "return_3d": None, "return_5d": None,
                  "mfe": None, "mae": None, "label_status": "OK"}
        if sp["outcome"] == "DATA_MISSING":
            cf["label_status"] = "CENSORED_TAIL"
        basis = sp["basis"]
        if cf["label_status"] == "CENSORED_TAIL" and sp["outcome"] != "DATA_MISSING":
            basis += "；CENSORED_TAIL：日线增量窗口未走完，缺失字段绝不填 0"
        outcome.write_outcome(
            conn, cid, LABEL_VERSION, outcome=sp["outcome"],
            triggered_at=trig_at, entry_price=sp["entry"],
            basis=basis, overlap_note=overlap if sort_order != 5 else None,
            label_status=cf["label_status"],
            return_1d=cf["return_1d"], return_3d=cf["return_3d"],
            return_5d=cf["return_5d"], mfe=cf["mfe"], mae=cf["mae"])

    conn.close()
    return {
        "session_id": session_id, "version_id": version_id,
        "trading_date": str(day.date()), "pre_close": pre_close,
        "levels": {"x1": x1, "x2": x2, "x_huge": x_huge},
        "candidates": cand, "fired": {k: [str(t) for t in v]
                                      for k, v in fired.items()},
        "inputs_hash": inputs_hash,
        "daily_last_date": str(daily_df["date"].iloc[-1].date()) if len(daily_df) else None,
    }


def main():
    import pprint
    info = ensure_demo()
    print("演示链路已落库（is_demo=True，trading_date={}）：".format(info["trading_date"]))
    pprint.pprint(info, sort_dicts=False)


if __name__ == "__main__":
    main()
