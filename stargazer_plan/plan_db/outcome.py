"""五种结局分类与未执行反事实（plan_db/outcome.py，规格第 3、4 节）。


五种结局（严格区分，绝不混淆）：
    EXECUTED      触发 → 人工批准 → 有成交记录
    NOT_TRIGGERED 全天未触发（状态机 TRIGGER 输出为空）
    DECLINED      触发 → 人工否决（human_decision=REJECT）
    NO_DECISION   触发了但人工未响应（超时，区别于否决）
    DATA_MISSING  分钟数据缺失无法判定（绝不猜）

反事实（规格第 4 节）：
- 仅对 NOT_TRIGGERED / DECLINED / NO_DECISION 计算；
- 触发过的计划 basis 记 would_fill_at=trigger_price_if_fired；
- 未触发的计划基于"若触发价出现"的假设价，basis 显式记录假设，
  不得伪装成真实成交；
- 日线窗口未走完（增量延迟）必须 CENSORED_TAIL，缺失字段一律 NULL 不填 0；
- 重叠事件只标"事件后反应"，不声称因果。
"""
import pandas as pd

HORIZONS = (1, 3, 5)

OUTCOMES = ("EXECUTED", "NOT_TRIGGERED", "DECLINED", "NO_DECISION", "DATA_MISSING")


def classify_outcome(minute_present, triggered, decision_actions, has_execution):
    """纯函数结局分类。

    minute_present    : 当日分钟数据是否存在（缺失 → DATA_MISSING，绝不猜测）
    triggered         : 状态机是否产生过 TRIGGER 事件（未失效抑制前）
    decision_actions  : 人工决策动作集合（如 {'APPROVE'} / {'REJECT'} / set()）
    has_execution     : 是否存在 action <> 'NONE' 的成交记录
    """
    if not minute_present:
        return "DATA_MISSING"
    if not triggered:
        return "NOT_TRIGGERED"
    if has_execution:
        return "EXECUTED"
    if "REJECT" in decision_actions:
        return "DECLINED"
    return "NO_DECISION"


def counterfactual_from_daily(daily_df, event_day, entry_price):
    """按日线计算 1/3/5 日反事实收益与 mfe/mae。

    返回 dict：return_1d/3d/5d、mfe、mae（比例）、label_status。
    窗口不足 5 个交易日（日线增量延迟）→ label_status='CENSORED_TAIL'，
    缺失字段一律 None（不填 0）。
    """
    res = {"return_1d": None, "return_3d": None, "return_5d": None,
           "mfe": None, "mae": None, "label_status": "CENSORED_TAIL"}
    if entry_price is None:
        res["label_status"] = "CENSORED_TAIL"
        return res
    fut = daily_df[daily_df["date"] > pd.Timestamp(event_day)].head(max(HORIZONS))
    for n in HORIZONS:
        if len(fut) >= n:
            res["return_{}d".format(n)] = float(fut["close"].iloc[n - 1] / entry_price - 1.0)
    if len(fut):
        res["mfe"] = float(fut["high"].max() / entry_price - 1.0)
        res["mae"] = float(fut["low"].min() / entry_price - 1.0)
    if len(fut) >= max(HORIZONS):
        res["label_status"] = "OK"
    return res


def write_outcome(conn, candidate_id, label_version, outcome=None, triggered_at=None,
                  entry_price=None, basis=None, overlap_note=None,
                  return_1d=None, return_3d=None, return_5d=None,
                  mfe=None, mae=None, label_status="OK"):
    """落库 outcome_attribution；同 (candidate_id, label_version) 重复写入只刷新
    计算字段不留重复行（ON CONFLICT DO UPDATE，与 Corvus 标签账本同口径）。"""
    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO stargazer_plan.outcome_attribution "
            "(candidate_id, label_version, outcome, triggered_at, entry_price, "
            " return_1d, return_3d, return_5d, mfe, mae, label_status, "
            " counterfactual_basis, overlap_note) "
            "VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s) "
            "ON CONFLICT (candidate_id, label_version) DO UPDATE SET "
            " outcome = EXCLUDED.outcome, triggered_at = EXCLUDED.triggered_at, "
            " entry_price = EXCLUDED.entry_price, return_1d = EXCLUDED.return_1d, "
            " return_3d = EXCLUDED.return_3d, return_5d = EXCLUDED.return_5d, "
            " mfe = EXCLUDED.mfe, mae = EXCLUDED.mae, "
            " label_status = EXCLUDED.label_status, "
            " counterfactual_basis = EXCLUDED.counterfactual_basis, "
            " overlap_note = EXCLUDED.overlap_note, "
            " computed_at = now()",
            (candidate_id, label_version, outcome, triggered_at, entry_price,
             return_1d, return_3d, return_5d, mfe, mae, label_status,
             basis, overlap_note),
        )
    conn.commit()
