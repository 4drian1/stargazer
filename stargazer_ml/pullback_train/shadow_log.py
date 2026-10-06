# -*- coding: utf-8 -*-
"""影子阶段决策留痕系统（2026-09-09 设计，对应评审提出的影子验证协议）。

每天模型留下一条决策记录（shadow_decisions 表）：
  股票/信号日/决策点 + P_risk/P_opp/象限(FOCUS/WAIT/WATCH/REJECT) + OOD
  + 参考EV（训练池同象限均值，非预测）+ 高热解释字段
  + model_version（打分所用冻结模型版本，取 shadow_t5_frozen.json 的 _frozen_at）
  + 人工操作（buy/skip + 原因 + 实际成交价 fill_price）——盘中人工录入，
    action 只属于候选×决策点维度，绝不写 sell（卖出是账户事件，走 trades 体系）
  + 人工提前退出（exit_price/exit_ts/exited_early）——卖出时 best-effort 回填，
    与 nr_5d 反事实标签并列，用于比较"提前退出 vs 持有到 T+5"，不污染 action 分组
  + 最终3日/5日扣费净收益、STOP——T+6 后由 backfill 自动回填

内生性注意（评审第4点）：action 是人工选择的结果（买的是认同的），不是随机
样本，只能当分层变量（人机一致时是否赚钱），不能直接当训练标签/监督信号。

积累后回答：FOCUS是否真的优于WATCH、REJECT是否避开亏损、AM/PM哪个头
更可靠、哪些月份/状态失效、人工否决是否有价值、模型为什么选错、OOD是否
正确警告。前瞻通过标准（评审拍板）：
  FOCUS 5日扣费收益为正；FOCUS优于WATCH；STOP率不恶化；多月稳定；
  不依赖少数股票或板块。
"""
import json
import os
import sqlite3
import time

import numpy as np
import pandas as pd

OUT_DIR = "D:/stargazer/stargazer_ml/pullback_train"
LBL = {y: os.path.join(OUT_DIR, f"dual_head_labels_{y}.parquet")
       for y in (2025, 2026)}
KT = "2.0"

SCHEMA = """CREATE TABLE IF NOT EXISTS shadow_decisions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    sig_date TEXT, dp_date TEXT, code TEXT, dp INTEGER, head TEXT,
    p_risk REAL, p_opp REAL, quadrant TEXT, ood INTEGER, ood_hits INTEGER,
    ev3_ref REAL, ev5_ref REAL,
    runup_5d REAL, runup_10d REAL, runup_20d REAL, pos20 REAL, rsi14 REAL,
    turnover_sum_5d REAL, turnover_max_5d REAL, limit_up_10d REAL,
    atr_ref REAL, entry REAL, src TEXT, subtype TEXT,
    model_version TEXT,
    action TEXT, action_reason TEXT, action_ts TEXT, fill_price REAL,
    exit_price REAL, exit_ts TEXT, exited_early INTEGER,
    nr_3d REAL, nr_5d REAL, stop_5d INTEGER, outcome_t2 TEXT, censored INTEGER,
    filled_at TEXT,
    UNIQUE(sig_date, code, dp))"""

# 评审第1/3/5点新增列：老库（CREATE TABLE IF NOT EXISTS 不会改结构）需要增量迁移
_ADDED_COLUMNS = (
    ("model_version", "TEXT"),
    ("fill_price", "REAL"),
    ("exit_price", "REAL"),
    ("exit_ts", "TEXT"),
    ("exited_early", "INTEGER"),
)


def connect(db_path):
    os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
    conn = sqlite3.connect(db_path, timeout=20)
    conn.execute(SCHEMA)
    have = {r[1] for r in conn.execute("PRAGMA table_info(shadow_decisions)")}
    for name, typ in _ADDED_COLUMNS:
        if name not in have:
            conn.execute(f"ALTER TABLE shadow_decisions ADD COLUMN {name} {typ}")
    conn.commit()
    return conn


def upsert_decisions(db_path, rows, model_version=None):
    """影子建议行（t5_shadow_advice.json 的 rows）落决策表；已有人工操作的
    行不覆盖（保留 action_* / fill_price / exit_*），只更新模型字段。
    model_version：本次打分所用冻结模型版本（bundle.meta._frozen_at）；
    行内自带 model_version 时以行内为准。"""
    conn = connect(db_path)
    n_ins = n_upd = 0
    for r in rows:
        cols = ("sig_date", "dp_date", "code", "dp", "head", "p_risk", "p_opp",
                "quadrant", "ood", "ood_hits", "ev3_ref", "ev5_ref",
                "runup_5d", "runup_10d", "runup_20d", "pos20", "rsi14",
                "turnover_sum_5d", "turnover_max_5d", "limit_up_10d",
                "atr_ref", "entry", "src", "subtype", "model_version")
        vals = [_int_if_digit(r.get("date")), _int_if_digit(r.get("dp_date")),
                str(r.get("code")),
                r.get("dp"), "AM" if int(r.get("dp") or 0) < 1200 else "PM",
                r.get("P_risk"), r.get("P_opp"), r.get("quadrant"),
                int(bool(r.get("ood"))), r.get("ood_hits"),
                r.get("ev3_ref"), r.get("ev5_ref"),
                r.get("runup_5d"), r.get("runup_10d"), r.get("runup_20d"),
                r.get("pos20"), r.get("rsi14"), r.get("turnover_sum_5d"),
                r.get("turnover_max_5d"), r.get("limit_up_10d"),
                r.get("atr_ref"), r.get("entry"), r.get("src"), r.get("subtype"),
                r.get("model_version") or model_version]
        cur = conn.execute(
            f"INSERT INTO shadow_decisions ({','.join(cols)}) "
            f"VALUES ({','.join('?' * len(cols))}) "
            "ON CONFLICT(sig_date, code, dp) DO UPDATE SET "
            + ",".join(f"{c}=excluded.{c}" for c in cols[3:]),
            vals)
        n_ins += 1
    conn.commit(); conn.close()
    return n_ins


def _int_if_digit(v):
    s = str(v)
    return int(s) if s.isdigit() else s


def record_action(db_path, sig_date, code, dp, action, reason="",
                  fill_price=None):
    """人工录入候选×决策点上的操作：action 只允许 buy / skip。

    评审第1点：sell 是持仓事件，不属于候选决策维度，这里显式拒绝；
    人工提前退出走 record_exit，写 exit_* 列，与 action 分组隔离。
    fill_price：点击买入时的实际成交价（评审第5点），用于精确计算
    "人工实际收益"，不必事后反查分钟快照。
    必须绑定具体 sig_date + dp（评审第2点）：前端按钮所在行=记录行，
    严禁对一只票的全部决策点批量记 buy。
    """
    if action not in ("buy", "skip"):
        raise ValueError(f"action 只允许 buy/skip，收到 {action!r}（卖出请走 trades/exit）")
    conn = connect(db_path)
    if fill_price not in (None, ""):
        try:
            fill_price = float(fill_price)
        except (TypeError, ValueError):
            fill_price = None
    if action == "buy" and fill_price is not None:
        conn.execute(
            "UPDATE shadow_decisions SET action=?, action_reason=?, action_ts=?, "
            "fill_price=? WHERE sig_date=? AND code=? AND dp=?",
            (action, reason, time.strftime("%Y-%m-%d %H:%M:%S"), fill_price,
             _int_if_digit(sig_date), str(code), int(dp)))
    else:
        conn.execute(
            "UPDATE shadow_decisions SET action=?, action_reason=?, action_ts=? "
            "WHERE sig_date=? AND code=? AND dp=?",
            (action, reason, time.strftime("%Y-%m-%d %H:%M:%S"),
             _int_if_digit(sig_date), str(code), int(dp)))
    conn.commit(); conn.close()


def record_exit(db_path, code, exit_price, reason=""):
    """卖出时 best-effort 回填最近一笔未平仓的人工买入决策（评审第1点）。

    不新建/修改 action 行，只写 exit_price/exit_ts/exited_early：
      exited_early=1 卖出发生在 T+5 观察窗内（dp_date + 7 自然日近似），
                 0 为窗口外退出；分析时与 nr_5d（持有到期反事实）并列比较。
    返回被标记的行 id；没有未平仓买入时返回 None（不报错，可能是模型外买入）。
    """
    conn = connect(db_path)
    row = conn.execute(
        "SELECT id, dp_date FROM shadow_decisions "
        "WHERE code=? AND action='buy' AND exit_price IS NULL "
        "ORDER BY sig_date DESC, dp DESC LIMIT 1",
        (str(code),)).fetchone()
    if not row:
        conn.close(); return None
    rid, dp_date = row
    exited_early = None
    try:
        d = str(dp_date)[:8].replace("-", "")
        if len(d) == 8 and d.isdigit():
            import datetime as _dt
            horizon_end = _dt.datetime.strptime(d, "%Y%m%d") + _dt.timedelta(days=7)
            exited_early = int(_dt.datetime.now() <= horizon_end)
    except (TypeError, ValueError):
        pass
    conn.execute(
        "UPDATE shadow_decisions SET exit_price=?, exit_ts=?, exited_early=? "
        "WHERE id=?",
        (float(exit_price), time.strftime("%Y-%m-%d %H:%M:%S"),
         exited_early, rid))
    conn.commit(); conn.close()
    return rid


def backfill_outcomes(db_path, years=(2025, 2026)):
    """用 dual_head_labels 回填最终3/5日收益；只填未回填行。"""
    conn = connect(db_path)
    todo = pd.read_sql(
        "SELECT sig_date, dp_date, code, dp FROM shadow_decisions "
        "WHERE nr_5d IS NULL", conn)
    if not len(todo):
        conn.close(); return 0
    n = 0
    for y in years:
        lbl = pd.read_parquet(LBL[y])
        lbl["code"] = lbl["code"].astype(str)
        sub = todo[todo["dp_date"].astype(str).str.startswith(str(y))]
        if not len(sub):
            continue
        m = sub.copy()
        m["dp_date"] = m["dp_date"].astype(int)
        m["code"] = m["code"].astype(str)
        lbl_y = lbl.copy()
        lbl_y["code"] = lbl_y["code"].astype(str)
        m = m.merge(lbl_y, on=["code", "dp_date", "dp"], how="inner")
        for _, r in m.iterrows():
            conn.execute(
                "UPDATE shadow_decisions SET nr_3d=?, nr_5d=?, stop_5d=?, "
                "outcome_t2=?, censored=?, filled_at=? "
                "WHERE sig_date=? AND code=? AND dp=?",
                (float(r["o_net_ret_3d"]) if pd.notna(r["o_net_ret_3d"]) else None,
                 float(r["o_net_ret_5d"]) if pd.notna(r["o_net_ret_5d"]) else None,
                 int(r["o_stop_5d"]) if pd.notna(r["o_stop_5d"]) else None,
                 r.get(f"outcome_t{KT}"),
                 int(r["censored"]) if "censored" in m.columns
                 and pd.notna(r["censored"]) else None,
                 time.strftime("%Y-%m-%d %H:%M:%S"),
                 _int_if_digit(r["sig_date"]), str(r["code"]), int(r["dp"])))
            n += 1
    conn.commit(); conn.close()
    return n


def analyze(db_path, out_dir=OUT_DIR):
    """对应影子验证协议的七个问题；样本不足时如实报告 N。"""
    conn = connect(db_path)
    df = pd.read_sql("SELECT * FROM shadow_decisions WHERE nr_5d IS NOT NULL",
                     conn)
    conn.close()
    print(f"已回填样本 {len(df):,} 行（有收益结果）")
    if len(df) < 100:
        print("样本不足（<100），只出基础分布，不做结论。")
    rows_path = os.path.join(out_dir, "shadow_decisions_filled.csv")
    df.to_csv(rows_path, index=False)

    def grp(d, keys, name):
        g = d.groupby(keys).agg(
            n=("nr_5d", "size"), nr3=("nr_3d", "mean"),
            nr5=("nr_5d", "mean"), stop5=("stop_5d", "mean")).reset_index()
        g.insert(0, "cut", name)
        return g

    parts = [grp(df, ["quadrant"], "Q"),
             grp(df, ["quadrant", "head"], "Q×头")]
    parts.append(grp(df.assign(
        month=df["sig_date"].astype(str).str[:6]), ["quadrant", "month"], "Q×月"))
    parts.append(grp(df.assign(
        year=df["sig_date"].astype(str).str[:4]), ["quadrant", "year"], "Q×年"))
    parts.append(grp(df, ["ood"], "OOD"))
    parts.append(grp(df[df["action"].notna()], ["quadrant", "action"],
                     "人工操作"))
    res = pd.concat([p for p in parts if p is not None], ignore_index=True)
    out = os.path.join(out_dir, "shadow_analysis.csv")
    res.to_csv(out, index=False)
    print(res.to_string(index=False))
    print("->", out, "|", rows_path)
    return res


if __name__ == "__main__":
    import sys
    db = sys.argv[1] if len(sys.argv) > 1 else \
        "D:/stargazer/stargazer_desktop/data/model_state.sqlite"
    cmd = sys.argv[2] if len(sys.argv) > 2 else "analyze"
    if cmd == "backfill":
        print("backfilled", backfill_outcomes(db))
    else:
        analyze(db)
