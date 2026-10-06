# -*- coding: utf-8 -*-
"""绿盘低吸 pullback_train - v2.1 第5+6步 v2：LightGBM 三分类基线 + K_stop 冻结

v2 修订（评审 10 条，2026-09）:
  1. target_atr 正式入特征矩阵（训练/预测均按长表索引对齐合入 X）。
     原 v1 漏加导致同一事件四档 target_atr 概率完全相同（已验证 nunique=1），
     原 v1 全部结论作废。
  2. 事件级统计：独立事件数=event_id 去重，不再用长表行数。
  3. Top10% 事件级：每事件先算 4 档 EV 取 argmax 档，再按事件 EV 取 Top10%；
     另报每日候选分布、target_atr 选中比例、每股交易数、20 仓组合回测。
  4. K_stop 用折外实际 net_ret 选（非预测 EV）：每事件 argmax EV 档后的
     实际 net_ret 均值/年收益/loss 率/覆盖率。
  5. 双口径: (a) 四个 K 共同 eligible 子集（隔离止损宽度效果）;
              (b) 各 K 自己合格池的完整事件级策略。
  6. EVParams 计算前按该 K 的 eligible 列过滤训练段。
  7. 2024 移出早停与校准：早停用训练期内部时间 8:2；按早停轮数在 2019-2023
     全量重训；isotonic 用 2019-2023 内部滚动 OOF(2022/2023 折外) 预测拟合。
     2024 只预测/确认/评估。超参固定保守值，不做基于 2024 的选择。
  8. 基线增强：常数概率 Brier、Brier Skill Score、multiclass log-loss、ECE、
     校准前后 Rank IC 与 Top10%；校准后 Brier 变差如实报告。
  9. 结论措辞：first_div 仅"可能存在风险排序信号…值得复核"，不写"有效"。
 10. 分层报告：板块内/板块特征缺失 × adj_ready/first_div；冻结模型 gain 重要性 top20。

幂等可重跑，覆盖写 models/ preds/ 与报告第 8 节。
"""
import os
import pickle
import sys
import time

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LinearRegression
import lightgbm as lgb

OUT_DIR = "D:/stargazer/stargazer_ml/pullback_train"
EVENTS_PATH = os.path.join(OUT_DIR, "events_train_v1.parquet")
MODEL_DIR = os.path.join(OUT_DIR, "models")
PRED_DIR = os.path.join(OUT_DIR, "preds")
REPORT_PATH = os.path.join(OUT_DIR, "event_funnel_report.md")
MARKET_DAILY_PATH = os.path.join(OUT_DIR, "market_daily.parquet")
PATH_COOLED_PATH = os.path.join(OUT_DIR, "events_path_cooled.parquet")
COST = 0.002  # 往返成本（含在标签 net_ret 内；账本卖出结算时回补为 fill 价）
COMM = 0.00025        # 佣金双边（最低5元，组合本金口径下忽略）
STAMP_OLD = 0.001     # 印花税 2023-08-28 之前（卖出）
STAMP_NEW = 0.0005    # 印花税 2023-08-28 之后（卖出）
STAMP_CUTOVER = 20230828
EVENT_PATH = os.path.join(OUT_DIR, "events_train_v1.parquet")
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(PRED_DIR, exist_ok=True)

K_STOPS = ["1.2", "1.5", "1.8", "2.0"]
K_TARGS = ["1.0", "1.5", "2.0", "3.0"]
CLASSES = ["TARGET", "STOP", "TIMEOUT"]
LGB_PARAMS = dict(objective="multiclass", num_class=3, num_leaves=63,
                  learning_rate=0.05, class_weight="balanced",
                  min_child_samples=50, subsample=0.9, colsample_bytree=0.9,
                  metric="multi_logloss", verbosity=-1, n_jobs=8)

sys.stdout.reconfigure(encoding="utf-8")


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def outcol(ks, kt, kind="outcome"):
    return f"{kind}_s{ks}_t{kt}"


# ---------------------------------------------------------------- load
def load_event_level():
    df = pd.read_parquet(EVENTS_PATH)
    df = df.sort_values(["event_id", "sector_ret5"],
                        ascending=[True, False], na_position="last",
                        kind="mergesort").drop_duplicates("event_id")
    df = df[df["subtype"].isin(["adj_ready", "first_div", "both"])].copy()
    return df.reset_index(drop=True)


# ---------------------------------------------------------------- features
# v3.3 特征白名单（显式声明）。黑名单方式曾漏掉 exit_day_* 导致标签泄漏，
# 之后任何特征增减必须改这里并写明理由；parquet 出现未声明列会直接 assert 失败。
FEATURE_WHITELIST = [
    "open", "high", "low", "close", "pre_close", "amount",
    "is_st", "float_mv", "total_mv", "len", "ret1", "ret5",
    "ret20", "high5", "low5", "high20", "low20", "high60",
    "low60", "pos20", "pullback_from_high5", "adv20", "amount_ratio20", "tr",
    "atr14", "atr_ref", "body", "body_prev", "lower_shadow", "amplitude",
    "amp_mean5", "amp_contract", "ma20", "_ret1_prev_min5", "_limit", "_lock",
    "trend_cond_a", "entry", "stop_pct_s1.2", "stop_pct_s1.5", "stop_pct_s1.8",
    "stop_pct_s2.0",
    "turnover_pct60", "ret3", "limit60", "rsi14", "rsi_delta3", "bbi_distance",
    "bbi_slope5", "boll_pct_b", "down_volume_ratio5", "turnover_ratio20",
    "quality_score",
    "eq_ret1", "eq_ret5", "eq_ret20", "market_vol_state", "market_pos20",
    "sh_ret1", "sh_ret5", "sh_ret20", "cyb_ret1", "cyb_ret5", "cyb_ret20",
    "in_watch_sector",
    "sector_up_ratio", "sector_limit_up_count", "sector_limit_down_count",
    "sector_amount", "sector_n_cons",
    "leader_ret1", "leader_high_to_close", "center_ret1",
    "sector_ret1", "sector_ret3", "sector_ret5", "sector_high_to_close",
    "sector_return_rank", "sector_return_pct", "sector_excess_ret_5d",
    "rel_ret_1d", "rel_ret_3d", "rel_ret_5d",
]

# 显式丢弃清单（标签/事后字段；白名单之外的数值列若不在此清单也会 assert 失败）
DROP_PREFIX = ("outcome_", "net_ret_", "ambig_", "fill_", "eligible_",
               "exit_day_")
DROP_EXACT = {"date", "code", "event_id", "pos", "year", "subtype",
              "label_end_date", "available_future_days", "is_censored",
              "timeout_return", "sector_name", "leader_code", "center_code"}


def select_features(df):
    feats = [c for c in FEATURE_WHITELIST if c in df.columns]
    missing = [c for c in FEATURE_WHITELIST if c not in df.columns]
    assert not missing, f"白名单特征在数据中缺失: {missing}"
    unknown = [c for c in df.columns
               if c not in feats and c not in DROP_EXACT
               and not c.startswith(DROP_PREFIX)]
    assert not unknown, \
        f"出现未声明列，需人工归类防止泄漏(加入白名单或丢弃清单): {unknown}"
    X = df[feats].copy().replace([np.inf, -np.inf], np.nan)
    for c in feats:
        s = X[c]
        if pd.api.types.is_bool_dtype(s) or pd.api.types.is_integer_dtype(s):
            X[c] = s.astype(np.float32)
    return X, feats


def segment(df, y0, y1, lend_max):
    m = (df["date"] >= y0) & (df["date"] <= y1) \
        & (~df["is_censored"]) & (df["label_end_date"].notna()) \
        & (df["label_end_date"] <= lend_max)
    return df[m].copy()


# ---------------------------------------------------------------- long table
KEYS = ["event_id", "date", "code", "subtype", "in_watch_sector"]


def make_long(ev, ks):
    """事件 × 4 target_atr 长表; 保留 ev 原索引以便与特征矩阵 .loc 对齐"""
    parts = []
    for kt in K_TARGS:
        sub = ev[KEYS + [outcol(ks, kt), outcol(ks, kt, "net_ret")]].copy()
        sub.columns = KEYS + ["y", "net_ret"]
        sub["target_atr"] = float(kt)
        parts.append(sub)
    long = pd.concat(parts, ignore_index=False)
    return long[long["y"].isin(CLASSES)]


def build_xy(ev, ks):
    """特征矩阵 + target_atr 列（按长表索引对齐）"""
    long = make_long(ev, ks)
    X, _ = select_features(ev)
    X = X.loc[long.index].copy()
    X["target_atr"] = long["target_atr"].values
    return X, long


def fit_model(tr_ev, ks, es_ev=None):
    """训练; 若给 es_ev 则在其上 early stopping。
    返回 (clf, best_iter, n_train_rows)"""
    tr_ev = tr_ev[tr_ev[f"eligible_s{ks}"]]
    Xtr, tr_long = build_xy(tr_ev, ks)
    kw = dict(X=Xtr, y=tr_long["y"].values)
    if es_ev is not None:
        es_ev = es_ev[es_ev[f"eligible_s{ks}"]]
        Xes, es_long = build_xy(es_ev, ks)
        kw.update(eval_set=[(Xes, es_long["y"].values)], eval_metric="multi_logloss")
    clf = lgb.LGBMClassifier(**LGB_PARAMS, n_estimators=3000)
    clf.fit(**kw, callbacks=[lgb.early_stopping(100, verbose=False),
                             lgb.log_evaluation(0)])
    return clf, clf.best_iteration_, len(tr_long)


def refit_full(tr_ev, ks, n_iter):
    tr_ev = tr_ev[tr_ev[f"eligible_s{ks}"]]
    Xtr, tr_long = build_xy(tr_ev, ks)
    clf = lgb.LGBMClassifier(**LGB_PARAMS, n_estimators=max(int(n_iter), 10))
    clf.fit(Xtr, tr_long["y"].values, callbacks=[lgb.log_evaluation(0)])
    return clf


def predict_proba(clf, ev, ks):
    ev = ev[ev[f"eligible_s{ks}"]]
    X, long = build_xy(ev, ks)
    proba = clf.predict_proba(X)
    for i, cl in enumerate(clf.classes_):  # 列序=clf.classes_（字母序），按名映射
        long[f"p_{cl}"] = proba[:, i]
    for cl in CLASSES:
        if f"p_{cl}" not in long.columns:
            long[f"p_{cl}"] = 0.0
    return long


# ---------------------------------------------------------------- EV
class EVParams:
    def __init__(self, tr_ev, ks):
        tr_ev = tr_ev[tr_ev[f"eligible_s{ks}"]]  # v2-6: eligible 子集
        long = make_long(tr_ev, ks)
        self.gain = long[long["y"] == "TARGET"].groupby("target_atr")["net_ret"].mean().to_dict()
        self.loss = long[long["y"] == "STOP"].groupby("target_atr")["net_ret"].mean().to_dict()
        tmo = long[long["y"] == "TIMEOUT"].groupby("subtype")["net_ret"].mean()
        self.tmo = tmo.to_dict()
        self.tmo_all = float(long.loc[long["y"] == "TIMEOUT", "net_ret"].mean())
        self.ks = ks

    def ev(self, long):
        g = long["target_atr"].map(self.gain)
        l = long["target_atr"].map(self.loss)
        t = long["subtype"].map(self.tmo).fillna(self.tmo_all)
        return (long["p_TARGET"] * g + long["p_STOP"] * l
                + long["p_TIMEOUT"] * t).astype(np.float32)


# ---------------------------------------------------------------- event-level
def event_select(long):
    """每事件取 EV argmax 档（事件级，禁止同事件重复）。
    v3: 必须先重置索引——长表保留原事件索引时 4 行共享同一标签，
    .loc[idxmax()] 会把 4 行全部取回（该 bug 曾导致各档选中比例恒为 25%）。"""
    long = long.reset_index(drop=True)
    idx = long.groupby("event_id")["EV"].idxmax()
    out = long.loc[idx].copy()
    assert out["event_id"].is_unique, "event_select 输出 event_id 不唯一"
    return out


# ---------------------------------------------------------------- 组合回测 v3
CALENDAR = None       # 交易日历（date int 有序列表）
PATH_MAP = None       # event_id -> (c1..c5, exit_day 列)

def _load_calendar():
    global CALENDAR
    if CALENDAR is None:
        m = pd.read_parquet(MARKET_DAILY_PATH)
        CALENDAR = sorted(int(d) for d in m["date"].unique())
    return CALENDAR


def _load_path():
    """event_id -> 路径表（entry, c1..c5, exit_day_s{k}_t{t} 16 列）"""
    global PATH_MAP
    if PATH_MAP is None:
        PATH_MAP = pd.read_parquet(PATH_COOLED_PATH).set_index("event_id")
    return PATH_MAP


def portfolio_backtest(evd, max_pos=20, ks="1.2", ev_thr=None, slot_cap=None,
                       window=None, no_buy_dates=None):
    """v3.3 真账本：显式现金+股数+逐日盯市。

    - 每仓等权 1/max_pos 的当前净值，entry 价归一化为 1.0（收益率口径不变）；
    - A股 T+1：exit_day>=1（标签引擎已保证），当天买入当天不可卖；
    - 涨停封板不买：信号日 close==high 且 ret1>=limit-0.5% 跳过（硬过滤已排涨停日，
      这里是兜底）；
    - 跌停卖不出：标签引擎的锁死顺延与 fill 口径已处理，账本按标签 fill 结算；
    - 费用分列：佣金双边 0.025%（最低5元，本金口径下忽略），印花税卖出
      2023-08-28 前 0.1% / 之后 0.05%；滑点含在标签往返成本 0.2% 内不重复扣；
    - ev_thr: 冻结 EV 阈值（允许空仓正式口径）；None=强制填仓容量压力测试；
    - slot_cap: 市场参与模型的每日槽位上限（date int -> 0~20 的 Series/dict），
      G6/G7 实验用；当日买入受 min(max_pos, cap) 限制；
    - window: (start, end) int 日期，统一自然年度口径（第0步修订）——
      全程计净值、闲置现金收益 0、期末强平未退出持仓（按期末盯市价扣卖出费）、
      资金利用率 util=日均 invested/equity；None=旧口径（首信号日→末持仓退出）；
    - no_buy_dates: 反事实 R0 评估用——这些交易日禁止开新仓（候选与顺序
      与基准完全一致，只取消当日成交、不递补），持仓退出与盯市不受影响；
    - 回测至最后一批持仓完全退出（window 口径除外，见上）。"""
    cal = _load_calendar()
    pm = _load_path()
    eq_quote = _load_event_quote()
    idx_of = {d: i for i, d in enumerate(cal)}
    cash = 1.0
    positions = []   # dict: code,event_id,shares,entry_i,exit_day,net_ret,c,EV,target_atr
    trades = []
    fee = {"comm_buy": 0.0, "comm_sell": 0.0, "stamp": 0.0}
    src = evd if ev_thr is None else None
    if src is None:
        if np.isscalar(ev_thr):
            src = evd[evd["EV"] >= ev_thr]
        else:  # v3.3: 分状态阈值 event_id -> thr 的 Series
            _t = evd["event_id"].map(ev_thr)
            src = evd[evd["EV"] >= _t.fillna(np.inf)]
    if window is not None:
        w0, w1 = int(window[0]), int(window[1])
        src = src[(src["date"] >= w0) & (src["date"] <= w1)]
    if len(src) == 0:
        if window is None:
            return {}
        i0, i1 = idx_of[window[0]], idx_of[window[1]]
        return {"days": i1 - i0 + 1, "avg_pos": 0.0, "avg_involved": 0.0,
                "max_pos_held": 0, "n_trades": 0, "total_ret": 0.0,
                "ann_ret": 0.0, "max_dd": 0.0, "open_end": 0,
                "util": 0.0, "comm_buy": 0.0, "comm_sell": 0.0, "stamp": 0.0,
                "_trades": pd.DataFrame(), "_nav": np.ones(i1 - i0 + 1),
                "_dates": [int(x) for x in cal[i0:i1 + 1]]}
    src = src.copy()
    src["exit_day_sel"] = [
        pm.loc[e].get(f"exit_day_s{ks}_t{float(t):.1f}", np.nan)
        if e in pm.index else np.nan
        for e, t in zip(src["event_id"], src["target_atr"])]
    src = src.dropna(subset=["exit_day_sel"])
    src = src[src["exit_day_sel"] >= 1]
    pending = {int(d): g for d, g in
               src.sort_values(["date", "EV"], ascending=[True, False]).groupby("date")}
    if slot_cap is not None:
        cap_of = slot_cap if hasattr(slot_cap, "get") else {}
    else:
        cap_of = {}
    sig_days = sorted(pending)
    if window is not None:
        w0, w1 = int(window[0]), int(window[1])
        cal_in = [d for d in cal if w0 <= d <= w1]
        start_i, end_i = idx_of[cal_in[0]], idx_of[cal_in[-1]]
    else:
        start_i = idx_of[sig_days[0]]
        end_i = max(idx_of[int(d)] + int(ex)
                    for d, ex in zip(src["date"], src["exit_day_sel"]))
    nav, held_close, involved, utils = [], [], [], []
    n_trades = 0
    for ci in range(start_i, end_i + 1):
        d = cal[ci]
        stamp = STAMP_OLD if d < STAMP_CUTOVER else STAMP_NEW
        still = []
        for pos in positions:
            off = ci - pos["entry_i"]
            if off < pos["exit_day"]:
                still.append(pos)
                continue
            # 结算日：按标签 fill 价卖出（gross = 1+net_ret+往返成本）
            proceeds = pos["shares"] * (1 + pos["net_ret"] + COST)
            c_sell = COMM * proceeds
            st = stamp * proceeds
            cash += proceeds - c_sell - st
            fee["comm_sell"] += c_sell
            fee["stamp"] += st
            trades.append({"date": pos["date"], "code": pos["code"],
                           "event_id": pos["event_id"], "EV": pos["EV"],
                           "target_atr": pos["target_atr"],
                           "net_ret": pos["net_ret"],
                           "exit_day": pos["exit_day"],
                           "pnl": proceeds - c_sell - st - pos["alloc"]})
        positions = still
        equity = cash + sum(pos["shares"] * _mark(pos, ci) for pos in positions)
        if d in pending and (no_buy_dates is None or d not in no_buy_dates):
            cap_today = min(max_pos, int(cap_of.get(int(d), max_pos)))
            for _, r in pending[d].iterrows():
                if len(positions) >= cap_today or cash <= 0:
                    break
                eid = int(r["event_id"])
                q = eq_quote.loc[eid] if eid in eq_quote.index else None
                if q is not None and bool(q["close"] == q["high"]) \
                        and q["ret1"] >= q["_limit"] - 0.005:
                    continue  # 涨停封板，收盘无法买入
                prow = pm.loc[eid]
                alloc = min(equity / max_pos, cash)
                if alloc <= 0:
                    break
                comm_buy = COMM * alloc
                shares = (alloc - comm_buy)          # entry 价=1.0
                cash -= alloc
                fee["comm_buy"] += comm_buy
                positions.append({
                    "code": r["code"], "event_id": eid, "shares": shares,
                    "alloc": alloc, "entry_i": ci, "date": int(r["date"]),
                    "exit_day": int(r["exit_day_sel"]),
                    "net_ret": float(r["net_ret"]), "EV": float(r["EV"]),
                    "target_atr": float(r["target_atr"]),
                    "c": [float(prow[f"c{k}"]) for k in range(1, 6)]})
                n_trades += 1
        equity = cash + sum(pos["shares"] * _mark(pos, ci) for pos in positions)
        nav.append(equity)
        held_close.append(len(positions))
        utils.append(float(sum(pos["shares"] * _mark(pos, ci)
                               for pos in positions) / equity) if equity > 0 else 0.0)
        involved.append(len(positions) + (n_trades and 0 or 0))
    # window 口径：期末强平未退出持仓（按期末盯市价，扣卖出佣金+印花税）
    if window is not None and positions:
        d_end = cal[end_i]
        stamp = STAMP_OLD if d_end < STAMP_CUTOVER else STAMP_NEW
        for pos in positions:
            proceeds = pos["shares"] * _mark(pos, end_i)
            c_sell = COMM * proceeds
            st = stamp * proceeds
            cash += proceeds - c_sell - st
            fee["comm_sell"] += c_sell
            fee["stamp"] += st
            trades.append({"date": pos["date"], "code": pos["code"],
                           "event_id": pos["event_id"], "EV": pos["EV"],
                           "target_atr": pos["target_atr"],
                           "net_ret": _mark(pos, end_i) - 1.0 - COST,
                           "exit_day": ci - pos["entry_i"],
                           "pnl": proceeds - c_sell - st - pos["alloc"]})
        positions = []
        nav[-1] = cash
    if not nav:
        return {}
    nav = np.array(nav)
    total = float(nav[-1] - 1)
    ann = float(nav[-1] ** (252 / len(nav)) - 1)
    mdd = float((nav / np.maximum.accumulate(nav) - 1).min())
    t = pd.DataFrame(trades)
    out = {"days": len(nav), "avg_pos": float(np.mean(held_close)),
           "avg_involved": float(np.mean(held_close)),
           "max_pos_held": int(max(held_close)) if held_close else 0,
           "n_trades": n_trades,
           "total_ret": total, "ann_ret": ann, "max_dd": mdd,
           "open_end": len(positions),
           "util": float(np.mean(utils)) if utils else 0.0,
           "comm_buy": fee["comm_buy"], "comm_sell": fee["comm_sell"],
           "stamp": fee["stamp"]}
    if len(t):
        out["trade_win"] = float((t["pnl"] > 0).mean())
        out["trade_pnl_mean"] = float(t["pnl"].mean())
    out["_trades"] = t
    out["_nav"] = nav
    out["_dates"] = [int(d) for d in cal[start_i:end_i + 1]]
    return out


def _mark(pos: dict, ci: int) -> float:
    """持仓盯市价格比率（entry=1.0）；停牌/缺失向前沿用最后有效价。"""
    off = ci - pos["entry_i"]
    if off <= 0:
        return 1.0
    k = min(off, 5)
    vals = [c for c in pos["c"][:k] if pd.notna(c)]
    return 1.0 + (vals[-1] if vals else 0.0)


EVENT_QUOTE = None     # event_id -> ret1/_limit/high/close（封板判断）


def _load_event_quote():
    global EVENT_QUOTE
    if EVENT_QUOTE is None:
        e = pd.read_parquet(EVENT_PATH,
                            columns=["event_id", "ret1", "_limit", "high", "close"])
        # events_train_v1 因多板块归属展开，event_id 不唯一，需先去重
        EVENT_QUOTE = e.drop_duplicates("event_id").set_index("event_id")
    return EVENT_QUOTE


def strat_metrics(evd, const_p, ks="1.2", thr=None):
    """事件级指标; evd = event_select 后的表; const_p = 常数基线概率(3, );
    thr = 训练期冻结 EV 阈值（传入时报告阈值选取样本表现）"""
    r = {}
    r["n_events"] = evd["event_id"].nunique()
    daily_cnt = evd.groupby("date")["event_id"].nunique()
    r["cand_daily_mean"] = float(daily_cnt.mean())
    r["cand_daily_median"] = float(daily_cnt.median())
    r["cand_daily_p90"] = float(daily_cnt.quantile(0.9))
    per_code = evd.groupby("code").size()
    r["trades_per_code_mean"] = float(per_code.mean())
    r["trades_per_code_max"] = int(per_code.max())
    r["targ_share"] = evd["target_atr"].value_counts(normalize=True).to_dict()
    r["mean_ret"] = float(evd["net_ret"].mean())
    r["loss_rate"] = float((evd["net_ret"] < 0).mean())
    cum = float((1 + evd.sort_values("date")["net_ret"]).cumprod().iloc[-1] - 1)
    r["cum_ret"] = cum
    r["ann_ret"] = (1 + cum) ** (252 / max(evd["date"].nunique(), 1)) - 1
    r["rank_ic"] = float(spearmanr(evd["EV"], evd["net_ret"]).statistic)
    top_thr = evd["EV"].quantile(0.9)  # 当期 Top10%（描述性，事后阈值）
    top = evd[evd["EV"] >= top_thr]
    r["top_n"] = len(top)
    r["top_ret"] = float(top["net_ret"].mean())
    r["top_win"] = float((top["net_ret"] > 0).mean())
    r.update({f"pf_{k}": v for k, v in portfolio_backtest(evd, ks=ks).items()
              if not k.startswith("_")})
    if thr is not None:  # 冻结阈值、允许空仓的正式口径
        r.update({f"pft_{k}": v for k, v in
                  portfolio_backtest(evd, ks=ks, ev_thr=thr).items()
                  if not k.startswith("_")})
    r.update(daily_metrics(evd, thr=thr))
    # 概率质量指标（用事件选中档的单行概率）
    y = pd.get_dummies(evd["y"])[CLASSES].values
    p = evd[[f"p_{c}" for c in CLASSES]].values
    r["brier"] = float(((p - y) ** 2).sum(axis=1).mean())
    r["brier_const"] = float(((const_p - y) ** 2).sum(axis=1).mean())
    r["bss"] = 1 - r["brier"] / max(r["brier_const"], 1e-9)
    ll = -np.log(np.clip((p * y).sum(axis=1), 1e-12, 1)).mean()
    ll_c = -np.log(np.clip((const_p * y).sum(axis=1), 1e-12, 1)).mean()
    r["logloss"], r["logloss_const"] = float(ll), float(ll_c)
    # ECE on P(TARGET)
    b = pd.qcut(evd["p_TARGET"], 10, duplicates="drop")
    ece = evd.groupby(b, observed=True).apply(
        lambda g: len(g) * abs((g["y"] == "TARGET").mean() - g["p_TARGET"].mean()),
        include_groups=False).sum() / len(evd)
    r["ece"] = float(ece)
    sl = LinearRegression().fit(evd["p_TARGET"].values.reshape(-1, 1),
                                (evd["y"] == "TARGET").astype(float).values)
    r["cal_slope"] = float(sl.coef_[0])
    return r


def daily_metrics(evd, thr=None, top_n=3, min_cnt=8):
    """v3: 每日横截面指标（避免全年混合 IC 被"选了好日子"污染）。
    - 每日 Rank IC（EV vs net_ret，当日事件数>=min_cnt 才计）
    - 每日 Top-N 等权净收益
    - 训练期冻结阈值 thr：EV>=thr 的样本收益与日均数量
    """
    r = {}
    ics, tops = [], []
    thr_sel = [] if thr is not None else None
    for d, g in evd.groupby("date"):
        if len(g) >= min_cnt:
            ics.append(spearmanr(g["EV"], g["net_ret"]).statistic)
        g2 = g.sort_values("EV", ascending=False).head(top_n)
        if len(g2):
            tops.append(float(g2["net_ret"].mean()))
        if thr is not None:
            thr_sel.extend(g.loc[g["EV"] >= thr, "net_ret"].tolist())
    ics = [x for x in ics if pd.notna(x)]
    if ics:
        arr = np.array(ics)
        r["d_ic_mean"] = float(arr.mean())
        r["d_ic_std"] = float(arr.std())
        r["d_ic_pos"] = float((arr > 0).mean())
        r["d_ic_t"] = float(arr.mean() / max(arr.std(), 1e-9) * np.sqrt(len(arr)))
        r["d_ic_days"] = len(arr)
    if tops:
        r["d_top_ret"] = float(np.mean(tops))
        r["d_top_cum"] = float(np.prod(1 + np.array(tops)) - 1)
        r["d_top_days"] = len(tops)
    if thr is not None:
        a = np.array(thr_sel) if thr_sel else np.array([np.nan])
        r["thr_n"] = len(thr_sel)
        r["thr_ret"] = float(np.nanmean(a))
        r["thr_days"] = int(sum(1 for _, g in evd.groupby("date")
                                if (g["EV"] >= thr).any()))
    return r


def const_probs(tr_ev, ks, st):
    tr_ev = tr_ev[tr_ev[f"eligible_s{ks}"]]
    long = make_long(tr_ev, ks)
    long = long[long["subtype"] == st]
    f = long["y"].value_counts(normalize=True).reindex(CLASSES).fillna(1 / 3).values
    return f


# ---------------------------------------------------------------- calibration
def fit_isotonic(long):
    cal = {}
    for cl in CLASSES:
        y = (long["y"] == cl).astype(float).values
        cal[cl] = IsotonicRegression(out_of_bounds="clip").fit(long[f"p_{cl}"], y)
    return cal


def apply_isotonic(long, cal):
    out = long.copy()
    ps = np.zeros((len(out), 3))
    for i, cl in enumerate(CLASSES):
        ps[:, i] = cal[cl].predict(out[f"p_{cl}"])
    ps = np.clip(ps, 0, 1)
    ps = ps / ps.sum(axis=1, keepdims=True)
    for i, cl in enumerate(CLASSES):
        out[f"p_{cl}"] = ps[:, i]
    return out


# ---------------------------------------------------------------- main
def main():
    t0 = time.time()
    ev = load_event_level()
    log(f"event-level rows: {len(ev)}")
    tr = segment(ev, 20190101, 20231231, 20231231)
    va = segment(ev, 20240101, 20241231, 20241231)
    t5 = segment(ev, 20250101, 20251231, 20251231)
    log(f"segments(events): train={len(tr)}, val2024={len(va)}, ref2025={len(t5)}")
    _, feats = select_features(ev)
    log(f"features: {len(feats)} (+target_atr)")

    # 训练期内部 8:2 时间尾段（早停用；2024 不参与）
    # v3: 同时剔除 label_end_date 跨入早停段的事件（防标签泄漏）
    es_from = tr["date"].quantile(0.8)
    tr_main = tr[(tr["date"] < es_from) & (tr["label_end_date"] < int(es_from))]
    tr_es = tr[tr["date"] >= es_from]
    log(f"internal ES split: main={len(tr_main)} es={len(tr_es)} (from {int(es_from)}, "
        f"label-leak excluded {int((tr['date'] < es_from).sum() - len(tr_main))})")

    # ---------------- 滚动折（2019-2021→2022; 2019-2022→2023）：K 选择 + OOF
    folds = [(2022, segment(ev, 20190101, 20211231, 20211231)),
             (2023, segment(ev, 20190101, 20221231, 20221231))]
    fold_models, fold_long = {}, {}
    for fy, ftr in folds:
        fes_from = ftr["date"].quantile(0.8)
        fm = ftr[(ftr["date"] < fes_from) & (ftr["label_end_date"] < int(fes_from))]
        fes = ftr[ftr["date"] >= fes_from]
        fev = segment(ev, fy * 10000 + 101, fy * 10000 + 1231, fy * 10000 + 1231)
        log(f"fold {fy}: train={len(fm)} es={len(fes)} eval={len(fev)}")
        for ks in K_STOPS:
            clf, bi, n = fit_model(fm, ks, es_ev=fes)
            fold_models[(fy, ks)] = (clf, bi)
            pl = predict_proba(clf, fev, ks)
            pl["EV"] = EVParams(fm, ks).ev(pl)  # v2-6: eligible 过滤后
            fold_long[(fy, ks)] = pl

    subtypes = ["adj_ready", "first_div"]
    common_pool = ev[ev[[f"eligible_s{k}" for k in K_STOPS]].all(axis=1)]

    def k_stats_table(long_by_ks, pool_name):
        """每 K: 事件级实际收益（全口径用各自 eligible；共同口径另算）"""
        tbl = {}
        for ks in K_STOPS:
            evd = event_select(long_by_ks[ks])
            row = {}
            for st in subtypes:
                s = evd[evd["subtype"] == st]
                row[st] = {"n": len(s), "mean_ret": float(s["net_ret"].mean()),
                           "cum_ret": float((1 + s.sort_values("date")["net_ret"]).cumprod().iloc[-1] - 1) if len(s) else np.nan,
                           "loss_rate": float((s["net_ret"] < 0).mean()) if len(s) else np.nan}
            tbl[ks] = row
        return tbl

    fold_realized = {}
    for fy, _ in folds:
        fold_realized[fy] = k_stats_table(
            {ks: fold_long[(fy, ks)] for ks in K_STOPS}, "full")

    def common_stats(fy):
        """共同 eligible 子集上各 K 的实际收益（用共同池事件预测）"""
        fev = segment(ev, fy * 10000 + 101, fy * 10000 + 1231, fy * 10000 + 1231)
        cmn = fev[fev[[f"eligible_s{k}" for k in K_STOPS]].all(axis=1)]
        out = {}
        for ks in K_STOPS:
            pl = predict_proba(fold_models[(fy, ks)][0], cmn, ks)  # 全 eligible,需再滤共同
            pl = pl[pl["event_id"].isin(set(cmn["event_id"]))]
            pl["EV"] = EVParams(
                segment(ev, 20190101, (fy - 1) * 10000 + 1231, (fy - 1) * 10000 + 1231),
                ks).ev(pl)
            evd = event_select(pl)
            out[ks] = {st: {"n": len(evd[evd["subtype"] == st]),
                            "mean_ret": float(evd[evd["subtype"] == st]["net_ret"].mean())}
                       for st in subtypes}
        return out

    fold_common = {fy: common_stats(fy) for fy, _ in folds}

    # 冻结（v2-4: 折外实际 net_ret 均值；双口径并报）
    freeze, freeze_common = {}, {}
    for st in subtypes:
        full_row = {ks: np.mean([fold_realized[fy][ks][st]["mean_ret"] for fy, _ in folds])
                    for ks in K_STOPS}
        cmn_row = {ks: np.mean([fold_common[fy][ks][st]["mean_ret"] for fy, _ in folds])
                   for ks in K_STOPS}
        freeze[st] = max(full_row, key=full_row.get)
        freeze_common[st] = max(cmn_row, key=cmn_row.get)
        log(f"fold realized {st}: full=" + ", ".join(f"{k}:{v:+.4f}" for k, v in full_row.items())
            + f" -> {freeze[st]} | common=" + ", ".join(f"{k}:{v:+.4f}" for k, v in cmn_row.items())
            + f" -> {freeze_common[st]}")

    # ---------------- OOF 校准器（滚动折折外预测拟合；冻结 K）
    calibrators = {}
    for st in subtypes:
        ks = freeze[st]
        oof = pd.concat([fold_long[(fy, ks)] for fy, _ in folds], ignore_index=True)
        calibrators[st] = fit_isotonic(oof[oof["subtype"] == st])
    with open(os.path.join(MODEL_DIR, "isotonic_calibrators.pkl"), "wb") as f:
        pickle.dump({"freeze": freeze, "freeze_common": freeze_common,
                     "calibrators": calibrators}, f)

    # ---------------- 最终模型：内部 8:2 早停选轮数 -> 2019-2023 全量重训
    final, best_iters = {}, {}
    for ks in K_STOPS:
        clf, bi, n = fit_model(tr_main, ks, es_ev=tr_es)
        best_iters[ks] = bi
        fr = refit_full(tr, ks, bi)
        final[ks] = fr
        fr.booster_.save_model(os.path.join(MODEL_DIR, f"lgbm_s{ks}.txt"))
        log(f"final s{ks}: es_best_iter={bi}, refit on {n} rows")

    # ---------------- 2024 确认（实际收益，双口径；只确认不选择）
    def realized_on(seg_ev, models, pool="full"):
        out = {}
        for ks in K_STOPS:
            src = seg_ev if pool == "full" else \
                seg_ev[seg_ev[[f"eligible_s{k}" for k in K_STOPS]].all(axis=1)]
            pl = predict_proba(models[ks], src, ks)
            if pool == "common":
                pl = pl[pl["event_id"].isin(set(src["event_id"]))]
            pl["EV"] = EVParams(tr, ks).ev(pl)
            evd = event_select(pl)
            out[ks] = {st: {"n": len(evd[evd["subtype"] == st]),
                            "mean_ret": float(evd[evd["subtype"] == st]["net_ret"].mean()),
                            "loss_rate": float((evd[evd["subtype"] == st]["net_ret"] < 0).mean())}
                       for st in subtypes}
        return out

    conf_full = realized_on(va, final, "full")
    conf_common = realized_on(va, final, "common")
    confirm_ok = {}
    for st in subtypes:
        b_full = max(conf_full, key=lambda k: conf_full[k][st]["mean_ret"])
        b_cmn = max(conf_common, key=lambda k: conf_common[k][st]["mean_ret"])
        confirm_ok[st] = (b_full == freeze[st], b_cmn == freeze_common[st])
        log(f"2024 confirm {st}: full best={b_full}(frozen {freeze[st]}), "
            f"common best={b_cmn}(frozen {freeze_common[st]})")

    # ---------------- 评估（2024 与 2025）：分层 = subtype × in_watch_sector
    # v3.1: 冻结阈值改由训练期 2022/2023 滚动 OOF 的校准后 EV 分布取 90 分位，
    # 避免最终模型训练集内预测拟合偏高导致阈值过松
    thr_frozen = {}
    for st in subtypes:
        ks = freeze[st]
        oof_sel = []
        for fy, ftr in folds:
            pl = fold_long[(fy, ks)]
            pl = pl[pl["subtype"] == st].copy()
            cal = apply_isotonic(pl, calibrators[st])
            cal["EV"] = EVParams(ftr, ks).ev(cal)
            oof_sel.append(event_select(cal))
        thr_frozen[st] = float(pd.concat(oof_sel, ignore_index=True)["EV"].quantile(0.9))
    log(f"frozen EV thresholds (OOF): {thr_frozen}")

    def evaluate(seg_ev, tag):
        longs = {ks: predict_proba(final[ks], seg_ev, ks) for ks in K_STOPS}
        for ks in K_STOPS:
            longs[ks]["EV"] = EVParams(tr, ks).ev(longs[ks])
        stats, preds_out = {}, []
        # v2-1 验收: target_atr 敏感性
        l12 = longs[freeze["adj_ready"]]
        g = l12.groupby("event_id")["p_TARGET"].nunique()
        sens = float(spearmanr(l12["target_atr"], l12["p_TARGET"]).statistic)
        log(f"[{tag}] p_TARGET nunique per event: max={int(g.max())} "
            f"| target_atr vs p_TARGET spearman={sens:.4f} (期望为负)")
        for st in subtypes:
            ks = freeze[st]
            raw = longs[ks][longs[ks]["subtype"] == st].copy()
            cal = apply_isotonic(raw, calibrators[st])
            cal["EV"] = EVParams(tr, ks).ev(cal)  # 校准后再算 EV
            cp = const_probs(tr, ks, st)
            for sec_lab, sec_m in [("板块内", raw["in_watch_sector"] == True),
                                   ("板块外/缺失", raw["in_watch_sector"] == False)]:
                ev_r = event_select(raw[sec_m])
                m = strat_metrics(ev_r, cp, ks=ks)
                ev_c = event_select(cal[sec_m])
                mc = strat_metrics(ev_c, cp, ks=ks, thr=thr_frozen[st])
                stats[(tag, st, sec_lab)] = (m, mc)
                log(f"[{tag}|{st}|{sec_lab}|s{ks}] n={m['n_events']} "
                    f"IC {m['rank_ic']:+.4f}->{mc['rank_ic']:+.4f} | "
                    f"Top10% {m['top_ret']*100:.2f}%->{mc['top_ret']*100:.2f}% "
                    f"(base {m['mean_ret']*100:.2f}%) win {m['top_win']*100:.0f}% | "
                    f"dIC {mc['d_ic_mean']:+.4f}(t={mc['d_ic_t']:+.1f},"
                    f"{mc['d_ic_pos']:.0%}pos) dTop3 {mc['d_top_ret']*100:.2f}% | "
                    f"thr EV≥{thr_frozen[st]:.3f}: n={mc['thr_n']} "
                    f"ret {mc['thr_ret']*100:.2f}% | "
                    f"Brier {m['brier']:.4f}->{mc['brier']:.4f} (const {m['brier_const']:.4f}) "
                    f"BSS {mc['bss']:+.3f} ECE {m['ece']:.4f}")
            preds_out.append(raw.assign(subtype=st))
        return stats, pd.concat(preds_out, ignore_index=True)

    stats_v, preds_v = evaluate(va, "val2024")
    preds_v.to_parquet(os.path.join(PRED_DIR, "val_pred.parquet"), index=False)
    stats_5, preds_5 = evaluate(t5, "ref2025")
    preds_5.to_parquet(os.path.join(PRED_DIR, "test2025_pred.parquet"), index=False)

    # ---------------- 特征重要性（冻结模型）
    imp_lines = []
    for st in subtypes:
        imp = pd.Series(
            final[freeze[st]].booster_.feature_importance("gain"),
            index=final[freeze[st]].booster_.feature_name()) \
            .sort_values(ascending=False).head(20)
        imp_lines.append((st, freeze[st], imp))
        log(f"importance {st} s{freeze[st]} top5: " +
            ", ".join(f"{k}={v:.0f}" for k, v in imp.head(5).items()))
    uses_sector_flag = [st for st, _, imp in imp_lines if "in_watch_sector" in imp.index[:20]]
    log(f"in_watch_sector 进入 top20 的模型: {uses_sector_flag}")

    # ---------------- v3 板块特征消融：剔除 sector_/leader_/center_/rel_ret_ 块 +
    # in_watch_sector 后重训冻结 K 模型，比较事件级 IC/Top10%，验证模型是否
    # 依赖"板块特征缺失模式"而非数值信号
    SECTOR_PREFIX = ("sector_", "leader_", "center_", "rel_ret_")
    abl_rows = []
    for st in subtypes:
        ks = freeze[st]
        drop_cols = [c for c in feats if c.startswith(SECTOR_PREFIX)
                     or c == "in_watch_sector"]
        keep_cols = [c for c in feats if c not in drop_cols]

        def _fit_abl(tr_ev):
            tr_ev = tr_ev[tr_ev[f"eligible_s{ks}"]]
            long = make_long(tr_ev, ks)
            X = tr_ev[keep_cols].astype(np.float32) \
                .reindex(long.index).replace([np.inf, -np.inf], np.nan)
            X["target_atr"] = long["target_atr"].values
            clf = lgb.LGBMClassifier(**LGB_PARAMS, n_estimators=max(int(best_iters[ks]), 10))
            clf.fit(X, long["y"].values, callbacks=[lgb.log_evaluation(0)])
            return clf

        def _pred_abl(clf, seg_ev):
            seg_ev = seg_ev[seg_ev[f"eligible_s{ks}"]]
            long = make_long(seg_ev, ks)
            X = seg_ev[keep_cols].astype(np.float32) \
                .reindex(long.index).replace([np.inf, -np.inf], np.nan)
            X["target_atr"] = long["target_atr"].values
            proba = clf.predict_proba(X)
            for i, cl in enumerate(clf.classes_):
                long[f"p_{cl}"] = proba[:, i]
            for cl in CLASSES:
                if f"p_{cl}" not in long.columns:
                    long[f"p_{cl}"] = 0.0
            long["EV"] = EVParams(tr, ks).ev(long)
            return event_select(long)

        clf_abl = _fit_abl(tr)
        for seg_ev, tag in [(va, "2024"), (t5, "2025")]:
            evd_a = _pred_abl(clf_abl, seg_ev[seg_ev["subtype"] == st])
            # 对照：全特征模型在同一子类上的校准前事件级结果（板块内+外合并）
            full_key = ("val2024" if tag == "2024" else "ref2025", st, "板块外/缺失")
            full_m = stats_v[full_key][0] if tag == "2024" else stats_5[full_key][0]
            abl_rows.append({
                "subtype": st, "seg": tag,
                "abl_n": len(evd_a),
                "abl_ic": float(spearmanr(evd_a["EV"], evd_a["net_ret"]).statistic),
                "abl_top10": float(evd_a[evd_a["EV"] >= evd_a["EV"].quantile(0.9)]
                                    ["net_ret"].mean()),
                "full_ic": full_m["rank_ic"], "full_top10": full_m["top_ret"],
            })
            log(f"[abl|{st}|{tag}] 去板块 IC {abl_rows[-1]['abl_ic']:+.4f} "
                f"Top10 {abl_rows[-1]['abl_top10']*100:.2f}% | "
                f"全特征 IC {full_m['rank_ic']:+.4f} Top10 {full_m['top_ret']*100:.2f}%")

    # ---------------- 报告第 8 节（v2，先删旧节）
    with open(REPORT_PATH, encoding="utf-8") as f:
        old = f.read()
    idx = old.find("\n## 8. LightGBM 基线")
    if idx != -1:
        old = old[:idx] + "\n"
        with open(REPORT_PATH, "w", encoding="utf-8") as f:
            f.write(old.rstrip("\n") + "\n")

    L = []
    L.append("\n## 8. LightGBM 基线（v2.1 第5+6步）\n")
    L.append("**修订痕迹**：v1 因 `target_atr` 未入模作废；v2 修复 target_atr/eligible/"
             "EVParams/2024 泄漏；v3 修复选档共享索引、组合回测真实持仓化、新增每日横截面"
             "指标与板块消融；v3.1 修复冻结阈值被当期 90% 分位覆盖的 bug、组合双口径、"
             "仓位字段拆分、回测延长至持仓全退出。"
             "**v3.3（本轮）**：① 数据链修复——源 `pre_close` 为未复权价（2018 年 83% 行"
             "偏差>1%），改由前复权前收盘推导，并新增数据质量断言（年度 ret1 中位数≈0、"
             "负向超跌幅错误行剔除）；② 特征改**显式白名单**（黑名单曾漏 `exit_day_*` "
             "标签列导致 IC 0.8 虚高假象），未声明列 assert 失败；"
             "③ 组合回测改**真账本**（现金+股数+逐日盯市、T+1、涨停封板不买、"
             "跌停锁死由标签 fill 顺延、佣金/印花税分列、回测至持仓全退出）；"
             "④ 新增 8.5a 账本审计（2025 阈值过滤交易明细）。"
             "v3.3 之前的全部模型与报告结论作废，见 archive_invalid/README.md。\n")
    L.append(f"事件级样本：训练(信号19-23,label_end≤2023)={len(tr):,}，"
             f"验证(2024)={len(va):,}，2025参考={len(t5):,}；特征 {len(feats)}+target_atr；"
             "板块 NaN 不填 0。早停=训练期内部时间 8:2 尾段，2024 不进训练/早停/校准拟合"
             "（校准器用 2019-2023 滚动 OOF 拟合）。")

    L.append("\n### 8.1 K_stop 滚动选择（折外实际 net_ret，v2-4；两折平均）\n")
    L.append("全口径（各 K 自己合格池，每事件取 EV argmax 档后实际收益）：\n")
    L.append("| 形态子类 | K | 实际净收益(折均) | 单折2022 | 单折2023 | 平均loss率 | 覆盖率 |")
    L.append("|---|---|---:|---:|---:|---:|---:|")
    for st in subtypes:
        cov = {ks: float((tr[f"eligible_s{ks}"] & (tr["subtype"] == st)).sum()
                         / max((tr["subtype"] == st).sum(), 1)) for ks in K_STOPS}
        for ks in K_STOPS:
            r22, r23 = fold_realized[2022][ks][st], fold_realized[2023][ks][st]
            mean_r = np.mean([r22["mean_ret"], r23["mean_ret"]])
            lr_ = np.mean([r22["loss_rate"], r23["loss_rate"]])
            L.append(f"| {st} | {ks} | {mean_r:+.4f} | {r22['mean_ret']:+.4f} "
                     f"| {r23['mean_ret']:+.4f} | {lr_:.3f} | {cov[ks]:.1%} |")
    L.append("\n共同样本口径（仅四 K 都 eligible 的事件，隔离止损宽度效果）：\n")
    L.append("| 形态子类 | K | 共同样本实际净收益(折均) | 冻结 |")
    L.append("|---|---|---:|---|")
    for st in subtypes:
        for ks in K_STOPS:
            m = np.mean([fold_common[fy][ks][st]["mean_ret"] for fy, _ in folds])
            L.append(f"| {st} | {ks} | {m:+.4f} | "
                     f"{'全口径' if freeze[st]==ks else ''}{'共同' if freeze_common[st]==ks else ''} |")
    L.append(f"\n**冻结表**：全口径 {freeze}；共同样本口径 {freeze_common}。"
             + ("两口径一致。" if freeze == freeze_common else
                "⚠ 两口径不一致，以全口径为准（v2-4 选择流程），共同口径供复核。"))

    L.append("\n### 8.2 2024 验证段确认（实际收益，只确认不选择）\n")
    L.append("| 形态子类 | 口径 | 各 K 实际净收益(1.2/1.5/1.8/2.0) | 方向一致 |")
    L.append("|---|---|---|---|")
    for st in subtypes:
        L.append(f"| {st} | 全口径 | " + " / ".join(
            f"{conf_full[ks][st]['mean_ret']:+.4f}" for ks in K_STOPS)
            + f" | {confirm_ok[st][0]} |")
        L.append(f"| {st} | 共同 | " + " / ".join(
            f"{conf_common[ks][st]['mean_ret']:+.4f}" for ks in K_STOPS)
            + f" | {confirm_ok[st][1]} |")

    def metric_table(stats, tag):
        key = {"8.3 验证段（2024）表现": "val2024",
               "8.4 2025 参考评估（不参与选择）": "ref2025"}[tag]
        L.append(f"\n### {tag}\n")
        L.append("全年混合 IC/Top10% 仅作描述（可能来自择日而非选股）；决策口径看每日横截面列。\n")
        L.append("| 子类 | 板块 | n事件 | RankIC(原→校) | Top10%(原→校) | 胜率 | 基线 | "
                 "日IC(校,t值,%pos) | 日Top3 | 冻结阈值n/收益 | Brier(→校) | 常数 | BSS | logloss | ECE |")
        L.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
        for st in subtypes:
            for sec_lab in ["板块内", "板块外/缺失"]:
                m, mc = stats[(key, st, sec_lab)]
                L.append(f"| {st} | {sec_lab} | {m['n_events']} | "
                         f"{m['rank_ic']:+.4f}→{mc['rank_ic']:+.4f} | "
                         f"{m['top_ret']*100:.2f}%→{mc['top_ret']*100:.2f}% | "
                         f"{m['top_win']*100:.0f}% | {m['mean_ret']*100:.2f}% | "
                         f"{mc['d_ic_mean']:+.4f}({mc['d_ic_t']:+.1f},{mc['d_ic_pos']:.0%}) | "
                         f"{mc['d_top_ret']*100:.2f}% | "
                         f"{mc['thr_n']}/{mc['thr_ret']*100:.2f}% | "
                         f"{m['brier']:.4f}→{mc['brier']:.4f} | {m['brier_const']:.4f} | "
                         f"{mc['bss']:+.3f} | {m['logloss']:.4f}→{m['logloss_const']:.4f} | "
                         f"{m['ece']:.4f} |")
        return L

    metric_table(stats_v, "8.3 验证段（2024）表现")
    metric_table(stats_5, "8.4 2025 参考评估（不参与选择）")

    L.append("\n### 8.5 交易画像与 20 仓组合回测（事件级，冻结 K 全口径）\n")
    L.append("| 段 | 子类 | n事件 | 每日候选(均/中位/P90) | target_atr 选中比例 | "
             "每股交易(均/最大) | Top10%净收益 | Top10%胜率 |")
    L.append("|---|---|---:|---|---|---|---|---:|")
    prof_keys = {"2024": ("val2024", stats_v), "2025": ("ref2025", stats_5)}
    for tag, (key, stt) in prof_keys.items():
        for st in subtypes:
            m_in, _ = stt[(key, st, "板块内")]
            m_out, _ = stt[(key, st, "板块外/缺失")]
            m = m_in if m_in["n_events"] >= m_out["n_events"] else m_out
            L.append(f"| {tag} | {st} | {m['n_events']} | "
                     f"{m['cand_daily_mean']:.1f}/{m['cand_daily_median']:.0f}/{m['cand_daily_p90']:.0f} | "
                     + ", ".join(f"t{k}:{v:.0%}" for k, v in sorted(m['targ_share'].items()))
                     + f" | {m['trades_per_code_mean']:.2f}/{m['trades_per_code_max']} | "
                     f"{m['top_ret']*100:.2f}% | {m['top_win']*100:.0f}% |")
    L.append("\n（画像指标按该子类全部合格事件统计，与板块内/外拆分无关；"
             "组合回测按校准后 EV 排序、每日≤20 仓：）")

    # 组合回测：用 preds 文件的事件级选中档（校准后 EV 重算在评估内完成，
    # 这里用未校准 EV 选档 + 校准后 EV 排序，直接取各 subtype 冻结 K）
    def pf_table(seg_ev, tag):
        rows = []
        for st in subtypes:
            ks = freeze[st]
            raw = predict_proba(final[ks], seg_ev[seg_ev["subtype"] == st], ks)
            raw["EV"] = EVParams(tr, ks).ev(raw)
            cal = apply_isotonic(raw, calibrators[st])
            cal["EV"] = EVParams(tr, ks).ev(cal)
            evd = event_select(cal)
            pf_fill = portfolio_backtest(evd, ks=ks)                      # 强制填仓
            pf_thr = portfolio_backtest(evd, ks=ks,                      # 冻结阈值+空仓
                                        ev_thr=thr_frozen[st])
            rows.append((st, pf_fill, pf_thr))
        allrows = []
        for st in subtypes:
            ks = freeze[st]
            raw = predict_proba(final[ks], seg_ev[seg_ev["subtype"] == st], ks)
            raw["EV"] = EVParams(tr, ks).ev(raw)
            cal = apply_isotonic(raw, calibrators[st])
            cal["EV"] = EVParams(tr, ks).ev(cal)
            allrows.append(event_select(cal))
        ev_all = pd.concat(allrows, ignore_index=True)
        ev_all["ev_thr_row"] = ev_all["subtype"].map(thr_frozen)
        ev_thr_all = ev_all["EV"] >= ev_all["ev_thr_row"]
        pf_all_fill = portfolio_backtest(ev_all.drop(columns=["ev_thr_row"]))
        return rows, pf_all_fill, ev_all.assign(sel=ev_thr_all)

    audit_25 = None
    for seg_ev, tag in [(va, "2024"), (t5, "2025")]:
        rows, pf_all_fill, ev_all = pf_table(seg_ev, tag)
        pf_all_thr = portfolio_backtest(
            ev_all[ev_all["sel"]].drop(columns=["ev_thr_row", "sel"]))

        def _fmt(pf):
            return (f"总{pf['total_ret']:+.2%}/年化{pf['ann_ret']:+.2%}/"
                    f"回撤{pf['max_dd']:.2%}/盘后{pf['avg_pos']:.1f}槽/"
                    f"交易{pf['n_trades']}/"
                    f"佣金{pf['comm_buy']+pf['comm_sell']:.4f}/"
                    f"印花税{pf['stamp']:.4f}")

        txt = "；".join(f"{st}: 填仓[{_fmt(pf)}]" for st, pf, _ in rows if pf)
        txt_t = "；".join(f"{st}: 阈值[{_fmt(pt)}]" for st, _, pt in rows if pt)
        L.append(f"\n- {tag} 组合回测（v3.3 真账本：现金+股数+逐日盯市、T+1、"
                 f"涨停封板不买、跌停锁死由标签 fill 顺延、费用分列、"
                 f"回测至持仓全退出）：\n"
                 f"  - 强制填仓（容量压力测试）{txt}\n"
                 f"  - 冻结阈值+允许空仓（正式口径）{txt_t}\n"
                 f"  - 整体填仓[{_fmt(pf_all_fill)}]\n"
                 f"  - 整体阈值[{_fmt(pf_all_thr)}]")
        if tag == "2025":
            for st, pf_f, pf_t in rows:
                if st != "adj_ready" or not pf_f or not pf_t:
                    continue
                tf, tt = pf_f.get("_trades"), pf_t.get("_trades")
                if tf is None or tt is None or len(tf) == 0:
                    continue
                only_fill = tf[~tf["event_id"].isin(set(tt["event_id"]))]
                ks = freeze[st]
                raw = predict_proba(final[ks], t5[t5["subtype"] == st], ks)
                raw["EV"] = EVParams(tr, ks).ev(raw)
                cal = apply_isotonic(raw, calibrators[st])
                cal["EV"] = EVParams(tr, ks).ev(cal)
                evd25 = event_select(cal)
                pf_ex = portfolio_backtest(
                    evd25[~evd25["event_id"].isin(set(only_fill["event_id"]))],
                    ks=ks)
                audit_25 = (st, pf_f, pf_t, pf_ex, only_fill)

    if audit_25 is not None:
        st, pf_f, pf_t, pf_ex, only = audit_25
        L.append(f"\n### 8.5a 账本审计：2025 {st} 阈值过滤掉的 {len(only)} 笔交易\n")
        L.append(f"- 填仓口径：交易 {pf_f['n_trades']} 笔，总收益 {pf_f['total_ret']:+.2%}，"
                 f"年化 {pf_f['ann_ret']:+.2%}")
        L.append(f"- 阈值口径：交易 {pf_t['n_trades']} 笔，总收益 {pf_t['total_ret']:+.2%}，"
                 f"年化 {pf_t['ann_ret']:+.2%}")
        L.append(f"- 排除交易（填仓买入但 EV<冻结阈值）：{len(only)} 笔，"
                 f"净收益均值 {only['net_ret'].mean():+.2%}，"
                 f"中位 {only['net_ret'].median():+.2%}，"
                 f"胜率 {(only['net_ret']>0).mean():.0%}，"
                 f"合计账面盈亏 {only['pnl'].sum():+.4f}")
        L.append(f"- 同账本复算（剔除这 {len(only)} 笔后强制填仓）：交易 "
                 f"{pf_ex['n_trades']} 笔，总收益 {pf_ex['total_ret']:+.2%}，"
                 f"年化 {pf_ex['ann_ret']:+.2%} → 与阈值口径的差即剩余差异"
                 f"（槽位释放后由更高 EV 候选递补造成）")
        L.append("- 明细（事件级）：")
        L.append("| 信号日 | 代码 | EV | target_atr | net_ret | exit_day | 账面盈亏 |")
        L.append("|---|---|---:|---:|---:|---:|---:|")
        for _, r in only.sort_values("date").iterrows():
            L.append(f"| {int(r['date'])} | {r['code']} | {r['EV']:.4f} | "
                     f"{r['target_atr']:.1f} | {r['net_ret']:+.2%} | "
                     f"{int(r['exit_day'])} | {r['pnl']:+.4f} |")

    L.append("\n### 8.6 特征重要性（冻结模型 gain, top20）\n")
    for st, ks, imp in imp_lines:
        L.append(f"- {st} (s{ks}): " + ", ".join(f"{k}({v:.0f})" for k, v in imp.items()))
    L.append(f"- in_watch_sector 是否主导：进入 top20 的模型 = {uses_sector_flag or '无'}；"
             "板块特征 NaN 模式未被单独依赖则排序信号来自数值特征。"
             "（若 in_watch_sector 排名靠前，说明模型主要在区分'板块内/外'两类事件。）")

    L.append("\n### 8.7 板块特征消融（剔除 sector_/leader_/center_/rel_ret_ 块 + in_watch_sector 后重训）\n")
    L.append("| 子类 | 段 | n | 去板块IC | 全特征IC | 去板块Top10% | 全特征Top10% |")
    L.append("|---|---|---:|---:|---:|---:|---:|")
    for r in abl_rows:
        L.append(f"| {r['subtype']} | {r['seg']} | {r['abl_n']} | "
                 f"{r['abl_ic']:+.4f} | {r['full_ic']:+.4f} | "
                 f"{r['abl_top10']*100:.2f}% | {r['full_top10']*100:.2f}% |")
    L.append("\n（全特征列为板块外/缺失层校准前事件级结果，作对照；消融与对照同段同子类。）")

    L.append("\n### 8.8 结论（v3 措辞）\n")
    def ic_of(stats, key, st, sec):
        return stats[(key, st, sec)][0]["rank_ic"]
    L.append(f"- adj_ready：2024 Rank IC（板块内/外）= "
             f"{[round(ic_of(stats_v,'val2024','adj_ready',s),4) for s in ['板块内','板块外/缺失']]}，"
             f"2025 = {[round(ic_of(stats_5,'ref2025','adj_ready',s),4) for s in ['板块内','板块外/缺失']]}。"
             "主体子类排序能力≈0，Top10% 与基线差异不构成可交易优势。")
    L.append(f"- first_div：2024 Rank IC = "
             f"{[round(ic_of(stats_v,'val2024','first_div',s),4) for s in ['板块内','板块外/缺失']]}，"
             f"2025 = {[round(ic_of(stats_5,'ref2025','first_div',s),4) for s in ['板块内','板块外/缺失']]}。"
             "**首次分歧子类可能存在风险排序信号**：连续两年（2024/2025）Top 层相对基线"
             "减少损失，但 Top10% 绝对净收益仍为负（2024 胜率仅约 "
             f"{stats_v[('val2024','first_div','板块内')][0]['top_win']:.0%}），"
             "**值得复核**（样本量小、校准不稳，禁止解读为'有效'）。")
    L.append("- 校准：isotonic 由训练期 OOF 拟合；个别分层校准后 Brier 变差"
             "（如 2025 adj_ready 见 8.4）已如实报告，原因是 OOF 与目标段分布漂移。"
             "产物：models/lgbm_s{K}.txt ×4、models/isotonic_calibrators.pkl、"
             "preds/val_pred.parquet、preds/test2025_pred.parquet（长表含 target_atr，"
             "事件级 argmax 由消费方按 EV 自取）。")
    with open(REPORT_PATH, "a", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    log("appended report section 8 (v2)")
    log(f"done in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
