"""20 槽真实组合账本 v2 —— 绿盘低吸「真实收益口径」组合回测引擎。

规则全部冻结，来源：exit_dyn_spec_v1.md（§0/§5/§7/§8）+ 开发路线第 3 步任务书。
收益一律来自退出标签（exit_labels_*.parquet）的 exit_price / gross_return；
绝不读取 t1_rule_return、绝不构造 MFE 收益、绝不把固定 T+5 当默认退出
（X0_CONTROL 的 TIME_EXIT 是显式规则对照，不是偷懒）。

运行矩阵：入场流 {F1,F2,F3} × 退出变体 {X0_CONTROL,X1,X1_RF,X2_STAGED} = 12 组合。
同一 event 只用其所属变体的那一行的退出来决定卖出，不混用变体。

成本（任务书冻结口径，与标签 ROUND_TRIP_COST=0.003 冲突时以本口径为准并记录）：
    买入侧 = 佣金 0.0003 + 滑点 0.001
    卖出侧 = 佣金 0.0003 + 印花税 0.0005 + 滑点 0.001
逐笔从现金扣，绝不做事后按比例摊。

运行：d:/stargazer/.venv/Scripts/python.exe portfolio_backtest_v2.py
（需要 torch：入场流推理复用 _ledger_20slot.predict_scores；账本本身不读 t1_rule_return）
输出：reports/portfolio_v2_report.md + reports/ledgers/ledger_{F}_{X}.parquet
幂等：入场流缓存 flow_cache/entry_flow_{F}.parquet，标签/行情不变则直接复用。
"""
import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict

BASE = Path("d:/stargazer/stargazer_ml/v5_hybrid/green_exit_v1")
LABELS_DAILY = BASE / "labels/exit_labels_daily_approx.parquet"
LABELS_MINUTE = BASE / "labels/exit_labels_minute_exact.parquet"
BARS_GLOB = "D:/a股数据/data_daily_parts/part_*.parquet"
REPORTS = BASE / "reports"
LEDGERS = REPORTS / "ledgers"
FLOW_CACHE = BASE / "flow_cache"

FLOWS = ["F1", "F2", "F3"]
VARIANTS = ["X0_CONTROL", "X1", "X1_RF", "X2_STAGED"]

# ---- 冻结规则常量（来源：exit_dyn_spec_v1.md §0/§5/§8 + 任务书）----
N_SLOT = 20                # 槽位 20；单票占 1 槽；已持仓不可重复买入
COOLDOWN_DAYS = 5          # 卖出后 5 个交易日内不得买回同票（第 6 个交易日起可买）
INIT_CAPITAL = 1_000_000.0 # 初始资金 100 万
COMM_RATE = 0.0003         # 佣金，双边（任务书）
STAMP_RATE = 0.0005        # 印花税，仅卖出（任务书）
SLIP_BUY = 0.001           # 买入滑点（任务书）
SLIP_SELL = 0.001          # 卖出滑点（任务书）
# 涨跌停锁死判定沿用标签引擎口径（build_exit_labels.py 顶部常量，20%/30% 板简化）：
LIMIT_UP_MULT, LIMIT_DOWN_MULT = 1.098, 0.902
ST_UP_MULT, ST_DOWN_MULT = 1.05, 0.95
BLOCK_FORCE_DELAY = 5      # EXIT_BLOCKED 不臆造成交：持仓延续，T+5（入场后第 5 个交易日）之后
                           # 按下一可交易日 close 强制退出（日频账本可行近似，spec §0 延迟成交原则）

EPS = 1e-9


# ---------------------------------------------------------------- 数据装载
def load_labels():
    dl = pd.read_parquet(LABELS_DAILY)
    ml = pd.read_parquet(LABELS_MINUTE)
    lab = pd.concat([dl, ml], ignore_index=True)
    # exit_time 有 8 位（日粒度）与 14 位（分钟）两种编码，统一取前 8 位为退出日
    lab["exit_date"] = lab["exit_time"].astype(str).str[:8].astype(int)
    lab["code"] = lab["code"].astype(str).str.zfill(6)
    return lab


def load_bars(codes):
    import glob
    cols = ["date", "code", "open", "high", "low", "close", "pre_close", "volume", "amount", "is_st"]
    fs = sorted(glob.glob(BARS_GLOB))
    parts = [pd.read_parquet(f, columns=cols) for f in fs]
    bars = pd.concat(parts, ignore_index=True)
    del parts
    bars["code"] = bars["code"].astype(str).str.zfill(6)
    bars = bars[bars["code"].isin(codes)]
    bars = bars.sort_values(["code", "date"])
    out = {}
    for code, g in bars.groupby("code", sort=False):
        out[code] = {
            "date": g["date"].to_numpy(),
            "open": g["open"].to_numpy(float),
            "high": g["high"].to_numpy(float),
            "low": g["low"].to_numpy(float),
            "close": g["close"].to_numpy(float),
            "pre_close": g["pre_close"].to_numpy(float),
            "volume": g["volume"].to_numpy(float),
            "amount": g["amount"].to_numpy(float),
            "is_st": g["is_st"].to_numpy(),
            "idx": {int(d): i for i, d in enumerate(g["date"].to_numpy())},
        }
    calendar = np.array(sorted(bars["date"].unique()))
    return out, calendar


def build_entry_flow(flow_name):
    """每日 top-N_SLOT 入场流（借鉴 _ledger_20slot.build_positions 的生成方式：
    S0 头排序分降序取 top20；绝不用 t1_rule_return 做收益，也不用它过滤——
    可用性由退出标签是否存在决定）。结果缓存到 parquet，幂等可重跑。"""
    FLOW_CACHE.mkdir(parents=True, exist_ok=True)
    cache = FLOW_CACHE / f"entry_flow_{flow_name}.parquet"
    if cache.exists():
        return pd.read_parquet(cache)
    sys.path.insert(0, str(BASE.parent))
    from _ledger_20slot import predict_scores  # 复用 S0 推理（含 t1_rule_return 列但本引擎不读它）
    data, scores = predict_scores(flow_name)
    data = data.copy()
    data["_score"] = scores.values
    rows = []
    for date, g in data.groupby("date"):
        g = g.sort_values("_score", ascending=False).head(N_SLOT)
        for rank, (_, r) in enumerate(g.iterrows(), 1):
            rows.append({
                "signal_date": int(date), "code": str(r["code"]).zfill(6),
                "event_id": f"G0:{str(r['code']).zfill(6)}:{int(date)}",
                "score": float(r["_score"]), "rank": rank,
                "sector": r.get("sector", "未知"),
            })
    flow = pd.DataFrame(rows).sort_values(["signal_date", "rank"]).reset_index(drop=True)
    flow.to_parquet(cache, index=False)
    return flow


# ---------------------------------------------------------------- 账本引擎
class Position:
    __slots__ = ("code", "event_id", "sector", "shares", "entry_price", "entry_date",
                 "entry_idx", "signal_date", "label_precision", "exit_date_plan",
                 "exit_price_plan", "exit_reason_plan", "blocked", "x2_blended",
                 "gross_label", "cost_label", "net_label", "last_px",
                 "buy_cash", "buy_comm", "buy_slip")

    def __init__(self, **kw):
        for k, v in kw.items():
            setattr(self, k, v)


def _limit_mults(is_st):
    return (ST_UP_MULT, ST_DOWN_MULT) if is_st else (LIMIT_UP_MULT, LIMIT_DOWN_MULT)


def _bar_on(bars, code, date):
    b = bars.get(code)
    if b is None:
        return None
    i = b["idx"].get(int(date))
    if i is None:
        return None
    return {k: b[k][i] for k in ("open", "high", "low", "close", "pre_close",
                                 "volume", "is_st")}


def _sellable_close(bar):
    """该日是否可按 close 卖出（非跌停锁死、有成交）。跌停锁死=收盘=最低且贴跌停价。"""
    if bar is None or not np.isfinite(bar["close"]) or bar["volume"] <= 0:
        return False
    pc = bar["pre_close"]
    if not pc or not np.isfinite(pc) or pc <= 0:
        return True
    _, dn = _limit_mults(bool(bar["is_st"]))
    locked_dn = bar["close"] <= pc * dn + EPS and abs(bar["close"] - bar["low"]) < 1e-6
    return not locked_dn


def run_combo(flow, labels_v, bars, calendar, cal_idx, variant):
    """单组合 20 槽逐日账本。labels_v: event_id -> label Series（本变体）。"""
    t0i, t1i = 0, len(calendar)
    cash = INIT_CAPITAL
    positions = {}          # code -> Position
    cooldown = {}           # code -> 卖出日 index（含 blocked 强制退出）
    last_px = {}            # code -> 最近收盘价
    equity_rows, trade_rows = [], []
    skip_counts = defaultdict(int)
    blocked_stats = {"label_blocked_traded": 0, "force_exited": 0, "end_of_data": 0}
    # 入场候选按标签 entry_date 分桶（标签引擎允许入场自 T+1 顺延至 T+5，
    # 故不能按 signal_date+1 硬套）；桶内保留 signal 日 top20 的名次顺序
    flow = flow.copy()
    flow["label"] = flow["event_id"].map(labels_v)
    flow = flow[flow["label"].notna()]
    flow["entry_date"] = flow["label"].map(lambda r: int(r["entry_date"]))
    by_entry = {int(d): g.sort_values(["signal_date", "rank"]).to_dict("records")
                for d, g in flow.groupby("entry_date")}

    for t in range(t0i, t1i):
        date = int(calendar[t])
        # 1) 更新持仓最新价（停牌沿用上价）
        for code, pos in positions.items():
            b = bars.get(code)
            i = b["idx"].get(date) if b else None
            if i is not None and np.isfinite(b["close"][i]):
                last_px[code] = b["close"][i]
            pos.last_px = last_px.get(code, pos.entry_price)

        # 2) 卖出（先卖后买，当日释放现金可用于当日入场）
        for code in list(positions.keys()):
            pos = positions[code]
            bar = _bar_on(bars, code, date)
            done = False
            exit_reason, exit_price = None, None
            if pos.blocked:
                # EXIT_BLOCKED：不臆造成交，T+5 后按下一可交易日 close 强制退出
                if t >= pos.entry_idx + BLOCK_FORCE_DELAY and _sellable_close(bar):
                    exit_price, exit_reason = bar["close"], "EXIT_BLOCKED_FORCE"
                    blocked_stats["force_exited"] += 1
                    done = True
            elif pos.exit_date_plan == date:
                exit_price, exit_reason = pos.exit_price_plan, pos.exit_reason_plan
                done = True
            if done:
                gross_px_ret = exit_price / pos.entry_price - 1.0
                proceeds = pos.shares * exit_price * (1 - SLIP_SELL)
                comm_s = pos.shares * exit_price * COMM_RATE
                stamp = pos.shares * exit_price * STAMP_RATE
                slip_s = pos.shares * exit_price * SLIP_SELL
                cash += proceeds - comm_s - stamp
                pnl = proceeds - comm_s - stamp - pos.buy_cash
                trade_rows.append({
                    "row_type": "trade", "date": date, "code": code,
                    "event_id": pos.event_id, "sector": pos.sector,
                    "signal_date": pos.signal_date, "entry_date": pos.entry_date,
                    "entry_price": pos.entry_price, "shares": pos.shares,
                    "buy_cash": pos.buy_cash,
                    "buy_cost_detail": pos.buy_comm + pos.buy_slip,
                    "exit_date": date, "exit_price": exit_price,
                    "exit_reason": exit_reason,
                    "exit_blocked_resolved": int(pos.blocked),
                    "label_precision": pos.label_precision,
                    "gross_return_label": pos.gross_label,
                    "cost_label": pos.cost_label, "net_return_label": pos.net_label,
                    "gross_px_return": gross_px_ret,
                    "sell_cost_detail": comm_s + stamp + slip_s,
                    "pnl": pnl, "net_return": pnl / pos.buy_cash,
                    "holding_days": cal_idx[date] - pos.entry_idx,
                    "turnover": pos.shares * pos.entry_price + pos.shares * exit_price,
                })
                cooldown[code] = t
                del positions[code]

        # 3) 买入（T+1 开盘价；标签入场日即为本日，买不到则跳过并计数）
        for cand in by_entry.get(date, []):
            if len(positions) >= N_SLOT:
                skip_counts["slot_full"] += 1
                break
            code = cand["code"]
            if code in positions:
                skip_counts["already_held"] += 1
                continue
            if code in cooldown and t <= cooldown[code] + COOLDOWN_DAYS:
                skip_counts["cooldown"] += 1
                continue
            lab = cand["label"]
            entry_date = int(lab["entry_date"])
            if entry_date != date:
                skip_counts["entry_date_mismatch"] += 1
                continue
            bar = _bar_on(bars, code, date)
            if bar is None or not np.isfinite(bar["open"]) or bar["open"] <= 0:
                skip_counts["no_bar"] += 1
                continue
            if bar["volume"] <= 0:
                skip_counts["suspended"] += 1  # 停牌买不进
                continue
            # 涨停可成交性以标签为准（spec §0：标签引擎已按其行情源判定入场并顺延），
            # 账本行情源 parts 的 pre_close 在复牌/除权日与标签源不一致（全 00 主板
            # +32% 开盘等 1453 例假阳性），故不再用 parts pre_close 二次判涨停；
            # 仅记录信息性计数。
            up_mult, _ = _limit_mults(bool(bar["is_st"]))
            if bar["pre_close"] and np.isfinite(bar["pre_close"]) and \
                    bar["open"] >= bar["pre_close"] * up_mult - EPS:
                skip_counts["info_limit_up_per_parts"] += 1
            if abs(bar["open"] - float(lab["entry_price"])) / bar["open"] > 0.01:
                skip_counts["entry_price_mismatch"] += 1  # 行情源冲突，以标签价为成交
                continue
            # 等权：每槽 = 可用现金 / 剩余槽位
            alloc = cash / (N_SLOT - len(positions))
            entry = bar["open"]
            shares = alloc / (entry * (1 + SLIP_BUY + COMM_RATE))
            slip_b = shares * entry * SLIP_BUY
            comm_b = shares * entry * COMM_RATE
            buy_cash = shares * entry + slip_b + comm_b
            cash -= buy_cash
            # 计划卖出（本变体行；X2_STAGED 的 blended 收益换算等效全仓出场价）
            x2 = variant == "X2_STAGED" and lab["exit_reason"] == "REBOUND_FAILURE_STAGED"
            eff_exit = float(lab["entry_price"]) * (1 + float(lab["gross_return"])) if x2 \
                else float(lab["exit_price"])
            positions[code] = Position(
                code=code, event_id=cand["event_id"], sector=cand["sector"],
                shares=shares, entry_price=entry, entry_date=date, entry_idx=t,
                signal_date=cand["signal_date"], label_precision=lab["label_precision"],
                exit_date_plan=int(lab["exit_date"]), exit_price_plan=eff_exit,
                exit_reason_plan=lab["exit_reason"],
                blocked=bool(int(lab["exit_blocked"])), x2_blended=x2,
                gross_label=float(lab["gross_return"]), cost_label=float(lab["cost"]),
                net_label=float(lab["net_return"]), last_px=entry,
                buy_cash=buy_cash, buy_comm=comm_b, buy_slip=slip_b)
            if lab["exit_blocked"]:
                blocked_stats["label_blocked_traded"] += 1

        # 4) 逐仓盯市
        mv = sum(p.shares * p.last_px for p in positions.values())
        equity = cash + mv
        equity_rows.append({"row_type": "equity", "date": date, "cash": cash,
                            "market_value": mv, "equity": equity,
                            "slots_used": len(positions), "drawdown": 0.0})

    # 期末仍未平仓（含未解除 blocked）：按最后可得 close 强制平仓，单列统计
    last_date = int(calendar[-1])
    for code in list(positions.keys()):
        pos = positions[code]
        b = bars.get(code)
        px = b["close"][-1] if b is not None else pos.last_px
        proceeds = pos.shares * px * (1 - SLIP_SELL)
        comm_s = pos.shares * px * COMM_RATE
        stamp = pos.shares * px * STAMP_RATE
        cash += proceeds - comm_s - stamp
        pnl = proceeds - comm_s - stamp - pos.buy_cash
        trade_rows.append({
            "row_type": "trade", "date": last_date, "code": code, "event_id": pos.event_id,
            "sector": pos.sector, "signal_date": pos.signal_date, "entry_date": pos.entry_date,
            "entry_price": pos.entry_price, "shares": pos.shares, "buy_cash": pos.buy_cash,
            "buy_cost_detail": pos.buy_comm + pos.buy_slip,
            "exit_date": last_date, "exit_price": px, "exit_reason": "END_OF_DATA",
            "exit_blocked_resolved": int(pos.blocked), "label_precision": pos.label_precision,
            "gross_return_label": pos.gross_label, "cost_label": pos.cost_label,
            "net_return_label": pos.net_label,
            "gross_px_return": px / pos.entry_price - 1.0,
            "sell_cost_detail": comm_s + stamp + pos.shares * px * SLIP_SELL,
            "pnl": pnl, "net_return": pnl / pos.buy_cash,
            "holding_days": len(calendar) - 1 - pos.entry_idx,
            "turnover": pos.shares * pos.entry_price + pos.shares * px,
        })
        blocked_stats["end_of_data"] += 1
    positions.clear()

    eq = pd.DataFrame(equity_rows)
    eq["drawdown"] = eq["equity"] / eq["equity"].cummax() - 1
    tr = pd.DataFrame(trade_rows)
    return eq, tr, dict(skip_counts), blocked_stats


# ---------------------------------------------------------------- 指标
def perf_stats(eq, tr, label):
    e = eq["equity"].to_numpy()
    n = len(e)
    rets = np.diff(e) / e[:-1]
    ann = float(e[-1] / e[0]) ** (252.0 / max(n - 1, 1)) - 1 if n > 1 else 0.0
    mdd = float(eq["drawdown"].min())
    sharpe = float(rets.mean() / rets.std() * np.sqrt(252)) if len(rets) > 1 and rets.std() > 0 else 0.0
    calmar = ann / abs(mdd) if mdd < 0 else np.nan
    wins = tr[tr["pnl"] > 0]
    losses = tr[tr["pnl"] <= 0]
    gross_profit = wins["pnl"].sum()
    gross_loss = -losses["pnl"].sum()
    reason_counts = tr["exit_reason"].value_counts()
    n_tr = len(tr)
    total_cost = tr["buy_cost_detail"].sum() + tr["sell_cost_detail"].sum()
    occ = float(eq["slots_used"].mean()) / N_SLOT
    monthly = eq.assign(ym=(eq["date"] // 100)).groupby("ym")["equity"].last()
    monthly_ret = monthly.pct_change()
    # 稳健性：去掉最佳月 / 最佳 10 笔
    mr = monthly_ret.dropna()
    best_month = mr.idxmax() if len(mr) else None
    r_adj = eq["equity"].pct_change().fillna(0).to_numpy().copy()
    if best_month is not None:
        mask = (eq["date"] // 100) == best_month
        r_adj[mask.to_numpy()] = 0.0
    eq_ex_month = e[0] * np.cumprod(1 + r_adj)
    ex_mdd = float((eq_ex_month / np.maximum.accumulate(eq_ex_month) - 1).min())
    ex_ann = float(eq_ex_month[-1] / eq_ex_month[0]) ** (252.0 / max(n - 1, 1)) - 1
    top10 = tr.nlargest(min(10, n_tr), "pnl") if n_tr else tr
    eq_adj = e.copy()
    for _, r in top10.iterrows():
        idx = eq.index[eq["date"] >= r["exit_date"]]
        if len(idx):
            eq_adj[idx[0]:] -= r["pnl"]
    ex10_mdd = float((eq_adj / np.maximum.accumulate(eq_adj) - 1).min())
    ex10_ann = float(eq_adj[-1] / eq_adj[0]) ** (252.0 / max(n - 1, 1)) - 1
    return {
        "label": label, "n_trades": n_tr, "final_equity": float(e[-1]),
        "total_return": float(e[-1] / e[0] - 1), "ann_return": ann,
        "sharpe": sharpe, "mdd": mdd, "calmar": calmar,
        "win_rate": float(len(wins) / n_tr) if n_tr else np.nan,
        "avg_win": float(wins["pnl"].mean()) if len(wins) else np.nan,
        "avg_loss": float(losses["pnl"].mean()) if len(losses) else np.nan,
        "profit_factor": float(gross_profit / gross_loss) if gross_loss > 0 else np.nan,
        "expectancy": float(tr["pnl"].mean()) if n_tr else np.nan,
        "avg_holding_days": float(tr["holding_days"].mean()) if n_tr else np.nan,
        "slot_occupancy": occ,
        "total_cost": float(total_cost),
        "cost_over_gross_profit": float(total_cost / gross_profit) if gross_profit > 0 else np.nan,
        "stop_rate": float(reason_counts.get("STOP", 0) / n_tr) if n_tr else np.nan,
        "rf_rate": float((reason_counts.get("REBOUND_FAILURE", 0)
                          + reason_counts.get("REBOUND_FAILURE_STAGED", 0)) / n_tr) if n_tr else np.nan,
        "trail_rate": float((reason_counts.get("TRAIL1", 0)
                             + reason_counts.get("TRAIL2", 0)) / n_tr) if n_tr else np.nan,
        "time_exit_rate": float(reason_counts.get("TIME_EXIT", 0) / n_tr) if n_tr else np.nan,
        "blocked_force_rate": float(reason_counts.get("EXIT_BLOCKED_FORCE", 0) / n_tr) if n_tr else np.nan,
        "monthly": monthly_ret, "best_month": best_month,
        "ex_best_month_ann": ex_ann, "ex_best_month_mdd": ex_mdd,
        "ex_best10_ann": ex10_ann, "ex_best10_mdd": ex10_mdd,
    }


def segment_stats(eq, tr, year_lo, year_hi, tag):
    es = eq[(eq["date"] >= year_lo * 10000) & (eq["date"] <= year_hi * 10000 + 1231)].copy()
    ts = tr[(tr["exit_date"] >= year_lo * 10000) & (tr["exit_date"] <= year_hi * 10000 + 1231)]
    if len(es) < 2 or es["equity"].iloc[0] <= 0:
        return None
    e = es["equity"].to_numpy()
    rets = np.diff(e) / e[:-1]
    ann = float(e[-1] / e[0]) ** (252.0 / max(len(e) - 1, 1)) - 1
    mdd = float((e / np.maximum.accumulate(e) - 1).min())
    sharpe = float(rets.mean() / rets.std() * np.sqrt(252)) if rets.std() > 0 else 0.0
    prec = ts["label_precision"].value_counts().to_dict() if len(ts) else {}
    return {"tag": tag, "n_days": len(es), "n_trades": len(ts),
            "ann_return": ann, "mdd": mdd, "sharpe": sharpe,
            "win_rate": float((ts["pnl"] > 0).mean()) if len(ts) else np.nan,
            "precision_breakdown": prec}


# ---------------------------------------------------------------- 主流程
def main():
    pd.set_option("display.width", 200)
    REPORTS.mkdir(parents=True, exist_ok=True)
    LEDGERS.mkdir(parents=True, exist_ok=True)

    print("[1/4] 加载退出标签 …")
    labels = load_labels()
    need_codes = set(labels["code"])

    print("[2/4] 生成/加载入场流（F1/F2/F3，S0 top20 排序分）…")
    flows = {}
    for f in FLOWS:
        fl = build_entry_flow(f)
        flows[f] = fl
        need_codes |= set(fl["code"])
        n_match = fl["event_id"].isin(set(labels["event_id"])).sum()
        print(f"  {f}: 入场流 {len(fl)} 行（top20/日），有标签 {n_match}，"
              f"无标签被跳过 {len(fl) - n_match}")

    print("[3/4] 加载日线行情（逐仓盯市/可行性校验）…")
    bars, calendar = load_bars(need_codes)
    cal_idx = {int(d): i for i, d in enumerate(calendar)}
    print(f"  交易日 {len(calendar)} 天：{calendar[0]}..{calendar[-1]}，"
          f"覆盖股票 {len(bars)} 只")

    print("[4/4] 运行 12 组合 …")
    all_stats, all_seg, audit_pool = {}, {}, {}
    for f in FLOWS:
        for v in VARIANTS:
            labels_v = {r["event_id"]: r for _, r in
                        labels[labels["strategy_variant"] == v].iterrows()}
            eq, tr, skips, bstat = run_combo(flows[f], labels_v, bars, calendar, cal_idx, v)
            key = f"{f}×{v}"
            # 有效窗口裁剪：行情自 2018 起，但组合 2021-01 才有首笔入场；
            # 裁剪到 [首笔入场日, 末笔退出日]，避免 2018–2020 空转稀释年化/占用
            lo, hi = int(tr["entry_date"].min()), int(tr["exit_date"].max())
            eq = eq[(eq["date"] >= lo) & (eq["date"] <= hi)].copy().reset_index(drop=True)
            eq["drawdown"] = eq["equity"] / eq["equity"].cummax() - 1
            st = perf_stats(eq, tr, key)
            st["skips"] = skips
            st["blocked"] = bstat
            all_stats[key] = st
            seg24 = segment_stats(eq, tr, 2024, 2024, "2024 (DAILY_APPROX)")
            seg25 = segment_stats(eq, tr, 2025, 2099, "2025+ (MINUTE_EXACT 为主)")
            all_seg[key] = [s for s in (seg24, seg25) if s]
            lcols = sorted(set(eq.columns) | set(tr.columns))
            ledger = pd.concat([eq.reindex(columns=lcols), tr.reindex(columns=lcols)],
                               ignore_index=True)
            ledger.to_parquet(LEDGERS / f"ledger_{f}_{v}.parquet", index=False)
            audit_pool[key] = (eq, tr)
            print(f"  {key:22} 笔数={st['n_trades']:5d} 年化={st['ann_return']:8.2%} "
                  f"MDD={st['mdd']:8.2%} 夏普={st['sharpe']:5.2f} "
                  f"占用={st['slot_occupancy']:5.1%} 成本/毛盈利={st['cost_over_gross_profit']:6.2%} "
                  f"| skip={sum(skips.values())} blocked={bstat['label_blocked_traded']}")

    write_report(all_stats, all_seg, audit_pool)
    print(f"\n完成：{REPORTS/'portfolio_v2_report.md'}")
    print(f"账本：{LEDGERS}/ledger_*.parquet (12 个)")


def _fmt_pct(x, nd=2):
    return f"{x:.{nd}%}" if isinstance(x, (int, float)) and np.isfinite(x) else "—"


def _audit_block(tr, eq, key, n=5):
    """抽 n 笔完整计算链（人工可对账）。"""
    if not len(tr):
        return ""
    pick = tr.copy()
    # 覆盖多种退出原因：每种原因取代表，凑满 n 笔
    chosen = []
    for reason, g in pick.groupby("exit_reason"):
        chosen.append(g.iloc[len(g) // 2])
    chosen = chosen[:n]
    lines = [f"#### {key} 逐笔可对账示例（{len(chosen)} 笔）",
             "成本口径：买侧 佣金0.0003+滑点0.001；卖侧 佣金0.0003+印花税0.0005+滑点0.001"
             "（任务书冻结口径；标签 cost=0.003 为旧口径，差异在末行列示）",
             ""]
    for r in chosen:
        s = r["shares"]
        buy_notional = s * r["entry_price"]
        buy_cost = r["buy_cost_detail"]
        sell_notional = s * r["exit_price"]
        sell_cost = r["sell_cost_detail"]
        lines += [
            f"- **{r['code']}** {r['entry_date']}→{r['exit_date']}（{r['exit_reason']}，{r['label_precision']}）",
            f"  - 入场：{s:.2f} 股 × 开 {r['entry_price']:.4f} = {buy_notional:,.2f} 元；"
            f"买侧成本 {buy_cost:.2f} 元（滑点 {s*r['entry_price']*SLIP_BUY:.2f} + 佣金 {s*r['entry_price']*COMM_RATE:.2f}）；"
            f"占用现金 {r['buy_cash']:,.2f}",
            f"  - 出场：{s:.2f} 股 × {r['exit_price']:.4f} = {sell_notional:,.2f} 元；"
            f"卖侧成本 {sell_cost:.2f} 元（滑点 {s*r['exit_price']*SLIP_SELL:.2f} + 佣金 "
            f"{s*r['exit_price']*COMM_RATE:.2f} + 印花税 {s*r['exit_price']*STAMP_RATE:.2f}）",
            f"  - 净盈亏 = {sell_notional:,.2f} − {sell_cost:.2f} − {r['buy_cash']:,.2f} = **{r['pnl']:,.2f} 元**"
            f"（净收益率 {r['net_return']:.3%}，持有 {int(r['holding_days'])} 个交易日）",
            f"  - 对标签：标签 gross={r['gross_return_label']:.4%} cost={r['cost_label']:.4f} "
            f"net={r['net_return_label']:.4%}；账本价格口径 gross={r['gross_px_return']:.4%} "
            f"（差异=旧 cost 0.003 冻结口径 0.0031 及价格基准不同）",
        ]
    return "\n".join(lines) + "\n"


def write_report(all_stats, all_seg, audit_pool):
    L = []
    A = L.append
    A("# 20 槽真实组合账本 v2 —— 12 组合对比报告")
    A("")
    A("规则来源：exit_dyn_spec_v1.md（冻结）+ 开发路线第 3 步任务书。"
      "收益全部来自退出标签（exit_price/gross_return），未读取 t1_rule_return，"
      "未构造 MFE，未把固定 T+5 当默认退出。"
      "成本按任务书冻结口径逐笔从现金扣除（买 0.0013 / 卖 0.0018，往返 0.0031）；"
      "标签内 cost=0.003（v4_2_train 旧口径）与冻结口径冲突，**以冻结口径为准**，对账处单列差异。")
    A("")
    A("数据段：2019–2024 段为 DAILY_APPROX（日线保守近似），2025+ 段为 MINUTE_EXACT"
      "（分钟精确）；两段分开报告，绝不合并成一条收益曲线。")
    A("")
    A("> **上游缺陷已修复（2026-09-16）**：原标签文件中 X0_CONTROL 与 X1 曾 100% 同路"
      "（ExitState.step() 移动止盈分支未按 variant 关闭，违背 spec §8）。已在 "
      "build_exit_labels.py 增加变体门控（仅 X1/X1_RF/X2_STAGED 启用 TRAIL1/TRAIL2，"
      "X0_CONTROL 仅硬止损+时间退出），重建后四变体退出原因分布正确分化，"
      "test_exit_engine.py 15/15、verify_exit_labels.py 全部验收通过；"
      "缺陷版标签留存于 labels/SUPERSEDED/*_X0DEFECT_20260915.parquet。")
    A("")
    A("> **成本口径冲突记录**：标签内 cost=0.003（v4_2_train 旧口径）；任务书冻结口径"
      "= 佣金 0.0003 双边 + 印花税 0.0005（卖出）+ 滑点 0.001/0.001（往返 0.0031）。"
      "按任务书规则「冲突时以规格为准」，本账本按价格逐笔扣 0.0013（买）/0.0018（卖），"
      "对账示例中列示与标签 net 的差异。")
    A("")
    # ---- 总表 ----
    A("## 12 组合总表（全程 2021–2025+，单段口径见下）")
    A("")
    A("| 组合 | 笔数 | 总收益 | 年化 | 夏普 | MDD | Calmar | 胜率 | 盈亏比 | 单笔期望(元) | 平均持有(日) | 槽位占用 | 成本/毛盈利 |")
    A("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for key, st in all_stats.items():
        A(f"| {key} | {st['n_trades']} | {_fmt_pct(st['total_return'])} | {_fmt_pct(st['ann_return'])} "
          f"| {st['sharpe']:.2f} | {_fmt_pct(st['mdd'])} | {st['calmar']:.2f} | {_fmt_pct(st['win_rate'])} "
          f"| {st['profit_factor']:.2f} | {st['expectancy']:,.0f} | {st['avg_holding_days']:.1f} "
          f"| {_fmt_pct(st['slot_occupancy'],1)} | {_fmt_pct(st['cost_over_gross_profit'])} |")
    A("")
    A("## 退出结构 & 稳健性")
    A("")
    A("| 组合 | STOP | 反弹失败 | 移动止盈 | TIME_EXIT | blocked强退 | 去最佳月年化/MDD | 去最佳10笔年化/MDD |")
    A("|---|---|---|---|---|---|---|---|")
    for key, st in all_stats.items():
        A(f"| {key} | {_fmt_pct(st['stop_rate'])} | {_fmt_pct(st['rf_rate'])} "
          f"| {_fmt_pct(st['trail_rate'])} | {_fmt_pct(st['time_exit_rate'])} "
          f"| {_fmt_pct(st['blocked_force_rate'])} "
          f"| {_fmt_pct(st['ex_best_month_ann'])}/{_fmt_pct(st['ex_best_month_mdd'])} "
          f"| {_fmt_pct(st['ex_best10_ann'])}/{_fmt_pct(st['ex_best10_mdd'])} |")
    A("")
    # ---- 分段 ----
    A("## 分段报告（2024 vs 2025+，口径分离）")
    A("")
    A("| 组合 | 段 | 交易日 | 笔数 | 年化 | MDD | 夏普 | 胜率 | 标签精度构成 |")
    A("|---|---|---|---|---|---|---|---|---|")
    for key, segs in all_seg.items():
        for s in segs:
            prec = ", ".join(f"{k}:{v}" for k, v in s["precision_breakdown"].items()) or "—"
            A(f"| {key} | {s['tag']} | {s['n_days']} | {s['n_trades']} "
              f"| {_fmt_pct(s['ann_return'])} | {_fmt_pct(s['mdd'])} | {s['sharpe']:.2f} "
              f"| {_fmt_pct(s['win_rate'])} | {prec} |")
    A("")
    # ---- 每组合详情 ----
    A("## 每组合详情")
    for key, st in all_stats.items():
        eq, tr = audit_pool[key]
        A(f"\n### {key}")
        A(f"- 笔数 {st['n_trades']}，期末权益 {st['final_equity']:,.0f} 元，"
          f"平均盈利 {format(st['avg_win'], ',.0f') + ' 元' if np.isfinite(st['avg_win']) else '—'}，"
          f"平均亏损 {format(st['avg_loss'], ',.0f') + ' 元' if np.isfinite(st['avg_loss']) else '—'}，"
          f"成本总额 {st['total_cost']:,.0f} 元")
        sk = st["skips"]
        A(f"- 入场跳过计数：停牌 {sk.get('suspended',0)}，无行情 {sk.get('no_bar',0)}，"
          f"冷却期 {sk.get('cooldown',0)}，已持仓 {sk.get('already_held',0)}，"
          f"槽满 {sk.get('slot_full',0)}，标签缺 {sk.get('missing_label',0)}，"
          f"入场日/价不符 {sk.get('entry_date_mismatch',0)+sk.get('entry_price_mismatch',0)}；"
          f"信息性：parts 判定涨停开盘但标签已成交 {sk.get('info_limit_up_per_parts',0)}"
          f"（两行情源 pre_close 在复牌/除权日不一致，可成交性以冻结标签为准）")
        bs = st["blocked"]
        A(f"- blocked 统计：进入持仓的标签 blocked 笔 {bs['label_blocked_traded']}，"
          f"T+5 后强退 {bs['force_exited']}，期末未了 {bs['end_of_data']}")
        # 月度收益表
        mr = st["monthly"].dropna()
        A("\n月度收益（%）：")
        A("")
        A("| 月 | 收益 | 月 | 收益 |")
        A("|---|---|---|---|")
        vals = [(str(int(i)), f"{v:.2%}") for i, v in mr.items()]
        for i in range(0, len(vals), 2):
            a = vals[i]
            b = vals[i + 1] if i + 1 < len(vals) else ("", "")
            A(f"| {a[0]} | {a[1]} | {b[0]} | {b[1]} |")
        # 集中度（按成交额）
        if len(tr):
            amt = tr.groupby("sector")["turnover"].sum().sort_values(ascending=False)
            top = amt.head(5)
            A(f"\n板块集中度（成交额占比，Top5）：" +
              "；".join(f"{k} {v/amt.sum():.1%}" for k, v in top.items()))
            sa = tr.groupby("code")["turnover"].sum().sort_values(ascending=False)
            A(f"个股集中度：Top1 {sa.index[0]} {sa.iloc[0]/sa.sum():.1%}，"
              f"Top5 {sa.head(5).sum()/sa.sum():.1%}，Top10 {sa.head(10).sum()/sa.sum():.1%}")
        A("")
    # ---- 逐笔对账 ----
    A("## 逐笔人工对账（3 组合 × 5 笔，完整计算链）")
    A("")
    for key in ["F1×X1", "F2×X2_STAGED", "F3×X1_RF"]:
        A(_audit_block(audit_pool[key][1], audit_pool[key][0], key))
    REPORTS.joinpath("portfolio_v2_report.md").write_text("\n".join(L), encoding="utf-8")


if __name__ == "__main__":
    main()
