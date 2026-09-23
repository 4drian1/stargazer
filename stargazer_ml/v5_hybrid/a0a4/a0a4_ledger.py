"""a0a4_ledger — X0-style 20 槽真实账本 (OOF 预测, test 折内, 无未来信息).

机制 (预注册):
  每个交易日 t (plan_date, 决策于 P_CLOSE): 候选 = 当日 OOF 预测 (timing 轨, seed 均值)
  按 pred 降序选取, 填充空槽 (最多 20 并发持仓), 同股不重复持仓,
  同股退出后冷却 5 个交易日; 成交日 = D = t 的下一交易日 (09:31 fill, 见执行引擎)
  仓位 = 当前权益 / 20 (每笔等权); 退出 = x0_exit_sim 的 X0-style 规则
  权益按已实现收益复利 (无日内 MTM, 近似口径已注明)

A0 控制臂: 用旧系统自身分数 intraday_dip_score 代替模型预测排序 (V0 历史对照)。

产出: x0_ledger/{a1..a4,a0_control}/trades.parquet + summary.json
      x0_ledger/ledger_report.json
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
import a0a4_config as C  # noqa

sys.path.insert(0, str(C.PREFLIGHT))
import trading_calendar as TC  # noqa

LEDGER = C.RUN / "x0_ledger"
SLOTS = C.X0_STYLE["slots"]
COOLDOWN_TD = 5


def run_ledger(trades_df: pd.DataFrame, cal: list, label: str) -> dict:
    """trades_df: sample_id, symbol, plan_date, fill_day, pred, net, exit_date."""
    t0 = time.time()
    cal_arr = np.asarray(cal)
    tds = {d: i for i, d in enumerate(cal)}
    df = trades_df.sort_values(["plan_date", "pred"], ascending=[True, False])
    cash = 1.0
    equity = 1.0
    active = []          # dicts: code, exit_i, stake, net
    cooldown = {}        # code -> td index until blocked
    trades = []
    occ_sum = n_days = 0
    plan_dates = sorted(df["plan_date"].unique())
    first_i, last_i = None, None
    for t in plan_dates:
        ti = tds.get(int(t))
        if ti is None:
            continue
        if first_i is None:
            first_i = ti
        # 结算到期持仓
        still = []
        for p in active:
            if tds.get(int(p["exit_date"]), 10**9) <= ti:
                cash += p["stake"] * (1.0 + p["net"])
                trades.append({**p["meta"], "net": p["net"],
                               "stake": p["stake"]})
                cooldown[p["code"]] = ti + COOLDOWN_TD
            else:
                still.append(p)
        active = still
        # 冷却到期
        cooldown = {c: u for c, u in cooldown.items() if u > ti}
        occupied = {p["code"] for p in active}
        free = SLOTS - len(active)
        day = df[df["plan_date"] == t]
        n_days += 1
        occ_sum += len(active)
        last_i = ti
        if free <= 0:
            continue
        taken = 0
        for r in day.itertuples(index=False):
            if taken >= free:
                break
            code = r.symbol
            if code in occupied or code in cooldown:
                continue
            stake = cash / SLOTS if free > 0 else 0.0
            if stake <= 0:
                break
            cash -= stake
            active.append({"code": code, "exit_date": int(r.exit_date),
                           "stake": stake, "net": float(r.net),
                           "meta": {"sample_id": r.sample_id, "symbol": code,
                                    "plan_date": int(r.plan_date),
                                    "fill_day": int(r.fill_day),
                                    "pred": float(r.pred)}})
            occupied.add(code)
            taken += 1
    # 清算剩余
    for p in active:
        cash += p["stake"] * (1.0 + p["net"])
        trades.append({**p["meta"], "net": p["net"], "stake": p["stake"]})
    total = cash - 1.0
    n_td = max(1, (last_i - first_i + 1) if first_i is not None else 1)
    ann = (1.0 + total) ** (252.0 / n_td) - 1.0 if total > -1 else -1.0
    tr = pd.DataFrame(trades)
    if len(tr):
        eq = 1.0 * (1.0 + tr.sort_values("plan_date")["net"]).cumprod()
        peak = eq.cummax()
        mdd = float((eq / peak - 1.0).min())
    else:
        mdd = 0.0
    summ = {"label": label, "n_trades": int(len(tr)),
            "win_rate": _r(float((tr["net"] > 0).mean())) if len(tr) else None,
            "avg_net": _r(float(tr["net"].mean())) if len(tr) else None,
            "median_net": _r(float(tr["net"].median())) if len(tr) else None,
            "worst_net": _r(float(tr["net"].min())) if len(tr) else None,
            "total_return": _r(total), "ann_return": _r(ann),
            "max_drawdown_trade_level": _r(mdd),
            "avg_slot_occupancy": _r(occ_sum / max(1, n_days)),
            "n_decision_days": int(n_days), "span_trading_days": int(n_td),
            "elapsed_s": round(time.time() - t0, 1)}
    return summ, tr


def _r(x, nd=6):
    return None if x is None or not np.isfinite(x) else round(float(x), nd)


def main():
    LEDGER.mkdir(parents=True, exist_ok=True)
    cal = [int(x) for x in TC.load_calendar()]
    tds = {d: i for i, d in enumerate(cal)}

    def next_td(d):
        for x in cal:
            if x > d:
                return x
        return None

    exits = pd.read_parquet(C.DATASETS / "a14_x0_exits.parquet")
    uni = pd.read_parquet(C.DATASETS / "a14_universe.parquet",
                          columns=["sample_id", "symbol", "plan_date", "d_date",
                                   "timing_eligible", "intraday_dip_score"])
    base = exits.merge(uni, on="sample_id")
    base = base[base["timing_eligible"] == 1]
    base["fill_day"] = base["d_date"]

    report = {}
    for tier in ("a1", "a2", "a3", "a4"):
        p = pd.read_parquet(C.TIER_DIRS[tier] / "predictions.parquet")
        p = p[p["track"] == "timing"]
        oof = p.groupby("sample_id").agg(pred=("pred", "mean")).reset_index()
        t = base.merge(oof, on="sample_id")
        summ, tr = run_ledger(t[["sample_id", "symbol", "plan_date", "fill_day",
                                  "pred", "x0_net_return", "x0_exit_date"]]
                              .rename(columns={"x0_net_return": "net",
                                               "x0_exit_date": "exit_date"}),
                              cal, tier)
        d = LEDGER / tier
        d.mkdir(exist_ok=True)
        tr.to_parquet(d / "trades.parquet", index=False)
        (d / "summary.json").write_text(json.dumps(summ, ensure_ascii=False, indent=2),
                                        encoding="utf-8")
        report[tier] = summ
        print(f"[ledger] {tier}: {summ['n_trades']} trades ann={summ['ann_return']} "
              f"win={summ['win_rate']} mdd={summ['max_drawdown_trade_level']}")
    # A0 控制臂 (V0 宇宙 + 旧分数排序, 用 a0 自己的退出模拟)
    exits0 = pd.read_parquet(C.DATASETS / "a0_x0_exits.parquet")
    uni0 = pd.read_parquet(C.DATASETS / "a0_universe.parquet",
                           columns=["sample_id", "symbol", "plan_date", "d_date",
                                    "timing_eligible", "intraday_dip_score"])
    base0 = exits0.merge(uni0, on="sample_id")
    base0 = base0[base0["timing_eligible"] == 1]
    base0["fill_day"] = base0["d_date"]
    t = base0.copy()
    t["pred"] = t["intraday_dip_score"]
    summ, tr = run_ledger(t[["sample_id", "symbol", "plan_date", "fill_day",
                              "pred", "x0_net_return", "x0_exit_date"]]
                          .rename(columns={"x0_net_return": "net",
                                           "x0_exit_date": "exit_date"}),
                          cal, "a0_control")
    d = LEDGER / "a0_control"
    d.mkdir(exist_ok=True)
    tr.to_parquet(d / "trades.parquet", index=False)
    (d / "summary.json").write_text(json.dumps(summ, ensure_ascii=False, indent=2),
                                    encoding="utf-8")
    report["a0_control"] = summ
    print(f"[ledger] a0_control: {summ['n_trades']} trades ann={summ['ann_return']}")
    (LEDGER / "ledger_report.json").write_text(
        json.dumps({"x0_style": C.X0_STYLE, "slots": SLOTS,
                    "cooldown_trading_days": COOLDOWN_TD, "tiers": report},
                   ensure_ascii=False, indent=2, default=str), encoding="utf-8")


if __name__ == "__main__":
    main()
