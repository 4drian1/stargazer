"""pool_backtest.py — GREEN-WATCH 池 top-N 入场流 × 20 槽账本（X0_CONTROL 退出）。

账本规则完全复用 green_exit_v1/portfolio_backtest_v2.py（import 其 run_combo /
perf_stats / segment_stats / load_bars / load_labels）：20 槽 / 冷却 5 日 / 不可重复持仓 /
T+1 开盘入场 / 逐日盯市 / 成本按冻结口径买 0.0013 + 卖 0.0018 逐笔扣。
收益来自 X0_CONTROL 标签行（硬止损+时间退出；标签 cost=0.003 旧口径与冻结 0.0031
的差异只在报告注明，账本按冻结口径重算净盈亏）。

入场流（全部只用 T 日盘后信息：primary_score + tier）：
  GW_N20 / GW_N10      池 WATCH+ 每日 top-N（混合 primary_setup）
  GW_DIP_N20 / GW_DIP_N10  仅 INTRADAY_DIP_SETUP，按 intraday_dip_score 排序
  GW_BOT_N20 / GW_BOT_N10  仅 BOTTOM_SETUP，按 bottom_score 排序（不混排）
  GW_RAND_N20          每日从池 WATCH+ 随机抽 20（seed=42，幂等）

基线：F1×X0_CONTROL 既有账本（green_exit_v1/reports/ledgers/ledger_F1_X0_CONTROL.parquet）。

输出：ledgers/ledger_*.parquet + flow_*.parquet + run 摘要打印。
运行：d:/stargazer/.venv/Scripts/python.exe pool_backtest.py
"""
import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE.parent / "green_exit_v1"))
import portfolio_backtest_v2 as pbv  # noqa: E402

POOL_EV = BASE / "pool_events.parquet"
LEDGERS = BASE / "ledgers"
FLOWS_D = BASE / "flows"
F1_LEDGER = Path("d:/stargazer/stargazer_ml/v5_hybrid/green_exit_v1/reports/ledgers/ledger_F1_X0_CONTROL.parquet")
WATCH_TIERS = ["WATCH", "HIGH_PRIORITY_WATCH"]


def build_flows(ev):
    FLOWS_D.mkdir(parents=True, exist_ok=True)
    flows = {}
    elig = ev[(ev["matched"]) & (ev["primary_tier"].isin(WATCH_TIERS))].copy()
    print(f"WATCH+ 且已匹配标签的池事件：{len(elig)} 行，"
          f"{elig['signal_date'].nunique()} 个信号日")

    def mk(name, sub, score, n, seed=None):
        rows = []
        for d, g in sub.groupby("signal_date"):
            g = g.sort_values(score, ascending=False)
            if seed is not None:
                g = g.sample(n=min(n, len(g)), random_state=seed + int(d) % 100000)
            else:
                g = g.head(n)
            for rank, (_, r) in enumerate(g.iterrows(), 1):
                rows.append({"signal_date": int(d), "code": r["code"],
                             "event_id": r["event_id"], "score": float(r[score]),
                             "rank": rank, "sector": "NA"})
        f = pd.DataFrame(rows).sort_values(["signal_date", "rank"]).reset_index(drop=True)
        f.to_parquet(FLOWS_D / f"flow_{name}.parquet", index=False)
        flows[name] = f
        print(f"  {name}: {len(f)} 行 / {f['signal_date'].nunique()} 日")
        return f

    dip = elig[elig["primary_setup"] == "INTRADAY_DIP_SETUP"]
    bot = elig[elig["primary_setup"] == "BOTTOM_SETUP"]
    mk("GW_N20", elig, "primary_score", 20)
    mk("GW_N10", elig, "primary_score", 10)
    mk("GW_DIP_N20", dip, "intraday_dip_score", 20)
    mk("GW_DIP_N10", dip, "intraday_dip_score", 10)
    mk("GW_BOT_N20", bot, "bottom_score", 20)
    mk("GW_BOT_N10", bot, "bottom_score", 10)
    mk("GW_RAND_N20", elig, "primary_score", 20, seed=42)
    return flows


def trim_stats(eq, tr, label, segments=((2021, 2023), (2024, 2024), (2025, 2099))):
    lo, hi = int(tr["entry_date"].min()), int(tr["exit_date"].max())
    eq = eq[(eq["date"] >= lo) & (eq["date"] <= hi)].copy().reset_index(drop=True)
    eq["drawdown"] = eq["equity"] / eq["equity"].cummax() - 1
    st = pbv.perf_stats(eq, tr, label)
    segs = []
    for y0, y1 in segments:
        s = pbv.segment_stats(eq, tr, y0, y1, f"{y0}-{min(y1,2026)}")
        if s:
            segs.append(s)
    return eq, tr, st, segs


def main():
    pd.set_option("display.width", 220)
    LEDGERS.mkdir(parents=True, exist_ok=True)
    print("[1/4] 加载池事件 …")
    ev = pd.read_parquet(POOL_EV)
    ev["code"] = ev["code"].astype(str).str.zfill(6)
    ev = ev[ev["matched"]].copy()

    print("[2/4] 构建入场流 …")
    flows = build_flows(ev)

    print("[3/4] 加载标签与行情 …")
    labels = pbv.load_labels()
    x0 = labels[labels["strategy_variant"] == "X0_CONTROL"]
    labels_v = {r["event_id"]: r for _, r in x0.iterrows()}
    need_codes = set(ev["code"]) | set(x0["code"])
    bars, calendar = pbv.load_bars(need_codes)
    cal_idx = {int(d): i for i, d in enumerate(calendar)}
    print(f"  交易日 {len(calendar)}：{calendar[0]}..{calendar[-1]}，行情票 {len(bars)}")

    print("[4/4] 运行账本 …")
    results = {}
    for name, fl in flows.items():
        eq, tr, skips, bstat = pbv.run_combo(fl, labels_v, bars, calendar, cal_idx,
                                             "X0_CONTROL")
        eq, tr, st, segs = trim_stats(eq, tr, name)
        st["skips"], st["blocked"] = skips, bstat
        results[name] = (st, segs)
        lcols = sorted(set(eq.columns) | set(tr.columns))
        pd.concat([eq.reindex(columns=lcols), tr.reindex(columns=lcols)],
                  ignore_index=True).to_parquet(LEDGERS / f"ledger_{name}_X0.parquet",
                                                index=False)
        print(f"  {name:12s} 笔={st['n_trades']:5d} 年化={st['ann_return']:8.2%} "
              f"MDD={st['mdd']:8.2%} 夏普={st['sharpe']:5.2f} 胜率={st['win_rate']:6.2%} "
              f"占用={st['slot_occupancy']:5.1%} 单笔期望={st['expectancy']:8.0f}元")

    # ---- F1 基线（既有账本，同一指标口径重算）----
    print("[基线] F1×X0_CONTROL 既有账本 …")
    f1 = pd.read_parquet(F1_LEDGER)
    eq1 = f1[f1["row_type"] == "equity"].reset_index(drop=True)
    tr1 = f1[f1["row_type"] == "trade"].reset_index(drop=True)
    _, _, st1, segs1 = trim_stats(eq1, tr1, "F1xX0")
    results["F1xX0"] = (st1, segs1)
    print(f"  F1xX0       笔={st1['n_trades']:5d} 年化={st1['ann_return']:8.2%} "
          f"MDD={st1['mdd']:8.2%} 夏普={st1['sharpe']:5.2f} 胜率={st1['win_rate']:6.2%}")

    # 汇总落盘供报告使用
    summary = {}
    for k, (st, segs) in results.items():
        summary[k] = {kk: vv for kk, vv in st.items()
                      if kk not in ("monthly", "skips", "blocked")
                      and not isinstance(vv, (pd.Series, dict))}
        summary[k]["skips"] = dict(st.get("skips", {}))
        summary[k]["blocked"] = dict(st.get("blocked", {}))
        summary[k]["segments"] = segs
    (BASE / "backtest_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    print(f"\n→ {LEDGERS}/ledger_*.parquet + backtest_summary.json")


if __name__ == "__main__":
    main()
