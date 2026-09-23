"""pool_backtest_e.py — EXIT-GREEN-v2 结构消融：E0–E5 × {ALL, BOTTOM, DIP} 20 槽账本。

入场流：池 WATCH+ 每日 top-20（ALL 按 primary_score；BOTTOM/DIP 各按本类分数，不混排）。
账本规则：完全复用 portfolio_backtest_v2.run_combo（20 槽/冷却 5 日/T+1 开盘/
逐日盯市/成本买 0.0013 + 卖 0.0018）。收益来自 E 系标签（DAILY_APPROX 全历史）。
指标：年化/MDD/夏普/胜率/平均盈利/平均亏损/盈亏比/单笔期望/MFE capture
（退出价 ÷ 持有期日线最高价，均值与中位）/exit_reason 归因（笔数 × 均净收益）。

输出：ablation_summary.json + ledgers_e/ledger_{layer}_{variant}.parquet
运行：python pool_backtest_e.py
"""
import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE.parent / "green_exit_v1"))
import portfolio_backtest_v2 as pbv  # noqa: E402

POOL_F = BASE / "pool_daily.parquet"
LAB_F = BASE / "labels_pool_e" / "exit_labels_pool_daily_approx.parquet"
LEDGERS = BASE / "ledgers_e"
VARIANTS = ["E0", "E1", "E2", "E3", "E4", "E5"]
LAYERS = {"ALL": None, "BOTTOM": "BOTTOM_SETUP", "DIP": "INTRADAY_DIP_SETUP"}
N_TOP = 20


def build_flows(pool):
    w = pool[pool["primary_tier"].isin(["WATCH", "HIGH_PRIORITY_WATCH"])].copy()
    flows = {}
    for name, setup in LAYERS.items():
        sub = w if setup is None else w[w["primary_setup"] == setup]
        score = "primary_score" if setup is None else \
            ("bottom_score" if setup == "BOTTOM_SETUP" else "intraday_dip_score")
        rows = []
        for d, g in sub.groupby("date"):
            g = g.sort_values(score, ascending=False).head(N_TOP)
            for rank, (_, r) in enumerate(g.iterrows(), 1):
                rows.append({"signal_date": int(d), "code": r["code"],
                             "event_id": f"GW:{r['code']}:{int(d)}",
                             "score": float(r[score]), "rank": rank, "sector": "NA"})
        flows[name] = pd.DataFrame(rows).sort_values(["signal_date", "rank"])
        print(f"  flow {name}: {len(flows[name])} 行 / "
              f"{flows[name]['signal_date'].nunique()} 日")
    return flows


def mfe_capture(tr, bars, cal_idx):
    """退出价 / 持有期（T+1入场日..退出日）日线最高价。"""
    caps = []
    for _, r in tr.iterrows():
        b = bars.get(r["code"])
        if b is None:
            caps.append(np.nan)
            continue
        i0 = b["idx"].get(int(r["entry_date"]))
        i1 = b["idx"].get(int(r["exit_date"]))
        if i0 is None or i1 is None:
            caps.append(np.nan)
            continue
        hh = b["high"][i0:i1 + 1]
        hh = hh[np.isfinite(hh)]
        caps.append(r["exit_price"] / hh.max() if len(hh) and hh.max() > 0 else np.nan)
    return pd.Series(caps, index=tr.index)


def run_layer_variant(flow, labels_v, bars, calendar, cal_idx, tag):
    eq, tr, skips, bstat = pbv.run_combo(flow, labels_v, bars, calendar, cal_idx, "E")
    if not len(tr):
        return None, None, None, None
    lo, hi = int(tr["entry_date"].min()), int(tr["exit_date"].max())
    eq = eq[(eq["date"] >= lo) & (eq["date"] <= hi)].copy().reset_index(drop=True)
    eq["drawdown"] = eq["equity"] / eq["equity"].cummax() - 1
    st = pbv.perf_stats(eq, tr, tag)
    tr = tr.copy()
    tr["mfe_capture"] = mfe_capture(tr, bars, cal_idx)
    segs = []
    for y0, y1 in ((2021, 2023), (2024, 2024), (2025, 2099)):
        s = pbv.segment_stats(eq, tr, y0, y1, f"{y0}-{min(y1, 2026)}")
        if s:
            segs.append(s)
    return eq, st, tr, segs


def main():
    pd.set_option("display.width", 220)
    LEDGERS.mkdir(parents=True, exist_ok=True)
    print("[1/4] 池 + E 标签 …")
    pool = pd.read_parquet(POOL_F)
    pool["code"] = pool["code"].astype(str).str.zfill(6)
    lab = pd.read_parquet(LAB_F)
    lab["code"] = lab["code"].astype(str).str.zfill(6)
    lab["exit_date"] = lab["exit_time"].astype(str).str[:8].astype(int)
    print(f"  池 {len(pool)} 行；E 标签 {len(lab)} 行 "
          f"（{lab['strategy_variant'].value_counts().to_dict()}）")

    print("[2/4] 入场流 …")
    flows = build_flows(pool)

    print("[3/4] 行情 …")
    need = set(lab["code"])
    for f in flows.values():
        need |= set(f["code"])
    bars, calendar = pbv.load_bars(need)
    cal_idx = {int(d): i for i, d in enumerate(calendar)}
    print(f"  交易日 {len(calendar)}，票 {len(bars)}")

    print("[4/4] 18 组账本 …")
    summary, ledgers_tr = {}, {}
    for layer, flow in flows.items():
        for v in VARIANTS:
            lv = lab[lab["strategy_variant"] == v]
            labels_v = {r["event_id"]: r for _, r in lv.iterrows()}
            tag = f"{layer}×{v}"
            eq, st, tr, segs = run_layer_variant(flow, labels_v, bars, calendar,
                                                 cal_idx, tag)
            if st is None:
                print(f"  {tag}: 无成交")
                continue
            # 归因与 MFE
            tr2 = tr.copy()
            tr2["yr"] = tr2["exit_date"] // 10000
            attr = tr2.groupby("exit_reason").agg(
                n=("pnl", "size"), mean_net=("net_return", "mean"),
                mean_mfe=("mfe_capture", "mean")).reset_index()
            wins, losses = tr2[tr2["pnl"] > 0], tr2[tr2["pnl"] <= 0]
            summary[tag] = dict(
                layer=layer, variant=v, n_trades=int(st["n_trades"]),
                ann=st["ann_return"], mdd=st["mdd"], sharpe=st["sharpe"],
                win=st["win_rate"], avg_win=float(wins["pnl"].mean()),
                avg_loss=float(losses["pnl"].mean()),
                profit_factor=st["profit_factor"], expectancy=st["expectancy"],
                mfe_mean=float(tr2["mfe_capture"].mean()),
                mfe_med=float(tr2["mfe_capture"].median()),
                slot_occupancy=st["slot_occupancy"],
                avg_holding=st["avg_holding_days"],
                attribution=attr.to_dict("records"), segments=segs)
            ledgers_tr[tag] = tr2
            lcols = sorted(set(eq.columns) | set(tr.columns))
            pd.concat([eq.reindex(columns=lcols), tr.reindex(columns=lcols)],
                      ignore_index=True).to_parquet(
                LEDGERS / f"ledger_{layer}_{v}.parquet", index=False)
            print(f"  {tag:14s} 笔={st['n_trades']:5d} 年化={st['ann_return']:7.2%} "
                  f"MDD={st['mdd']:7.2%} 胜率={st['win_rate']:5.1%} "
                  f"MFE均值={summary[tag]['mfe_mean']:.3f} 期望={st['expectancy']:7.0f}元")

    (BASE / "ablation_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=1, default=str),
        encoding="utf-8")
    print(f"\n→ {LEDGERS}/ + ablation_summary.json")


if __name__ == "__main__":
    main()
