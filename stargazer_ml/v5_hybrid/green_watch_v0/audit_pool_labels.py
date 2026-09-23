"""audit_pool_labels.py — 池宇宙标签 50 条抽样人工对账。

抽 50 条池标签（覆盖 X0/X1/X1_RF/X2、DAILY/MINUTE、STOP/TIME_EXIT/TRAIL/RF 等原因），
用独立的朴素 pandas 代码从原始行情重放退出路径，逐条打印可对账明细
（入场判定、逐日 OHLC vs 止损线、退出触发点），并自动校验与标签一致。

运行：python audit_pool_labels.py [--n 50] [--seed 11]
输出：审计打印（stdout）+ audit_pool_labels_50.txt
"""
import argparse
import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent
BARS_GLOB = "D:/a股数据/data_daily_parts/part_*.parquet"


def load_px(code):
    parts = [pd.read_parquet(f, columns=["date", "code", "open", "high", "low",
                                         "close"]) for f in sorted(glob.glob(BARS_GLOB))]
    df = pd.concat(parts, ignore_index=True)
    df["code"] = df["code"].astype(str).str.zfill(6)
    df = df[df["code"] == code].sort_values("date").reset_index(drop=True)
    df["pc"] = df["close"].shift(1).fillna(df["close"])
    return df


def naive_replay(px, rec):
    """朴素重放 X0 语义（日线口径）：返回 (ok, 说明列表)。"""
    L = []
    i = px.index[px["date"] == rec["signal_date"]]
    if not len(i):
        return False, ["signal_date 非交易日"]
    i = int(i[0])
    entry_idx = None
    for k in range(1, 6):
        cand = i + k
        if cand >= len(px):
            break
        r = px.iloc[cand]
        if r["open"] >= r["pc"] * 1.098 - 1e-9:
            L.append(f"T+{k} {int(r['date'])} 涨停开盘顺延")
            continue
        entry_idx = cand
        break
    if entry_idx is None or entry_idx + 1 >= len(px):
        return False, L + ["无法入场"]
    e = px.iloc[entry_idx]
    L.append(f"入场 T+{entry_idx-i} {int(e['date'])} open={e['open']:.3f} "
             f"(标签 entry_price={rec['entry_price']:.3f})")
    t5 = min(entry_idx + 4, len(px) - 1)
    stop = rec["initial_stop_price"]
    for d in range(entry_idx + 1, t5 + 1):
        r = px.iloc[d]
        locked = r["close"] <= r["pc"] * 0.902 + 1e-9 and abs(r["close"] - r["low"]) < 1e-6
        L.append(f"  {int(r['date'])} O={r['open']:.3f} H={r['high']:.3f} "
                 f"L={r['low']:.3f} C={r['close']:.3f} stop={stop:.3f}"
                 f"{' 跌停锁死' if locked else ''}")
        if r["open"] <= stop:
            got = ("STOP", r["open"], int(r["date"]))
            break
        if r["low"] <= stop:
            got = ("STOP", stop, int(r["date"]))
            break
    else:
        r = px.iloc[t5]
        got = ("TIME_EXIT", r["close"], int(r["date"]))
    L.append(f"朴素重放: {got[0]} price={got[1]:.4f} date={got[2]}")
    ok = (got[0] == rec["exit_reason"] and abs(got[1] - rec["exit_price"]) < 1e-6
          and got[2] == int(str(rec["exit_time"])[:8])
          and abs(e["open"] - rec["entry_price"]) < 1e-9)
    return ok, L


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--seed", type=int, default=11)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    out_lines = []

    def P(s=""):
        print(s)
        out_lines.append(str(s))

    for prec in ("daily_approx", "minute_exact"):
        df = pd.read_parquet(BASE / "labels_pool" / f"exit_labels_pool_{prec}.parquet")
        # 分层抽样：覆盖各变体与退出原因
        picks = []
        for (v, rsn), g in df.groupby(["strategy_variant", "exit_reason"]):
            k = min(max(1, args.n // 12), len(g))
            picks.append(g.sample(k, random_state=args.seed))
        sel = pd.concat(picks).sample(frac=1, random_state=args.seed).head(args.n)
        n_ok = 0
        P(f"\n{'='*78}\n== {prec} 抽查 {len(sel)} 条 ==\n{'='*78}")
        cache = {}
        for rec in sel.itertuples(index=False):
            code = str(rec.code).zfill(6)
            if code not in cache:
                cache[code] = load_px(code)
            px = cache[code]
            if prec == "minute_exact":
                # 分钟口径：只校验日线骨架（入场日/入场价/退出日/原因日线一致性），
                # 分钟级触发点在标签引擎内已 bit 级对账（validate_labels_pool.py）
                sub = dict(entry_price=rec.entry_price, initial_stop_price=rec.initial_stop_price,
                           exit_reason=rec.exit_reason, exit_price=rec.exit_price,
                           exit_time=rec.exit_time, signal_date=rec.signal_date)
                ok, L = naive_replay(px, sub)
                P(f"\n[{rec.strategy_variant}] {code} T={rec.signal_date} "
                  f"-> {rec.exit_reason} @ {rec.exit_time} px={rec.exit_price:.3f} "
                  f"net={rec.net_return:+.3%}（分钟口径：日线骨架校验{'OK' if ok else '差异见下'}）")
                for l in L:
                    P("  " + l)
                n_ok += ok
            else:
                sub = dict(entry_price=rec.entry_price, initial_stop_price=rec.initial_stop_price,
                           exit_reason=rec.exit_reason, exit_price=rec.exit_price,
                           exit_time=rec.exit_time, signal_date=rec.signal_date)
                ok, L = naive_replay(px, sub)
                P(f"\n[{rec.strategy_variant}] {code} T={rec.signal_date} "
                  f"atr_ref={rec.atr_ref:.3f} -> {rec.exit_reason} @ {rec.exit_time} "
                  f"px={rec.exit_price:.3f} net={rec.net_return:+.3%} "
                  f"{'OK' if ok else '<<< 不一致'}")
                for l in L:
                    P("  " + l)
                n_ok += ok
        P(f"\n{prec}: 朴素重放一致 {n_ok}/{len(sel)}")
    (BASE / "audit_pool_labels_50.txt").write_text("\n".join(out_lines),
                                                   encoding="utf-8")
    print(f"\n-> audit_pool_labels_50.txt")


if __name__ == "__main__":
    main()
