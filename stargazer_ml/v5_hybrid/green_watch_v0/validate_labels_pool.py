"""validate_labels_pool.py — 池宇宙标签引擎小规模验证（全量跑之前必跑）。

验证 A（bit 级）：抽样 (code, signal_date)，用**原引擎** build_exit_labels.run_exit
（同一 parts 行情构造 raw_* px）与 label_engine_pool 的输出逐字段对比
（X0_CONTROL/X1/X1_RF/X2_STAGED 四变体 + E0 vs X0 日线等价性）。
验证 B（源差异解释）：与旧 F1 宇宙标签（stockdb raw+qfq 源）在重叠事件上对比
net_return，量化并解释差异（行情源不同：parts 前复权 vs stockdb raw）。

运行：python validate_labels_pool.py [--n-codes 100] [--per-code 4]
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent
GE = Path("d:/stargazer/stargazer_ml/v5_hybrid/green_exit_v1")
sys.path.insert(0, str(GE))
sys.path.insert(0, str(BASE))
import build_exit_labels as eng  # noqa: E402
import label_engine_pool as mine  # noqa: E402

POOL_F = BASE / "pool_daily.parquet"
OLD_DAILY = GE / "labels/exit_labels_daily_approx.parquet"
OLD_MINUTE = GE / "labels/exit_labels_minute_exact.parquet"

CMP_COLS = ["entry_date", "entry_price", "atr_ref", "initial_stop_price",
            "exit_time", "exit_price", "exit_reason", "exit_blocked",
            "gross_return", "net_return", "holding_days", "label_precision"]


def engine_px(px):
    df = pd.DataFrame({"date": px["date"], "raw_open": px["open"],
                       "raw_high": px["high"], "raw_low": px["low"],
                       "raw_close": px["close"]})
    return df


def run_engine_variant(pxdf, i, atr_ref, variant, minute_df, pcm, stm, code, sd):
    return eng.run_exit(pxdf, i, atr_ref, variant, minute_df, pcm, stm, code, sd, None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-codes", type=int, default=100)
    ap.add_argument("--per-code", type=int, default=4)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    pool = pd.read_parquet(POOL_F, columns=["date", "code", "primary_tier"])
    pool["code"] = pool["code"].astype(str).str.zfill(6)
    w = pool[pool["primary_tier"].isin(["WATCH", "HIGH_PRIORITY_WATCH"])]
    codes = sorted(w["code"].unique())
    # 抽样：前 1/3（含 F1 重叠高概率）+ 随机 2/3
    head = codes[: args.n_codes // 3]
    rest = rng.choice(codes[args.n_codes // 3:], args.n_codes - len(head), replace=False)
    pick = sorted(set(head) | set(rest.tolist()))
    print(f"抽样票 {len(pick)}")

    px_all = mine.load_parts(set(pick))
    stats = {"n": 0, "exact": 0, "mismatch": []}
    old_cmp = []
    old_daily = pd.read_parquet(OLD_DAILY)
    old_minute = pd.read_parquet(OLD_MINUTE)
    old = pd.concat([old_daily, old_minute], ignore_index=True)
    old = old[old["strategy_variant"] == "X0_CONTROL"]
    old = old.sort_values("label_precision").drop_duplicates("event_id", keep="last")
    old_ix = {(r["code"], int(r["signal_date"])): r for _, r in old.iterrows()}

    for code in pick:
        px = px_all.get(code)
        if px is None:
            continue
        n = len(px["date"])
        pc, atr14 = mine._prep_code(px)
        loc = {int(d): i for i, d in enumerate(px["date"])}
        dates = w[w["code"] == code]["date"].to_numpy()
        if not len(dates):
            continue
        sel = rng.choice(dates, min(args.per_code, len(dates)), replace=False)
        minute = mine.load_minute_slices(code)
        min_slice, vwap_map = minute if minute else (None, None)
        pxdf = engine_px(px)
        pcm = {int(pxdf.iloc[k]["date"]):
               float(pxdf.iloc[k - 1]["raw_close"]) if k > 0
               else float(pxdf.iloc[k]["raw_close"]) for k in range(len(pxdf))}
        stm = {int(pxdf.iloc[k]["date"]): bool(px["is_st"][k]) for k in range(len(pxdf))}
        mdf = pd.read_parquet(mine.MINUTE_DIR / f"{code}.parquet") \
            if (mine.MINUTE_DIR / f"{code}.parquet").exists() else None
        for sd in sel:
            sd = int(sd)
            i = loc.get(sd)
            if i is None:
                continue
            entry_idx = mine._find_entry(px, pc, i, n)
            if entry_idx is None:
                continue
            atr_ref = mine._atr_ref(atr14, entry_idx)
            if atr_ref is None:
                continue
            # ---- 我的引擎 ----
            mine_out = mine.run_event(px, pc, atr14, loc, sd, n, min_slice, vwap_map,
                                      ("X0_CONTROL", "X1", "X1_RF", "X2_STAGED", "E0"))
            # ---- 原引擎 ----
            e_atr = eng.compute_atr_ref(pxdf, entry_idx)
            for variant in ("X0_CONTROL", "X1", "X1_RF", "X2_STAGED"):
                mdf_y = mdf if sd >= 20250101 else None   # 引擎由调用方按年份门控分钟
                rec_e = run_engine_variant(pxdf, i, e_atr, variant, mdf_y, pcm, stm,
                                           code, sd)
                rec_m = mine_out.get(variant) if mine_out else None
                stats["n"] += 1
                if rec_e is None and rec_m is None:
                    stats["exact"] += 1
                    continue
                if rec_e is None or rec_m is None:
                    stats["mismatch"].append((code, sd, variant, "None 不一致"))
                    continue
                diffs = []
                for c in CMP_COLS:
                    a, b = rec_e.get(c), rec_m.get(c)
                    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
                        if not (np.isfinite(a) and np.isfinite(b)) or abs(a - b) > 1e-9:
                            diffs.append((c, a, b))
                    elif str(a) != str(b):
                        diffs.append((c, a, b))
                if diffs:
                    stats["mismatch"].append((code, sd, variant, diffs))
                else:
                    stats["exact"] += 1
            # ---- E0(日线) vs X0(日线) 等价性 ----
            e0 = mine_out.get("E0")
            x0 = mine_out.get("X0_CONTROL")
            if e0 and x0 and e0["label_precision"] == "DAILY_APPROX" \
               and x0["label_precision"] == "DAILY_APPROX":
                stats["n"] += 1
                d = [(c, x0[c], e0[c]) for c in CMP_COLS
                     if isinstance(x0[c], (int, float))
                     and abs(float(x0[c]) - float(e0[c])) > 1e-9]
                if d:
                    stats["mismatch"].append((code, sd, "E0vsX0", d))
                else:
                    stats["exact"] += 1
            # ---- 与旧 F1 标签对比（仅 X0）----
            key = (code, sd)
            if key in old_ix and mine_out.get("X0_CONTROL"):
                o = old_ix[key]
                m = mine_out["X0_CONTROL"]
                old_cmp.append(dict(code=code, sd=sd, prec=o["label_precision"],
                                    old_net=o["net_return"], new_net=m["net_return"],
                                    old_reason=o["exit_reason"],
                                    new_reason=m["exit_reason"],
                                    old_exit=o["exit_price"], new_exit=m["exit_price"]))

    print(f"\n== 验证 A：原引擎 vs 池引擎（含 E0≡X0 日线）==")
    print(f"对比 {stats['n']} 条，一致 {stats['exact']}，不一致 {len(stats['mismatch'])}")
    for m in stats["mismatch"][:10]:
        print("  MISMATCH:", m)

    print(f"\n== 验证 B：旧 F1 标签(stockdb) vs 池引擎(parts)，X0 重叠 = {len(old_cmp)} ==")
    if old_cmp:
        b = pd.DataFrame(old_cmp)
        b["diff"] = b["new_net"] - b["old_net"]
        print(b["diff"].describe().to_string())
        print(f"|diff|>0.5% 的条数：{(b['diff'].abs() > 0.005).sum()}")
        print("退出原因不一致：", (b["old_reason"] != b["new_reason"]).sum())
        big = b[b["diff"].abs() > 0.01]
        if len(big):
            print("大差异示例（前 5）：")
            print(big.head(5).to_string())


if __name__ == "__main__":
    main()
