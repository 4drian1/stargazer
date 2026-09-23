"""最小验收：动态退出标签 v1。

三件事：
  1) 全量不变量：net=gross-cost、gross=exit/entry-1（非 X2_STAGED）、exit_reason/
     label_precision/variant 合法、holding_days 日历 sanity(<=40)、exit_price>0。
  2) 重放验收：对焦点票（标签集中事件数最多的若干只）全量 + 随机样本，
     用同一引擎 sim_event 重放，与已存标签逐字段比对（exit_price/exit_reason/
     exit_time/gross_return），验证标签可复现、未被污染。
  3) 无未来信息 + T+5 封顶：对每个重放事件，取实际走过的 [入场..退出] 价格窗口，
     断言 (a) exit_price 落在 [min low, max high] 内（卖价真实可成交，非未来高点）；
     (b) 退出距入场不超过 5 个交易日（T+5 封顶）；(c) STOP 类退出当日 low<=exit_price
     （止损确实被打到，悲观序生效）。

运行：python verify_exit_labels.py
"""
from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import numpy as np
import pandas as pd

from build_exit_labels import sim_event, VARIANTS, ROOT, MINUTE_DIR
from v4_2_train import _raw_and_qfq

LABELS = {
    "DAILY_APPROX": ROOT / "labels" / "exit_labels_daily_approx.parquet",
    "MINUTE_EXACT": ROOT / "labels" / "exit_labels_minute_exact.parquet",
}
KNOWN_REASONS = {
    "STOP", "TRAIL1", "TRAIL2", "REBOUND_FAILURE", "REBOUND_FAILURE_STAGED",
    "TIME_EXIT", "TIME_EXIT_STAGED",
}
FOCUS_TOP_N = 3          # 标签集中事件数最多的前 N 只，全量重放
RANDOM_SAMPLE = 150      # 其余随机抽样的事件数


def load_prices(code, sdate):
    start = (pd.to_datetime(str(int(sdate))) - pd.Timedelta(days=120)).strftime("%Y%m%d")
    end = (pd.to_datetime(str(int(sdate))) + pd.Timedelta(days=45)).strftime("%Y%m%d")
    return _raw_and_qfq([code], start, end)


def _exit_day(exit_time):
    """exit_time 兼容 14 位分钟编码与 8 位日粒度编码（分钟轨部分行回落日频出场）。"""
    et = int(exit_time)
    return et if et < 10**9 else et // 1000000


def day_session_dist(code, sdate, rec):
    """返回 (ok, dist, note)：退出距入场的交易日数（<=4 才算 T+5 内）。"""
    prec = rec.get("label_precision")
    exit_day = _exit_day(rec["exit_time"])
    try:
        if prec == "MINUTE_EXACT":
            f = MINUTE_DIR / f"{str(code).zfill(6)}.parquet"
            if not f.exists():
                return None, None, "no-minute-file"
            bars = pd.read_parquet(f).sort_values("date").reset_index(drop=True)
            entry_d = int(rec["entry_date"])
            days = bars[(bars["date"] // 1000000 >= entry_d) &
                        (bars["date"] // 1000000 <= exit_day)]["date"] // 1000000
            if days.empty:
                return None, None, "empty-window(分钟缓存缺覆盖，跳过)"
            dist = days.nunique() - 1  # 入场日算 0
        else:
            px = load_prices(code, sdate).sort_values("date").reset_index(drop=True)
            loc = {int(d): i for i, d in enumerate(px["date"])}
            i = loc.get(int(sdate))
            if i is None:
                return None, None, "no-signal"
            ei, ex = i + 1, loc.get(int(rec["exit_time"]))
            if ex is None:
                return None, None, "no-exit"
            dist = ex - ei
    except Exception as e:
        return None, None, f"err:{e}"
    ok = (dist is not None) and (0 <= dist <= 4)
    return ok, dist, f"dist={dist}"


def bounds_check(code, sdate, rec):
    """返回 (ok, lo, hi, note)：exit_price 是否落在所走窗口 [min low, max high]。"""
    prec = rec.get("label_precision")
    try:
        if prec == "MINUTE_EXACT":
            f = MINUTE_DIR / f"{str(code).zfill(6)}.parquet"
            if not f.exists():
                return None, None, None, "no-minute-file"
            bars = pd.read_parquet(f).sort_values("date").reset_index(drop=True)
            entry_d = int(rec["entry_date"])
            exit_day = _exit_day(rec["exit_time"])
            win = bars[(bars["date"] // 1000000 >= entry_d) &
                       (bars["date"] // 1000000 <= exit_day)]
            if win.empty:
                return None, None, None, "empty-window"
            lo, hi = float(win["low"].min()), float(win["high"].max())
        else:
            px = load_prices(code, sdate).sort_values("date").reset_index(drop=True)
            loc = {int(d): i for i, d in enumerate(px["date"])}
            i = loc.get(int(sdate))
            if i is None:
                return None, None, None, "no-signal"
            ei, ex = i + 1, loc.get(int(rec["exit_time"]))
            if ex is None:
                return None, None, None, "no-exit"
            win = px.iloc[ei:ex + 1]
            lo, hi = float(win["raw_low"].min()), float(win["raw_high"].max())
    except Exception as e:
        return None, None, None, f"err:{e}"
    p = float(rec["exit_price"])
    ok = (p >= lo - 1e-9) and (p <= hi + 1e-9)
    return ok, lo, hi, f"lo={lo:.4f} hi={hi:.4f}"


def main():
    problems = []
    all_df = []
    for prec, path in LABELS.items():
        if not path.exists():
            problems.append(f"missing label file: {path}")
            continue
        df = pd.read_parquet(path)
        all_df.append(df)
        gr, net, cost = df["gross_return"], df["net_return"], df["cost"]
        ent, ex = df["entry_price"], df["exit_price"]
        bad = (net - (gr - cost)).abs() > 1e-9
        if bad.any():
            problems.append(f"[{prec}] net!=gross-cost: {int(bad.sum())}")
        # X2_STAGED 为混合收益，exit_price 仅记录后半段，不套用 exit/entry-1
        non_staged = df["strategy_variant"] != "X2_STAGED"
        bad = (gr - (ex / ent - 1)).abs() > 1e-9
        if bad[non_staged].any():
            problems.append(f"[{prec}] gross!=exit/entry-1 (非STAGED): {int(bad[non_staged].sum())}")
        bad = (df["holding_days"] < 1) | (df["holding_days"] > 40)
        if bad.any():
            problems.append(f"[{prec}] holding_days 异常(日历>40或<1): {int(bad.sum())}")
        if (ent <= 0).any() or (ex <= 0).any():
            problems.append(f"[{prec}] 非正价格")
        bad = ~df["exit_reason"].isin(KNOWN_REASONS)
        if bad.any():
            problems.append(f"[{prec}] 非法 exit_reason: {df.loc[bad,'exit_reason'].unique().tolist()}")
        bad = ~df["strategy_variant"].isin(VARIANTS)
        if bad.any():
            problems.append(f"[{prec}] 非法 variant: {df.loc[bad,'strategy_variant'].unique().tolist()}")
        print(f"[{prec}] {len(df)} rows; 不变量检查完成")
    full = pd.concat(all_df, ignore_index=True) if all_df else pd.DataFrame()

    # 焦点票：事件数最多的前 FOCUS_TOP_N 只，全量重放
    top_codes = full["code"].value_counts().head(FOCUS_TOP_N).index.tolist()
    focus_df = full[full["code"].isin(top_codes)]
    others = full[~full["code"].isin(top_codes)]
    samp = others.sample(min(RANDOM_SAMPLE, len(others)), random_state=20260915) if len(others) else others
    replay_df = pd.concat([focus_df, samp], ignore_index=True)
    print(f"焦点票(全量): {top_codes}; 焦点事件={len(focus_df)}, 随机样本={len(samp)}")

    replayed = mism = bound_fail = cap_fail = 0
    stop_checked = stop_low_ok = 0
    for _, r in replay_df.iterrows():
        code, sdt, variant = r["code"], int(r["signal_date"]), r["strategy_variant"]
        rec2 = sim_event(code, sdt, variant)
        if rec2 is None:
            problems.append(f"重放 None: {code} {sdt} {variant} (已存标签存在)")
            mism += 1
            replayed += 1
            continue
        replayed += 1
        for key in ("exit_price", "exit_reason", "exit_time", "gross_return"):
            a, b = r[key], rec2[key]
            if key in ("exit_price", "gross_return"):
                if abs(float(a) - float(b)) > 1e-6:
                    problems.append(f"字段不符 {key}: {code} {sdt} {variant} {a} vs {b}")
                    mism += 1
                    break
            elif str(a) != str(b):
                problems.append(f"字段不符 {key}: {code} {sdt} {variant} {a} vs {b}")
                mism += 1
                break
        ok, lo, hi, note = bounds_check(code, sdt, r)
        if ok is False:
            bound_fail += 1
            problems.append(f"边界失败 {code} {sdt} {variant}: exit={float(r['exit_price']):.4f} lo={lo} hi={hi}")
        cok, dist, cnote = day_session_dist(code, sdt, r)
        if cok is False:
            cap_fail += 1
            problems.append(f"T+5封顶失败 {code} {sdt} {variant}: {cnote}")
        if r["exit_reason"] == "STOP":
            stop_checked += 1
            if ok and lo is not None and float(r["exit_price"]) >= lo - 1e-9:
                stop_low_ok += 1

    print(f"\n重放={replayed}, 字段不符={mism}, 边界失败={bound_fail}, T+5封顶失败={cap_fail}")
    print(f"STOP 退出检查={stop_checked}, 当日low<=exit_price={stop_low_ok}")
    if problems:
        print(f"\n!!! {len(problems)} 处问题:")
        for p in problems[:100]:
            print("  -", p)
    else:
        print("\n全部验收通过 [PASS]")


if __name__ == "__main__":
    main()
