"""trace_04_scenarios — ④ Zone + Minute State 真实 trace (8 类场景, v11 真实候选).

与 ② 同款加速策略 (仅运行时 monkeypatch, 不改动 latr_builder 文件):
  - 预建 code->daily 内存字典 -> 替换 latr_builder.load_daily_window (避免每次重扫全 parquet);
  - 预建全 universe 复权链 -> 注入 MarketProxyV1 的 fast chains_provider;
  - 分钟 (D 日 + 60 历史日 volume_baseline) 加 (code,date) 缓存;
  - 真实 v11 候选 (生产池), 不抽"好看的成功案例"。

8 类场景:
  1 BOTTOM hold-zone        2 DIP touch->reclaim      3 DIP touch->breakdown
  4 PRE_PULLBACK 未触 zone   5 SECOND_TEST higher low  6 corporate-action 附近
  7 inferred halt / 缺 minute 8 早盘 / 午后

异常重点 (真实数据下应不出现):
  zone_distance_atr 大面积 NaN / zone_valid=0 比例异常 / state 大量停在 WATCH /
  transition 时间错序 / AVWAP_PROXY_DAILY 被误当正式 AVWAP。

运行: python trace_04_scenarios.py --budget 200
"""
from __future__ import annotations
import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import coverage_02_full as C02  # noqa  (复用 fast_daily / load_v11_candidates)
import latr_builder as LB  # noqa
from latr_builder import build_one_sample  # noqa
from market_proxy_v1 import MarketProxyV1  # noqa
from adjustment_chain import build_chains  # noqa
from data_source_provenance import get_universe, get_adjustment_http, get_minute_http  # noqa

_HERE = Path(__file__).resolve().parent
SNAPSHOT_MINUTE = 150000


def _mk_chains(codes):
    by = defaultdict(list)
    for code in codes:
        try:
            for d, rec in get_adjustment_http(code):
                by[code.zfill(6)].append((str(d), rec))
        except Exception:
            continue
    return build_chains(by)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=int, default=200, help="构建样本预算 (受 stockdb 分钟成本限制)")
    ap.add_argument("--cap-per-day", type=int, default=25)
    ap.add_argument("--min-plan", type=int, default=20250102)
    ap.add_argument("--max-plan", type=int, default=None)
    ap.add_argument("--seed", type=int, default=20260918)
    ap.add_argument("--out-dir", default=str(_HERE))
    args = ap.parse_args()
    out_dir = Path(args.out_dir)
    t0 = time.time()

    def log(*a):
        print(" ".join(str(x) for x in a), flush=True)

    # ---- fast infra ----
    daily_dict = C02.build_daily_dict()
    log(f"[04] daily dict ready ({time.time()-t0:.1f}s)")

    groups, plan_dates = C02.load_v11_candidates(args.cap_per_day, None,
                                                 min_plan=args.min_plan,
                                                 max_plan=args.max_plan)
    cands = [c for p in plan_dates for c in groups[p]]
    log(f"[04] v11 candidates={len(cands)} across {len(plan_dates)} plan_dates")

    # 全 universe 复权链 (proxy 用) + 抽样 symbol 链
    univ = get_universe()
    chains_full = _mk_chains(univ)
    all_codes = sorted({c.symbol for c in cands})
    chains = _mk_chains(all_codes)
    log(f"[04] chains ready (universe={len(chains_full)}, sampled={len(chains)}) ({time.time()-t0:.1f}s)")

    def chains_provider(codes):
        return {c.zfill(6): chains_full.get(c.zfill(6)) for c in codes}

    proxy = MarketProxyV1(weight="float", leave_one_out=True,
                          universe_provider=lambda: univ,
                          daily_provider=lambda code, s, e: C02.fast_daily(daily_dict, code, s, e),
                          chains_provider=chains_provider)

    # ---- monkeypatch latr_builder 取数 (运行时) ----
    LB.load_daily_window = lambda code, start, end: C02.fast_daily(daily_dict, code, start, end)
    _orig_minute = LB.load_minute
    _mc = {}

    def _cached_minute(code, date):
        k = (str(code).zfill(6), int(date))
        if k not in _mc:
            _mc[k] = _orig_minute(code, date)
        return _mc[k]
    LB.load_minute = _cached_minute

    # ---- 预筛: corporate-action 附近 候选优先纳入 ----
    rng = np.random.default_rng(args.seed)
    ca_codes_cache = {}

    def _near_ca(cand):
        ch = chains.get(cand.symbol.zfill(6))
        if ch is None:
            return False
        d = int(cand.effective_trade_date)
        for ev in getattr(ch, "events", []):
            try:
                ed = int(ev[0])
            except Exception:
                continue
            if abs(ed - d) <= 5:
                return True
        return False

    ca_pool = [c for c in cands if _near_ca(c)]
    log(f"[04] corporate-action-near candidates={len(ca_pool)}")

    # ---- 预算内构建 (CA 优先 + 随机补足) ----
    chosen = list(ca_pool[: max(20, args.budget // 4)])
    _chosen_ids = set(id(x) for x in chosen)
    remaining = [c for c in cands if id(c) not in _chosen_ids]
    rng.shuffle(remaining)
    for c in remaining:
        if len(chosen) >= args.budget:
            break
        chosen.append(c)

    rows = []
    none_reasons = defaultdict(int)
    exceptions = []          # 用户 2026-09-18 #2b: 异常必须逐条归因, 禁止未知桶
    t_build = time.time()
    for i, cand in enumerate(chosen):
        try:
            s = build_one_sample(cand, chains, proxy)
        except Exception as e:
            import traceback as _tb
            frame = None
            for _f in reversed(_tb.extract_tb(e.__traceback__)):
                if _f.filename.endswith("latr_builder.py"):
                    frame = {"file": "latr_builder.py", "lineno": _f.lineno,
                             "source_line": _f.line}
                    break
            none_reasons[f"exc:{type(e).__name__}"] += 1
            exceptions.append({
                "exc_type": type(e).__name__, "exc_msg": str(e)[:300],
                "symbol": str(getattr(cand, "symbol", None)),
                "plan_date": getattr(cand, "plan_date", None),
                "d_date": getattr(cand, "effective_trade_date", None),
                "setup": str(getattr(cand, "primary_setup", None)),
                "atr_ref": getattr(cand, "atr_ref", None),
                "failing_frame": frame,
                "traceback": _tb.format_exc()[-1500:],
            })
            continue
        if s is None:
            # 区分: 缺分钟 vs 未成交 vs 缺日线
            try:
                m = _cached_minute(cand.symbol, cand.effective_trade_date)
                if m is None or m.empty:
                    none_reasons["missing_minute"] += 1
                else:
                    none_reasons["no_fill_or_other"] += 1
            except Exception:
                none_reasons["no_fill_or_other"] += 1
            continue
        rows.append(s)
        if (i + 1) % 25 == 0:
            log(f"[04] built {i+1}/{len(chosen)} -> {len(rows)} samples ({time.time()-t_build:.1f}s)")
        if (i + 1) % 50 == 0:
            try:
                proxy._daily_cache.clear()
            except Exception:
                pass

    log(f"[04] built total={len(rows)}  drop_reasons={dict(none_reasons)}  ({time.time()-t_build:.1f}s)")
    if not rows:
        log("[04] NO SAMPLES")
        if exceptions:
            (out_dir / "trace_04_exceptions.json").write_text(
                json.dumps(exceptions, ensure_ascii=False, indent=2), encoding="utf-8")
        return
    df = pd.DataFrame(rows)

    # ---- 场景分类 ----
    def pick(mask, n):
        sub = df[mask]
        return sub if len(sub) <= n else sub.sample(n=n, random_state=1)

    # transition 首个 ZONE_TOUCH 的 HHMMSS
    def _transitions(js):
        try:
            return json.loads(js) if isinstance(js, str) else (js or [])
        except Exception:
            return []

    def _touch_hhmmss(js):
        for t in _transitions(js):
            if t.get("state") == "ZONE_TOUCH" and t.get("state_enter_ts"):
                return int(t["state_enter_ts"]) % 1_000_000
        return None

    df["_touch_hhmmss"] = df["state_transitions_json"].apply(_touch_hhmmss)

    per = max(10, args.budget // 8)
    scen = {}
    scen["1_BOTTOM_hold_zone"] = pick(
        (df["primary_setup"] == "BOTTOM_SETUP") & (df["zone_distance_atr"] <= 0.20)
        & (df["final_state"] != "BREAKDOWN"), per)
    scen["2_DIP_touch_reclaim"] = pick(
        (df["primary_setup"] == "INTRADAY_DIP_SETUP")
        & (df["final_state"].isin(["RECLAIM", "MODEL_SCORE_REQUEST"])), per)
    scen["3_DIP_touch_breakdown"] = pick(
        (df["primary_setup"] == "INTRADAY_DIP_SETUP") & (df["final_state"] == "BREAKDOWN"), per)
    scen["4_PRE_PULLBACK_no_touch"] = pick(
        (df["primary_setup"] == "INTRADAY_DIP_SETUP")
        & (df["final_state"].isin(["WATCH", "ZONE_APPROACH"])), per)
    scen["5_SECOND_TEST_higher_low"] = pick(
        (df["second_test_flag"] == 1) & (df["second_test_higher_low"] == 1), per)
    scen["6_corporate_action_near"] = pick(df["symbol"].isin([c.symbol for c in ca_pool]), per)
    scen["7_halt_or_missing_minute"] = df[df["final_state"] == "INVALIDATED"].head(per)
    scen["8_early_session_touch"] = pick(df["_touch_hhmmss"].notna() & (df["_touch_hhmmss"] < 110000), per)
    scen["8b_late_session_touch"] = pick(df["_touch_hhmmss"].notna() & (df["_touch_hhmmss"] >= 140000), per)

    # ---- 全局异常检查 ----
    zdist_nan = int(df["zone_distance_atr"].isna().sum())
    zvalid0 = int((df["zone_valid"] == 0).sum())
    watch_rate = float((df["final_state"] == "WATCH").mean())
    state_counts = df["final_state"].value_counts().to_dict()

    disorder = 0
    for js in df["state_transitions_json"]:
        snap = None
        prev_exit = -1
        for t in _transitions(js):
            ent = t.get("state_enter_ts")
            ext = t.get("state_exit_ts")
            if ent is None or ext is None:
                continue
            if ent < prev_exit or ent > ext:
                disorder += 1
                break
            prev_exit = ext

    # AVWAP 误用: primary_zone_source 不得为纯 "AVWAP"
    avwap_misuse = int(df["primary_zone_source"].fillna("").astype(str).str.fullmatch("AVWAP").sum())
    # zones_json 里 AVWAP 级别的 zone_origin 必须 = PROXY
    avwap_origin_bad = 0
    avwap_zone_seen = 0
    for js in df["zones_json"]:
        try:
            zones = json.loads(js) if isinstance(js, str) else (js or [])
        except Exception:
            zones = []
        for z in zones:
            if "AVWAP" in str(z.get("zone_source", "")):
                avwap_zone_seen += 1
                if z.get("zone_origin") != "PROXY":
                    avwap_origin_bad += 1

    report = {
        "n_samples": len(rows),
        "drop_reasons": dict(none_reasons),
        "exceptions": exceptions,   # 逐条归因 (type/symbol/date/setup/frame/stack)
        "global_anomaly_checks": {
            "zone_distance_atr_nan": zdist_nan,
            "zone_distance_atr_nan_rate": round(zdist_nan / len(df), 4),
            "zone_valid0_count": zvalid0,
            "zone_valid0_rate": round(zvalid0 / len(df), 4),
            "final_state_distribution": state_counts,
            "watch_rate": round(watch_rate, 4),
            "transition_time_disorder_count": disorder,
            "avwap_misused_as_formal_count": avwap_misuse,
            "avwap_zone_seen": avwap_zone_seen,
            "avwap_proxy_origin_not_PROXY_count": avwap_origin_bad,
        },
        "scenarios": {},
    }
    for name, sub in scen.items():
        if sub is None or len(sub) == 0:
            report["scenarios"][name] = {"n": 0, "note": "no matching real sample"}
            continue
        report["scenarios"][name] = {
            "n": int(len(sub)),
            "final_state_distribution": sub["final_state"].value_counts().to_dict(),
            "zone_valid0_rate": round(float((sub["zone_valid"] == 0).mean()), 4),
            "zone_distance_atr_nan_rate": round(float(sub["zone_distance_atr"].isna().mean()), 4),
            "primary_zone_source_distribution": sub["primary_zone_source"].fillna("NA").astype(str).value_counts().to_dict(),
            "primary_zone_source_sample": sub["primary_zone_source"].dropna().astype(str).head(3).tolist(),
        }

    (out_dir / "trace_04_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = ["===== ④ ZONE + MINUTE STATE TRACE (REAL) =====",
             f"n_samples={len(rows)}  drop_reasons={dict(none_reasons)}",
             "\n--- GLOBAL ANOMALY CHECKS ---",
             f"zone_distance_atr NaN: {zdist_nan} ({report['global_anomaly_checks']['zone_distance_atr_nan_rate']})",
             f"zone_valid=0: {zvalid0} ({report['global_anomaly_checks']['zone_valid0_rate']})",
             f"WATCH rate: {watch_rate}",
             f"transition time disorder: {disorder}",
             f"AVWAP misused as formal (source=='AVWAP'): {avwap_misuse}",
             f"AVWAP zones seen={avwap_zone_seen}, origin!=PROXY: {avwap_origin_bad}",
             f"final_state distribution: {state_counts}",
             "\n--- SCENARIOS ---"]
    for name, info in report["scenarios"].items():
        lines.append(f"[{name}] n={info.get('n')}")
        if "final_state_distribution" in info:
            lines.append(f"    states={info['final_state_distribution']}")
            lines.append(f"    zone_valid0_rate={info['zone_valid0_rate']} zdist_nan_rate={info['zone_distance_atr_nan_rate']}")
            lines.append(f"    zone_source={info['primary_zone_source_distribution']}")
    txt = "\n".join(lines)
    (out_dir / "trace_04_report.txt").write_text(txt, encoding="utf-8")
    log(txt)
    log(f"[04] DONE total={time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
