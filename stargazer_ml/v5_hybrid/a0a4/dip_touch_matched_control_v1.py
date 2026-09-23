"""DIP_TOUCH_EVENT_TIME_MATCHED_CONTROL_V1 (用户 2026-09-22 预注册规格).

问题定义 (非"Touch 因果效应检验" — Touch 由价格路径决定, date+stage+minute 匹配
不足以消除全部选择偏差; 本实验只回答):
  在同一时点、同类 DIP 候选中, 发生 ZONE_TOUCH 的股票, 之后是否比尚未 Touch 的更好?
  仅据此决定 Touch 有无资格升级为 Trigger.

Risk set (a14 抽样宇宙全部 is_dip 样本):
  Treatment : INTRADAY_DIP_SETUP=1 且 first real ZONE_TOUCH at S
              (v2 精确语义: in_zone=(lo<=upper)&(hi>=lower) 首根, = arm_t0)
  Control   : 同 effective_trade_date + 同 dip_stage, 在 S 时刻 尚未 Touch
              (first_touch_ts > S 或全天未触 — 条件等价于 [开盘,S] 无 Touch, S 时刻可知),
              仍 candidate-valid (is_dip), 未 structural breakdown (未触前结构上不可能),
              可交易 (S 前存在 <=10 分钟内活跃 bar)
  禁止用"全天 never-touch"筛 control (S 时刻不可知的未来信息).

匹配:
  Primary    : trade_date + dip_stage + minute_of_day (control 用同一 S 作 synthetic signal)
  Sensitivity: + |zone_distance@S前一根bar 差| <= 0.25 ATR
               + |realized_dip@S前一根bar 差|  <= 0.25 ATR
               + 同 float_mv bucket (P 日 float_mv 五分位)
  匹配变量只用 S 之前状态; 绝不用 Touch 后产生的 recovery/reclaim 信息.

执行 (两臂完全一致):
  signal_ts = S ; fill = first legal minute > S (execution_engine.execute,
  09:31 规则 / 涨停停牌不可成交); exec_status != FILLED → 双臂对称排除 (② 规则, 不伪造).
  exit = H1 (D 的下一交易日) close raw, 复用 a14 冻结口径
         exit_price = actual_fill_a14 * (1 + timing_return_h1)
  proxy_ret 复用冻结 MarketProxyV1 (D open→H1 close, LOO, plan_date 权重, 与 fill 分钟无关):
         proxy_ret = timing_return_h1 − timing_market_alpha_h1
  alpha = stock_ret(fill→H1) − proxy_ret ; 另计 raw ret / MFE / MAE / loss.

输出: incremental (T − matched C) alpha + bootstrap CI95 (按 plan_date 聚类) + win rate,
      拆 year / fold / dip_stage / pre-post 20260701; BOTH(BOTTOM∩DIP) 仅描述不裁决.

预注册裁决标准:
  Touch−Control ≈ 0 且 CI95 含 0 且多 fold 不稳定
      → DIP_TOUCH_EVENT_NOT_INCREMENTAL → 不再注册 TIMING-TRIGGER-v2.1
  增量明显为正 且 CI/多数 fold 支持 且 pre/post 无无法解释反转
      → DIP_TOUCH_EVENT_VALIDATED → 才值得构建 DIP causal Timing Dataset
"""
from __future__ import annotations
import json, math, pickle, sys, time
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
import a0a4_config as C  # noqa
sys.path.insert(0, str(_HERE.parent / "step9_preflight"))
import trading_calendar as TC  # noqa
from data_source_provenance import get_minute_http, get_adjustment_http, get_daily_parquet  # noqa
from adjustment_chain import build_chains  # noqa
from execution_engine import execute  # noqa
import zone_state_engine_v2 as Z2  # noqa  (仅 sanity 校验)
from zone_state_engine import Zone  # noqa  (仅 sanity 校验)

OUT = C.RUN / "dip_touch_matched_control_v1"
OUT.mkdir(parents=True, exist_ok=True)

CALIPER_ZD = 0.25     # ATR
CALIPER_RD = 0.25     # ATR
LIVE_WINDOW = 600     # S 前 10 分钟内有 bar 才算活跃 (HHMMSS 近似)
N_BOOT = 2000
BREAKDOWN_BARS = 3
STRUCT_BREAK_DEPTH_ATR = 1.0


# ---------------- v2 精确复刻: 全天 touch / breakdown 首次时刻 ----------------
def scan_day(ts_arr, hi, lo, cl, lower, upper, atr):
    """复刻 zone_state_engine_v2 的 touch/below/break_low 语义 (无 score break)."""
    touched = False
    touch_ts = None
    below_bars = 0
    broke = False
    break_low = math.inf
    breakdown_ts = None
    for i in range(len(ts_arr)):
        h_, l_ = hi[i], lo[i]
        in_zone = (l_ <= upper) and (h_ >= lower)
        below = l_ < lower
        if in_zone and not touched:
            touched = True
            touch_ts = int(ts_arr[i])
        if touched:
            if below:
                below_bars += 1
                break_low = min(break_low, l_)
                if breakdown_ts is None and (below_bars >= BREAKDOWN_BARS
                                             or (lower - break_low) >= STRUCT_BREAK_DEPTH_ATR * atr):
                    breakdown_ts = int(ts_arr[i])
            else:
                below_bars = 0
                if broke:
                    broke = False
                    break_low = math.inf
    return touch_ts, breakdown_ts


def pre_state(ts_arr, cl, lo, open_first, S, mid, atr):
    """S 之前最后一根完成 bar 的 zone distance 与开盘以来 realized dip (全部 S 前, causal)."""
    mask = ts_arr < S
    if not mask.any():
        return np.nan, np.nan
    i = int(np.max(np.nonzero(mask)[0]))
    zd = (cl[i] - mid) / atr if atr > 0 else np.nan
    rd = (open_first - lo[:i + 1].min()) / atr if atr > 0 else np.nan
    return zd, rd


def _fold_of(plan_date: int) -> str:
    for f in C.FOLDS:
        if f["test_lo"] <= plan_date <= f["test_hi"]:
            return f"f{f['fold']}"
    return "pre_fold"


def main():
    t0 = time.time()
    a = pd.read_parquet(C.DATASETS / "a14_universe.parquet")
    d = a[a["is_dip"].astype(bool)].copy()
    d["code"] = d["sample_id"].str.split("|").str[0]
    d = d.reset_index(drop=True)
    print(f"[A] dip samples n={len(d)} codes={d['code'].nunique()}")

    # ---- 复权链 (cache) ----
    adj_pk = OUT / "adj_chains.pkl"
    if adj_pk.exists():
        by_sym = pickle.load(open(adj_pk, "rb"))
    else:
        by_sym = {}
        for i, code in enumerate(sorted(d["code"].unique())):
            by_sym[code] = [(str(dt), rec) for dt, rec in get_adjustment_http(code)]
            if (i + 1) % 500 == 0:
                print(f"  adj {i+1}/{d['code'].nunique()} {time.time()-t0:.0f}s")
        pickle.dump(by_sym, open(adj_pk, "wb"))
    chains = build_chains(by_sym)
    print(f"[A] chains {len(chains)} {time.time()-t0:.0f}s")

    # ---- daily lookup (H1 low/high + P float_mv) ----
    daily = get_daily_parquet()
    daily = daily[daily["code"].isin(set(d["code"]))]
    dlow = {(r.code, int(r.date)): float(r.low) for r in daily.itertuples()}
    dhigh = {(r.code, int(r.date)): float(r.high) for r in daily.itertuples()}
    dmv = {(r.code, int(r.date)): float(r.float_mv) for r in daily.itertuples()
           if hasattr(r, "float_mv") and not pd.isna(r.float_mv)}
    del daily
    TC.load_calendar()
    print(f"[A] daily lookup {time.time()-t0:.0f}s")

    # ---- 分钟 fetch (cache) + 全天 scan ----
    min_pk = OUT / "minute_cache.parquet"
    if min_pk.exists():
        mc = pd.read_parquet(min_pk)
        minute = {k: v.sort_values("ts").reset_index(drop=True)
                  for k, v in mc.groupby("sample_id")}
        print(f"[A] minute cache {len(minute)} samples")
    else:
        minute = {}
        rows = []
        for i, r in enumerate(d.itertuples()):
            m = get_minute_http(r.code, str(int(r.effective_trade_date)))
            if m is None or m.empty:
                continue
            m = m.rename(columns={"date": "ts"}) if "ts" not in m.columns else m
            for c_ in ("ts", "open", "high", "low", "close", "volume"):
                if c_ in m:
                    m[c_] = pd.to_numeric(m[c_], errors="coerce")
            m = m.dropna(subset=["ts", "open", "high", "low", "close"]).sort_values("ts")
            m["ts"] = m["ts"].astype(np.int64)
            m = m[(m["ts"] // 1000000) == int(r.effective_trade_date)]
            if m.empty:
                continue
            m = m.reset_index(drop=True)
            minute[r.sample_id] = m
            mm = m.copy(); mm["sample_id"] = r.sample_id
            rows.append(mm)
            if (i + 1) % 500 == 0:
                print(f"  minute {i+1}/{len(d)} cached={len(minute)} {time.time()-t0:.0f}s")
        pd.concat(rows, ignore_index=True).to_parquet(min_pk, index=False)
        print(f"[A] minute fetched {len(minute)} samples {time.time()-t0:.0f}s")

    # ---- per-sample state ----
    states = {}
    attrition = defaultdict(int)
    for r in d.itertuples():
        sid = r.sample_id
        chain = chains.get(r.code)
        if chain is None:
            attrition["NO_CHAIN"] += 1
            continue
        m = minute.get(sid)
        if m is None or m.empty:
            attrition["NO_MINUTE"] += 1
            continue
        # ---- primary zone 重建 (与 latr select_primary_zone 完全一致: (-zone_strength, |zone_distance_atr|)) ----
        # 已验证: 300/300 样本上, 该 zone + 本扫描 精确复现 a14 存储的 time_since_zone_touch.
        # 注意: a14 顶层 zone_lower/upper/mid 是池(build_pool)窄支撑位, 不是引擎 zone — v0 误用已作废.
        try:
            zs = json.loads(r.zones_json) if isinstance(r.zones_json, str) else r.zones_json
        except Exception:
            zs = []
        if not zs:
            attrition["NO_ZONES"] += 1
            continue
        prim = sorted(zs, key=lambda z: (-int(z.get("zone_strength", 0)),
                                          abs(float(z.get("zone_distance_atr", 1e9)))))[0]
        scale = (chain.factor_at(int(r.effective_trade_date)) or 1.0) / \
                (chain.factor_at(int(r.plan_date)) or 1.0)
        ts_arr = m["ts"].to_numpy(np.int64)
        hi = m["high"].to_numpy(float) * scale
        lo = m["low"].to_numpy(float) * scale
        cl = m["close"].to_numpy(float) * scale
        op = float(m["open"].iloc[0]) * scale
        lower, upper = float(prim["zone_lower"]), float(prim["zone_upper"])
        mid = float(prim["zone_mid"])
        atr = float(r.atr_ref)
        if not (np.isfinite(lower) and np.isfinite(upper) and np.isfinite(mid) and atr > 0):
            attrition["BAD_ZONE"] += 1
            continue
        tt, bd = scan_day(ts_arr, hi, lo, cl, lower, upper, atr)
        # 冻结口径复用: exit(H1 close raw) 与 proxy_ret
        exit_px = float(r.actual_fill) * (1.0 + float(r.timing_return_h1))
        tr, al = float(r.timing_return_h1), float(r.timing_market_alpha_h1)
        proxy = (tr - al) if (np.isfinite(tr) and np.isfinite(al)) else np.nan
        h1 = TC.next_trading_day(int(r.effective_trade_date))
        h1 = int(h1) if h1 is not None else None
        states[sid] = dict(
            code=r.code, plan_date=int(r.plan_date), D=int(r.effective_trade_date),
            dip_stage=r.dip_stage, is_bottom=bool(r.is_bottom),
            touch_ts=tt, breakdown_ts=bd, scale=scale, atr=atr,
            lower=lower, upper=upper, mid=mid, open_adj=op,
            ts=ts_arr, hi=hi, lo=lo, cl=cl,
            exit_px=exit_px, proxy=proxy, h1=h1,
            h1_low=dlow.get((r.code, h1)) if h1 else None,
            h1_high=dhigh.get((r.code, h1)) if h1 else None,
            fmv=dmv.get((r.code, int(r.plan_date))),
        )
    print(f"[A] states {len(states)} attrition={dict(attrition)} {time.time()-t0:.0f}s")

    # touch rate by stage
    sdf = pd.DataFrame([{"stage": s["dip_stage"], "touched": s["touch_ts"] is not None}
                        for s in states.values()])
    print("[A] touch rate by stage:",
          sdf.groupby("stage")["touched"].agg(["mean", "size"]).round(3).to_dict("index"))

    # ---- sanity: 与 run_state_engine_v2 的 arm_t0 对齐 (30 样本) ----
    n_chk = mism = 0
    for sid, s in states.items():
        if n_chk >= 30:
            break
        if s["touch_ts"] is None:
            continue
        n_chk += 1
        m = minute[sid]
        z = Zone(zone_id="Z", zone_source="X", zone_lower=s["lower"], zone_upper=s["upper"],
                 zone_mid=s["mid"], zone_timestamp=s["plan_date"], zone_width_atr=0.0,
                 zone_strength=1, zone_source_count=1, zone_distance_atr=0.0)
        out = Z2.run_state_engine_v2(z, m, s["scale"], "INTRADAY_DIP_SETUP", s["atr"],
                                     int(f"{s['D']}150000"))
        if out["arm_t0"] != s["touch_ts"]:
            mism += 1
            print(f"  MISMATCH {sid}: mine={s['touch_ts']} engine={out['arm_t0']}")
    print(f"[A] sanity arm_t0 vs engine: checked={n_chk} mismatch={mism}")
    assert mism == 0, "touch 复刻与 v2 引擎不一致, 中止"

    # ---- float_mv bucket (五分位) ----
    fmv_vals = [s["fmv"] for s in states.values() if s["fmv"] is not None and s["fmv"] > 0]
    qs = np.quantile(fmv_vals, [0.2, 0.4, 0.6, 0.8]) if fmv_vals else []
    for s in states.values():
        v = s["fmv"]
        s["fmv_bucket"] = int(np.searchsorted(qs, v)) if (v is not None and v > 0 and len(qs)) else -1

    # ---- Phase B: strata ----
    by_day = defaultdict(list)
    for sid, s in states.items():
        by_day[s["D"]].append(sid)

    records = []
    strat_info = []
    sid_n = 0
    exec_fail_c = set()
    for day in sorted(by_day):
        pool = by_day[day]
        treated = [x for x in pool if states[x]["touch_ts"] is not None]
        for tid in treated:
            T = states[tid]
            S = T["touch_ts"]
            exT = execute(S, minute[tid])
            if exT["exec_status"] != "FILLED":
                attrition["T_EXEC_NOT_FILLED"] += 1
                continue
            if T["breakdown_ts"] is not None and T["breakdown_ts"] <= exT["fill_ts"]:
                attrition["T_BREAKDOWN_BEFORE_FILL"] += 1
                continue
            zdT, rdT = pre_state(T["ts"], T["cl"], T["lo"], T["open_adj"], S, T["mid"], T["atr"])
            ctrls = []
            for cid in pool:
                if cid == tid:
                    continue
                Cx = states[cid]
                if Cx["dip_stage"] != T["dip_stage"]:
                    continue
                if Cx["touch_ts"] is not None and Cx["touch_ts"] <= S:
                    continue  # S 时刻已 Touch (或同刻) -> 非 control
                # 活跃性: S 前最后一根 bar 距 S <= 10 分钟
                mask = Cx["ts"] <= S
                if not mask.any():
                    attrition["C_NOT_LIVE"] += 1
                    continue
                lastb = int(Cx["ts"][np.max(np.nonzero(mask)[0])])
                if lastb < S - LIVE_WINDOW * 100:
                    attrition["C_STALE"] += 1
                    continue
                if (cid, S) in exec_fail_c:
                    continue
                exC = execute(S, minute[cid])
                if exC["exec_status"] != "FILLED":
                    exec_fail_c.add((cid, S))
                    continue
                zdC, rdC = pre_state(Cx["ts"], Cx["cl"], Cx["lo"], Cx["open_adj"], S,
                                     Cx["mid"], Cx["atr"])
                ctrls.append((cid, exC, zdC, rdC))
            if not ctrls:
                attrition["T_NO_CONTROL"] += 1
                continue
            sid_n += 1
            strat_info.append(dict(stratum_id=sid_n, plan_date=T["plan_date"], D=day,
                                   dip_stage=T["dip_stage"], S=S, treat=tid,
                                   n_ctrl=len(ctrls), is_both=T["is_bottom"]))

            def _rec(sample_id, st, ex, zd, rd, arm, sens_ok):
                fill = ex["actual_fill"]
                ret = st["exit_px"] / fill - 1.0
                alpha = ret - st["proxy"] if np.isfinite(st["proxy"]) else np.nan
                mraw = minute[sample_id]
                post = mraw[mraw["ts"] >= ex["fill_ts"]]
                lo_d = float(post["low"].min()) if not post.empty else None
                hi_d = float(post["high"].max()) if not post.empty else None
                mae = (min(x for x in [lo_d, st["h1_low"]] if x is not None) / fill - 1.0) \
                    if (lo_d is not None or st["h1_low"] is not None) else np.nan
                mfe = (max(x for x in [hi_d, st["h1_high"]] if x is not None) / fill - 1.0) \
                    if (hi_d is not None or st["h1_high"] is not None) else np.nan
                return dict(stratum_id=sid_n, sample_id=sample_id, arm=arm,
                            plan_date=st["plan_date"], D=day, dip_stage=st["dip_stage"],
                            S=S, zd=zd, rd=rd, fmv_bucket=st["fmv_bucket"],
                            sens_ok=sens_ok, ret=ret, alpha=alpha, mfe=mfe, mae=mae,
                            loss=int(ret < 0), is_both=st["is_bottom"],
                            fold=_fold_of(st["plan_date"]),
                            prepost="POST" if st["plan_date"] >= 20260701 else "PRE")

            records.append(_rec(tid, T, exT, zdT, rdT, "T", True))
            for cid, exC, zdC, rdC in ctrls:
                sens_ok = (np.isfinite(zdT) and np.isfinite(zdC) and np.isfinite(rdT)
                           and np.isfinite(rdC)
                           and abs(zdT - zdC) <= CALIPER_ZD and abs(rdT - rdC) <= CALIPER_RD
                           and T["fmv_bucket"] >= 0
                           and states[cid]["fmv_bucket"] == T["fmv_bucket"])
                records.append(_rec(cid, states[cid], exC, zdC, rdC, "C", sens_ok))

    R = pd.DataFrame(records)
    R.to_parquet(OUT / "records.parquet", index=False)
    pd.DataFrame(strat_info).to_parquet(OUT / "strata.parquet", index=False)
    print(f"[B] strata={sid_n} records={len(R)} attrition={dict(attrition)} {time.time()-t0:.0f}s")

    # ---- analysis ----
    def _agg(df, tag, sens=False):
        if sens:
            df = df[df["sens_ok"]]
        t = df[df["arm"] == "T"]
        c = df[df["arm"] == "C"]
        tm = t.groupby("stratum_id")["alpha"].mean().rename("ta")
        trm = t.groupby("stratum_id")["ret"].mean().rename("tr")
        cm = c.groupby("stratum_id")["alpha"].mean().rename("ca")
        crm = c.groupby("stratum_id")["ret"].mean().rename("cr")
        n_ct = c.groupby("stratum_id").size().rename("n_ctrl")
        pd_ = t.groupby("stratum_id")["plan_date"].first().rename("plan_date")
        strat = pd.concat([tm, trm, cm, crm, n_ct, pd_], axis=1).dropna(subset=["ta", "ca"])
        inc_a = strat["ta"] - strat["ca"]
        inc_r = strat["tr"] - strat["cr"]
        out = {"tag": tag, "sens": sens, "n_strata": int(len(strat)),
               "n_treat": int(len(t)), "n_ctrl_obs": int(len(c)),
               "treat_alpha": _r(t["alpha"].mean()), "ctrl_alpha": _r(c["alpha"].mean()),
               "treat_ret": _r(t["ret"].mean()), "ctrl_ret": _r(c["ret"].mean()),
               "inc_alpha": _r(inc_a.mean()), "inc_ret": _r(inc_r.mean()),
               "treat_mfe": _r(t["mfe"].mean()), "treat_mae": _r(t["mae"].mean()),
               "ctrl_mfe": _r(c["mfe"].mean()), "ctrl_mae": _r(c["mae"].mean()),
               "treat_loss": _r(t["loss"].mean()), "ctrl_loss": _r(c["loss"].mean())}
        # win rates
        out["win_stratum"] = _r((strat["ta"] > strat["ca"]).mean())
        merged = t[["stratum_id", "alpha"]].merge(
            c[["stratum_id", "alpha"]], on="stratum_id", suffixes=("_t", "_c"))
        out["win_pairwise"] = _r((merged["alpha_t"] > merged["alpha_c"]).mean())
        # bootstrap CI (cluster by plan_date)
        if len(strat) >= 5:
            rng = np.random.default_rng(42)
            dates = strat["plan_date"].to_numpy()
            by = defaultdict(np.ndarray)
            for dt in np.unique(dates):
                by[dt] = inc_a[dates == dt].to_numpy()
            uniq = np.array(sorted(by.keys()))
            boots = []
            for _ in range(N_BOOT):
                pick = rng.choice(uniq, size=len(uniq), replace=True)
                vals = np.concatenate([by[x] for x in pick])
                boots.append(float(vals.mean()))
            lo_, hi_ = np.percentile(boots, [2.5, 97.5])
            out["inc_alpha_ci95"] = [_r(lo_), _r(hi_)]
            out["boot_p_two_sided"] = _r(2 * min(np.mean(np.array(boots) <= 0),
                                                 np.mean(np.array(boots) >= 0)))
        return out, strat

    def _r(x, nd=5):
        return None if x is None or not np.isfinite(x) else round(float(x), nd)

    results = {}
    results["primary"], _ = _agg(R, "PRIMARY")
    results["sensitivity"], _ = _agg(R, "SENSITIVITY", sens=True)
    for dim in ("fold", "dip_stage", "prepost"):
        for val, sub in R.groupby(dim):
            key = f"{dim}={val}"
            results[key], _ = _agg(sub, key)
            if dim == "prepost":
                results[key + "_sens"], _ = _agg(sub, key + "_sens", sens=True)
    # BOTH 仅描述
    both = R[R["is_both"]]
    if len(both) >= 30:
        results["BOTH_descriptive"], _ = _agg(both, "BOTH_DESCRIPTIVE")
    else:
        results["BOTH_descriptive"] = {"n_records": int(len(both)), "skipped": "n<30"}

    results["attrition"] = dict(attrition)
    results["pre_registered_verdict_rule"] = {
        "NOT_INCREMENTAL": "inc_alpha≈0 & CI95 含 0 & 多 fold 不稳定 -> 不注册 TIMING-TRIGGER-v2.1",
        "VALIDATED": "增量明显为正 & CI/多数 fold 支持 & pre/post 无无法解释反转 -> 才构建 DIP causal Dataset",
    }
    with open(OUT / "report.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)

    p = results["primary"]
    print("\n===== DIP_TOUCH_EVENT_TIME_MATCHED_CONTROL_V1 (PRIMARY) =====")
    print(f"strata={p['n_strata']} treat={p['n_treat']} ctrl_obs={p['n_ctrl_obs']}")
    print(f"treat_alpha={p['treat_alpha']} ctrl_alpha={p['ctrl_alpha']} "
          f"inc_alpha={p['inc_alpha']} CI95={p.get('inc_alpha_ci95')} "
          f"p={p.get('boot_p_two_sided')}")
    print(f"inc_ret={p['inc_ret']} win_stratum={p['win_stratum']} win_pairwise={p['win_pairwise']}")
    print(f"MFE/MAE: T={p['treat_mfe']}/{p['treat_mae']} C={p['ctrl_mfe']}/{p['ctrl_mae']} "
          f"loss: T={p['treat_loss']} C={p['ctrl_loss']}")
    for k in ("fold=f1", "fold=f2", "fold=f3", "fold=f4", "dip_stage=PRE_PULLBACK",
              "dip_stage=PULLBACK_ACTIVE", "dip_stage=NEAR_ZONE",
              "prepost=PRE", "prepost=POST"):
        if k in results:
            v = results[k]
            print(f"  {k}: n_strata={v.get('n_strata')} inc_alpha={v.get('inc_alpha')} "
                  f"CI95={v.get('inc_alpha_ci95')}")
    s = results["sensitivity"]
    print(f"SENSITIVITY(caliper): n_strata={s.get('n_strata')} inc_alpha={s.get('inc_alpha')} "
          f"CI95={s.get('inc_alpha_ci95')}")
    print(f"attrition={dict(attrition)}")
    print(f"[done {time.time()-t0:.0f}s] -> {OUT}")


if __name__ == "__main__":
    main()
