"""a0a4_build — A0/A1-A4 数据集构建 (分片并行 + 断点续跑).

subcommands:
  sample                    确定性抽样宇宙 -> datasets/{a0,a14}_universe_candidates.parquet
  worker --pool {a0,a14} --shard i --nshards N [--limit M]
                            构建一个分片 (JSONL 增量落盘, 可续跑)
  merge --pool {a0,a14}     合并分片 -> datasets/{a0,a14}_universe.parquet + build_manifest
  status                    各分片进度

加速 (均不改冻结逻辑, 数值恒等):
  - a0a4_fast_proxy: 同 plan_date 共享 proxy 窗口, leave-one-out 走恒等代数;
    安装前真实探针对拍, 每 250 样本用纯原版路径复检一行 alpha 字段。
  - 分钟数据 ThreadPool 预取 (前视 2 个候选), 20ms/次并发拉取。
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
import a0a4_config as C  # noqa
import a0a4_fast_proxy as FP  # noqa

sys.path.insert(0, str(C.PREFLIGHT))

SHARDS = C.DATASETS / "shards"


# ---------------------------------------------------------------- utils
def _jdefault(o):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, float) and not np.isfinite(o):
        return None
    return str(o)


def _dumps(obj) -> str:
    return json.dumps(obj, ensure_ascii=False, default=_jdefault)


def _safe_round(x, nd=5):
    if x is None:
        return None
    try:
        xf = float(x)
    except (TypeError, ValueError):
        return None
    return round(xf, nd) if np.isfinite(xf) else None


# ---------------------------------------------------------------- sample
def cmd_sample(args):
    C.DATASETS.mkdir(parents=True, exist_ok=True)
    spec = dict(C.SAMPLE_SPEC)
    v1_all = pd.read_parquet(C.POOL_V1, columns=[
        "code", "plan_date", "effective_trade_date", "dip_gate"])
    v1_all = v1_all[v1_all["effective_trade_date"].notna()]
    v1_dipgate = dict(zip(zip(v1_all["code"].astype(str).str.zfill(6),
                              v1_all["plan_date"].astype(int)),
                          v1_all["dip_gate"].astype(bool)))

    for name, path in (("a0", C.POOL_V1), ("a14", C.POOL_V11)):
        t0 = time.time()
        df = pd.read_parquet(path)
        df = df[df["effective_trade_date"].notna()]
        df["_plan"] = df["plan_date"].astype(int)
        df["_eff"] = df["effective_trade_date"].astype(int)
        df = df[(df["_eff"] > df["_plan"])
                & (df["_eff"] >= spec["minute_start"])
                & (df["_plan"] >= spec["min_plan"])
                & (df["_plan"] <= spec["max_plan"])]
        rng = np.random.default_rng(spec["seed"])
        parts = []
        for p, g in df.groupby("_plan"):
            n = min(spec["cap_per_day"], len(g))
            parts.append(g.sample(n=n, random_state=int(rng.integers(0, 1_000_000))))
        s = pd.concat(parts).copy()
        s["sample_id"] = (s["code"].astype(str).str.zfill(6) + "|"
                          + s["_plan"].astype(str))
        s["is_bottom"] = s["bottom_gate"].astype(bool)
        s["dual_hit_score40"] = ((s["bottom_score"] >= 40)
                                 & (s["intraday_dip_score"] >= 40))
        if name == "a14":
            s["is_dip"] = s["dip_eligible_v11"].astype(bool)
            keys = list(zip(s["code"].astype(str).str.zfill(6), s["_plan"].astype(int)))
            s["v1_rejected_v11_accepted"] = [
                int(bool(elig) and not v1_dipgate.get(k, False))
                for elig, k in zip(s["dip_eligible_v11"].astype(bool), keys)]
        else:
            s["is_dip"] = s["dip_gate"].astype(bool)
            s["v1_rejected_v11_accepted"] = 0
        out = s.drop(columns=["_plan", "_eff"])
        f = C.DATASETS / f"{name}_universe_candidates.parquet"
        out.to_parquet(f, index=False)
        print(f"[sample] {name}: {len(out)} rows, {out['plan_date'].nunique()} days "
              f"-> {f} ({time.time()-t0:.1f}s)")
        print(f"[sample] {name} is_bottom={int(out['is_bottom'].sum())} "
              f"is_dip={int(out['is_dip'].sum())} "
              f"dual_gate={int((out['is_bottom'] & out['is_dip']).sum())} "
              f"dual_score40={int(out['dual_hit_score40'].sum())} "
              f"v1rej_v11acc={int(out['v1_rejected_v11_accepted'].sum())}")


# ---------------------------------------------------------------- worker
class _PrefetchMinute:
    """(code,date)->Future 缓存 + 前视预取; 不改变 load_minute 语义 (同源函数)."""

    def __init__(self, orig, cal, n_threads=6, horizon=60):
        self._orig = orig
        self._cal = np.asarray(sorted(cal), dtype=np.int64)
        self._h = horizon
        self._cache: dict = {}
        self._pool = ThreadPoolExecutor(max_workers=n_threads)

    def need_dates(self, d_date: int):
        i = int(np.searchsorted(self._cal, int(d_date), side="left"))
        past = self._cal[max(0, i - self._h):i]
        return [int(d_date)] + [int(x) for x in past]

    def prefetch(self, code, d_date):
        code = str(code).zfill(6)
        for d in self.need_dates(d_date):
            k = (code, d)
            if k not in self._cache:
                self._cache[k] = self._pool.submit(self._orig, code, d)

    def hard_clear(self):
        self._cache.clear()

    def __call__(self, code, date):
        code = str(code).zfill(6)
        k = (code, int(date))
        f = self._cache.get(k)
        if f is None:
            f = self._pool.submit(self._orig, code, int(date))
            self._cache[k] = f
        return f.result()


def _harness_extras(row, mc, daily_dict, C02):
    """harness 派生特征 (预注册公式见 a0a4_config.DERIVED_FORMULAS)。"""
    out = {}
    code, d, atr = row["symbol"], int(row["d_date"]), float(row["atr_ref"])
    m = mc(code, d)
    if m is not None and not m.empty and atr > 0:
        ts = m["ts"].to_numpy("int64")
        mod = ts % 1_000_000
        mins = (mod // 10000) * 60 + (mod // 100 % 100)
        low = m["low"].to_numpy(float)
        high = m["high"].to_numpy(float)
        o = float(m["open"].iloc[0])
        open_min = int(mins[0])
        for k in (5, 30):
            w = mins <= open_min + k
            depth = (o - float(low[w].min())) / atr if w.any() else None
            out[f"dip_depth_{k}m_atr"] = _safe_round(depth, 5)
            out[f"dip_speed_{k}m"] = _safe_round(
                (depth / k) if depth is not None else None, 6)
        out["dip_speed_15m"] = _safe_round(
            (row.get("open_dip_atr") or np.nan) / 15.0, 6)
        fill = row.get("actual_fill")
        fts = row.get("fill_ts")
        if fill is not None and fts:
            fill = float(fill)
            fmod = int(fts) % 1_000_000
            fmin = (fmod // 10000) * 60 + (fmod // 100 % 100)
            post = mins >= fmin
            w30 = post & (mins <= fmin + 30)
            out["mae_30m"] = _safe_round(
                float(low[w30].min()) / fill - 1 if w30.any() else None, 6)
            out["mfe_30m"] = _safe_round(
                float(high[w30].max()) / fill - 1 if w30.any() else None, 6)
            out["mae_day"] = _safe_round(
                float(low[post].min()) / fill - 1 if post.any() else None, 6)
            out["mfe_day"] = _safe_round(
                float(high[post].max()) / fill - 1 if post.any() else None, 6)
            nxt = C02.fast_daily(daily_dict, code, d, d + 30)
            nxt = nxt[nxt["date"] > d] if not nxt.empty else nxt
            if len(nxt) and out.get("mae_day") is not None:
                nl, nh = float(nxt["low"].iloc[0]), float(nxt["high"].iloc[0])
                dlow, dhigh = out["mae_day"], out["mfe_day"]
                out["mae_h1"] = _safe_round(min(fill * (1 + dlow), nl) / fill - 1, 6)
                out["mfe_h1"] = _safe_round(max(fill * (1 + dhigh), nh) / fill - 1, 6)
            out["failed_rebound"] = (int(out["mfe_30m"] < 0.01)
                                     if out.get("mfe_30m") is not None else None)
    try:
        zones = json.loads(row.get("zones_json") or "[]")
        out["n_zones"] = len(zones)
    except Exception:
        out["n_zones"] = None
    if row.get("zone_upper") is not None and row.get("zone_lower") is not None and atr > 0:
        out["zone_band_atr"] = _safe_round(
            (float(row["zone_upper"]) - float(row["zone_lower"])) / atr, 5)
    else:
        out["zone_band_atr"] = None
    src = row.get("primary_zone_source")
    out["zone_n_sources"] = (len(str(src).split("|")) if src else 0)
    return out


def cmd_worker(args):
    import coverage_02_full as C02  # noqa
    import latr_builder as LB  # noqa
    import trading_calendar as TC  # noqa
    from market_proxy_v1 import MarketProxyV1  # noqa
    from fast_infra import mk_chains  # noqa
    from data_source_provenance import get_universe  # noqa
    from green_watch_candidate_loader import CandidateRow  # noqa

    t0 = time.time()

    def log(*a):
        print(f"[w{args.shard}]", *a, flush=True)

    SHARDS.mkdir(parents=True, exist_ok=True)
    uni = pd.read_parquet(C.DATASETS / f"{args.pool}_universe_candidates.parquet")
    plans = sorted(uni["plan_date"].astype(int).unique())
    n = len(plans)
    lo_i = args.shard * n // args.nshards
    hi_i = (args.shard + 1) * n // args.nshards
    shard_plans = set(plans[lo_i:hi_i])
    sub = uni[uni["plan_date"].astype(int).isin(shard_plans)].copy()
    sub = sub.sort_values(["plan_date", "code"])       # date-major: proxy 窗口复用
    if args.limit:
        sub = sub.head(args.limit)
    log(f"pool={args.pool} shard={args.shard}/{args.nshards} "
        f"plans={min(shard_plans)}..{max(shard_plans)} candidates={len(sub)}")

    # ---- 上下文 ----
    daily_dict = C02.build_daily_dict()
    univ = get_universe()
    chains = mk_chains(univ)
    proxy = MarketProxyV1(
        weight="float", leave_one_out=True,
        universe_provider=lambda: univ,
        daily_provider=lambda c, s, e: C02.fast_daily(daily_dict, c, s, e),
        chains_provider=lambda cs: {c.zfill(6): chains.get(c.zfill(6)) for c in cs})
    LB.load_daily_window = lambda code, s, e: C02.fast_daily(daily_dict, code, s, e)
    cal = [int(x) for x in TC.load_calendar()]
    mc = _PrefetchMinute(LB.load_minute, cal)
    LB.load_minute = mc
    log(f"context ready: daily_dict={len(daily_dict)} chains={len(chains)} "
        f"({time.time()-t0:.1f}s)")

    # ---- fast proxy (等价对拍通过才安装) ----
    probes = []
    for _, r in sub.head(3).iterrows():
        d = int(r["effective_trade_date"])
        h1 = int(TC.next_trading_day(d))
        h3 = int(TC.next_trading_day(TC.next_trading_day(h1)))
        probes.append((int(r["plan_date"]), d, "open", h3, "close", str(r["code"]).zfill(6)))
        probes.append((int(r["plan_date"]), d, "open", h1, "close", str(r["code"]).zfill(6)))
    ok, rep = FP.install_fast_proxy(proxy, probes)
    log(f"fast_proxy installed={ok} equiv: n={rep['n_tests']} "
        f"max_diff={rep['max_abs_diff']:.2e} passed={rep['passed']}")
    if not ok:
        log("ABORT: fast proxy equivalence FAILED")
        sys.exit(2)

    base = SHARDS / f"{args.pool}_shard{args.shard}of{args.nshards}"
    jl, rj, ex, eq = (Path(str(base) + s) for s in (
        ".jsonl", ".rejects.jsonl", ".exc.jsonl", ".equiv.jsonl"))
    done = set()
    if jl.exists():
        with open(jl, encoding="utf-8") as f:
            for line in f:
                try:
                    done.add(json.loads(line)["sample_id"])
                except Exception:
                    continue
    log(f"resume: {len(done)} already done")

    recs = list(sub.iterrows())
    t_build = time.time()
    n_ok = n_rej = n_exc = n_eq = n_eq_fail = 0
    cur_plan = None
    for i, (_, r) in enumerate(recs):
        sid = r["sample_id"]
        code, plan, d = str(r["code"]).zfill(6), int(r["plan_date"]), int(r["effective_trade_date"])
        if plan != cur_plan:
            cur_plan = plan
            try:
                proxy._daily_cache.clear()
            except Exception:
                pass
        cand = CandidateRow(
            symbol=code, plan_date=plan, effective_trade_date=d,
            primary_setup=str(r["primary_setup"]),
            bottom_score=float(r["bottom_score"]),
            intraday_dip_score=float(r["intraday_dip_score"]),
            atr_ref=float(r["atr_ref"]),
            bottom_tier=str(r.get("bottom_tier", "") or ""),
            dip_tier=str(r.get("dip_tier", "") or ""),
            primary_tier=str(r.get("primary_tier", "") or ""),
            bottom_subtype=str(r.get("bottom_subtype", "") or ""),
            raw={k: (None if pd.isna(v) else v) for k, v in r.items()},
        )
        if sid in done:
            continue
        mc.prefetch(code, d)                       # 当前 + 前视 2
        if i + 1 < len(recs):
            mc.prefetch(str(recs[i + 1][1]["code"]).zfill(6),
                        int(recs[i + 1][1]["effective_trade_date"]))
        if i + 2 < len(recs):
            mc.prefetch(str(recs[i + 2][1]["code"]).zfill(6),
                        int(recs[i + 2][1]["effective_trade_date"]))
        try:
            cand.validate()
        except AssertionError:
            with open(rj, "a", encoding="utf-8") as f:
                f.write(_dumps({"sample_id": sid, "reason": "contract_assert"}) + "\n")
            n_rej += 1
            continue
        try:
            s = LB.build_one_sample(cand, chains, proxy)
            if s:
                s["sample_id"] = sid
                s.update(_harness_extras(s, mc, daily_dict, C02))
                # 等价抽检: 对最近真实调用重放原版 compute_proxy_return 对拍
                if n_ok > 0 and n_ok % 250 == 0:
                    try:
                        n_ch, dmax, okq = FP.recheck_equivalence(proxy, 4)
                        with open(eq, "a", encoding="utf-8") as f:
                            f.write(_dumps({"sample_id": sid, "n_calls": n_ch,
                                            "max_abs_diff": dmax,
                                            "pass": bool(okq)}) + "\n")
                        n_eq += 1
                        if not okq:
                            n_eq_fail += 1
                    except Exception as ee:  # noqa
                        n_eq_fail += 1
                        with open(eq, "a", encoding="utf-8") as f:
                            f.write(_dumps({"sample_id": sid, "error": str(ee)[:200],
                                            "pass": False}) + "\n")
                with open(jl, "a", encoding="utf-8") as f:
                    f.write(_dumps(s) + "\n")
                n_ok += 1
            else:
                with open(rj, "a", encoding="utf-8") as f:
                    f.write(_dumps({"sample_id": sid, "reason": "none_return"}) + "\n")
                n_rej += 1
        except Exception as e:  # noqa
            with open(ex, "a", encoding="utf-8") as f:
                f.write(_dumps({"sample_id": sid, "symbol": cand.symbol,
                                "plan_date": cand.plan_date,
                                "d_date": cand.effective_trade_date,
                                "exc_type": type(e).__name__, "exc_msg": str(e)[:300],
                                "traceback": traceback.format_exc()[-1500:]}) + "\n")
            n_exc += 1
            continue
        if (n_ok + n_rej + n_exc) % 25 == 0:
            log(f"built {n_ok} rej={n_rej} exc={n_exc} of {len(recs)} "
                f"({time.time()-t_build:.1f}s)")
        if (n_ok + n_rej + n_exc) % 100 == 0:
            mc.hard_clear()
    log(f"DONE ok={n_ok} rej={n_rej} exc={n_exc} equiv_checks={n_eq} "
        f"equiv_fail={n_eq_fail} total={time.time()-t0:.1f}s")


# ---------------------------------------------------------------- merge
def cmd_merge(args):
    import glob as _glob
    import trading_calendar as TC
    C.DATASETS.mkdir(parents=True, exist_ok=True)
    files = [f for f in sorted(_glob.glob(str(SHARDS / f"{args.pool}_shard*of*.jsonl")))
             if not f.endswith((".rejects.jsonl", ".exc.jsonl", ".equiv.jsonl"))]
    rows, rej, exc = [], [], []
    n_bad_lines = 0
    for fp in files:
        with open(fp, encoding="utf-8") as f:
            for line in f:
                try:
                    rows.append(json.loads(line))
                except Exception:
                    n_bad_lines += 1
                    continue
        rp = Path(fp).with_name(Path(fp).name.replace(".jsonl", ".rejects.jsonl"))
        ep = Path(fp).with_name(Path(fp).name.replace(".jsonl", ".exc.jsonl"))
        for pp, sink in ((rp, rej), (ep, exc)):
            if pp.exists():
                with open(pp, encoding="utf-8") as f:
                    sink.extend(json.loads(l) for l in f if l.strip())
    df = pd.DataFrame(rows)
    if df.empty:
        print(f"[merge] {args.pool}: NO ROWS")
        return
    df = df.drop_duplicates(subset=["sample_id"], keep="last")
    uni = pd.read_parquet(C.DATASETS / f"{args.pool}_universe_candidates.parquet")
    keep = ["sample_id"] + [c for c in (
        "dip_stage", "dip_eligible_v11", "is_bottom", "is_dip", "dual_hit_score40",
        "v1_rejected_v11_accepted", "candidate_contract_version",
        "source_data_snapshot_id", "intraday_dip_score") if c in uni.columns] + \
        [f for f, _ in C.A1_CORE if f not in ("setup_type", "dip_stage")]
    keep = [c for c in keep if c in uni.columns]
    uni_join = uni[keep]
    df = df.merge(uni_join, on="sample_id", how="left", validate="1:1")
    out = C.DATASETS / f"{args.pool}_universe.parquet"
    df.to_parquet(out, index=False)
    eq_files = _glob.glob(str(SHARDS / f"{args.pool}_shard*.equiv.jsonl"))
    n_eq = sum(1 for fp in eq_files for _ in open(fp, encoding="utf-8"))
    n_eq_fail = 0
    for fp in eq_files:
        for line in open(fp, encoding="utf-8"):
            try:
                if not json.loads(line).get("pass"):
                    n_eq_fail += 1
            except Exception:
                pass
    man = {
        "pool": args.pool,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "n_samples": len(df),
        "n_bad_jsonl_lines": n_bad_lines,
        "n_rejects": len(rej),
        "n_exceptions": len(exc),
        "reject_by_reason": pd.Series([r.get("reason") for r in rej])
            .value_counts().to_dict() if rej else {},
        "exception_by_type": pd.Series([e.get("exc_type") for e in exc])
            .value_counts().to_dict() if exc else {},
        "alpha_equiv_checks": n_eq, "alpha_equiv_fail": n_eq_fail,
        "sample_spec": C.SAMPLE_SPEC,
        "folds": C.FOLDS,
        "config_fingerprint": C.config_fingerprint(),
        "source_data_snapshot_id": TC.data_epoch_fingerprint(),
        "label_status_counts": df["label_status"].value_counts().to_dict()
        if "label_status" in df else {},
    }
    man["input_hashes"] = {
        "pool_parquet": C.sha256_file(C.POOL_V1 if args.pool == "a0" else C.POOL_V11),
        **C.frozen_hashes(),
    }
    man["hashes"] = man["input_hashes"]
    (C.DATASETS / f"{args.pool}_build_manifest.json").write_text(
        json.dumps(man, ensure_ascii=False, indent=2, default=_jdefault),
        encoding="utf-8")
    print(f"[merge] {args.pool}: {len(df)} rows -> {out}")
    print(f"[merge] rejects={len(rej)} exceptions={len(exc)} "
          f"equiv={n_eq} equiv_fail={n_eq_fail}")
    print(f"[merge] label_status={man['label_status_counts']}")


# ---------------------------------------------------------------- status
def cmd_status(args):
    import glob as _glob
    for pool in ("a0", "a14"):
        files = sorted(_glob.glob(str(SHARDS / f"{pool}_shard*.jsonl")))
        tot = 0
        for f in files:
            if f.endswith((".rejects.jsonl", ".exc.jsonl", ".equiv.jsonl")):
                continue
            n = sum(1 for _ in open(f, encoding="utf-8"))
            tot += n
            print(f"  {Path(f).name}: {n}")
        print(f"  [{pool}] total built = {tot}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("sample")
    w = sub.add_parser("worker")
    w.add_argument("--pool", required=True, choices=["a0", "a14"])
    w.add_argument("--shard", type=int, required=True)
    w.add_argument("--nshards", type=int, required=True)
    w.add_argument("--limit", type=int, default=0)
    m = sub.add_parser("merge")
    m.add_argument("--pool", required=True, choices=["a0", "a14"])
    sub.add_parser("status")
    args = ap.parse_args()
    {"sample": cmd_sample, "worker": cmd_worker,
     "merge": cmd_merge, "status": cmd_status}[args.cmd](args)


if __name__ == "__main__":
    main()
