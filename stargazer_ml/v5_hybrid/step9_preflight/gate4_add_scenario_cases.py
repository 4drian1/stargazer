"""gate4_add_scenario_cases — 把 ④ 场景扫描命中的真实案例注入 Gate4 样本集.

背景 (用户 2026-09-18 裁定):
  随机抽样的 Gate4 样本里 3 个预注册场景命中数为 0; 定向扫描 (trace_04b_scenario_scan)
  在 2000 个候选里找到了 DIP_TO_BREAKDOWN=8 / SECOND_TEST_HIGHER_LOW=1 / LATE_SESSION_TOUCH=4。
  -> 结论是「抽样没抽到」而不是「数据里没有」。

  正确收尾: 把这些真实案例当作强制分层, 走完整的 build_one_sample (含 label/proxy/zone)
  并入 Gate4 样本集, 让 TRACE_SCENARIO_COVERAGE_PASS 通过真实样本 PRESENT 而不是靠证书兜底。

输出:
  gate4_samples.csv      合并后的样本集 (原文件备份为 gate4_samples.base.csv)
  scenario_cases_built.csv  本次注入案例的构建结果

运行: python gate4_add_scenario_cases.py
"""
from __future__ import annotations
import json
import shutil
import sys
import time
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent))

import coverage_02_full as C02  # noqa
import gate4_harness as G  # noqa
from fast_infra import build_context, mk_chains  # noqa
from latr_builder import build_one_sample  # noqa

_HERE = Path(__file__).resolve().parent
NEED = ["code", "plan_date", "effective_trade_date", "primary_setup", "bottom_score",
        "intraday_dip_score", "atr_ref", "bottom_tier", "dip_tier", "primary_tier",
        "bottom_subtype", "dip_stage"]


def main():
    t0 = time.time()

    def log(*a):
        print(" ".join(str(x) for x in a), flush=True)

    cert = json.loads((_HERE / "scenario_cert.json").read_text(encoding="utf-8"))
    wanted = {}
    for scen, v in cert.items():
        for e in v.get("examples", []):
            wanted[(str(e["symbol"]).zfill(6), int(e["plan_date"]))] = (scen, int(e["d_date"]))
    log(f"[add] 场景案例 {len(wanted)} 个: {sorted(wanted.keys())}")

    avail = set(pq.read_schema(C02.V11_POOL).names)
    cols = [c for c in NEED if c in avail]
    df = pd.read_parquet(C02.V11_POOL, columns=cols)
    df["_plan"] = df["plan_date"].astype(int)
    df["_code"] = df["code"].astype(str).str.zfill(6)
    sel = df[[(c, p) in wanted for c, p in zip(df["_code"], df["_plan"])]]
    log(f"[add] pool 命中 {len(sel)} 行 (期望 {len(wanted)})")
    miss = set(wanted) - {(c, p) for c, p in zip(sel["_code"], sel["_plan"])}
    if miss:
        log(f"[add] WARN 池中未找到: {sorted(miss)}")
    cands = []
    for _, r in sel.iterrows():
        try:
            cands.append(C02._build_row(r))
        except Exception as e:  # noqa
            log(f"[add] _build_row failed {r.get('_code')} {r.get('_plan')}: {e}")
    log(f"[add] CandidateRow 构建 {len(cands)}")

    _ctx, _c, _p = build_context(cap_per_day=1, min_plan=20250101)
    ctx = _ctx
    chains = mk_chains(sorted({c.symbol for c in cands}))
    proxy = ctx.make_proxy(leave_one_out=True)
    log(f"[add] chains={len(chains)}; 开始完整构建 ...")

    rows, rejects, excs = [], [], []
    import traceback
    for cand in cands:
        try:
            s = build_one_sample(cand, chains, proxy, reject_log=rejects)
            if s:
                rows.append(s)
        except Exception as e:  # noqa
            excs.append({"symbol": cand.symbol, "plan_date": cand.plan_date,
                         "d_date": cand.effective_trade_date, "exc_type": type(e).__name__,
                         "exc_msg": str(e)[:300], "traceback": traceback.format_exc()[-2000:]})
    log(f"[add] built={len(rows)} rejects={len(rejects)} exceptions={len(excs)} ({time.time()-t0:.0f}s)")
    if rejects:
        log(f"[add] rejects: {json.dumps(rejects, ensure_ascii=False)[:600]}")
    if excs:
        log(f"[add] exceptions: {json.dumps([{k: e[k] for k in ('symbol','plan_date','d_date','exc_type','exc_msg')} for e in excs], ensure_ascii=False)}")
        (_HERE / "gate4_scenario_cases_exceptions.json").write_text(
            json.dumps(excs, ensure_ascii=False, indent=2), encoding="utf-8")

    new = pd.DataFrame(rows)
    if new.empty:
        log("[add] 无新增样本; 退出")
        return
    new.to_csv(_HERE / "scenario_cases_built.csv", index=False, encoding="utf-8")
    present = G._scenario_present(new)
    log(f"[add] 本批场景命中: {json.dumps(present)}")

    base = _HERE / "gate4_samples.csv"
    old = pd.DataFrame()
    if base.exists():
        bak = _HERE / "gate4_samples.base.csv"
        if not bak.exists():
            shutil.copy2(base, bak)
        old = pd.read_csv(base)
        log(f"[add] 现有样本 {len(old)} 行")
        merged = pd.concat([old, new], ignore_index=True)
    else:
        merged = new
    key = ["symbol", "plan_date", "effective_trade_date"]
    merged["symbol"] = merged["symbol"].astype(str).str.zfill(6)
    merged = merged.drop_duplicates(subset=key, keep="first").reset_index(drop=True)
    merged.to_csv(base, index=False, encoding="utf-8")
    log(f"[add] gate4_samples.csv 合并后 {len(merged)} 行 (原 {len(old)} 行)")
    log("[add] 下一步: python gate4_run_real.py --from-samples")


if __name__ == "__main__":
    main()
