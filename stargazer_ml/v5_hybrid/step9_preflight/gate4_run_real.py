"""gate4_run_real — ⑤ Formal Gate4 harness on REAL v11 samples.

对真实构建的样本 DataFrame 执行 gate4_harness.run_gate4 (11 项检查)。
SAMPLE_INTEGRITY_PASS 与 SAMPLE_SELECTION_LEAK_PASS 作为一票否决项。

本版新增 (用户 2026-09-18 裁定):
  - 工程性拒绝账本 (不可 replay 的候选, 含显式原因) -> gate4_attrition.json
  - 异常归因 (symbol/date/setup/exc_type/stack) -> gate4_attrition.json
    (禁止把 exc:TypeError 作为未知桶带进 Final Freeze)
  - 全量样本落盘 gate4_samples.csv (可复算, 不必重跑取数)
  - 消费 trace_04b_scenario_scan 生成的 scenario_cert.json (场景覆盖证书)

verdict:
  PASS             全部检查通过
  CONDITIONAL_PASS 仅 TRACE_SCENARIO_COVERAGE_PASS 未定 (等待场景证书/定向扫描)
  FAIL             其它检查失败

运行: python gate4_run_real.py --budget 300
"""
from __future__ import annotations
import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from fast_infra import build_context, mk_chains  # noqa
from latr_builder import build_one_sample  # noqa
from gate4_harness import run_gate4  # noqa

_HERE = Path(__file__).resolve().parent
ONLY_SOFT = {"TRACE_SCENARIO_COVERAGE_PASS"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=int, default=300, help="构建样本预算")
    ap.add_argument("--cap-per-day", type=int, default=25)
    ap.add_argument("--min-plan", type=int, default=20250102)
    ap.add_argument("--max-plan", type=int, default=None)
    ap.add_argument("--out-dir", default=str(_HERE))
    ap.add_argument("--from-samples", action="store_true",
                    help="跳过取数, 直接对已落盘的 gate4_samples.csv 重跑检查 (秒级, 用于场景证书就绪后复评)")
    args = ap.parse_args()
    out_dir = Path(args.out_dir)
    t0 = time.time()

    def log(*a):
        print(" ".join(str(x) for x in a), flush=True)

    if args.from_samples:
        csv = out_dir / "gate4_samples.csv"
        df = pd.read_csv(csv)
        log(f"[gate4] --from-samples: {csv} rows={len(df)}")
        _evaluate(df, out_dir, len(df), None, log)
        return

    ctx, cands, plan_dates = build_context(args.cap_per_day, args.min_plan, args.max_plan)
    log(f"[gate4] candidates={len(cands)} across {len(plan_dates)} plan_dates ({time.time()-t0:.1f}s)")

    if len(cands) > args.budget:
        step = max(1, len(cands) // args.budget)
        cands = cands[::step][: args.budget]
    chains = mk_chains(sorted({c.symbol for c in cands}))
    proxy = ctx.make_proxy(leave_one_out=True)
    log(f"[gate4] chains={len(chains)}, building {len(cands)} samples ...")

    rows = []
    reject_log = []
    exc_log = []
    import traceback
    for i, cand in enumerate(cands):
        try:
            s = build_one_sample(cand, chains, proxy, reject_log=reject_log)
            if s:
                rows.append(s)
        except Exception as e:
            exc_log.append({
                "symbol": getattr(cand, "symbol", None),
                "plan_date": getattr(cand, "plan_date", None),
                "d_date": getattr(cand, "effective_trade_date", None),
                "primary_setup": getattr(cand, "primary_setup", None),
                "atr_ref": getattr(cand, "atr_ref", None),
                "exc_type": type(e).__name__, "exc_msg": str(e)[:300],
                "traceback": traceback.format_exc()[-2500:],
            })
        if (i + 1) % 25 == 0:
            log(f"[gate4] built {i+1}/{len(cands)} -> {len(rows)} ({time.time()-t0:.1f}s)")
        if (i + 1) % 100 == 0:
            try:
                proxy._daily_cache.clear()
            except Exception:
                pass

    log(f"[gate4] total samples: {len(rows)} ({time.time()-t0:.1f}s) "
        f"rejects={len(reject_log)} exceptions={len(exc_log)}")
    if not rows:
        log("[gate4] NO SAMPLES — cannot run Gate4")
        sys.exit(3)

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "gate4_samples.csv", index=False, encoding="utf-8")
    attrition = {
        "n_attempted": len(cands), "n_built": len(rows),
        "n_rejected_engineering": len(reject_log), "n_exceptions": len(exc_log),
        "reject_by_reason": pd.Series([r["reason"] for r in reject_log]).value_counts().to_dict()
        if reject_log else {},
        "exception_by_type": pd.Series([e["exc_type"] for e in exc_log]).value_counts().to_dict()
        if exc_log else {},
        "rejects": reject_log[:200], "exceptions": exc_log[:50],
        "note": "exc:* 必须逐条归因 (类型/symbol/date/setup/stack); 不得作为未知桶进 Freeze",
    }
    (out_dir / "gate4_attrition.json").write_text(
        json.dumps(attrition, ensure_ascii=False, indent=2), encoding="utf-8")
    _evaluate(df, out_dir, len(cands), attrition, log)


def _evaluate(df, out_dir, n_attempted, attrition, log):
    attrition = attrition or {}
    # 场景证书 (由 trace_04b_scenario_scan.py 生成; 缺失则场景检查标 UNVERIFIED)
    cert_path = out_dir / "scenario_cert.json"
    cert = json.loads(cert_path.read_text(encoding="utf-8")) if cert_path.exists() else None
    if cert is None:
        log("[gate4] WARN: scenario_cert.json 缺失 -> TRACE_SCENARIO_COVERAGE_PASS 将 UNVERIFIED")
    else:
        log(f"[gate4] scenario_cert.json loaded: "
            f"{ {k: v.get('status') for k, v in cert.items()} }")

    report = run_gate4(df, scenario_cert=cert)

    failing = [k for k, v in report.items() if k != "OVERALL" and not v["pass"]]
    if not failing:
        verdict = "PASS"
    elif set(failing) <= ONLY_SOFT:
        verdict = "CONDITIONAL_PASS"
    else:
        verdict = "FAIL"

    log("\n===== GATE4 REPORT (REAL) =====")
    for name, res in report.items():
        if name == "OVERALL":
            continue
        log(f"  {name:30s} : {'PASS' if res['pass'] else 'FAIL'}")
        if not res["pass"]:
            log(f"      detail: {json.dumps(res['detail'], ensure_ascii=False)[:600]}")
    integ = report["SAMPLE_INTEGRITY_PASS"]["detail"]
    leak = report["SAMPLE_SELECTION_LEAK_PASS"]["detail"]
    log(f"\n  research_tier (NOT a filter): {json.dumps(integ['research_tier']['distribution'], ensure_ascii=False)}")
    log(f"  eligibility_counts: {json.dumps(integ['eligibility_counts'], ensure_ascii=False)}")
    log(f"  selection-leak: {json.dumps(leak, ensure_ascii=False)}")
    scen = report["TRACE_SCENARIO_COVERAGE_PASS"]["detail"]
    log(f"  scenario status: {json.dumps({k: v['status'] for k, v in scen['scenarios'].items()}, ensure_ascii=False)}")
    log(f"\n  OVERALL: {'PASS' if report['OVERALL'] else 'FAIL'}")
    log(f"\n  VERDICT: {verdict}")

    out = out_dir / "gate4_real_report.json"
    out.write_text(json.dumps({"verdict": verdict, "n_samples": len(df),
                               "failing_checks": failing,
                               "attrition": {k: v for k, v in attrition.items()
                                             if k not in ("rejects", "exceptions")},
                               "report": report}, ensure_ascii=False, indent=2),
                   encoding="utf-8")
    log(f"[gate4] wrote {out}")
    if verdict == "FAIL":
        sys.exit(2)


if __name__ == "__main__":
    main()
