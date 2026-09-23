"""merge_02_recover — 将 14 个受影响日期的复测结果合并回 ② 全量 coverage 产物.

背景: coverage_02_full.py 旧版用 g[0] 推导 D/H1/H3 锚点 (单点脆弱), 使 11 个中段
日期整日 alpha 假性为 0 (见 verify_02_ref_single_point.py)。修复后仅复测这 14 日期,
本脚本把修正行替换回全量 CSV, 并用与 coverage_02_full 完全相同的聚合/告警逻辑重算
coverage_02_summary.json。

原产物先备份为 *.pre_fix.bak。
"""
from __future__ import annotations
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve().parent
MAIN_CSV = _HERE / "coverage_02_by_date.csv"
MAIN_JSON = _HERE / "coverage_02_summary.json"
REC_CSV = _HERE / "coverage_02_recover" / "coverage_02_by_date.csv"

COV_COLS = ["s0_return_h3_cov", "s0_market_alpha_h3_cov", "timing_return_h1_cov",
            "timing_market_alpha_h1_cov", "alpha_label_available_cov",
            "proxy_weight_coverage", "proxy_constituent_count"]


def build_summary(df: pd.DataFrame) -> dict:
    overall = {
        "n_plan_dates": len(df),
        "date_range": [int(df["plan_date"].min()), int(df["plan_date"].max())] if len(df) else None,
        "total_sampled": int(df["n_sampled"].sum()),
        "total_built": int(df["n_built"].sum()),
        "s0_return_h3_cov_mean": round(float(df["s0_return_h3_cov"].mean()), 4),
        "s0_return_h3_cov_min": round(float(df["s0_return_h3_cov"].min()), 4),
        "s0_market_alpha_h3_cov_mean": round(float(df["s0_market_alpha_h3_cov"].mean()), 4),
        "s0_market_alpha_h3_cov_min": round(float(df["s0_market_alpha_h3_cov"].min()), 4),
        "timing_return_h1_cov_mean": round(float(df["timing_return_h1_cov"].mean()), 4),
        "timing_return_h1_cov_min": round(float(df["timing_return_h1_cov"].min()), 4),
        "timing_market_alpha_h1_cov_mean": round(float(df["timing_market_alpha_h1_cov"].mean()), 4),
        "timing_market_alpha_h1_cov_min": round(float(df["timing_market_alpha_h1_cov"].min()), 4),
        "alpha_label_available_cov_mean": round(float(df["alpha_label_available_cov"].mean()), 4),
        "alpha_label_available_cov_min": round(float(df["alpha_label_available_cov"].min()), 4),
        "proxy_weight_coverage_mean": round(float(df["proxy_weight_coverage"].mean()), 4),
        "proxy_weight_coverage_min": round(float(df["proxy_weight_coverage"].min()), 4),
        "proxy_constituent_count_mean": int(df["proxy_constituent_count"].mean()),
        "proxy_constituent_count_min": int(df["proxy_constituent_count"].min()),
        "proxy_constituent_count_max": int(df["proxy_constituent_count"].max()),
    }
    median_constit = float(df["proxy_constituent_count"].median()) if len(df) else 0
    alerts = []
    for _, r in df.iterrows():
        reasons = []
        if r["proxy_constituent_count"] < 0.6 * median_constit:
            reasons.append(f"proxy_constituent_count={int(r['proxy_constituent_count'])} "
                           f"< 0.6*median({int(median_constit)})")
        if r["proxy_weight_coverage"] < 0.5:
            reasons.append(f"proxy_weight_coverage={r['proxy_weight_coverage']} < 0.5")
        if r["s0_return_h3_cov"] < 0.9:
            reasons.append(f"s0_return_h3_cov={r['s0_return_h3_cov']} < 0.9")
        if r["timing_return_h1_cov"] < 0.9:
            reasons.append(f"timing_return_h1_cov={r['timing_return_h1_cov']} < 0.9")
        if r["alpha_label_available_cov"] < 0.9:
            reasons.append(f"alpha_label_available_cov={r['alpha_label_available_cov']} < 0.9")
        if reasons:
            alerts.append({"plan_date": int(r["plan_date"]), "reasons": reasons})
    return {"overall": overall, "alerts": alerts, "n_alerts": len(alerts)}


def main():
    main_df = pd.read_csv(MAIN_CSV)
    rec_df = pd.read_csv(REC_CSV)
    assert set(rec_df["plan_date"]).issubset(set(main_df["plan_date"])), "recover dates not in main"

    # 备份
    shutil.copyfile(MAIN_CSV, MAIN_CSV.with_suffix(".pre_fix.bak.csv"))
    shutil.copyfile(MAIN_JSON, MAIN_JSON.with_suffix(".pre_fix.bak.json"))

    fixed = main_df.drop(columns=["__src"], errors="ignore")
    fixed = fixed.set_index("plan_date")
    rec = rec_df.set_index("plan_date")
    common = [c for c in fixed.columns if c in rec.columns]
    changed = sorted(set(rec.index))
    for d in changed:
        for c in common:
            fixed.loc[d, c] = rec.loc[d, c]
    out = fixed.reset_index().sort_values("plan_date").reset_index(drop=True)

    # 验证: 14 行确实被替换
    print(f"[merge] replaced {len(changed)} dates: {changed}")
    for d in changed:
        r = out[out["plan_date"] == d].iloc[0]
        print(f"  {d}: s0_cov={r['s0_return_h3_cov']} alpha_s0_cov={r['s0_market_alpha_h3_cov']} "
              f"alpha_avail={r['alpha_label_available_cov']}")

    out.to_csv(MAIN_CSV, index=False, encoding="utf-8")
    summ = build_summary(out)
    MAIN_JSON.write_text(json.dumps(summ, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n[merge] OVERALL:", json.dumps(summ["overall"], ensure_ascii=False))
    print(f"[merge] n_alerts={summ['n_alerts']}")
    for a in summ["alerts"]:
        print(f"  ALERT {a['plan_date']}: {a['reasons']}")
    print(f"[merge] wrote {MAIN_CSV} / {MAIN_JSON}")


if __name__ == "__main__":
    main()
