"""a0a4_report — A0-A4 第一轮汇总: Q1-Q4 裁决 + 强制切片 + 账本。

产出: REPORT.md + report.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
import a0a4_config as C  # noqa

TIERS = ("a0", "a1", "a2", "a3", "a4")


def _load(t, f):
    p = C.TIER_DIRS[t] / f
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def _fmt(x, pct=False, nd=4):
    if x is None:
        return "n/a"
    if pct:
        return f"{x*100:+.2f}%"
    return f"{x:+.{nd}f}"


def main():
    out = {"generated_at": pd.Timestamp.now().isoformat()}
    checks = json.loads((C.RUN / "checks.json").read_text(encoding="utf-8"))
    m = {t: _load(t, "metrics.json") for t in TIERS}
    out["checks"] = {
        "FEATURE_DELTA_PASS": checks["FEATURE_DELTA_PASS"],
        "SAMPLE_IDENTITY_PASS": checks["SAMPLE_IDENTITY_PASS"],
    }
    # ---- 主表: timing / s0 两轨的 IC 与 top 经济学 ----
    rows = []
    for track in ("timing", "s0"):
        for t in ("a1", "a2", "a3", "a4"):
            mm = (m.get(t) or {}).get("tracks", {}).get(track)
            if not mm:
                continue
            rows.append({
                "track": track, "tier": t,
                "ic_mean": mm["ic_mean"], "ic_std": mm["ic_std"],
                "ic_ir": mm["ic_ir"],
                "top_label": mm["pooled_oof"] if False else None,
                "pooled_label_mean": mm["pooled_oof"].get("label_mean"),
                "top_label_mean": None,  # 见 per_fold 汇总
                "loss_rate": mm["pooled_oof"].get("loss_rate"),
                "failed_rebound": mm["pooled_oof"].get("failed_rebound_rate"),
            })
        # top bucket per fold 汇总 (加权)
        for t in ("a1", "a2", "a3", "a4"):
            mm = (m.get(t) or {}).get("tracks", {}).get(track)
            if not mm:
                continue
            tops = [v["top"] for v in mm["per_fold"].values()
                    if isinstance(v, dict) and v.get("top", {}).get("n", 0) > 0]
            if tops:
                n = sum(x["n"] for x in tops)
                wmean = sum(x["label_mean"] * x["n"] for x in tops
                            if x.get("label_mean") is not None) / max(1, n)
                for r in rows:
                    if r["track"] == track and r["tier"] == t:
                        r["top_label_mean"] = round(wmean, 6)
    out["ablation_table"] = rows

    # ---- Q1-Q4 ----
    def ic(t, track="timing"):
        return ((m.get(t) or {}).get("tracks", {}).get(track) or {}).get("ic_mean")

    def top_econ(t, track="timing"):
        mm = (m.get(t) or {}).get("tracks", {}).get(track)
        if not mm:
            return None
        tops = [v["top"] for v in mm["per_fold"].values()
                if isinstance(v, dict) and v.get("top", {}).get("n", 0) > 0]
        if not tops:
            return None
        n = sum(x["n"] for x in tops)
        lab = sum((x.get("label_mean") or 0) * x["n"] for x in tops) / max(1, n)
        alp = sum((x.get("alpha_mean") or 0) * x["n"] for x in tops) / max(1, n)
        mae = sum((x.get("mae_30m") or 0) * x["n"] for x in tops) / max(1, n)
        return {"n": n, "top_label_mean": lab, "top_alpha_mean": alp,
                "top_mae_30m": mae}

    q = {}
    m0 = m.get("a0") or {}
    q["Q1_candidate_increment"] = {
        "question": "GREEN-WATCH-v1.1 相比 V0 是否真的改变了负基线? (A0->A1 = Candidate System Increment)",
        "a0_universe_timing_label_mean": (m0.get("universe", {}).get("timing") or {})
            .get("label_mean"),
        "a0_old_score_top10_timing_label_mean": (m0.get("old_score_top10", {})
            .get("timing") or {}).get("label_mean"),
        "a1_top_timing_label_mean": (top_econ("a1") or {}).get("top_label_mean"),
        "a1_ic": ic("a1"),
    }
    q["Q2_zone_increment"] = {
        "question": "Zone/Structural Liquidity 有没有独立增量?",
        "ic_a1": ic("a1"), "ic_a2": ic("a2"),
        "top_a1": (top_econ("a1") or {}).get("top_label_mean"),
        "top_a2": (top_econ("a2") or {}).get("top_label_mean"),
    }
    q["Q3_dip_increment"] = {
        "question": "Dip depth/speed 有没有独立增量?",
        "ic_a2": ic("a2"), "ic_a3": ic("a3"),
        "top_a2": (top_econ("a2") or {}).get("top_label_mean"),
        "top_a3": (top_econ("a3") or {}).get("top_label_mean"),
    }
    q["Q4_recovery_increment"] = {
        "question": "Recovery/Reclaim 是否是最关键的 Timing 信息?",
        "ic_a3": ic("a3"), "ic_a4": ic("a4"),
        "top_a3": (top_econ("a3") or {}).get("top_label_mean"),
        "top_a4": (top_econ("a4") or {}).get("top_label_mean"),
    }
    out["questions"] = q

    # ---- 强制切片 (timing 轨, A4) ----
    a4 = ((m.get("a4") or {}).get("tracks", {}).get("timing") or {}).get("slices")
    out["a4_timing_slices"] = a4
    vj = (a4 or {}).get("V1_REJECTED_V11_ACCEPTED")
    out["falsification_slice"] = {
        "question": "v1_rejected_v11_accepted (DIP Gate 证伪组) 表现如何?",
        "detail": vj,
        "criteria": "H3/timing 更好 + MAE 不恶化 + loss rate 可控 -> v1.1 修改有价值",
    }
    # pre/post float mv (a4 与 a1 都报)
    out["float_mv_regime_slices"] = {
        t: (((m.get(t) or {}).get("tracks", {}).get("timing") or {}).get("slices")
            or {}).get("FLOAT_MV_PRE_20260701", {}) for t in ("a1", "a4")}
    out["float_mv_regime_slices_post"] = {
        t: (((m.get(t) or {}).get("tracks", {}).get("timing") or {}).get("slices")
            or {}).get("FLOAT_MV_POST_20260701", {}) for t in ("a1", "a4")}

    # ---- 账本 ----
    lp = C.RUN / "x0_ledger" / "ledger_report.json"
    if lp.exists():
        out["x0_ledger"] = json.loads(lp.read_text(encoding="utf-8"))

    (C.RUN / "report.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8")

    # ---- Markdown ----
    L = ["# A0-A4 第一轮报告 (Green V1 Verdict 前置)", ""]
    L.append(f"- SAMPLE_IDENTITY_PASS = **{out['checks']['SAMPLE_IDENTITY_PASS']}**")
    L.append(f"- FEATURE_DELTA_PASS = **{out['checks']['FEATURE_DELTA_PASS']}**")
    L.append("")
    L.append("## 消融主表 (IC = OOF Rank IC, 3 seed x 4 fold)")
    L.append("")
    L.append("| track | tier | IC mean | IC std | IC IR | top label | pooled label | loss rate | failed rebound |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        L.append(f"| {r['track']} | {r['tier']} | {_fmt(r['ic_mean'])} "
                 f"| {_fmt(r['ic_std'])} | {_fmt(r['ic_ir'])} "
                 f"| {_fmt(r['top_label_mean'], pct=True)} "
                 f"| {_fmt(r['pooled_label_mean'], pct=True)} "
                 f"| {_fmt(r['loss_rate'], pct=True)} "
                 f"| {_fmt(r['failed_rebound'], pct=True)} |")
    L.append("")
    L.append("## Q1-Q4")
    L.append("")
    for k, v in q.items():
        L.append(f"### {k}")
        L.append(f"- {v['question']}")
        for kk, vv in v.items():
            if kk != "question":
                L.append(f"- {kk}: {vv}")
        L.append("")
    L.append("## 证伪切片 v1_rejected_v11_accepted (A4 timing)")
    if vj:
        for kk, vv in vj.items():
            L.append(f"- {kk}: {vv}")
    L.append("")
    L.append("## FLOAT_MV regime (强制切片)")
    for t in ("a1", "a4"):
        pre = out["float_mv_regime_slices"].get(t, {})
        post = out["float_mv_regime_slices_post"].get(t, {})
        L.append(f"- {t}: pre n={pre.get('n')} label={pre.get('label_mean')} / "
                 f"post n={post.get('n')} label={post.get('label_mean')}")
    L.append("")
    if "x0_ledger" in out:
        L.append("## X0 20 槽账本")
        L.append("")
        L.append("| tier | trades | ann | win | avg net | MDD(trade) | occ |")
        L.append("|---|---|---|---|---|---|---|")
        for t, s in out["x0_ledger"].get("tiers", {}).items():
            L.append(f"| {t} | {s['n_trades']} | {_fmt(s['ann_return'], pct=True)} "
                     f"| {_fmt(s['win_rate'], pct=True)} "
                     f"| {_fmt(s['avg_net'], pct=True)} "
                     f"| {_fmt(s['max_drawdown_trade_level'], pct=True)} "
                     f"| {_fmt(s['avg_slot_occupancy'], nd=1)} |")
    (C.RUN / "REPORT.md").write_text("\n".join(L), encoding="utf-8")
    print(f"[report] -> {C.RUN / 'REPORT.md'}")


if __name__ == "__main__":
    main()
