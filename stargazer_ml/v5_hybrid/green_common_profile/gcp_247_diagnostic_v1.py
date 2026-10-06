# -*- coding: utf-8 -*-
"""现有可信子集经济诊断 v1 — 按 FROZEN 合同执行一次零网络描述性诊断.

执行前: 逐件复验合同 §8 冻结哈希 + manifest 5 文件, 任一不一致即中止并报审计.
输入: gcp_247_phase1_status_p2b1.json (人口/轨道), gcp_adj_cache_acceptance.json (公司行为三档证据).
口径: 合同 §1/§3/§9/§10 — 子集=已闭环轨道, 扣除跨段/公司行为②/尺度未通过; 公司行为③与 EMPTY 不排除;
NULL 不入分位; 缺失与残仓单列; 持有期双量纲 (交易日历 + 按日期版本网格计数, 禁止时间戳直减).
输出: gcp_247_diagnostic_v1.json/.md (RUN 新文件; 写前查预算). 不产生 PASS/FAIL 盈利裁决.
"""
from __future__ import annotations
import sys, json, time, hashlib
from pathlib import Path
from collections import Counter, defaultdict

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import dual_path_execution_v0_1_1 as D

BASE = HERE / "gcp_round_b_small_sample_v1"
OUT_JSON = BASE / "gcp_247_diagnostic_v1.json"
OUT_MD = BASE / "gcp_247_diagnostic_v1.md"
WARN = "非新盲测、非全策略表现"


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def dir_size(p: Path) -> int:
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file())


def write_checked(path: Path, text: str):
    if dir_size(BASE) + len(text.encode("utf-8")) > 50 * 1024 * 1024:
        raise RuntimeError("budget exceeded (stop)")
    path.write_text(text, encoding="utf-8")


FROZEN = {
    "population_status": ("gcp_247_phase1_status_p2b1.json", "44283a733b409e72f81cbbabb11b1987af4c32ba5a8dbf51283f6c655f5799c7"),
    "population_status_md": ("gcp_247_phase1_status_p2b1.md", "f9965da7819ae1a85f692ba45217fbd022d21ddfcfe1a9407595b82a72ad8010"),
    "needs_v0_1_2": ("gcp_247_phase2_needs_v0_1_2.json", "50176792425c7e738b3874cabcca42893e3be164483a1926152667b9af46fafc"),
    "coverage_scope": ("gcp_247_coverage_scope_v0_1_2.json", "ce68a46bc56e7288e85d381e7dab7be8b7d9625e7ea2fd7c0129b268dbebbb9a"),
    "p2b1_frozen_list": ("p2b1_collect/frozen_request_list.json", "31d27779a81353a7aa296c705e9c34014f0d4bdcbd5decaf04a3617fac68008a"),
    "p2b1_qa_report": ("p2b1_collect/collection_report.json", "c6e48efd347e2dfaa46c8b93f71e83d5baebdb8d42a9f4176f8d1d1a38a9994f"),
    "execution_contract": ("../contracts/dual_path_execution_contract_v0_1_1_final.md", "9c22c37e5e9bcc4e874f063730f19b138fac27f0c89de04d6b39cc1c0a8b852d"),
    "executor": ("../dual_path_execution_v0_1_1.py", "aee8e60bc00951a3f18975d7a01dda7c64af27a9eb7d5e1592cd8c3716aa91cb"),
    "frozen_247_scope": ("gcp_247_coverage_verify.json", "f2974391fcff26e75bea450426093679bb8547e17125d21589f5108ddfb9b673"),
    "adj_acceptance": ("gcp_adj_cache_acceptance.json", "15007a2526da0bb49f05a8e4cc52c195f5787b1c982a4cff5b00ee9b108505db"),
}


def verify_hashes() -> dict:
    res = {}
    ok_all = True
    for role, (rel, expect) in FROZEN.items():
        p = (BASE / rel).resolve()
        actual = sha256_file(p) if p.exists() else None
        m = (actual == expect)
        ok_all &= m
        res[role] = {"path": rel, "match": m, "sha256": actual}
    man = json.loads((HERE / "contracts" / "dual_path_execution_v0_1_1_manifest.json").read_text(encoding="utf-8"))
    for f in man["files"]:
        p = Path("D:/stargazer") / f["path"]
        m = sha256_file(p) == f["sha256"]
        ok_all &= m
        res[f"manifest:{f['role']}"] = {"match": m}
    if not ok_all:
        bad = [k for k, v in res.items() if v.get("match") is False]
        raise SystemExit(f"执行前哈希复验失败: {bad} -> 中止并报审计")
    return res


def trading_minutes(entry_ts: int, exit_ts: int, cal, ci) -> int:
    """按日期版本网格计数: 严格晚于入场成交槽且不晚于退出成交槽的槽数."""
    n = 0
    d0, d1 = entry_ts // 1000000, exit_ts // 1000000
    i = ci[d0]
    while i < len(cal) and cal[i] <= d1:
        d = cal[i]
        for s in D.day_grid(d):
            ts = d * 1000000 + s
            if entry_ts < ts <= exit_ts:
                n += 1
        i += 1
    return n


def main():
    t0 = time.time()
    hv = verify_hashes()
    status = json.loads((BASE / "gcp_247_phase1_status_p2b1.json").read_text(encoding="utf-8"))
    adj = {(r["code"], int(r["E"])): r for r in
           json.loads((BASE / "gcp_adj_cache_acceptance.json").read_text(encoding="utf-8"))["per_event"]}
    cal = D.load_calendar()
    ci = {d: i for i, d in enumerate(cal)}

    def pct(xs, q):
        return round(float(np.percentile(np.asarray(xs, dtype=float), q)), 6) if xs else None

    def dist(xs):
        return {"n": len(xs), "mean": (round(float(np.mean(xs)), 6) if xs else None),
                "median": pct(xs, 50), "p5": pct(xs, 5), "p25": pct(xs, 25),
                "p75": pct(xs, 75), "p95": pct(xs, 95)}

    groups = defaultdict(lambda: {"nets": [], "rows": [], "hold_d": [], "hold_m": []})
    excl = defaultdict(Counter)
    corp_tier3, residuals, null_detail = [], [], defaultdict(Counter)
    pop_seg_events = Counter(e["segment"] for e in status["events"])
    exit_kind = defaultdict(Counter)

    for e in status["events"]:
        code, E_, seg = e["code"], e["E"], e["segment"]
        a = adj.get((code, int(E_)), {})
        for tr in ("H3", "H5"):
            t = e.get(tr)
            key = (seg, tr)
            if not t:
                excl[key]["NO_FILL_CONFIRMED(事件未成交)"] += 1
                null_detail[(seg, tr)]["no_fill_event"] += 1
                continue
            st = t.get("status")
            if not (st == "SIM_COMPLETE" and t.get("fully_exited") and t.get("net") is not None):
                # NULL / 未闭环 (缺失, 单列, 不入分位)
                if t.get("remaining_shares"):
                    residuals.append({"code": code, "E": E_, "track": tr, "segment": seg,
                                      "remaining_shares": t.get("remaining_shares"),
                                      "status": st, "reason": t.get("reason")})
                    excl[key][f"RESIDUAL({st})"] += 1
                    null_detail[(seg, tr)]["residual"] += 1
                else:
                    excl[key][f"NULL({st})"] += 1
                    null_detail[(seg, tr)][st or "NULL"] += 1
                continue
            # 已闭环 -> 扣除项判定
            if (e.get("cross_segment_label") or {}).get(tr):
                excl[key]["CROSS_SEGMENT_LABEL"] += 1
                continue
            if a.get("status") not in ("OK", "EMPTY") or a.get("in_failures_log"):
                excl[key]["CORP_TIER2_UNKNOWN"] += 1
                continue
            # 尺度证据: 已闭环轨道已通过引擎依赖校验(网格/量价) + 采集QA, 无未通过项
            g = groups[key]
            net = float(t["net"])
            g["nets"].append(net)
            entry_ts = (t.get("sim_fill") or {}).get("ts")
            exit_ts = t.get("last_fill_ts")
            hd = hm = None
            if entry_ts and exit_ts and (entry_ts // 1000000) in ci and (exit_ts // 1000000) in ci:
                hd = ci[exit_ts // 1000000] - ci[entry_ts // 1000000]
                hm = trading_minutes(int(entry_ts), int(exit_ts), cal, ci)
                g["hold_d"].append(hd)
                g["hold_m"].append(hm)
            kinds = tuple(sorted({s.get("kind") for s in (t.get("sells") or [])}))
            kind = kinds[0] if len(kinds) == 1 else ("MIXED" if kinds else "NONE")
            exit_kind[key][kind] += 1
            g["rows"].append({"code": code, "E": E_, "net": net,
                              "exit_date": (exit_ts // 1000000) if exit_ts else None,
                              "entry_date": (entry_ts // 1000000) if entry_ts else None,
                              "kind": kind, "hold_days": hd, "hold_minutes": hm,
                              "corp_in_window": bool(a.get("adj_in_window"))})
            if a.get("adj_in_window"):
                corp_tier3.append({"code": code, "E": E_, "track": tr, "segment": seg,
                                   "net": net, "note": "窗口内有公司行为(OK), 另行检查; 未排除"})

    # ---- 汇总 ----
    tables = {}
    for (seg, tr), g in sorted(groups.items()):
        nets = g["nets"]
        n = len(nets)
        dates = sorted({r["exit_date"] for r in g["rows"] if r["exit_date"]})
        small = (n < 30) or (len(dates) < 20)
        d = dist(nets)
        d.update({
            "unique_events": len({r["E"] for r in g["rows"]}),
            "unique_codes": len({r["code"] for r in g["rows"]}),
            "n_exit_dates": len(dates),
            "n_e_dates": len({r["E"] for r in g["rows"]}),
            "availability_rate": round(n / (2 * pop_seg_events[seg]), 4),
            "population_tracks": 2 * pop_seg_events[seg],
            "population_events": pop_seg_events[seg],
            "small_sample": small,
            "quantile_note": ("小样本, 仅描述; 分位不稳定" if small else None),
            "warning": WARN,
            "exit_kind": dict(exit_kind[(seg, tr)]),
            "holding_days": dist(g["hold_d"]),
            "holding_minutes": dist(g["hold_m"]),
        })
        worst = sorted(g["rows"], key=lambda r: r["net"])[:min(10, n)]
        tables[f"{seg}/{tr}"] = {"distribution": d, "worst_cases": worst}
    excl_tab = {f"{k[0]}/{k[1]}": dict(v) for k, v in sorted(excl.items())}
    null_tab = {f"{k[0]}/{k[1]}": dict(v) for k, v in sorted(null_detail.items())}

    out = {"phase": "GCP_247_DIAGNOSTIC_V1_DESCRIPTIVE",
           "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
           "network_requests": 0,
           "contract": "contracts/trusted_subset_diagnostic_contract_v1.md (FROZEN 描述性诊断 v1)",
           "no_pass_fail_verdict": True,
           "hash_verification": hv,
           "population": {"class_counts": status["summary"]["class_counts"],
                          "class_by_segment": status["summary"]["class_by_segment"],
                          "track_status": status["summary"]["track_status"],
                          "segment_events": dict(pop_seg_events)},
           "subset_tables": tables,
           "exclusion_reasons": excl_tab,
           "null_and_missing_breakdown": null_tab,
           "residual_positions": {"n": len(residuals), "rows": residuals,
                                  "note": "残仓只报股数, 不折现金; 净收益 NULL, 不入分位"},
           "corp_action_tier3_listing": {"n": len(corp_tier3), "rows": corp_tier3,
                                         "note": "窗口内有公司行为(证据OK), 另行检查; 未排除; EMPTY=已确认可用不排除"},
           "definitions": {"net": "C_sell/C_buy-1 (含滑点+费用); 仅已闭环非NULL轨道",
                           "holding_days": "真实交易日历 index 差 (禁止时间戳直减)",
                           "holding_minutes": "按日期版本网格槽计数, 入场槽<槽<=退出槽",
                           "dates_count": "last_fill 所在交易日去重; n_e_dates=E日去重(备查)",
                           "percentile": "线性插值; 小样本(n<30 或日期<20)注明不稳定",
                           "null_policy": "NULL 不入分位, 不假定亏损或零; 缺失与残仓单列"},
           "warning": WARN,
           "elapsed_s": round(time.time() - t0, 1)}
    write_checked(OUT_JSON, json.dumps(out, ensure_ascii=False, indent=1))

    # ---- md ----
    L = ["# 可信子集描述性诊断 v1 (零网络, 按 FROZEN 合同)", "",
         f"- 生成: {out['generated_at']} | 哈希复验: 全过 (10 输入 + manifest 5 文件) | 用时 {out['elapsed_s']}s",
         f"- **{WARN}**; 本报告不产生 PASS/FAIL 盈利裁决; NULL 不入分位, 缺失与残仓单列.", "",
         "## 全人口状态 (分母)", "",
         f"- class: {out['population']['class_counts']}",
         f"- 分段事件数: {out['population']['segment_events']}",
         f"- 轨道状态: H3={out['population']['track_status']['H3']} H5={out['population']['track_status']['H5']}", "",
         "## 子集收益分布 (已闭环非NULL; 含小样本警示)", "",
         "| 段/轨道 | n轨道 | 事件 | 股票 | 退出日数 | 可用率 | 均值 | 中位 | P5 | P25 | P75 | P95 | 警示 |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for k, tb in tables.items():
        d = tb["distribution"]
        L.append(f"| {k} | {d['n']} | {d['unique_events']} | {d['unique_codes']} | {d['n_exit_dates']} | "
                 f"{d['availability_rate']*100:.1f}% | {d['mean']} | {d['median']} | {d['p5']} | {d['p25']} | "
                 f"{d['p75']} | {d['p95']} | {'小样本,仅描述' if d['small_sample'] else '-'} |")
    L += ["", "## 持有期双量纲 (均值 | 中位数)", ""]
    for k, tb in tables.items():
        hd, hm = tb["distribution"]["holding_days"], tb["distribution"]["holding_minutes"]
        L.append(f"- {k}: 交易日 {hd['mean']} | {hd['median']} ; 交易分钟 {hm['mean']} | {hm['median']} (n={hd['n']})")
    L += ["", "## 退出方式构成", ""]
    for k, tb in tables.items():
        L.append(f"- {k}: {tb['distribution']['exit_kind']}")
    L += ["", "## 最差例 (每段每轨道 min(10,n); 不据此改规则)", ""]
    for k, tb in tables.items():
        L.append(f"### {k}")
        for r in tb["worst_cases"]:
            L.append(f"- {r['code']}@{r['E']} net={r['net']} {r['kind']} 持有{r['hold_days']}日/{r['hold_minutes']}分"
                     + (" [窗口内有公司行为]" if r["corp_in_window"] else ""))
    L += ["", "## 排除原因统计 (人口分母内)", ""]
    for k, v in excl_tab.items():
        L.append(f"- {k}: {v}")
    L += ["", "## 缺失与残仓单列", "",
          f"- 残仓轨道: {len(residuals)} (remaining_shares 股数单列, 不折现金; 明细见 json)",
          "- NULL/缺失分解: " + json.dumps(null_tab, ensure_ascii=False),
          f"- 公司行为③另行检查清单: {len(corp_tier3)} 轨 (明细见 json); EMPTY=已确认可用未排除", "",
          "## 声明", "",
          f"- {WARN}; test 子集样本量与构成偏差见上表警示; 跨期有效性须新前瞻时段另行批准.",
          "- 未做: 账户级回撤 / PASS-FAIL 裁决 / 阈值调整 / 训练 / 按收益挑样本.", ""]
    write_checked(OUT_MD, "\n".join(L))
    print(f"diagnostic done ({time.time()-t0:.1f}s): groups={ {k: tb['distribution']['n'] for k, tb in tables.items()} }")
    print(f"exclusions: { {k: sum(v.values()) for k, v in excl_tab.items()} }")
    print(f"residuals={len(residuals)} corp_tier3={len(corp_tier3)}")


if __name__ == "__main__":
    main()
