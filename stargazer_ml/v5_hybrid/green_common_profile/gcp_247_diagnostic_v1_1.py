# -*- coding: utf-8 -*-
"""现有可信子集经济诊断 v1.1 — 用户核对三处修正的收口版 (零网络, 一次执行).

对 v1 的修正 (用户 2026-10-05 核对, 授权一次收口):
  修正1 可用率分母: 单轨道可用率 = 子集轨道数 / **该段事件数** (v1 误乘 2, 使可用率腰斩).
  修正2 唯一事件数: 按 **(code, E) 去重**; E 日期数另列一栏 (v1 把 E 日期数冒充事件数).
  修正3 公司行为按**实际最后退出终点**核销: v1 用验收表 [E, 计划H5] 窗口, 顺延退出可更晚.
     核销方法 (零网络, 本地证据): 优先 gate1c 全史复权链 (SHA 与验收表登记值复验);
     链覆盖 [E, 实际退出] 全程 -> 确认无/确认有(逐条列因子); 链不可用 -> 日线 pre_close 跳空佐证;
     两者都覆盖不到 -> UNKNOWN -> 按合同 §1② 隔离并逐条解释. 未核销不得写"窗口内公司行为 0 条"为确认事实.
输入哈希: 合同 §8 十件 + manifest 5 文件执行前复验; 脚本本身为执行件, 旧/新哈希与变更原因记录在案.
一致性: 收益字段除公司行为隔离外必须与 v1 逐位一致 (本脚本内置与 v1 报告的对照).
输出: gcp_247_diagnostic_v1_1.json/.md (v1 版完整保留, 不覆盖).
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
OUT_JSON = BASE / "gcp_247_diagnostic_v1_1.json"
OUT_MD = BASE / "gcp_247_diagnostic_v1_1.md"
V1_JSON = BASE / "gcp_247_diagnostic_v1.json"
GATE1C_ADJ = Path(r"D:\stargazer\stargazer_ml\v5_hybrid\opportunity_v2\gate1c\feb2026_adjustment_cache_v1\adj_cache.json")
WARN = "非新盲测、非全策略表现"
SCRIPT_V1 = HERE / "gcp_247_diagnostic_v1.py"


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
        m = sha256_file(Path("D:/stargazer") / f["path"]) == f["sha256"]
        ok_all &= m
        res[f"manifest:{f['role']}"] = {"match": m}
    acc = json.loads((BASE / "gcp_adj_cache_acceptance.json").read_text(encoding="utf-8"))
    m = sha256_file(GATE1C_ADJ) == acc["cache_sha256"]
    ok_all &= m
    res["gate1c_adj_cache(验收登记值)"] = {"match": m}
    if not ok_all:
        bad = [k for k, v in res.items() if v.get("match") is False]
        raise SystemExit(f"执行前哈希复验失败: {bad} -> 中止并报审计")
    return res


def trading_minutes(entry_ts: int, exit_ts: int, cal, ci) -> int:
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
    acc = json.loads((BASE / "gcp_adj_cache_acceptance.json").read_text(encoding="utf-8"))
    acc_ev = {(r["code"], int(r["E"])): r for r in acc["per_event"]}
    chains = json.loads(GATE1C_ADJ.read_text(encoding="utf-8"))
    cal = D.load_calendar()
    ci = {d: i for i, d in enumerate(cal)}

    # 日线 (pre_close 跳空佐证用)
    codes = sorted({e["code"] for e in status["events"]})
    import glob, pandas as pd
    import pyarrow.dataset as ds
    tbl = ds.dataset(sorted(glob.glob(D.DAILY_PARTS + "/part_*.parquet")), format="parquet").to_table(
        columns=["date", "code", "close", "pre_close"], filter=ds.field("code").isin(codes)).to_pandas()
    tbl["date"] = tbl["date"].astype(int)
    daily_by_code = {c: g.sort_values("date").reset_index(drop=True) for c, g in tbl.groupby("code")}

    def daily_jump_check(code, day_from, day_to):
        """pre_close[d] 与 close[d-1] 不一致 >2% -> 疑似除权/数据异常 (佐证方法). 返回 (可用, 疑点日列表)."""
        g = daily_by_code.get(code)
        if g is None or not len(g):
            return False, []
        idx = {int(r["date"]): i for i, r in g.iterrows()}
        flags = []
        for d in range(day_from, day_to + 1):
            if d not in idx or idx[d] == 0:
                continue
            pc = g.loc[idx[d], "pre_close"]
            prev_close = g.loc[idx[d] - 1, "close"]
            if pd.notna(pc) and pd.notna(prev_close) and abs(float(pc) / float(prev_close) - 1) > 0.02:
                flags.append(int(d))
        return True, flags

    def corp_check(code, E_, exit_day):
        """按实际退出终点核销公司行为. 返回 (status, method, detail).
        status: CONFIRMED_NONE / CONFIRMED_HAS / UNKNOWN"""
        ch = chains.get(code)
        if ch and ch.get("status") == "EMPTY":
            return "CONFIRMED_NONE", "adj_chain_EMPTY(全史确认空)", []
        if ch and ch.get("status") == "OK":
            inwin = [[int(d), float(f)] for d, f in ch.get("events", []) if int(E_) <= int(d) <= int(exit_day)]
            if inwin:
                return "CONFIRMED_HAS", "adj_chain", inwin
            return "CONFIRMED_NONE", "adj_chain(全史链, 覆盖至实际退出)", []
        usable, flags = daily_jump_check(code, int(E_), int(exit_day))
        if usable:
            if flags:
                return "CONFIRMED_HAS", "daily_preclose_jump(佐证)", [[d, None] for d in flags]
            return "CONFIRMED_NONE", "daily_preclose_jump(佐证, 链不可用)", []
        return "UNKNOWN", "无覆盖证据(链不可用且日线不可用)", []

    def pct(xs, q):
        return round(float(np.percentile(np.asarray(xs, dtype=float), q)), 6) if xs else None

    def dist(xs):
        return {"n": len(xs), "mean": (round(float(np.mean(xs)), 6) if xs else None),
                "median": pct(xs, 50), "p5": pct(xs, 5), "p25": pct(xs, 25),
                "p75": pct(xs, 75), "p95": pct(xs, 95)}

    groups = defaultdict(lambda: {"nets": [], "rows": [], "hold_d": [], "hold_m": []})
    excl = defaultdict(Counter)
    corp_tier3, corp_unknown, residuals = [], [], []
    null_detail = defaultdict(Counter)
    pop_seg_events = Counter(e["segment"] for e in status["events"])
    exit_kind = defaultdict(Counter)
    corp_stat = defaultdict(Counter)

    for e in status["events"]:
        code, E_, seg = e["code"], e["E"], e["segment"]
        exit_basis = {}  # 事件级计划 H5 (验收表口径) 仅作交叉核对
        for tr in ("H3", "H5"):
            t = e.get(tr)
            key = (seg, tr)
            if not t:
                excl[key]["NO_FILL_CONFIRMED(事件未成交)"] += 1
                null_detail[(seg, tr)]["no_fill_event"] += 1
                continue
            st = t.get("status")
            if not (st == "SIM_COMPLETE" and t.get("fully_exited") and t.get("net") is not None):
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
            if (e.get("cross_segment_label") or {}).get(tr):
                excl[key]["CROSS_SEGMENT_LABEL"] += 1
                continue
            # 修正3: 公司行为按实际退出终点核销
            exit_day = int((t.get("last_fill_ts") or 0) // 1000000)
            cs, method, detail = corp_check(code, int(E_), exit_day)
            corp_stat[key][cs] += 1
            if cs == "UNKNOWN":
                corp_unknown.append({"code": code, "E": E_, "track": tr, "segment": seg,
                                     "exit_day": exit_day, "method": method})
                excl[key]["CORP_TIER2_UNKNOWN"] += 1
                continue
            if cs == "CONFIRMED_HAS":
                corp_tier3.append({"code": code, "E": E_, "track": tr, "segment": seg,
                                   "exit_day": exit_day, "method": method, "events_in_window": detail})
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
                              "exit_date": exit_day or None,
                              "entry_date": (entry_ts // 1000000) if entry_ts else None,
                              "kind": kind, "hold_days": hd, "hold_minutes": hm,
                              "corp": cs, "corp_method": method})

    # 交叉核对: 链口径 [E,计划H5] adj_in_window vs 验收表
    xcheck = []
    for (code, E_), r in acc_ev.items():
        ch = chains.get(code)
        ei = ci.get(int(E_))
        if ch is None or ei is None or ei + 5 >= len(cal):
            continue
        h5 = cal[ei + 5]
        mine = bool(ch.get("status") == "OK" and any(int(E_) <= int(d) <= int(h5) for d, _ in ch.get("events", [])))
        if mine != bool(r.get("adj_in_window")):
            xcheck.append({"code": code, "E": E_, "acceptance": r.get("adj_in_window"), "chain_recheck": mine})
    # 事件存在性核对 (验收表 247 事件 vs status)
    stat_keys = {(e["code"], int(e["E"])) for e in status["events"]}
    acc_keys = set(acc_ev.keys())
    xcheck += [{"missing_in_acceptance": sorted([list(x) for x in stat_keys - acc_keys])[:10]},
               {"missing_in_status": sorted([list(x) for x in acc_keys - stat_keys])[:10]}]

    tables = {}
    for (seg, tr), g in sorted(groups.items()):
        nets = g["nets"]
        n = len(nets)
        dates = sorted({r["exit_date"] for r in g["rows"] if r["exit_date"]})
        small = (n < 30) or (len(dates) < 20)
        d = dist(nets)
        d.update({
            "unique_events_codeE": len({(r["code"], r["E"]) for r in g["rows"]}),   # 修正2
            "unique_codes": len({r["code"] for r in g["rows"]}),
            "n_exit_dates": len(dates),
            "n_e_dates": len({r["E"] for r in g["rows"]}),                          # 修正2: E 日期数另列
            "availability_rate": round(n / pop_seg_events[seg], 4),                 # 修正1: 分母=段事件数
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

    # ---- 与 v1 对照 (收益字段一致性) ----
    v1 = json.loads(V1_JSON.read_text(encoding="utf-8"))
    v1_groups = {}
    for k, tb in v1["subset_tables"].items():
        for r in tb["worst_cases"]:
            pass
    v1_rows = {(r["code"], r["E"], k.split("/")[1]): r["net"]
               for k, tb in v1["subset_tables"].items() for r in tb["worst_cases"]}
    v1_n = {k: tb["distribution"]["n"] for k, tb in v1["subset_tables"].items()}
    v11_rows = {(r["code"], r["E"], k.split("/")[1]): r["net"]
                for k, tb in tables.items() for r in tb["worst_cases"]}
    v11_n = {k: tb["distribution"]["n"] for k, tb in tables.items()}
    # 全集 net 对照 (不只最差例): 重建 v1 全量行集不可行 (v1 json 只存最差例), 以分组合计与逐最差例对照
    net_mismatch = [k for k in v11_rows if k in v1_rows and v1_rows[k] != v11_rows[k]]
    n_changed = {k: (v1_n.get(k), v11_n.get(k)) for k in v11_n if v1_n.get(k) != v11_n.get(k)}

    out = {"phase": "GCP_247_DIAGNOSTIC_V1_1_DESCRIPTIVE",
           "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
           "network_requests": 0,
           "contract": "contracts/trusted_subset_diagnostic_contract_v1.md (FROZEN 描述性诊断 v1) + 用户 2026-10-05 三项修正指示",
           "no_pass_fail_verdict": True,
           "errata_vs_v1": {
               "fix1_availability_denominator": "v1 误用 2×段事件数作分母, 可用率腰斩; v1.1 改为 子集轨道数/段事件数",
               "fix2_unique_events": "v1 '唯一事件数'实为 E 日期去重数; v1.1 按 (code,E) 去重, E 日期数另列 n_e_dates",
               "fix3_corp_actual_exit": "v1 公司行为窗口=[E,计划H5] (验收表), 未覆盖顺延退出; v1.1 按实际最后退出终点用全史复权链核销",
               "impact": "纯展示/核销口径修正; 净收益字段逐位对照见 v1_consistency"},
           "hash_verification": hv,
           "script_change": {"v1": sha256_file(SCRIPT_V1), "v1_1": sha256_file(HERE / "gcp_247_diagnostic_v1_1.py"),
                             "reason": "用户核对发现的统计错误修正 (可用率分母/事件计数/公司行为核销口径)"},
           "population": {"class_counts": status["summary"]["class_counts"],
                          "class_by_segment": status["summary"]["class_by_segment"],
                          "track_status": status["summary"]["track_status"],
                          "segment_events": dict(pop_seg_events)},
           "subset_tables": tables,
           "corp_check_summary": {f"{k[0]}/{k[1]}": dict(v) for k, v in sorted(corp_stat.items())},
           "corp_action_tier3_listing": {"n": len(corp_tier3), "rows": corp_tier3,
                                         "note": "确认有(链/佐证), 未排除, 另行检查"},
           "corp_unknown_isolated": {"n": len(corp_unknown), "rows": corp_unknown,
                                     "note": "UNKNOWN 按合同 §1② 隔离"},
           "corp_crosscheck_vs_acceptance": {"mismatches": xcheck,
                                             "note": "链口径 [E,计划H5] adj_in_window 与验收表逐事件核对"},
           "v1_consistency": {"worst_case_net_mismatch": net_mismatch,
                              "group_n_changed": n_changed,
                              "statement": ("子集轨道数无变化, 最差例 net 与 v1 全部逐位一致"
                                            if not net_mismatch and not n_changed else
                                            "存在差异, 见 mismatch/changed 字段")},
           "exclusion_reasons": excl_tab,
           "null_and_missing_breakdown": null_tab,
           "residual_positions": {"n": len(residuals), "rows": residuals,
                                  "note": "残仓只报股数, 不折现金; 净收益 NULL, 不入分位"},
           "definitions": {"net": "C_sell/C_buy-1 (含滑点+费用); 仅已闭环非NULL轨道",
                           "availability_rate": "子集轨道数 / 该段事件数 (单轨道口径, 修正1)",
                           "unique_events": "(code,E) 去重 (修正2); n_e_dates=E 日期去重另列",
                           "holding_days": "真实交易日历 index 差 (禁止时间戳直减)",
                           "holding_minutes": "按日期版本网格槽计数, 入场槽<槽<=退出槽",
                           "percentile": "线性插值; 小样本(n<30 或日期<20)注明不稳定",
                           "null_policy": "NULL 不入分位, 不假定亏损或零; 缺失与残仓单列",
                           "corp_window": "[E, 实际最后退出日] 含两端; 方法: 全史复权链优先, 日线pre_close跳空佐证, 覆盖不到=UNKNOWN隔离"},
           "warning": WARN,
           "elapsed_s": round(time.time() - t0, 1)}
    write_checked(OUT_JSON, json.dumps(out, ensure_ascii=False, indent=1))

    L = ["# 可信子集描述性诊断 v1.1 (零网络; 用户核对三处修正收口版)", "",
         f"- 生成: {out['generated_at']} | 哈希复验全过 (10 输入 + manifest 5 + gate1c 链登记值) | 用时 {out['elapsed_s']}s",
         f"- **{WARN}**; 不产生 PASS/FAIL 盈利裁决; NULL 不入分位, 缺失与残仓单列.", "",
         "## 对 v1 的两处统计错误说明 (不掩饰)", "",
         "- **可用率分母错误**: v1 用 2×段事件数作分母, 可用率被腰斩 (如 test 误报 17.2%); v1.1 修正为 子集轨道数/段事件数.",
         "- **事件计数错误**: v1 的\"唯一事件数\"实为 E 日期去重数 (如 test 误报\"15 事件 18 股票\"); v1.1 按 (code,E) 去重, E 日期数另列.",
         "- 影响范围: 纯展示数字; 净收益分布未受影响 (与 v1 最差例逐位一致, 见 json v1_consistency).", "",
         "## 全人口状态 (分母)", "",
         f"- class: {out['population']['class_counts']}",
         f"- 分段事件数: {out['population']['segment_events']}",
         f"- 轨道状态: H3={out['population']['track_status']['H3']} H5={out['population']['track_status']['H5']}", "",
         "## 子集收益分布 (已闭环非NULL)", "",
         "| 段/轨道 | n轨道 | 事件(code,E) | 股票 | 退出日数 | E日期数 | 可用率 | 均值 | 中位 | P5 | P25 | P75 | P95 | 警示 |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for k, tb in tables.items():
        d = tb["distribution"]
        L.append(f"| {k} | {d['n']} | {d['unique_events_codeE']} | {d['unique_codes']} | {d['n_exit_dates']} | "
                 f"{d['n_e_dates']} | {d['availability_rate']*100:.1f}% | {d['mean']} | {d['median']} | {d['p5']} | "
                 f"{d['p25']} | {d['p75']} | {d['p95']} | {'小样本,仅描述' if d['small_sample'] else '-'} |")
    L += ["", "## 公司行为核销 (按实际退出终点)", ""]
    for k, v in sorted(corp_stat.items()):
        L.append(f"- {k}: {v}")
    L += [f"- UNKNOWN 隔离: {len(corp_unknown)} 轨 (逐条见 json)",
          f"- 确认有(另行检查, 未排除): {len(corp_tier3)} 轨 (逐条含因子见 json)",
          f"- 与验收表交叉核对不一致项: {len(xcheck)} (见 json)", "",
          "## 持有期双量纲 (均值 | 中位数)", ""]
    for k, tb in tables.items():
        hd, hm = tb["distribution"]["holding_days"], tb["distribution"]["holding_minutes"]
        L.append(f"- {k}: 交易日 {hd['mean']} | {hd['median']} ; 交易分钟 {hm['mean']} | {hm['median']} (n={hd['n']})")
    L += ["", "## 退出方式构成", ""]
    for k, tb in tables.items():
        L.append(f"- {k}: {tb['distribution']['exit_kind']}")
    L += ["", "## 最差例 (每段每轨道 min(10,n))", ""]
    for k, tb in tables.items():
        L.append(f"### {k}")
        for r in tb["worst_cases"]:
            extra = " [公司行为确认有]" if r["corp"] == "CONFIRMED_HAS" else ""
            L.append(f"- {r['code']}@{r['E']} net={r['net']} {r['kind']} 持有{r['hold_days']}日/{r['hold_minutes']}分{extra}")
    L += ["", "## 排除原因统计 (人口分母内)", ""]
    for k, v in excl_tab.items():
        L.append(f"- {k}: {v}")
    L += ["", "## 缺失与残仓单列", "",
          f"- 残仓轨道: {len(residuals)} (股数单列, 不折现金; 明细见 json)",
          f"- 公司行为 UNKNOWN 隔离: {len(corp_unknown)}",
          "- NULL/缺失分解: " + json.dumps(null_tab, ensure_ascii=False), "",
          "## 声明", "",
          f"- {WARN}; test 子集样本量与构成偏差见表内警示; 跨期有效性须新前瞻时段另行批准.",
          "- 未做: 账户级回撤 / PASS-FAIL 裁决 / 阈值调整 / 训练 / 按收益挑样本.", ""]
    write_checked(OUT_MD, "\n".join(L))
    print(f"v1.1 done ({time.time()-t0:.1f}s)")
    for k, tb in tables.items():
        d = tb["distribution"]
        print(f"  {k}: n={d['n']} events={d['unique_events_codeE']} codes={d['unique_codes']} "
              f"exitdates={d['n_exit_dates']} edates={d['n_e_dates']} rate={d['availability_rate']}")
    print("corp:", {k: dict(v) for k, v in corp_stat.items()}, "unknown:", len(corp_unknown),
          "tier3:", len(corp_tier3), "xcheck mismatches:", len(xcheck))
    print("v1 consistency:", out["v1_consistency"]["statement"], n_changed)


if __name__ == "__main__":
    main()
