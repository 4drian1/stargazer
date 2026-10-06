# -*- coding: utf-8 -*-
"""《买点增量对照研究合同》v1 执行 — 两臂复放 + 预注册统计 (零网络, 一次连续完成).

臂 A = 冻结 S (entry.action_ts); 臂 B = E 日 09:35 决策 (signal_ts=E*1e6+93500, 09:36 槽 open 成交).
两臂共用执行器 v0.1.1 **未改动** — 唯一差异 = 传入的 signal_ts (决策时点); 派生差异 =
入场价→各臂止损价/退出路径/依赖终点. 运行脚本为本件 (登记哈希); 基线臂无需扩展执行器.
步骤 (用户授权四步): 1 复验 (合同8类+manifest5+链登记值) + 分钟语义钉死核对;
2 两臂复放 247 冻结事件, 依赖按各臂实际路径重新核销 (公司行为=全史复权链, 跨段, 缺槽, 闭环终点;
链可用性可复用, 窗口核销不可复用); 3 中间输出 (每臂每事件每轨道状态 + 重合组47条一致性核对 + 有效配对);
4 冻结规则统计 (H3 主: E日聚类 bootstrap 种子20261005/万次/双侧95%CI, val 主判读;
条件② val 臂A H3 绝对经济性; H5 辅助; 三档出口 + 区分 证据不足/未显示增量/有增量但绝对差).
零网络零采集; 产物 gcp_247_two_arm_v1.json/.md (RUN 新文件); 不覆盖历史件; 写前查预算.
"""
from __future__ import annotations
import sys, json, time, hashlib, glob
from pathlib import Path
from collections import Counter, defaultdict

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import dual_path_execution_v0_1_1 as D

BASE = HERE / "gcp_round_b_small_sample_v1"
OUT_JSON = BASE / "gcp_247_two_arm_v1.json"
OUT_MD = BASE / "gcp_247_two_arm_v1.md"
GATE1C_ADJ = Path(r"D:\stargazer\stargazer_ml\v5_hybrid\opportunity_v2\gate1c\feb2026_adjustment_cache_v1\adj_cache.json")
SEED = 20261005
N_BOOT = 10000
WARN = "非新盲测、非全策略表现"
NEXT_SEG = {"train": 20260309, "val": 20260610}

FROZEN = {
    "frozen_247_scope": ("gcp_247_coverage_verify.json", "f2974391fcff26e75bea450426093679bb8547e17125d21589f5108ddfb9b673"),
    "execution_contract": ("../contracts/dual_path_execution_contract_v0_1_1_final.md", "9c22c37e5e9bcc4e874f063730f19b138fac27f0c89de04d6b39cc1c0a8b852d"),
    "executor": ("../dual_path_execution_v0_1_1.py", "aee8e60bc00951a3f18975d7a01dda7c64af27a9eb7d5e1592cd8c3716aa91cb"),
    "diagnostic_v1_1": ("gcp_247_diagnostic_v1_1.json", "2ba5a89fb91794dbc9982bb20a509ba73f162ffa19b8bd1617f612ac86f2d0fa"),
    "minute_grid": ("minute_grid.json", "723c578c6b5380f038f6e0f72d16adfcabaf3df95cbf20554743b244ad10b9e9"),
    "step1_frozen": ("step1_frozen_inputs.json", "c7772fa3b5d7b605f4772a44def599929108b3eb66d64bf57d62e51ae98c4602"),
    "adj_acceptance": ("gcp_adj_cache_acceptance.json", "15007a2526da0bb49f05a8e4cc52c195f5787b1c982a4cff5b00ee9b108505db"),
}


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


def verify_hashes() -> dict:
    res, ok_all = {}, True
    for role, (rel, expect) in FROZEN.items():
        actual = sha256_file((BASE / rel).resolve())
        m = (actual == expect)
        ok_all &= m
        res[role] = {"match": m}
    man = json.loads((HERE / "contracts" / "dual_path_execution_v0_1_1_manifest.json").read_text(encoding="utf-8"))
    for f in man["files"]:
        m = sha256_file(Path("D:/stargazer") / f["path"]) == f["sha256"]
        ok_all &= m
        res[f"manifest:{f['role']}"] = {"match": m}
    acc = json.loads((BASE / "gcp_adj_cache_acceptance.json").read_text(encoding="utf-8"))
    m = sha256_file(GATE1C_ADJ) == acc["cache_sha256"]
    ok_all &= m
    res["gate1c_adj_chain"] = {"match": m}
    if not ok_all:
        raise SystemExit(f"复验失败: {[k for k,v in res.items() if not v['match']]} -> 中止")
    return res


def cluster_boot(values_by_cluster: dict, n_boot=N_BOOT, seed=SEED):
    """E 日聚类 bootstrap 均值 + 双侧 95% CI (有放回抽簇, 重复簇按次数计)."""
    keys = sorted(values_by_cluster)
    n_c = len(keys)
    if n_c < 2:
        return None
    cid, vals = [], []
    for i, k in enumerate(keys):
        for v in values_by_cluster[k]:
            cid.append(i)
            vals.append(v)
    cid = np.asarray(cid)
    vals = np.asarray(vals, dtype=float)
    idx_per_c = [np.where(cid == i)[0] for i in range(n_c)]
    rng = np.random.default_rng(seed)
    means = np.empty(n_boot)
    for b in range(n_boot):
        picks = rng.integers(0, n_c, size=n_c)
        idx = np.concatenate([idx_per_c[p] for p in picks])
        means[b] = vals[idx].mean()
    return {"n_clusters": n_c, "mean": round(float(vals.mean()), 6),
            "ci_lower": round(float(np.percentile(means, 2.5)), 6),
            "ci_upper": round(float(np.percentile(means, 97.5)), 6),
            "n_boot": n_boot, "seed": seed}


def slim_track(t):
    if not t:
        return None
    sells = t.get("sells") or []
    return {"status": t.get("status"), "reason": t.get("reason"),
            "fully_exited": t.get("fully_exited"), "remaining_shares": t.get("remaining_shares"),
            "net": t.get("net"), "label_end_ts": t.get("label_end_ts"),
            "last_fill_ts": t.get("last_fill_ts"), "deferral_d0": t.get("deferral_d0"),
            "n_sells": len(sells), "sell_kinds": sorted({s.get("kind") for s in sells}),
            "missing": [{"type": m.get("type"), "date": m.get("date")} for m in (t.get("missing_dependencies") or [])]}


def corp_check(code, E_, exit_day, chains, daily_by_code):
    import pandas as pd
    ch = chains.get(code)
    if ch and ch.get("status") == "EMPTY":
        return "CONFIRMED_NONE", "adj_chain_EMPTY"
    if ch and ch.get("status") == "OK":
        inwin = [[int(d), float(f)] for d, f in ch.get("events", []) if int(E_) <= int(d) <= int(exit_day)]
        return ("CONFIRMED_HAS", "adj_chain") if inwin else ("CONFIRMED_NONE", "adj_chain")
    g = daily_by_code.get(code)
    if g is not None and len(g):
        idx = {int(r["date"]): i for i, r in g.iterrows()}
        flags = []
        for d in range(int(E_), int(exit_day) + 1):
            if d in idx and idx[d] > 0:
                pc, prev = g.loc[idx[d], "pre_close"], g.loc[idx[d] - 1, "close"]
                if pd.notna(pc) and pd.notna(prev) and abs(float(pc) / float(prev) - 1) > 0.02:
                    flags.append(d)
        return ("CONFIRMED_HAS", "daily_preclose_jump") if flags else ("CONFIRMED_NONE", "daily_preclose_jump")
    return "UNKNOWN", "no_coverage"


def main():
    t0 = time.time()
    hv = verify_hashes()
    rows = json.loads((BASE / "gcp_247_coverage_verify.json").read_text(encoding="utf-8"))["rows"]
    chains = json.loads(GATE1C_ADJ.read_text(encoding="utf-8"))
    cal = D.load_calendar()
    ci_cal = {d: i for i, d in enumerate(cal)}

    # 分钟时间语义钉死核对 (步骤1): 09:35 bar 完成于 09:36:00; 决策 signal_slot=093500 -> 下一槽 093600
    slots = D.fill_slots(20250110)
    nxt = min(s for s in slots if s > 93500)
    assert 93500 in D.day_grid(20250110) and 93500 in slots and nxt == 93600
    semantics_check = {"bar_093500_completes_at": "09:36:00", "decision_uses_bars_completed_by": "09:35:00",
                       "next_fill_slot_after_093500": 93600, "ok": True}

    # 数据装载 (与 p2b1 重判同读取路径: deferral + p2b1 插入)
    D.MIN_DIRS.insert(0, BASE / "deferral_collect_v2" / "by_pair")
    D.MIN_DIRS.insert(1, BASE / "p2b1_collect" / "by_pair")
    codes = sorted({r["code"] for r in rows})
    import pandas as pd
    import pyarrow.dataset as ds
    COLS = ["date", "code", "open", "high", "low", "close", "volume", "pre_close", "amount", "is_st"]
    tbl = ds.dataset(sorted(glob.glob(D.DAILY_PARTS + "/part_*.parquet")), format="parquet").to_table(
        columns=COLS, filter=ds.field("code").isin(codes)).to_pandas()
    tbl["date"] = tbl["date"].astype(int)
    daily_by_code = {c: g.drop_duplicates("date").sort_values("date").reset_index(drop=True)
                     for c, g in tbl.groupby("code")}
    mbd_cache = {}
    def get_minute(code):
        if code not in mbd_cache:
            mbd_cache[code] = D.load_minute_code(code)[0]
        return mbd_cache[code]

    events, overlap_check = [], []
    n_overlap = 0
    for r in rows:
        code, E_ = r["code"], int(r["E"])
        seg = r["segment"]
        S = r.get("entry", {}).get("action_ts")
        overlap = bool(S and int(S) % 1000000 == 93500)
        if overlap:
            n_overlap += 1
        mbd = get_minute(code)
        dd = daily_by_code.get(code)
        ev = {"code": code, "E": E_, "segment": seg, "setup": r.get("setup"),
              "board": r.get("board"), "frozen_S": S, "overlap_S0935": overlap}
        for arm, sig in (("A", int(S) if S else None), ("B", int(E_) * 1000000 + 93500)):
            try:
                res = D.execute_event(code, E_, r["board"], None, r.get("setup"), signal_ts=sig,
                                      minute_by_day=mbd, daily_df=dd, cal=cal)
            except Exception as ex:
                res = {"status": "EXECUTE_ERROR", "error": repr(ex)[:200]}
            ent = res.get("entry") or {}
            slim_ent = {"status": ent.get("status"), "fill_ts": ent.get("sim_fill_ts"),
                        "fill_price": ent.get("sim_fill_price"), "shares": ent.get("shares"),
                        "D_stop": ent.get("D_stop"), "stop_price": ent.get("stop_price")}
            ev[arm] = {"exec_status": res.get("status"), "entry": slim_ent,
                       "H3": slim_track(res.get("H3")), "H5": slim_track(res.get("H5"))}
        if overlap:
            oa, ob = ev["A"], ev["B"]
            diffs = []
            for tr in ("H3", "H5"):
                ta, tb = oa.get(tr) or {}, ob.get(tr) or {}
                for f in ("status", "fully_exited", "net", "label_end_ts", "last_fill_ts",
                          "remaining_shares", "n_sells", "sell_kinds"):
                    if ta.get(f) != tb.get(f):
                        diffs.append(f"{tr}.{f}: {ta.get(f)} vs {tb.get(f)}")
            if oa["entry"] != ob["entry"]:
                diffs.append(f"entry: {oa['entry']} vs {ob['entry']}")
            overlap_check.append({"code": code, "E": E_, "identical": not diffs, "diffs": diffs[:10]})
        events.append(ev)

    # ---- 依赖按各臂实际路径重新核销 (公司行为/跨段), 组装分析用轨道表 ----
    def analyze(arm):
        """返回 {(seg,tr): {'eligible':[row], 'cross':n, 'corp_unknown':[...]}} 及逐事件配对基础."""
        out = defaultdict(lambda: {"eligible": [], "cross": 0, "corp_unknown": [], "corp_has": []})
        for ev in events:
            code, E_, seg = ev["code"], ev["E"], ev["segment"]
            a = ev[arm]
            for tr in ("H3", "H5"):
                t = a.get(tr)
                key = (seg, tr)
                if not t or not (t.get("status") == "SIM_COMPLETE" and t.get("fully_exited")
                                 and t.get("net") is not None):
                    continue
                exit_day = int((t.get("last_fill_ts") or 0) // 1000000)
                if seg in NEXT_SEG and exit_day >= NEXT_SEG[seg]:
                    out[key]["cross"] += 1
                    continue
                cs, method = corp_check(code, E_, exit_day, chains, daily_by_code)
                if cs == "UNKNOWN":
                    out[key]["corp_unknown"].append({"code": code, "E": E_, "exit_day": exit_day})
                    continue
                row = {"code": code, "E": E_, "net": float(t["net"]), "exit_day": exit_day,
                       "overlap": ev["overlap_S0935"], "corp": cs, "corp_method": method,
                       "kind": (t.get("sell_kinds") or ["NONE"])[0] if len(t.get("sell_kinds") or []) == 1 else "MIXED"}
                out[key]["eligible"].append(row)
                if cs == "CONFIRMED_HAS":
                    out[key]["corp_has"].append(row)
        return out

    anaA, anaB = analyze("A"), analyze("B")

    # ---- 有效配对 (两臂均 eligible, 以 (code,E,tr) 配对) ----
    def pairs_for(tr):
        """tr='H3'/'H5'; 返回 {seg: {'deltas':{Edate:[Δ]}, 'a_vals':{Edate:[netA]}, 'rows':[...]}}"""
        out = {}
        for seg in ("train", "val", "test"):
            ra = {(r["code"], r["E"]): r for r in anaA[(seg, tr)]["eligible"]}
            rb = {(r["code"], r["E"]): r for r in anaB[(seg, tr)]["eligible"]}
            common = sorted(set(ra) & set(rb))
            rows_p, dlt, av = [], defaultdict(list), defaultdict(list)
            for k in common:
                d = round(ra[k]["net"] - rb[k]["net"], 6)
                rows_p.append({"code": k[0], "E": k[1], "netA": ra[k]["net"], "netB": rb[k]["net"],
                               "delta": d, "overlap": ra[k]["overlap"],
                               "exitA": ra[k]["exit_day"], "exitB": rb[k]["exit_day"]})
                dlt[k[1]].append(d)
                av[k[1]].append(ra[k]["net"])
            out[seg] = {"rows": rows_p, "n_pairs": len(rows_p), "n_dates": len(dlt),
                        "delta_boot": cluster_boot(dlt), "a_boot": cluster_boot(av),
                        "overlap_pairs": sum(1 for r in rows_p if r["overlap"])}
        return out

    pairs_H3, pairs_H5 = pairs_for("H3"), pairs_for("H5")

    # ---- 两臂人口状态 (按段) ----
    def arm_pop(arm):
        out = {}
        for seg in ("train", "val", "test"):
            for tr in ("H3", "H5"):
                st = Counter()
                fill_n = 0
                for ev in events:
                    if ev["segment"] != seg:
                        continue
                    a = ev[arm]
                    if (a.get("entry") or {}).get("status") == "SIM_FILLED":
                        fill_n += 1
                    t = a.get(tr)
                    if not t:
                        st["NO_TRACK(事件未成交)"] += 1
                    else:
                        st[t.get("status") or "NONE"] += 1
                        if t.get("remaining_shares"):
                            st["(含)残仓"] += 1
                out[f"{seg}/{tr}"] = {"track_status": dict(st), "events": sum(1 for e in events if e['segment']==seg),
                                      "entries_filled": fill_n}
        return out

    popA, popB = arm_pop("A"), arm_pop("B")

    # ---- 主检验与三档出口 (val 主判读) ----
    def gate(seg_pair):
        d = seg_pair["delta_boot"]
        if d is None:
            return {"verdict": "无法确认增量", "why": "簇数<2, 无法计算 CI"}
        small = seg_pair["n_pairs"] < 30 or seg_pair["n_dates"] < 20
        cond1 = (d["mean"] > 0) and (d["ci_lower"] > 0)
        a = seg_pair["a_boot"]
        cond2 = (a is not None) and (a["mean"] > 0) and (a["ci_lower"] > 0)
        if small:
            return {"verdict": "无法确认增量(小样本)", "cond1": cond1, "cond2": cond2,
                    "small_sample": True, "why": f"n={seg_pair['n_pairs']} 日期={seg_pair['n_dates']}"}
        if cond1 and cond2:
            v = "达预注册证据门"
        elif cond1 and not cond2:
            v = "买点有改善但绝对收益仍差"
        elif (not cond1) and d["mean"] <= 0:
            v = "未显示增量(方向非正)"
        else:
            v = "无法确认增量(CI跨零)"
        return {"verdict": v, "cond1_increment": cond1, "cond2_absolute": bool(cond2),
                "small_sample": False}

    verdict_H3 = {seg: gate(pairs_H3[seg]) for seg in ("train", "val", "test")}
    verdict_primary = verdict_H3["val"]

    # ---- 与 v1.1 交叉核对 (臂A 应复现 v1.1 子集规模) ----
    v11 = json.loads((BASE / "gcp_247_diagnostic_v1_1.json").read_text(encoding="utf-8"))
    xcheck = {}
    for seg in ("train", "val", "test"):
        for tr in ("H3", "H5"):
            v11_n = v11["subset_tables"][f"{seg}/{tr}"]["distribution"]["n"]
            armA_n = len(anaA[(seg, tr)]["eligible"])
            v11_mean = v11["subset_tables"][f"{seg}/{tr}"]["distribution"]["mean"]
            armA_vals = [r["net"] for r in anaA[(seg, tr)]["eligible"]]
            armA_mean = round(float(np.mean(armA_vals)), 6) if armA_vals else None
            xcheck[f"{seg}/{tr}"] = {"v1_1_n": v11_n, "armA_n": armA_n, "n_match": v11_n == armA_n,
                                     "v1_1_mean": v11_mean, "armA_mean": armA_mean,
                                     "mean_match": v11_mean == armA_mean}

    # ---- 汇总 json ----
    out = {"phase": "GCP_247_TWO_ARM_REPLAY_V1",
           "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
           "network_requests": 0,
           "contract": "contracts/buy_point_increment_study_contract_v1.md (FROZEN; 执行授权=用户2026-10-05单独指示)",
           "authorization": {"seed": SEED, "n_boot": N_BOOT, "ci": "two-sided 95%",
                             "baseline": "B1 (E日09:35决策->09:36槽open成交)"},
           "semantics_check": semantics_check,
           "arm_difference_manifest": {
               "statement": "两臂唯一执行差异 = 传入 execute_event 的 signal_ts (臂A=冻结S/行动分支, 臂B=093500); "
                            "派生差异 = 入场价->各臂止损价(D_stop,stop_price)/退出路径/依赖终点(label_end_ts). "
                            "成本/参与率/板块规则/T+1/顺延d0+5/限价/缺槽语义完全一致 (同一未改动执行器 v0.1.1).",
               "executor_sha256": sha256_file(HERE / "dual_path_execution_v0_1_1.py"),
               "runner_script_sha256": sha256_file(HERE / "gcp_247_two_arm_replay_v1.py"),
               "note": "基线臂无需扩展执行器; 运行脚本为本执行件 (哈希登记)"},
           "hash_verification": hv,
           "n_events": len(events), "n_overlap_S0935": n_overlap,
           "overlap_consistency": {"n_identical": sum(1 for o in overlap_check if o["identical"]),
                                   "n_diff": sum(1 for o in overlap_check if not o["identical"]),
                                   "diffs": [o for o in overlap_check if not o["identical"]],
                                   "qa_note": "同一 signal_ts 走同一未改动执行器, 期望全同; 若有差异先查实现"},
           "population_by_arm": {"A": popA, "B": popB},
           "eligible_counts": {f"{arm}/{s}/{t}": len(ana[(s, t)]["eligible"])
                               for arm, ana in (("A", anaA), ("B", anaB))
                               for s in ("train", "val", "test") for t in ("H3", "H5")},
           "cross_segment": {f"{arm}/{s}/{t}": ana[(s, t)]["cross"]
                             for arm, ana in (("A", anaA), ("B", anaB))
                             for s in ("train", "val", "test") for t in ("H3", "H5")},
           "corp_unknown": {f"{arm}/{s}/{t}": ana[(s, t)]["corp_unknown"]
                            for arm, ana in (("A", anaA), ("B", anaB))
                            for s in ("train", "val", "test") for t in ("H3", "H5")},
           "corp_has": {f"{arm}/{s}/{t}": len(ana[(s, t)]["corp_has"])
                        for arm, ana in (("A", anaA), ("B", anaB))
                        for s in ("train", "val", "test") for t in ("H3", "H5")},
           "pairs_H3": {seg: {k: (v if k != "rows" else v[:0]) for k, v in segd.items()} | {"rows_n": len(segd["rows"])}
                        for seg, segd in pairs_H3.items()},
           "pairs_H5": {seg: {k: (v if k != "rows" else v[:0]) for k, v in segd.items()} | {"rows_n": len(segd["rows"])}
                        for seg, segd in pairs_H5.items()},
           "pair_rows_H3_val": pairs_H3["val"]["rows"],
           "verdict_H3_by_segment": verdict_H3,
           "verdict_primary_val_H3": verdict_primary,
           "armA_vs_v1_1_crosscheck": xcheck,
           "warning": WARN,
           "elapsed_s": round(time.time() - t0, 1)}
    # 保留全量配对行 (供审计), 分段存
    out["pair_rows_full_H3"] = {seg: pairs_H3[seg]["rows"] for seg in pairs_H3}
    out["pair_rows_full_H5"] = {seg: pairs_H5[seg]["rows"] for seg in pairs_H5}
    out.pop("pairs_H3"); out.pop("pairs_H5")
    write_checked(OUT_JSON, json.dumps(out, ensure_ascii=False, indent=1))

    # ---- md ----
    def fmt_boot(b):
        return "-" if b is None else f"{b['mean']} [{b['ci_lower']}, {b['ci_upper']}] (簇{b['n_clusters']})"

    L = ["# 两臂复放与买点增量对照 v1 (零网络, 按 FROZEN 合同)", "",
         f"- 生成: {out['generated_at']} | 复验全过 | 用时 {out['elapsed_s']}s | {WARN}",
         "- 臂 A = 冻结 S1/S2 买点; 臂 B = B1 (E 日 09:35 决策 → 09:36 槽 open). "
         "两臂唯一差异 = 决策时点（派生：入场价→各臂止损价/退出路径/依赖终点）；执行器 v0.1.1 未改动。", "",
         "## 1. 复验与分钟语义", "",
         f"- 哈希复验: {sum(1 for v in hv.values() if v.get('match'))}/{len(hv)} 全过 (合同8类+manifest5+链登记值)",
         f"- 分钟语义: 09:35 bar 于 09:36:00 完成; 决策用 ≤09:35:00 已完成 bar; 下一成交槽 09:36 open (核对 ok)", "",
         "## 2. 人口状态 (每臂 轨道状态计数 / 事件数 / 入场成交数)", "",
         "| 臂 | 段 | 轨道 | 事件数 | 入场成交 | 状态计数 |", "|---|---|---|---|---|---|"]
    for arm in ("A", "B"):
        for seg in ("train", "val", "test"):
            for tr in ("H3", "H5"):
                p = out["population_by_arm"][arm][f"{seg}/{tr}"]
                L.append(f"| {arm} | {seg} | {tr} | {p['events']} | {p['entries_filled']} | {p['track_status']} |")
    L += ["", "## 3. 分析可用轨道 (复核销后: 扣除跨段/公司行为UNKNOWN) 与两臂不对称", "",
          "| 臂 | 段 | 轨道 | eligible | 跨段隔离 | 公司行为UNKNOWN | 公司行为有 |", "|---|---|---|---|---|---|---|"]
    for arm in ("A", "B"):
        for seg in ("train", "val", "test"):
            for tr in ("H3", "H5"):
                k = f"{arm}/{seg}/{tr}"
                L.append(f"| {arm} | {seg} | {tr} | {out['eligible_counts'][k]} | {out['cross_segment'][k]} | "
                         f"{len(out['corp_unknown'][k])} | {out['corp_has'][k]} |")
    oc = out["overlap_consistency"]
    L += ["", "## 4. 重合组一致性 (47 条, 逐条核对成交数量/止损/退出路径/依赖终点)", "",
          f"- 全同: {oc['n_identical']}/47 | 有差异: {oc['n_diff']}/47"]
    for d in oc["diffs"]:
        L.append(f"  - 差异! {d['code']}@{d['E']}: {d['diffs']}")
    if oc["n_diff"] == 0:
        L.append("- QA 结论: 全同为预期（同一 signal_ts 走同一执行器）；无实现问题，无一条被解释成买点增量。")
    else:
        L.append("- QA 结论: 存在差异 → 按合同先查实现，本报告不解释成买点增量。")
    L += ["", "## 5. 有效配对与日期覆盖", "",
          "| 轨道 | 段 | 配对数 | 日期簇 | 其中重合组配对 | Δ bootstrap 均值 [95%CI] | 臂A绝对 net bootstrap |", "|---|---|---|---|---|---|---|"]
    for tr, pd_ in (("H3", pairs_H3), ("H5", pairs_H5)):
        for seg in ("train", "val", "test"):
            s = pd_[seg]
            L.append(f"| {tr} | {seg} | {s['n_pairs']} | {s['n_dates']} | {s['overlap_pairs']} | "
                     f"{fmt_boot(s['delta_boot'])} | {fmt_boot(s['a_boot'])} |")
    L += ["", "## 6. H3 主检验 (val 主判读; 种子20261005/万次/双侧95%CI)", "",
          f"- val Δ: {fmt_boot(pairs_H3['val']['delta_boot'])}",
          f"- val 臂A 绝对经济性: {fmt_boot(pairs_H3['val']['a_boot'])}",
          f"- 判定: **{verdict_primary['verdict']}** ({json.dumps({k: v for k, v in verdict_primary.items() if k != 'verdict'}, ensure_ascii=False)})",
          f"- train(解释): {verdict_H3['train']['verdict']} | test(支持反证,不计票): {verdict_H3['test']['verdict']}",
          "- 区分: 证据不足(配对/日期不够或CI跨零) ≠ 买点未显示增量(方向非正) ≠ 有增量但绝对收益仍差", "",
          "## 7. H5 辅助 (不入主裁决)", ""]
    for seg in ("train", "val", "test"):
        s = pairs_H5[seg]
        L.append(f"- {seg}: Δ {fmt_boot(s['delta_boot'])} ; 臂A {fmt_boot(s['a_boot'])}")
    L += ["", "## 8. 左尾披露 (val H3 配对 Δ 最差 min(10,n))", ""]
    worst = sorted(pairs_H3["val"]["rows"], key=lambda r: r["delta"])[:min(10, len(pairs_H3["val"]["rows"]))]
    for r in worst:
        L.append(f"- {r['code']}@{r['E']} Δ={r['delta']} (A={r['netA']} B={r['netB']})"
                 + (" [重合组]" if r["overlap"] else ""))
    L += ["", "## 9. 臂A 与 v1.1 交叉核对 (应复现子集)", ""]
    allok = all(v["n_match"] and v["mean_match"] for v in xcheck.values())
    for k, v in xcheck.items():
        L.append(f"- {k}: v1.1 n={v['v1_1_n']} vs 臂A n={v['armA_n']} {'OK' if v['n_match'] else 'MISMATCH'}; "
                 f"均值 {'OK' if v['mean_match'] else 'MISMATCH'} ({v['armA_mean']})")
    L += ["", f"- 总判定: {'交叉核对全过' if allok else '存在不一致, 须 QA'}", "",
          "## 声明", "",
          f"- {WARN}; 事后条件化诊断; 无 PASS/FAIL 以外推广声明; 未调参/未训练/未自动进个性层.", ""]
    write_checked(OUT_MD, "\n".join(L))

    print(f"two-arm done ({time.time()-t0:.1f}s)")
    print("overlap:", oc["n_identical"], "/", n_overlap, "identical")
    for seg in ("train", "val", "test"):
        s3 = pairs_H3[seg]
        print(f" H3 {seg}: pairs={s3['n_pairs']} dates={s3['n_dates']} dBoot={s3['delta_boot']} aBoot={s3['a_boot']}")
    print("verdicts:", {s: v['verdict'] for s, v in verdict_H3.items()})
    print("v1.1 xcheck all ok:", allok)


if __name__ == "__main__":
    main()
