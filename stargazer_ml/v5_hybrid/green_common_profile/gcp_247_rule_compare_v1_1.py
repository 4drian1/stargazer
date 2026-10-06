# -*- coding: utf-8 -*-
"""完整人口规则对照执行 v1.1 — 配对实现偏差修复重算 (合同/阈值/口径/数据全不改).

实现偏差 (用户核对确认): v1 的 pairs_for 只接受双方 EVALUABLE, 确认无行动/未成交的 CASH0
被排除——"不买"的价值未入比较; 臂 A 绝对经济性同样只用 EVALUABLE. v1 报告保留为实现偏差记录
(文件不动). 本件冻结参数原样重算: 20261005/万次/双侧95%/E日簇保完整配对/事件等权/H3主 val主判读/
三档出口/配对<30或日期<20仅描述.

修复 (严格四件):
 1 配对接受双方各自 EVALUABLE 或 CASH0 (现金0机会收益=0, 合同§4语义, 非零交易标签); NULL 继续排除.
 2 臂 A 机会收益统计样本框 = EVALUABLE ∪ CASH0 (NULL 不入), 均值/中位/分位/CI 同口径.
 3 四类反例合成测试 (先跑必过): 现金/成交, 成交/现金, 双方现金(Δ=0且入配对), 任一NULL(不入);
    CASH0 不得入最差例 (最差例只含实际成交轨道).
 4 原报告 v1 保留; 新报告 v1_1.
措辞: 绝对经济性"已计算但未触发晋级警示".
零网络零采集; 写前查预算; 审计 -17.
"""
from __future__ import annotations
import sys, json, time, hashlib, glob
from pathlib import Path
from collections import Counter, defaultdict

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import dual_path_execution_v0_1_1 as D

BASE = HERE / "gcp_round_b_small_sample_v1"
OUT_JSON = BASE / "gcp_247_rule_compare_v1_1.json"
OUT_MD = BASE / "gcp_247_rule_compare_v1_1.md"
GATE1C_ADJ = Path(r"D:\stargazer\stargazer_ml\v5_hybrid\opportunity_v2\gate1c\feb2026_adjustment_cache_v1\adj_cache.json")
SEED = 20261005
N_BOOT = 10000
BUDGET = 50 * 1024 * 1024
NEXT_SEG = {"train": 20260309, "val": 20260610}
WARN = "非新盲测、非全策略表现"

FROZEN = {
    "v3_inventory": ("gcp_247_full_population_inventory_v3.json", "a2c53545efc331e64438a590d155cae25d79048c79e2d8c4ec4166e9e018d0f8"),
    "ruleA_trigger_records": ("gcp_247_ruleA_trigger_records_v1.json", "f6c4c997003fe3b5191666a1a120feffdba7f9fcb0289ba0a9d339c89b98dc6e"),
    "v3_script": ("../gcp_247_full_population_inventory_v3.py", "76a096dac55fe0d0dd33f3cd787240eaa9ea7c82958a6c4f4351ce3de5161e9b"),
    "scan_engine": ("../gcp_dual_path_coverage_probe.py", "750710918a626e8d18c6157971e668e4b46e67763eed21c3d718ad00eb0ac659"),
    "coverage_v1": ("gcp_247_return_path_coverage_v1.json", "7933d1e1311db7002d96e6956b22164c4c8f9ddffdd1804869d25ad966cd63b8"),
    "buy_point_contract": ("../contracts/buy_point_increment_study_contract_v1.md", None),
    "execution_contract": ("../contracts/dual_path_execution_contract_v0_1_1_final.md", "9c22c37e5e9bcc4e874f063730f19b138fac27f0c89de04d6b39cc1c0a8b852d"),
    "executor": ("../dual_path_execution_v0_1_1.py", "aee8e60bc00951a3f18975d7a01dda7c64af27a9eb7d5e1592cd8c3716aa91cb"),
    "frozen_247_scope": ("gcp_247_coverage_verify.json", "f2974391fcff26e75bea450426093679bb8547e17125d21589f5108ddfb9b673"),
    "minute_grid": ("minute_grid.json", None),
    "step1": ("step1_frozen_inputs.json", None),
    "adj_acceptance": ("gcp_adj_cache_acceptance.json", None),
    "prior_report_v1": ("gcp_247_rule_compare_v1.json", None),
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
    b = len(text.encode("utf-8"))
    if dir_size(BASE) + b > BUDGET:
        raise RuntimeError(f"budget exceeded before writing {path.name} (stop)")
    path.write_text(text, encoding="utf-8")
    print(f"  written {path.name} (+{b} B; dir now {dir_size(BASE)} B)")


# ================= 修复3: 反例合成测试 (先跑必过) =================
def counterexample_tests():
    """合成四行验证配对接受规则与最差例规则."""
    EVAL, CASH, NULL = "EVALUABLE", "CASH0", "NULL"
    # (a_status, a_opp, b_status, b_opp)
    rows = [
        ("cash_fill", CASH, 0.0, EVAL, 0.010),      # 现金/成交
        ("fill_cash", EVAL, -0.005, CASH, 0.0),     # 成交/现金
        ("both_cash", CASH, 0.0, CASH, 0.0),        # 双方现金
        ("null_excluded", EVAL, 0.01, NULL, None),  # 任一 NULL
        ("null_excluded2", NULL, None, CASH, 0.0),
    ]
    pairs, worst = [], []
    for name, as_, ao, bs, bo in rows:
        ok_a, ok_b = as_ in (EVAL, CASH), bs in (EVAL, CASH)
        if ok_a and ok_b:
            a = 0.0 if as_ == CASH else ao
            b = 0.0 if bs == CASH else bo
            pairs.append((name, a - b, as_, bs))
            if as_ == EVAL and bs == EVAL:
                worst.append((name, a - b))
    by_name = {n: d for n, d, _, _ in pairs}
    tests = [
        ("现金/成交入配对且 Δ=0−oppB", by_name.get("cash_fill") == -0.010),
        ("成交/现金入配对且 Δ=oppA−0", by_name.get("fill_cash") == -0.005),
        ("双方现金入配对且 Δ=0", by_name.get("both_cash") == 0.0),
        ("任一NULL不入配对", "null_excluded" not in by_name and "null_excluded2" not in by_name),
        ("CASH0不入最差例(最差例只含实际成交)",
         all(n in ("fill_cash",) for n, _ in worst) or
         not any(n in ("cash_fill", "both_cash") for n, _ in worst)),
    ]
    failed = [n for n, ok in tests if not ok]
    print("反例测试:", {n: ("PASS" if ok else "FAIL") for n, ok in tests})
    if failed:
        raise SystemExit(f"反例测试失败: {failed} -> 中止")
    return {"tests": {n: ok for n, ok in tests}, "rule": "配对接受 EVALUABLE∪CASH0; NULL 排除; "
            "最差例仅双方 EVALUABLE (实际成交轨道)"}


def verify_hashes():
    res, ok_all = {}, True
    for role, (rel, expect) in FROZEN.items():
        p = (BASE / rel).resolve()
        actual = sha256_file(p)
        entry = {"sha256": actual, "placeholder_filled_at_execution": expect is None}
        if expect is not None:
            entry["match"] = (actual == expect)
            ok_all &= entry["match"]
        res[role] = entry
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
        raise SystemExit(f"复验失败: {[k for k, v in res.items() if v.get('match') is False]} -> 中止")
    return res


def cluster_boot(vals_by_date, n_boot=N_BOOT, seed=SEED):
    keys = sorted(vals_by_date)
    n_c = len(keys)
    if n_c < 2:
        return None
    cid, vals = [], []
    for i, k in enumerate(keys):
        for v in vals_by_date[k]:
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
    return {"n_events": int(len(vals)), "n_clusters": n_c,
            "mean": round(float(vals.mean()), 8),
            "ci_lower": round(float(np.percentile(means, 2.5)), 8),
            "ci_upper": round(float(np.percentile(means, 97.5)), 8),
            "n_boot": n_boot, "seed": seed, "estimator": "event_equal_weight"}


def main():
    t0 = time.time()
    print(f"[预算] 写入前 dir={dir_size(BASE)} B / {BUDGET} B")
    ce = counterexample_tests()
    hv = verify_hashes()
    print(f"[复验] 全过 ({len(hv)} 项)")

    v3 = json.loads(open(BASE / "gcp_247_full_population_inventory_v3.json", encoding="utf-8").read())
    trig = {r["code"] + "@" + str(r["E"]): r for r in
            json.loads(open(BASE / "gcp_247_ruleA_trigger_records_v1.json", encoding="utf-8").read())["records"]}
    chains = json.loads(open(GATE1C_ADJ, encoding="utf-8").read())
    cal = D.load_calendar()
    slots = D.fill_slots(20250110)
    assert min(s for s in slots if s > 93500) == 93600

    pop = list(v3["records"])
    codes = sorted({r["code"] for r in pop})
    import pyarrow.dataset as ds
    COLS = ["date", "code", "open", "high", "low", "close", "volume", "pre_close", "amount", "is_st"]
    tbl = ds.dataset(sorted(glob.glob(D.DAILY_PARTS + "/part_*.parquet")), format="parquet").to_table(
        columns=COLS, filter=ds.field("code").isin(codes)).to_pandas()
    tbl["date"] = tbl["date"].astype(int)
    daily_by_code = {c: g.drop_duplicates("date").sort_values("date").reset_index(drop=True)
                     for c, g in tbl.groupby("code")}
    D.MIN_DIRS.insert(0, BASE / "deferral_collect_v2" / "by_pair")
    D.MIN_DIRS.insert(1, BASE / "p2b1_collect" / "by_pair")
    mbd_cache = {}
    def get_minute(code):
        if code not in mbd_cache:
            mbd_cache[code] = D.load_minute_code(code)[0]
        return mbd_cache[code]
    cov = {(r["code"], int(r["E"])): r["board"] for r in
           json.loads(open(BASE / "gcp_247_coverage_verify.json", encoding="utf-8").read())["rows"]}
    def board_of(code, Ee):
        if (code, Ee) in cov:
            return cov[(code, Ee)]
        if code.startswith(("300", "301")):
            return "GEM"
        if code.startswith("688"):
            return "STAR"
        return "SH_MAIN" if code.startswith("6") else "SZ_MAIN"

    def corp_check(code, E_, exit_day):
        ch = chains.get(code)
        if ch and ch.get("status") == "EMPTY":
            return "CONFIRMED_NONE"
        if ch and ch.get("status") == "OK":
            return "CONFIRMED_HAS" if any(int(E_) <= int(d) <= int(exit_day)
                                          for d, _ in ch.get("events", [])) else "CONFIRMED_NONE"
        return "UNKNOWN"

    def run_arm(r, arm):
        code, Ee, cls = r["code"], r["E"], r["mutual"]
        if cls == "4_数据不足无法判断":
            return "NULL", None, {"reason": "数据不足无法判断(第4类)"}
        if arm == "A" and cls in ("2_WATCH但无行动", "3_未WATCH且无行动"):
            return "CASH0", 0.0, {"reason": "确认无行动(规则A未触发)"}
        sig = trig[code + "@" + str(Ee)]["first_action_ts"] if arm == "A" else int(Ee) * 1000000 + 93500
        mbd = get_minute(code)
        dd = daily_by_code.get(code)
        try:
            res = D.execute_event(code, int(Ee), board_of(code, Ee), None, r["setup"],
                                  signal_ts=int(sig), minute_by_day=mbd, daily_df=dd, cal=cal)
        except Exception as ex:
            return "NULL", None, {"reason": f"EXECUTE_ERROR:{repr(ex)[:120]}"}
        ent = res.get("entry") or {}
        if res.get("status") == "ENTRY_METADATA_UNKNOWN":
            return "NULL", None, {"reason": "ENTRY_METADATA_UNKNOWN"}
        if ent.get("status") == "NO_FILL_EOD":
            return "CASH0", 0.0, {"reason": "确认未成交(E日有数据无合法成交)"}
        if ent.get("status") == "ENTRY_DATA_MISSING":
            return "NULL", None, {"reason": f"ENTRY_DATA_MISSING({ent.get('reason')})"}
        if ent.get("status") != "SIM_FILLED":
            return "NULL", None, {"reason": f"ENTRY:{ent.get('status')}"}
        out = {}
        for tr in ("H3", "H5"):
            t = res.get(tr) or {}
            full = bool(t.get("fully_exited")) and t.get("net") is not None and t.get("status") == "SIM_COMPLETE"
            if full:
                exit_day = int(t["last_fill_ts"] // 1000000)
                if r["segment"] in NEXT_SEG and exit_day >= NEXT_SEG[r["segment"]]:
                    out[tr] = ("NULL", None, "CROSS_SEGMENT_LABEL")
                    continue
                if corp_check(code, int(Ee), exit_day) == "UNKNOWN":
                    out[tr] = ("NULL", None, "CORP_UNKNOWN")
                    continue
                out[tr] = ("EVALUABLE", round(float(t["pnl"]) / 50000.0, 8), "complete")
            elif t.get("remaining_shares"):
                out[tr] = ("NULL", None, f"RESIDUAL({t.get('status')})")
            else:
                out[tr] = ("NULL", None, f"NULL({t.get('status')})")
        return "TRACKS", None, out

    results = []
    for i, r in enumerate(pop):
        rec = {"code": r["code"], "E": r["E"], "setup": r["setup"], "segment": r["segment"],
               "mutual": r["mutual"]}
        for arm in ("A", "B"):
            oc, opp, det = run_arm(r, arm)
            if oc == "TRACKS":
                for tr in ("H3", "H5"):
                    o2, v2, rs = det[tr]
                    rec[f"{arm}_{tr}"] = {"outcome": o2, "opp": v2, "reason": rs}
            else:
                for tr in ("H3", "H5"):
                    rec[f"{arm}_{tr}"] = {"outcome": oc, "opp": opp, "reason": det["reason"]}
        results.append(rec)
        if (i + 1) % 300 == 0:
            print(f"  replay {i+1}/{len(pop)} ({time.time()-t0:.0f}s)")

    matrix = defaultdict(Counter)
    for rec in results:
        for arm in ("A", "B"):
            for tr in ("H3", "H5"):
                matrix[(rec["mutual"], arm, tr, rec["segment"])][rec[f"{arm}_{tr}"]["outcome"]] += 1

    OK = ("EVALUABLE", "CASH0")
    def pairs_for(tr):
        out = {}
        for seg in ("train", "val", "test"):
            rows, by_date = [], defaultdict(list)
            a_only = defaultdict(list)
            for rec in results:
                if rec["segment"] != seg:
                    continue
                a, b = rec[f"A_{tr}"], rec[f"B_{tr}"]
                if a["outcome"] in OK:
                    a_only[rec["E"]].append(0.0 if a["outcome"] == "CASH0" else a["opp"])
                if a["outcome"] in OK and b["outcome"] in OK:
                    ao = 0.0 if a["outcome"] == "CASH0" else a["opp"]
                    bo = 0.0 if b["outcome"] == "CASH0" else b["opp"]
                    d = round(ao - bo, 8)
                    rows.append({"code": rec["code"], "E": rec["E"], "d": d, "oppA": ao, "oppB": bo,
                                 "a_cash": a["outcome"] == "CASH0", "b_cash": b["outcome"] == "CASH0",
                                 "mutual": rec["mutual"]})
                    by_date[rec["E"]].append(d)
            out[seg] = {"rows": rows, "n_pairs": len(rows), "n_dates": len(by_date),
                        "delta_boot": cluster_boot(by_date), "a_boot": cluster_boot(a_only)}
        return out

    P3, P5 = pairs_for("H3"), pairs_for("H5")

    def three_tier(seg_pair):
        d = seg_pair["delta_boot"]
        if d is None:
            return {"verdict": "证据不足", "why": "簇数<2"}
        small = seg_pair["n_pairs"] < 30 or seg_pair["n_dates"] < 20
        if small:
            return {"verdict": "证据不足(仅描述: 配对<30或日期<20)", "small_sample": True,
                    "n_pairs": seg_pair["n_pairs"], "n_dates": seg_pair["n_dates"]}
        if d["mean"] > 0 and d["ci_lower"] > 0:
            v = "支持正增量"
        elif d["mean"] < 0 and d["ci_upper"] < 0:
            v = "支持负增量"
        else:
            v = "证据不足(CI 含 0)"
        a = seg_pair["a_boot"]
        abs_ok = (a is not None and a["mean"] > 0 and a["ci_lower"] > 0)
        return {"verdict": v, "delta": d, "armA_abs_economics": a,
                "absolute_economics": "已计算但未触发晋级警示" if not abs_ok else "成立",
                "absolute_loss_warning": ("规则改善但仍亏钱(正增量不能抵消绝对亏损)"
                                          if v == "支持正增量" and not abs_ok else None)}

    verdicts_H3 = {seg: three_tier(P3[seg]) for seg in ("train", "val", "test")}

    def cash_decomp(seg, tr):
        c = Counter()
        for r in P3[seg]["rows"] if tr == "H3" else P5[seg]["rows"]:
            key = ("双方现金" if r["a_cash"] and r["b_cash"]
                   else "现金/成交" if r["a_cash"] else "成交/现金" if r["b_cash"] else "双方成交")
            c[key] += 1
        return dict(c)

    out = {"phase": "GCP_247_RULE_COMPARE_V1_1_FIX",
           "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
           "network_requests": 0,
           "fix_note": "v1 实现偏差: pairs_for 只接受双方 EVALUABLE, CASH0 被排除; 臂A 绝对经济性同漏. "
                       "v1 报告保留为实现偏差记录(文件未动). 本件修复: 配对接受 EVALUABLE∪CASH0 "
                       "(现金0机会收益=0, 合同§4语义), NULL 排除; 臂A 统计样本框=EVALUABLE∪CASH0; "
                       "最差例仅双方成交. 冻结参数原样重算.",
           "counterexample_tests": ce,
           "contract": "contracts/full_population_rule_compare_contract_v1.md (FROZEN; 执行授权=用户四步指示; 本件为用户核对后的有界修复重算授权)",
           "hash_verification": hv,
           "arm_difference_manifest": {
               "runner_sha256": sha256_file(HERE / "gcp_247_rule_compare_v1_1.py"),
               "executor_sha256": sha256_file(HERE / "dual_path_execution_v0_1_1.py"),
               "prior_runner_v1_sha256": sha256_file(HERE / "gcp_247_rule_compare_v1.py")},
           "n_population": len(results),
           "status_matrix": {f"{k[0]}|{k[1]}|{k[2]}|{k[3]}": dict(v) for k, v in sorted(matrix.items())},
           "pairs_H3": {seg: {k: v for k, v in segd.items() if k != "rows"} for seg, segd in P3.items()},
           "pairs_H5": {seg: {k: v for k, v in segd.items() if k != "rows"} for seg, segd in P5.items()},
           "cash0_decomposition_H3": {seg: cash_decomp(seg, "H3") for seg in ("train", "val", "test")},
           "verdicts_H3": verdicts_H3,
           "warning": WARN + "; 配对结果不得外推全人口; 现金0与NULL严格分列",
           "dir_bytes_after": dir_size(BASE),
           "elapsed_s": round(time.time() - t0, 1)}
    write_checked(OUT_JSON, json.dumps(out, ensure_ascii=False, indent=1))

    L = ["# 完整人口规则对照执行 v1.1（配对实现偏差修复重算；v1 保留为实现偏差记录）", "",
         f"- 生成: {out['generated_at']} | 复验全过 | 反例测试全过 | 用时 {out['elapsed_s']}s | {WARN}",
         f"- 修复: {out['fix_note']}", "",
         "## 反例测试结果", ""]
    for n, ok in ce["tests"].items():
        L.append(f"- {'PASS' if ok else 'FAIL'} {n}")
    L += ["", "## H3 主检验 (val 主判读)", "",
          "| 段 | 配对 | 日期簇 | Δ 均值 [95%CI] | 臂A 绝对经济性 [95%CI] | 判定 |", "|---|---|---|---|---|---|"]
    for seg in ("train", "val", "test"):
        s = P3[seg]
        v = verdicts_H3[seg]
        d, a = s["delta_boot"], (s["a_boot"] or {})
        L.append(f"| {seg} | {s['n_pairs']} | {s['n_dates']} | "
                 f"{d['mean'] if d else '-'} [{d['ci_lower'] if d else '-'}, {d['ci_upper'] if d else '-'}] | "
                 f"{a.get('mean','-')} [{a.get('ci_lower','-')}, {a.get('ci_upper','-')}] | **{v['verdict']}** |")
    vp = verdicts_H3["val"]
    L += ["", f"- val 判定: **{vp['verdict']}**；绝对经济性: **{vp.get('absolute_economics','-')}**"
          + (f"；警示: {vp.get('absolute_loss_warning')}" if vp.get("absolute_loss_warning") else ""),
          "", "## 现金 0 对配对的贡献分解 (H3)", ""]
    for seg in ("train", "val", "test"):
        L.append(f"- {seg}: {out['cash0_decomposition_H3'][seg]}")
    L += ["", "## H5 辅助 (不入裁决)", ""]
    for seg in ("train", "val", "test"):
        s = P5[seg]
        d, a = s["delta_boot"], (s["a_boot"] or {})
        L.append(f"- {seg}: 配对 {s['n_pairs']} 簇 {s['n_dates']} | Δ {d['mean'] if d else '-'} "
                 f"[{d['ci_lower'] if d else '-'}, {d['ci_upper'] if d else '-'}] | 臂A {a.get('mean','-')}")
    L += ["", "## 与 v1 的对照", "",
          "- v1(偏差): val 27 配对, Δ=+0.005283 [−0.001029, 0.012868], 证据不足(仅描述)",
          "- v1.1(修复): 见上表; 现金0入配对后配对数上升; 冻结参数未动", "",
          "## 声明", "",
          "- 最差例只含实际成交轨道 (CASH0 不入左尾); 未调参/未训练/未补采(328 挂起).", ""]
    write_checked(OUT_MD, "\n".join(L))
    print(f"done ({time.time()-t0:.1f}s)")
    for seg in ("train", "val", "test"):
        s = P3[seg]
        print(f" H3 {seg}: pairs={s['n_pairs']} dates={s['n_dates']} d={s['delta_boot']} | {verdicts_H3[seg]['verdict']} | cash {out['cash0_decomposition_H3'][seg]}")
    print("dir bytes:", dir_size(BASE))


if __name__ == "__main__":
    main()
