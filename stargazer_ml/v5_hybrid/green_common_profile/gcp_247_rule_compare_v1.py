# -*- coding: utf-8 -*-
"""完整人口规则对照执行 (合同 v1 FROZEN 单独执行授权: 用户 2026-10-05 四步指示).

第1步 复验: 合同 §8 全部输入哈希 (填实项+占位项实测) + manifest 5 + 分钟语义 + 两臂唯一差异.
第2步 全人口两臂复放: 1,013 人口; 臂A=S1/S2 首触发 (187 条取自触发记录件, 其余确认无行动);
  臂B=09:35 基线 (E*1e6+93500); H3/H5 双轨; 依赖按各臂实际路径重新核销 (公司行为链/跨段/缺槽/闭环终点).
  机会收益口径 (§4): 无行动/确认未成交=0(现金, 非零交易标签); 全闭环=pnl/50000; 数据不足/元数据未知/残仓=NULL.
  0 与 NULL 全程分列.
第3步 全人口状态同报: 互斥四类 × 臂 × 轨道 状态矩阵 (可评价/现金0/NULL), 配对结果不得外推.
第4步 主检验: H3 机会收益差 Δ=A−B (双方可评价配对); bootstrap 种子20261005/万次/双侧95%/
  E日簇重抽保完整配对/事件等权均值; val 主判读三档出口; 臂A 绝对经济性独立警示; H5 辅助;
  train 解释/test 支持反证; 配对<30 或日期<20 仅描述.
运行三守住: 全人口同报 / 0 与 NULL 严格分列 / 每写前核验预算.
零网络零采集; 产物 gcp_247_rule_compare_v1.json/.md (RUN 新文件); 审计 -16.
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
OUT_JSON = BASE / "gcp_247_rule_compare_v1.json"
OUT_MD = BASE / "gcp_247_rule_compare_v1.md"
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
        raise RuntimeError(f"budget exceeded before writing {path.name} (stop, no truncation)")
    path.write_text(text, encoding="utf-8")
    print(f"  written {path.name} (+{b} B; dir now {dir_size(BASE)} B)")


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
        raise SystemExit(f"复验失败: {[k for k, v in res.items() if v.get('match') is False]} -> 中止报审计")
    return res


def cluster_boot(pairs_by_date: dict, n_boot=N_BOOT, seed=SEED):
    """E 日簇有放回重抽, 同簇保留完整配对, 事件等权均值; 双侧 95% CI."""
    keys = sorted(pairs_by_date)
    n_c = len(keys)
    if n_c < 2:
        return None
    cid, vals = [], []
    for i, k in enumerate(keys):
        for v in pairs_by_date[k]:
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
    hv = verify_hashes()
    print(f"[复验] 全过 ({len(hv)} 项)")

    v3 = json.loads(open(BASE / "gcp_247_full_population_inventory_v3.json", encoding="utf-8").read())
    trig = {r["code"] + "@" + str(r["E"]): r for r in
            json.loads(open(BASE / "gcp_247_ruleA_trigger_records_v1.json", encoding="utf-8").read())["records"]}
    chains = json.loads(open(GATE1C_ADJ, encoding="utf-8").read())
    cal = D.load_calendar()
    ci_cal = {d: i for i, d in enumerate(cal)}

    slots = D.fill_slots(20250110)
    assert min(s for s in slots if s > 93500) == 93600

    pop = [r for r in v3["records"]]  # 1,013
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

    def corp_check(code, E_, exit_day):
        ch = chains.get(code)
        if ch and ch.get("status") == "EMPTY":
            return "CONFIRMED_NONE"
        if ch and ch.get("status") == "OK":
            return "CONFIRMED_HAS" if any(int(E_) <= int(d) <= int(exit_day)
                                          for d, _ in ch.get("events", [])) else "CONFIRMED_NONE"
        return "UNKNOWN"

    def run_arm(r, arm):
        """返回 (outcome, opp, detail). outcome: EVALUABLE / CASH0 / NULL."""
        code, Ee, cls = r["code"], r["E"], r["mutual"]
        if cls == "4_数据不足无法判断":
            return "NULL", None, {"reason": "数据不足无法判断(第4类)"}
        if arm == "A" and cls in ("2_WATCH但无行动", "3_未WATCH且无行动"):
            return "CASH0", 0.0, {"reason": "确认无行动(规则A未触发)"}
        sig = None
        if arm == "A":
            sig = trig[code + "@" + str(Ee)]["first_action_ts"]
        else:
            sig = int(Ee) * 1000000 + 93500
        mbd = get_minute(code)
        dd = daily_by_code.get(code)
        try:
            res = D.execute_event(code, int(Ee), r["board"] if "board" in r else board_of(code, Ee),
                                  None, r["setup"], signal_ts=int(sig),
                                  minute_by_day=mbd, daily_df=dd, cal=cal)
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
            st = t.get("status")
            full = bool(t.get("fully_exited")) and t.get("net") is not None and st == "SIM_COMPLETE"
            if full:
                exit_day = int(t["last_fill_ts"] // 1000000)
                seg = r["segment"]
                if seg in NEXT_SEG and exit_day >= NEXT_SEG[seg]:
                    out[tr] = ("NULL", None, "CROSS_SEGMENT_LABEL")
                    continue
                cs = corp_check(code, int(Ee), exit_day)
                if cs == "UNKNOWN":
                    out[tr] = ("NULL", None, "CORP_UNKNOWN")
                    continue
                opp = round(float(t["pnl"]) / 50000.0, 8)
                out[tr] = ("EVALUABLE", opp, f"complete; corp={cs}")
            elif t.get("remaining_shares"):
                out[tr] = ("NULL", None, f"RESIDUAL({st})")
            else:
                out[tr] = ("NULL", None, f"NULL({st})")
        return "TRACKS", None, out

    # board 映射 (v3 records 无 board 字段; 从 247 coverage 或 251 cache 取, 缺失按代码段推断)
    cov = {(r["code"], int(r["E"])): r["board"] for r in
           json.loads(open(BASE / "gcp_247_coverage_verify.json", encoding="utf-8").read())["rows"]}
    def board_of(code, Ee):
        if (code, Ee) in cov:
            return cov[(code, Ee)]
        if code.startswith(("300", "301")):
            return "GEM"
        if code.startswith("688"):
            return "STAR"
        if code.startswith("6"):
            return "SH_MAIN"
        return "SZ_MAIN"

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
        if (i + 1) % 200 == 0:
            print(f"  replay {i+1}/{len(pop)} ({time.time()-t0:.0f}s)")

    # ---------- 第3步: 状态矩阵 ----------
    matrix = defaultdict(Counter)
    for rec in results:
        for arm in ("A", "B"):
            for tr in ("H3", "H5"):
                matrix[(rec["mutual"], arm, tr, rec["segment"])][rec[f"{arm}_{tr}"]["outcome"]] += 1

    # ---------- 第4步: 主检验 ----------
    def pairs_for(tr):
        out = {}
        for seg in ("train", "val", "test"):
            rows, by_date = [], defaultdict(list)
            for rec in results:
                if rec["segment"] != seg:
                    continue
                a, b = rec[f"A_{tr}"], rec[f"B_{tr}"]
                if a["outcome"] == "EVALUABLE" and b["outcome"] == "EVALUABLE":
                    d = round(a["opp"] - b["opp"], 8)
                    rows.append({"code": rec["code"], "E": rec["E"], "d": d,
                                 "oppA": a["opp"], "oppB": b["opp"], "mutual": rec["mutual"]})
                    by_date[rec["E"]].append(d)
            a_only = defaultdict(list)
            for rec in results:
                if rec["segment"] != seg:
                    continue
                a = rec[f"A_{tr}"]
                if a["outcome"] == "EVALUABLE":
                    a_only[rec["E"]].append(a["opp"])
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
        abs_warn = not (a is not None and a["mean"] > 0 and a["ci_lower"] > 0)
        return {"verdict": v, "delta": d, "armA_abs_economics": a,
                "absolute_loss_warning": ("规则改善但仍亏钱(正增量不能抵消绝对亏损)" if v == "支持正增量" and abs_warn
                                          else ("绝对经济性不成立" if abs_warn else None))}

    verdicts_H3 = {seg: three_tier(P3[seg]) for seg in ("train", "val", "test")}

    out = {"phase": "GCP_247_RULE_COMPARE_V1",
           "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
           "network_requests": 0,
           "contract": "contracts/full_population_rule_compare_contract_v1.md (FROZEN; 执行授权=用户2026-10-05四步指示)",
           "hash_verification": hv,
           "arm_difference_manifest": {
               "statement": "执行器 v0.1.1 未改动; 两臂唯一差异=signal_ts "
                            "(臂A=触发记录件 first_action_ts[187条]/确认无行动不成交; 臂B=E日093500); "
                            "派生=各臂入场价->止损价/退出路径/依赖终点",
               "runner_sha256": sha256_file(HERE / "gcp_247_rule_compare_v1.py"),
               "executor_sha256": sha256_file(HERE / "dual_path_execution_v0_1_1.py")},
           "n_population": len(results),
           "status_matrix": {f"{k[0]}|{k[1]}|{k[2]}|{k[3]}": dict(v) for k, v in sorted(matrix.items())},
           "pairs_H3": {seg: {k: v for k, v in segd.items() if k != "rows"} for seg, segd in P3.items()},
           "pairs_H5": {seg: {k: v for k, v in segd.items() if k != "rows"} for seg, segd in P5.items()},
           "pair_rows_H3_val": P3["val"]["rows"],
           "verdicts_H3": verdicts_H3,
           "warning": WARN + "; 配对结果不得外推全人口; 现金0与NULL严格分列",
           "dir_bytes_after": dir_size(BASE),
           "elapsed_s": round(time.time() - t0, 1)}
    write_checked(OUT_JSON, json.dumps(out, ensure_ascii=False, indent=1))

    L = ["# 完整人口规则对照执行 v1 (零网络, 合同 v1 FROZEN)", "",
         f"- 生成: {out['generated_at']} | 复验全过 ({len(hv)} 项) | 用时 {out['elapsed_s']}s | {WARN}",
         "- 主指标=H3 机会收益差 Δ=A−B (已闭环净PnL÷50,000); 现金0与NULL严格分列; 配对结果不得外推全人口.", "",
         "## 全人口状态矩阵 (互斥类|臂|轨道|段 -> 可评价/现金0/NULL)", ""]
    for k, v in sorted(out["status_matrix"].items()):
        L.append(f"- {k}: {v}")
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
    L += ["", f"- val 判定: **{vp['verdict']}**" + (f"；绝对经济性警示: {vp.get('absolute_loss_warning')}"
          if vp.get("absolute_loss_warning") else ""),
          "- 三档出口: 支持正增量(均值>0且CI下界>0) / 证据不足 / 支持负增量(均值<0且CI上界<0); "
          "'规则改善但仍亏钱'为独立警示, 正增量不能抵消绝对亏损.", "",
          "## H5 辅助 (不入裁决)", ""]
    for seg in ("train", "val", "test"):
        s = P5[seg]
        d, a = s["delta_boot"], (s["a_boot"] or {})
        L.append(f"- {seg}: Δ {d['mean'] if d else '-'} [{d['ci_lower'] if d else '-'}, {d['ci_upper'] if d else '-'}] "
                 f"(配对 {s['n_pairs']}/簇 {s['n_dates']}); 臂A {a.get('mean','-')}")
    L += ["", "## 与 247 漏斗两臂结果对照", "",
          "- 买点合同(247人口, 单笔净收益口径): val Δ=+0.004749 [0.00122, 0.009003], 判定'买点有改善但绝对收益仍差'.",
          "- 本执行(1,013 人口, 机会收益口径, 全行动/无行动覆盖): 见上表; 口径与人口不同, 只作方向参照, 不并表.", "",
          "## 声明", "",
          "- 未调参/未训练/未补采(328 挂起); 全人口状态已同报; 0 与 NULL 分列.", ""]
    write_checked(OUT_MD, "\n".join(L))
    print(f"done ({time.time()-t0:.1f}s)")
    for seg in ("train", "val", "test"):
        s = P3[seg]
        print(f" H3 {seg}: pairs={s['n_pairs']} dates={s['n_dates']} dBoot={s['delta_boot']} | verdict={verdicts_H3[seg]['verdict']}")
    print("dir bytes:", dir_size(BASE))


if __name__ == "__main__":
    main()
