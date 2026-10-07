# -*- coding: utf-8 -*-
"""共性背景价值研究 第一阶段执行 (合同 v1 FROZEN 单独执行授权): 覆盖核验+分层描述+固定规则对照.

范围: 零网络; 既有冻结输入只读; 同一 1,013 人口同一 S1/S2; 现金 0 与 NULL 严格分列;
分层边界只在 train 计算随后冻结 (setup 按 A/B; contraction/vol_cv10/箱体相对宽度 train 中位数二分);
不按收益挑组; 不拟合 LR; 不补采 328; 不调阈值.
两臂 (合同 §3): 基线臂=全 1,013 观察+冻结 S1/S2 (规则对照 v1_1 臂A口径); 过滤臂=盘前
contraction<=1.0 才观察+同一 S1/S2, 被拒->现金 0, contraction 缺失->NULL (无法判断).
主比较=同事件配对 Δ=过滤臂-基线臂 (日期聚类 bootstrap 20261005/万次/双侧95%);
val 主判读/train 解释/test 描述(非新盲测).
产物: gcp_round_b_bg_value_v1/{coverage_check,stratified_description,fixed_rule_compare}.json/.md + progress.md.
预算: 旧目录(47.75MiB)+新目录累计 <=50MiB, 逐写前核验, 不足即停不删旧件.
"""
from __future__ import annotations
import sys, json, time, hashlib, glob
from pathlib import Path
from collections import Counter, defaultdict

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import gcp_dual_path_coverage_probe as PROBE
import dual_path_execution_v0_1_1 as D

OLD = HERE / "gcp_round_b_small_sample_v1"
NEW = HERE / "gcp_round_b_bg_value_v1"
BUDGET = 50 * 1024 * 1024
SEED = 20261005
N_BOOT = 10000
LEFT_TAIL = -0.05
CONTRACT = "contracts/background_value_study_contract_v1.md"


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def dir_size(p: Path) -> int:
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file()) if p.exists() else 0


def write_checked(path: Path, text: str):
    b = len(text.encode("utf-8"))
    if dir_size(OLD) + dir_size(NEW) + b > BUDGET:
        raise RuntimeError(f"budget exceeded (old+new) before writing {path.name} (stop, 不删旧件)")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    print(f"  written {path.name} (+{b} B; old+new now {dir_size(OLD)+dir_size(NEW)} B)")


def verify_hashes():
    checks = {
        "v3_inventory": (OLD / "gcp_247_full_population_inventory_v3.json", "a2c53545efc331e64438a590d155cae25d79048c79e2d8c4ec4166e9e018d0f8"),
        "trigger_records": (OLD / "gcp_247_ruleA_trigger_records_v1.json", "f6c4c997003fe3b5191666a1a120feffdba7f9fcb0289ba0a9d339c89b98dc6e"),
        "feature_pipeline": (HERE / "gcp_dual_path_coverage_probe.py", "750710918a626e8d18c6157971e668e4b46e67763eed21c3d718ad00eb0ac659"),
        "execution_contract": (HERE / "contracts" / "dual_path_execution_contract_v0_1_1_final.md", "9c22c37e5e9bcc4e874f063730f19b138fac27f0c89de04d6b39cc1c0a8b852d"),
        "executor": (HERE / "dual_path_execution_v0_1_1.py", "aee8e60bc00951a3f18975d7a01dda7c64af27a9eb7d5e1592cd8c3716aa91cb"),
        "rule_compare_contract": (HERE / "contracts" / "full_population_rule_compare_contract_v1.md", None),
        "this_contract": (HERE / CONTRACT, None),
    }
    res, ok_all = {}, True
    for role, (p, expect) in checks.items():
        actual = sha256_file(p)
        entry = {"sha256": actual}
        if expect:
            entry["match"] = (actual == expect)
            ok_all &= entry["match"]
        res[role] = entry
    man = json.loads((HERE / "contracts" / "dual_path_execution_v0_1_1_manifest.json").read_text(encoding="utf-8"))
    for f in man["files"]:
        m = sha256_file(Path("D:/stargazer") / f["path"]) == f["sha256"]
        ok_all &= m
        res[f"manifest:{f['role']}"] = {"match": m}
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
        means[b] = vals[np.concatenate([idx_per_c[p] for p in picks])].mean()
    return {"mean": round(float(vals.mean()), 8), "ci_lower": round(float(np.percentile(means, 2.5)), 8),
            "ci_upper": round(float(np.percentile(means, 97.5)), 8), "n": int(len(vals)), "n_clusters": n_c}


def main():
    t0 = time.time()
    print(f"[预算] old+new={dir_size(OLD)+dir_size(NEW)} B / {BUDGET} B")
    hv = verify_hashes()
    print(f"[复验] 全过 ({len(hv)} 项)")

    pop = json.loads(open(OLD / "gcp_247_full_population_inventory_v3.json", encoding="utf-8").read())["records"]
    trig = {r["code"] + "@" + str(r["E"]): r for r in
            json.loads(open(OLD / "gcp_247_ruleA_trigger_records_v1.json", encoding="utf-8").read())["records"]}

    panel = PROBE.get_daily_parquet()
    panel["code"] = panel["code"].astype(str).str.zfill(6)
    panel["date"] = panel["date"].astype(int)
    cal = sorted(int(d) for d in panel["date"].unique())
    daily_by_code = {c: g.sort_values("date").reset_index(drop=True) for c, g in panel.groupby("code")}

    def bg_features(r):
        """四盘前字段; 全部缺失返回 None (不填默认值)."""
        dd = daily_by_code.get(r["code"])
        if dd is None:
            return None
        loc = int(dd["date"].searchsorted(int(r["P"]), side="right"))
        if loc < 20:
            return None
        d_sub = dd.iloc[:loc]
        atr_ref = PROBE.E.build_atr_ref(d_sub)
        calibs = PROBE.evaluate_calibers(d_sub, atr_ref)
        cb = calibs.get("main", {})
        if not cb.get("valid"):
            return None
        a14 = pd.Series(np.maximum(
            d_sub["high"].astype(float).to_numpy() - d_sub["low"].astype(float).to_numpy(),
            np.abs(d_sub["high"].astype(float).to_numpy()
                   - np.concatenate([[d_sub["close"].astype(float).to_numpy()[0]],
                                     d_sub["close"].astype(float).to_numpy()[:-1]]))
        )).rolling(14).mean().to_numpy()
        amed60 = float(np.nanmedian(a14[-60:])) if len(a14) >= 14 else np.nan
        contraction = atr_ref / amed60 if (np.isfinite(amed60) and amed60 > 0) else np.nan
        vol_arr = d_sub["volume"].astype(float).to_numpy()
        last10 = vol_arr[-10:]
        cv = float(np.std(last10, ddof=1) / np.mean(last10)) if np.mean(last10) > 0 else np.nan
        close_p = float(d_sub["close"].astype(float).iloc[-1])
        width_rel = (float(cb["upper"]) - float(cb["lower"])) / close_p if close_p > 0 else np.nan
        out = {"contraction": (round(float(contraction), 6) if np.isfinite(contraction) else None),
               "vol_cv10": (round(cv, 6) if np.isfinite(cv) else None),
               "box_width_rel": (round(float(width_rel), 6) if np.isfinite(width_rel) else None)}
        if all(v is None for v in out.values()):
            return None
        return out

    # ---------- 特征与覆盖 ----------
    for r in pop:
        f = bg_features(r)
        r["feat"] = f or {}
        r["contraction"] = (f or {}).get("contraction")
        r["vol_cv10"] = (f or {}).get("vol_cv10")
        r["box_width_rel"] = (f or {}).get("box_width_rel")
    cov_tab = defaultdict(lambda: Counter())
    for r in pop:
        has = r["feat"] != {}
        cov_tab[(r["segment"], r["mutual"][:1], r["setup"])]["has_all" if has and
                all(r[k] is not None for k in ("contraction", "vol_cv10", "box_width_rel")) else "missing"] += 1
    con_missing = [r for r in pop if r["contraction"] is None]

    # ---------- 冻结分层边界 (train 中位数) ----------
    def train_median(key):
        vals = [r[key] for r in pop if r["segment"] == "train" and r.get(key) is not None]
        return round(float(np.median(vals)), 6)
    bounds = {"contraction": train_median("contraction"), "vol_cv10": train_median("vol_cv10"),
              "box_width_rel": train_median("box_width_rel")}
    def layer(r, key):
        v = r.get(key)
        if v is None:
            return "NA"
        return "hi" if v > bounds[key] else "lo"

    # ---------- 基线臂复放 (同 v1_1 臂A口径) ----------
    D.MIN_DIRS.insert(0, OLD / "deferral_collect_v2" / "by_pair")
    D.MIN_DIRS.insert(1, OLD / "p2b1_collect" / "by_pair")
    mbd_cache = {}
    def get_minute(code):
        if code not in mbd_cache:
            mbd_cache[code] = D.load_minute_code(code)[0]
        return mbd_cache[code]
    cov_board = {(x["code"], int(x["E"])): x["board"] for x in
                 json.loads(open(OLD / "gcp_247_coverage_verify.json", encoding="utf-8").read())["rows"]}
    def board_of(c, E):
        if (c, E) in cov_board:
            return cov_board[(c, E)]
        if c.startswith(("300", "301")):
            return "GEM"
        if c.startswith("688"):
            return "STAR"
        return "SH_MAIN" if c.startswith("6") else "SZ_MAIN"
    chains = json.loads(open(r"D:\stargazer\stargazer_ml\v5_hybrid\opportunity_v2\gate1c\feb2026_adjustment_cache_v1\adj_cache.json", encoding="utf-8").read())
    NEXT_SEG = {"train": 20260309, "val": 20260610}

    def h3_outcome(r):
        """基线臂单事件 H3 结果: (outcome, opp, net, reason). outcome: EVALUABLE/CASH0/NULL."""
        cls = r["mutual"]
        if cls == "4_数据不足无法判断":
            return ("NULL", None, None, "class4_数据不足")
        if cls in ("2_WATCH但无行动", "3_未WATCH且无行动"):
            return ("CASH0", 0.0, None, "确认无行动")
        tr = trig[r["code"] + "@" + str(r["E"])]
        res = D.execute_event(r["code"], int(r["E"]), board_of(r["code"], r["E"]), None, r["setup"],
                              signal_ts=int(tr["first_action_ts"]), minute_by_day=get_minute(r["code"]),
                              daily_df=daily_by_code.get(r["code"]), cal=cal)
        ent = res.get("entry") or {}
        if res.get("status") == "ENTRY_METADATA_UNKNOWN":
            return ("NULL", None, None, "ENTRY_METADATA_UNKNOWN")
        if ent.get("status") == "NO_FILL_EOD":
            return ("CASH0", 0.0, None, "确认未成交")
        if ent.get("status") != "SIM_FILLED":
            return ("NULL", None, None, f"ENTRY:{ent.get('status')}")
        t = res.get("H3") or {}
        if t.get("status") == "SIM_COMPLETE" and t.get("fully_exited") and t.get("net") is not None:
            exit_day = int(t["last_fill_ts"] // 1000000)
            if r["segment"] in NEXT_SEG and exit_day >= NEXT_SEG[r["segment"]]:
                return ("NULL", None, None, "CROSS_SEGMENT_LABEL")
            ch = chains.get(r["code"])
            if not (ch and ch.get("status") in ("OK", "EMPTY")):
                return ("NULL", None, None, "CORP_UNKNOWN")
            return ("EVALUABLE", round(float(t["pnl"]) / 50000.0, 8), float(t["net"]), "COMPLETE")
        if t.get("remaining_shares"):
            return ("NULL", None, None, f"RESIDUAL({t.get('status')})")
        return ("NULL", None, None, f"NULL({t.get('status')})")

    for i, r in enumerate(pop):
        oc, opp, net, rs = h3_outcome(r)
        r["base"] = {"outcome": oc, "opp": opp, "net": net, "reason": rs}
        # 过滤臂: contraction 缺失 -> NULL(无法判断); >1.0 -> 拒观察现金 0; 否则=基线
        if r["contraction"] is None:
            r["filt"] = {"outcome": "NULL", "opp": None, "net": None, "reason": "contraction缺失_无法判断"}
        elif r["contraction"] > 1.0:
            r["filt"] = {"outcome": "CASH0", "opp": 0.0, "net": None, "reason": "背景过滤拒绝观察"}
        else:
            r["filt"] = dict(r["base"])
        if (i + 1) % 300 == 0:
            print(f"  replay {i+1}/{len(pop)} ({time.time()-t0:.0f}s)")

    # ---------- 产物 1: coverage_check ----------
    cov_out = {"phase": "BG_VALUE_COVERAGE_CHECK_V1",
               "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"), "network_requests": 0,
               "fields": ["contraction", "vol_cv10", "box_width_rel", "setup"],
               "n_population": len(pop),
               "contraction_missing_n": len(con_missing),
               "contraction_missing_detail_head": [{"code": r["code"], "E": r["E"], "segment": r["segment"],
                                                    "mutual": r["mutual"]} for r in con_missing[:30]],
               "contraction_missing_by_cell": {f"{k[0]}|{k[1]}|{k[2]}": dict(v) for k, v in sorted(cov_tab.items())},
               "filter_arm_undecidable_n": len(con_missing),
               "note": "contraction 缺失的候选过滤臂=NULL(无法判断), 基线臂不受影响; 其余三字段覆盖见表",
               "hash_verification": hv}
    write_checked(NEW / "coverage_check.json", json.dumps(cov_out, ensure_ascii=False, indent=1))
    cm = [f"- {seg}|{mu}|{st}: {dict(v)}" for (seg, mu, st), v in sorted(cov_tab.items())]
    write_checked(NEW / "coverage_check.md", "\n".join([
        "# 背景价值 覆盖核验 v1", "",
        f"- 生成: {cov_out['generated_at']} | 人口 {len(pop)} | 零网络",
        f"- **contraction 缺失 {len(con_missing)} 条** -> 过滤臂无法判断 (NULL), 基线臂不受影响",
        f"- 其中含第 1 类(有行动) {sum(1 for r in con_missing if r['mutual'].startswith('1'))} 条", "",
        "## 覆盖率 (段|互斥类|setup)", ""] + cm + ["",
        "- 字段: contraction/vol_cv10/box_width_rel 来自 probe 管线 (P 及以前); setup 候选字段; 无默认值填补.", ""]))

    # ---------- 产物 2: stratified_description (探索描述) ----------
    strat = defaultdict(lambda: {"n": 0, "actioned": 0, "closed": 0, "opps": [], "tails": 0})
    for r in pop:
        keys = [f"setup={r['setup']}"]
        for fkey in ("contraction", "vol_cv10", "box_width_rel"):
            keys.append(f"{fkey}={layer(r, fkey)}")
        for kk in keys:
            s = strat[(r["segment"], kk)]
            s["n"] += 1
            if r["base"]["outcome"] == "EVALUABLE":
                s["actioned"] += 1 if r["mutual"].startswith("1") else 0
                s["closed"] += 1
                s["opps"].append(r["base"]["opp"])
                if r["base"]["net"] is not None and r["base"]["net"] <= LEFT_TAIL:
                    s["tails"] += 1
    strat_out = {"phase": "BG_VALUE_STRATIFIED_DESCRIPTION_V1",
                 "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"), "network_requests": 0,
                 "frozen_boundaries_train_median": bounds,
                 "boundary_rule": "train 中位数二分, 冻结后应用于 val/test; setup 按 A/B; 不按收益挑组",
                 "layers": {f"{seg}|{kk}": {"n": v["n"], "closed": v["closed"],
                                             "opp_mean": (round(float(np.mean(v["opps"])), 8) if v["opps"] else None),
                                             "left_tail_n": v["tails"]}
                            for (seg, kk), v in sorted(strat.items())},
                 "warning": "分层仅探索描述; 不按哪组收益好挑规则 (合同 §5)"}
    write_checked(NEW / "stratified_description.json", json.dumps(strat_out, ensure_ascii=False, indent=1))
    L = ["# 背景价值 分层描述 v1 (探索描述, 不按收益挑规则)", "",
         f"- 生成: {strat_out['generated_at']} | 冻结边界(train 中位数): {bounds}", "",
         "| 段 | 层 | 候选 | 闭环 | 机会收益均值 | 左尾≤−5% |", "|---|---|---|---|---|---|"]
    for (seg, kk), v in sorted(strat.items()):
        L.append(f"| {seg} | {kk} | {v['n']} | {v['closed']} | "
                 f"{round(float(np.mean(v['opps'])),6) if v['opps'] else '-'} | {v['tails']} |")
    L += ["", "- 注: 层内机会收益均值仅探索陈述, 不作规则选择依据; val/test 用冻结边界.", ""]
    write_checked(NEW / "stratified_description.md", "\n".join(L))

    # ---------- 产物 3: fixed_rule_compare ----------
    def arm_boot(seg, arm):
        by = defaultdict(list)
        nets = []
        for r in pop:
            if r["segment"] != seg:
                continue
            a = r[arm]
            if a["outcome"] in ("EVALUABLE", "CASH0"):
                by[r["E"]].append(a["opp"] if a["outcome"] == "EVALUABLE" else 0.0)
                if a["outcome"] == "EVALUABLE":
                    nets.append((r["E"], a["opp"], a["net"]))
        nby = defaultdict(list)
        for d, o, _ in nets:
            nby[d].append(o)
        tails = sum(1 for _, _, net in nets if net is not None and net <= LEFT_TAIL)
        return {"opp_boot": cluster_boot(by), "traded_boot": cluster_boot(nby),
                "n_evaluable": sum(len(v) for v in by.values()), "n_traded": len(nets),
                "left_tail_n": tails}

    def paired(seg):
        by = defaultdict(list)
        kinds = Counter()
        for r in pop:
            if r["segment"] != seg:
                continue
            a, b = r["filt"], r["base"]
            if a["outcome"] in ("EVALUABLE", "CASH0") and b["outcome"] in ("EVALUABLE", "CASH0"):
                ao = a["opp"] if a["outcome"] == "EVALUABLE" else 0.0
                bo = b["opp"] if b["outcome"] == "EVALUABLE" else 0.0
                by[r["E"]].append(round(ao - bo, 8))
                if a["outcome"] == "CASH0" and b["outcome"] == "EVALUABLE":
                    kinds["过滤拒/基线成交"] += 1
                elif a["outcome"] == "CASH0" and b["outcome"] == "CASH0":
                    kinds["双现金"] += 1
        return {"delta_boot": cluster_boot(by), "n_pairs": sum(len(v) for v in by.values()),
                "n_dates": len(by), "pair_kinds": dict(kinds)}

    cmp_out = {"phase": "BG_VALUE_FIXED_RULE_COMPARE_V1",
               "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"), "network_requests": 0,
               "arms": {"baseline": "全 1,013 观察+冻结 S1/S2 (规则对照 v1_1 臂A口径)",
                        "filter": "盘前 contraction≤1.0 才观察+同一 S1/S2; 拒绝->现金0; contraction缺失->NULL"},
               "main_test": "同事件配对 Δ=过滤臂-基线臂, 日期聚类 bootstrap 20261005/万次/双侧95%",
               "by_segment": {}}
    for seg in ("train", "val", "test"):
        cmp_out["by_segment"][seg] = {"base": arm_boot(seg, "base"), "filt": arm_boot(seg, "filt"),
                                      "paired": paired(seg),
                                      "note": ("解释(样本内对照)" if seg == "train" else
                                               "主判读" if seg == "val" else "描述(非新盲测)")}
    vp = cmp_out["by_segment"]["val"]["paired"]["delta_boot"]
    verdict = None
    if vp:
        if vp["mean"] > 0 and vp["ci_lower"] > 0:
            verdict = "配对增量为正(CI 不含 0)"
        elif vp["mean"] < 0 and vp["ci_upper"] < 0:
            verdict = "配对增量为负(CI 不含 0)"
        else:
            verdict = "无法确认(CI 含 0 或样本不足)"
    cmp_out["val_verdict"] = verdict
    write_checked(NEW / "fixed_rule_compare.json", json.dumps(cmp_out, ensure_ascii=False, indent=1))

    def fmt(b):
        return "-" if not b else f"{b['mean']} [{b['ci_lower']}, {b['ci_upper']}]"

    L = ["# 背景价值 固定规则对照 v1 (两臂配对检验)", "",
         f"- 生成: {cmp_out['generated_at']} | 零网络 | 主检验: 同事件配对 Δ (过滤臂−基线臂)",
         "- 现金 0 与 NULL 严格分列; 条件子集结论不外推全人口; test 非新盲测.", "",
         "## 分段对比 (H3 机会收益; 净收益口径同引擎)", "",
         "| 段 | 臂 | 机会收益均值 [95%CI] | 成交子集 [95%CI] | 可评价 | 成交 | 左尾≤−5% |",
         "|---|---|---|---|---|---|---|"]
    for seg in ("val", "train", "test"):
        s = cmp_out["by_segment"][seg]
        for arm in ("base", "filt"):
            v = s[arm]
            L.append(f"| {seg}({s['note']}) | {'基线' if arm=='base' else '过滤'} | {fmt(v['opp_boot'])} | "
                     f"{fmt(v['traded_boot'])} | {v['n_evaluable']} | {v['n_traded']} | {v['left_tail_n']} |")
        p = s["paired"]
        L.append(f"| {seg} | 配对 Δ | {fmt(p['delta_boot'])} | 配对 {p['n_pairs']}/簇 {p['n_dates']} | "
                 f"{p['pair_kinds']} | | |")
    L += ["", f"- **val 主判读出口: {verdict}**", "",
          "## 全人口可用率", ""]
    for seg in ("train", "val", "test"):
        rs = [r for r in pop if r["segment"] == seg]
        c = Counter(r["base"]["outcome"] for r in rs)
        c2 = Counter(r["filt"]["outcome"] for r in rs)
        L.append(f"- {seg}: 基线臂 {dict(c)} | 过滤臂 {dict(c2)} (人口 {len(rs)})")
    L += ["", "## 声明", "",
          "- contraction≤1.0 为固定参照(非新发现); 不按收益挑组; 不拟合 LR; 328 挂起.", ""]
    write_checked(NEW / "fixed_rule_compare.md", "\n".join(L))

    write_checked(NEW / "progress.md", "\n".join([
        "# 背景价值研究 进度", "",
        f"- 第一阶段执行完成: {time.strftime('%Y-%m-%d %H:%M:%S')} | 用时 {time.time()-t0:.1f}s | 零网络",
        "- 步骤: 复验(全过) -> 覆盖核验 -> 分层边界冻结(train 中位数) -> 基线臂复放 -> 过滤臂派生 -> 配对检验",
        f"- val 出口: {verdict}",
        f"- 预算: 旧 {dir_size(OLD)} B + 新 {dir_size(NEW)} B = {dir_size(OLD)+dir_size(NEW)} B (< {BUDGET})",
        "- 下一阶段(如需): 轻量基线/训练须单独授权.", ""]))
    print(f"stage1 done ({time.time()-t0:.1f}s); val verdict: {verdict}")
    for seg in ("train", "val", "test"):
        s = cmp_out["by_segment"][seg]
        print(f" {seg}: base {fmt(s['base']['opp_boot'])} | filt {fmt(s['filt']['opp_boot'])} | Δ {fmt(s['paired']['delta_boot'])}")
    print("old+new:", dir_size(OLD) + dir_size(NEW), "B")


if __name__ == "__main__":
    main()
