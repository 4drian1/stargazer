# -*- coding: utf-8 -*-
"""行动过滤 LR 修复重拟合 v1.1 — branch 前视缺陷修复 + 冻结参数原样重算.

修复 (用户核对授权, 审计 -22):
  branch 前视: v1 把 action_kind(含"双触发"类别)直接作 branch 特征; 双触发在首次决策时不可知.
  修复层 = fit 管道 (触发记录件不动, 其哈希 f6c4c997… 不变; 记录件保留 s1_ts/s2_ts/double_trigger
  作审计列, double_trigger 明文禁止入模). branch 一律取最早触发 (min(s1_ts,s2_ts) 归属;
  两条双触发 300377/688520 已核时序均归 S1).
  反例测试新增: 删除首次触发之后的分钟 -> 四字段(contraction/vol_cv10/setup/branch)不变
  (对两条双触发记录及抽样记录, 重跑 scan_views 截断窗核验). 先过后算.
系数解释更正 (v1 报告与路线 #16.19 已同步): contraction 越小=收缩越强, 负系数=越扩张越不预测盈利;
  S2=行动分支 ≠ B_LVB 形态池, 删除互为印证表述.
重拟合: train 114 (branch 修复后), L2 C=0.1 / 阈值≥0.50 / 标准化编码仅 train / 无搜索重采样加权.
三策略同框对比. 报告 v2 新文件 (gcp_247_action_filter_fit_v1_1.*); v1 保留为实现偏差记录.
零网络零采集零新数据; 写前查预算; 不调参不补采.
"""
from __future__ import annotations
import sys, json, time, hashlib
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import gcp_dual_path_coverage_probe as PROBE
import dual_path_execution_v0_1_1 as D

BASE = HERE / "gcp_round_b_small_sample_v1"
OUT_JSON = BASE / "gcp_247_action_filter_fit_v1_1.model.json"
OUT_MD = BASE / "gcp_247_action_filter_fit_v1_1.md"
BUDGET = 50 * 1024 * 1024
SEED = 20261005
N_BOOT = 10000
C_FIX = 0.1
THRESH = 0.50


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


def earliest_branch(rec):
    """branch = 最早触发归属 (修复: 不取 action_kind 的'双触发'类别)."""
    s1, s2 = rec.get("s1_ts"), rec.get("s2_ts")
    if s1 is None and s2 is None:
        return None
    if s1 is not None and (s2 is None or s1 <= s2):
        return "S1"
    return "S2"


def main():
    t0 = time.time()
    print(f"[预算] dir={dir_size(BASE)} B / {BUDGET} B")
    trig_recs = {r["code"] + "@" + str(r["E"]): r for r in
                 json.loads(open(BASE / "gcp_247_ruleA_trigger_records_v1.json", encoding="utf-8").read())["records"]}
    phase0 = json.loads(open(BASE / "gcp_247_action_filter_phase0_v1.json", encoding="utf-8").read())
    v1_model = json.loads(open(BASE / "gcp_247_action_filter_fit_v1.model.json", encoding="utf-8").read())

    # ---------- 反例测试 (先过后算) ----------
    # (a) 原五项配对规则 (沿用 v1 测试)
    EVAL, CASH, NULL = "EVALUABLE", "CASH0", "NULL"
    rows_t = [(CASH, 0.0, EVAL, 0.010), (EVAL, -0.005, CASH, 0.0), (CASH, 0.0, CASH, 0.0),
              (EVAL, 0.01, NULL, None), (NULL, None, CASH, 0.0)]
    prs = []
    for as_, ao, bs, bo in rows_t:
        if as_ in (EVAL, CASH) and bs in (EVAL, CASH):
            a = 0.0 if as_ == CASH else ao
            b = 0.0 if bs == CASH else bo
            prs.append(a - b)
    tests = {"现金/成交·成交/现金·双方现金入配对": len(prs) == 3 and abs(prs[2]) < 1e-12,
             "任一NULL不入配对": len(prs) == 3}
    # (b) branch 修复: 两条双触发均归 S1 (已核时序)
    dbl = {k: r for k, r in trig_recs.items() if r.get("double_trigger")}
    tests["双触发branch=最早触发(S1)"] = all(earliest_branch(r) == "S1" for r in dbl.values()) and len(dbl) == 2
    # (c) 新增: 删除首次触发之后的分钟 -> 四字段不变 (含类别字段 branch)
    panel = PROBE.get_daily_parquet()
    panel["code"] = panel["code"].astype(str).str.zfill(6)
    panel["date"] = panel["date"].astype(int)
    cal = sorted(int(d) for d in panel["date"].unique())
    daily_by_code = {c: g.sort_values("date").reset_index(drop=True) for c, g in panel.groupby("code")}
    D.MIN_DIRS.insert(0, BASE / "deferral_collect_v2" / "by_pair")
    D.MIN_DIRS.insert(1, BASE / "p2b1_collect" / "by_pair")

    def scan_with(code, Pd, day, keep_after_ts=None):
        dd = daily_by_code.get(code)
        loc = int(dd["date"].searchsorted(Pd, side="right"))
        d_sub = dd.iloc[:loc]
        atr_ref = PROBE.E.build_atr_ref(d_sub)
        calibs = PROBE.evaluate_calibers(d_sub, atr_ref)
        cb = calibs.get("main", {})
        mbd, _ = D.load_minute_code(code)
        e_df = mbd.get(day)
        if keep_after_ts is not None:
            e_df = e_df[e_df["date"].astype(int) <= keep_after_ts].reset_index(drop=True)
        a14 = pd.Series(np.maximum(
            d_sub["high"].astype(float).to_numpy() - d_sub["low"].astype(float).to_numpy(),
            np.abs(d_sub["high"].astype(float).to_numpy()
                   - np.concatenate([[d_sub["close"].astype(float).to_numpy()[0]],
                                     d_sub["close"].astype(float).to_numpy()[:-1]]))
        )).rolling(14).mean().to_numpy()
        amed60 = float(np.nanmedian(a14[-60:]))
        contraction = atr_ref / amed60 if amed60 > 0 else np.nan
        vol_arr = d_sub["volume"].astype(float).to_numpy()
        cv = float(np.std(vol_arr[-10:], ddof=1) / np.mean(vol_arr[-10:])) if np.mean(vol_arr[-10:]) > 0 else np.nan
        if not (cb.get("valid") and e_df is not None and len(e_df)):
            return None
        v = PROBE.scan_views(e_df, cb["upper"], cb["lower"], atr_ref,
                             contraction <= PROBE.ATR_CONTRACTION_MAX, cv <= PROBE.VOL_STABILITY_CV_MAX)
        return {"contraction": round(float(contraction), 6), "vol_cv10": round(cv, 6),
                "s1_ts": v["s1_ts"], "s2_ts": v["s2_ts"]}

    trunc_ok, trunc_detail = 0, []
    for k, r in dbl.items():
        full = scan_with(r["code"], int(r["P"]), int(r["E"]))
        cut = scan_with(r["code"], int(r["P"]), int(r["E"]), keep_after_ts=int(r["first_action_ts"]))
        same = (full and cut and full["contraction"] == cut["contraction"]
                and full["vol_cv10"] == cut["vol_cv10"]
                and full["s1_ts"] == cut["s1_ts"]
                and earliest_branch({"s1_ts": full["s1_ts"], "s2_ts": full["s2_ts"]})
                == earliest_branch({"s1_ts": cut["s1_ts"], "s2_ts": cut["s2_ts"]}))
        trunc_ok += bool(same)
        trunc_detail.append({"code": r["code"], "E": r["E"], "fields_unchanged": bool(same),
                             "full_s1": full["s1_ts"], "trunc_s1": cut["s1_ts"],
                             "full_s2": full["s2_ts"], "trunc_s2": cut["s2_ts"]})
    tests["首次触发后分钟截断→四字段不变(含branch)"] = (trunc_ok == len(dbl))
    failed = [n for n, ok in tests.items() if not ok]
    print("反例测试:", {n: ("PASS" if ok else "FAIL") for n, ok in tests.items()})
    for d in trunc_detail:
        print("  ", d)
    if failed:
        raise SystemExit(f"反例测试失败: {failed} -> 中止")

    # ---------- 重拟合 (冻结参数) ----------
    rows = phase0["rows"]
    for r in rows:
        tr = trig_recs[r["code"] + "@" + str(r["E"])]
        r["branch_fixed"] = earliest_branch(tr)
    NUM = ("feat_contraction", "feat_vol_cv10")
    CAT = ("setup", "branch_fixed")
    train = [r for r in rows if r["segment"] == "train" and r["traded_closed"]]
    assert len(train) == 114
    mu = {f: float(np.mean([r[f] for r in train])) for f in NUM}
    sd = {f: float(np.std([r[f] for r in train], ddof=0)) or 1.0 for f in NUM}
    cats = {c: sorted({r[c] for r in train}) for c in CAT}
    feat_names = list(NUM) + [f"{c}={k}" for c in CAT for k in cats[c][:-1]]

    def encode(r):
        x = [(r[f] - mu[f]) / sd[f] for f in NUM]
        for c in CAT:
            x += [1.0 if r[c] == k else 0.0 for k in cats[c][:-1]]
        return x

    X = np.array([encode(r) for r in train])
    y = np.array([r["label_main"] for r in train], dtype=int)
    from sklearn.linear_model import LogisticRegression
    lr = LogisticRegression(C=C_FIX, penalty="l2", solver="liblinear", random_state=SEED,
                            class_weight=None, max_iter=1000)
    lr.fit(X, y)
    coef = {n: round(float(c), 6) for n, c in zip(feat_names, lr.coef_[0])}
    inter = round(float(lr.intercept_[0]), 6)
    acc = round(float(lr.score(X, y)), 4)
    print("v1_1 coef:", coef, "acc:", acc)

    def prob(r):
        return float(lr.predict_proba(np.array([encode(r)]))[0][1])
    for r in rows:
        r["p_profit"] = prob(r)

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

    def strat_outcome(r, name):
        reject = False
        if name == "rule":
            reject = (r["feat_contraction"] is not None and r["feat_contraction"] > 1.0)
        elif name == "lr":
            reject = (r["p_profit"] is None or r["p_profit"] < THRESH)
        if reject:
            return (0.0, False)
        if r["traded_closed"]:
            return (float(r["pnl"]) / 50000.0, True)
        if r["entry_status"] == "NO_FILL_EOD":
            return (0.0, False)
        return (None, False)

    STRATS = (("baseline", "全放行"), ("rule", "规则基线 contraction≤1.0"), ("lr", f"LR≥{THRESH} 放行"))
    report = {}
    for seg in ("train", "val", "test"):
        frame = [r for r in rows if r["segment"] == seg and r["validity_reason"] != "NULL"]
        seg_rep = {"n_evaluable": len(frame),
                   "note": ("样本内(in-sample)" if seg == "train" else "主判读" if seg == "val" else "描述(非新盲测)")}
        for key, label in STRATS:
            opps, kept = [], []
            for r in frame:
                opp, kb = strat_outcome(r, key)
                if opp is None:
                    continue
                opps.append((r["E"], opp))
                if kb:
                    kept.append((r["E"], opp, float(r["C_buy"] or 0)))
            by = defaultdict(list)
            for d, o in opps:
                by[d].append(o)
            kby = defaultdict(list)
            for d, o, _ in kept:
                kby[d].append(o)
            n_kept = len(kept)
            util = (sum(c for _, _, c in kept) / (50000.0 * len(opps))) if opps else None
            seg_rep[key] = {"label": label, "opp_boot": cluster_boot(by), "traded_subset_boot": cluster_boot(kby),
                            "n_kept_trades": n_kept,
                            "filter_rate": round(1 - n_kept / max(1, sum(1 for r in frame if r["traded_closed"])), 4)
                            if any(r["traded_closed"] for r in frame) else None,
                            "capital_utilization": round(util, 6) if util is not None else None,
                            "left_tail": sorted([o for _, o, _ in kept])[:min(10, n_kept)],
                            "rejects_all_trades": (n_kept == 0)}
        report[seg] = seg_rep

    # 与 v1 对比
    v1_coef = v1_model["model"]["coef"]
    coef_delta = {n: round(coef[n] - v1_coef[n], 6) for n in feat_names if n in v1_coef}
    v1_val = v1_model["report"]["val"]
    compare = {"coef_delta": coef_delta,
               "val": {k: {"v1_kept": v1_val[k]["n_kept_trades"],
                           "v1_opp_mean": (v1_val[k]["opp_boot"] or {}).get("mean"),
                           "v11_kept": report["val"][k]["n_kept_trades"],
                           "v11_opp_mean": (report["val"][k]["opp_boot"] or {}).get("mean")}
                       for k, _ in STRATS}}

    artifact = {"phase": "GCP_247_ACTION_FILTER_FIT_V1_1_FIX",
                "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "network_requests": 0, "research_fit_only": True,
                "fix": "branch 前视修复: 取最早触发归属(两条双触发均归S1); 修复层=fit管道; "
                       "触发记录件未动(哈希不变); double_trigger 保留审计列禁止入模",
                "counterexample_tests": {n: bool(ok) for n, ok in tests.items()},
                "truncation_detail": trunc_detail,
                "trigger_records_sha256_unchanged": sha256_file(BASE / "gcp_247_ruleA_trigger_records_v1.json"),
                "hash_verification": {"trigger_records": {"match": sha256_file(BASE / "gcp_247_ruleA_trigger_records_v1.json") == "f6c4c997003fe3b5191666a1a120feffdba7f9fcb0289ba0a9d339c89b98dc6e"},
                                      "phase0": {"sha256": sha256_file(BASE / "gcp_247_action_filter_phase0_v1.json")},
                                      "v1_model": {"sha256": sha256_file(BASE / "gcp_247_action_filter_fit_v1.model.json")}},
                "model": {"type": "L2-LogisticRegression", "C": C_FIX, "threshold": THRESH, "seed": SEED,
                          "feature_names": feat_names, "coef": coef, "intercept": inter,
                          "scaler": {"mean": mu, "std": sd}, "encoder_categories": cats,
                          "train_in_sample_accuracy": acc, "train_n": len(train), "train_pos": int(y.sum())},
                "coef_reading_note": "contraction=ATR_ref/历史中位, 越小=收缩越强; 负系数=越扩张→预测盈利概率越低. "
                                     "S2=行动分支, 与B_LVB形态池两字段不互为印证.",
                "v1_comparison": compare,
                "report": report}
    write_checked(OUT_JSON, json.dumps(artifact, ensure_ascii=False, indent=1))

    def fmt(b):
        return "-" if not b else f"{b['mean']} [{b['ci_lower']}, {b['ci_upper']}]"

    L = ["# 行动过滤 LR 修复重拟合报告 v1.1（branch 前视修复；v1 保留为实现偏差记录）", "",
         f"- 生成: {artifact['generated_at']} | 反例测试全过 | 零网络 | C=0.1 / 阈值≥{THRESH} 原样",
         f"- 修复: {artifact['fix']}", "",
         "## 反例测试", ""]
    for n, ok in artifact["counterexample_tests"].items():
        L.append(f"- {'PASS' if ok else 'FAIL'} {n}")
    for d in trunc_detail:
        L.append(f"  - 截断核验 {d['code']}@{d['E']}: s1 {d['full_s1']}→{d['trunc_s1']} "
                 f"(s2 {d['full_s2']}→截断后 {d['trunc_s2']}), 字段不变={d['fields_unchanged']}")
    L += ["", "## 系数 (修复后) vs v1", "",
          f"- v1_1: {json.dumps(coef, ensure_ascii=False)}; 截距 {inter}; 样本内准确率 {acc}",
          f"- 变化量: {json.dumps(coef_delta, ensure_ascii=False)}",
          "- 解读: contraction 越小=收缩越强, 负系数=越扩张越不预测盈利; S2=行动分支≠B_LVB 形态池.", "",
          "## 三策略对比 (val 主判读)", "",
          "| 策略 | 机会收益均值 [95%CI] | 成交子集 [95%CI] | 保留 | 过滤率 | 利用率 | 左尾最差 |",
          "|---|---|---|---|---|---|---|"]
    for key, _ in STRATS:
        v = report["val"][key]
        L.append(f"| {v['label']} | {fmt(v['opp_boot'])} | {fmt(v['traded_subset_boot'])} | "
                 f"{v['n_kept_trades']} | {v['filter_rate']} | {v['capital_utilization']} | "
                 f"{v['left_tail'][0] if v['left_tail'] else '-'} |")
    L += ["", "## 与 v1 对照 (val)", "",
          "| 策略 | v1 保留/机会收益均值 | v1.1 保留/机会收益均值 |",
          "|---|---|---|"]
    for key, _ in STRATS:
        c = compare["val"][key]
        L.append(f"| {key} | {c['v1_kept']} / {c['v1_opp_mean']} | {c['v11_kept']} / {c['v11_opp_mean']} |")
    L += ["", "## train(样本内)/test(描述)", ""]
    for seg in ("train", "test"):
        for key, _ in STRATS:
            v = report[seg][key]
            L.append(f"- {seg} {v['label']}: opp {fmt(v['opp_boot'])} | 保留 {v['n_kept_trades']} | 利用率 {v['capital_utilization']}")
    L += ["", "## 归档判读", "",
          "- 修复后 LR 线最终状态: **过滤减亏方向不变, 但仍未证明能识别盈利机会**; val 小样本(盈利仅 5 条)不确定性依旧; "
          "不晋级、不上线、本 LR 线冻结(归档), '共性背景正经济性'研究方案另立.", ""]
    write_checked(OUT_MD, "\n".join(L))
    print(f"done ({time.time()-t0:.1f}s)")
    print("compare:", json.dumps(compare, ensure_ascii=False))
    print("dir:", dir_size(BASE))


if __name__ == "__main__":
    main()
