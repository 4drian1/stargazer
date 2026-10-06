# -*- coding: utf-8 -*-
"""共享行动过滤 LR 第一轮研究性拟合 (合同 v1 FROZEN + §8.7 补签; 用户单独授权).

第1步 复验: 合同输入哈希 (触发记录件/第〇阶段盘点/规则对照报告/coverage_v1/执行合同/执行器/manifest5).
第2步 拟合: train 114 成交闭环样本 (正53/负61); 特征=冻结四字段; 标准化与编码仅 train 拟合;
  固定 L2 C=0.1 (liblinear); 不搜索/不重采样/不类别加权. 标签=第〇阶段逐事件逐轨道记录.
第3步 三策略同可评价样本框对比: 全放行 / contraction<=1.0 / LR(>=0.50 放行, §8.7 补签).
第4步 报告 (合同 §4 全含): 机会收益分布(拒绝->现金0, NULL 隔离不补0)/成交子集收益单独列/
  左尾(仅成交轨道)/过滤率与保留成交数/资金利用率/E日聚类CI(20261005/万次/双侧95);
  val 主判读(30 条有效标签, 盈利仅 5 条, 不确定性如实标注); train 样本内标注; test 描述.
第5步 硬性判读: 拒绝所有交易不能称为成功 -> 保留成交数/绝对收益(成交子集)/资金利用率单列.
零网络零采集; 不自动上线; 产物 gcp_247_action_filter_fit_v1.json/.md (模型工件+报告, RUN 新文件).
"""
from __future__ import annotations
import sys, json, time, hashlib
from pathlib import Path
from collections import defaultdict

import numpy as np

HERE = Path(__file__).resolve().parent
BASE = HERE / "gcp_round_b_small_sample_v1"
OUT_JSON = BASE / "gcp_247_action_filter_fit_v1.model.json"
OUT_MD = BASE / "gcp_247_action_filter_fit_v1.md"
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


def verify_hashes():
    checks = {
        "trigger_records": (BASE / "gcp_247_ruleA_trigger_records_v1.json", "f6c4c997003fe3b5191666a1a120feffdba7f9fcb0289ba0a9d339c89b98dc6e"),
        "phase0_inventory": (BASE / "gcp_247_action_filter_phase0_v1.json", None),
        "rule_compare_v1_1": (BASE / "gcp_247_rule_compare_v1_1.json", None),
        "coverage_v1": (BASE / "gcp_247_return_path_coverage_v1.json", None),
        "execution_contract": (HERE / "contracts" / "dual_path_execution_contract_v0_1_1_final.md", "9c22c37e5e9bcc4e874f063730f19b138fac27f0c89de04d6b39cc1c0a8b852d"),
        "executor": (HERE / "dual_path_execution_v0_1_1.py", "aee8e60bc00951a3f18975d7a01dda7c64af27a9eb7d5e1592cd8c3716aa91cb"),
        "contract_v1": (HERE / "contracts" / "action_filter_model_research_contract_v1.md", None),
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
        idx = np.concatenate([idx_per_c[p] for p in picks])
        means[b] = vals[idx].mean()
    return {"mean": round(float(vals.mean()), 8),
            "ci_lower": round(float(np.percentile(means, 2.5)), 8),
            "ci_upper": round(float(np.percentile(means, 97.5)), 8),
            "n": int(len(vals)), "n_clusters": n_c}


def main():
    t0 = time.time()
    print(f"[预算] dir={dir_size(BASE)} B / {BUDGET} B")
    hv = verify_hashes()
    print(f"[复验] 全过 ({len(hv)} 项)")
    rows = json.loads(open(BASE / "gcp_247_action_filter_phase0_v1.json", encoding="utf-8").read())["rows"]

    NUM = ("feat_contraction", "feat_vol_cv10")
    CAT = ("setup", "branch")
    def feats(r):
        return [r["feat_contraction"], r["feat_vol_cv10"], r["setup"], r["branch"]]

    train = [r for r in rows if r["segment"] == "train" and r["traded_closed"]]
    assert len(train) == 114 and sum(r["label_main"] for r in train) == 53
    mu = {f: float(np.mean([r[f] for r in train])) for f in NUM}
    sd = {f: float(np.std([r[f] for r in train], ddof=0)) or 1.0 for f in NUM}
    cats = {c: sorted({r[c] for r in train}) for c in CAT}

    def encode(r):
        x = [(r[f] - mu[f]) / sd[f] for f in NUM]
        for c in CAT:
            x += [1.0 if r[c] == k else 0.0 for k in cats[c][:-1]]  # drop-last
        return x

    feat_names = list(NUM) + [f"{c}={k}" for c in CAT for k in cats[c][:-1]]
    X = np.array([encode(r) for r in train])
    y = np.array([r["label_main"] for r in train], dtype=int)

    from sklearn.linear_model import LogisticRegression
    lr = LogisticRegression(C=C_FIX, penalty="l2", solver="liblinear", random_state=SEED,
                            class_weight=None, max_iter=1000)
    lr.fit(X, y)
    coef = {n: round(float(c), 6) for n, c in zip(feat_names, lr.coef_[0])}
    inter = round(float(lr.intercept_[0]), 6)
    train_acc = float(lr.score(X, y))
    print("coef:", coef, "intercept:", inter, "train_acc(in-sample):", round(train_acc, 4))

    def prob(r):
        return float(lr.predict_proba(np.array([encode(r)]))[0][1])

    for r in rows:
        r["p_profit"] = prob(r) if all(r[f] is not None for f in NUM) else None

    def strat_outcome(r, name):
        """返回 (opp, kept_trade_bool). 可评价框内: 拒绝->0; 放行->按实际成交/未成交."""
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
            return (0.0, False)  # 放行但确认未成交 -> 现金 0
        return (None, False)  # NULL 隔离

    STRATS = (("baseline", "全放行"), ("rule", "规则基线 contraction≤1.0"), ("lr", f"LR≥{THRESH} 放行"))
    report = {}
    for seg in ("train", "val", "test"):
        frame = [r for r in rows if r["segment"] == seg and r["validity_reason"] != "NULL"]
        seg_rep = {"n_evaluable": len(frame),
                   "n_isolated_null": sum(1 for r in rows if r["segment"] == seg and r["validity_reason"] == "NULL"),
                   "note": ("样本内(in-sample)" if seg == "train" else
                            "主判读" if seg == "val" else "描述(非新盲测)")}
        for key, label in STRATS:
            opps, kept = [], []
            for r in frame:
                opp, kept_b = strat_outcome(r, key)
                if opp is None:
                    continue
                opps.append((r["E"], opp))
                if kept_b:
                    kept.append((r["E"], opp, float(r["C_buy"] or 0)))
            by_date = defaultdict(list)
            for d, o in opps:
                by_date[d].append(o)
            kby = defaultdict(list)
            for d, o, _ in kept:
                kby[d].append(o)
            n_kept = len(kept)
            util = (sum(c for _, _, c in kept) / (50000.0 * len(opps))) if opps else None
            seg_rep[key] = {
                "label": label,
                "opp_boot": cluster_boot(by_date),
                "traded_subset_boot": cluster_boot(kby),
                "n_kept_trades": n_kept,
                "filter_rate": round(1 - n_kept / max(1, sum(1 for r in frame if r["traded_closed"])), 4)
                if any(r["traded_closed"] for r in frame) else None,
                "capital_utilization": round(util, 6) if util is not None else None,
                "left_tail": sorted([o for _, o, _ in kept])[:min(10, n_kept)],
                "rejects_all_trades": (n_kept == 0),
                "reject_all_note": ("拒绝所有交易: 不得称为成功 (硬性判读规则)" if n_kept == 0 else None),
            }
        report[seg] = seg_rep
        print(seg, {k: (v.get("n_kept_trades"), (v.get("opp_boot") or {}).get("mean"))
                    for k, v in seg_rep.items() if isinstance(v, dict)})

    artifact = {"phase": "GCP_247_ACTION_FILTER_FIT_V1_MODEL",
                "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "network_requests": 0, "research_fit_only": True,
                "contract": "contracts/action_filter_model_research_contract_v1.md (FROZEN + §8.7)",
                "hash_verification": hv,
                "model": {"type": "L2-LogisticRegression", "C": C_FIX, "solver": "liblinear",
                          "threshold": THRESH, "seed": SEED,
                          "feature_names": feat_names, "coef": coef, "intercept": inter,
                          "scaler": {"mean": mu, "std": sd},
                          "encoder_categories": cats,
                          "train_in_sample_accuracy": round(train_acc, 4),
                          "train_n": len(train), "train_pos": int(y.sum())},
                "hard_rule": "拒绝所有交易不能称为成功: 保留成交数/成交子集绝对收益/资金利用率必须单列",
                "runner_sha256_note": "见报告 json runner_sha256",
                "report": report}
    write_checked(OUT_JSON, json.dumps(artifact, ensure_ascii=False, indent=1))

    def fmt(b):
        return "-" if not b else f"{b['mean']} [{b['ci_lower']}, {b['ci_upper']}]"

    L = ["# 共享行动过滤 LR 第一轮拟合报告 v1 (研究性; 非个性模型/非全量训练)", "",
         f"- 生成: {artifact['generated_at']} | 复验全过 | 零网络 | C=0.1 固定 / 阈值 ≥{THRESH} (§8.7 补签)",
         "- 标签=第〇阶段逐事件逐轨道记录; train=样本内; val=主判读(30 有效标签, 盈利仅 5 条, **结论可能不确定, 如实标注**); test=描述.", "",
         "## 拟合概况 (train, in-sample)", "",
         f"- n=114 (正 53/负 61); 样本内准确率 {round(train_acc,4)} (**样本内, 不夸大**)",
         f"- 系数: {json.dumps(coef, ensure_ascii=False)}; 截距 {inter}",
         "- 方向简述: 正系数=该特征越大越预测盈利(标准化后); 仅为样本内相关结构, 非因果.", "",
         "## 三策略对比 (同可评价样本框; 拒绝→现金 0; NULL 隔离不补 0)", "",
         "| 段 | 策略 | 机会收益均值 [95%CI] | 成交子集均值 [95%CI] | 保留成交 | 过滤率 | 资金利用率 | 左尾最差 |", 
         "|---|---|---|---|---|---|---|---|"]
    for seg in ("val", "train", "test"):
        s = report[seg]
        for key, _ in STRATS:
            v = s[key]
            L.append(f"| {seg}({s['note']}) | {v['label']} | {fmt(v['opp_boot'])} | {fmt(v['traded_subset_boot'])} | "
                     f"{v['n_kept_trades']} | {v['filter_rate']} | {v['capital_utilization']} | "
                     f"{v['left_tail'][0] if v['left_tail'] else '-'} |")
            if v["rejects_all_trades"]:
                L.append(f"| | ⚠ {v['reject_all_note']} | | | | | | |")
    L += ["", "## 硬性判读", "",
          "- '拒绝所有交易'情形: " + ("出现 -> 见上表 ⚠ 行, 不得称为成功" if any(
              report[seg][k]["rejects_all_trades"] for seg in report for k, _ in STRATS) else "未出现"),
          "- 经济评价为主: 相对全放行的改善必须同时看保留成交数/成交子集绝对收益/资金利用率, 不是分类准确率.", "",
          "## 声明", "",
          "- 研究性拟合; 不自动上线; 不追加训练/补采(328 挂起); 盈利概率高≠期望收益为正.", ""]
    write_checked(OUT_MD, "\n".join(L))
    print(f"fit done ({time.time()-t0:.1f}s); dir={dir_size(BASE)} B")


if __name__ == "__main__":
    main()
