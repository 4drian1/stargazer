# -*- coding: utf-8 -*-
"""第三步 e: 双路径共享轻量模型 · 固定配置研究性拟合 (签署+单独授权后执行).

严格按 DUAL_PATH_FIT_CONTRACT_V0_1 §1-§9 执行:
- 只读冻结输入 (r1_fit_features_v0_1_1.json), 复验全部上游哈希 == manifest.
- 仅 train H3 (77) 拟合; LR(C=0.1, l2, lbfgs/等价IRLS, class_weight=None, max_iter=1000, tol=1e-4, random_state=20260928).
- 数值列 z-score / 分类列 one-hot, 仅 train 拟合; 未知类别->0(参考水平)+标记, 不崩溃.
- 放行规则 p>=0.50.
- 评估: 三档对照(常数概率37/77 / 简单规则contraction<=1.0 / LR) + 全放行; 主比较=同事件机会收益差 (LR-全放行, LR-简单规则);
  E日簇重抽保配对 (cluster_boot); 自身收益CI与增量CI分报; 左尾; 不收敛/非有限即停不换参.
- 零网络零采集; 不声称泛化有效/继承旧门; 完成三档对照即停.
产物: gcp_step3e_fit_fixed.json/.md; 审计 ENGINEERING-2026-10-08-35.
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

HERE = Path(__file__).resolve().parent
RUN = HERE / "gcp_round_b_small_sample_v1"
FEAT = RUN / "r1_fit_features_v0_1_1.json"
MANI = RUN / "r1_fit_manifest_v0_1_1.json"
OUT_JSON = RUN / "gcp_step3e_fit_fixed.json"
OUT_MD = RUN / "gcp_step3e_fit_fixed.md"
AUDIT = HERE / "gcp_audit_log.md"

SEED = 20261008
N_BOOT = 10000
BUDGET = 55 * 1024 * 1024


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def dir_size(p: Path) -> int:
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file())


def cluster_boot(pairs_by_date: dict, n_boot=N_BOOT, seed=SEED):
    """E 日簇有放回重抽, 同簇保留完整配对, 事件等权均值; 双侧 95% CI. (复用 gcp_247_rule_compare_v1.cluster_boot)"""
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
            "mean": float(vals.mean()),
            "ci_lower": float(np.percentile(means, 2.5)),
            "ci_upper": float(np.percentile(means, 97.5)),
            "n_boot": n_boot, "seed": seed, "estimator": "event_equal_weight"}


def cluster_boot_diff(by_date_A: dict, by_date_B: dict, n_boot=N_BOOT, seed=SEED):
    """臂间同事件机会收益差 (差值的簇重抽). A,B 为各 E日->[net...]; 重抽簇后算 (meanA - meanB)."""
    dates = sorted(set(by_date_A) | set(by_date_B))
    n = len(dates)
    if n < 2:
        return None
    sumA = {c: float(np.sum(by_date_A.get(c, []))) for c in dates}
    cntA = {c: len(by_date_A.get(c, [])) for c in dates}
    sumB = {c: float(np.sum(by_date_B.get(c, []))) for c in dates}
    cntB = {c: len(by_date_B.get(c, [])) for c in dates}
    rng = np.random.default_rng(seed)
    deltas = np.empty(n_boot)
    for b in range(n_boot):
        picks = rng.integers(0, n, size=n)
        sA = sum(sumA[dates[p]] for p in picks); cA = sum(cntA[dates[p]] for p in picks)
        sB = sum(sumB[dates[p]] for p in picks); cB = sum(cntB[dates[p]] for p in picks)
        mA = sA / cA if cA > 0 else np.nan
        mB = sB / cB if cB > 0 else np.nan
        deltas[b] = mA - mB
    # 点估计 (全集)
    sA = float(np.sum([sumA[c] for c in dates])); cA = int(np.sum([cntA[c] for c in dates]))
    sB = float(np.sum([sumB[c] for c in dates])); cB = int(np.sum([cntB[c] for c in dates]))
    point = (sA / cA) - (sB / cB) if (cA > 0 and cB > 0) else np.nan
    finite = deltas[np.isfinite(deltas)]
    if finite.size == 0:
        return {"mean": float(point), "ci_lower": None, "ci_upper": None,
                "n_boot": n_boot, "seed": seed, "n_clusters": n, "finite_boot": 0}
    return {"mean": float(point),
            "ci_lower": float(np.percentile(finite, 2.5)),
            "ci_upper": float(np.percentile(finite, 97.5)),
            "n_boot": n_boot, "seed": seed, "n_clusters": n, "finite_boot": int(finite.size)}


def verify_frozen():
    man = json.loads(MANI.read_text(encoding="utf-8"))
    table = {
        "r1_exec_labels_v0_1_2.json": RUN / "r1_exec_labels_v0_1_2.json",
        "r1_label_admission_v0_1_2.json": RUN / "r1_label_admission_v0_1_2.json",
        "gcp_dual_path_coverage_probe.py": HERE / "gcp_dual_path_coverage_probe.py",
        "gcp_round_b_engine.py": HERE / "gcp_round_b_engine.py",
        "dual_path_execution_v0_1_1.py": HERE / "dual_path_execution_v0_1_1.py",
        "gcp_adj_cache_acceptance.json": RUN / "gcp_adj_cache_acceptance.json",
        "gcp_step3c_fit_readiness.py": HERE / "gcp_step3c_fit_readiness.py",
        "r1_fit_readiness_v0_1_1.json": RUN / "r1_fit_readiness_v0_1_1.json",
        "gcp_step3d_fit_features.py": HERE / "gcp_step3d_fit_features.py",
        "r1_fit_features_v0_1_1.json": FEAT,
    }
    res, ok = {}, True
    for role, expect in man["sha256"].items():
        actual = sha256_file(table[role])
        m = (actual == expect)
        ok &= m
        res[role] = {"expect": expect, "actual": actual, "match": m}
    if not ok:
        bad = [k for k, v in res.items() if not v["match"]]
        raise SystemExit(f"冻结输入复验失败: {bad} -> 中止, 不拟合")
    return res


def fit_lr(X, y, C=0.1, max_iter=1000, tol=1e-4):
    """L2 logistic (sklearn 优先; 否则 IRLS 等价凸解). 返回 beta, solver_name, converged."""
    try:
        from sklearn.linear_model import LogisticRegression
        m = LogisticRegression(C=C, penalty="l2", solver="lbfgs", class_weight=None,
                               fit_intercept=False, max_iter=max_iter, tol=tol, random_state=20260928)
        m.fit(X, y)
        return m.coef_[0], m.intercept_[0], "sklearn.LogisticRegression(lbfgs,C=0.1)", True
    except Exception:
        pass
    n, p = X.shape
    lam = 1.0 / C
    beta = np.zeros(p)
    pen = np.ones(p); pen[0] = 0.0  # 截距不罚
    Xt = X.T
    converged = False
    for _ in range(max_iter):
        eta = X @ beta
        p_hat = 1.0 / (1.0 + np.exp(-eta))
        g = (Xt @ (p_hat - y)) / n + lam * pen * beta
        S = p_hat * (1 - p_hat)
        H = (Xt * S) @ X / n + np.diag(lam * pen)
        try:
            step = np.linalg.solve(H, g)
        except np.linalg.LinAlgError:
            step = np.linalg.lstsq(H, g, rcond=None)[0]
        beta = beta - step
        if np.max(np.abs(g)) < tol:
            converged = True
            break
    return beta, beta[0], "IRLS(l2,C=0.1,Newton等价lbfgs)", converged


def main():
    t0 = time.time()
    print("=" * 70)
    print("第三步 e: 固定配置研究性拟合 (零网络/零采集)")
    print("=" * 70)
    fv = verify_frozen()
    print(f"[复验] 冻结输入全过 ({len(fv)} 项)")

    feat = json.loads(FEAT.read_text(encoding="utf-8"))
    rows = feat["H3_primary"]
    # 仅用有标签事件
    rows = [r for r in rows if r.get("net") is not None]
    print(f"[人口] H3 准入事件 (有标签) = {len(rows)}")

    by_seg = defaultdict(list)
    for r in rows:
        by_seg[r["segment"]].append(r)
    for s in ("train", "val", "test"):
        print(f"  {s}: {len(by_seg[s])}")

    def feats(rs):
        return np.array([[r["contraction"], r["vol_cv10"],
                          1.0 if r["setup"] != REF_SETUP else 0.0,
                          1.0 if r["branch"] != REF_BRANCH else 0.0] for r in rs], dtype=float)

    # 列顺序: [intercept, contraction_z, vol_cv10_z, setup_nonref, branch_nonref]
    train = by_seg["train"]
    REF_SETUP = sorted({r["setup"] for r in train})[0]
    REF_BRANCH = sorted({r["branch"] for r in train})[0]
    Xtr_raw = feats(train); ytr = np.array([r["label"] for r in train], dtype=float)
    mu = Xtr_raw[:, :2].mean(0); sd = Xtr_raw[:, :2].std(0)
    sd = np.where(sd == 0, 1.0, sd)

    def design(rs):
        X = feats(rs)  # (n,4): contraction, vol_cv10, setup_flag, branch_flag
        n = X.shape[0]
        Xz = np.zeros((n, 5))
        Xz[:, 0] = 1.0                       # intercept (fit_intercept=False)
        Xz[:, 1] = (X[:, 0] - mu[0]) / sd[0]  # contraction z
        Xz[:, 2] = (X[:, 1] - mu[1]) / sd[1]  # vol_cv10 z
        Xz[:, 3] = X[:, 2]                    # setup (drop-first, ref=REF_SETUP)
        Xz[:, 4] = X[:, 3]                    # branch (drop-first, ref=REF_BRANCH)
        return Xz

    Xtr = design(train)
    beta, intercept, solver, converged = fit_lr(Xtr, ytr)
    print(f"[拟合] solver={solver}; converged={converged}; 截距={intercept:.4f}")
    if not converged:
        raise SystemExit("LR 未收敛 -> 停止, 不换参 (合同 §3.1)")
    coef = {"intercept": float(beta[0]),
            "contraction_z": float(beta[1]), "vol_cv10_z": float(beta[2]),
            "setup_nonref(1=非%s)" % REF_SETUP: float(beta[3]),
            "branch_nonref(1=非%s)" % REF_BRANCH: float(beta[4])}

    def proba(rs):
        X = design(rs)
        eta = X @ beta
        p = 1.0 / (1.0 + np.exp(-eta))
        if not np.all(np.isfinite(p)):
            raise SystemExit("预测含非有限值 -> 停止, 不换参 (合同 §3.1)")
        return p

    const_prob = float(ytr.mean())  # 37/77
    print(f"[基线] 常数概率={const_prob:.4f} (37/77)")

    # 各臂评估
    def arm_eval(rs, pass_mask, p=None):
        passed = [rs[i] for i in range(len(rs)) if pass_mask[i]]
        nets = [r["net"] for r in passed]
        by_date = defaultdict(list)
        for r in passed:
            by_date[int(r["E"])].append(r["net"])
        boot = cluster_boot(by_date) if nets else None
        return {"pass": len(passed), "filter_rate": len(passed) / len(rs) if rs else 0.0,
                "n_evaluable": len(nets),
                "own_return_mean": float(np.mean(nets)) if nets else None,
                "own_return_ci": [boot["ci_lower"], boot["ci_upper"]] if boot else None,
                "brier": float(np.mean((np.array(p) - np.array([r["label"] for r in rs])) ** 2)) if p is not None else None}

    segments_out = {}
    increments_out = {}
    left_tail_out = {}
    for s in ("train", "val", "test"):
        rs = by_seg[s]
        if not rs:
            continue
        p = proba(rs)
        net_arr = np.array([r["net"] for r in rs])
        c_arr = np.array([r["contraction"] for r in rs])
        all_mask = np.ones(len(rs), bool)
        sr_mask = c_arr <= 1.0
        lr_mask = p >= 0.50
        arms = {
            "all_pass": arm_eval(rs, all_mask),
            "simple_rule": arm_eval(rs, sr_mask),
            "lr": arm_eval(rs, lr_mask, p),
            "constant_prob": {"brier": float(np.mean((const_prob - np.array([r["label"] for r in rs])) ** 2))},
        }
        # 增量 (簇重抽)
        by_all = defaultdict(list); by_sr = defaultdict(list); by_lr = defaultdict(list)
        for i, r in enumerate(rs):
            by_all[int(r["E"])].append(r["net"])
            if sr_mask[i]:
                by_sr[int(r["E"])].append(r["net"])
            if lr_mask[i]:
                by_lr[int(r["E"])].append(r["net"])
        inc = {
            "LR_minus_allpass": cluster_boot_diff(by_all, by_lr),
            "LR_minus_simple": cluster_boot_diff(by_sr, by_lr),
            "simple_minus_allpass": cluster_boot_diff(by_all, by_sr),
        }
        # 左尾
        order = np.argsort(net_arr)
        worst = net_arr[order[:min(10, len(net_arr))]]
        n_le = int(np.sum(net_arr <= -0.05))
        left_tail_out[s] = {"worst_min10": [float(x) for x in worst],
                            "min_net": float(net_arr.min()),
                            "n_le_minus5pct": n_le}
        segments_out[s] = {"n": len(rs), "n_evaluable": len(rs), "arms": arms}
        increments_out[s] = inc
        print(f"[{s}] n={len(rs)} | LR pass={arms['lr']['pass']} "
              f"own={arms['lr']['own_return_mean']} | LR-全放行 Δ={inc['LR_minus_allpass']['mean']:.4f} "
              f"[{inc['LR_minus_allpass']['ci_lower']:.4f},{inc['LR_minus_allpass']['ci_upper']:.4f}]")

    # 描述性判定 (非晋级裁决)
    def describe(inc):
        if inc is None:
            return "证据不足(簇<2)"
        m, lo, hi = inc["mean"], inc["ci_lower"], inc["ci_upper"]
        if m > 0 and lo > 0:
            return "正增量(CI下界>0)"
        if m < 0 and hi < 0:
            return "负增量(CI上界<0)"
        return "证据不足(CI含0)"

    verdicts = {s: {"LR_minus_allpass": describe(increments_out[s]["LR_minus_allpass"]),
                    "LR_minus_simple": describe(increments_out[s]["LR_minus_simple"])}
               for s in segments_out}

    out = {
        "contract": "contracts/dual_path_fit_contract_v0_1_draft.md (已签署; 拟合单独授权)",
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "network_requests": 0, "training_performed": True, "data_collection_performed": False,
        "frozen_input_verification": {k: v["match"] for k, v in fv.items()},
        "model": {"solver": solver, "C": 0.1, "penalty": "l2", "class_weight": None,
                  "max_iter": 1000, "tol": 1e-4, "random_state": 20260928,
                  "converged": bool(converged), "n_train": len(train), "constant_prob": const_prob},
        "coefficients": coef,
        "ref_levels": {"setup_ref": REF_SETUP, "branch_ref": REF_BRANCH},
        "standardization": {"contraction_mean": float(mu[0]), "contraction_std": float(sd[0]),
                            "vol_cv10_mean": float(mu[1]), "vol_cv10_std": float(sd[1])},
        "segments": segments_out,
        "increments": increments_out,
        "left_tail": left_tail_out,
        "verdicts_descriptive": verdicts,
        "primary_read": "val",
        "warning": "小样本研究性拟合; 覆盖34.1%限制已接受; 不声称泛化有效/继承旧门; 配对结果不外延全人口; 非晋级裁决",
        "dir_bytes_after": dir_size(RUN),
        "elapsed_s": round(time.time() - t0, 1),
    }
    text = json.dumps(out, ensure_ascii=False, indent=1)
    if dir_size(RUN) + len(text.encode("utf-8")) > BUDGET:
        raise RuntimeError("预算超限, 停机")
    OUT_JSON.write_text(text, encoding="utf-8", newline="")

    # MD
    L = ["# 双路径共享轻量模型 · 固定配置拟合 v1 (零网络, 合同 v0.1)", "",
         f"- 生成: {out['generated_at']} | 复验冻结输入全过 | 用时 {out['elapsed_s']}s",
         f"- 模型: {solver}; C=0.1, l2, class_weight=None, max_iter=1000, tol=1e-4, seed=20260928; 收敛={converged}; n_train={len(train)}",
         f"- 常数概率基线 = {const_prob:.4f} (37/77, 仅 Brier 评价, 非放行)", "",
         "## 系数 (z-score 数值列, 参考水平 one-hot)", ""]
    for k, v in coef.items():
        L.append(f"- {k}: {v:+.4f}")
    L += ["", "## 各段各臂结果 (val 主判读; train 拟合/样本内; test 描述性)", "",
          "| 段 | 臂 | 放行数 | 过滤率 | 自身收益均值 | 自身收益[95%CI] | Brier |",
          "|---|---|---|---|---|---|---|"]
    for s in ("train", "val", "test"):
        if s not in segments_out:
            continue
        seg = segments_out[s]
        for arm in ("all_pass", "simple_rule", "lr", "constant_prob"):
            a = seg["arms"][arm]
            own = a.get("own_return_mean")
            ci = a.get("own_return_ci")
            own_s = f"{own*100:.2f}%" if own is not None else "-"
            ci_s = f"[{ci[0]*100:.2f}%,{ci[1]*100:.2f}%]" if ci else "-"
            br = a.get("brier")
            br_s = f"{br:.4f}" if br is not None else "-"
            L.append(f"| {s} | {arm} | {a.get('pass','-')} | {a.get('filter_rate',0)*100:.1f}% | {own_s} | {ci_s} | {br_s} |")
    L += ["", "## 同事件机会收益差 (增量, E日簇重抽)", "",
          "| 段 | 对比 | 均值 | [95%CI] | 描述判定 |",
          "|---|---|---|---|---|"]
    for s in ("train", "val", "test"):
        if s not in increments_out:
            continue
        for key, label in (("LR_minus_allpass", "LR−全放行"), ("LR_minus_simple", "LR−简单规则"), ("simple_minus_allpass", "简单−全放行")):
            inc = increments_out[s][key]
            if inc is None:
                L.append(f"| {s} | {label} | - | - | 簇<2 |")
                continue
            L.append(f"| {s} | {label} | {inc['mean']*100:.2f}% | [{inc['ci_lower']*100:.2f}%,{inc['ci_upper']*100:.2f}%] | {describe(inc)} |")
    L += ["", "## 左尾 (实际成交轨道, net=C_sell/C_buy-1)", ""]
    for s in ("train", "val", "test"):
        if s not in left_tail_out:
            continue
        lt = left_tail_out[s]
        L.append(f"- {s}: 最差10例={['%.2f%%' % (x*100) for x in lt['worst_min10']]}; "
                 f"全局最小={lt['min_net']*100:.2f}%; net≤−5% 个数={lt['n_le_minus5pct']}")
    L += ["", "## 声明", "",
          "- 非晋级裁决; 完成三档对照即停 (无阈值搜索/无上线).",
          "- 覆盖34.1%限制已显式接受 (非门语义); 结果不外延全人口.",
          "- RAW 尺度证据仅限 E 日收盘尺度一致; 不声称全部历史/整条持仓路径同尺度.",
          "- 32 只冲突码继续隔离; 未修全部历史数据; 未补采 354 对; 未训练逐股模型.", ""]
    OUT_MD.write_text("\n".join(L), encoding="utf-8", newline="")

    log = f"""

## 工程记录 ENGINEERING-2026-10-08-35：双路径共享轻量模型 · 固定配置拟合（第三步 e）

- **执行依据**：合同 DUAL_PATH_FIT_CONTRACT_V0_1 已签署 + 用户 2026-10-08「审阅通过，可以进行下一步」单独授权。
- **冻结复验**：全部上游输入 SHA256 == r1_fit_manifest_v0_1_1.json（10 项全过），拟合前未篡改。
- **拟合**：仅 train H3（77）拟合；LR = {solver}；C=0.1 / l2 / class_weight=None / max_iter=1000 / tol=1e-4 / seed=20260928；收敛={converged}；常数概率基线=37/77≈{const_prob:.4f}（仅 Brier，非放行）。
- **系数**：{coef}。
- **三档对照**：全放行 / 简单规则(contraction≤1.0) / LR(p≥0.50)；主比较=同事件机会收益差（LR−全放行、LR−简单规则），E 日簇重抽保配对，自身收益 CI 与增量 CI 分报；左尾单列。
- **范围**：零网络零采集零补采；不修 32 只冲突码、不补采 354 对、不训练逐股模型、不搜阈值、不上线。
- **预算**：RUN 累计 {dir_size(RUN)} B / 55 MiB。产物 gcp_step3e_fit_fixed.json/.md。
- **结论姿态**：描述性判定（非晋级裁决）；覆盖 34.1% 限制已接受；结果不外延全人口。
"""
    if "ENGINEERING-2026-10-08-35" not in AUDIT.read_text(encoding="utf-8"):
        with open(AUDIT, "a", encoding="utf-8", newline="") as f:
            f.write(log)
    print(f"写入: {OUT_JSON.name} / {OUT_MD.name}; 审计 -35; 累计 {dir_size(RUN)} B")


if __name__ == "__main__":
    main()
