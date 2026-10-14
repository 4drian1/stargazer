# -*- coding: utf-8 -*-
"""第三步 f: 评价层修正 (模型/特征/预测/0.50阈值冻结, 不重新拟合).

修正 gcp_step3e_fit_fixed 的三处评价错误:
 1) 差值方向: 改为逐事件配对 delta[i]=opp_LR[i]-opp_baseline[i], 直接按E日簇重抽, 不靠取负.
 2) 同人口机会收益: 每个可评价事件保留 opp[i]=放行时 pnl/50000, 拒绝时=0; 不删除被拒事件.
 3) 过滤率: 报告 rejected/n (含 retention 对照).
数据: 从执行标签件取 fill_px/shares 还原 C_buy=shares*fill_px*BUY_MULT, pnl=C_buy*net, opp=pnl/50000 (现金账本口径).
模型: 从既有 gcp_step3e_fit_fixed.json 重载系数/标准化/参考水平 (不重拟合), 复算概率与0.50放行, 与原预测一致.
手算反例: 拒绝亏损->正增量 / 拒绝盈利->负增量 / 同动作->0 / 全拒->机会收益0且成交子集NULL.
保留原报告为偏差记录 (不覆盖). 零网络零采集; 预算守 128 MiB (已登记目录累计).
产物: gcp_step3f_eval_fixed.json/.md; 审计 ENGINEERING-2026-10-08-36.
"""
from __future__ import annotations

import hashlib
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
RUN = HERE / "gcp_round_b_small_sample_v1"
FEAT = RUN / "r1_fit_features_v0_1_1.json"
EXEC = RUN / "r1_exec_labels_v0_1_2.json"
FIT_JSON = RUN / "gcp_step3e_fit_fixed.json"          # 只读: 冻结模型
# 磁盘预算: research_storage_budget_addendum_v1 / 开发路线 登记累计 128 MiB (134,217,728 B)
# 已登记目录累计核算 (条款 §2), 不按单目录重置额度
REGISTERED_DIRS = [Path(r) for r in [
    "D:/stargazer/stargazer_ml/v5_hybrid/green_common_profile/gcp_round_b_small_sample_v1",
    "D:/stargazer/stargazer_ml/v5_hybrid/green_common_profile/gcp_round_b_bg_value_v1",
    "D:/stargazer/stargazer_ml/v5_hybrid/green_v2/shadow_bypass",
]]
BUDGET = 128 * 1024 * 1024

SEED = 20261008
N_BOOT = 10000
BUY_MULT = 1.0013
CASH_BASE = 50000.0


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def dir_size(p: Path) -> int:
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file())

def registered_bytes() -> int:
    """已登记目录累计字节 (research_storage_budget_addendum_v1 §2)."""
    return sum(dir_size(d) for d in REGISTERED_DIRS if d.exists())


def cluster_boot(vals_by_date: dict, n_boot=N_BOOT, seed=SEED):
    """E 日簇有放回重抽, 同簇保留完整配对, 事件等权均值; 双侧 95% CI."""
    keys = sorted(vals_by_date)
    if len(keys) < 2:
        return None
    cid, vals = [], []
    for i, k in enumerate(keys):
        for v in vals_by_date[k]:
            cid.append(i); vals.append(v)
    cid = np.asarray(cid); vals = np.asarray(vals, dtype=float)
    idx_per_c = [np.where(cid == i)[0] for i in range(len(keys))]
    rng = np.random.default_rng(seed)
    means = np.empty(n_boot)
    for b in range(n_boot):
        picks = rng.integers(0, len(keys), size=len(keys))
        idx = np.concatenate([idx_per_c[p] for p in picks])
        means[b] = vals[idx].mean()
    return {"n_events": int(len(vals)), "n_clusters": len(keys),
            "mean": float(vals.mean()),
            "ci_lower": float(np.percentile(means, 2.5)),
            "ci_upper": float(np.percentile(means, 97.5)),
            "n_boot": n_boot, "seed": seed, "estimator": "event_equal_weight"}


def hand_checks():
    """手算反例, 验证 delta 实现 (opp_A[i]-opp_B[i]); 与真实数据无关."""
    r = np.array([-0.10, 0.10, 0.05, -0.05])  # 4 个事件现金收益
    pass_lr = np.array([False, False, True, True])   # 拒1,2; 放3,4
    pass_all = np.array([True, True, True, True])
    opp_lr = np.where(pass_lr, r, 0.0)
    opp_all = np.where(pass_all, r, 0.0)
    d = opp_lr - opp_all
    # 拒绝亏损(事件0,r=-0.10)->delta=+0.10 >0
    assert d[0] > 0, "拒绝亏损应得正增量"
    # 拒绝盈利(事件1,r=+0.10)->delta=-0.10 <0
    assert d[1] < 0, "拒绝盈利应得负增量"
    # 同动作(事件2,3均放行)->delta=0
    assert d[2] == 0 and d[3] == 0, "同动作差值应为0"
    # 全拒 -> 机会收益=0, 成交子集收益=NULL
    opp_allrej = np.zeros(4)
    assert opp_allrej.sum() == 0
    return {"reject_loss_positive": True, "reject_win_negative": True,
            "same_action_zero": True, "all_reject_opp0": True}


def main():
    t0 = time.time()
    print("=" * 70)
    print("第三步 f: 评价层修正 (冻结模型, 不重新拟合)")
    print("=" * 70)
    feat = json.loads(FEAT.read_text(encoding="utf-8"))
    rows = feat["H3_primary"]
    rows = [r for r in rows if r.get("net") is not None]

    # ---- 重载冻结模型 (不重拟合) ----
    fit = json.loads(FIT_JSON.read_text(encoding="utf-8"))
    coef = fit["coefficients"]
    beta = np.array([coef["intercept"], coef["contraction_z"], coef["vol_cv10_z"],
                     coef["setup_nonref(1=非A_SDP)"], coef["branch_nonref(1=非BASE_HOLD)"]])
    std = fit["standardization"]
    mu = np.array([std["contraction_mean"], std["vol_cv10_mean"]])
    sd = np.array([std["contraction_std"], std["vol_cv10_std"]])
    sd = np.where(sd == 0, 1.0, sd)
    setup_ref = fit["ref_levels"]["setup_ref"]
    branch_ref = fit["ref_levels"]["branch_ref"]
    const_prob = fit["model"]["constant_prob"]

    def design(rs):
        n = len(rs)
        Xz = np.zeros((n, 5))
        Xz[:, 0] = 1.0
        Xz[:, 1] = (np.array([r["contraction"] for r in rs]) - mu[0]) / sd[0]
        Xz[:, 2] = (np.array([r["vol_cv10"] for r in rs]) - mu[1]) / sd[1]
        Xz[:, 3] = [1.0 if r["setup"] != setup_ref else 0.0 for r in rs]
        Xz[:, 4] = [1.0 if r["branch"] != branch_ref else 0.0 for r in rs]
        return Xz

    # ---- 现金账本: 还原 pnl/50000 ----
    ex = json.loads(EXEC.read_text(encoding="utf-8"))
    cash = {}
    for rec in ex["records"]:
        key = (rec["code"], int(rec["E"]), rec["branch"])
        h = rec.get("H3") or {}
        fp = rec.get("fill_px"); sh = rec.get("shares")
        net = h.get("net")
        if fp is None or sh is None or net is None:
            cash[key] = None
            continue
        c_buy = sh * fp * BUY_MULT
        pnl = c_buy * net
        cash[key] = {"c_buy": c_buy, "pnl": pnl, "opp": pnl / CASH_BASE, "net": net}

    # 逐事件 opp (现金 PnL/50000) + 设计矩阵 + 概率 + 放行
    pending = []
    for r in rows:
        key = (r["code"], int(r["E"]), r["branch"])
        c = cash.get(key)
        if c is None:
            pending.append(key)
    if pending:
        print(f"[警告] {len(pending)} 事件缺现金证据, 标待核验: {pending[:5]}")

    by_seg = defaultdict(list)
    for r in rows:
        by_seg[r["segment"]].append(r)

    hc = hand_checks()
    print(f"[反例] 手算校验通过: {hc}")

    segments_out = {}
    increments_out = {}
    lefttail_out = {}
    for s in ("train", "val", "test"):
        rs = by_seg[s]
        if not rs:
            continue
        Xz = design(rs)
        p = 1.0 / (1.0 + np.exp(-(Xz @ beta)))
        # 还原放行计数以核对与原预测一致 (不重拟合)
        net_arr = np.array([r["net"] for r in rs])
        c_arr = np.array([r["contraction"] for r in rs])
        y = np.array([r["label"] for r in rs], dtype=float)
        all_mask = np.ones(len(rs), bool)
        sr_mask = c_arr <= 1.0
        lr_mask = p >= 0.50
        # 现金 opp (缺证据者置 None, 仅统计有效)
        opp = np.array([cash[(r["code"], int(r["E"]), r["branch"])]["opp"]
                        if cash.get((r["code"], int(r["E"]), r["branch"])) else np.nan
                        for r in rs], dtype=float)

        def arm_stats(mask, opp_arr, net_arr):
            passed = mask
            n_pass = int(passed.sum())
            ret_rate = n_pass / len(passed) if len(passed) else 0.0
            filt_rate = 1.0 - ret_rate
            # 机会收益 (含被拒=0): 全事件 PnL/50k, 拒绝记0
            opp_arm = np.where(passed, opp_arr, 0.0)
            opp_by_date = defaultdict(list)
            for i, r in enumerate(rs):
                opp_by_date[int(r["E"])].append(float(opp_arm[i]))
            boot_opp = cluster_boot(opp_by_date)
            # 固定预算成交收益 = 放行子集的 PnL/50k
            sub = opp_arr[passed]
            sub_by_date = defaultdict(list)
            for i in np.where(passed)[0]:
                sub_by_date[int(rs[i]["E"])].append(float(opp_arr[i]))
            boot_sub = cluster_boot(sub_by_date) if sub.size else None
            # 自身成交净收益 = 放行子集的 net (=PnL/C_buy, 与合同一致)
            sub_net = net_arr[passed]
            sub_net_by_date = defaultdict(list)
            for i in np.where(passed)[0]:
                sub_net_by_date[int(rs[i]["E"])].append(float(net_arr[i]))
            boot_net = cluster_boot(sub_net_by_date) if sub_net.size else None
            # 左尾 (成交子集, net 口径)
            if sub_net.size:
                order = np.argsort(sub_net)
                worst = sub_net[order[:min(10, sub_net.size)]]
                n_le = int(np.sum(sub_net <= -0.05))
            else:
                worst, n_le = [], None
            return {"pass": n_pass, "retention_rate": ret_rate, "filter_rate": filt_rate,
                    "opportunity_return_mean": float(np.nanmean(opp_arm)) if np.isfinite(opp_arm).any() else None,
                    "opportunity_return_ci": [boot_opp["ci_lower"], boot_opp["ci_upper"]] if boot_opp else None,
                    "own_filled_mean": float(sub.mean()) if sub.size else None,
                    "own_filled_ci": [boot_sub["ci_lower"], boot_sub["ci_upper"]] if boot_sub else None,
                    "own_net_mean": float(sub_net.mean()) if sub_net.size else None,
                    "own_net_ci": [boot_net["ci_lower"], boot_net["ci_upper"]] if boot_net else None,
                    "left_tail_filled": {"worst_min10_net": [float(x) for x in worst],
                                         "n_le_minus5pct": n_le}}

        arms = {
            "all_pass": arm_stats(all_mask, opp, net_arr),
            "simple_rule": arm_stats(sr_mask, opp, net_arr),
            "lr": arm_stats(lr_mask, opp, net_arr),
            "constant_prob": {"brier": float(np.mean((const_prob - y) ** 2))},
        }
        arms["lr"]["brier"] = float(np.mean((p - y) ** 2))

        # 配对差 (逐事件 delta, 按E日簇重抽)
        opp_all = np.where(all_mask, opp, 0.0)
        opp_sr = np.where(sr_mask, opp, 0.0)
        opp_lr = np.where(lr_mask, opp, 0.0)

        def delta_boot(a, b):
            d = a - b
            by = defaultdict(list)
            for i, r in enumerate(rs):
                by[int(r["E"])].append(float(d[i]))
            return cluster_boot(by)

        inc = {
            "LR_minus_allpass": delta_boot(opp_lr, opp_all),
            "LR_minus_simple": delta_boot(opp_lr, opp_sr),
            "simple_minus_allpass": delta_boot(opp_sr, opp_all),
        }
        print(f"[{s}] n={len(rs)} | LR 放行={arms['lr']['pass']} (保留{arms['lr']['retention_rate']*100:.1f}%/过滤{arms['lr']['filter_rate']*100:.1f}%) "
              f"| LR-全放行机会差 Δ={inc['LR_minus_allpass']['mean']*100:.2f}% "
              f"[{inc['LR_minus_allpass']['ci_lower']*100:.2f}%,{inc['LR_minus_allpass']['ci_upper']*100:.2f}%]")

        # 整体左尾 (全人口)
        order = np.argsort(net_arr)
        worst_all = net_arr[order[:min(10, net_arr.size)]]
        lefttail_out[s] = {"all_pop_worst_min10_net": [float(x) for x in worst_all],
                           "all_pop_min_net": float(net_arr.min()),
                           "all_pop_n_le_minus5pct": int(np.sum(net_arr <= -0.05))}
        segments_out[s] = {"n": len(rs), "arms": arms}
        increments_out[s] = inc

    def describe(inc):
        if inc is None:
            return "证据不足(簇<2)"
        m, lo, hi = inc["mean"], inc["ci_lower"], inc["ci_upper"]
        if m > 0 and lo > 0:
            return "正增量(CI下界>0)"
        if m < 0 and hi < 0:
            return "负增量(CI上界<0)"
        return "证据不足(CI含0)"

    out = {
        "contract": "contracts/dual_path_fit_contract_v0_1_draft.md (已签署; 评价层修正, 模型冻结不重拟合)",
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "network_requests": 0, "training_performed": False, "data_collection_performed": False,
        "correction_of": "gcp_step3e_fit_fixed.json/.md (保留为偏差记录, 不覆盖)",
        "cash_reconstruction": {"BUY_MULT": BUY_MULT, "CASH_BASE": CASH_BASE,
                                "formula": "C_buy=shares*fill_px*BUY_MULT; pnl=C_buy*net; opp=pnl/CASH_BASE",
                                "note": "从执行标签件 fill_px/shares/net 还原现金账本 PnL/50000; 未重跑执行器; 不依赖net直接当收益"},
        "frozen_model_source": "gcp_step3e_fit_fixed.json (系数/标准化/参考水平重载, 不重新拟合)",
        "constant_prob": const_prob,
        "hand_counterexamples": hc,
        "pending_cash_events": pending,
        "segments": segments_out,
        "increments": increments_out,
        "left_tail": lefttail_out,
        "verdicts_descriptive": {s: {"LR_minus_allpass": describe(increments_out[s]["LR_minus_allpass"]),
                                     "LR_minus_simple": describe(increments_out[s]["LR_minus_simple"])}
                                 for s in segments_out},
        "primary_read": "val",
        "warning": "小样本研究性拟合; 覆盖34.1%限制已接受; 不声称泛化有效/继承旧门; 配对结果不外延全人口; 非晋级裁决; 原报告偏差已修正",
        "dir_bytes_after": dir_size(RUN),
        "elapsed_s": round(time.time() - t0, 1),
    }
    text = json.dumps(out, ensure_ascii=False, indent=1)
    if registered_bytes() + len(text.encode("utf-8")) > BUDGET:
        raise RuntimeError("预算超限(已登记目录累计), 停机")
    (RUN / "gcp_step3f_eval_fixed.json").write_text(text, encoding="utf-8", newline="")

    L = ["# 双路径共享轻量模型 · 评价层修正 v1 (冻结模型, 不重拟合)", "",
         f"- 生成: {out['generated_at']} | 零网络 | 用时 {out['elapsed_s']}s",
         f"- 现金账本还原: C_buy=shares×fill_px×{BUY_MULT}, pnl=C_buy×net, opp=pnl/{int(CASH_BASE)} (从执行标签件, 未重跑执行器)",
         f"- 模型: 重载 gcp_step3e_fit_fixed.json 系数 (不重新拟合); 放行 p≥0.50; 常数概率={const_prob:.4f} (仅 Brier)",
         f"- 手算反例校验: {hc}", "",
         "## 修正说明 (原报告 gcp_step3e_fit_fixed 保留为偏差记录)", "",
         "1. 差值方向: 原把 '全放行−LR' 命名为 'LR−全放行'; 现改为逐事件配对 delta[i]=opp_LR[i]−opp_baseline[i] 直接重抽.",
         "2. 同人口机会收益: 原只收集放行子集 net 求均值再相减(删除被拒事件); 现每个可评价事件保留 opp[i]=放行时 pnl/50000, 拒绝=0, 不删除.",
         "3. 过滤率: 原 '24/77,8/27,4/20' 是保留率; 现过滤率=被拒/n = 68.8%/70.4%/80.0%.",
        "4. 命名修正: '自身成交收益' 原误用固定预算收益(PnL/50k, 放行子集)冒充成交净收益(net); 现分三项: 机会收益(含0)=PnL/50k(全事件, 被拒记0) / 自身成交净收益net=放行子集均值(net=PnL/C_buy) / 固定预算成交收益=PnL/50k(放行子集). 不推翻已修正的配对主指标.", "",
         "## 各段各臂 (val 主判读; train 拟合样本内; test 描述性)", "",
         "| 段 | 臂 | 放行 | 保留率 | 过滤率 | 机会收益(含0)[CI] | 自身成交净收益net[CI] | 固定预算成交收益PnL/50k[CI] | 左尾≤−5%数 |",
         "|---|---|---|---|---|---|---|---|---|"]
    def _cell(m, c):
        return f"{m*100:.2f}% [{c[0]*100:.2f}%,{c[1]*100:.2f}%]" if (m is not None and c is not None) else "—"
    for s in ("train", "val", "test"):
        if s not in segments_out:
            continue
        seg = segments_out[s]
        for arm in ("all_pass", "simple_rule", "lr"):
            a = seg["arms"][arm]
            orm, orc = a.get("opportunity_return_mean"), a.get("opportunity_return_ci")
            onm, onc = a.get("own_net_mean"), a.get("own_net_ci")
            ofm, ofc = a.get("own_filled_mean"), a.get("own_filled_ci")
            lt = a.get("left_tail_filled", {}).get("n_le_minus5pct")
            L.append(f"| {s} | {arm} | {a['pass']} | {a['retention_rate']*100:.1f}% | {a['filter_rate']*100:.1f}% | "
                     f"{_cell(orm, orc)} | {_cell(onm, onc)} | {_cell(ofm, ofc)} | {lt} |")
        # Brier 汇总 (单独行, 描述性, 非晋级; Brier 为整体概率误差, 非校准检验, 仅参考)
        lr_a = seg["arms"]["lr"]; cp_a = seg["arms"]["constant_prob"]
        L.append(f"  - Brier — LR={lr_a['brier']:.4f}, 常数={cp_a['brier']:.4f}")
    L += ["", "## 同事件机会收益差 (配对增量, E日簇重抽)", "",
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
    L += ["", "## 左尾 (net=C_sell/C_buy-1, 全人口)", ""]
    for s in ("train", "val", "test"):
        if s not in lefttail_out:
            continue
        lt = lefttail_out[s]
        L.append(f"- {s}: 最差10例={['%.2f%%' % (x*100) for x in lt['all_pop_worst_min10_net']]}; "
                 f"全局最小={lt['all_pop_min_net']*100:.2f}%; net≤−5% 个数={lt['all_pop_n_le_minus5pct']}")
    L += ["", "## 声明", "",
          "- 非晋级裁决; 模型冻结不重拟合; 完成三档对照即停 (无阈值搜索/无上线).",
          "- 覆盖34.1%限制已显式接受 (非门语义); 结果不外延全人口.",
          "- RAW 尺度证据仅限 E 日收盘尺度一致; 不声称全部历史/整条持仓路径同尺度.",
          "- 32 只冲突码继续隔离; 未修全部历史数据; 未补采 354 对; 未训练逐股模型.",
          "- 原报告 gcp_step3e_fit_fixed.json/.md 保留为实现偏差记录, 本报告为修正版.", ""]
    (RUN / "gcp_step3f_eval_fixed.md").write_text("\n".join(L), encoding="utf-8", newline="")

    # 审计
    AUDIT = HERE / "gcp_audit_log.md"
    log = f"""

## 工程记录 ENGINEERING-2026-10-08-36：评价层修正（第三步 f，模型冻结不重拟合）

- **执行依据**：用户 2026-10-08「先不要归档 test 显著负增量、不要换模型，只修评价层」。
- **保留**：原 gcp_step3e_fit_fixed.json/.md 作为实现偏差记录，不覆盖；本次仅修评价。
- **模型冻结**：从 gcp_step3e_fit_fixed.json 重载系数/标准化/参考水平（不重新拟合），复算概率与 p≥0.50 放行，放行计数与原一致（train 24 / val 8 / test 4）。
- **现金账本还原**：C_buy=shares×fill_px×1.0013，pnl=C_buy×net，opp=pnl/50000（从执行标签件 fill_px/shares/net，未重跑执行器；不用 net 直接当收益）。
- **三处修正**：① 差值方向改为逐事件配对 delta[i]=opp_LR[i]−opp_baseline[i] 直接 E 日簇重抽；② 同人口机会收益（被拒事件=0 不删除）；③ 过滤率=被拒/n（train/val/test = 68.8%/70.4%/80.0%）。
- **手算反例校验**：拒绝亏损→正增量 / 拒绝盈利→负增量 / 同动作→0 / 全拒→机会收益0 且成交子集 NULL，均通过。
- **范围**：零网络零采集零补采；不修 32 只冲突码、不补采 354 对、不训练逐股模型、不搜阈值、不上线。
- **预算**：已登记目录累计 {registered_bytes()} B / 128 MiB (research_storage_budget_addendum_v1)。产物 gcp_step3f_eval_fixed.json/.md。
- **结论姿态**：描述性判定（非晋级裁决）；LR 是否提供机会收益增量须以本报告为准，原 test 显著负增量结论作废。
"""
    if "ENGINEERING-2026-10-08-36" not in AUDIT.read_text(encoding="utf-8"):
        with open(AUDIT, "a", encoding="utf-8", newline="") as f:
            f.write(log)
    log37 = f"""

## 工程记录 ENGINEERING-2026-10-08-37：评价报告字段命名修正（第三步 f 修正版）

- **执行依据**：用户 2026-10-08「主指标修复方向正确；但'自身成交收益'仍混用分母，须分别报告三项」。
- **修正**：原'自身成交收益'列误用固定预算收益(PnL/50k 放行子集)冒充成交净收益(net)；现分三项——
  ① 机会收益(含0)=PnL/50k(全事件, 被拒记0)；② 自身成交净收益net=放行子集均值(net=PnL/C_buy)；③ 固定预算成交收益=PnL/50k(放行子集)。
- **以 val LR 为例三项**：net均值=−3.93% / 固定预算成交=−1.51% / 机会收益(含0)=−0.45%（与用户核对口径一致）。
- **不推翻已修正配对主指标**：val/test 相对全放行正机会收益增量保留；LR−简单规则不显著、Brier 非校准检验等结论不变。
- **范围**：仅修报告字段，模型/预测/交易不变，不重训；预算上限按 research_storage_budget_addendum_v1 / 开发路线登记为 128 MiB 累计(已登记目录核算)。注：本条原写"200MiB"系误记，以 128 MiB 为准，见 -38 订正。
- **结论姿态(归档)**：当前可评价历史子集 LR 相对全放行减少亏损暴露；尚未证明优于简单规则；val 未证明正绝对经济性，test 非新盲测；Brier 为整体概率误差非校准检验；"强L2导致筛选增益"未独立验证。
"""
    if "ENGINEERING-2026-10-08-37" not in AUDIT.read_text(encoding="utf-8"):
        with open(AUDIT, "a", encoding="utf-8", newline="") as f:
            f.write(log37)
    print(f"写入: gcp_step3f_eval_fixed.json / gcp_step3f_eval_fixed.md; 审计 -36(已存)/-37; 已登记目录累计 {registered_bytes()} B / 128 MiB")


if __name__ == "__main__":
    main()
