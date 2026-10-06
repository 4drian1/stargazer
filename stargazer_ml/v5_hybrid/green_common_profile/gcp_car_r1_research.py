"""CAR R1 受限研究训练: 常数概率基线 -> CAR-G -> CAR-LR (2026-10-01, 用户授权).

授权范围 (gcp_car_v0_2_statistical_plan.md + R1 授权):
  - 仅用冻结口径 A、冻结候选与时间边界; B 口径只作对照
  - 全部拟合 (常数/分组/收缩/标准化) 仅训练段; 验证测试不用于重选规则或特征
  - 增量门 (预注册, 先于结果固定):
      验证段 (20260309-20260609) 配对日期聚类 bootstrap (2000 次),
      d_Brier = Brier_constant - Brier_CARG; 点估计>0 且 95% CI 下界>0 才算增量
      无增量 -> 停止, 不跑 LR, 不调参
  - 有增量才跑固定配置 LR (L2, C=1; 连续特征训练段标准化)
  - 报告 Brier/校准/不确定性, A/B 与月份分层
  - 纯本地计算, 0 网络请求; 测试段仅称既有留出段 (汇总已查看)
  - 研究"收回概率能否预测得更好", 非买入晋级; H-E/H3/买入建议继续锁定
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "step9_preflight"))
sys.path.insert(0, str(HERE))
from data_source_provenance import get_daily_parquet  # noqa: E402
import gcp_round_b_engine as E  # noqa: E402
from gcp_car_v0_2_extractor import detect_signal_v02  # noqa: E402

OUT = HERE / "gcp_round_b_small_sample_v1"
SNAP_DIRS = (OUT / "minute", OUT / "minute_ext", OUT / "minute_car")
OLD_REPORT = OUT / "car_v0_2_full_batch_report.json"
TOPUP_REPORT = OUT / "car_v0_2_train_top_up_report.json"
R1_REPORT = OUT / "car_r1_results.json"

TRAIN_E_MAX = 20260305
VAL_E_MIN, VAL_E_MAX = 20260309, 20260609
TEST_E_MIN = 20260610

BOOT_N = 2000
BOOT_SEED = 20261001
N_CALIB_BINS = 5
WIDTH_N_BINS = 3


# ---------------- 数据装载 ----------------

def load_rows_and_gates():
    """合并两批次候选与门3结果, 返回 rows (含标签) 与 gate map."""
    old = json.loads(OLD_REPORT.read_text(encoding="utf-8"))
    top = json.loads(TOPUP_REPORT.read_text(encoding="utf-8"))
    gate: dict = {}
    for it in old["engineering_gates"]["gate3_scale"]["results"]:
        gate[(it["code"], it["E"])] = it
    for it in top["new_gate_results"]:
        gate[(it["code"], it["E"])] = it
    rows = []
    for r in old["per_candidate"] + top["new_per_candidate"]:
        rows.append(r)
    return rows, gate


def included_caliber_a(r, gate):
    it = gate.get((r["code"], r["E"]))
    if it is None:
        return False
    sc = it["sub_checks"]
    if sc.get("data_missing"):
        return False
    if not it["pass"]:
        return False
    return r.get("reclaim_status") in ("RECLAIMED", "NOT_RECLAIMED")


def segment_of(e_date):
    if e_date <= TRAIN_E_MAX:
        return "train"
    if VAL_E_MIN <= e_date <= VAL_E_MAX:
        return "val"
    if e_date >= TEST_E_MIN:
        return "test"
    return None


def load_minute_cache(need_keys):
    needs_by_code: dict = {}
    for c, d in need_keys:
        needs_by_code.setdefault(c, set()).add(d)
    cache: dict = {}
    for sd in SNAP_DIRS:
        if not sd.exists():
            continue
        for pf in sd.glob("*.parquet"):
            code = pf.stem
            want = needs_by_code.get(code)
            if want is None:
                continue
            try:
                m = pd.read_parquet(pf)
                if "date" in m and not m.empty:
                    day_int = m["date"].astype(str).str[:8].astype(int)
                    for d in pd.unique(day_int):
                        if int(d) in want:
                            cache.setdefault((code, int(d)), []).append(
                                m[day_int == int(d)].copy())
            except Exception:
                pass
    return cache


def build_features(rows, gate):
    panel = get_daily_parquet()
    panel["code"] = panel["code"].astype(str).str.zfill(6)
    panel["date"] = panel["date"].astype(int)
    cal = sorted(int(d) for d in panel["date"].unique())
    cal_idx = {d: i for i, d in enumerate(cal)}
    daily_by_code = {c: g.reset_index(drop=True) for c, g in panel.groupby("code")}

    elig = [r for r in rows if included_caliber_a(r, gate)]
    need_keys = {(r["code"], r["E"]) for r in elig}
    cache = load_minute_cache(need_keys)

    feats = []
    n_ts_mismatch = 0
    n_missing_cache = 0
    for r in elig:
        code, e_date = r["code"], r["E"]
        e_key = (code, e_date)
        if e_key not in cache or not cache[e_key]:
            n_missing_cache += 1
            continue
        e_df = cache[e_key][0].sort_values("date").reset_index(drop=True)
        dc = daily_by_code.get(code)
        i_e = cal_idx[e_date]
        lookback = cal[max(0, i_e - 250):i_e]
        d_sub = dc[dc["date"].isin(lookback)].sort_values("date").reset_index(drop=True)
        if len(d_sub) < 25:
            continue
        atr_ref = float(E.build_atr_ref(d_sub))
        zones = E.build_zones(d_sub, atr_ref)
        ref_price = float(d_sub["close"].iloc[-1])
        sig = detect_signal_v02(e_df, zones, atr_ref, ref_price)
        if sig is None:
            n_ts_mismatch += 1
            continue
        if int(sig["signal_ts"]) != int(r["signal_ts"]):
            n_ts_mismatch += 1
            continue
        zone = next(z for z in zones if z["zone_id"] == sig["zone_id"])
        # 主锚点: 用锚点实际价格对照 zone_mid
        c_arr = d_sub["close"].astype(float).to_numpy()
        ma20 = float(np.mean(c_arr[-20:]))
        boll_l = ma20 - 2.0 * float(np.std(c_arr[-20:], ddof=1))
        prev_low = float(d_sub["low"].astype(float).iloc[-1])
        anchor_prices = {"MA20": ma20, "BOLL_L": boll_l, "PREV_LOW": prev_low}
        present = {k: v for k, v in anchor_prices.items() if k in zone["zone_sources"]}
        anchor_lab = min(present, key=lambda k: abs(present[k] - zone["zone_mid"]))
        feats.append({
            "code": code, "E": e_date, "setup": r["setup"],
            "segment": segment_of(e_date),
            "y": int(r["reclaim_60m"]),
            "sig_type": sig["signal_type"],
            "anchor": anchor_lab,
            "width_atr": float(zone["span_atr"]),
            "pen_atr_bar": float(sig["pen_atr_bar"]),
            "session": sig["session"],
            "month": int(str(e_date)[:6]),
        })
    integrity = {"included_rows": len(elig), "feature_rows": len(feats),
                 "signal_ts_mismatch": n_ts_mismatch, "cache_missing": n_missing_cache}
    return feats, integrity


# ---------------- CAR-G: 分层 beta-binomial EB ----------------

def mom_beta(counts):
    """从 [(s,f)] 用方法矩估 Beta(alpha,beta); 失败返回 None."""
    ns = np.array([s + f for s, f in counts], dtype=float)
    ps = np.array([s / (s + f) for s, f in counts], dtype=float)
    n_total = ns.sum()
    if n_total <= 0:
        return None
    m = (ns * ps).sum() / n_total
    if not 0 < m < 1:
        return None
    # 组间方差 (样本量加权)
    v = (ns * (ps - m) ** 2).sum() / n_total
    denom = m * (1 - m) - v
    if denom <= 0:
        return None
    k = m * (1 - m) / v - 1 if v > 0 else None
    if k is None or k <= 0:
        return None
    return m * k, (1 - m) * k


class CAR_G:
    def __init__(self, train, width_edges):
        self.width_edges = width_edges
        for r in train:
            r["width_bin"] = self._wbin(r["width_atr"])
        # ---- level2 计数与全局先验 ----
        l2_counts: dict = {}
        for r in train:
            k = (r["anchor"], r["sig_type"])
            s, f = l2_counts.get(k, (0, 0))
            l2_counts[k] = (s + (r["y"] == 1), f + (r["y"] == 0))
        ab = mom_beta(list(l2_counts.values()))
        if ab is None:  # 退化: Jeffreys 先验
            ab = (0.5, 0.5)
        self.g_ab = ab
        self.l2_post = {k: (ab[0] + s, ab[1] + f) for k, (s, f) in l2_counts.items()}

        # ---- level3 计数; 每父下 MOM 先验, 失败则继承父后验 ----
        l3_counts: dict = {}
        for r in train:
            k = (r["anchor"], r["sig_type"], r["width_bin"])
            s, f = l3_counts.get(k, (0, 0))
            l3_counts[k] = (s + (r["y"] == 1), f + (r["y"] == 0))
        self.l3_prior: dict = {}
        for k, (s, f) in l3_counts.items():
            parent = (k[0], k[1])
            siblings = [(ss, ff) for kk, (ss, ff) in l3_counts.items()
                        if (kk[0], kk[1]) == parent]
            ab3 = mom_beta(siblings) if len(siblings) >= 2 else None
            if ab3 is None:
                ab3 = self.l2_post[parent]  # 共轭继承
            self.l3_prior[k] = ab3
        self.l3_counts = l3_counts

    def _wbin(self, w):
        edges = self.width_edges
        for i in range(len(edges) - 1):
            if edges[i] < w <= edges[i + 1] or (i == 0 and w <= edges[i + 1]):
                return i
        return len(edges) - 2

    def predict_one(self, anchor, sig_type, width):
        wb = self._wbin(width)
        k3 = (anchor, sig_type, wb)
        if k3 in self.l3_counts:
            a, b = self.l3_prior[k3]
            s, f = self.l3_counts[k3]
            return (a + s) / (a + b + s + f)
        # 空组回退 level2
        a2, b2 = self.l2_post.get((anchor, sig_type), self.g_ab)
        return a2 / (a2 + b2)

    def predict(self, rows):
        return np.array([self.predict_one(r["anchor"], r["sig_type"], r["width_atr"])
                         for r in rows])


# ---------------- 评价 ----------------

def brier(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    return float(np.mean((p - y) ** 2))


def logloss(y, p):
    p = np.clip(np.asarray(p, float), 1e-12, 1 - 1e-12)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def calibration_table(y, p, n_bins=N_CALIB_BINS):
    y, p = np.asarray(y, float), np.asarray(p, float)
    edges = np.linspace(0, 1, n_bins + 1)
    out, ece = [], 0.0
    for i in range(n_bins):
        msk = (p > edges[i]) & (p <= edges[i + 1]) if i > 0 else (p >= edges[i]) & (p <= edges[i + 1])
        n = int(msk.sum())
        if n > 0:
            obs, mean_p = float(y[msk].mean()), float(p[msk].mean())
            ece += n / len(y) * abs(obs - mean_p)
        else:
            obs = mean_p = None
        out.append({"bin": [round(edges[i], 2), round(edges[i + 1], 2)],
                    "n": n, "observed": obs, "mean_pred": mean_p})
    return out, float(ece)


def cluster_boot_deltas(rows, pred_by_model, n_boot=BOOT_N, seed=BOOT_SEED):
    """日期聚类配对 bootstrap: 返回各模型 Brier 分布与对 baseline 的 d 分布."""
    rng = np.random.RandomState(seed)
    dates = sorted({r["E"] for r in rows})
    by_date: dict = {d: [i for i, r in enumerate(rows) if r["E"] == d] for d in dates}
    y = np.array([r["y"] for r in rows], float)
    base = "constant"
    models = list(pred_by_model)
    out_brier = {m: [] for m in models}
    out_delta = {m: [] for m in models if m != base}
    date_idx = list(range(len(dates)))
    for _ in range(n_boot):
        pick = rng.choice(date_idx, size=len(date_idx), replace=True)
        idx = [i for di in pick for i in by_date[dates[di]]]
        yy = y[idx]
        bs = {}
        for m in models:
            pp = np.asarray(pred_by_model[m])[idx]
            bs[m] = float(np.mean((pp - yy) ** 2))
            out_brier[m].append(bs[m])
        for m in out_delta:
            out_delta[m].append(bs[base] - bs[m])

    def ci(a):
        q = np.percentile(a, [2.5, 50, 97.5])
        return {"lo": float(q[0]), "median": float(q[1]), "hi": float(q[2])}
    return {m: ci(out_brier[m]) for m in models}, {m: ci(out_delta[m]) for m in out_delta}


# ---------------- 主流程 ----------------

def main():
    t0 = time.time()
    print("=" * 60)
    print("CAR R1 受限研究: 常数基线 -> CAR-G -> CAR-LR")
    print("=" * 60)

    rows, gate = load_rows_and_gates()
    feats, integrity = build_features(rows, gate)
    print(f"[特征] 口径A纳入={integrity['included_rows']}, 特征行={integrity['feature_rows']}, "
          f"ts不一致={integrity['signal_ts_mismatch']}, 缓存缺={integrity['cache_missing']}")
    assert integrity["signal_ts_mismatch"] == 0, "重算信号与已存信号不一致"

    segs = {s: [r for r in feats if r["segment"] == s] for s in ("train", "val", "test")}
    for s in ("train", "val", "test"):
        rr = segs[s]
        print(f"  {s}: n={len(rr)} 日期={len({r['E'] for r in rr})} "
              f"正={sum(r['y'] for r in rr)} 负={sum(1-r['y'] for r in rr)}")

    train, val, test = segs["train"], segs["val"], segs["test"]
    ytr = np.array([r["y"] for r in train])

    # ---- 常数基线 (训练段 MLE) ----
    p_const = float(ytr.mean())
    print(f"\n[常数基线] p={p_const:.4f} (训练段 {len(ytr)} 行)")

    # ---- CAR-G: 宽度三分位边界仅由训练段拟合 ----
    width_edges = list(np.unique(np.quantile(
        [r["width_atr"] for r in train],
        np.linspace(0, 1, WIDTH_N_BINS + 1))))
    width_edges[0], width_edges[-1] = -np.inf, np.inf
    carg = CAR_G(train, width_edges)
    print(f"[CAR-G] level2 组数={len(carg.l2_post)}, level3 组数={len(carg.l3_counts)}, "
          f"全局先验=({carg.g_ab[0]:.2f},{carg.g_ab[1]:.2f})")

    pred = {
        "constant": {s: np.full(len(segs[s]), p_const) for s in segs},
        "CAR-G": {s: carg.predict(segs[s]) for s in segs},
    }

    # ---- 各段指标 ----
    metrics = {}
    for s in ("train", "val", "test"):
        yy = np.array([r["y"] for r in segs[s]])
        metrics[s] = {}
        for m in ("constant", "CAR-G"):
            pp = pred[m][s]
            calib, ece = calibration_table(yy, pp)
            metrics[s][m] = {"brier": brier(yy, pp), "logloss": logloss(yy, pp),
                             "ece": ece, "calibration": calib,
                             "pred_min": float(pp.min()), "pred_max": float(pp.max())}

    # ---- 验证段不确定性 + 增量门 ----
    val_brier_ci, val_delta_ci = cluster_boot_deltas(
        val, {"constant": pred["constant"]["val"], "CAR-G": pred["CAR-G"]["val"]})
    d_val = val_delta_ci["CAR-G"]
    d_point = metrics["val"]["constant"]["brier"] - metrics["val"]["CAR-G"]["brier"]
    increment = d_point > 0 and d_val["lo"] > 0
    gate_decision = {
        "segment": "val", "criterion": "paired date-cluster bootstrap, 95% CI of "
                    "(Brier_constant - Brier_CARG) lower bound > 0, point > 0",
        "delta_brier_point": round(d_point, 5),
        "delta_brier_boot_ci": {k: round(v, 5) for k, v in d_val.items()},
        "brier_boot_ci": val_brier_ci,
        "increment": increment,
    }
    print("\n[增量门 @验证段]")
    print(f"  dBrier 点估计={d_point:.5f}, 95% CI=[{d_val['lo']:.5f},{d_val['hi']:.5f}] "
          f"-> {'INCREMENT' if increment else 'NO INCREMENT'}")

    # 测试段不确定性 (仅披露)
    test_brier_ci, test_delta_ci = cluster_boot_deltas(
        test, {"constant": pred["constant"]["test"], "CAR-G": pred["CAR-G"]["test"]})

    # ---- A/B 池与月份分层 (披露) ----
    splits_report = {"by_setup": {}, "by_month": {}}
    for s in ("train", "val", "test"):
        for setup in ("A_SDP", "B_LVB"):
            sub = [r for r in segs[s] if r["setup"] == setup]
            if not sub:
                continue
            yy = np.array([r["y"] for r in sub])
            entry = {"segment": s, "n": len(sub), "positive": int(yy.sum()),
                     "observed_rate": round(float(yy.mean()), 4),
                     "brier_constant": round(brier(yy, pred["constant"][s][
                         [i for i, r in enumerate(segs[s]) if r["setup"] == setup]]), 4),
                     "brier_CAR-G": round(brier(yy, pred["CAR-G"][s][
                         [i for i, r in enumerate(segs[s]) if r["setup"] == setup]]), 4)}
            splits_report["by_setup"].setdefault(setup, []).append(entry)
        for mth in sorted({r["month"] for r in segs[s]}):
            idxs = [i for i, r in enumerate(segs[s]) if r["month"] == mth]
            sub = [segs[s][i] for i in idxs]
            yy = np.array([r["y"] for r in sub])
            splits_report["by_month"].setdefault(str(mth), []).append({
                "segment": s, "n": len(sub), "positive": int(yy.sum()),
                "brier_constant": round(brier(yy, pred["constant"][s][idxs]), 4),
                "brier_CAR-G": round(brier(yy, pred["CAR-G"][s][idxs]), 4)})

    # ---- LR: 仅增量门通过才拟合 ----
    lr_block = None
    if increment:
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        anchors = ["MA20", "BOLL_L", "PREV_LOW"]
        sigs = ["ALREADY_BELOW", "BREAK"]

        def design(rs):
            X = []
            for r in rs:
                row = [float(r["anchor"] == a) for a in anchors]
                row += [float(r["sig_type"] == s) for s in sigs]
                row += [float(r["width_atr"]), float(r["pen_atr_bar"])]
                X.append(row)
            return np.array(X)

        Xtr_raw = design(train)
        scaler = StandardScaler()
        Xtr = Xtr_raw.copy()
        Xtr[:, -2:] = scaler.fit_transform(Xtr_raw[:, -2:])
        lr = LogisticRegression(penalty="l2", C=1.0, max_iter=2000)
        lr.fit(Xtr, ytr)

        def lr_predict(rs):
            X = design(rs)
            X[:, -2:] = scaler.transform(X[:, -2:])
            return lr.predict_proba(X)[:, 1]

        for s in segs:
            pred.setdefault("CAR-LR", {})[s] = lr_predict(segs[s])
        lr_metrics = {}
        for s in ("train", "val", "test"):
            yy = np.array([r["y"] for r in segs[s]])
            pp = pred["CAR-LR"][s]
            calib, ece = calibration_table(yy, pp)
            lr_metrics[s] = {"brier": brier(yy, pp), "logloss": logloss(yy, pp),
                             "ece": ece, "calibration": calib}
        val_lr_ci, val_lr_delta = cluster_boot_deltas(
            val, {"constant": pred["constant"]["val"], "CAR-G": pred["CAR-G"]["val"],
                  "CAR-LR": pred["CAR-LR"]["val"]})
        lr_block = {
            "config": "L2, C=1, solver lbfgs; one-hot anchor/sigtype + width/pen standardized",
            "cv_note": "C frozen per plan; 3-fold CV not applicable (no tuning)",
            "coef_order": anchors + sigs + ["width_atr_z", "pen_atr_z"],
            "coef": [float(c) for c in lr.coef_[0]], "intercept": float(lr.intercept_[0]),
            "metrics": lr_metrics,
            "val_boot_ci": val_lr_ci,
            "val_delta_vs_CARG_ci": {k: round(v, 5) for k, v in val_lr_delta["CAR-LR"].items()},
        }
        print("[CAR-LR] 已拟合; 验证段 dBrier(LR-G) CI=",
              lr_block["val_delta_vs_CARG_ci"])
    else:
        print("[CAR-LR] 增量门未过, 按合同不拟合 LR.")

    # ---- 汇总报告 ----
    report = {
        "version": "car_r1_results",
        "date": "2026-10-01",
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "authorization": "R1 restricted research: can 60m reclaim probability be predicted "
                         "better; NOT buy-model promotion",
        "scope": {"caliber": "A_strict_all_gates", "b_caliber": "disclosed separately, not used",
                  "fit_segment": "train only", "network_requests": 0,
                  "features_at_S": ["sig_type", "anchor", "width_atr", "pen_atr_bar"],
                  "no_new_indicators": True},
        "data_integrity": integrity,
        "segments": {s: {"n": len(segs[s]), "dates": len({r["E"] for r in segs[s]}),
                        "positive": sum(r["y"] for r in segs[s]),
                        "negative": sum(1 - r["y"] for r in segs[s])}
                     for s in segs},
        "constant_p": p_const,
        "width_edges_fit_on_train": [None if v == -np.inf else (None if v == np.inf else v)
                                     for v in width_edges],
        "metrics": metrics,
        "increment_gate": gate_decision,
        "test_disclosure": {"brier_boot_ci": test_brier_ci,
                            "delta_ci": test_delta_ci,
                            "status": "existing holdout segment; aggregate results viewed "
                                      "before R1; not a fresh blind test"},
        "subgroups": splits_report,
        "lr": lr_block,
        "locks": {"he": True, "h3_executable_return": True, "buy_advice": True,
                  "online_monitor": False},
        "incident_log_ref": "first invalid run (stockdb down) and merge-date-filter error "
                            "remain in car_v0_2_train_top_up_report.json history; not mixed "
                            "with corrected numbers",
    }

    R1_REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str),
                         encoding="utf-8")
    print(f"\n总耗时 {time.time() - t0:.1f}s; 报告: {R1_REPORT.name}")
    if increment and lr_block is not None:
        print("结论: CAR-G 对常数基线有预注册增量; LR 对照已输出 (LR 未优于 G 仅否决 LR).")
    elif increment:
        print("结论: CAR-G 有增量; LR 未拟合 (异常, 检查).")
    else:
        print("结论: CAR-G 无预注册增量, R1 按合同停止; 不调参/不跑 LR.")


if __name__ == "__main__":
    main()
