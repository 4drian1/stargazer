# -*- coding: utf-8 -*-
"""影子模式打分脚本（stargazer_desktop 子进程调用，§18.33）。

读取冻结四头（shadow_models/shadow_t5_frozen.pkl），对给定信号日 T 的
G-SHOCK 候选（来自离线特征表，entry_type=vwap、除权有效）打分：
  - P_risk：AM-R1（dp 1000/1030）或 PM-R2（dp 1330/1400/1430/1450）
  - P_opp ：AM-O4 / PM-O6（Isotonic 校准后概率）
  - 四象限（展示层，相对训练基准率启发式，非买入指令）：
      低风险(P_risk<=base)  & 高机会(P_opp>=0.5) -> FOCUS（仅人工复核）
      高风险(P_risk>=1.5base)& 高机会             -> WAIT
      低风险                & 低机会             -> WATCH
      其他                                    -> REJECT
    并附参考EV（训练池同象限历史均值 ev3_ref/ev5_ref，非模型预测）。
  - OOD 警告（ood_thresholds.json）：runup_10d>P99.5 或 ≥3 个高热指标>P99
    -> OUT_OF_DISTRIBUTION，仅观察，覆盖四象限显示。

模式：
  --as-of YYYYMMDD  打分并写 <output-dir>/t5_shadow_advice.json（默认最新信号日）
  --validate2026    离线回放验证：冻结头打 2026 池，对照 ablation_t7_2026.csv
                    的 R1/R2 hi30%STOP（容差 ±0.5pp），确认口径无漂移后才允许上线
用法：
  python shadow_batch.py --as-of 20260831 --output-dir <cache>
  python shadow_batch.py --validate2026
"""
import argparse
import json
import os
import pickle
import time

import numpy as np
import pandas as pd

import ablation_opp as AO
import ablation_risk as AR
import data_io

OUT_DIR = "D:/stargazer/stargazer_ml/pullback_train"
PKL = os.path.join(OUT_DIR, "shadow_models", "shadow_t5_frozen.pkl")
OOD = os.path.join(OUT_DIR, "ood_thresholds.json")
KT = "2.0"
HEADS = AR.HEADS  # AM (1000,1030) / PM (1330,1400,1430,1450)
# 2026 特征表统一在根目录（与 ablation_t7 一致）；样本表走 data_io 治理入口
FEAT = {
    2025: {"PF": AR.PF, "ST": AR.ST, "TPM": AR.TPM, "T6": AR.T6,
           "T3": AO.T3, "T4": AO.T4, "LBL": AR.LBL},
    2026: {"PF": "minute_path_features_2026.parquet",
           "ST": "minute_second_test_2026.parquet",
           "TPM": "t2pm_features_2026.parquet",
           "T6": "t6_features_2026.parquet",
           "T3": "minute_t3_vwap_2026.parquet",
           "T4": "minute_t4_proxy_2026.parquet",
           "LBL": "dual_head_labels_2026.parquet"},
}
T7 = {y: os.path.join(OUT_DIR, f"t7_features_{y}.parquet") for y in (2025, 2026)}
HEAT3 = ("runup_10d", "turnover_sum_5d", "rsi14", "pos20", "limit_up_10d",
         "turnover_max_5d", "runup_20d", "rsi_over75_10d")


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_bundle():
    with open(PKL, "rb") as f:
        return pickle.load(f)


def build_pool(year, dps, as_of=None):
    """与 export/build_pool 同口径；as_of 限定信号日（默认全年）。"""
    df = data_io.load_samples(year)
    df = df[df["dp"].isin(dps)].copy()
    if as_of is not None:
        df = df[df["date"] == as_of].copy()
    for k in ("PF", "ST", "TPM", "T6", "T3", "T4"):
        m = pd.read_parquet(os.path.join(OUT_DIR, FEAT[year][k]))
        m["code"] = m["code"].astype(str)
        df["code"] = df["code"].astype(str)
        df = df.merge(m, on=["code", "dp_date", "dp"], how="left")
    df = df[(df["entry_type"] == "vwap") & (df["censored"] == 0)].copy()
    if "price_scale_valid" not in df.columns:
        df["price_scale_valid"] = True
    df = df[df["price_scale_valid"].astype(bool)].copy()
    return df


def attach_t7(year, df):
    m = pd.read_parquet(T7[year])
    m["code"] = m["code"].astype(str)
    df["code"] = df["code"].astype(str)
    m = m.drop(columns=["date"])
    return df.merge(m, on=["code", "dp_date", "dp"], how="left")


def score(bundle, df):
    """对池打分，返回带 P_risk/P_opp/quadrant 的副本。"""
    df = df.copy()
    df["P_risk"], df["P_opp"] = np.nan, np.nan
    for head, dps in HEADS.items():
        m = df["dp"].isin(dps)
        if not m.any():
            continue
        rk, ok = f"{head}_R1" if head == "AM" else f"{head}_R2", \
                 f"{head}_O4" if head == "AM" else f"{head}_O6"
        fr = bundle["meta"][rk]["features"]
        fo = bundle["meta"][ok]["features"]
        df.loc[m, "P_risk"] = bundle["calibrators"][rk].predict(
            bundle["models"][rk].predict_proba(df.loc[m, fr])[:, 1])
        df.loc[m, "P_opp"] = bundle["calibrators"][ok].predict(
            bundle["models"][ok].predict_proba(df.loc[m, fo])[:, 1])
    base_r = {h: float(np.mean([bundle["meta"][f"{h}_R1" if h == "AM" else f"{h}_R2"]["base_rate"]]))
              for h in HEADS}
    quad = np.where(df["P_opp"].isna(), "UNSCORED", "REJECT")
    for h in HEADS:
        m = df["dp"].isin(HEADS[h]) & df["P_opp"].notna()
        lo_r, hi_r = df.loc[m, "P_risk"] <= base_r[h], \
                     df.loc[m, "P_risk"] >= 1.5 * base_r[h]
        hi_o = df.loc[m, "P_opp"] >= 0.5
        q = pd.Series("REJECT", index=df.index[m])
        q[lo_r & hi_o] = "FOCUS"   # 低风险高机会：仅人工复核，不自动买入
        q[hi_r & hi_o] = "WAIT"
        q[lo_r & ~hi_o] = "WATCH"
        quad[m] = q
    df["quadrant"] = quad
    return df


def attach_ood(df):
    with open(OOD, encoding="utf-8") as f:
        th = json.load(f)["features"]
    df["ood_hits"] = 0
    for c in HEAT3:
        if c in df.columns and c in th:
            df["ood_hits"] += (df[c] > th[c]["p99_5"]).fillna(False).astype(int)
    primary = df["runup_10d"] > th["runup_10d"]["p99_5"] if "runup_10d" in df.columns \
        else pd.Series(False, index=df.index)
    df["ood"] = (primary.fillna(False) | (df["ood_hits"] >= 3))
    return df


def advice_rows(df):
    keep = ["code", "name", "date", "dp_date", "dp", "src", "subtype",
            "entry", "atr_ref", "P_risk", "P_opp", "quadrant", "ood",
            "ood_hits", "ev3_ref", "ev5_ref", "model_version",
            "runup_5d", "runup_10d", "runup_20d", "pos20",
            "rsi14", "turnover_sum_5d", "turnover_max_5d", "limit_up_10d",
            "shock_atr", "downside_atr", "recovery", "kill_speed_atr",
            "vwap_reclaim_atr", "new_low_after_945"]
    have = [c for c in keep if c in df.columns]
    rows = df[have].copy()
    rows["strategy"] = "t5_shadow"
    for c in ("P_risk", "P_opp"):
        rows[c] = rows[c].round(4)
    for c in rows.columns:
        if rows[c].dtype == float:
            rows[c] = rows[c].round(4)
    return json.loads(rows.to_json(orient="records", force_ascii=False))


def cmd_as_of(a):
    bundle = load_bundle()
    year = a.as_of // 10000
    pool = build_pool(year, tuple(data_io.EXPECTED_DPS), as_of=a.as_of)
    if not len(pool):
        raise SystemExit(f"信号日 {a.as_of} 无候选（检查该日是否已构建样本/特征表）")
    pool = attach_t7(year, pool)
    pool = score(bundle, pool)
    pool = attach_ood(pool)
    pool = pool[pool["P_opp"].notna()].copy()  # 945 无头覆盖，不进影子建议
    # 参考EV（训练池同象限历史均值，非预测）+ 字段对齐决策表
    for h in HEADS:
        m = pool["dp"].isin(HEADS[h])
        ref = bundle.get("ev_ref", {}).get(h, {})
        pool.loc[m, "ev3_ref"] = pool.loc[m, "quadrant"].map(
            {q: v["ev3_ref"] for q, v in ref.items()})
        pool.loc[m, "ev5_ref"] = pool.loc[m, "quadrant"].map(
            {q: v["ev5_ref"] for q, v in ref.items()})
    model_version = str(bundle["meta"].get("_frozen_at") or "")
    pool["model_version"] = model_version
    if a.decisions_db:
        import shadow_log
        n = shadow_log.upsert_decisions(a.decisions_db, pool.to_dict("records"),
                                        model_version=model_version)
        log(f"决策留痕 {n} 行 -> {a.decisions_db}（model_version={model_version}）")
        nf = shadow_log.backfill_outcomes(a.decisions_db)
        if nf:
            log(f"到期回填 {nf} 行（最终3/5日收益已自动填入）")
    rows = advice_rows(pool)
    os.makedirs(a.output_dir, exist_ok=True)
    out = os.path.join(a.output_dir, "t5_shadow_advice.json")
    payload = {"as_of": a.as_of, "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
               "model_version": model_version,
               "frozen": bundle["meta"]["_frozen_list"],
               "n": len(rows), "rows": rows}
    with open(out, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=1)
    log(f"信号日 {a.as_of}: {len(rows)} 行 -> {out}")
    log("象限分布 " + str(pool["quadrant"].value_counts().to_dict())
        + "；OOD " + str(int(pool["ood"].sum())) + " 只")


def cmd_validate(a):
    """回放验证：冻结头打 2026 池 hi30%，对照 ablation_t7_2026.csv。"""
    from ablation_t7 import load_pool as t7_pool
    bundle = load_bundle()
    ref = pd.read_csv(os.path.join(OUT_DIR, "ablation_t7_2026.csv"))
    ok_all = True
    for head in HEADS:
        df = t7_pool(2026, HEADS[head])
        df = df[df["date"] <= 20260831]
        for k in ("T3", "T4", "T6"):  # t7_pool 不含机会头特征，补合并
            m = pd.read_parquet(os.path.join(OUT_DIR, FEAT[2026][k]))
            m["code"] = m["code"].astype(str)
            df["code"] = df["code"].astype(str)
            df = df.merge(m, on=["code", "dp_date", "dp"], how="left")
        df = score(bundle, df)
        q70 = df["P_risk"].quantile(0.7)
        hi = df[df["P_risk"] >= q70]
        got = float(hi["r_sbr_t2"].mean())
        r = ref[(ref["head"] == head) & (ref["label"] == "y0")
                & (ref["model"] == ("R1" if head == "AM" else "R2"))].iloc[0]
        want = float(r["hi_rate"])
        ok = abs(got - want) <= 0.005
        ok_all &= ok
        log(f"[{head}-R] 回放 hi30%STOP={got:.2%} vs 消融 {want:.2%} -> "
            f"{'OK' if ok else 'MISMATCH'}")
        okey = f"{head}_O4" if head == "AM" else f"{head}_O6"
        q70o = df["P_opp"].quantile(0.7)
        hio = df[df["P_opp"] >= q70o]
        log(f"[{head}-O] 回放 hi30% t1_bt_5d={hio['o_t1_bt_5d'].mean():.1%} "
            f"净收益5d={hio['o_net_ret_5d'].mean():+.3%}（对照 §18.29 方向）")
    print("VALIDATE2026:", "PASS" if ok_all else "FAIL")


def cmd_replay(a):
    """历史重放回测：对整年每个信号日重放影子留痕（与前瞻同口径）。

    象限切训练基准率（绝对阈值），故整池一次打分与逐日打分等价。
    注意：2025 为训练期内（in-sample，偏乐观）；2026 已观察（可否定）。
    真正的样本外只有前瞻影子。
    """
    import shadow_log
    bundle = load_bundle()
    model_version = str(bundle["meta"].get("_frozen_at") or "")
    for year in (int(y) for y in str(a.years).split(",")):
        pool = build_pool(year, tuple(data_io.EXPECTED_DPS))
        if a.since:
            pool = pool[pool["date"] >= a.since].copy()
        pool = score(bundle, pool)
        pool = attach_ood(pool)
        pool = pool[pool["P_opp"].notna()].copy()
        pool["model_version"] = model_version
        for h in HEADS:
            m = pool["dp"].isin(HEADS[h])
            ref = bundle.get("ev_ref", {}).get(h, {})
            pool.loc[m, "ev3_ref"] = pool.loc[m, "quadrant"].map(
                {q: v["ev3_ref"] for q, v in ref.items()})
            pool.loc[m, "ev5_ref"] = pool.loc[m, "quadrant"].map(
                {q: v["ev5_ref"] for q, v in ref.items()})
        n = shadow_log.upsert_decisions(a.decisions_db, pool.to_dict("records"),
                                        model_version=model_version)
        log(f"[replay {year}] 留痕 {n} 行（{pool['date'].nunique()} 个信号日，"
            f"model_version={model_version}）")
    nf = shadow_log.backfill_outcomes(a.decisions_db)
    log(f"回填 {nf} 行")
    shadow_log.analyze(a.decisions_db)


def cmd_backfill(a):
    import shadow_log
    n = shadow_log.backfill_outcomes(a.decisions_db)
    print(f"backfilled {n} rows -> {a.decisions_db}")
    shadow_log.analyze(a.decisions_db)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--as-of", type=int, default=None, help="信号日 YYYYMMDD")
    ap.add_argument("--output-dir", default=OUT_DIR)
    ap.add_argument("--validate2026", action="store_true")
    ap.add_argument("--decisions-db", default=None,
                    help="影子决策留痕 sqlite（默认不写）")
    ap.add_argument("--backfill-decisions", action="store_true",
                    help="回填最终3/5日收益并出分析")
    ap.add_argument("--replay", action="store_true",
                    help="历史重放回测：整年逐信号日留痕+回填+分析")
    ap.add_argument("--years", default="2025,2026", help="回放年份，逗号分隔")
    ap.add_argument("--since", type=int, default=None, help="只回放 >=该信号日")
    a = ap.parse_args()
    t0 = time.time()
    if a.replay:
        if not a.decisions_db:
            raise SystemExit("--replay 需要 --decisions-db")
        cmd_replay(a)
    elif a.backfill_decisions:
        if not a.decisions_db:
            raise SystemExit("--backfill-decisions 需要 --decisions-db")
        cmd_backfill(a)
    elif a.validate2026:
        cmd_validate(a)
    else:
        as_of = a.as_of
        if as_of is None:
            s = data_io.load_samples(2026)
            as_of = int(s["date"].max())
            log(f"未指定 --as-of，取 2026 表最新信号日 {as_of}")
        cmd_as_of(argparse.Namespace(**{**vars(a), "as_of": as_of}))
    log(f"done {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
