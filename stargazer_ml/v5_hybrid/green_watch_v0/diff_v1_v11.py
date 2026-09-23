"""v1 -> v1.1 Candidate Diff (Step 2, 终版).

修正: 全程只用 DIP-gate 定义 (v1_dip = dip_gate|DIP_SCORE>=40; v11_dip = dip_eligible_v11),
避免把"全集候选 4 格"的 v11_only 与"DIP-gate 2x2"的 v11_only 两个不同分母混用。
限制共同交易日窗口, 消除 v1 快照日期较短的干扰。

输出 (对应审计要求):
  1. v1/v11 日期窗口 + 共同窗口
  2. (date,code) / (plan_date,code) 唯一键重复校验
  3. DIP-gate 2x2: both / v1_only / v11_only (同一分母)
  4. v11_only 的 dip_stage 覆盖率 (PRE_PULLBACK/PULLBACK_ACTIVE/NEAR_ZONE/NULL/UNKNOWN/OTHER)
     + stage_coverage_ratio
  5. v1_only 被 v11 删除的原因诊断 (prior_strength/structure/clear_dist/breakdown flag)
  6. V1_ONLY / V11_ONLY / BOTH 三组在统一特征上的分布
不挑参数、不判收益; 仅审计 v1.1 是否按设计改变了 Candidate Population。
"""
import pandas as pd
import numpy as np

V1 = "green_watch_v0/pool_daily.parquet"
V11 = "green_watch_v0/pool_daily_v11.parquet"

v1_cols = ["date", "code", "primary_setup", "dip_gate", "intraday_dip_score"]
v11_cols = ["date", "code", "primary_setup", "dip_eligible_v11", "dip_stage",
            "intraday_dip_score", "primary_tier", "dip_tier",
            "prior_strength_pass", "structure_intact_flag", "clear_distribution_flag",
            "structural_breakdown_flag", "liquidity_pass",
            "ma20_slope5", "prior_impulse_return", "ret_20d", "ret_60d",
            "dist_from_20d_high", "dist_from_60d_high", "zone_distance_atr",
            "support_break_depth_atr", "distribution_score_raw"]

out = []
def log(*a):
    s = " ".join(str(x) for x in a)
    out.append(s); print(s)

log("load v1 ...")
v1 = pd.read_parquet(V1, columns=v1_cols)
log("load v11 ...")
v11 = pd.read_parquet(V11, columns=v11_cols)

# ---------- 1. 日期窗口 ----------
d1 = (int(v1.date.min()), int(v1.date.max()))
d11 = (int(v11.date.min()), int(v11.date.max()))
common = sorted(set(v1.date.unique()) & set(v11.date.unique()))
log(f"\n[1] v1 日期窗口 {d1[0]}~{d1[1]} ({v1.date.nunique()} 交易日)")
log(f"    v11 日期窗口 {d11[0]}~{d11[1]} ({v11.date.nunique()} 交易日)")
log(f"    共同窗口     {min(common)}~{max(common)} ({len(common)} 交易日)")
v1 = v1[v1.date.isin(common)].copy()
v11 = v11[v11.date.isin(common)].copy()
log(f"    限制共同窗口后: v1={len(v1)} 行, v11={len(v11)} 行")

# ---------- 2. 唯一键重复校验 ----------
for f, df in [("v1", v1), ("v11", v11)]:
    dup_dc = df.duplicated(subset=["date", "code"]).sum()
    log(f"[{f}] (date,code) 重复行 = {dup_dc}")
    if "plan_date" in df.columns:
        log(f"[{f}] (plan_date,code) 重复行 = {df.duplicated(subset=['plan_date','code']).sum()}")
    else:
        log(f"[{f}] 无 plan_date 列 (唯一键用 date)")

# ---------- 3. DIP-gate 2x2 (共同窗口, 单一分母) ----------
v1_dip = set(map(tuple, v1[(v1.dip_gate == True) | (v1.intraday_dip_score >= 40)][["date", "code"]].values))
v11_dip = set(map(tuple, v11[v11.dip_eligible_v11 == True][["date", "code"]].values))
both = v1_dip & v11_dip
v1_only_set = v1_dip - v11_dip
v11_only_set = v11_dip - v1_dip
log(f"\n[3] DIP-gate 2x2 (共同窗口)")
log(f"    v1 标 DIP   = {len(v1_dip)}")
log(f"    v11 标 DIP  = {len(v11_dip)}")
log(f"    both_accepted = {len(both)}")
log(f"    v1_only (旧收新弃)  = {len(v1_only_set)}  ({len(v1_only_set)/len(v1_dip):.1%} of v1 DIP)")
log(f"    v11_only (旧弃新收) = {len(v11_only_set)}  <- 新增'强势回调'类")

# ---------- 4. v11_only dip_stage 覆盖率 ----------
m11 = v11.set_index(["date", "code"])
v11_only_rows = m11.loc[[k for k in v11_only_set if k in m11.index]]
stage = v11_only_rows["dip_stage"]
vc = stage.value_counts(dropna=False)
pre = int((stage == "PRE_PULLBACK").sum())
pul = int((stage == "PULLBACK_ACTIVE").sum())
nez = int((stage == "NEAR_ZONE").sum())
nul = int(stage.isna().sum())
unk = int(((~stage.isna()) & (~stage.isin(["PRE_PULLBACK", "PULLBACK_ACTIVE", "NEAR_ZONE"]))).sum())
oth = unk
classified = pre + pul + nez
cov = classified / len(v11_only_rows) if len(v11_only_rows) else float("nan")
log(f"\n[4] v11_only stage coverage (n={len(v11_only_rows)})")
log(f"    PRE_PULLBACK = {pre}")
log(f"    PULLBACK_ACTIVE = {pul}")
log(f"    NEAR_ZONE = {nez}")
log(f"    NULL = {nul}")
log(f"    UNKNOWN = {unk}")
log(f"    OTHER = {oth}")
log(f"    stage_coverage_ratio = {cov:.4f}  (目标≈1.0)")

# ---------- 5. v1_only 被删原因诊断 (用 v11 的 flag 字段) ----------
v1_only_feat = m11.loc[[k for k in v1_only_set if k in m11.index]]
n_missing = len(v1_only_set) - len(v1_only_feat)
log(f"\n[5] v1_only 删除原因诊断 (在 v11 取到特征 n={len(v1_only_feat)}, 缺失={n_missing})")
ps = v1_only_feat["prior_strength_pass"].astype(float)
si = v1_only_feat["structure_intact_flag"].astype(float)
cd = v1_only_feat["clear_distribution_flag"].astype(float)
sb = v1_only_feat["structural_breakdown_flag"].astype(float)
lq = v1_only_feat["liquidity_pass"].astype(float)
log(f"    prior_strength_fail (PriorStrength 弱)      = {int((ps==0).sum())} ({ (ps==0).mean():.1%})")
log(f"    structure_broken     (StructureIntact 失败)= {int((si==0).sum())} ({ (si==0).mean():.1%})")
log(f"    clear_distribution   (明显派发)            = {int((cd==1).sum())} ({ (cd==1).mean():.1%})")
log(f"    structural_breakdown(结构破坏)            = {int((sb==1).sum())} ({ (sb==1).mean():.1%})")
log(f"    liquidity_fail      (LiquidityPass 失败)  = {int((lq==0).sum())} ({ (lq==0).mean():.1%})")
# 通过全部 5 个闸门仍被删 = 真正的异常 (应为 ~0)
clean_but_dropped = ((ps == 1) & (si == 1) & (cd == 0) & (sb == 0) & (lq == 1))
any_fail = ((ps == 0) | (si == 0) | (cd == 1) | (sb == 1) | (lq == 0))
log(f"    >>> 任一闸门失败 (gate 语义内删除)        = {int(any_fail.sum())} ({any_fail.mean():.1%})")
log(f"    >>> 看似'好股却被判删' (通过全部5闸门仍删) = {int(clean_but_dropped.sum())} ({clean_but_dropped.mean():.1%})")

# ---------- 6. V1_ONLY / V11_ONLY / BOTH 特征分布 ----------
def block(df, label):
    log(f"\n  ===== {label} (n={len(df)}) =====")
    for col in ["prior_impulse_return", "ret_20d", "ret_60d", "ma20_slope5",
                "dist_from_20d_high", "dist_from_60d_high", "zone_distance_atr",
                "support_break_depth_atr", "distribution_score_raw"]:
        s = df[col].astype(float)
        log(f"    {col:24s} median={s.median():+.4f}  p10={s.quantile(.1):+.4f}  p90={s.quantile(.9):+.4f}")
    for col in ["prior_strength_pass", "structure_intact_flag"]:
        log(f"    {col:24s} mean={df[col].astype(float).mean():.3f}")
    log(f"    primary_tier 分布: " + df["primary_tier"].value_counts().to_dict().__str__())

both_rows = m11.loc[[k for k in both if k in m11.index]]
block(v1_only_feat, "V1_ONLY (v1收v11弃, 在v11中特征)")
block(v11_only_rows, "V11_ONLY (v11收v1弃)")
block(both_rows, "BOTH (v1&v11均收)")

log("\n[DONE]")
with open("green_watch_v0/diff_v1_v11_out.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(out))
log("[written] green_watch_v0/diff_v1_v11_out.txt")
