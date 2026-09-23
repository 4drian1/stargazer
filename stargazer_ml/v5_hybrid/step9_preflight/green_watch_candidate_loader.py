"""GreenWatchCandidateLoader — pool_daily.parquet 与 LATR 之间的薄契约层.

职责:
  - 从 pool_daily.parquet 读取候选 (单一事实源, 不复制第二份候选生成逻辑)
  - 严格校验契约字段 (symbol / plan_date / effective_trade_date /
    primary_setup / bottom_score / intraday_dip_score / atr_ref / tiers)
  - 标准化为 CandidateRow, 交给 latr_builder 做 D 日重放
  - LATR 不得重新扫描/重算; atr_ref / setup / scores 一律继承 CandidateRow

可选物化 green_watch_candidates.parquet 仅作为确定性投影/缓存
(相同 candidate 指纹 + source_data_snapshot_id, 不含任何新生成逻辑)。

注: build_pool 不产出 bottom_subtype, 该字段在契约中为 OPTIONAL, 缺失时为 ""。
"""
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
import pandas as pd

_HERE = Path(__file__).resolve().parent
POOL_PARQUET = _HERE.parent / "green_watch_v0" / "pool_daily.parquet"
CANDIDATE_PARQUET = _HERE / "green_watch_candidates.parquet"  # 可选缓存(确定性投影)

REQUIRED_FIELDS = [
    "code", "plan_date", "effective_trade_date",
    "primary_setup", "bottom_score", "intraday_dip_score", "atr_ref",
    "bottom_tier", "dip_tier", "primary_tier",
]
OPTIONAL_FIELDS = ["bottom_subtype", "bottom_gate", "candidate_contract_version",
                   "source_data_snapshot_id",
                   # v1.1 DIP 分解字段 (供 step2 v1→v1.1 Diff 与后续分析)
                   "dip_eligible_v11", "dip_stage", "dip_quality_score", "dip_readiness_score",
                   "prior_strength_pass", "structure_intact_flag", "structural_breakdown_flag",
                   "clear_distribution_flag", "liquidity_pass", "zone_distance_atr",
                   "prior_impulse_return", "ma20_slope_5", "ret_60d", "dist_from_60d_high",
                   "support_break_depth_atr", "distribution_score_raw"]


@dataclass
class CandidateRow:
    symbol: str
    plan_date: int
    effective_trade_date: int
    primary_setup: str
    bottom_score: float
    intraday_dip_score: float
    atr_ref: float
    bottom_tier: str = ""
    dip_tier: str = ""
    primary_tier: str = ""
    bottom_subtype: str = ""
    raw: dict = field(default_factory=dict)

    def validate(self):
        # 用户 ③.4 断言
        assert self.plan_date < self.effective_trade_date, \
            f"plan_date({self.plan_date}) >= effective_trade_date({self.effective_trade_date})"
        assert self.atr_ref and self.atr_ref > 0, "atr_ref missing/non-positive"
        return self


def load_pool_df() -> pd.DataFrame:
    if not POOL_PARQUET.exists():
        raise FileNotFoundError(
            f"pool_daily.parquet 不存在: {POOL_PARQUET} (先运行 green_watch_v0/build_pool.py)")
    return pd.read_parquet(POOL_PARQUET)


def _norm_plan_date(plan_date):
    if plan_date is None:
        return None
    if isinstance(plan_date, (tuple, list)) and len(plan_date) == 2:
        return ("range", int(plan_date[0]), int(plan_date[1]))
    if isinstance(plan_date, (list, tuple)):
        return ("in", [int(x) for x in plan_date])
    return ("eq", int(plan_date))


def load_green_watch_candidates(plan_date=None, effective_trade_date=None,
                                setup_type=None):
    """从 pool_daily.parquet 读取并严格校验候选。

    plan_date: None(全部) | int | (lo,hi) 区间 | list[int]
    effective_trade_date: None(不过滤) | int (精确匹配)
    setup_type: None(全部) | "BOTTOM_SETUP" | "INTRADAY_DIP_SETUP"

    返回 list[CandidateRow] (已过滤掉 effective_trade_date 缺失 / 不晚于 plan_date 的行)。
    """
    df = load_pool_df()
    missing = [c for c in REQUIRED_FIELDS if c not in df.columns]
    if missing:
        raise AssertionError(f"pool_daily.parquet 缺契约字段: {missing}")

    if plan_date is not None:
        mode = _norm_plan_date(plan_date)
        if mode[0] == "eq":
            df = df[df["plan_date"] == mode[1]]
        elif mode[0] == "range":
            df = df[(df["plan_date"] >= mode[1]) & (df["plan_date"] <= mode[2])]
        elif mode[0] == "in":
            df = df[df["plan_date"].isin(mode[1])]
    if effective_trade_date is not None:
        df = df[df["effective_trade_date"] == int(effective_trade_date)]
    if setup_type is not None:
        df = df[df["primary_setup"] == setup_type]

    # 过滤: effective_trade_date 缺失 (数据末日无下一交易日) 或不晚于 plan_date
    df = df[df["effective_trade_date"].notna()]
    df["_plan"] = df["plan_date"].astype(int)
    df["_eff"] = df["effective_trade_date"].astype(int)
    df = df[df["_eff"] > df["_plan"]]
    df = df.drop(columns=["_plan", "_eff"])

    out = []
    for _, r in df.iterrows():
        row = CandidateRow(
            symbol=str(r["code"]).zfill(6),
            plan_date=int(r["plan_date"]),
            effective_trade_date=int(r["effective_trade_date"]),
            primary_setup=str(r["primary_setup"]),
            bottom_score=float(r["bottom_score"]),
            intraday_dip_score=float(r["intraday_dip_score"]),
            atr_ref=float(r["atr_ref"]),
            bottom_tier=str(r.get("bottom_tier", "")),
            dip_tier=str(r.get("dip_tier", "")),
            primary_tier=str(r.get("primary_tier", "")),
            bottom_subtype=str(r.get("bottom_subtype", "")) if "bottom_subtype" in df else "",
            raw=r.to_dict(),
        )
        try:
            row.validate()
            out.append(row)
        except AssertionError:
            continue
    return out


def materialize_cache(plan_date=None, setup_type=None, force=False):
    """可选: 物化 green_watch_candidates.parquet 作为确定性投影/缓存。

    必须是 pool_daily 的纯投影, 含相同 source_data_snapshot_id 指纹;
    不允许包含任何新的候选生成逻辑。
    """
    if CANDIDATE_PARQUET.exists() and not force:
        return CANDIDATE_PARQUET
    df = load_pool_df()
    cols = [c for c in (REQUIRED_FIELDS + OPTIONAL_FIELDS) if c in df.columns]
    sub = df[cols]
    mode = _norm_plan_date(plan_date)
    if mode is not None:
        if mode[0] == "eq":
            sub = sub[sub["plan_date"] == mode[1]]
        elif mode[0] == "range":
            sub = sub[(sub["plan_date"] >= mode[1]) & (sub["plan_date"] <= mode[2])]
    if setup_type is not None:
        sub = sub[sub["primary_setup"] == setup_type]
    sub.to_parquet(CANDIDATE_PARQUET, index=False)
    return CANDIDATE_PARQUET


def candidate_statistics(plan_date=None):
    """③ 记录项: 正式 Gate4 前固定输出候选分布 (不修改任何规则, 仅供审计)。

    覆盖: candidates_per_day / by_setup / by_tier / bottom∩intraday_dip 重叠 /
    score 分布 / WATCH·HIGH_PRIORITY·NEAR_MISS 分布。
    """
    df = load_pool_df()
    df = df[df["effective_trade_date"].notna()]
    if plan_date is not None:
        mode = _norm_plan_date(plan_date)
        if mode[0] == "eq":
            df = df[df["plan_date"] == mode[1]]
        elif mode[0] == "range":
            df = df[(df["plan_date"] >= mode[1]) & (df["plan_date"] <= mode[2])]
    per_day = df.groupby("plan_date").size()
    stats = {
        "total_candidates": int(len(df)),
        "date_range": [int(df["plan_date"].min()), int(df["plan_date"].max())],
        "candidates_per_day": {
            "mean": round(float(per_day.mean()), 1),
            "median": float(per_day.median()),
            "min": int(per_day.min()), "max": int(per_day.max()),
        },
        "by_setup": df["primary_setup"].value_counts().to_dict(),
        "by_primary_tier": df["primary_tier"].value_counts().to_dict(),
        "by_bottom_tier": df["bottom_tier"].value_counts().to_dict(),
        "by_dip_tier": df["dip_tier"].value_counts().to_dict(),
        "bottom_score_quantiles": {f"p{int(q * 100)}": round(float(df["bottom_score"].quantile(q)), 1)
                                    for q in (0.1, 0.25, 0.5, 0.75, 0.9)},
        "intraday_dip_score_quantiles": {f"p{int(q * 100)}": round(float(df["intraday_dip_score"].quantile(q)), 1)
                                         for q in (0.1, 0.25, 0.5, 0.75, 0.9)},
        "dual_both_scores_ge_40": int(((df["bottom_score"] >= 40) & (df["intraday_dip_score"] >= 40)).sum()),
        "bottom_only": int(((df["bottom_score"] >= 40) & (df["intraday_dip_score"] < 40)).sum()),
        "dip_only": int(((df["bottom_score"] < 40) & (df["intraday_dip_score"] >= 40)).sum()),
        "bottom_subtype_present": bool("bottom_subtype" in df.columns
                                       and (df["bottom_subtype"].astype(str) != "").any()),
    }
    return stats


if __name__ == "__main__":
    import sys as _sys, json as _json
    _plan = None
    args = _sys.argv[1:]
    if "--plan-date" in args:
        i = args.index("--plan-date")
        if i + 1 < len(args):
            v = args[i + 1]
            _plan = int(v) if "-" not in v else (int(v.split("-")[0]), int(v.split("-")[1]))
    if args and args[0] == "--stats":
        print(_json.dumps(candidate_statistics(_plan), ensure_ascii=False, indent=2))
    else:
        print("usage: python green_watch_candidate_loader.py --stats [--plan-date YYYYMMDD|lo-hi]")
