"""canonicalize_dataset.py — 把 events_SUPERSEDED_V2_labeled.parquet 规范化为 C3 生产数据契约。

产出：
- event_cluster.parquet      1 行 / event_cluster_id（cluster 级标签 + 代表性文档字段）
- event_document_link.parquet 全量公告级（文档字段 + event_cluster_id + available_at）

同时落地 A（canonical 有效性重建）：
- corp_action_within_horizon==True 的样本：
    label_valid = False
    label_invalid_reason = CORPORATE_ACTION_WINDOW
    training_eligible = False
  原始 price/return/path 字段与 materiality raw 一律保留（不重算 alpha）。

设计要点（与现 walk_forward 训练域完全对齐）：
- 代表性文档行优先取 cluster 内 P0 公告（保证 tier/category 与现有 drop_duplicates 行为一致）。
- cluster_tier = 若 cluster 含 P0 则为 P0，否则取首行 tier（P0 判定不变 -> 训练域不变）。
"""
from __future__ import annotations

import datetime as dt

import pandas as pd

OUT = "out"
SRC = f"{OUT}/events_SUPERSEDED_V2_labeled.parquet"
CLUSTER_OUT = f"{OUT}/event_cluster.parquet"
DOC_OUT = f"{OUT}/event_document_link.parquet"


def log(m: str):
    print(f"[{dt.datetime.now():%H:%M:%S}] {m}", flush=True)


def main():
    log(f"读取 {SRC} ...")
    ev = pd.read_parquet(SRC)
    ev = ev.drop(columns=[c for c in ev.columns if c.startswith("__")])
    n_rows = len(ev)
    log(f"行数(公告级)= {n_rows}, 唯一 cluster= {ev['event_cluster_id'].nunique()}")

    doc_cols = ["title", "announce_time", "raw_url", "title_hash",
                "dedup_key", "category", "tier", "matched_keyword"]

    # 代表性文档行：cluster 内优先 P0（与 walk_forward 现有 tier==P0 过滤 + drop_duplicates 对齐）
    ev["_is_p0"] = (ev["tier"] == "P0")
    ev_sorted = ev.sort_values(["event_cluster_id", "_is_p0"],
                               ascending=[True, False], kind="mergesort")
    rep = (ev_sorted.groupby("event_cluster_id", sort=False).head(1)
           .set_index("event_cluster_id"))

    # cluster_tier 修正：含 P0 即 P0（避免首行非 P0 但存在 P0 公告时漏 cluster）
    p0_per_cluster = ev.groupby("event_cluster_id")["_is_p0"].any()
    rep["tier"] = rep.index.map(
        lambda cid: "P0" if p0_per_cluster.get(cid, False) else rep.loc[cid, "tier"])

    # ---- A: canonical 有效性重建 ----
    corp = rep["corp_action_within_horizon"].fillna(False).astype(bool)
    before_valid = rep["label_valid"].value_counts(dropna=False).to_dict()
    # label_invalid_reason 全 None 被存为 float64，先转 object 才能赋字符串
    rep["label_invalid_reason"] = rep["label_invalid_reason"].astype(object)
    rep.loc[corp, "label_valid"] = False
    rep.loc[corp, "label_invalid_reason"] = "CORPORATE_ACTION_WINDOW"
    rep["training_eligible"] = (rep["label_valid"] == True)
    after_valid = rep["label_valid"].value_counts(dropna=False).to_dict()
    log(f"label_valid before={before_valid}")
    log(f"label_valid after ={after_valid}  (corp 置 False 数= {int(corp.sum())})")
    log(f"training_eligible True= {int(rep['training_eligible'].sum())} / {len(rep)}")

    # 写 cluster 主表（保留所有列：cluster 标签 + 代表性文档字段）
    rep_out = rep.reset_index()
    rep_out.to_parquet(CLUSTER_OUT, index=False)
    log(f"写 {CLUSTER_OUT}: {len(rep_out)} 行")

    # 文档 link 表
    doc = ev[doc_cols + ["event_cluster_id", "code", "announce_date",
                         "name", "label_status", "available_at_precision"]].copy()
    doc = doc.rename(columns={"announce_date": "available_at_date"})
    doc.to_parquet(DOC_OUT, index=False)
    log(f"写 {DOC_OUT}: {len(doc)} 行")

    # ---- C 的 cluster 级 UNMAPPED 统计 ----
    n_clusters = len(rep)
    rep_cat = rep["category"]
    unmapped_cluster = int((rep_cat == "UNMAPPED").sum())
    p0_mask = (rep["tier"] == "P0")
    p0_clusters = int(p0_mask.sum())
    p0_unmapped = int(((rep_cat == "UNMAPPED") & p0_mask).sum())
    # 含任何/全部 UNMAPPED 公告的 cluster
    any_unmapped = ev.groupby("event_cluster_id")["category"].apply(
        lambda s: (s == "UNMAPPED").any())
    all_unmapped = ev.groupby("event_cluster_id")["category"].apply(
        lambda s: (s == "UNMAPPED").all())
    log("=== C cluster 级 UNMAPPED 统计 ===")
    log(f"clusters 总数= {n_clusters}")
    log(f"代表性 category==UNMAPPED 的 cluster= {unmapped_cluster} "
        f"({unmapped_cluster/n_clusters:.1%})")
    log(f"P0 cluster 总数= {p0_clusters}, 其中代表性 UNMAPPED= {p0_unmapped} "
        f"({p0_unmapped/p0_clusters:.1%})")
    log(f"含>=1 UNMAPPED 公告的 cluster= {int(any_unmapped.sum())} "
        f"({int(any_unmapped.sum())/n_clusters:.1%})")
    log(f"全部公告 UNMAPPED 的 cluster= {int(all_unmapped.sum())} "
        f"({int(all_unmapped.sum())/n_clusters:.1%})")


if __name__ == "__main__":
    main()
