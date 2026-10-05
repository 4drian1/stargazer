"""unmapped_audit.py — C 最小闭环审计：UNMAPPED 内部文档构成。

问题：UNMAPPED 占公告行 52%、cluster 代表性 44%，是否把明显不同的语义全塞进垃圾桶？
方法：对 UNMAPPED 公告标题做关键词分桶（ROUTINE_DISCLOSURE / POTENTIALLY_MATERIAL /
AMBIGUOUS / TRUE_MAPPING_MISS），并统计最频繁标题模板。
目的：判断 taxonomy 覆盖质量，NOT 用结果改模型。
"""
from __future__ import annotations

import datetime as dt
import re

import pandas as pd

DOC = "out/event_document_link.parquet"


def log(m: str):
    print(f"[{dt.datetime.now():%H:%M:%S}] {m}", flush=True)


ROUTINE = ["报告", "半年报", "季报", "年度", "独立意见", "独立董事", "审计报告",
           "内部控制", "社会责任", "可持续", "提示性公告", "进展公告", "补充公告",
           "更正公告", "问询函", "关注函", "问询", "摘要", "说明", "通知", "公告书",
           "权益变动", "简式", "详式"]
MATERIAL = ["股权", "质押", "冻结", "担保", "诉讼", "仲裁", "中标", "合同", "增持",
            "减持", "回购", "分红", "送转", "派息", "业绩", "停牌", "复牌", "并购",
            "重组", "资产", "投资", "融资", "借款", "授信", "理财", "关联交易",
            "重大", "风险", "处罚", "警示", "退市", "要约", "定增", "配股", "股权激励"]


def bucket(t: str) -> str:
    if not isinstance(t, str) or not t.strip():
        return "EMPTY"
    for k in ROUTINE:
        if k in t:
            return "ROUTINE_DISCLOSURE"
    for k in MATERIAL:
        if k in t:
            return "POTENTIALLY_MATERIAL"
    if len(t) <= 6:
        return "AMBIGUOUS_SHORT"
    return "TRUE_MAPPING_MISS"


def main():
    log(f"读取 {DOC} ...")
    doc = pd.read_parquet(DOC, columns=["event_cluster_id", "title", "category"])
    unm = doc[doc["category"] == "UNMAPPED"].copy()
    n = len(unm)
    log(f"UNMAPPED 公告行= {n} ({n/len(doc):.1%})")

    unm["bucket"] = unm["title"].map(bucket)
    vc = unm["bucket"].value_counts()
    log("=== UNMAPPED 分桶 ===")
    for k, v in vc.items():
        log(f"  {k}: {v} ({v/n:.1%})")

    # 最频繁标题模板（前 30，去空白后）
    unm["t_norm"] = unm["title"].astype(str).str.strip()
    top = unm["t_norm"].value_counts().head(30)
    log("=== UNMAPPED 最高频标题(top30) ===")
    for t, c in top.items():
        log(f"  [{c:>6}] {t}")

    # 与整体(含 mapped)对比：mapped 公告的 bucket 分布（用同样的标题规则）
    mapped = doc[doc["category"] != "UNMAPPED"].copy()
    mapped["bucket"] = mapped["title"].map(bucket)
    log("=== 对照：MAPPED 公告分桶 ===")
    for k, v in mapped["bucket"].value_counts().items():
        log(f"  {k}: {v} ({v/len(mapped):.1%})")


if __name__ == "__main__":
    main()
