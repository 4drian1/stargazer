"""存量候选重校验（revalidate.py）：校验器/分类法/证据约束升级后，用新规则重算既有候选。

只重算未人工评审过的行（PENDING/NEEDS_REVIEW/SCHEMA_VALIDATED/P2_ROUTED），
HUMAN_ACCEPTED / HUMAN_REJECTED 一律不动。

用法：python -m corvus.revalidate
"""
import json
from collections import Counter

from .db import get_conn, init_db
from .event_taxonomy import (CONTAINER_TYPES, map_canonical, tier_of,
                             underlying_hint)
from .extract_candidates import load_source_levels, read_pdf_text, read_raw_text
from .validate import validate_candidate


def revalidate_all(conn, verbose=True):
    levels = load_source_levels()
    rows = conn.execute(
        """SELECT c.candidate_id, c.raw_id, c.source_id, c.event_type, c.entities,
                  c.action, c.amount, c.official_scope, c.effective_date,
                  c.candidate_status,
                  r.title, r.raw_path, r.mime, r.detail_mime, r.published_at,
                  r.body_status
           FROM information_event_candidate c JOIN raw_news r ON r.raw_id = c.raw_id
           WHERE c.candidate_status IN ('PENDING','NEEDS_REVIEW','SCHEMA_VALIDATED',
                                        'AUTO_VALIDATED','P2_ROUTED')""").fetchall()
    n_up = 0
    stat = Counter()
    elig = Counter()
    for c in rows:
        title, text = (None, None)
        if c["raw_path"]:
            title, text = read_raw_text(c["raw_path"], c["mime"])
            if not text and c["detail_mime"] == "application/pdf":
                text = read_pdf_text(c["raw_path"])
        else:
            title, text = None, None

        # P2_ROUTED 或无语义字段的行：只补 canonical 映射与分层
        if c["candidate_status"] == "P2_ROUTED" or not c["event_type"]:
            canon, note = map_canonical(c["event_type"], c["title"])
            tier = tier_of(canon) if canon else "UNKNOWN"
            conn.execute(
                """UPDATE information_event_candidate SET canonical_event_type=?,
                   materiality_tier=?, mapping_status=?, model_eligible=0,
                   is_container_document=?,
                   underlying_event_types=? WHERE candidate_id=?""",
                (canon, tier, "MAPPED" if canon else "UNMAPPED_CONTEXT",
                 1 if canon in CONTAINER_TYPES else 0,
                 json.dumps(underlying_hint(canon), ensure_ascii=False)
                 if canon else None, c["candidate_id"]))
            stat[f"P2_ROUTED->{tier}"] += 1
            n_up += 1
            continue

        ents = json.loads(c["entities"]) if c["entities"] else None
        parsed = {"event_type": c["event_type"], "entities": ents, "action": c["action"],
                  "amount": c["amount"], "official_scope": c["official_scope"],
                  "effective_date": c["effective_date"]}
        # 正文实读成功即视为 FULL_TEXT（镜像列主要服务 SSE 审计）；读不到则按库状态或 METADATA_ONLY
        if c["body_status"] in ("METADATA_ONLY", "FAILED"):
            eff_body = c["body_status"]
        elif text and len(text.strip()) >= 50:
            eff_body = "FULL_TEXT"
        else:
            eff_body = "METADATA_ONLY"
        rc = validate_candidate(levels.get(c["source_id"], "AGGREGATOR"), parsed,
                                c["title"], text, c["published_at"],
                                body_status=eff_body)
        conn.execute(
            """UPDATE information_event_candidate SET rule_check=?,
               candidate_status=?, canonical_event_type=?, materiality_tier=?,
               mapping_status=?, model_eligible=?, is_container_document=?,
               underlying_event_types=?, effective_date_raw=?,
               effective_date_normalized=?, effective_date_status=?,
               effective_date_evidence=?, amount_raw_text=?, amount_unit=?,
               amount_type=?, amount_status=?, amount_evidence=?,
               assertion_status=? WHERE candidate_id=?""",
            (json.dumps({"checks": rc["checks"], "status": rc["status"],
                         "warnings": rc["warnings"]}, ensure_ascii=False),
             rc["status"], rc["canonical_event_type"], rc["materiality_tier"],
             rc["mapping_status"], rc["model_eligible"], rc["is_container_document"],
             json.dumps(rc["underlying_event_types"], ensure_ascii=False),
             rc["effective_date_raw"], rc["effective_date_normalized"],
             rc["effective_date_status"], rc["effective_date_evidence"],
             rc["amount_raw_text"], rc["amount_unit"], rc["amount_type"],
             rc["amount_status"], rc["amount_evidence"], rc["assertion_status"],
             c["candidate_id"]))
        stat[rc["status"]] += 1
        elig[rc["materiality_tier"]] += 1
        n_up += 1
    conn.commit()
    if verbose:
        print(f"重校验 {n_up} 条；状态分布: {dict(stat)}")
        print(f"model_eligible 分布(按 tier): {dict(elig)}")
    return n_up


def main(argv=None):
    init_db()
    conn = get_conn()
    try:
        revalidate_all(conn)
    finally:
        conn.close()


if __name__ == "__main__":
    main()
