"""事件正式化与实体映射（event_loop.py，第 3 步 3.2/3.3）。

formalize_events：SQLite 候选（model_eligible=1）→ PG information_event。
- 幂等锚 UNIQUE(raw_id, event_version)；重复运行不产生重复行；
- 去重：同 body_hash + canonical 类型 + 规范化标题只正式化一个 canonical；
- available_at = published_at（缺省退 collected_at），是标签起点唯一依据；
- P1 容器文档不进（候选侧 model_eligible 已挡，这里再挡一次）。

map_entities：规则优先的实体映射（规格 3.3）。
- RULE_CODE_OFFICIAL：原文"证券代码/代码：XXXXXX"（含名称佐证）conf=0.95；
- RULE_NAME_MATCH：公司全称命中（parquet name→code 表）conf=0.80；
- BOARD_KEYWORD：标题命中用户观察板块关键词 conf=0.60；
- LLM_CANDIDATE：Qwen 实体提名未获规则证实 conf=0.40（不可直接训练）；
- 宏观/法规/制裁类 → MARKET/GLOBAL 映射，不强行贴个股。
"""
import hashlib
import json
import re
import sqlite3
from pathlib import Path

from .db import get_conn as get_sqlite
from .extract_candidates import load_source_levels, read_pdf_text, read_raw_text
from .event_taxonomy import map_canonical
from .pgstore import get_pg_conn, init_pg_schema

_CODE_RE = re.compile(r"(?<!\d)(\d{6})(?!\d)")
_CODE_LABEL_RE = re.compile(r"(?:证券代码|证券简称|股票代码|代码)[：: ]*(\d{6})")
_NORM_TITLE = re.compile(r"[\s\W_]+", re.UNICODE)

DATA_PARTS = Path(r"D:/a股数据/data_daily_parts")
_DESKTOP_DATA = Path(__file__).resolve().parent.parent.parent / "stargazer_desktop" / "data"

MARKET_TYPES = {"MACRO_POLICY", "REGULATION_RELEASE"}          # → MARKET
GLOBAL_TYPES = {"SANCTION_TRADE", "MACRO_POLICY"}              # 宏观兼具 GLOBAL 属性
SECTOR_TYPE = "INDUSTRY_POLICY"


def _norm(t):
    return _NORM_TITLE.sub("", str(t or ""))


def _sha(s):
    return hashlib.sha256(str(s).encode("utf-8")).hexdigest()


# ---------------------------------------------------------------- 名称→代码表
_name_map_cache = None


def load_name_map():
    """从日线 parquet 的 name 列建 公司名→代码 映射（全称优先；重名取最新）。"""
    global _name_map_cache
    if _name_map_cache is not None:
        return _name_map_cache
    import pandas as pd
    m = {}
    for f in sorted(DATA_PARTS.glob("part_*.parquet")):
        df = pd.read_parquet(f, columns=["code", "name", "date"])
        df = df.sort_values("date").drop_duplicates("code", keep="last")
        for _, r in df.iterrows():
            nm = str(r["name"] or "").strip()
            if len(nm) >= 3:          # 全名/常用名，短简称（≤2字）不用于自动映射
                m[nm] = r["code"]
    _name_map_cache = m
    return m


def load_watch_boards():
    """用户观察板块（桌面程序配置，只读借用）。"""
    p = _DESKTOP_DATA / "user_watch_sectors.json"
    if not p.exists():
        return []
    try:
        d = json.loads(p.read_text(encoding="utf-8"))
        return [str(x) for x in (d.get("sectors") or d if isinstance(d, list) else [])]
    except Exception:
        return []


# ---------------------------------------------------------------- 正式化
def formalize_events(pg, sl, verbose=True):
    """把 SQLite 中 model_eligible=1 的候选正式化进 PG。返回 {candidate_id: event_id}。"""
    init_pg_schema(pg)
    levels = load_source_levels()
    rows = sl.execute(
        """SELECT c.*, r.title, r.url, r.published_at, r.collected_at, r.raw_path,
                  r.mime, r.detail_mime, r.body_hash
           FROM information_event_candidate c JOIN raw_news r ON r.raw_id=c.raw_id
           WHERE c.model_eligible=1 AND c.candidate_status='SCHEMA_VALIDATED'""").fetchall()
    # 去重键：body_hash + 类型 + 规范化标题（同一公告多源报道只正式化一次）
    seen, mapping, n_new, n_dup = {}, {}, 0, 0
    for c in rows:
        key = (c["body_hash"] or _sha(c["raw_id"]),
               c["canonical_event_type"], _norm(c["title"]))
        if key in seen:
            n_dup += 1
            mapping[c["candidate_id"]] = seen[key]
            continue
        available_at = c["published_at"] or c["collected_at"]
        eff_norm = c["effective_date_normalized"]  # DATE 字符串或 None
        cur = pg.execute(
            """INSERT INTO information_event
               (candidate_id, raw_id, title, canonical_event_type, event_type_raw,
                materiality_tier, assertion_status, source_id, source_level,
                source_url, body_hash, evidence_span, entities, action,
                amount_value, amount_unit, amount_type, amount_status, amount_evidence,
                official_scope, effective_date_raw, effective_date_normalized,
                effective_date_status, effective_date_evidence,
                published_at, available_at, effective_at, first_seen_at,
                event_version, model_eligible)
               VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,1,TRUE)
               ON CONFLICT (raw_id, event_version) DO NOTHING
               RETURNING event_id""",
            (c["candidate_id"], c["raw_id"], c["title"], c["canonical_event_type"],
             c["event_type"], c["materiality_tier"], c["assertion_status"],
             c["source_id"], levels.get(c["source_id"]), c["url"],
             c["body_hash"] or _sha(c["raw_id"]),
             (c["effective_date_evidence"] or c["amount_evidence"]
              or str(c["title"] or ""))[:200],
             c["entities"], c["action"],
             c["amount"], c["amount_unit"], c["amount_type"], c["amount_status"],
             c["amount_evidence"], c["official_scope"],
             c["effective_date_raw"], eff_norm, c["effective_date_status"],
             c["effective_date_evidence"],
             c["published_at"], available_at,
             eff_norm, c["collected_at"]))
        got = cur.fetchone()
        if got:
            n_new += 1
            seen[key] = got[0]
            mapping[c["candidate_id"]] = got[0]
        else:  # 已存在（幂等重跑）
            eid = pg.execute(
                "SELECT event_id FROM information_event WHERE raw_id=%s AND event_version=1",
                (c["raw_id"],)).fetchone()[0]
            seen[key] = eid
            mapping[c["candidate_id"]] = eid
    pg.commit()
    if verbose:
        print(f"正式化：新增 {n_new}，去重合并 {n_dup}，可建模候选共 {len(rows)} 条")
    return mapping


def _valid_code(c):
    """A 股代码有效性：60/68 沪主板科创、00/30 深主板创业、8/4/92 北交所。"""
    return bool(c) and len(c) == 6 and c[0] in "03468"


# ---------------------------------------------------------------- 实体映射
def _event_text(sl, raw_id):
    r = sl.execute("SELECT raw_path, mime, detail_mime FROM raw_news WHERE raw_id=?",
                   (raw_id,)).fetchone()
    if not r or not r["raw_path"]:
        return ""
    try:
        title, text = read_raw_text(r["raw_path"], r["mime"])
        if not text and r["detail_mime"] == "application/pdf":
            text = read_pdf_text(r["raw_path"])
    except FileNotFoundError:
        # 从机合并行原文未拷回：如实返回空文本（evidence 降级），不得崩整条链
        return ""
    return f"{title or ''}\n{text or ''}"


def map_entities(pg, sl, verbose=True):
    """为每个正式事件生成实体链接（幂等：ON CONFLICT DO NOTHING）。"""
    name_map = load_name_map()
    boards = load_watch_boards()
    events = pg.execute(
        """SELECT event_id, candidate_id, raw_id, title, canonical_event_type,
                  entities, source_id
           FROM information_event WHERE event_version=1""").fetchall()
    n_link = 0
    for e in events:
        eid, cand_id, raw_id, title, canon = e[0], e[1], e[2], e[3], e[4]
        ents = []
        try:
            ents = json.loads(e[5] or "[]")
        except Exception:
            pass
        text = _event_text(sl, raw_id)[:20000]
        links = []   # (entity_type, entity_id, relation, method, conf, evidence)

        # 1) 规则：证券代码（标签佐证，最高优先；有效性过滤，PDF 页码等数字串不得入库）
        codes = []
        for m in _CODE_LABEL_RE.finditer(text):
            if _valid_code(m.group(1)):
                codes.append((m.group(1), text[max(0, m.start()-30):m.end()+30]))
        for m in _CODE_RE.finditer(str(title or "")):
            if _valid_code(m.group(1)):
                codes.append((m.group(1), f"标题: {title[:60]}"))
        for code, ev in codes:
            links.append(("DIRECT_STOCK", code, "PRIMARY",
                          "RULE_CODE_OFFICIAL", 0.95, ev.replace("\n", " ")[:80]))

        # 2) 规则：公司全称命中
        if not any(l[0] == "DIRECT_STOCK" for l in links):
            for nm, cd in name_map.items():
                if len(nm) >= 4 and nm in text:
                    links.append(("DIRECT_STOCK", cd, "PRIMARY",
                                  "RULE_NAME_MATCH", 0.80, f"全称命中: {nm}"))
                    break  # 一只公告默认一个主体公司

        # 3) LLM 提名（未获规则证实，低置信，仅供审计/后续复核）
        known = {l[1] for l in links}
        for ent in ents:
            s = str(ent).strip()
            if (len(s) >= 4 and s not in known
                    and not _CODE_RE.fullmatch(s)
                    and s not in text[:0]):   # 占位，保持结构清晰
                links.append(("UNKNOWN_SCOPE", s[:60], "SECONDARY",
                              "LLM_CANDIDATE", 0.40, f"Qwen提名: {s}"))

        # 4) 宏观/法规/制裁 → 市场/全球；行业政策 → 板块关键词
        if canon in MARKET_TYPES:
            links.append(("MARKET", "A_SHARE_MARKET", "PRIMARY",
                          "RULE_TAXONOMY", 0.90, f"{canon} 属宏观/法规类"))
        if canon in GLOBAL_TYPES:
            links.append(("GLOBAL", "GLOBAL_MACRO", "AFFECTED",
                          "RULE_TAXONOMY", 0.85, f"{canon} 含外围传导属性"))
        if canon == SECTOR_TYPE:
            hit = next((b for b in boards if b and b in (title or "")), None)
            if hit:
                links.append(("SECTOR", hit, "PRIMARY",
                              "BOARD_KEYWORD", 0.60, f"标题命中观察板块: {hit}"))
            else:
                links.append(("UNKNOWN_SCOPE", "UNMAPPED_SECTOR", "PRIMARY",
                              "RULE_TAXONOMY", 0.50, "行业政策但板块未映射"))

        # 5) 落库（幂等）
        seen_link = set()
        for et, eid_s, rel, meth, conf, ev in links:
            k = (et, eid_s, meth)
            if k in seen_link:
                continue
            seen_link.add(k)
            pg.execute(
                """INSERT INTO event_entity_link
                   (event_id, entity_type, entity_id, relation_type, mapping_method,
                    mapping_confidence, evidence)
                   VALUES (%s,%s,%s,%s,%s,%s,%s) ON CONFLICT DO NOTHING""",
                (eid, et, eid_s, rel, meth, conf, ev[:200]))
            n_link += 1
    pg.commit()
    if verbose:
        print(f"实体映射：写入 {n_link} 条链接（{len(events)} 个事件）")
    return n_link


def main(argv=None):
    sl = get_sqlite()
    pg = get_pg_conn()
    try:
        init_pg_schema(pg)
        formalize_events(pg, sl)
        map_entities(pg, sl)
    finally:
        sl.close()
        pg.close()


if __name__ == "__main__":
    main()
