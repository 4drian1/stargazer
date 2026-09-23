# 临时补丁脚本：qc_extract.py 适配 SCHEMA_VALIDATED + 分类法新字段（ASCII 锚点）
import io

p = "corvus/qc_extract.py"
src = io.open(p, encoding="utf-8").read()

def rep(old, new):
    global src
    assert old in src, f"NOT FOUND: {old[:60]!r}"
    src = src.replace(old, new, 1)

# 1) 删除 event_type_canonical 回写（validate 不再返回该键）
rep('''                status = rc["status"]
                if p.get("event_type") is None and rc.get("event_type_canonical"):
                    p["event_type"] = rc.get("event_type_canonical")
            else:''',
    '''                status = rc["status"]
            else:''')

# 2) parse 失败分支的 rc 结构补新字段
rep('''"status": "NEEDS_REVIEW", "warnings": [], "event_type_canonical": None}''',
    '''"status": "NEEDS_REVIEW", "warnings": [],
                      "canonical_event_type": None, "materiality_tier": None,
                      "is_container_document": 0, "underlying_event_types": []}''')

# 3) 状态计数改名
rep('''total["auto_validated" if status == "AUTO_VALIDATED" else "needs_review"] += 1''',
    '''total["schema_validated" if status == "SCHEMA_VALIDATED" else "needs_review"] += 1''')

# 4) INSERT 语句带新列
rep('''INSERT INTO information_event_candidate
                   (raw_id, source_id, event_type, entities, action, amount,
                    official_scope, effective_date, extract_raw, parse_ok, latency_s,
                    model_version, rule_check, candidate_status, created_at)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (rid, r["source_id"], p.get("event_type"),
                 json.dumps(p.get("entities"), ensure_ascii=False)
                 if p.get("entities") is not None else None,
                 p.get("action"), p.get("amount"), p.get("official_scope"),''',
    '''INSERT INTO information_event_candidate
                   (raw_id, source_id, event_type, canonical_event_type, materiality_tier,
                    is_container_document, underlying_event_types,
                    entities, action, amount, official_scope, effective_date,
                    extract_raw, parse_ok, latency_s, model_version, rule_check,
                    candidate_status, created_at)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (rid, r["source_id"], p.get("event_type"), rc.get("canonical_event_type"),
                 rc.get("materiality_tier"), rc.get("is_container_document", 0),
                 json.dumps(rc.get("underlying_event_types") or [], ensure_ascii=False),
                 json.dumps(p.get("entities"), ensure_ascii=False)
                 if p.get("entities") is not None else None,
                 p.get("action"), p.get("amount"), p.get("official_scope"),''')

# 5) verbose 打印 canonical 类型
rep('''f"status={status} type={p.get('event_type')}", flush=True)
    return total''',
    '''f"status={status} type={rc.get('canonical_event_type') or p.get('event_type')}",
                      flush=True)
    total.setdefault("schema_validated", 0)
    return total''')

io.open(p, "w", encoding="utf-8", newline="\n").write(src)
print("patched OK")
