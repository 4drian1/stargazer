"""计划版本接口（按 interface_contract v2 §4.2 对齐）。

合同响应：{ok, versions:[{session_id,trading_date,version_no,status(FROZEN/SUPERSEDED),
frozen_at,content_hash,candidate_count,summary}], error?, reason?}
- versions[] 来自真实历史归档 frozen_*.json（独立历史来源）；同一 session_id 内
  version_no 最大者 FROZEN，其余 SUPERSEDED。缺历史明确说明，不用当前文件充当多版本。
- 现行计划按生产实际读取规则 trade_plan_v1.load_plan() 读取（安装目录明确指定），
  作为附加字段 current 返回；只读，绝不改写生产文件。
- PG 旧档案单列（pg_old_archive），口令/查询来自 config，不硬编码。
- 支持 ?session_id= 可选过滤。

注意：本模块为 ui_next 新接口的独立命名实现（避免与生产已有模块重名）。
"""
import sys
import json
from pathlib import Path

_DESKTOP = Path(__file__).resolve().parents[1] / "stargazer_desktop"
if str(_DESKTOP) not in sys.path:
    sys.path.insert(0, str(_DESKTOP))

from ui_backend_config import desktop_dir, plan_archive_dir, pg_dsn, pg_plan_version_query


def _load_archive_entries(adir: "Path | None"):
    """frozen_*.json → 合同版本条目（status 先占位，调用方按 session 判 FROZEN/SUPERSEDED）。"""
    entries = []
    if not adir or not Path(adir).is_dir():
        return entries
    for f in sorted(Path(adir).glob("frozen_*.json")):
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
        except Exception:
            continue
        exported = data.get("exported_at")
        for s in data.get("sessions") or []:
            entries.append({
                "session_id": s.get("session_id"),
                "trading_date": data.get("trading_date"),
                "version_no": s.get("version_no"),
                "status": "FROZEN",
                "frozen_at": exported,
                "content_hash": s.get("content_hash"),
                "candidate_count": len(s.get("candidates") or []),
                "summary": s.get("session_tag"),
            })
    return entries


def _mark_frozen_status(entries):
    max_ver = {}
    for e in entries:
        k = e["session_id"]
        if k not in max_ver or (e["version_no"] or 0) > max_ver[k]:
            max_ver[k] = e["version_no"] or 0
    for e in entries:
        e["status"] = "FROZEN" if (e["version_no"] or 0) == max_ver.get(e["session_id"]) else "SUPERSEDED"
    return entries


def _read_current(loader):
    """现行计划：生产实际读取规则 trade_plan_v1.load_plan()；只读。"""
    try:
        plan = loader()
    except Exception as exc:
        return {"source_rule": "trade_plan_v1.load_plan()", "status": "ERROR",
                "error": str(exc)}
    if not plan:
        return {"source_rule": "trade_plan_v1.load_plan()", "status": "NO_PLAN"}
    import trade_plan_v1
    return {"source_rule": "trade_plan_v1.load_plan()",
            "path": str(trade_plan_v1.TRADE_PLAN_FILE),
            "status": "CURRENT",
            "version": plan.get("version"),
            "created_ts": plan.get("created_ts"),
            "data_date": plan.get("data_date"),
            "monitor_date": plan.get("monitor_date"),
            "assistant_plan_id": plan.get("assistant_plan_id"),
            "stocks_count": len(plan.get("stocks") or [])}


def plan_versions(session_id=None, current_loader=None, archive_dir=None,
                  dsn=None, psycopg_mod=None):
    import trade_plan_v1
    loader = current_loader or trade_plan_v1.load_plan
    current = _read_current(loader)

    adir = archive_dir if archive_dir is not None else plan_archive_dir()
    entries = _mark_frozen_status(_load_archive_entries(adir))
    if session_id is not None:
        entries = [e for e in entries if str(e["session_id"]) == str(session_id)]
    entries.sort(key=lambda e: (str(e.get("frozen_at") or ""), e["version_no"] or 0),
                 reverse=True)

    pg = pg_old_archive(dsn=dsn, psycopg_mod=psycopg_mod)

    notes = []
    if entries:
        notes.append("历史版本来自 frozen_*.json 独立归档（同一 session 最新版本 FROZEN，其余 SUPERSEDED）。")
    else:
        notes.append("未找到真实历史归档（frozen_*.json 缺失）；未用当前计划文件充当多版本。")
    notes.append("现行计划按生产实际读取规则 trade_plan_v1.load_plan() 读取（只读）。")

    has_current = current.get("status") == "CURRENT"
    resp = {"ok": bool(entries) or has_current,
            "versions": entries,
            "current": current,
            "pg_old_archive": pg,
            "archive": {"dir": str(adir) if adir else None,
                        "kind": "frozen_*.json（独立历史来源）"},
            "note": "".join(notes)}
    if not resp["ok"]:
        resp["reason"] = "PLAN_SOURCES_UNAVAILABLE:现行计划与历史归档均不可读"
    return resp


def pg_old_archive(dsn=None, psycopg_mod=None):
    dsn = dsn if dsn is not None else pg_dsn()
    if not dsn:
        return {"source": "pg_old_archive", "status": "NOT_CONFIGURED",
                "note": "PG 旧档案 DSN 未配置；口令不硬编码，由环境变量/可选配置提供。"}
    query = pg_plan_version_query()
    if psycopg_mod is None:
        try:
            import psycopg as psycopg_mod
        except Exception as exc:
            return {"source": "pg_old_archive", "status": "DRIVER_MISSING",
                    "error": str(exc)}
    try:
        with psycopg_mod.connect(dsn, connect_timeout=5) as conn:
            if not query:
                conn.execute("SELECT 1")
                return {"source": "pg_old_archive", "status": "CONNECTED",
                        "note": "已连接 PG 旧档案；版本列表查询未配置"
                                "（STARGAZER_PG_PLAN_VERSION_QUERY），未执行任何查询。"}
            rows = conn.execute(query).fetchall()
        return {"source": "pg_old_archive", "status": "AVAILABLE",
                "row_count": len(rows), "rows": [list(r) for r in rows],
                "note": "PG 旧档案版本列表（查询来自配置，非硬编码）。"}
    except Exception as exc:
        return {"source": "pg_old_archive", "status": "ERROR", "error": str(exc)}
