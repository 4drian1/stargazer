"""全球市场统一总览（按 interface_contract v2 §4.3 与前端 GlobalOverviewItem 对齐）。

合同响应：{ok, mode, note, generated_at(ISO+偏移), items[], reason?}
每项 GlobalOverviewItem：variety, category, value, unit, change, change_unit, basis,
market, timezone, quote_time(ISO+偏移), quote_flag, scrape_time, status(OK/SAMPLE/
MISSING/ERROR:.../STALE), reason, sample。

纪律（含两轮审查修正）：
- GET 零网络：只读隔离行情快照（isolated/，绝不读生产文件、绝不现场采集）。
- 快照缺失 → 默认全部 MISSING（不提供样本）；样本骨架仅在显式演示模式
  （STARGAZER_DEMO_MODE=1）下给出，且 decision_use 恒为 false。
- 独立检查样本标记与每项行情时间，不能只看快照生成时间：
  * 任一项 sample:true → 该项 SAMPLE，整体 decision_use=false；
  * 单项 quote_time 无法解析/来自未来 → ERROR:TIME_INVALID；
  * 单项 quote_time 距今超过 STARGAZER_GLOBAL_OVERVIEW_MAX_AGE_HOURS → STALE；
  * 快照 generated_at 无法解析/来自未来/整体过期 → 整体降级。
- 缺失、样本、过期数据不得整体标成可用于决策（decision_use=false）。
- 新闻流与全球行情分开：本接口只输出行情品种，不含新闻 items。

注意：本模块为 ui_next 新接口的独立命名实现（避免与生产已有模块重名）。
"""
import sys
import json
import math
import os
from pathlib import Path
from datetime import datetime, timedelta, timezone as _tz

_DESKTOP = Path(__file__).resolve().parents[1] / "stargazer_desktop"
if str(_DESKTOP) not in sys.path:
    sys.path.insert(0, str(_DESKTOP))

from ui_backend_config import global_overview_snapshot_path, global_overview_max_age_hours

# 第一批品种目录（variety, category, unit, market, timezone, sample_capable）
CATALOG = [
    ("道琼斯工业平均指数", "美股指数", "点", "美国", "America/New_York", True),
    ("纳斯达克综合指数", "美股指数", "点", "美国", "America/New_York", True),
    ("标普500", "美股指数", "点", "美国", "America/New_York", True),
    ("COMEX黄金", "贵金属", "美元/盎司", "美国", "America/New_York", True),
    ("COMEX白银", "贵金属", "美元/盎司", "美国", "America/New_York", True),
    ("WTI原油", "能源", "美元/桶", "美国", "America/New_York", True),
    ("布伦特原油", "能源", "美元/桶", "欧洲", "Europe/London", True),
    ("日经225", "股指", "点", "日本", "Asia/Tokyo", False),
    ("恒生指数", "股指", "点", "香港", "Asia/Hong_Kong", False),
    ("欧洲斯托克50", "股指", "点", "欧洲", "Europe/Paris", False),
    ("中国10年期国债收益率", "国债", "%", "中国", "Asia/Shanghai", False),
    ("美国10年期国债收益率", "国债", "%", "美国", "America/New_York", False),
]

# 样本值：仅显式演示模式提供，sample=true 明确标注，绝不进入决策
SAMPLE_VALUES = {
    "道琼斯工业平均指数": 43000.0, "纳斯达克综合指数": 18500.0, "标普500": 5800.0,
    "COMEX黄金": 2650.0, "COMEX白银": 31.0, "WTI原油": 70.0, "布伦特原油": 74.0,
}

_FUTURE_TOLERANCE_S = 300


def _parse_iso(ts):
    if ts is None:
        return None
    try:
        s = str(ts).strip()
        if s.endswith("Z"):
            s = s[:-1] + "+00:00"
        dt = datetime.fromisoformat(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=_tz.utc)
        return dt
    except (ValueError, TypeError):
        return None


def _now_iso():
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _is_demo():
    return os.environ.get("STARGAZER_DEMO_MODE", "").strip() == "1"


def _base_item(variety, category, unit, market, tz):
    return {"variety": variety, "category": category, "unit": unit,
            "market": market, "timezone": tz, "value": None, "change": None,
            "change_unit": None, "basis": None, "quote_time": None,
            "quote_flag": None, "scrape_time": None,
            "status": "MISSING", "reason": None, "sample": False}


def _skeleton(demo: bool, covered_hint=None):
    """无快照：默认全部 MISSING（零网络、无样本）；demo=True 才给样本骨架。
    covered_hint 非 None 表示处于快照模式（未覆盖品种按 MISSING+未覆盖说明）。"""
    items = []
    for variety, category, unit, market, tz, sample_ok in CATALOG:
        it = _base_item(variety, category, unit, market, tz)
        if demo and sample_ok:
            it.update({"value": SAMPLE_VALUES[variety], "status": "SAMPLE",
                       "sample": True,
                       "reason": "样本值（显式演示模式），仅界面联调，不可用于决策"})
        else:
            it["reason"] = ("快照缺失，暂无该品种行情（不提供样本）"
                            if covered_hint is None else "快照未覆盖该品种（第一批未接源）")
        items.append(it)
    return items


def _load_snapshot(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    except Exception:
        return "UNREADABLE"


def _degrade_all(items, status, reason):
    for it in items:
        it["status"] = status
        it["reason"] = reason


def _is_daily_bond(item, row):
    """国债收益率是日度发布序列，不能按分时行情的小时级时效判定。"""
    return (item.get("category") == "国债" and
            str(row.get("quote_flag") or "") == "AKSHARE_BOND_ZH_US_RATE" and
            bool(row.get("data_date") or row.get("quote_time")))


def _daily_bond_stale(item, row, now, max_age_hours):
    """按日度发布日期判定新鲜度；返回 True 时写入具体 STALE 原因。

    基点差(change/bp)由采集端用两个**不同有效发布日**的同口径值计算；
    这里只读结果，不重算、不拿单日值冒充变动。
    """
    day = str(row.get("data_date") or "")[:10] or str(row.get("quote_time") or "")[:10]
    try:
        published = datetime.fromisoformat(day).date()
    except (ValueError, TypeError):
        item["reason"] = f"日度发布日期无法解析（{day}），按过期处理，不得标为决策可用"
        return True
    now_day = now.astimezone(_tz(timedelta(hours=8))).date()
    lag = (now_day - published).days
    if lag < 0:
        item["reason"] = f"日度发布日期 {day} 来自未来，不得标为决策可用"
        return True
    # 周五发布遇周末要到周一才更新；日度数据至少允许到下一个有效发布日。
    allowed = max(4, math.ceil(float(max_age_hours) / 24.0))
    if lag > allowed:
        item["reason"] = (f"日度发布日 {day} 距今 {lag} 个自然日（允许 ≤{allowed} 天），"
                          f"已过期，决策不可用；不按小时级行情口径误判")
        return True
    return False


def global_overview(snapshot_path=None, max_age_hours=None):
    """GET 契约：零网络，只读隔离快照；缺失/过期/时间异常/样本标记逐级降级。"""
    path = Path(snapshot_path) if snapshot_path else global_overview_snapshot_path()
    max_age = global_overview_max_age_hours() if max_age_hours is None else max_age_hours
    now = datetime.now().astimezone()

    snap = _load_snapshot(path)
    if snap is None:
        demo = _is_demo()
        return {"ok": True, "mode": "sample" if demo else "missing",
                "generated_at": _now_iso(), "items": _skeleton(demo),
                "decision_use": False, "snapshot_path": str(path),
                "note": ("显式演示模式：第一批 SAMPLE/MISSING 骨架，样本值不可用于决策。"
                         if demo else
                         "隔离行情快照缺失：全部 MISSING（不提供样本，不伪造数据）；"
                         "样本骨架仅在显式演示模式（STARGAZER_DEMO_MODE=1）提供。")}
    if snap == "UNREADABLE":
        items = _skeleton(False)
        _degrade_all(items, "ERROR:SNAPSHOT_UNREADABLE", "快照文件无法解析，已整体降级")
        return {"ok": True, "mode": "snapshot_error", "generated_at": _now_iso(),
                "items": items, "decision_use": False,
                "snapshot_path": str(path),
                "note": "隔离行情快照无法解析，整体降级，不可用于决策。"}

    items = _skeleton(False, covered_hint=True)
    by_variety = {it["variety"]: it for it in items}
    for row in snap.get("items") or []:
        it = by_variety.get(row.get("variety"))
        if it is None:
            continue  # 不在第一批目录中的品种不展示（避免自造结构）
        it.update({"value": row.get("value"), "change": row.get("change"),
                   "change_unit": row.get("change_unit"),
                   "basis": row.get("basis"),
                   "market": row.get("market") or it["market"],
                   "timezone": row.get("timezone") or it["timezone"],
                   "quote_time": row.get("quote_time"),
                   "quote_flag": row.get("quote_flag"),
                   "scrape_time": row.get("scrape_time"),
                   "sample": bool(row.get("sample", False))})
        # 保留采集端写入的具体缺失原因（如日经225 东财断连/字段无效），
        # 不得用通用「无数值」把它覆盖掉。
        collected_reason = str(row.get("reason") or "").strip()
        date_only = _is_daily_bond(it, row) or it["quote_flag"] == "STOXX_LAST_CLOSE"
        # 独立检查样本标记：样本永远不是 OK，不进入决策
        if it["sample"]:
            it["status"] = "SAMPLE"
            it["reason"] = "快照标记 sample:true，样本值不可用于决策"
            continue
        qt = _parse_iso(it["quote_time"])
        if it["value"] is None:
            it["status"] = "MISSING"
            it["reason"] = collected_reason or "快照无该品种数值"
        elif qt is None:
            it["status"] = "ERROR:TIME_INVALID"
            it["reason"] = f"quote_time 无法解析（{it['quote_time']}），不得标为决策可用"
        elif qt > now + timedelta(seconds=_FUTURE_TOLERANCE_S):
            it["status"] = "ERROR:TIME_INVALID"
            it["reason"] = "quote_time 来自未来，不得标为决策可用"
        elif date_only and _daily_bond_stale(it, row, now, max_age):
            it["status"] = "STALE"
        elif not date_only and (now - qt).total_seconds() > max_age * 3600.0:
            it["status"] = "STALE"
            it["reason"] = (f"行情时间已过期（{round((now - qt).total_seconds() / 3600.0, 1)}"
                            f" 小时，超过 {max_age} 小时），决策不可用")
        else:
            it["status"] = "OK"
            it["quote_flag"] = it["quote_flag"] or "LAST_CLOSE"
            it["reason"] = None  # 不残留骨架的「快照未覆盖该品种」文字

    # 快照级时间校验（generated_at）
    gen = _parse_iso(snap.get("generated_at"))
    mode, decision_use = "snapshot", True
    if gen is None:
        mode, decision_use = "time_invalid", False
        _degrade_all(items, "ERROR:TIME_INVALID",
                     "快照 generated_at 无法解析，整体降级，不得标为决策可用")
    elif gen > now + timedelta(seconds=_FUTURE_TOLERANCE_S):
        mode, decision_use = "time_future", False
        _degrade_all(items, "ERROR:TIME_INVALID",
                     "快照 generated_at 来自未来，整体降级，不得标为决策可用")
    else:
        age_h = (now - gen).total_seconds() / 3600.0
        if age_h > max_age:
            mode, decision_use = "stale", False
            for it in items:
                if it["status"] == "OK":
                    it["status"] = "STALE"
                    it["reason"] = f"快照已过期（{round(age_h, 1)} 小时），决策不可用"
        else:
            decision_use = all(it["status"] == "OK" and not it["sample"]
                               for it in items)

    return {"ok": True, "mode": mode, "generated_at": _now_iso(),
            "items": items, "decision_use": decision_use,
            "snapshot_path": str(path),
            "snapshot_generated_at": snap.get("generated_at"),
            "note": ("真实隔离快照可用。" if mode == "snapshot" and decision_use else
                     "快照存在但未达可用标准（缺失/样本/过期/时间异常），已按项降级。")}
