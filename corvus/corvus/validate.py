"""规则校验（validate.py）：Qwen 事实抽取的字段校验 + 证据约束派生（规格 v1.2）。

两层职责：
1. rule_check：格式/字典/时间语义/交叉核对（决定 SCHEMA_VALIDATED / NEEDS_REVIEW）；
2. 证据约束派生（规格 v1.2，用户批复 2026-09-13）：
   - effective_date 四层：raw/normalized/status(EXPLICIT|INFERRED|CONFLICT|MISSING)/evidence
     原文明确完整年月日=EXPLICIT 可用；只有月日=INFERRED 不进模型；与原文冲突=CONFLICT；
     原文没有=NULL(MISSING)；禁止 Qwen 补年份；
   - amount 四层：raw_text/value/unit/type/status(EXPLICIT|CALCULATED|AMBIGUOUS|MISSING)/evidence
     Qwen 只允许复制原文；高精度小数等原文找不到的=CALCULATED；只有 EXPLICIT 可进模型；
     P1 容器文档金额默认不参与建模；
   - assertion_status：AFFIRMED/NEGATED/UNCERTAIN/HISTORICAL_REFERENCE
     P0 安全门：关键词+标题绝不能直接产出可建模 P0——必须 AFFIRMED + 正文完整
     (FULL_TEXT/MIRROR_MATCHED) + 映射成功 + 证据段存在；
   - model_eligible：只有全部证据约束过的字段才允许进事件影响模型。

SCHEMA_VALIDATED 只代表"通过格式与字典检查"，不代表事实已被验证。
"""
import re

from .event_taxonomy import (CONTAINER_TYPES, map_canonical, tier_of,
                             underlying_hint)

_NULL_DATE_WORDS = ("未提及", "无", "暂未", "尚未", "不涉及", "null", "none", "n/a")

_DATE_PATTERNS = [
    re.compile(r"(\d{4})\s*年\s*(\d{1,2})\s*月\s*(\d{1,2})\s*日"),
    re.compile(r"(\d{4})[-/.](\d{1,2})[-/.](\d{1,2})"),
    re.compile(r"(\d{4})\s*年\s*(\d{1,2})\s*月"),           # 缺日，按当月 1 日
]

_CN_DIGIT = {"〇": 0, "零": 0, "一": 1, "二": 2, "两": 2, "三": 3, "四": 4,
             "五": 5, "六": 6, "七": 7, "八": 8, "九": 9}


def _cn_year(s):
    """'二〇二六'/'二零二六' → 2026（年份逐位转）。"""
    if len(s) == 4 and all(ch in _CN_DIGIT for ch in s):
        return int("".join(str(_CN_DIGIT[ch]) for ch in s))
    return None


def _cn_num(s):
    """简单中文数词：十/十一/二十/二十三 → int；不支持更复杂形式。"""
    if s.isdigit():
        return int(s)
    if "十" in s:
        left, _, right = s.partition("十")
        tens = _CN_DIGIT.get(left, 1) if left else 1
        ones = _CN_DIGIT.get(right, 0) if right else 0
        return tens * 10 + ones
    if all(ch in _CN_DIGIT for ch in s) and s:
        return _CN_DIGIT[s[-1]] if len(s) == 1 else None
    return None


_FULL_DATE_RE = re.compile(
    r"(20\d{2})\s*年\s*(\d{1,2})\s*月\s*(\d{1,2})\s*日|(20\d{2})[-/.](\d{1,2})[-/.](\d{1,2})"
    r"|([二〇零一二三四五六七八九]{4})\s*年\s*([一二两三四五六七八九十\d]{1,3})\s*月\s*([一二两三四五六七八九十\d]{1,3})\s*日")
_MONTH_DAY_RE = re.compile(r"(\d{1,2})\s*月\s*(\d{1,2})\s*日")

# 否定/历史语境（排除公告套话"不存在虚假记载/重大遗漏/误导性陈述"）
_NEG_RE = re.compile(r"(不存在|未发生|未涉及|不涉及|无相关|已解除|予以否认|否认|澄清公告|自查未发现|未发现)([^。；\n]{0,18})(处罚|违规|诉讼|仲裁|减持|退市|立案|风险|违约|占用)")
_BOILERPLATE = ("虚假记载", "重大遗漏", "误导性陈述", "保证公告内容")
_HIST_RE = re.compile(r"(历史上|近五年|近三年|近两年|曾经)([^。；\n]{0,15})(处罚|违规|诉讼|退市|立案|违约)")


def parse_effective_date(value):
    """返回 (date_iso 或 None, note)。None + note 表示合法 null；无法解析抛 ValueError。"""
    if value is None:
        return None, "null（字段缺失）"
    s = str(value).strip()
    if s == "" or s.lower() in _NULL_DATE_WORDS or any(w in s for w in ("未提及", "暂未", "尚未")):
        return None, f"null（{s or '空串'}）"
    for i, pat in enumerate(_DATE_PATTERNS):
        m = pat.search(s)
        if m:
            y, mo = int(m.group(1)), int(m.group(2))
            d = int(m.group(3)) if i < 2 else 1
            from datetime import date
            return date(y, mo, d).isoformat(), f"解析自: {s[:30]}"
    raise ValueError(f"无法解析日期: {s[:40]}")


def _evidence(text, start, width=60):
    return (text or "")[max(0, start - 10):start + width].replace("\n", " ")


def derive_effective_date(parsed, text):
    """四层派生 effective_date。返回 (raw, normalized, status, evidence)。"""
    raw = parsed.get("effective_date")
    raw_s = str(raw).strip() if raw is not None else ""
    text = text or ""
    if not raw_s or raw_s.lower() in _NULL_DATE_WORDS:
        return (raw_s or None, None, "MISSING", None)
    fulls = []   # (normalized, span_start)
    for m in _FULL_DATE_RE.finditer(text):
        if m.group(1) or m.group(4):          # 阿拉伯数字：YYYY年M月D日 / YYYY-M-D
            y = int(m.group(1) or m.group(4))
            mo, d = int(m.group(2) or m.group(5)), int(m.group(3) or m.group(6))
        else:                                  # 中文数字：二〇二六年九月十一日
            y = _cn_year(m.group(7))
            mo, d = _cn_num(m.group(8)), _cn_num(m.group(9))
            if y is None or mo is None or d is None:
                continue
        try:
            from datetime import date as _date
            norm = _date(y, mo, d).isoformat()
        except ValueError:
            continue
        fulls.append((norm, m.start()))
    try:
        q_val, _ = parse_effective_date(raw)
    except ValueError:
        q_val = None
    if q_val and any(n == q_val for n, _ in fulls):
        start = next(s for n, s in fulls if n == q_val)
        return (raw_s, q_val, "EXPLICIT", _evidence(text, start))
    if q_val and fulls:
        return (raw_s, q_val, "CONFLICT",
                f"Qwen={q_val} 原文完整日期={[n for n, _ in fulls][:3]}")
    if q_val and not fulls:
        # 原文无完整日期但 Qwen 给了完整值 → 疑补年份/编造
        return (raw_s, q_val, "CONFLICT", "原文无完整年月日，Qwen 值来源不明（疑补年份）")
    # Qwen 值不可解析
    md = _MONTH_DAY_RE.search(raw_s)
    if md:
        return (raw_s, None, "INFERRED", f"Qwen 仅给月日 '{md.group(0)}'，缺年份不推断")
    return (raw_s, None, "CONFLICT", f"Qwen 日期输出不可解析: {raw_s[:30]}")


def _plain_num(x):
    """数值的精确字符串形式（%g 只有 6 位有效数字，23246.07 会被修约成 23246.1 导致误判）。"""
    s = f"{x:.10f}".rstrip("0").rstrip(".")
    return s if s not in ("", "-0") else "0"


def derive_amount(parsed, text):
    """四层派生 amount。返回 (raw_text, value, unit, type, status, evidence)。"""
    amt = parsed.get("amount")
    text = text or ""
    if amt is None or not str(amt).strip():
        return (None, None, None, None, "MISSING", None)
    if isinstance(amt, str):
        # Qwen 偶发返回带格式字符串（"1,000万元"）：确定性程序归一化，转不了就 MISSING 不崩
        s = amt.replace(",", "").replace("万元", "").replace("万", "")                .replace("亿元", "").replace("元", "").replace("%", "").strip()
        try:
            amt = float(s)
        except ValueError:
            return (str(amt), None, None, None, "MISSING", f"amount 字符串不可解析: {amt[:20]}")
    a = float(amt)
    variants = {}
    if a == int(a):
        variants[str(int(a))] = None
    variants[_plain_num(a)] = None
    if a >= 1e4:
        variants[f"{_plain_num(a/1e4)}万"] = "万元"
    if a >= 1e8:
        variants[f"{_plain_num(a/1e8)}亿"] = "亿元"
    if 0 < a <= 100:
        variants[f"{_plain_num(a)}%"] = "%"
    norm_text = re.sub(r"[\s,，]", "", text)
    for v, unit in variants.items():
        nv = re.sub(r"[\s,，]", "", v)
        pos = norm_text.find(nv)
        if pos >= 0:
            status = "EXPLICIT"
            # 高精度小数原文不可得（norm 后匹配的只是前缀巧合）仍按命中处理，
            # 但分到元级以下精度时单独标 CALCULATED
            if abs(a) >= 1e6 and a != int(a):
                status = "CALCULATED"
            raw_span = re.sub(r"\s+", "", v)
            typ = "比例" if unit == "%" else "金额"
            return (str(amt), a, unit, typ, status, f"原文命中 '{v}'")
    # 原文找不到
    if abs(a) >= 1e6 and a != int(a):
        return (str(amt), a, None, "金额", "CALCULATED",
                f"{_plain_num(a)} 为高精度小数，疑模型合成计算值")
    return (str(amt), a, None, None, "AMBIGUOUS", f"原文找不到 {_plain_num(a)} 的任何字面形式")


def derive_assertion(text):
    """四态断言判定。NEGATED=否定语境；HISTORICAL_REFERENCE=历史陈述；否则 AFFIRMED。"""
    text = text or ""
    if not text.strip():
        return "UNCERTAIN"
    for m in _NEG_RE.finditer(text):
        seg = m.group(0)
        if not any(b in seg for b in _BOILERPLATE):
            return "NEGATED"
    if _HIST_RE.search(text):
        return "HISTORICAL_REFERENCE"
    return "AFFIRMED"


def validate_candidate(source_level, parsed, title, content, published_at,
                       body_status=None):
    """六字段规则校验 + 证据约束派生。body_status：raw_news 正文状态
    (FULL_TEXT/MIRROR_MATCHED/METADATA_ONLY/FAILED/None)，P0 门控用。"""
    checks = []
    warnings = []
    parsed = parsed or {}
    text = f"{title or ''}\n{content or ''}"

    def add(rule, ok, detail="", level=None):
        checks.append({"rule": rule, "pass": bool(ok),
                       "level": level or ("error" if not ok else "info"), "detail": detail})
        if ok and level == "warning":
            warnings.append(detail)

    # 0. 标准类别映射
    canon, map_note = map_canonical(parsed.get("event_type"), title)
    tier = tier_of(canon) if canon else None
    mapping_status = "MAPPED" if canon else "UNMAPPED_CONTEXT"
    if canon is None:
        add("event_type_mappable", False,
            f"'{parsed.get('event_type')}' 无法映射 → UNMAPPED_CONTEXT 观察池（不等于中性）")
    else:
        add("event_type_mappable", True,
            f"{canon}（{map_note or '直配'}）", level="warning" if map_note else "info")
        if map_note:
            warnings.append(map_note)

    # 1. entities 非空
    ents = parsed.get("entities")
    ok = isinstance(ents, list) and len([e for e in ents if str(e).strip()]) > 0
    add("entities_non_empty", ok, f"{len(ents) if isinstance(ents, list) else '非列表'} 个实体")

    # 2. action 非空
    add("action_non_empty", bool(str(parsed.get("action") or "").strip()),
        str(parsed.get("action") or "")[:50])

    # 3. effective_date 可解析或为合法 null
    try:
        eff, note = parse_effective_date(parsed.get("effective_date"))
        add("effective_date_parseable", True, note)
    except ValueError as e:
        eff = None
        add("effective_date_parseable", False, str(e))

    # 4. 来源等级封顶（confidence 待启用）
    if source_level == "AGGREGATOR":
        add("source_cap_aggregator", True, "AGGREGATOR 源：confidence 封顶 0.4 规则待启用")
    else:
        add("source_cap_aggregator", True, f"{source_level} 源无封顶")

    # 5. 时间语义（未来/即时正常；追溯 >5 天才拦，≤5 天决议日宽限）
    if eff and published_at:
        pub_day = str(published_at)[:10]
        if eff < pub_day:
            from datetime import date as _date
            y1, m1, d1 = map(int, eff.split("-"))
            y2, m2, d2 = map(int, pub_day.split("-"))
            gap = (_date(y2, m2, d2) - _date(y1, m1, d1)).days
            if gap <= 5:
                add("effective_after_publish", True,
                    f"决议/会议日期 {eff} 早于发布日 {gap} 天（披露常态，warning）",
                    level="warning")
            else:
                add("effective_after_publish", False,
                    f"RETROACTIVE：{eff} 早于发布 {gap} 天（疑编造年份或真追溯），需人工核")
        elif eff == pub_day:
            add("effective_after_publish", True, f"即时生效（{eff}）")
        else:
            add("effective_after_publish", True,
                f"未来生效（{eff} > {pub_day}，正常；可见性由 available_at 决定）")
    else:
        add("effective_after_publish", True, "无 effective_date 或无 published_at，跳过")

    # 6. 实体交叉核对
    if isinstance(ents, list) and ents:
        hit = [e for e in ents if str(e) and str(e) in text]
        add("entity_in_text", len(hit) > 0,
            f"命中: {hit[:3]}" if hit else f"entities={ents[:5]} 均未出现在标题/正文")
    else:
        add("entity_in_text", False, "无实体可核对")

    # 7. 金额交叉核对（schema 层；四层派生另见 derive_amount）
    amt = parsed.get("amount")
    if amt is not None and str(amt).strip():
        _raw, _v, _u, _t, a_status, a_note = derive_amount(parsed, text)
        add("amount_in_text", a_status in ("EXPLICIT", "MISSING"),
            f"{a_note}（四层判定 {a_status}）" if a_status != "EXPLICIT" else f"{a_note}")
    else:
        add("amount_in_text", True, "amount 为空，跳过")

    # 8. 证据约束四层派生
    eff_raw, eff_norm, eff_status, eff_evi = derive_effective_date(parsed, text)
    amt_raw, amt_val, amt_unit, amt_type, amt_status, amt_evi = derive_amount(parsed, text)
    assertion = derive_assertion(text)
    is_container = 1 if canon in CONTAINER_TYPES else 0
    add("assertion_gate", assertion == "AFFIRMED" or tier != "P0",
        f"assertion={assertion}" +
        ("" if assertion == "AFFIRMED" else "（P0 须 AFFIRMED 才可建模）"))

    # 9. model_eligible 总门控
    body_ok = body_status in ("FULL_TEXT", "MIRROR_MATCHED")
    schema_ok = all(c["pass"] for c in checks)
    if tier == "P0":
        eligible = int(schema_ok and assertion == "AFFIRMED" and body_ok
                       and amt_status in ("EXPLICIT", "MISSING")
                       and eff_status in ("EXPLICIT", "MISSING"))
    elif tier == "P1":
        # 容器文档默认不进模型（金额/日期字段在容器内无意义），识别底层事件后升级
        eligible = int(schema_ok and body_ok and is_container == 0
                       and assertion in ("AFFIRMED", "HISTORICAL_REFERENCE"))
    else:
        eligible = 0   # P2 / UNMAPPED_CONTEXT 一律不进模型
    if tier == "P0" and not eligible and schema_ok:
        add("p0_gate", False,
            f"P0 未过安全门：assertion={assertion}, body={body_status}, "
            f"amount={amt_status}, effective={eff_status}（关键词不得直接产出可建模 P0）")
    else:
        add("p0_gate", True, "非 P0 或已满足 P0 安全门")

    status = "SCHEMA_VALIDATED" if all(c["pass"] for c in checks) else "NEEDS_REVIEW"
    return {"checks": checks, "status": status, "warnings": warnings,
            "canonical_event_type": canon, "materiality_tier": tier if canon else "UNKNOWN",
            "mapping_status": mapping_status, "model_eligible": eligible,
            "is_container_document": is_container,
            "underlying_event_types": underlying_hint(canon) if canon else [],
            "effective_date_raw": eff_raw, "effective_date_normalized": eff_norm,
            "effective_date_status": eff_status, "effective_date_evidence": eff_evi,
            "amount_raw_text": amt_raw, "amount_unit": amt_unit,
            "amount_type": amt_type, "amount_status": amt_status,
            "amount_evidence": amt_evi, "assertion_status": assertion}
