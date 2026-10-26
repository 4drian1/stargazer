"""trade_plan_v1.py — 交易计划存储与规则判断（Stargazer 1.0 UI 框架接线）。

数据契约 dist/data/trade_plan.json：
    {version, created_ts, data_date, monitor_date,
     stocks: [{code, name, buy_zone: [lower, upper], invalidations: [...],
               valid_until, evidence_refs: [...]}],
     boards: [...]}
由 server 端 validate_plan 校验后原子落盘。plan_check 只按计划 JSON 做
**规则判断**（接近买入区/触发/失效）；模型决策位留空，见下方接入点注释。
"""
from __future__ import annotations

import json
import math
import os
import re
from pathlib import Path

import paths

CONTRACT = 'STARGAZER_TRADE_PLAN_V1'
TRADE_PLAN_FILE = Path(paths.data_file(
    'data/trade_plan.json' if paths.FROZEN else 'dist/data/trade_plan.json',
    seed_from_bundle=False))
# 现价高于买入区上限多少以内算"接近"（百分比）
NEAR_ZONE_PCT = 1.5

_TS14 = re.compile(r'^\d{14}$')
_D8 = re.compile(r'^\d{8}$')
_CODE = re.compile(r'^\d{6}$')


def validate_plan(plan: dict) -> list[str]:
    """契约校验：返回错误列表（空=合法）。"""
    errs = []
    if not isinstance(plan, dict):
        return ['PLAN_NOT_DICT']
    if plan.get('version') != 1:
        errs.append('VERSION_INVALID')
    if not _TS14.match(str(plan.get('created_ts', ''))):
        errs.append('CREATED_TS_INVALID')
    for k in ('data_date', 'monitor_date'):
        if not _D8.match(str(plan.get(k, ''))):
            errs.append(f'{k.upper()}_INVALID')
    stocks = plan.get('stocks')
    if not isinstance(stocks, list) or not stocks:
        errs.append('STOCKS_REQUIRED')
        stocks = []
    seen = set()
    for i, s in enumerate(stocks):
        tag = f'STOCKS[{i}]'
        if not isinstance(s, dict):
            errs.append(f'{tag}_NOT_DICT')
            continue
        code = str(s.get('code', ''))
        if not _CODE.match(code):
            errs.append(f'{tag}_CODE_INVALID')
        elif code in seen:
            errs.append(f'{tag}_CODE_DUP')
        else:
            seen.add(code)
        if not str(s.get('name', '')).strip():
            errs.append(f'{tag}_NAME_EMPTY')
        zone = s.get('buy_zone')
        if s.get('observe_only') is True and zone is None:
            pass  # No price region: observation only, never an executable price alert.
        elif (not isinstance(zone, list) or len(zone) != 2
                or not all(isinstance(x, (int, float)) and not isinstance(x, bool)
                           and math.isfinite(x) and x > 0 for x in zone)):
            errs.append(f'{tag}_BUY_ZONE_INVALID')
        elif zone[0] > zone[1]:
            errs.append(f'{tag}_BUY_ZONE_ORDER')
        inv = s.get('invalidations')
        if not isinstance(inv, list) or not all(isinstance(x, str) for x in inv):
            errs.append(f'{tag}_INVALIDATIONS_TYPE')
        if not _TS14.match(str(s.get('valid_until', ''))):
            errs.append(f'{tag}_VALID_UNTIL_INVALID')
        refs = s.get('evidence_refs')
        if not isinstance(refs, list) or not all(isinstance(x, str) for x in refs):
            errs.append(f'{tag}_EVIDENCE_REFS_TYPE')
    boards = plan.get('boards')
    if not isinstance(boards, list):
        errs.append('BOARDS_TYPE')
    return errs


def load_plan(path: Path | None = None) -> dict | None:
    """读计划；文件缺失/坏 JSON → None（UI 显示"暂无计划"）。"""
    file = path or TRADE_PLAN_FILE
    try:
        plan = json.loads(file.read_text(encoding='utf-8'))
    except (OSError, json.JSONDecodeError):
        return None
    return plan if isinstance(plan, dict) else None


def save_plan(plan: dict, path: Path | None = None) -> Path:
    """校验后原子落盘（临时文件+替换）；不合法抛 ValueError。"""
    errs = validate_plan(plan)
    if errs:
        raise ValueError('|'.join(errs[:8]))
    file = path or TRADE_PLAN_FILE
    file.parent.mkdir(parents=True, exist_ok=True)
    tmp = file.with_suffix('.tmp')
    tmp.write_text(json.dumps(plan, ensure_ascii=False), encoding='utf-8')
    os.replace(tmp, file)
    return file


def plan_stock(plan: dict, code: str) -> dict | None:
    for s in plan.get('stocks') or []:
        if str(s.get('code')) == str(code):
            return s
    return None


def plan_check(code: str, price, ts: str | None = None,
               plan: dict | None = None) -> dict:
    """规则判断（框架先行）：接近买入区/触发/失效/观察中。

    输入 code + 当前价 + 时刻（14位，缺省=本地现在）；输出结构化状态。
    # 模型决策接入点：数据链验收后在此接 decision_layer——
    # invalidations 的文本失效条件判定、触发后的综合建议均交由模型位，
    # 规则结果降级为模型输入特征；接入前 decision_source 恒为 'RULE'。
    """
    from datetime import datetime
    now_ts = ts or datetime.now().strftime('%Y%m%d%H%M%S')
    out = dict(schema=CONTRACT, code=str(code), price=price, ts=now_ts,
               status='ERROR', reason='', decision_source='RULE',
               decision_note='规则触发，模型决策待接入')
    if plan is None:
        plan = load_plan()
    if not plan:
        out.update(status='NO_PLAN', reason='暂无计划')
        return out
    stock = plan_stock(plan, code)
    if stock is None:
        out.update(status='NOT_IN_PLAN', reason='该股票不在当前交易计划中')
        return out
    out['stock'] = dict(code=stock.get('code'), name=stock.get('name'),
                        buy_zone=stock.get('buy_zone'),
                        valid_until=stock.get('valid_until'))
    try:
        px = float(price)
    except (TypeError, ValueError):
        out.update(status='ERROR', reason='PRICE_INVALID')
        return out
    if not math.isfinite(px) or px <= 0:
        out.update(status='ERROR', reason='PRICE_INVALID')
        return out
    if str(now_ts) > str(stock.get('valid_until', '')):
        out.update(status='EXPIRED', reason='计划有效期已过')
        return out
    if stock.get('entry_rule'):
        from golden_rules import VERSION
        if stock['entry_rule'].get('version') not in (VERSION,'TAROT_A_SECOND_CASH_V2'):
            out.update(status='ERROR', reason='UNKNOWN_ENTRY_RULE_VERSION')
            return out
        if stock['entry_rule'].get('version') == 'TAROT_A_SECOND_CASH_V2':
            from tarot_monitor_v2 import status_for
            out.update(status_for(plan, code, now_ts))
            return out
        from golden_monitor import status_for
        out.update(status_for(plan, code, now_ts), decision_source='GOLDEN_RULE_OBSERVATION',
                   decision_note='完整下杀止跌背离规则观察，不是自动买入')
        return out
    if stock.get('observe_only') is True and stock.get('buy_zone') is None:
        out.update(status='OBSERVE_ONLY', reason='无有效关注区域，仅观察，不生成价格触发提醒')
        return out
    zone = stock.get('buy_zone')
    if (not isinstance(zone, list) or len(zone) != 2 or any(
            not isinstance(v, (int, float)) or isinstance(v, bool) or not math.isfinite(v) or v <= 0 for v in zone)
            or zone[0] > zone[1]):
        out.update(status='ERROR', reason='BUY_ZONE_INVALID')
        return out
    lower, upper = zone
    if lower <= px <= upper:
        out.update(status='TRIGGERED', reason='现价进入计划买入区',
                   buy_zone=stock['buy_zone'])
    elif px > upper and (px - upper) / upper * 100 <= NEAR_ZONE_PCT:
        out.update(status='NEAR', reason='接近买入区（上方 %.2f%%）' % ((px - upper) / upper * 100),
                   buy_zone=stock['buy_zone'])
    elif px < lower:
        out.update(status='BELOW_ZONE', reason='现价低于买入区下限',
                   buy_zone=stock['buy_zone'])
    else:
        out.update(status='WATCHING', reason='观察中，未接近买入区')
    return out
