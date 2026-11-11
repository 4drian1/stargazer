"""Closed, per-plan execution parameters. Text never becomes executable code."""
import hashlib
import json
import math
import re

DEFAULT_POLICY = 'POSITION_TWO_OF_THREE_O20_CROSS1_V1'
DEFAULT = dict(required_votes=2, momentum_crosses_required=1, oversold_ratio=.20)


def validate(value=None):
    value = dict(DEFAULT if value is None else value)
    if set(value) != set(DEFAULT):
        raise ValueError('规则只支持三项所需票数、上穿次数和严重超卖比例；其他条件需先实现')
    for key in ('required_votes', 'momentum_crosses_required'):
        if type(value[key]) is not int or not 1 <= value[key] <= 3:
            raise ValueError('规则次数须为1至3的整数')
    ratio = value['oversold_ratio']
    if type(ratio) not in (int, float) or not math.isfinite(ratio) or not 0 < ratio < 1:
        raise ValueError('超卖比例须在0与1之间')
    return value


def policy(value=None):
    value = validate(value)
    if value == DEFAULT:
        return DEFAULT_POLICY
    return 'PLAN_CUSTOM_V1:' + hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()[:16]


def describe(value=None):
    p = validate(value)
    return ('位置到达必需；位置修复、动能、严重超卖三选%d；动能=跨日底背离＋%d次RSI6上穿RSI15＋当前快线在上；'
            '超卖=今日完整5分钟RSI6＜昨日固定低点RSI6×%g%%；到位即推，未通过持续观察，AI否决也推送，人工决定。'
            % (p['required_votes'], p['momentum_crosses_required'], p['oversold_ratio'] * 100))


def parse(text):
    """Only exact supported semantic phrases; broad paraphrases go through tool parameters."""
    if re.fullmatch(r'[\s，。！]*(?:用|采用|按|就用)?(?:默认(?:三选二)?|原来的|原规则|三选二)(?:规则|监控)?[\s，。！]*', text):
        return dict(DEFAULT)
    p = dict(DEFAULT)
    changed = False
    patterns = [('required_votes', r'三选([一二三123])', {'一': 1, '二': 2, '三': 3}),
                ('momentum_crosses_required', r'([一二两三123])次上穿', {'一': 1, '二': 2, '两': 2, '三': 3})]
    remaining = text
    for key, pattern, digits in patterns:
        hit = re.search(pattern, text)
        if hit:
            token = hit[1]
            p[key] = digits[token] if token in digits else int(token)
            remaining = remaining.replace(hit[0], '')
            changed = True
    hit = re.search(r'(?:超卖|比例)[^\d]{0,8}(\d+(?:\.\d+)?)\s*[%％]', remaining)
    if hit:
        p['oversold_ratio'] = float(hit[1]) / 100
        remaining = remaining.replace(hit[0], '')
        changed = True
    remaining = re.sub(r'默认|其余不变|其他不变|规则|改成|改为|使用|采用|设为|按|用|加|和|且|，|、|。|\s', '', remaining)
    return validate(p) if changed and not remaining else None
