"""Deterministic learning-record facts; LLM supplies only a bounded interpretation."""
from collections import Counter
from datetime import datetime
import math
import statistics

VERSION = 2
MAX_COMMENTARY_CHARS = 1200


def finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def text(value):
    return str(value if value is not None else '未知').replace('\n', ' ').replace('|', '\\|')[:100]


def percent(value):
    return f'{value:+.2f}%' if finite(value) else 'NULL（未知）'


def summary(facts):
    rows = facts.get('rows', [])
    touched = [r for r in rows if r.get('daily_touch') is True]
    untouched = [r for r in rows if r.get('daily_touch') is False]
    changes = [r['close_change_pct'] for r in touched if finite(r.get('close_change_pct'))]
    not_changes = [r['close_change_pct'] for r in untouched if finite(r.get('close_change_pct'))]
    valid = sorted([r for r in rows if finite(r.get('close_change_pct'))],
                   key=lambda r: (r['close_change_pct'], r['code']))
    examples, seen = [], set()
    for r in valid[:3] + list(reversed(valid[-3:])):
        if r['code'] not in seen:
            examples.append({k: r.get(k) for k in ('code', 'name', 'daily_touch', 'close_change_pct', 'price_source')})
            seen.add(r['code'])
    actual = facts.get('actual_trades') or {}
    if not isinstance(actual, dict):
        actual = {}
    return dict(data_date=facts.get('data_date'), observe_date=facts.get('observe_date'),
                candidate_count=facts['candidate_count'], detail_rows=len(rows),
                valid_daily_rows=facts['summary']['valid_daily_rows'],
                setup_counts=dict(Counter(r.get('setup') or 'UNKNOWN' for r in rows)),
                touch_known=len(touched)+len(untouched), touch_count=len(touched),
                touch_unknown=len(rows)-len(touched)-len(untouched),
                touch_change_n=len(changes), touch_mean=statistics.mean(changes) if changes else None,
                touch_median=statistics.median(changes) if changes else None,
                touch_positive=sum(v > 0 for v in changes),
                untouched_change_n=len(not_changes),
                untouched_mean=statistics.mean(not_changes) if not_changes else None,
                examples=examples, market=facts.get('market') or {'status': 'UNKNOWN'},
                seven_day_status_counts=dict(Counter(r.get('status') or 'UNKNOWN' for r in facts.get('seven_day', []))),
                actual_trades={k: actual.get(k) for k in ('status', 'realized_gross_pnl', 'realized_net_pnl')})


def prompt_context(facts):
    value = summary(facts)
    value['snapshot_hash'] = facts.get('snapshot_hash')
    value['examples_notice'] = '仅最多6个参考收盘变化样例，不代表全体；统计使用完整明细，日线不能证明承接/成交。'
    return value


def render(facts, commentary, *, confirmed=False):
    s = summary(facts)
    day = datetime.strptime(str(facts['observe_date']), '%Y%m%d').strftime('%Y-%m-%d')
    touch_rate = f"{s['touch_count']/s['touch_known']*100:.1f}%" if s['touch_known'] else 'NULL（未知）'
    setups = ' / '.join(f'{text(k)}={v}' for k, v in sorted(s['setup_counts'].items())) or '未知'
    states = ' / '.join(f'{text(k)}={v}' for k, v in sorted(s['seven_day_status_counts'].items())) or '未提供'
    actual = s['actual_trades']
    gross = actual.get('realized_gross_pnl')
    net = actual.get('realized_net_pnl')
    state = '人工确认' if confirmed else '待人工确认'
    lines = [f'## {day}（程序统计＋DeepSeek短判读，{state}）', '',
        f"- **数据日/观察日**：{s['data_date']} → {s['observe_date']}",
        f"- **名单**：完整候选 {s['candidate_count']} 只；明细 {s['detail_rows']} 条；{setups}。",
        f"- **市场背景**：{text(s['market'].get('status', 'UNKNOWN'))}；未提供已核验方向时不推断强弱或beta。",
        f"- **总体数字**：有效日K {s['valid_daily_rows']}/{s['candidate_count']}；"
        f"可判定区域触及 {s['touch_known']}，触区 {s['touch_count']}（{touch_rate}）；未知 {s['touch_unknown']}。",
        f"- **触区组参考收盘变化**：有效参考 {s['touch_change_n']}；均值 {percent(s['touch_mean'])}，"
        f"中位 {percent(s['touch_median'])}，高于参考收盘 {s['touch_positive']}/{s['touch_change_n']}；不是买入收益或K线收阳数。",
        f"- **未触区组**：有效参考 {s['untouched_change_n']}；均值 {percent(s['untouched_mean'])}。", '',
        '### 典型案例（程序选取参考收盘变化两端，不认定反弹/承接）', '',
        '| 代码 | 名称 | 日线触区 | 相对名单参考收盘 |', '|---|---|---|---:|']
    for r in s['examples']:
        touch = '是' if r['daily_touch'] is True else '否' if r['daily_touch'] is False else '未知'
        lines.append(f"| {text(r['code'])} | {text(r['name'])} | {touch} | {percent(r['close_change_pct'])} |")
    if not s['examples']:
        lines.append('| — | 未提供有效参考收盘样例 | 未知 | NULL |')
    lines += ['', '### 七日观察与真实成交（分账）', '', f'- 七日状态：{states}；未成熟/未验执行不补零。',
        f"- 独立真实成交账本：{text(actual.get('status'))}；已匹配毛盈亏 "
        f"{f'{gross:.2f} 元' if finite(gross) else 'NULL'}；净盈亏 {f'{net:.2f} 元' if finite(net) else 'NULL（费用/证据未知）'}。",
        '- 日线触区≠买入；七日MFE/MAE≠可兑现收益；未知口径不得用最高/最低价伪造成交。', '',
        '### 当日判读（AI辅助，需人工复核）', '', commentary.strip(), '', '### 当日索引', '',
        f"- {day}：触区组参考收盘均值 {percent(s['touch_mean'])}；有效参考 {s['touch_change_n']}。历史序列见已存旧篇。"]
    return '\n'.join(lines)
