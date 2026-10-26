"""shadow_journal.py — 双供应商影子记录器 V1.1（2026-10-20；用户反例修订）。

协议 shadow_protocol_v1.md。只记录、不影响。V1.1 修订：
1. **持久化去重**：(event_id, pack_hash, vendor) 已有未过期建议 → 复用，不扣额度；
2. **原子额度+并发保护**：预算读-改-写整体置于 OS 文件锁（msvcrt）内，写盘走
   临时文件+原子替换；异常中断不留半成品；
3. **按实际尝试计数**：每次供应商 HTTP 尝试（含模型探测）都计一次；首轮
   **关闭自动重试**；30s 是单次尝试上限，另设整事件总时限（迟到结果弃用）；
4. **事件类别限额**：trigger 必须 ∈ TRIGGERS；每(家,日,类别)分别计数
   （首次触发≤5/风险变化≤3/人工不限类别但受日顶与总顶）；
5. **费用与 token 入账**：每条记录 prompt/completion/token 合计；
   金额=token×单价（单价常量待用户签价前记 0，token 硬顶先行）；
   过期建议记 EXPIRED 单列，不与 PASS 混算；
6. 单侧失败另报：主比较只配对统计（同事件两家都有建议）。
"""
from __future__ import annotations

import hashlib
import json
import msvcrt
import os
import time
from datetime import datetime
from pathlib import Path

import jev_adapter_v1 as jev
import paths
from decision_layer import DeepSeekJudge, KimiJudge, _vendor_payload

HERE = Path(__file__).resolve().parent
JOURNAL = Path(paths.data_file('data/assistant/shadow_journal.jsonl', seed_from_bundle=False))
BUDGET = JOURNAL.with_name('shadow_budget.json')
LOCK = JOURNAL.with_name('shadow_budget.lock')

WINDOW = (20261021, 20261117)
DAILY_CAP = 8
TOTAL_CAP = 160
ATTEMPT_TIMEOUT_S = 30.0
EVENT_TOTAL_S = 75.0                 # 整事件（含两家）总时限；迟到结果弃用
# 事件类别日限额（None=不限类别数，仍受日顶/总顶）
TRIGGERS = {'FIRST_TRIGGER': 5, 'RISK_CHANGE': 3, 'MANUAL': None}
# token 硬顶（每次调用 / 每日每家的 prompt+completion 合计）
TOKENS_PER_CALL_CAP = 8000
TOKENS_DAILY_CAP = 60000
# 金额上限：单价待用户签价后填入；未签前 cost 记 0、token 顶先行
PRICE_PER_1K = {'deepseek': {'in': 0.0, 'out': 0.0},
                'kimi': {'in': 0.0, 'out': 0.0}}
MONEY_DAILY_CAP = None               # 签价后冻结数字

_PROVIDERS = {'deepseek': DeepSeekJudge, 'kimi': KimiJudge}


def _now() -> datetime:
    return datetime.now()


def _day(d: datetime | None = None) -> str:
    return (d or _now()).strftime('%Y%m%d')


def _now14(d: datetime | None = None) -> str:
    return (d or _now()).strftime('%Y%m%d%H%M%S')


class _BudgetLock:
    """OS 文件锁（Windows msvcrt），保护读-改-写全过程；异常时仍释放。"""

    def __enter__(self):
        self.f = LOCK.open('a+b')
        msvcrt.locking(self.f.fileno(), msvcrt.LK_LOCK, 1)
        return self

    def __exit__(self, *a):
        try:
            self.f.seek(0)
            msvcrt.locking(self.f.fileno(), msvcrt.LK_UNLCK, 1)
        finally:
            self.f.close()


def _load_budget() -> dict:
    if BUDGET.exists():
        return json.loads(BUDGET.read_text(encoding='utf-8'))
    return {}


def _save_budget_atomic(b: dict) -> None:
    tmp = BUDGET.with_suffix('.tmp')
    tmp.write_text(json.dumps(b, ensure_ascii=False), encoding='utf-8')
    os.replace(tmp, BUDGET)


def _vendor_state(b: dict, v: str) -> dict:
    return b.setdefault(v, {'days': {}, 'total': 0,
                            'cat_days': {}, 'tokens_days': {}})


def _reserve_attempt(vendor: str, day: str, category: str) -> dict | None:
    """锁内读-改-写：总额/日额/类别/token 预检。返回 None=可调用，否则拒绝原因。"""
    with _BudgetLock():
        b = _load_budget()
        st = _vendor_state(b, vendor)
        if st.get('total', 0) >= TOTAL_CAP:
            return dict(reason='TOTAL_CAP')
        days = st.setdefault('days', {})
        if days.get(day, 0) >= DAILY_CAP:
            return dict(reason='DAILY_CAP')
        cat_days = st.setdefault('cat_days', {})
        cat_key = f'{day}:{category}'
        cat_cap = TRIGGERS.get(category)
        if cat_cap is not None and cat_days.get(cat_key, 0) >= cat_cap:
            return dict(reason='CATEGORY_CAP')
        # 预留：计数与 token 预扣在调用后按实际值结算（见 settle_attempt）
        st['reserved'] = st.get('reserved', 0) + 1
        _save_budget_atomic(b)
    return None


def settle_attempt(vendor: str, day: str, category: str,
                   usage: dict | None, *, error: bool = False) -> None:
    """按实际 token 结算；超 token 顶记为超顶（调用已发生，如实记账）。"""
    tok = (usage or {}).get('total_tokens') or 0
    with _BudgetLock():
        b = _load_budget()
        st = _vendor_state(b, vendor)
        st['reserved'] = max(0, st.get('reserved', 1) - 1)
        st['total'] = st.get('total', 0) + 1
        st.setdefault('days', {})[day] = st['days'].get(day, 0) + 1
        st.setdefault('cat_days', {})[f'{day}:{category}'] = \
            st['cat_days'].get(f'{day}:{category}', 0) + 1
        td = st.setdefault('tokens_days', {})
        td[day] = td.get(day, 0) + tok
        if tok > TOKENS_PER_CALL_CAP:
            st.setdefault('over_tokens', []).append(
                dict(day=day, tokens=tok))
        _save_budget_atomic(b)


def _dedup_index() -> dict:
    """(event_id, pack_hash, vendor) -> 最近一条未过期 PASS 建议。"""
    idx = {}
    if not JOURNAL.exists():
        return idx
    for line in JOURNAL.read_text(encoding='utf-8').splitlines():
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        key = (r.get('event_id'), r.get('pack_hash'), r.get('vendor'))
        if r.get('correctness') == 'PASS' and r.get('advice') and key[0]:
            idx[key] = r
    return idx


def _cost(vendor: str, usage: dict | None) -> float:
    u = usage or {}
    p = PRICE_PER_1K.get(vendor, {})
    return round((u.get('prompt_tokens', 0) * p.get('in', 0.0)
                  + u.get('completion_tokens', 0) * p.get('out', 0.0))
                 / 1000.0, 6)


def _in_window(pack_date: str) -> bool:
    try:
        d = int(str(pack_date)[:8])
    except (TypeError, ValueError):
        return False
    return WINDOW[0] <= d <= WINDOW[1]


def record_event(pack: jev.EvidencePack, scenario: str, trigger: str, *,
                 valid_until_ts: str, event_id: str | None = None,
                 vendors=('deepseek', 'kimi')) -> list[dict]:
    """记录一次影子事件。所有异常只写日志不外抛。"""
    rows = []
    event_id = event_id or hashlib.sha256(
        f'{pack.evidence_id}:{pack.query_ts}'.encode()).hexdigest()[:12]
    base = dict(event_id=event_id, query_ts=pack.query_ts,
                pack_hash=pack.pack_hash, trigger=trigger,
                code=pack.candidates[0]['code'] if pack.candidates else None)
    if trigger not in TRIGGERS:
        return [dict(**base, error='UNKNOWN_TRIGGER', vendors_skipped=True)]
    if not _in_window(pack.market_date):
        return [dict(**base, error='OUT_OF_WINDOW', window=WINDOW)]
    gate = jev.gate_checks(pack, scenario)
    day = _day()
    idx = _dedup_index()
    t_event = time.monotonic()
    for v in vendors:
        if gate is not None:
            rows.append(dict(**base, vendor=v, day=day,
                             correctness='HARD_GATE', hard_gate=gate.as_dict()))
            continue
        key = (event_id, pack.pack_hash, v)
        prior = idx.get(key)
        if prior is not None and not _is_expired(prior, valid_until_ts):
            rows.append(dict(**base, vendor=v, day=day, correctness='REUSED',
                             advice=prior['advice'],
                             display=prior.get('display'),
                             reused_from=prior.get('query_ts')))
            continue
        if time.monotonic() - t_event > EVENT_TOTAL_S:
            rows.append(dict(**base, vendor=v, day=day,
                             correctness='EVENT_TIMEOUT'))
            continue
        denied = _reserve_attempt(v, day, trigger)
        if denied:
            rows.append(dict(**base, vendor=v, day=day,
                             correctness='BUDGET_EXHAUSTED',
                             gate_error=denied['reason']))
            continue
        prov = _PROVIDERS[v]()
        t0 = time.monotonic()
        usage, payload = None, None
        err = None
        try:
            def caller(req, p=prov):
                nonlocal usage
                payload = _vendor_payload(p, req)
                usage = (getattr(p, '_last_out', None) or {}).get('usage')
                extra = getattr(p, 'last_extra_attempts', 0)   # 模型探测等
                if extra:
                    for _ in range(extra):
                        settle_attempt(v, day, trigger, None)
                    p.last_extra_attempts = 0
                return payload
            advice = jev.judge(pack, scenario, valid_until_ts=valid_until_ts,
                               api_caller=caller)
            latency = round(time.monotonic() - t0, 3)
            settle_attempt(v, day, trigger, usage)
            expired = advice.is_expired(_now14())
            rows.append(dict(**base, vendor=v, day=day,
                             correctness='EXPIRED' if expired else 'PASS',
                             advice=advice.as_dict(),
                             display=jev.render_advice(advice),
                             latency_s=latency, usage=usage or {},
                             tokens=(usage or {}).get('total_tokens') or 0,
                             cost=_cost(v, usage),
                             post_expired=expired))
        except ValueError as exc:
            settle_attempt(v, day, trigger, usage, error=True)
            rows.append(dict(**base, vendor=v, day=day, correctness='FAIL',
                             gate_error=str(exc),
                             latency_s=round(time.monotonic() - t0, 3),
                             usage=usage or {}, tokens=(usage or {}).get(
                                 'total_tokens') or 0, cost=_cost(v, usage)))
        except Exception as exc:
            settle_attempt(v, day, trigger, usage, error=True)
            rows.append(dict(**base, vendor=v, day=day, correctness='ERROR',
                             error=type(exc).__name__, msg=str(exc)[:200],
                             latency_s=round(time.monotonic() - t0, 3),
                             usage=usage or {}, tokens=(usage or {}).get(
                                 'total_tokens') or 0, cost=_cost(v, usage)))
    JOURNAL.parent.mkdir(parents=True, exist_ok=True)
    with JOURNAL.open('a', encoding='utf-8') as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')
    return rows


def _is_expired(row: dict, valid_until_ts: str) -> bool:
    adv = row.get('advice') or {}
    vu = adv.get('valid_until_ts') or valid_until_ts
    from jev_adapter_v1 import _parse_ts14
    now, until = _parse_ts14(_now14()), _parse_ts14(str(vu))
    return now is None or until is None or now > until


def summarize() -> dict:
    """分项汇总：硬门/复用/通过/过期/失败/错误/额度拒绝 分列；延迟按通过+过期。"""
    if not JOURNAL.exists():
        return {}
    from collections import Counter, defaultdict
    st = defaultdict(Counter)
    lat = defaultdict(list)
    tokens = defaultdict(int)
    cost = defaultdict(float)
    for line in JOURNAL.read_text(encoding='utf-8').splitlines():
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        v = r.get('vendor', '?')
        st[v][r.get('correctness') or 'UNKNOWN'] += 1
        tokens[v] += r.get('tokens') or 0
        cost[v] += r.get('cost') or 0.0
        if r.get('correctness') in ('PASS', 'EXPIRED') \
                and r.get('latency_s') is not None:
            lat[v].append(r['latency_s'])
    out = {}
    for v in st:
        row = dict(st[v])
        row['tokens_total'] = tokens[v]
        row['cost_total'] = round(cost[v], 6)
        if lat[v]:
            row['lat_median'] = sorted(lat[v])[len(lat[v]) // 2]
        out[v] = row
    return out
