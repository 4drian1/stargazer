"""Offline plan/trigger/review contract. No market IO, model fit, orders or notifications."""
from copy import deepcopy
from datetime import datetime, timedelta, timezone
import hashlib
import json
import math

CST = timezone(timedelta(hours=8))
COMPONENTS = ('market_location', 'momentum', 'sector', 'leader')
STATUSES = {'PASS', 'WAIT', 'BLOCK', 'UNKNOWN'}


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     allow_nan=False, separators=(',', ':')).encode('utf8')).hexdigest()


def timestamp(value):
    if not isinstance(value, str):
        raise ValueError('TIMESTAMP_STRING_REQUIRED')
    dt = datetime.fromisoformat(value)
    if dt.tzinfo is None:
        raise ValueError('TIMEZONE_REQUIRED')
    return dt.astimezone(CST)


def day(value):
    if type(value) is not int or len(str(value)) != 8:
        raise ValueError('DATE_FORMAT')
    return datetime.strptime(str(value), '%Y%m%d').replace(tzinfo=CST)


def digest_valid(value):
    return isinstance(value, str) and len(value) == 64 and all(c in '0123456789abcdef' for c in value)


def sessions(date):
    d = day(date)
    return [(d.replace(hour=9, minute=30), d.replace(hour=11, minute=30)),
            (d.replace(hour=13), d.replace(hour=15))]


def in_session(t, date):
    return any(start <= t <= end for start, end in sessions(date))


def freeze_plan(content, calendar):
    """Pure in-memory snapshot. Calendar provenance/source admission remains external."""
    c = deepcopy(content)
    if calendar != sorted(set(calendar)):
        raise ValueError('CALENDAR_NOT_SORTED_UNIQUE')
    for d in calendar:
        if day(d).weekday() >= 5:
            raise ValueError('CALENDAR_WEEKEND')
    if c['P'] not in calendar or calendar.index(c['P']) + 1 >= len(calendar):
        raise ValueError('NEXT_SESSION_MISSING')
    if calendar[calendar.index(c['P']) + 1] != c['E']:
        raise ValueError('E_NOT_NEXT_TRADING_DAY')
    if not isinstance(c['code'], str) or len(c['code']) != 6 or not c['code'].isdigit():
        raise ValueError('CODE_FORMAT')
    lo, hi = c['watch_lower'], c['watch_upper']
    if any(type(x) not in (float, int) or not math.isfinite(x) or x <= 0 for x in (lo, hi)) or lo > hi:
        raise ValueError('INVALID_WATCH_REGION')
    for key in ('index_symbol', 'sector_id', 'rule_version'):
        if not isinstance(c[key], str) or not c[key].strip():
            raise ValueError('CONTEXT_IDENTITY_REQUIRED')
    if not digest_valid(c['leader_list_hash']) or not c['source_hashes'] or not all(
            digest_valid(h) for h in c['source_hashes']):
        raise ValueError('SOURCE_REFERENCE_REQUIRED')
    created = timestamp(c['created_at'])
    completed = timestamp(c['daily_bar_completed_at'])
    available = timestamp(c['daily_data_available_at'])
    known = timestamp(c['candidate_known_at'])
    if completed != day(c['P']).replace(hour=15) or not completed <= available <= created:
        raise ValueError('DAILY_COMPLETION_ARRIVAL_ORDER')
    if not known <= created < sessions(c['E'])[0][0]:
        raise ValueError('PLAN_NOT_KNOWN_BEFORE_NEXT_OPEN')
    c['calendar_content_hash'] = fingerprint(calendar)
    return dict(content=c, plan_hash=fingerprint(c))


def checked_plan(plan):
    if fingerprint(plan['content']) != plan['plan_hash']:
        raise ValueError('PLAN_INTEGRITY_MISMATCH')
    return plan['content']


def observation_status(fact, now, triggered_at, date, max_age):
    if not isinstance(fact, dict):
        return 'UNKNOWN', 'MISSING_COMPONENT'
    try:
        start, end, arrived = (timestamp(fact[k]) for k in ('bar_start', 'bar_end', 'available_at'))
        status = fact['status']
        if status not in STATUSES or not digest_valid(fact['evidence_hash']):
            return 'UNKNOWN', 'INVALID_STATUS_OR_REFERENCE'
    except (ValueError, KeyError, TypeError):
        return 'UNKNOWN', 'INVALID_COMPONENT_METADATA'
    if end - start != timedelta(minutes=15) or start.minute % 15 or start.second or start.microsecond:
        return 'UNKNOWN', 'NOT_COMPLETE_15MIN_INTERVAL'
    if not any(a <= start < end <= b for a, b in sessions(date)):
        return 'UNKNOWN', 'BAR_OUTSIDE_SESSION'
    if not end <= arrived <= now:
        return 'UNKNOWN', 'BAR_NOT_AVAILABLE_AT_DECISION'
    if end < triggered_at:
        return 'UNKNOWN', 'BAR_PRECEDES_TRIGGER'
    if (now - end).total_seconds() > max_age:
        return 'UNKNOWN', 'STALE_BAR'
    return status, 'SOURCE_UNKNOWN' if status == 'UNKNOWN' else 'OBSERVED'


def evaluate(plan, *, decision_at, max_quote_age_seconds, max_bar_age_seconds,
             quote=None, observations=None, previous=None):
    """Only technical conditions are evaluated; all results have action=NONE.

    PASS/BLOCK facts must be produced by separately validated feature rules/adapters.
    Evidence hashes establish references, NOT financial validity or source completeness.
    """
    c = checked_plan(plan)
    now = timestamp(decision_at)
    for cap in (max_quote_age_seconds, max_bar_age_seconds):
        if type(cap) is not int or cap <= 0:
            raise ValueError('AGE_POLICY_MUST_BE_EXPLICIT_POSITIVE_SECONDS')
    prior = None
    if previous is not None:
        prior = previous['content']
        if fingerprint(prior) != previous['state_hash']:
            raise ValueError('STATE_INTEGRITY_MISMATCH')
        if prior['plan_hash'] != plan['plan_hash']:
            raise ValueError('PLAN_VERSION_CONFLICT')
        if timestamp(prior['evaluated_at']) > now:
            raise ValueError('DECISION_CLOCK_REWIND')
    trigger_arrival = prior.get('trigger_available_at') if prior else None
    trigger_observed = prior.get('trigger_observed_at') if prior else None

    def result(state, reasons, component_states=None):
        content = dict(plan_hash=plan['plan_hash'], state=state, evaluated_at=decision_at,
            trigger_available_at=trigger_arrival, trigger_observed_at=trigger_observed,
            components=component_states or {}, reason_codes=reasons,
            action='NONE', human_confirmation_required=True,
            source_acceptance_certified=False, financial_validity_certified=False)
        return dict(content=content, state_hash=fingerprint(content))

    if now < timestamp(c['created_at']):
        raise ValueError('PLAN_NOT_AVAILABLE')
    if prior and prior['state'] in ('INVALIDATED', 'EXPIRED'):
        return result(prior['state'], ['TERMINAL_PLAN_NOT_REACTIVATED'])
    if now > sessions(c['E'])[-1][1]:
        return result('EXPIRED', ['PLAN_VALID_SESSION_ENDED'])
    if not in_session(now, c['E']):
        return result('OUT_OF_SESSION', ['NO_REVIEW_OR_ACTION_OUTSIDE_SESSION'])
    if trigger_arrival is None:
        if quote is None:
            return result('WAIT_TRIGGER', ['NO_OBSERVED_TOUCH'])
        try:
            observed, arrived = timestamp(quote['observed_at']), timestamp(quote['available_at'])
            price = quote['price']
            valid_quote = (quote['code'] == c['code'] and digest_valid(quote['evidence_hash'])
                and type(price) in (int, float) and math.isfinite(price) and price > 0
                and observed <= arrived <= now and in_session(observed, c['E'])
                and (now - observed).total_seconds() <= max_quote_age_seconds)
        except (KeyError, ValueError, TypeError):
            valid_quote = False
        if not valid_quote:
            return result('DATA_INSUFFICIENT', ['QUOTE_IDENTITY_TIME_OR_REFERENCE_INVALID'])
        if not c['watch_lower'] <= price <= c['watch_upper']:
            return result('WAIT_TRIGGER', ['OUTSIDE_PLANNED_OBSERVATION_REGION'])
        trigger_arrival, trigger_observed = quote['available_at'], quote['observed_at']
    if observations is None:
        return result('DATA_INSUFFICIENT', ['TOUCH_IS_NOT_MINUTE_CONFIRMATION'])
    if any(observations.get(k) != c[k] for k in ('code', 'index_symbol', 'sector_id', 'leader_list_hash')):
        return result('DATA_INSUFFICIENT', ['CONTEXT_IDENTITY_MISMATCH'])
    component_states, reasons, ends = {}, [], []
    for name in COMPONENTS:
        fact = observations.get(name)
        status, reason = observation_status(fact, now, timestamp(trigger_arrival), c['E'], max_bar_age_seconds)
        component_states[name] = status
        reasons.append(name + ':' + reason)
        if reason in ('OBSERVED', 'SOURCE_UNKNOWN'):
            ends.append(timestamp(fact['bar_end']))
    if len(set(ends)) > 1:
        return result('DATA_INSUFFICIENT', reasons + ['MIXED_BAR_CUTOFFS'], component_states)
    if 'BLOCK' in component_states.values():
        return result('INVALIDATED', reasons + ['FROZEN_INVALIDATION_CONDITION_OBSERVED'], component_states)
    if 'UNKNOWN' in component_states.values():
        return result('DATA_INSUFFICIENT', reasons, component_states)
    if 'WAIT' in component_states.values():
        return result('WATCH_WAIT', reasons, component_states)
    return result('CONDITIONS_MET', reasons + ['NOT_A_TRADE_OR_FILL_PERMISSION'], component_states)
