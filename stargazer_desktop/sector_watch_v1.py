"""Runtime-only expansion of explicitly authorized boards; never writes a plan."""
import copy
import threading

from golden_rules import digest

LOCK = threading.RLock()
AUTHORIZED = {}


def names(plan):
    return [b['name'] for b in plan.get('boards', []) if isinstance(b, dict) and isinstance(b.get('name'), str)]


def matches(execution, confirmed):
    if digest(execution) == digest(confirmed):
        return True
    parent = execution.get('board_parent_digest')
    with LOCK:
        return parent == digest(confirmed) and AUTHORIZED.get(parent) == digest(execution)


def execution_plan(plan, now, scans=None):
    """Retain original plan rows; add all formal board rows, never review fallbacks.

    The plan's board authorization persists until edited/disabled. Daily board
    inputs must be a completed session, with observation date equal to today.
    """
    if not names(plan):
        return plan
    if scans is None:
        # Board observations now use the independent premarket strong pool.
        # Do not inject its rows into Tarot admission, risk batches or buy signals.
        return plan
    day = int(now.strftime('%Y%m%d'))
    current = int(plan.get('monitor_date', 0)) == day
    rows = {r['code']: copy.deepcopy(r) for r in plan.get('rows', [])} if current else {}
    plan_codes=set(rows)
    origins, inputs, versions, warnings = {}, None, [], []
    pday = int(plan.get('data_date', 0)) if current else None
    for result in scans:
        requested = result.get('board_requested', result.get('board'))
        if requested not in names(plan):
            continue
        if result.get('status') != 'READY' or int(result.get('observation_date', 0)) != day:
            warnings.append(dict(board=requested, status=result.get('status'), reason=result.get('reason')))
            continue
        if pday is not None and int(result['data_date']) != pday:
            warnings.append(dict(board=requested, status='DAILY_SOURCE_DATE_MISMATCH'))
            continue
        pday = int(result['data_date'])
        source_inputs = result.get('monitor_rule_inputs')
        if not source_inputs or (inputs is not None and inputs != source_inputs):
            warnings.append(dict(board=requested, status='RULE_SOURCE_UNBOUND_OR_CHANGED'))
            continue
        inputs = source_inputs
        versions.append([result['board_code'], result['source_key']])
        for row in result.get('monitor_rows', []):
            rule = row.get('entry_rule') or {}
            if rule.get('eligible') is not True or rule.get('policy') != 'GOLDEN_RESONANCE_CROSSDAY_V4' or int(rule.get('p_date', 0)) != pday:
                continue
            code = row['code']
            if code in plan_codes:
                continue  # Already monitored by the plan; no second discovery.
            origins.setdefault(code, []).append(result['board'])
            rows.setdefault(code, copy.deepcopy(row))
    # Preserve the existing risk batch cap, and make incomplete scope visible.
    # Never silently drop stocks and claim all configured boards are monitored.
    if len(rows) > 128:
        result={**plan, 'board_watch_status': 'POPULATION_LIMIT',
                'board_watch_reason': '板块正式候选超过单批128只，未扩大自动处理范围',
                'board_watch_count': 0,'board_parent_digest':digest(plan)}
        with LOCK:
            AUTHORIZED.clear();AUTHORIZED[digest(plan)]=digest(result)
        return result
    if not origins:
        with LOCK:
            AUTHORIZED.pop(digest(plan),None)
        return plan
    result = copy.deepcopy(plan)
    result.update(rows=list(rows.values()), monitor_date=day, data_date=pday,
                  board_parent_digest=digest(plan), board_origins=origins,
                  board_source_versions=sorted(versions), board_watch_warnings=warnings,
                  board_watch_status='OBSERVING', board_watch_count=len(origins))
    if inputs is not None:
        result['rule_inputs'] = inputs
    result['stocks'] = [dict(code=r['code'], entry_rule=r['entry_rule']) for r in result['rows']]
    with LOCK:
        AUTHORIZED.clear()
        AUTHORIZED[digest(plan)] = digest(result)
    return result
