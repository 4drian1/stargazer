"""Persist index card levels from the post-close selection and verified intraday breaks."""
from datetime import datetime
import json
import logging
import threading
import time

from index_outlook_v1 import nearby_levels

_lock = threading.Lock()


def _file():
    from postclose_risk_runtime import directory
    return directory() / 'risk_outlook' / 'overview_index_maps.json'


def read():
    try:
        value = json.loads(_file().read_bytes())
        if value.get('version') == 1 and isinstance(value.get('maps'), dict) and 'requested_day' in value:
            return value
    except (OSError, ValueError, AttributeError):
        pass
    return dict(version=1, requested_day=None, maps={})


def _save(value):
    from pentacles_monitor import atomic_json
    atomic_json(_file(), value)


def request_postclose(day):
    day = int(day)
    datetime.strptime(str(day), '%Y%m%d')
    with _lock:
        state = read()
        if day > (state['requested_day'] or 0):
            state['requested_day'] = day
            _save(state)


def _below_two(row, line):
    times = row.get('last_two_times') or []
    closes = row.get('last_two_closes') or []
    if len(times) != 2 or len(closes) != 2 or times[-1] != row['asof']:
        return False
    a, b = [datetime.strptime(str(t), '%Y%m%d%H%M%S') for t in times]
    return a.date() == b.date() and (b - a).total_seconds() == 300 and all(c < line for c in closes)


def reconcile(now, names):
    """Worker-only I/O. Readiness and cutoffs belong to each index."""
    from index_minutes_v1 import latest_complete_cutoff
    from plan_evidence import archived_price_overlay

    today = int(now.strftime('%Y%m%d'))
    state = read()
    requested = state['requested_day']
    maps = state['maps']
    pending = [name for name in names if requested and maps.get(name, {}).get('selection_day', 0) < requested]
    postclose = requested * 1000000 + 150000 if pending and requested <= today else None
    available = latest_complete_cutoff(now, for_overview=True)
    current = available if available and available // 1000000 == today else None
    work = {}
    if postclose and available and available >= postclose:
        work[postclose] = pending
    if current:
        live = [name for name in names if name in maps and maps[name]['asof'] < current
                and ((maps[name]['nearby'] or {}).get('following') is not None or maps[name].get('risk_support'))]
        if live:
            work[current] = list(dict.fromkeys(work.get(current, []) + live))
    for cutoff, members in work.items():
        evidence = archived_price_overlay(cutoff, overview=True)
        with _lock:
            state = read()
            changed = False
            for name in members:
                row = evidence.get(name) or {}
                if row.get('status') != 'PRICE_FACTS_ONLY':
                    continue
                actual = row['asof']
                old = state['maps'].get(name) or {}
                if actual < old.get('asof', 0):
                    continue
                nearby = nearby_levels(name, row.get('short_wave_levels') or [], row.get('last_close'))
                postclose_publish = (cutoff == postclose and name in pending
                                     and old.get('selection_day', 0) < requested and state['requested_day'] == requested)
                if postclose_publish:
                    if actual != cutoff or not any(nearby.values()):
                        continue
                    selection_day = requested
                else:
                    following = (old.get('nearby') or {}).get('following')
                    if (cutoff != current or actual <= old.get('asof', 0) or following is None
                            or nearby['lower'] is None or actual // 1000000 != today
                            or not _below_two(row, following * .98)):
                        continue
                    selection_day = old['selection_day']
                state['maps'][name] = dict(asof=actual, selection_day=selection_day,
                    close=row.get('last_close'), session_low=row.get('session_low'),
                    nearby=nearby, source=row.get('source_verification'),
                    risk_support=(row.get('critical_position') or {}).get('lower'),
                    risk_active=False if postclose_publish else old.get('risk_active', False),
                    risk_day=None if postclose_publish else old.get('risk_day'),
                    reason='POSTCLOSE' if postclose_publish else 'INTRADAY_BREAK')
                changed = True
            if cutoff == current:
                for name in members:
                    saved = state['maps'].get(name) or {}
                    row = evidence.get(name) or {}
                    support, reference = saved.get('risk_support'), row.get('critical_position', {}).get('reference_close')
                    if (support and reference and row.get('status') == 'PRICE_FACTS_ONLY' and
                            row['asof'] // 1000000 == today):
                        active = _below_two(row, support) and row['last_close'] / reference - 1 < -.02
                        if active != saved.get('risk_active', False) or (active and saved.get('risk_day') != today):
                            saved['risk_active'] = active
                            saved['risk_day'] = today
                            changed = True
            if changed:
                _save(state)


def start(names):
    def work():
        while True:
            try:
                if not read()['requested_day']:
                    from pentacles_demo import _analysis_status
                    status = _analysis_status()
                    if status.get('status') == 'SUCCEEDED' and status.get('data_date'):
                        request_postclose(status['data_date'])
                reconcile(datetime.now(), names)
            except (OSError, ValueError):
                logging.getLogger(__name__).exception('Index overview maps could not be refreshed')
            time.sleep(30)
    threading.Thread(target=work, daemon=True, name='overview-index-maps').start()
