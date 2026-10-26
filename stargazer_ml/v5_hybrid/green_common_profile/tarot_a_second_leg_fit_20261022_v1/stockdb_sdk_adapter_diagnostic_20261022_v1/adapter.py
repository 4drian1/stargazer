"""Versioned research-only SDK grouping; frozen signal kernel stays unchanged."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'execution_alignment_diagnostic_20261022_v1'))
import diagnose_execution as d

VERSION = 'STOCKDB_SDK_RIGHT_END_ASSUMED_V1'


def completed_prefix(frame, code, day):
    if frame.empty:
        return [], 'DAY_MISSING'
    if frame.date.duplicated().any() or not frame.code.astype(str).str.zfill(6).eq(code).all():
        return [], 'IDENTITY_OR_DUPLICATE_INVALID'
    expected = set(d.source_grid(day))
    present = set((frame.date % 1000000).astype(int))
    if present - expected or any(not d.prior.valid_daily(row) for row in frame.to_dict('records')):
        return [], 'UNEXPECTED_OR_INVALID_SOURCE'
    lookup = {int(row['date']) % 1000000: row for row in frame.to_dict('records')}
    bars = []
    for stop in d.prior.kernel.FIVE_STOPS:
        end = stop // 10000 * 60 + stop % 10000 // 100
        required = [minute // 60 * 10000 + minute % 60 * 100 for minute in range(end - 4, end + 1)]
        if stop == 93500:
            required.insert(0, 93000)
        if stop == 130500 and 130000 in expected:
            required.insert(0, 130000)
        if not all(slot in lookup for slot in required):
            return bars, 'UNKNOWN_FROM_FIRST_INCOMPLETE_GROUP'
        source = [lookup[slot] for slot in required]
        bars.append(dict(t=stop, open=source[0]['open'], high=max(row['high'] for row in source),
                         low=min(row['low'] for row in source), close=source[-1]['close']))
    return bars, 'FULL_48_COMPLETE'


def events_for(record, prefix):
    if not prefix:
        return [], {}
    signal = d.prior.confirmations(prefix, [0] * len(prefix), float(record['node']), float(record['atr']))
    events = []
    for branch, field in [('RSI', 'rsi_index'), ('RECLAIM', 'reclaim_index')]:
        index = signal.get(field)
        if index is None:
            continue
        stamp = int(record['E']) * 1000000 + prefix[index]['t']
        events.append({key: record[key] for key in ['code', 'P', 'E', 'stage', 'node', 'atr', 'split']}
                      | {'branch': branch, 'confirmation_available_ts': stamp, 'last_source_bar_index_ts': stamp,
                         'confirmation_close': prefix[index]['close'], 'prefix_five_at_confirmation': index + 1,
                         'risk_status': 'UNBOUND', 'adapter': VERSION, 'clock_assumption': 'MINUTE_ENDPOINT_NOT_ARRIVAL_CERTIFIED',
                         'speed_status': 'NOT_APPLICABLE' if branch == 'RSI' else 'MAX_RECLAIM_DURATION_UNDEFINED',
                         'policy_status': 'PRICE_EVENT_ONLY_NOT_BUY_ADVICE'})
    return events, signal
