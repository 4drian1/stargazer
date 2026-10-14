"""Seven-session fixed-exit observations. Never choose future extrema as a sale."""
from datetime import datetime
import math


def positive(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value > 0


def evaluate(record, calendar, asof):
    """Calendar is exchange-session order; entry day counts as day one.

    Verified fill/scale/exit evidence are explicit, not inferred from price presence.
    Old tracker records can supply observations but never certified executable PnL.
    """
    result = {'status': 'ENTRY_UNKNOWN', 'mfe_pct': None, 'mae_pct': None,
              'day7_reference_return_pct': None, 'simulated_net_return_pct': None,
              'actual_pnl': None, 'rule': 'ENTRY_DAY_COUNTS_1_EXIT_DAY7_CLOSE_V1'}
    ep, ts = record.get('entry_price'), record.get('entry_ts')
    if not positive(ep) or not isinstance(ts, int):
        return result
    entry_day = ts // 1000000
    if calendar != sorted(set(calendar)) or entry_day not in calendar:
        result['status'] = 'CALENDAR_UNKNOWN'
        return result
    pos = calendar.index(entry_day)
    window = calendar[pos:pos + 7]
    observed = [d for d in window if d <= asof]
    bars = record.get('days') or {}
    daily = [bars.get(str(d), bars.get(d)) for d in observed]
    result.update(window=window, observed_sessions=len(observed), entry_price=ep)
    if any(not isinstance(b, dict) or any(not positive(b.get(k)) for k in ('high', 'low', 'close'))
           or not b['low'] <= b['close'] <= b['high'] for b in daily):
        result['status'] = 'DATA_INSUFFICIENT'
        return result
    # Entry day daily extrema may precede entry. Only use explicit post-entry extrema.
    path = list(daily)
    if path:
        first = path.pop(0)
        after = record.get('entry_day_post_entry')
        if isinstance(after, dict) and positive(after.get('high')) and positive(after.get('low')):
            path.insert(0, after)
    if path and record.get('scale_verified') is True:
        result['mfe_pct'] = (max(b['high'] for b in path) / ep - 1) * 100
        result['mae_pct'] = (min(b['low'] for b in path) / ep - 1) * 100
    if len(window) < 7 or len(observed) < 7:
        result['status'] = 'WINDOW_PENDING'
        return result
    result['exit_date'] = window[-1]
    if record.get('scale_verified') is not True:
        result['status'] = 'SCALE_UNVERIFIED'
        return result
    exit_px = daily[-1]['close']
    result['day7_reference_return_pct'] = (exit_px / ep - 1) * 100
    if not (record.get('entry_execution_verified') is True and
            record.get('day7_exit_execution_verified') is True and
            record.get('corporate_action_verified') is True):
        result['status'] = 'REFERENCE_ONLY_EXECUTION_UNVERIFIED'
        return result
    buy_cost, sell_cost = record.get('buy_cost_rate'), record.get('sell_cost_rate')
    if not all(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)
               and 0 <= v < 1 for v in (buy_cost, sell_cost)):
        result['status'] = 'COST_UNKNOWN'
        return result
    result['simulated_net_return_pct'] = (exit_px * (1 - sell_cost) / (ep * (1 + buy_cost)) - 1) * 100
    result['status'] = 'SIMULATION_COMPLETE'
    # Real PnL requires actual matched trades; never copy the simulated return.
    return result
