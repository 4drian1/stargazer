"""Pure as-of leader capture; no network, model, scoring weights or trading actions.

Engineering contract: caller supplies a frozen same-taxonomy comparison universe,
membership and provenance. Its factual historical validity is NOT certified here.
Ties and day-boundary policy deliberately remain unresolved rather than invented.
"""
from copy import deepcopy
from datetime import datetime
import math


def valid_ts(value):
    try:
        return len(str(value)) == 14 and datetime.strptime(str(value), '%Y%m%d%H%M%S') is not None
    except (TypeError, ValueError):
        return False


def eligible_quote(quote, cutoff, decision):
    return (quote is not None and quote.get('bar_end') == cutoff
            and valid_ts(quote.get('known_at')) and quote['known_at'] <= decision
            and isinstance(quote.get('day_return'), (int, float))
            and not isinstance(quote['day_return'], bool) and math.isfinite(quote['day_return']))


def advance(previous, decision, cutoff, universe, boards, membership, stocks):
    """Quotes use decimal day_return. membership is {board: {codes, known_at}}.

    universe = {id, taxonomy, boards, known_at, reference_rule}. Its id binds the
    membership version as well as board identities. Output is an observation only.
    """
    if not valid_ts(decision) or not valid_ts(cutoff) or cutoff > decision:
        raise ValueError('INVALID_DECISION_OR_CUTOFF')
    if cutoff // 1000000 != decision // 1000000:
        raise ValueError('DIFFERENT_TRADING_DAY')
    required = universe.get('boards', [])
    if (not required or len(set(required)) != len(required)
            or not universe.get('taxonomy') or not universe.get('id') or not universe.get('reference_rule')
            or not valid_ts(universe.get('known_at')) or universe['known_at'] > decision):
        raise ValueError('UNIVERSE_NOT_AVAILABLE')
    if previous and (previous['universe_id'] != universe['id'] or decision <= previous['decision']):
        raise ValueError('UNIVERSE_DRIFT_OR_NONINCREASING_DECISION')
    state = deepcopy(previous) if previous else {
        'universe_id': universe['id'], 'day': decision // 1000000,
        'captured_boards': {}, 'captured_stocks': {}}
    state.update(decision=decision, cutoff=cutoff, current_board=None, current_stock=None,
                 board_ties=[], stock_ties=[], action='NONE', training_ready=False)
    if state['day'] != decision // 1000000:
        state['status'] = 'DAY_BOUNDARY_POLICY_PENDING'
        return state
    if any(not eligible_quote(boards.get(board), cutoff, decision) for board in required):
        state['status'] = 'UNKNOWN_BOARD_RANK'
        return state
    best = max(boards[board]['day_return'] for board in required)
    winners = sorted(board for board in required if boards[board]['day_return'] == best)
    if len(winners) != 1:
        state.update(status='BOARD_TIE_PENDING', board_ties=winners)
        return state
    board = winners[0]
    state['current_board'] = board
    state['captured_boards'].setdefault(board, decision)
    relation = membership.get(board)
    if (not relation or not valid_ts(relation.get('known_at')) or relation['known_at'] > decision
            or not relation.get('codes') or len(set(relation['codes'])) != len(relation['codes'])):
        state['status'] = 'UNKNOWN_MEMBERSHIP'
        return state
    members = relation['codes']
    if any(not eligible_quote(stocks.get(code), cutoff, decision) for code in members):
        state['status'] = 'UNKNOWN_STOCK_RANK'
        return state
    top = max(stocks[code]['day_return'] for code in members)
    winners = sorted(code for code in members if stocks[code]['day_return'] == top)
    if len(winners) != 1:
        state.update(status='STOCK_TIE_PENDING', stock_ties=winners)
        return state
    stock = winners[0]
    state.update(status='OBSERVED', current_stock=stock)
    state['captured_stocks'].setdefault(board + '|' + stock, decision)
    return state
