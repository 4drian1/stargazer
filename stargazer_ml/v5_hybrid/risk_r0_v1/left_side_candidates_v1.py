"""Endpoint left-side evidence: causal, no future pivot confirmation or trade action."""
import numpy as np
import pandas as pd


def candidates(frame, lags=(1, 2, 3)):
    """One symbol/timeframe on its complete expected grid, including missing slots.

    CLOSE and oscillator endpoints share their bar cutoff. Intervening invalid
    prices block the comparison rather than silently compressing time.
    """
    if not lags or len(set(lags)) != len(lags) or any(type(k) is not int or k < 1 for k in lags):
        raise ValueError('INVALID_LOOKBACKS')
    required = ['close', 'rsi14', 'macd_dif', 'macd_hist', 'bbi']
    values = frame[required].astype(float)
    if np.isinf(values.to_numpy()).any():
        raise ValueError('INFINITE_INPUT')
    price = values.close
    valid_price = price.notna() & (price > 0)
    if ((values.bbi <= 0) & values.bbi.notna()).any():
        raise ValueError('INVALID_BBI')
    if ((values.rsi14 < 0) | (values.rsi14 > 100)).any():
        raise ValueError('INVALID_RSI')
    result = pd.DataFrame(index=frame.index)
    for lag in lags:
        valid = valid_price.rolling(lag + 1, min_periods=lag + 1).sum().eq(lag + 1)
        previous_price = price.shift(lag)
        price_change = price / previous_price - 1
        prefix = f'lag{lag}_'
        result[prefix + 'price_return'] = price_change.where(valid)
        pairs = {}
        for indicator, label in [('rsi14', 'rsi'), ('macd_dif', 'macd')]:
            x, previous = values[indicator], values[indicator].shift(lag)
            usable = valid & x.notna() & previous.notna()
            change = x - previous
            result[prefix + label + '_change' + ('_over_prior_close' if label == 'macd' else '')] = (
                change / previous_price if label == 'macd' else change).where(usable)
            for direction in ['bottom', 'top']:
                match = ((price_change < 0) & (change > 0) if direction == 'bottom'
                         else (price_change > 0) & (change < 0))
                column = prefix + label + '_' + direction + '_candidate'
                result[column] = match.astype(float).where(usable)
                pairs[label, direction] = result[column]
        for direction in ['bottom', 'top']:
            a, b = pairs['rsi', direction], pairs['macd', direction]
            result[prefix + direction + '_confluence_candidate'] = ((a == 1) & (b == 1)).astype(float).where(a.notna() & b.notna())
        hist, bbi = values.macd_hist, values.bbi
        result[prefix + 'macd_hist_change_over_prior_close'] = ((hist - hist.shift(lag)) / previous_price).where(valid & hist.notna() & hist.shift(lag).notna())
        result[prefix + 'bbi_distance_change'] = (price / bbi - price.shift(lag) / bbi.shift(lag)).where(valid & bbi.notna() & bbi.shift(lag).notna())
    return result.astype(float)
