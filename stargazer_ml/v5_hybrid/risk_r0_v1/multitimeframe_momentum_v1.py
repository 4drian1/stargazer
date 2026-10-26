"""Causal research indicators. Proposed initialization is not a trading contract."""
from copy import deepcopy
from collections import deque
import math


def positive(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value > 0


class SeededEMA:
    """SMA seed at period observations; no output before seed."""
    def __init__(self, period):
        if type(period) is not int or period < 1:
            raise ValueError('INVALID_PERIOD')
        self.period, self.seed, self.value = period, [], None

    def update(self, value):
        if value is None:
            self.seed, self.value = [], None
        elif self.value is None:
            self.seed.append(value)
            if len(self.seed) == self.period:
                self.value = sum(self.seed) / self.period
                self.seed = []
        else:
            self.value += 2 / (self.period + 1) * (value - self.value)
        return self.value


class MomentumState:
    def __init__(self, rsi_period, macd_periods, bbi_periods):
        periods = [rsi_period, *macd_periods, *bbi_periods]
        if rsi_period != 14 or any(type(p) is not int or p < 1 for p in periods) or len(macd_periods) != 3 or not bbi_periods:
            raise ValueError('INVALID_PERIODS')
        if macd_periods[0] >= macd_periods[1]:
            raise ValueError('MACD_FAST_MUST_BE_SHORTER')
        self.parameters = (rsi_period, tuple(macd_periods), tuple(bbi_periods))
        self.rsi_period, self.bbi_periods = rsi_period, tuple(bbi_periods)
        self.fast, self.slow, self.signal = [SeededEMA(p) for p in macd_periods]
        self.closes = deque(maxlen=max(bbi_periods))
        self.previous = self.gain = self.loss = self.last_bbi = None
        self.changes = []

    def branch(self, close):
        """Exactly one provisional observation; leave the completed state untouched."""
        return deepcopy(self).update(close)

    def update(self, close):
        if not positive(close):
            self.__init__(*self.parameters)
            return {k: None for k in ('rsi14', 'macd_dif', 'macd_dea', 'macd_hist', 'bbi',
                                      'bbi_slope', 'close_over_bbi_minus_one', 'price_above_bbi')}
        rsi = None
        if self.previous is not None:
            change = close - self.previous
            if self.gain is None:
                self.changes.append(change)
                if len(self.changes) == self.rsi_period:
                    self.gain = sum(max(c, 0) for c in self.changes) / self.rsi_period
                    self.loss = sum(max(-c, 0) for c in self.changes) / self.rsi_period
                    self.changes = []
            else:
                n = self.rsi_period
                self.gain = (self.gain * (n - 1) + max(change, 0)) / n
                self.loss = (self.loss * (n - 1) + max(-change, 0)) / n
            if self.gain is not None:
                rsi = 50. if self.gain == self.loss == 0 else 100. if self.loss == 0 else 100 - 100 / (1 + self.gain / self.loss)
        fast, slow = self.fast.update(close), self.slow.update(close)
        dif = fast - slow if fast is not None and slow is not None else None
        dea = self.signal.update(dif)
        self.closes.append(close)
        bbi = None
        if len(self.closes) == max(self.bbi_periods):
            values = list(self.closes)
            bbi = sum(sum(values[-p:]) / p for p in self.bbi_periods) / len(self.bbi_periods)
        slope = bbi / self.last_bbi - 1 if bbi is not None and self.last_bbi is not None else None
        self.previous, self.last_bbi = close, bbi
        return dict(rsi14=rsi, macd_dif=dif, macd_dea=dea,
                    macd_hist=2 * (dif - dea) if dea is not None else None,
                    bbi=bbi, bbi_slope=slope,
                    close_over_bbi_minus_one=close / bbi - 1 if bbi is not None else None,
                    price_above_bbi=float(close > bbi) if bbi is not None else None)


def confirmed_divergence(closes, oscillator, *, left, right, min_distance, max_distance):
    """Strict close pivots; one-bar event at confirmation, never backdate.

    Explicit proposal parameters only. Adjacent same-type pivots are compared;
    an unsuitable adjacent pivot is NOT skipped to find a better-looking pair.
    Missing prices reset pivot history; missing oscillator leaves the pair unknown.
    Caller provides the complete trading-bar grid (missing bars represented by None).
    """
    if len(closes) != len(oscillator):
        raise ValueError('UNEQUAL_LENGTH')
    if any(type(v) is not int or v < 1 for v in (left, right, min_distance, max_distance)) or min_distance > max_distance:
        raise ValueError('INVALID_PIVOT_PARAMETERS')
    result, previous, run_start = [], {'bottom': None, 'top': None}, 0
    for t, close in enumerate(closes):
        record = dict(bottom=None, top=None, pivot_index=None, previous_pivot_index=None,
                      confirmation_index=t)
        if not positive(close):
            previous, run_start = {'bottom': None, 'top': None}, t + 1
            result.append(record)
            continue
        p = t - right
        if p - left >= run_start:
            window = closes[p - left:t + 1]
            others = [v for j, v in enumerate(window) if j != left]
            kind = ('bottom' if all(closes[p] < v for v in others) else
                    'top' if all(closes[p] > v for v in others) else None)
            if kind:
                prev = previous[kind]
                record['pivot_index'], record['previous_pivot_index'] = p, prev
                if prev is not None and min_distance <= p - prev <= max_distance:
                    a, b = oscillator[prev], oscillator[p]
                    if all(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) for v in (a, b)):
                        condition = (closes[p] < closes[prev] and b > a if kind == 'bottom'
                                     else closes[p] > closes[prev] and b < a)
                        record[kind] = int(condition)
                previous[kind] = p
        result.append(record)
    return result
