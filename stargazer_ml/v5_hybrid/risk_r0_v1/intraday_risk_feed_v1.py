"""Intraday production feed for the full Pyxis lane: realtime current-day stock
minutes + TDX local history + public index mirror. Never falls back to the
legacy 7899 capture (stockdb); LocalCapture itself is untouched so training and
old replays keep their frozen source. (#18.188/W1)
"""
from pathlib import Path
import gzip as _gzip
import hashlib as _hashlib
import json

import msgpack

from live_postclose_source_v1 import LocalCapture, FIELDS, now_stamp

CONTRACT = 'INTRADAY_RISK_FEED_REALTIME_V1'
MIRROR = Path(__file__).resolve().parent / 'index_minutes_v1'


def _calendar_day(token):
    """Strict YYYYMMDD validation; illegal dates are format errors, not empty history."""
    from datetime import datetime
    try:
        return int(datetime.strptime(token, '%Y%m%d').strftime('%Y%m%d'))
    except ValueError:
        raise ValueError('REALTIME_PERIOD_FORMAT:' + token)


def trading_days():
    """Local trading-day calendar derived from the public index mirror."""
    import pandas as pd
    frame = pd.read_parquet(MIRROR / 'sse_5min.parquet')
    return sorted({int(d) for d in (frame['ts'] // 1000000).astype(int).tolist()})


def _native(code, day):
    """Local TDX native minute rows for one finished day, mapped to FIELDS."""
    from tdx_native_history_v1 import native_rows
    rows, receipt = native_rows(code, [day])
    out = [dict(code=code, date=int(r['date'] if 'date' in r else r['day']),
                open=r['open'], high=r['high'], low=r['low'], close=r['close'],
                volume=r.get('volume', 0), amount=r.get('amount', 0)) for r in rows]
    return out, receipt


class RealtimeCapture(LocalCapture):
    """Same capture contract as LocalCapture; fetch() sourced from the public
    realtime minute lane (current day) and local TDX files (history). The
    legacy 7899 request path is never invoked in this class."""

    # Current-day minutes for the scoring universe are supplied per stock by
    # the public realtime lane; the full-board context supply is decided
    # separately (#18.188 supply inventory), so the scoped lane must not be
    # read as a full-universe current-day source.
    scoped_current_day = True

    def _archived(self, code, period, rows):
        """Archive the exact consumed bytes locally (legacy receipt shape), so
        downstream CapturedMinutes can re-read them with hash verification."""
        raw = msgpack.packb([[0, r] for r in rows], use_bin_type=True)
        body = _gzip.compress(raw, mtime=0)
        name = 'raw_%s_%s.msgpack.gz' % (code, str(period).replace('*', 'all'))
        self.output.mkdir(parents=True, exist_ok=True)
        ref = self.save(name, body)
        return dict(period=period, feed=CONTRACT, provenance='TDX_NATIVE_LOCAL',
                    reused=False, path=ref['path'], sha256=ref['sha256'],
                    raw_sha256=_hashlib.sha256(raw).hexdigest(), observed_at=now_stamp())

    def fetch(self, code, period):
        day = self.cache_day
        if day is None:
            raise ValueError('REALTIME_FEED_REQUIRES_CACHE_DAY')
        token = str(period)
        days = trading_days()
        if token == str(day) + '*':
            previous = [d for d in days if d < int(day)]
            if not previous:
                raise ValueError('REALTIME_PREVIOUS_DAY_UNAVAILABLE')
            p_day = previous[-1]
            history, _ = _native(code, p_day)
            if not history:
                raise ValueError('REALTIME_PREVIOUS_DAY_MINUTES_MISSING:' + code)
            previous_close = float(history[-1]['close'])
            from live_minute_source_v1 import snapshot
            rows, receipt, extensions = snapshot(
                self.budget_root, code, p_day, self.required_cutoff,
                previous_close, history, self._budget)
            if not rows:
                raise ValueError('REALTIME_CURRENT_MINUTES_EMPTY:' + code)
            # The live lane archives the raw provider JSON as evidence; downstream
            # CapturedMinutes re-reads STANDARDIZED rows by contract, so archive a
            # standardized copy and keep the raw response linked for audit.
            saved = self._archived(code, period, rows)
            receipt = dict(receipt, path=saved['path'], sha256=saved['sha256'],
                           raw_sha256=saved['raw_sha256'],
                           raw_response_path=receipt.get('path'),
                           raw_response_sha256=receipt.get('sha256'),
                           raw_response_contract='PUBLIC_LANE_RAW_JSON')
            return rows, dict(receipt, period=period, feed=CONTRACT,
                              provenance='PUBLIC_REALTIME_LANE', extensions=extensions)
        if len(token) == 9 and token.endswith('*') and token[:8].isdigit():  # YYYYMMDD*
            target = _calendar_day(token[:8])
            if target >= int(day):
                raise ValueError('REALTIME_PERIOD_NOT_FINISHED:' + token)
            rows, _ = _native(code, target)
            if not rows:
                raise ValueError('REALTIME_HISTORY_DAY_EMPTY:' + code + ':' + token[:8])
            return rows, self._archived(code, period, rows)
        if len(token) == 7 and token.endswith('*') and token[:6].isdigit():  # YYYYMM* month
            from datetime import datetime
            try:
                datetime.strptime(token[:6], '%Y%m')
            except ValueError:
                raise ValueError('REALTIME_PERIOD_FORMAT:' + token)
            month, rows = token[:6], []
            for d in days:
                if str(d).startswith(month) and d < int(day):
                    part, _ = _native(code, d)
                    rows.extend(part)
            if not rows:
                raise ValueError('REALTIME_HISTORY_MONTH_EMPTY:' + code + ':' + month)
            return rows, self._archived(code, period, rows)
        raise ValueError('REALTIME_HISTORY_UNBOUND:' + token)


def _budget(self, size):
    self.disk_bytes += size


RealtimeCapture._budget = _budget
