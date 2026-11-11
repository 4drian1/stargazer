"""Read-only prefix inputs for DEV-REC-2; no training or historical-table lookup."""
from functools import lru_cache
from importlib.util import module_from_spec, spec_from_file_location
import hashlib
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pyarrow.dataset as ds

HERE = Path(__file__).resolve().parent
GREEN = HERE.parent / 'green_common_profile'
sys.path.insert(0,str(HERE.parents[2]/'stargazer_desktop'))
MASTER_SOURCE = GREEN / 'round_a_v0_1_impl_fix/full_master_table_v2.py'
TRIGGER_SOURCE = GREEN / 'gcp_round_b_small_sample_v1/pentacles_golden_v1/trigger_replay_v3.py'
DAILY_DIR = Path('D:/a股数据/data_daily_parts')
DAILY_COLUMNS = ['code', 'date', 'open', 'high', 'low', 'close', 'volume',
                 'pre_close', 'is_st', 'float_mv', 'amount']


class ComputeUnavailable(ValueError):
    pass


@lru_cache(maxsize=None)
def source_module(path):
    path = Path(path)
    spec = spec_from_file_location('shadow_' + path.stem, path)
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def calendar_dates():
    from tdx_daily_source_v2 import calendar_days
    return calendar_days()


def read_daily_prefix(code, p):
    from tdx_daily_source_v2 import read_prefix
    return read_prefix(code,p,DAILY_COLUMNS+['name'])


def daily_prefix(frame, code, p, require_metadata=True):
    if not set(DAILY_COLUMNS).issubset(frame.columns):
        raise ComputeUnavailable('DAILY_SCHEMA_INCOMPLETE')
    g = frame.copy()
    g['code'] = g['code'].astype(str).str.zfill(6)
    g = g[(g.code == code) & (g.date <= p)].sort_values('date')
    for _, duplicate in g[g.duplicated('date', keep=False)].groupby('date'):
        if len(duplicate[DAILY_COLUMNS].drop_duplicates()) != 1:
            raise ComputeUnavailable('DAILY_DUP_CONFLICT')
    g = g.drop_duplicates('date').reset_index(drop=True)
    if not len(g) or int(g.date.iloc[-1]) != p:
        raise ComputeUnavailable('P_DAILY_MISSING')
    prices = g[['open', 'high', 'low', 'close']].to_numpy(dtype=float)
    if (not np.isfinite(prices).all() or (prices <= 0).any()
            or (g.low > g[['open', 'close']].min(axis=1)).any()
            or (g.high < g[['open', 'close']].max(axis=1)).any()):
        raise ComputeUnavailable('DAILY_PRICE_INVALID')
    metadata = g[['volume', 'amount']].to_numpy(dtype=float)
    final = g[['pre_close','is_st','float_mv']].iloc[-1].to_numpy(dtype=float)
    if require_metadata and (not np.isfinite(metadata).all() or not np.isfinite(final).all()):
        raise ComputeUnavailable('DAILY_METADATA_UNKNOWN')
    return g


def golden_features(g, p, calendar):
    """Exactly reuse V2 training anchor kernel, rounding and simple-TR ATR recipe."""
    daily = {int(r.date): {k: float(getattr(r, k)) for k in ('open', 'high', 'low', 'close')}
             for r in g.itertuples()}
    anchors = source_module(MASTER_SOURCE).build_anchor_features(daily).set_index('date')
    previous = g.close.shift(1)
    tr = pd.concat([g.high - g.low, (g.high - previous).abs(),
                    (g.low - previous).abs()], axis=1).max(axis=1)
    atr = pd.Series(tr.rolling(14).mean().to_numpy(), index=g.date)
    past = [int(d) for d in calendar if int(d) <= p]
    base = past[-5:]
    if len(base) != 5 or base[-1] != p or any(d not in atr.index for d in base):
        raise ComputeUnavailable('ATR_BASE_INCOMPLETE')
    vals = atr.loc[base].to_numpy(dtype=float)
    if not np.isfinite(vals).all() or (vals <= 0).any():
        raise ComputeUnavailable('ATR_BASE_INCOMPLETE')
    ref = float(np.median(vals))
    row = anchors.loc[p]
    span = float(row.anchor_hi - row.anchor_lo)
    ap = float(atr.loc[p])
    if not np.isfinite(span) or span <= 0 or ap <= 0:
        raise ComputeUnavailable('GOLDEN_ANCHOR_NOT_READY')
    f = daily[p]['high'] - .618 * (daily[p]['high'] - daily[p]['low'])
    level = float(row.anchor_hi) - .618 * span
    return dict(F618=round(f, 4), atr_ref=round(ref, 6),
                plan_dist_atr=round((daily[p]['close'] - f) / -ref, 4),
                golden_dist_618_atr=round((daily[p]['close'] - level) / ap, 4),
                golden_wave_pos=round(float(row.wave_pos), 4))


def selection_row(frame, code, p, calendar):
    g = daily_prefix(frame, code, p)
    if str(GREEN) not in sys.path:
        sys.path.insert(0, str(GREEN))
    from gcp_v0_1_pools import process_code
    a, b = process_code(g)
    for pool, members in [('A_sdp', a), ('B_lvb', b)]:
        if not members.empty:
            selected = members[members.date == p]
            if len(selected):
                row = selected.iloc[0].copy()
                row['setup'] = row['setup_type']
                row['pool'] = pool
                for key, value in golden_features(g, p, calendar).items():
                    row[key] = value
                return row, g
    raise ComputeUnavailable('NOT_COMMON_CANDIDATE')


def read_minute_prefix(code, day, cutoff):
    import csv
    import json
    quarantine = HERE.parents[1] / 'pullback_train/quarantine_2026.csv'
    if quarantine.exists():
        with quarantine.open(encoding='utf-8-sig', newline='') as stream:
            if any(r['code'] == code and int(r['day']) == day for r in csv.DictReader(stream)):
                raise ComputeUnavailable('MINUTE_DAY_QUARANTINED')
    ledger = json.loads((HERE / 'relative_stall_5m_20261018_v1/raw_input_ledger.json').read_bytes())
    item = next((x for x in ledger if x['code'] == code), None)
    if item is None:
        raise ComputeUnavailable('MINUTE_LEDGER_MISSING')
    path = Path(item['path'])
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 ** 2), b''):
            digest.update(block)
    if digest.hexdigest() != item['sha256']:
        raise ComputeUnavailable('MINUTE_SHA256_MISMATCH')
    return ds.dataset(str(path), format='parquet').to_table(
        columns=['code', 'date', 'open', 'high', 'low', 'close'],
        filter=((ds.field('date') >= day * 1000000)
                & (ds.field('date') <= cutoff))).to_pandas()
