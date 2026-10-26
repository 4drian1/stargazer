"""CSI300 price-index source binding (roadmap Blocker 3).

The HISTORICAL_STOCK_DAY_V1 contract requires the 5 market fields to come from a
BOUND CSI300 price index (SH000300). No such source is present in data_daily_parts
(000300 / 399300 absent). This module declares the required schema and a loader that
REFUSES to substitute a proxy or zero-fill. Market fields stay NULL until a real
index source is bound via experiment config `market_source.bound_input`.

Forbidden per contract: individual-stock median proxy, zero-fill, pre_close.
"""
import json
import math
import hashlib
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent


class MarketIndexSourceUnbound(Exception):
    """Raised when no CSI300 price-index source is bound; market fields must stay NULL."""


# Accepted CSI300 identities. The pure price index (SH000300) is preferred; when it is
# unavailable in a stockdb instance, the 沪深300ETF (SH510300) is an accepted genuine
# CSI300 tracker. Both must carry adjustment=NONE and a non-empty source. A single-stock
# median proxy remains forbidden.
ACCEPTED_IDENTITIES = [
    {'symbol': 'SH000300', 'semantics': 'CSI300_PRICE_INDEX'},
    {'symbol': 'SH510300', 'semantics': 'CSI300_ETF_TRACKER'},
]

REQUIRED_SCHEMA = {
    'symbol': 'SH000300 or SH510300',
    'semantics': 'CSI300_PRICE_INDEX or CSI300_ETF_TRACKER',
    'fields': ['date', 'close'],          # date(int YYYYMMDD) -> close(float, common scale)
    'forbidden': ['individual_stock_median_proxy', 'zero_fill', 'pre_close'],
}


def load_csi300_closes(bound_input=None, *, allow_etf_tracker=False):
    """Return {date(int): close(float)} or raise MarketIndexSourceUnbound.

    Caller must pass the bound source (path/config). Never fabricates or proxies.
    """
    if bound_input is None:
        raise MarketIndexSourceUnbound(
            'no CSI300 price-index source bound; market fields stay NULL '
            '(contract forbids individual-stock median proxy / zero-fill)')
    p = Path(bound_input)
    if not p.exists():
        raise MarketIndexSourceUnbound(f'bound_input not found: {bound_input}')
    binding = p.with_name(p.name + '.binding.json')
    if not binding.is_file():
        raise MarketIndexSourceUnbound('CSI300 identity/source binding missing')
    meta = json.loads(binding.read_bytes())
    tracker=meta.get('symbol')=='SH510300' and meta.get('semantics')=='CSI300_ETF_TRACKER'
    if tracker and not allow_etf_tracker:
        raise MarketIndexSourceUnbound('ETF tracker requires a separately versioned explicit choice')
    if tracker and (meta.get('adjustment')!='VERIFIED_COMMON_SCALE'
                    or not meta.get('adjustment_basis_refs')):
        raise MarketIndexSourceUnbound('raw ETF distributions are not index risk returns')
    if (not any(meta.get('symbol') == a['symbol'] and meta.get('semantics') == a['semantics']
                for a in ACCEPTED_IDENTITIES)
            or meta.get('adjustment') != ('VERIFIED_COMMON_SCALE' if tracker else 'NONE')
            or not meta.get('source')):
        raise MarketIndexSourceUnbound('CSI300 identity/price semantics mismatch')
    for ref in meta.get('adjustment_basis_refs',[]) if tracker else []:
        evidence=Path(ref['path'])
        if not evidence.is_absolute(): evidence=p.parent/evidence
        if not evidence.is_file() or hashlib.sha256(evidence.read_bytes()).hexdigest()!=ref.get('sha256'):
            raise MarketIndexSourceUnbound('ETF adjustment evidence hash mismatch')
    digest = hashlib.sha256(p.read_bytes()).hexdigest()
    if meta.get('sha256') != digest:
        raise MarketIndexSourceUnbound('CSI300 input hash mismatch')
    if p.suffix.lower() == '.parquet':
        import pyarrow.parquet as pq
        t = pq.read_table(p, columns=['date', 'close'])
        dates = t.column('date').to_pylist()
        closes = t.column('close').to_pylist()
    elif p.suffix.lower() == '.csv':
        import csv
        dates, closes = [], []
        with p.open(encoding='utf-8-sig', newline='') as f:
            for row in csv.DictReader(f):
                dates.append(row['date'])
                closes.append(row['close'])
    else:
        raise MarketIndexSourceUnbound('unsupported index source format')
    out = {}
    for d, c in zip(dates, closes):
        s = str(d)
        try:
            if isinstance(d, bool) or len(s) != 8 or not s.isdigit():
                raise ValueError('date must be YYYYMMDD')
            parsed = datetime.strptime(s, '%Y%m%d')
            if parsed.weekday() >= 5 or isinstance(c, bool):
                raise ValueError('weekend date or boolean price')
            price = float(c)
            if not math.isfinite(price) or price <= 0:
                raise ValueError('invalid price')
        except (TypeError, ValueError, OverflowError) as exc:
            raise MarketIndexSourceUnbound('invalid index row: ' + s) from exc
        day = int(s)
        if day in out and out[day] != price:
            raise MarketIndexSourceUnbound('conflicting index date: ' + s)
        out[day] = price
    if not out:
        raise MarketIndexSourceUnbound('bound source produced no valid close rows')
    if hashlib.sha256(p.read_bytes()).hexdigest() != digest:
        raise MarketIndexSourceUnbound('CSI300 input changed during read')
    return dict(sorted(out.items()))


def bind_from_experiment_config():
    """Read experiment config market_source.bound_input; raise if unbound."""
    base=HERE/'contracts/historical_stock_day_experiment_v1.json'
    cfg = json.loads(base.read_bytes())
    overlay=HERE/'contracts/historical_input_overlay_20261015.json'
    if overlay.exists():
        doc=json.loads(overlay.read_bytes())
        from research_budget_v2 import ADDENDUM
        if (doc.get('schema')!='HISTORICAL_INPUT_OVERLAY_V2'
                or doc.get('base_config_sha256')!=hashlib.sha256(base.read_bytes()).hexdigest()
                or doc.get('budget_addendum_sha256')!=hashlib.sha256(ADDENDUM.read_bytes()).hexdigest()):
            raise MarketIndexSourceUnbound('market overlay/base contract binding mismatch')
        source=doc['market_source'];p=Path(source['bound_input'])
        sidecar=p.with_name(p.name+'.binding.json')
        if (hashlib.sha256(p.read_bytes()).hexdigest()!=source.get('sha256')
                or hashlib.sha256(sidecar.read_bytes()).hexdigest()!=source.get('binding_sha256')):
            raise MarketIndexSourceUnbound('market overlay/input hash mismatch')
        return load_csi300_closes(p)
    return load_csi300_closes(cfg.get('market_source', {}).get('bound_input'))
