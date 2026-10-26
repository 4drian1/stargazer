"""Build a real HISTORICAL_SOURCE_BINDING_V1 from the live local stockdb (roadmap Blocker 4 prep).

For each in-scope code:
  - fetch RAW daily OHLC + 复权 from stockdb (read-only),
  - build the corporate-action factor series (cum forward-filled),
  - write prices parquet (code,date,open,high,low,close) and factors parquet (code,date,factor),
  - write per-code evidence with REAL verdicts derived from verify_corporate_action_v1
    (ACCEPTED only when the factor normalization actually removes ex-date gaps),
  - reference the verification report as basis_refs.
Then assemble source_binding.json using the DICT-REF schema required by
historical_stream_v1.bound_rows ({path, sha256} refs; absolute paths so cross-dir
files in PREP resolve).

Verdicts come from ACTUAL verification; nothing self-asserts ACCEPTED. Codes that fail
verification are reported and excluded from the binding (honest, not forced in).
"""
import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from stockdb_source_v1 import (read_daily_raw, read_adjustment, build_factor_series,
                               sha256_file, provenance)
from verify_corporate_action_v1 import verify_code, FIXED_CASES

HERE = Path(__file__).resolve().parent
PREP = HERE / 'historical_stock_day_preparation_v1'
VERIFY_REPORT = PREP / 'corporate_action_verification_v1.json'

PRICES_SCHEMA = pa.schema([('code', pa.string()), ('date', pa.int64()),
                          ('open', pa.float64()), ('high', pa.float64()),
                          ('low', pa.float64()), ('close', pa.float64())])
FACTOR_SCHEMA = pa.schema([('code', pa.string()), ('date', pa.int64()),
                           ('factor', pa.float64())])


def _ref(path):
    # Absolute path: disclosure/verify-report live in PREP (parent of out_dir), so
    # relative-to-binding-dir refs would not resolve. bound_file honors absolute paths.
    return dict(path=str(Path(path).resolve()), sha256=sha256_file(path))


def build_for_codes(codes, out_dir):
    out_dir = Path(out_dir).resolve()
    if not out_dir.is_relative_to(HERE) or out_dir==HERE:
        raise ValueError('OUTPUT_NOT_IN_REGISTERED_RISK_ROOT')
    if out_dir.exists(): raise ValueError('OUTPUT_EXISTS_NO_OVERWRITE')
    out_dir.mkdir(parents=True, exist_ok=False)
    verify_sha = sha256_file(VERIFY_REPORT) if VERIFY_REPORT.exists() else None
    accepted, all_dates, excluded = [], set(), []
    for code in codes:
        bars, dropped = read_daily_raw(code)
        if not bars:
            excluded.append(dict(code=code, reason='no_daily_data'))
            continue
        adj = read_adjustment(code)
        dates = [b['date'] for b in bars]
        factor = build_factor_series(adj, dates)
        all_dates.update(dates)
        # Same snapshot for diagnostics and data; no second live fetch for verification.
        verd = verify_code(code,bars=bars,adjustments=adj)
        ok = False  # internal continuity cannot certify RAW or missing corporate actions
        ppath = out_dir / f'prices_{code}.parquet'
        fpath = out_dir / f'factors_{code}.parquet'
        epath = out_dir / f'evidence_{code}.json'
        import io
        from prepare_historical_risk_v1 import bytes_used
        pbuf=io.BytesIO();fbuf=io.BytesIO()
        pq.write_table(pa.Table.from_pylist(
            [dict(code=code, date=d, open=b['open'], high=b['high'],
                  low=b['low'], close=b['close']) for b, d in zip(bars, dates)],
            schema=PRICES_SCHEMA), pbuf)
        pq.write_table(pa.Table.from_pylist(
            [dict(code=code, date=d, factor=factor[d]) for d in dates],
            schema=FACTOR_SCHEMA), fbuf)
        added=len(pbuf.getvalue())+len(fbuf.getvalue())
        used=sum(p.stat().st_size for p in out_dir.rglob('*') if p.is_file())
        from research_budget_v2 import TOTAL_CAP
        if used+added+65536>2*1024**2 or bytes_used()+added+65536>TOTAL_CAP:
            raise ValueError('SOURCE_BINDING_BUDGET_STOP')
        with ppath.open('xb') as f: f.write(pbuf.getvalue())
        with fpath.open('xb') as f: f.write(fbuf.getvalue())
        from_d, through_d = min(dates), max(dates)
        evidence = dict(
            code=code, scale='RAW',
            price_sha256=sha256_file(ppath),
            factor_sha256=sha256_file(fpath),
            calendar_sha256=None,   # back-filled after calendar written
            **{'from': from_d, 'through': through_d},
            scale_verdict='UNVERIFIED',
            corporate_action_coverage_verdict='UNVERIFIED',
            factor_method='PRODUCT_OF_MULT_AT_RECORDED_DATE',
            effective_dates_verified=False,
            basis_refs=[],
            n_adjustments=len(adj), dropped_bars=dropped,
            verify_verdict=verd['verdict'],
            note='Recorded dates only; continuity diagnostic cannot admit labels or prove coverage.')
        epath.write_text(json.dumps(evidence, ensure_ascii=False, indent=1), encoding='utf-8')
        if not ok:
            excluded.append(dict(code=code, reason='SOURCE_EVIDENCE_PENDING',
                                 verdict=verd['verdict']))
            continue
        accepted.append((code, ppath, fpath, epath))
    # calendar = union of all bar dates, sorted, no weekend
    cal = sorted(d for d in all_dates
                if datetime.strptime(str(d), '%Y%m%d').weekday() < 5)
    cpath = out_dir / 'calendar.json'
    cpath.write_text(json.dumps(cal, ensure_ascii=False), encoding='utf-8')
    cal_sha = sha256_file(cpath)
    # Back-fill calendar_sha256, THEN build refs (final evidence sha after rewrite).
    sources = []
    for code, ppath, fpath, epath in accepted:
        ev = json.loads(Path(epath).read_text(encoding='utf-8'))
        ev['calendar_sha256'] = cal_sha
        Path(epath).write_text(json.dumps(ev, ensure_ascii=False, indent=1), encoding='utf-8')
        sources.append(dict(code=code, prices=_ref(ppath), factors=_ref(fpath),
                            evidence=_ref(epath)))
    disclosure = PREP / 'SECURITY_COVERAGE_DISCLOSURE.md'
    binding = dict(
        schema='HISTORICAL_SOURCE_BINDING_V1',
        generated_at=datetime.now(timezone.utc).isoformat(timespec='seconds'),
        provenance=provenance(),
        calendar=_ref(cpath),
        security_scope_disclosure=_ref(disclosure),
        sources=sources,
        excluded=excluded,
        note='Diagnostic snapshot only; no ACCEPTED source is emitted from internal continuity. '
             'Sources remain pending until actual RAW/effective-date/coverage evidence is supplied.')
    bpath = out_dir / 'source_binding.json'
    bpath.write_text(json.dumps(binding, ensure_ascii=False, indent=1), encoding='utf-8')
    print(f'binding -> {bpath.name}: {len(sources)} accepted, {len(excluded)} excluded')
    for s in sources:
        print('  accepted', s['code'])
    for e in excluded:
        print('  excluded', e['code'], e['reason'])
    return bpath


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--codes', nargs='*', default=FIXED_CASES,
                    help='in-scope codes to bind (default: fixed engineering cases)')
    ap.add_argument('--out', default=str(PREP / 'real_binding_v1'))
    args = ap.parse_args()
    build_for_codes(args.codes, args.out)
