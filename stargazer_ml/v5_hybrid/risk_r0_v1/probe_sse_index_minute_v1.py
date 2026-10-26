"""Read-only probe: does local stockdb hold Shanghai Composite Index (上证综指) minute data?

No external network, no payment, no writes beyond stdout. We check:
  - universe for index-like codes (999999, 000300, 399001...)
  - daily + minute for candidate SSE Composite codes (999999, and 000001 as index)
  - the minute table content/columns
Target dates: 2025-01-02 / 01-03 / 01-06.
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'step9_preflight'))
from data_source_provenance import get_daily_http, get_minute_http, get_universe  # noqa


def summary_df(df, label):
    if df is None or getattr(df, 'empty', True):
        print(f'  {label}: EMPTY / None')
        return
    n = len(df)
    cols = list(df.columns)
    dates = None
    if 'date' in cols:
        d = df['date']
        try:
            dates = (int(d.min()), int(d.max()))
        except Exception:
            dates = '?'
    print(f'  {label}: rows={n} cols={cols}')
    if dates is not None:
        print(f'           date_range={dates}')


def probe_code(code, date='20250102'):
    print(f'[code {code}]')
    d = get_daily_http(code)
    summary_df(d, 'daily')
    m = get_minute_http(code, date)
    summary_df(m, f'minute@{date}')
    if m is not None and not getattr(m, 'empty', True):
        print('    minute head:', m.head(3).to_dict(orient='records'))


def main():
    print('=== universe index scan ===')
    try:
        uni = get_universe()
        idx_like = [c for c in uni if str(c).startswith(('999', '000300', '399', '000688'))
                    or str(c) in ('999999', '000001', '000300', '399001', '399006')]
        print('  universe size:', len(uni))
        print('  index-like codes sample:', sorted(set(str(c) for c in uni
              if str(c).startswith(('999', '0003', '399'))))[:40])
    except Exception as e:
        print('  universe ERR', type(e).__name__, str(e)[:120])

    for code in ['999999', '000001']:
        try:
            probe_code(code)
        except Exception as e:
            print(f'  probe {code} ERR', type(e).__name__, str(e)[:160])
        print()

    print('=== minute on target dates for 999999 ===')
    for dt in ['20250102', '20250103', '20250106']:
        try:
            m = get_minute_http('999999', dt)
            summary_df(m, f'minute@{dt}')
            if m is not None and not getattr(m, 'empty', True):
                print('    sample row:', m.iloc[0].to_dict())
        except Exception as e:
            print(f'  {dt} ERR', type(e).__name__, str(e)[:160])


if __name__ == '__main__':
    main()
