"""Exhaustive probe: does local stockdb hold the SSE Composite Index (上证综指) at all?

Try multiple index-code conventions; also scan the daily parquet universe for any
non-stock (index) code. Read-only. No external network, no writes, no payment.
"""
import sys
sys.path.insert(0, r'D:\stargazer\stargazer_ml\v5_hybrid\step9_preflight')
from data_source_provenance import _http_raw, _decode, get_daily_parquet


def table_len(table, code, date='*'):
    try:
        obj = _decode(_http_raw('get' if date == '*' else 'vals', table, code, date))
    except Exception as e:
        return -1, 'ERR %s' % type(e).__name__
    if isinstance(obj, list):
        rows = [v for _, v in obj if isinstance(v, dict)] if obj and isinstance(obj[0], (list, tuple)) \
            else [it for it in obj if isinstance(it, dict)]
        return len(rows), (rows[0].get('name') if rows and isinstance(rows[0], dict) else '')
    return 0, ''


def main():
    candidates = ['999999', '1A0001', 'sh000001', '000001.SH', 'sh999999',
                  '000300', '399001', '399006', '000688',
                  'sh000300', 'sz399001', 'sh000001.SH']
    print('=== HTTP probe (table 日k / 分钟k) ===')
    for c in candidates:
        nd, nm = table_len('日k', c, '*')
        nm_code = c.zfill(6)
        # minute uses zfill(6); for 8-char codes zfill does nothing useful, but try raw too
        nm2, _ = table_len('分钟k', c, '20250102*')
        nm3, _ = table_len('分钟k', c, '20250103*')
        print('%-12s daily=%-5s name=%-10s minute0102=%-5s minute0103=%-5s'
              % (c, nd, nm, nm2, nm3))

    print()
    print('=== daily parquet universe scan for non-stock codes ===')
    try:
        p = get_daily_parquet()
        allc = sorted(set(str(x) for x in p['code'].unique()))
        # normal A-share stock ranges
        def is_stock(code):
            if len(code) != 6:
                return False
            return (code[:3] in ('600', '601', '603', '605', '688', '000', '001', '002', '003', '004', '300')
                    or code[:2] in ('83', '87', '43', '92', '93'))  # BJ / NEEQ-ish
        nonstock = [c for c in allc if not is_stock(c)]
        print('total distinct codes:', len(allc))
        print('non-stock-like codes:', nonstock[:80])
        # any containing index-ish tokens
        idxish = [c for c in allc if c.startswith('999') or c.startswith('1A')
                  or c.startswith('399') or c in ('000300',)]
        print('index-ish codes present:', idxish)
    except Exception as e:
        print('parquet scan ERR', type(e).__name__, str(e)[:160])


if __name__ == '__main__':
    main()
