"""Read-only probe: akshare SSE Composite 000001.SH 1-min history + identity cross-check.
No writes, no payment, no training.
"""
import sys
import os
# Bypass the dead Windows system proxy so akshare can reach eastmoney directly.
os.environ.setdefault('NO_PROXY', '*')
os.environ.setdefault('no_proxy', '*')
os.environ.pop('HTTP_PROXY', None)
os.environ.pop('HTTPS_PROXY', None)

import pandas as pd
import akshare as ak

SYM = "000001"
TARGET = ["2025-01-02", "2025-01-03", "2025-01-06"]

print('akshare', ak.__version__)
print('\n=== 1-min fetch (index_zh_a_hist_min_em) ===')
df = ak.index_zh_a_hist_min_em(
    symbol=SYM, period="1",
    start_date="2025-01-02 09:30:00",
    end_date="2025-01-06 15:00:00",
)
print('shape:', df.shape)
print('columns:', list(df.columns))
print('head:\n', df.head(2).to_string())
print('tail:\n', df.tail(2).to_string())

tcol = df.columns[0]
ts = pd.to_datetime(df[tcol])
print('\nfirst ts:', ts.min(), ' last ts:', ts.max())
for d in TARGET:
    sel = ts.dt.strftime('%Y-%m-%d').eq(d)
    print(f'  {d}: bars={int(sel.sum())} '
          f'open(first)={df.loc[sel, "开盘"].iloc[0] if sel.any() else None} '
          f'close(last)={df.loc[sel, "收盘"].iloc[-1] if sel.any() else None}')

# interval check
sub = ts[ts.dt.strftime('%Y-%m-%d').eq('2025-01-02')].sort_values()
diffs = sub.diff().dropna().dt.total_seconds()
print('\n2025-01-02 inter-bar seconds: min=%.0f median=%.0f max=%.0f (expect ~60 for 1-min)'
      % (diffs.min(), diffs.median(), diffs.max()))

print('\n=== identity cross-check: sh000001 daily close on 2025-01-02 ===')
try:
    dly = ak.stock_zh_index_daily(symbol="sh000001")
    dly['date'] = pd.to_datetime(dly[dly.columns[0]]).dt.strftime('%Y-%m-%d')
    row = dly[dly['date'] == '2025-01-02']
    print('daily cols:', list(dly.columns))
    print(row.to_string())
    if len(row):
        dclose = float(row.iloc[0]['close'])
        mclose = df[ts.dt.strftime('%Y-%m-%d').eq('2025-01-02')]['收盘'].iloc[-1]
        print(f'  daily close={dclose}  vs 1-min last close={float(mclose)}  '
              f'-> match={abs(dclose-float(mclose))<1.0}')
except Exception as e:
    print('  daily cross-check ERR', type(e).__name__, str(e)[:160])

print('\n=== sanity: is this an index (~3200-3400), not a stock (~11)? ===')
print('2025-01-02 first-open =', float(df[ts.dt.strftime('%Y-%m-%d').eq('2025-01-02')]['开盘'].iloc[0]))
