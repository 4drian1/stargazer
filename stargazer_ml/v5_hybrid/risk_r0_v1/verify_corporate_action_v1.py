"""Corporate-action / price-caliber verification (roadmap Blocker 1 verification).

For fixed engineering cases, diagnoses the mult-rebuilt series at recorded dates:
  - raw_close drops on ex-dividend dates; the NORMALIZED (raw*cum) close must be
    continuous there (adjusted return ≈ normal daily move, not the dividend drop).
  - `pre_close` is never used as the raw previous close (the reader returns only OHLC
    and the prior close is derived from the series).
Internal continuity does not certify RAW, effective dates or event completeness.
"""
import json
from pathlib import Path

from stockdb_source_v1 import read_daily_raw, read_adjustment

# Fixed engineering cases: liquid blue chips with long, dividend-rich histories.
FIXED_CASES = ['600519', '000001', '600036', '000858', '300750',
               '601318', '600276', '002594', '600900', '688981']

RAW_DROP_THRESHOLD = 0.03     # a >3% single-day raw drop is a candidate corporate action
ADJ_CONTINUITY_MAX = 0.12     # after normalization, that drop must shrink below this


def verify_code(code, bars=None, adjustments=None):
    from stockdb_adjustment_v2 import continuity_diagnostic
    if bars is None:
        bars, dropped = read_daily_raw(code)
    else:
        dropped=0
    adj = read_adjustment(code) if adjustments is None else adjustments
    if not bars:
        return dict(code=code, verdict='NO_DATA', dates=0, dropped=dropped)
    result=continuity_diagnostic(bars,adj,ADJ_CONTINUITY_MAX)
    return dict(result,code=code,verdict='DIAGNOSTIC_ONLY',dates=len(bars),dropped=dropped,
                n_adjustments=len(adj),n_corp_actions_checked=len(result['checks']),
                note='Internal continuity is not a RAW/effective-date/coverage admission verdict')


def main():
    results = [verify_code(c) for c in FIXED_CASES]
    overall = 'DIAGNOSTIC_ONLY'
    total_unremoved = sum(sum(not c['inside_diagnostic_band'] for c in r.get('checks',[]))
                         for r in results)
    report = dict(generated_at=__import__('datetime').datetime.now(
                    __import__('datetime').timezone.utc).isoformat(timespec='seconds'),
                  scope='FIXED_ENGINEERING_CASES', cases=FIXED_CASES,
                  raw_drop_threshold=RAW_DROP_THRESHOLD,
                  adj_continuity_max=ADJ_CONTINUITY_MAX,
                  overall=overall, outside_diagnostic_band=total_unremoved,
                  per_code=results)
    stamp=__import__('datetime').datetime.now(__import__('datetime').timezone.utc).strftime('%Y%m%dT%H%M%S%f')
    out = Path('historical_stock_day_preparation_v1') / f'corporate_action_diagnostic_v2_{stamp}.json'
    out.parent.mkdir(exist_ok=True)
    data=json.dumps(report, ensure_ascii=False, indent=1).encode('utf-8')
    from prepare_historical_risk_v1 import bytes_used
    if bytes_used()+len(data)>128*1024**2: raise ValueError('REGISTERED_TOTAL_BUDGET')
    with out.open('xb') as f: f.write(data)
    print(f'overall={overall} unremoved_corp_actions={total_unremoved}')
    for r in results:
        print(f"  {r['code']} {r['verdict']} dates={r.get('dates')} "
              f"adj={r.get('n_adjustments')} checked={r.get('n_corp_actions_checked')} "
              f"unremoved={r.get('n_unremoved')}")
    print(f'wrote {out.name}')
    return True  # successful diagnostic execution, not label/coverage acceptance


if __name__ == '__main__':
    import sys
    sys.exit(0 if main() else 1)
