"""240 只异常股票逐行审计 v1 —— 只读，不修改任何生产代码。

回答的问题:
  - 每只坏股票究竟哪几行异常、异常字段是什么
  - 是单根K线还是整段数据损坏
  - 是否存在可核实的原始数据

用法:
  python -X utf8 pyxis_bad_stock_audit_v1.py --cutoff 20261016094500
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import numpy as np
import pandas as pd


def _row_reason(row):
    """单行异常的字段级原因。"""
    reasons = []
    for k in ('open', 'high', 'low', 'close'):
        v = row[k]
        if not isinstance(v, (int, float, np.floating, np.integer)):
            reasons.append(f'{k}:NOT_NUMERIC({type(v).__name__})')
        elif not np.isfinite(float(v)):
            reasons.append(f'{k}:NON_FINITE({v})')
        elif float(v) <= 0:
            reasons.append(f'{k}:NON_POSITIVE({v})')
    if reasons:
        return reasons
    o, h, l, c = (float(row['open']), float(row['high']),
                  float(row['low']), float(row['close']))
    if l > min(o, c):
        reasons.append(f'low({l})>min(open,close)')
    if h < max(o, c):
        reasons.append(f'high({h})<max(open,close)')
    return reasons


def audit_one(frame, cutoff):
    """对单只股票做逐行审计(向量化)。"""
    out = frame.loc[frame.date <= cutoff]
    total = int(len(out))
    rec = {'total_rows': total, 'dup_date': int(out.date.duplicated().sum()) if total else 0}

    if total == 0:
        rec['verdict'] = 'EMPTY_AFTER_CUTOFF'
        return rec

    need = ['date', 'open', 'high', 'low', 'close']
    missing = [k for k in need if k not in out.columns]
    if missing:
        rec['verdict'] = 'COLUMNS_MISSING'
        rec['missing_columns'] = missing
        return rec

    cols = ['open', 'high', 'low', 'close']
    num = out[cols].apply(pd.to_numeric, errors='coerce')
    p = num.to_numpy(dtype=float)
    row_bad = ~np.isfinite(p).all(axis=1) | (p <= 0).any(axis=1)
    ok = ~row_bad
    if ok.any():
        o, h, l, c = p[ok].T
        bad_ok = (l > np.minimum(o, c)) | (h < np.maximum(o, c))
        row_bad[np.flatnonzero(ok)[bad_ok]] = True

    bad_idx = np.flatnonzero(row_bad)
    rec['bad_rows'] = int(len(bad_idx))
    rec['bad_ratio'] = round(len(bad_idx) / total, 4) if total else 0
    samples = []
    for i in bad_idx[:5]:
        row = out.iloc[int(i)]
        samples.append({'date': int(row['date']), 'open': float(p[i, 0]),
                        'high': float(p[i, 1]), 'low': float(p[i, 2]),
                        'close': float(p[i, 3]),
                        'reasons': _row_reason({k: row[k] for k in cols})})
    rec['samples'] = samples

    if len(bad_idx):
        dates = out['date'].to_numpy()[bad_idx]
        rec['first_bad_date'] = int(dates[0])
        rec['last_bad_date'] = int(dates[-1])
        rec['distinct_bad_days'] = int(len({int(d) // 1000000 for d in dates}))
        gaps = np.diff(bad_idx)
        rec['contiguous_blocks'] = int(1 + (gaps > 1).sum())
        blocks = np.split(bad_idx, np.where(gaps > 1)[0] + 1) if len(gaps) else [bad_idx]
        rec['max_block_len'] = int(max(len(b) for b in blocks))
    # 结论分级
    if not len(bad_idx):
        rec['verdict'] = 'OK'
    elif rec['dup_date']:
        rec['verdict'] = 'DUPLICATE_DATE'
    elif len(bad_idx) == 1:
        rec['verdict'] = 'SINGLE_BAR'
    elif rec['contiguous_blocks'] == 1:
        rec['verdict'] = 'ONE_CONTIGUOUS_BLOCK'
    else:
        rec['verdict'] = 'MULTIPLE_BAD_SEGMENTS'
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cutoff', type=int, default=20261016094500)
    ap.add_argument('--out', default='_bad_stock_audit.json')
    args = ap.parse_args()

    from joint_shadow_snapshot_v1 import produce
    from joint_shadow_context_v1 import _prefix

    t_all = time.time()
    snapshot, _ = produce(args.cutoff, historical=True)
    research = snapshot.membership_scope == 'SNAPSHOT_RECONSTRUCTION_RESEARCH'
    codes = sorted({c for m in snapshot.boards.values() for c in m
                    if c in snapshot.stock_minutes})
    print(f'[produce] done, universe={len(codes)}', flush=True)

    bad, t0 = {}, time.time()
    cache_path = HERE / '_bad_codes_cache.json'
    if cache_path.exists():
        bad = json.loads(cache_path.read_text(encoding='utf8'))['bad']
        print(f'[precheck] from cache, bad={len(bad)}', flush=True)
    else:
        for code in codes:
            try:
                _prefix(snapshot.stock_minutes[code], args.cutoff,
                        snapshot.receipts.get(code), stock=True, research=research,
                        observed_at=snapshot.observed_at)
            except Exception as exc:
                s = str(exc)
                key = next((k for k in ('INVALID_SOURCE_PRICE', 'INVALID_SOURCE_OHLC',
                                        'SOURCE_RECEIPT_MISSING', 'SOURCE_CLOCK_UNBOUND',
                                        'INVALID_SOURCE_ACTIVITY', 'EMPTY_OR_DUPLICATE_SOURCE',
                                        'SOURCE_COLUMNS_MISSING') if k in s),
                           f'{type(exc).__name__}:{s[:60]}')
                bad[code] = key
        print(f'[precheck] {len(codes)} codes, bad={len(bad)}, {round(time.time()-t0,1)}s', flush=True)

    detail = {}
    for i, (code, err) in enumerate(sorted(bad.items()), 1):
        try:
            frame = snapshot.stock_minutes[code]
            rec = audit_one(frame, args.cutoff)
        except Exception as exc:
            rec = {'verdict': 'AUDIT_FAILED', 'error': f'{type(exc).__name__}:{str(exc)[:100]}'}
        rec['prefix_error'] = err
        detail[code] = rec
        if i % 20 == 0:
            print(f'[audit] {i}/{len(bad)}', flush=True)

    verdicts = {}
    for rec in detail.values():
        verdicts[rec['verdict']] = verdicts.get(rec['verdict'], 0) + 1
    err_types = {}
    for rec in detail.values():
        err_types[rec['prefix_error']] = err_types.get(rec['prefix_error'], 0) + 1

    report = {
        'cutoff': args.cutoff,
        'universe': len(codes), 'bad_count': len(bad),
        'bad_by_prefix_error': err_types,
        'verdict_distribution': verdicts,
        'recoverable_single_bar': [c for c, r in detail.items() if r.get('verdict') == 'SINGLE_BAR'],
        'recoverable_one_block': [c for c, r in detail.items()
                                  if r.get('verdict') == 'ONE_CONTIGUOUS_BLOCK'],
        'severe': [c for c, r in detail.items()
                   if r.get('verdict') in ('MULTIPLE_BAD_SEGMENTS', 'DUPLICATE_DATE',
                                           'EMPTY_AFTER_CUTOFF', 'COLUMNS_MISSING',
                                           'AUDIT_FAILED')],
        'detail': detail,
        'elapsed_s': round(time.time() - t_all, 1),
        'finished_at': datetime.now().isoformat(timespec='seconds'),
    }
    out_path = HERE / args.out
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf8')

    print('\n=== 审计摘要 ===')
    print('异常类型:', err_types)
    print('形态分布:', verdicts)
    print('单根K线(可考虑部分恢复):', len(report['recoverable_single_bar']))
    print('单段连续(可考虑部分恢复):', len(report['recoverable_one_block']))
    print('严重(整段损坏):', len(report['severe']))
    print('报告:', out_path)


if __name__ == '__main__':
    main()
