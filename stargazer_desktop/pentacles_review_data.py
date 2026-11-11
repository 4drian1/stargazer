"""Read-only local TDX daily evidence for a review. No orders or HTTP."""
from datetime import datetime
from contextlib import closing
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import time

import paths
from seven_day_review import positive, evaluate




def local_day(cfg, codes, day):
    """Exact-day synchronized mirror; conflicts remain unusable, never substitute yesterday."""
    from tdx_daily_source_v2 import day_path,read_day
    values={};receipts=[];missing=[]
    for code in sorted(codes):
        try:frame,receipt=read_day(day_path(code),int(day))
        except (OSError,ValueError) as exc:
            missing.append(dict(code=code,reason=str(exc)));continue
        row=frame[frame.date==int(day)]
        if not row.empty:values[code]=dict(row.iloc[0],code=code)
        receipts.append(receipt)
    return values,dict(source='TDX_LOCAL_RAW_DAILY',inputs=receipts,conflict_codes=[],missing=missing)


def actual_trades(cfg, codes, day):
    """Read the real ledger only. Daily matched gross PnL; fees absent => net NULL."""
    value = (cfg.get('agent') or {}).get('trades_db', 'data/trades.db')
    file = Path(value) if Path(value).is_absolute() else Path(paths.data_file(value, seed_from_bundle=False))
    result = {'status': 'NO_LEDGER', 'realized_gross_pnl': None, 'realized_net_pnl': None,
              'note': '真实成交独立记录，不归因模型；费用未记录，不计算实际净收益'}
    if not file.exists():
        return result
    try:
        with closing(sqlite3.connect(file.resolve().as_uri() + '?mode=ro', uri=True)) as conn:
            rows = conn.execute('SELECT code,trade_date,action,price,shares,id FROM trades ORDER BY trade_date,id LIMIT 10001').fetchall()
        if len(rows) > 10000:
            return dict(result, status='LEDGER_TOO_LARGE')
        inventory, pnl, matched, unknown = {}, 0.0, 0, False
        target = datetime.strptime(str(day), '%Y%m%d').date().isoformat()
        for code, date, action, price, shares, identifier in rows:
            if code not in codes or date > target:
                continue
            if not positive(price) or not isinstance(shares, int) or shares <= 0:
                unknown = True
                continue
            lots = inventory.setdefault(code, [])
            if action == 'buy':
                lots.append([shares, price, date])
            elif action == 'sell':
                remaining, cost = shares, 0.0
                while remaining and lots:
                    lot = lots[0]
                    take = min(remaining, lot[0])
                    if lot[2] >= date:
                        unknown = True
                        break
                    cost += take * lot[1]
                    remaining -= take
                    lot[0] -= take
                    if not lot[0]:
                        lots.pop(0)
                if remaining:
                    unknown = True
                elif date == target:
                    pnl += price * shares - cost
                    matched += 1
        return dict(result, status='MATCH_UNKNOWN' if unknown else 'MATCHED_GROSS_ONLY',
                    realized_gross_pnl=None if unknown or not matched else round(pnl, 2), matched_sales=matched)
    except (sqlite3.Error, ValueError, TypeError):
        return dict(result, status='LEDGER_UNAVAILABLE')


def collect(snapshot, day, root, cfg=None):
    datetime.strptime(str(day), '%Y%m%d')
    now = datetime.now()
    if day > int(now.strftime('%Y%m%d')) or (day == int(now.strftime('%Y%m%d')) and now.hour < 16):
        raise ValueError('请在16:00后复盘已完成交易日，不能使用未完成日K')
    source = json.dumps(snapshot, ensure_ascii=False, sort_keys=True, allow_nan=False).encode()
    facts = {'day': day, 'data_date': snapshot.get('data_date'),
             'observe_date': snapshot.get('monitor_date'), 'snapshot_hash': hashlib.sha256(source).hexdigest(),
             'candidate_count': len(snapshot['rows']), 'quote_touch_is_not_buy': True,
             'market': {'status': 'UNKNOWN'}, 'rows': [], 'seven_day': [],
             'actual_trades': [], 'errors': []}
    if day != snapshot.get('monitor_date'):
        raise ValueError('复盘日期须使用该名单的观察日；历史复盘请读取归档名单')
    try:
        fallback, evidence = local_day(cfg or {}, {r['code'] for r in snapshot['rows']}, day)
    except Exception:
        fallback, evidence = {}, {'status':'LOCAL_MIRROR_UNAVAILABLE'}
    facts['daily_mirror'] = evidence
    deadline = time.monotonic() + 90
    for row in snapshot['rows']:
        item = {'code': row['code'], 'name': row.get('name', ''), 'setup': row.get('setup'),
                'status': 'DATA_INSUFFICIENT', 'daily_touch': None, 'close_change_pct': None}
        try:
            if time.monotonic() >= deadline:
                raise ValueError('本轮行情读取时间预算已达，余下候选保留数据不足')
            data = [fallback[row['code']]] if row['code'] in fallback else []
            item['price_source'] = 'TDX_LOCAL_RAW_DAILY'
            if len(data) != 1:
                raise ValueError('缺失或重复日K')
            b = data[0]
            if not all(positive(b.get(k)) for k in ('open', 'high', 'low', 'close')):
                raise ValueError('日K价格无效')
            if not b['low'] <= min(b['open'], b['close']) <= max(b['open'], b['close']) <= b['high']:
                raise ValueError('OHLC不一致')
            item.update(status='DAILY_OBSERVATION', ohlc={k: b[k] for k in ('open', 'high', 'low', 'close')})
            ref = row.get('reference_price')
            if positive(ref):
                item['close_change_pct'] = (b['close'] / ref - 1) * 100
            zone = row.get('zone')
            if zone and positive(zone.get('zone_upper')):
                item['daily_touch'] = b['low'] <= zone['zone_upper']
                item['touch_note'] = '仅日线相对冻结区域描述；不证明触区时刻/成交/尺度核销'
        except Exception as exc:
            item['reason'] = type(exc).__name__
        facts['rows'].append(item)
    # Existing trackers remain backfilled observations, never become executable labels.
    for file in sorted(Path(root).glob('tracker*.json')):
        tracker = json.loads(file.read_text(encoding='utf-8'))
        for record in tracker.get('rows', []):
            dates = sorted(int(d) for d in (record.get('days') or {}))
            # Ad-hoc tracker flags are not executor evidence. Do not self-certify.
            observation = dict(record, scale_verified=False, entry_execution_verified=False,
                               corporate_action_verified=False, day7_exit_execution_verified=False)
            metrics = evaluate(observation, dates, day)
            facts['seven_day'].append({'code': record['code'], 'source': file.name,
                                      'backfilled': tracker.get('backfilled', True), **metrics})
    tracked_codes = {r['code'] for r in facts['seven_day']}
    for row in snapshot['rows']:
        if row['code'] not in tracked_codes:
            facts['seven_day'].append({'code':row['code'], 'status':'ENTRY_UNKNOWN',
                'mfe_pct':None, 'mae_pct':None, 'day7_reference_return_pct':None,
                'simulated_net_return_pct':None, 'actual_pnl':None,
                'note':'报价/日线触区不是认证入场，等待完整轨道证据'})
    touched = [r['close_change_pct'] for r in facts['rows'] if r['daily_touch'] is True
               and r['close_change_pct'] is not None]
    facts['summary'] = {'valid_daily_rows': sum(r['status'] == 'DAILY_OBSERVATION' for r in facts['rows']),
                        'daily_touch_count': len(touched),
                        'touch_close_mean_pct': sum(touched) / len(touched) if touched else None,
                        'touch_positive_count': sum(x > 0 for x in touched)}
    facts['actual_trades'] = actual_trades(cfg or {}, {r['code'] for r in snapshot['rows']}, day)
    return facts
