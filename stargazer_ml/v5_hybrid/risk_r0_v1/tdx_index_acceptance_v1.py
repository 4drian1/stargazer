"""Offline native-5m CSV/binary checks and ASSUMED_END price-only 15m diagnostics."""
from collections import Counter
from copy import deepcopy
import csv
from datetime import datetime, timedelta
import hashlib
import json
import math
from pathlib import Path
import struct

from minute_risk_features_v1 import encode, parse_ts
from price_verification_guarded_run import sha
from research_budget_v2 import bytes_used, TOTAL_CAP

HERE = Path(__file__).resolve().parent
OUT = HERE / 'tdx_index_acceptance_20261016_v1'
DATES = [20250102, 20250103, 20250106]
FIELDS = ('open', 'high', 'low', 'close', 'volume', 'amount')
CAP = 2*1024**2


def read_binary(body):
    if len(body) % 32:
        raise ValueError('BINARY_RECORD_SIZE')
    rows = []
    for packed, minute, o, h, l, c, amount, volume, reserved in struct.iter_unpack('<HHfffffiI', body):
        day = (packed//2048+2004)*10000 + (packed%2048//100)*100 + packed%2048%100
        if not 0 <= minute < 1440:
            raise ValueError('BINARY_CLOCK')
        ts = day*1000000 + (minute//60)*10000 + minute%60*100
        parse_ts(ts)
        rows.append(dict(date=ts, open=o, high=h, low=l, close=c, volume=volume, amount=amount))
    return rows


def read_csv(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != ['datetime', *FIELDS]:
            raise ValueError('CSV_SCHEMA')
        return [dict(date=encode(datetime.strptime(r['datetime'], '%Y-%m-%d %H:%M:%S')),
                     **{k: float(r[k]) for k in FIELDS}) for r in reader]


def grid(day, step):
    d = datetime.strptime(str(day), '%Y%m%d')
    return [encode(d.replace(hour=hour)+timedelta(minutes=start+i))
            for hour,start in ((9,30),(13,0)) for i in range(step,121,step)]


def aggregate_15(rows, dates):
    """END-labeled hypothesis only. Not a physical-interval/arrival certification."""
    if dates != sorted(set(dates)):
        raise ValueError('DATES_NOT_SORTED_UNIQUE')
    indexed = {}
    for r in rows:
        parse_ts(r['date'])
        if r['date']//1000000 not in dates or r['date'] not in grid(r['date']//1000000,5):
            raise ValueError('SOURCE_GRID_INVALID')
        if r['date'] in indexed:
            raise ValueError('DUPLICATE_TIMESTAMP')
        if any(type(r.get(k)) not in (int,float) or not math.isfinite(r[k]) for k in FIELDS):
            raise ValueError('NONFINITE')
        if min(r[k] for k in FIELDS[:4]) <= 0 or not r['low'] <= min(r['open'],r['close']) <= max(r['open'],r['close']) <= r['high']:
            raise ValueError('OHLC_INVALID')
        if r['volume'] < 0 or r['amount'] < 0:
            raise ValueError('NEGATIVE_VOLUME_AMOUNT')
        indexed[r['date']] = r
    bars = []
    for day in dates:
        for end in grid(day,15):
            dt = parse_ts(end)
            needed = [encode(dt-timedelta(minutes=i)) for i in (10,5,0)]
            present = all(k in indexed for k in needed)
            group = [indexed[k] for k in needed] if present else []
            bars.append(dict(day=day, bar_end=end, bar_start=encode(dt-timedelta(minutes=15)),
                slot=len(bars)%16, status='COMPLETE_UNDER_END_ASSUMPTION' if present else 'INCOMPLETE',
                missing_source_bars=[k for k in needed if k not in indexed],
                open=group[0]['open'] if present else None, close=group[-1]['close'] if present else None,
                high=max(r['high'] for r in group) if present else None,
                low=min(r['low'] for r in group) if present else None,
                volume=None, amount=None, time_semantics='ASSUMED_END_NOT_CERTIFIED',
                available_at=None, live_feature_admitted=False))
    return bars


def price_features(bars):
    previous, changes, gain, loss = None, [], None, None
    current, daily_open, day_high, day_low, intact = None, None, None, None, False
    result = []
    for b in bars:
        complete = b['status'] == 'COMPLETE_UNDER_END_ASSUMPTION'
        if b['day'] != current:
            current = b['day']
            daily_open, day_high, day_low = b['open'], None, None
            intact = complete and b['slot'] == 0
        intact &= complete
        rsi = change15 = day_return = drawdown = recovery = None
        if complete:
            c = b['close']
            day_high = max(day_high,b['high']) if day_high is not None else b['high']
            day_low = min(day_low,b['low']) if day_low is not None else b['low']
            if intact:
                day_return, drawdown, recovery = c/daily_open-1, c/day_high-1, c/day_low-1
            if previous is not None:
                diff = c - previous['close']
                if gain is None:
                    changes.append(diff)
                    if len(changes) == 14:
                        gain = sum(max(v,0) for v in changes)/14
                        loss = sum(max(-v,0) for v in changes)/14
                else:
                    gain, loss = (13*gain+max(diff,0))/14, (13*loss+max(-diff,0))/14
                if gain is not None:
                    rsi = 50. if gain == loss == 0 else 100. if loss == 0 else 100.-100./(1.+gain/loss)
                if previous['bar_end'] == b['bar_start']:
                    change15 = c/previous['close']-1
            previous = b
        else:
            previous, changes, gain, loss = None, [], None, None
        result.append(dict(bar_end=b['bar_end'], rsi14=rsi, return15m=change15, day_return=day_return,
            drawdown_from_day_high=drawdown, recovery_from_day_low=recovery,
            volume_feature=None, available_at=None, admitted=False))
    return result


def run():
    if OUT.exists():
        raise ValueError('OUTPUT_EXISTS_NO_RESTART')
    if bytes_used()+CAP > TOTAL_CAP:
        raise ValueError('REGISTERED_BUDGET')
    refs = {}

    def track(path):
        path = Path(path).resolve()
        refs[str(path)] = sha(path)
        return path

    for name in ('tdx_index_acceptance_v1.py','test_tdx_index_acceptance_v1.py','minute_risk_features_v1.py','research_budget_v2.py'):
        track(HERE/name)
    specs = [('000001.SH','user_sse_index_export','sse_composite_000001.SH',Path('D:/a股数据/沪指/sh999999.lc5')),
             ('399006.SZ','user_cyb_index_export','cyb_composite_399006',Path('D:/a股数据/创业板指/sz399006.lc5'))]
    indexes = []
    for symbol, folder, stem, source in specs:
        basename = stem+'_5min_20250102_20250106'
        csv_rows = read_csv(track(HERE/folder/(basename+'.csv')))
        meta = json.loads(track(HERE/folder/(basename+'.meta.json')).read_bytes())
        raw = read_binary(track(source).read_bytes())
        selected = [r for r in raw if r['date']//1000000 in DATES]
        if csv_rows != selected:
            raise ValueError('CSV_BINARY_NOT_EXACT')
        if [r['date'] for r in selected] != [k for day in DATES for k in grid(day,5)]:
            raise ValueError('TARGET_GRID_NOT_EXACT')
        bars = aggregate_15(csv_rows,DATES)
        fields = price_features(bars)
        checks = []
        for i,b in enumerate(bars):
            cutoff = b['bar_end']
            prefix = [r for r in csv_rows if r['date'] <= cutoff]
            pb = aggregate_15(prefix,DATES)
            if pb[:i+1] != bars[:i+1] or price_features(pb)[:i+1] != fields[:i+1]:
                raise ValueError('PREFIX_MISMATCH')
            changed = deepcopy(csv_rows)
            count = 0
            for r in changed:
                if r['date'] > cutoff:
                    count += 1
                    for k in FIELDS[:4]: r[k] *= 2
                    r['volume'] *= 3; r['amount'] *= 3
            fb = aggregate_15(changed,DATES)
            if fb[:i+1] != bars[:i+1] or price_features(fb)[:i+1] != fields[:i+1]:
                raise ValueError('PERTURBATION_MISMATCH')
            checks.append(dict(cutoff=cutoff,prefix_equal=True,mutated_rows=count,
                               future_test='PASS' if count else 'NOT_TESTED_NO_FUTURE'))
        ratios = [r['amount']/r['volume'] for r in csv_rows if r['volume'] > 0]
        indexes.append(dict(symbol=symbol, rows=144, source_rows=len(raw), meta_as_supplied=meta,
            csv_binary_exact=True, per_day_counts=dict(Counter(r['date']//1000000 for r in selected)),
            amount_over_raw_volume=dict(min=min(ratios),max=max(ratios)),
            volume_unit_status='UNVERIFIED_DO_NOT_USE', time_status='ASSUMED_END',
            rsi_nonnull=sum(f['rsi14'] is not None for f in fields), checks=checks,
            bars_and_price_fields=[dict(bar=b,fields=f) for b,f in zip(bars,fields)]))
    # Same timestamp identity only; not a certified same physical interval join.
    old_cases = []
    for name in ('minute_warmup_probe_20261016_v1/diagnostics.json',
                 'local_minute_snapshot_11_analysis_20261016_v1/analysis.json'):
        old_cases.extend(json.loads(track(HERE/name).read_bytes())['cases'])
    index_fields = {r['fields']['bar_end']:r['fields'] for r in indexes[0]['bars_and_price_fields']}
    alignment = []
    for c in old_cases:
        if not c.get('scenarios'): continue
        for s in c['scenarios']:
            for record in s['e_day']:
                ts = record['bar']['bar_end']
                alignment.append(dict(code=c['code'], E=c['E'], bar_end=ts,
                    stock_clock=s['assumption'], index_clock='ASSUMED_END',
                    stock_bar_status=record['bar']['status'], index_timestamp_present=ts in index_fields,
                    index_rsi_present=index_fields.get(ts,{}).get('rsi14') is not None,
                    physical_interval_match_verified=False, admitted=False))
    # Local .day comparison diagnoses units; it is not an independent vendor proof.
    day_path = track(Path('D:/a股数据/创业板指/sz399006.day'))
    daily = {r[0]:r for r in struct.iter_unpack('<IIIIIfII',day_path.read_bytes())}
    cyb = read_csv(HERE/'user_cyb_index_export/cyb_composite_399006_5min_20250102_20250106.csv')
    day_csv = track(HERE/'user_cyb_index_export/cyb_composite_399006_daily_20250102_20250106.csv')
    track(day_csv.with_suffix('.meta.json'))
    with day_csv.open(encoding='utf-8-sig',newline='') as stream:
        exported_daily = list(csv.DictReader(stream))
    comparisons = []
    for d in DATES:
        group = [r for r in cyb if r['date']//1000000 == d]
        reference = daily[d]
        export = next(r for r in exported_daily if int(r['datetime'].replace('-','')) == d)
        expected = dict(zip(FIELDS, [reference[1]/100,reference[2]/100,reference[3]/100,
                                     reference[4]/100,reference[6],reference[5]]))
        if any(not math.isclose(float(export[k]), expected[k], rel_tol=0, abs_tol=1e-10) for k in FIELDS):
            raise ValueError('DAILY_CSV_BINARY_MISMATCH')
        comparisons.append(dict(day=d, raw_5m_volume_sum=sum(r['volume'] for r in group),
            raw_day_volume=reference[6], volume_ratio=sum(r['volume'] for r in group)/reference[6],
            amount_ratio=sum(r['amount'] for r in group)/reference[5],
            close_difference=group[-1]['close']-reference[4]/100))
    report = dict(indexes=indexes, stock_alignment=alignment, cyb_day_comparison=comparisons,
        external_eastmoney_comparison_reproduced=False,
        external_comparison_reason='NO_RAW_EXTERNAL_REFERENCE_BOUND',
        old_sse_day_path_exists=Path('D:/a股数据/sh/lday/sh000001.day').exists(),
        time_verified=False, volume_verified=False, full_two_stage_ready=False, fit_ready=False,
        csv_and_sources_modified=False, fetch=False, training=False)
    for path,digest in refs.items():
        if sha(Path(path)) != digest: raise ValueError('INPUT_DRIFT')
    encode_json = lambda obj:(json.dumps(obj,ensure_ascii=False,allow_nan=False,indent=2)+'\n').encode('utf8')
    body = encode_json(report)
    manifest = encode_json(dict(inputs=[dict(path=p,sha256=h) for p,h in sorted(refs.items())],
        outputs=[dict(path='acceptance.json',bytes=len(body),sha256=hashlib.sha256(body).hexdigest())]))
    if len(body)+len(manifest)>CAP or bytes_used()+len(body)+len(manifest)>TOTAL_CAP:
        raise ValueError('WRITE_BUDGET')
    OUT.mkdir(exist_ok=False)
    for name,data in (('acceptance.json',body),('manifest.json',manifest)):
        with (OUT/name).open('xb') as stream: stream.write(data)
    print(json.dumps(dict(output=str(OUT),bytes=len(body)+len(manifest),index_bars_15m=96,
        stock_alignment_rows=len(alignment),fit_ready=False)))


if __name__ == '__main__': run()
