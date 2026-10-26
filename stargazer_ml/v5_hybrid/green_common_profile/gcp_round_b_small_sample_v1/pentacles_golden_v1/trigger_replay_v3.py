"""trigger_replay_v3.py — 触发回放 V3（用户 2026-10-19 缺槽语义收口）。

相对 V2 的增量：区分"正常午休延续"与"数据缺失"：
- 5分bar仍只在恰5根齐全时生成（缺槽不拼接）。
- 数据缺失组（非午休边界 11:30→13:05）→ **切段**：RSI 在新段重新预热；
  跨段不保留背离状态（激活中即记 VOID/DATA_GAP）；锚点搜索范围、
  止跌/跌不去比较基准均不跨段（缺失区间不能默认未发生更低价格）。
- 午休不切段（状态保留，RSI 连算）。
背离事件根不要求收阳；post_trigger 从下一根起；跨小时聚合用真实时间运算（V2 已修）。
"""
from __future__ import annotations

import json
import sys
import time
import urllib.request
import urllib.parse
import zipfile
from datetime import datetime, timedelta
from pathlib import Path

import msgpack

HERE = Path(__file__).resolve().parent
W_BAND = 0.2
RSI_N = 14
OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))
LUNCH_EDGE = (113000, 130500)


def five_stops():
    out = []
    t = datetime(2000, 1, 1, 9, 35)
    while t <= datetime(2000, 1, 1, 11, 30):
        out.append(t.hour * 10000 + t.minute * 100)
        t += timedelta(minutes=5)
    t = datetime(2000, 1, 1, 13, 5)
    while t <= datetime(2000, 1, 1, 15, 0):
        out.append(t.hour * 10000 + t.minute * 100)
        t += timedelta(minutes=5)
    return out


FIVE_STOPS = five_stops()


def sdb(t, timeout=20, retries=3):
    url = 'http://127.0.0.1:7899/?cmd=get&t=' + urllib.parse.quote(t)
    req = urllib.request.Request(url, headers={'Accept': 'application/x-msgpack'})
    for i in range(retries):
        try:
            d = msgpack.unpackb(OPENER.open(req, timeout=timeout).read(),
                                raw=False, strict_map_key=False)
            return d.get('data', d) if isinstance(d, dict) else d
        except Exception:
            if i == retries - 1:
                raise
            time.sleep(1.5)


def to_bars(mrows):
    bars = {}
    for r in (mrows or []):
        d = r[1] if isinstance(r, (list, tuple)) and len(r) == 2 else r
        if not isinstance(d, dict):
            continue
        try:
            ts = int(d['date'])
            o, h, l, c = (float(d['open']), float(d['high']),
                          float(d['low']), float(d['close']))
        except (KeyError, TypeError, ValueError):
            continue
        if min(o, h, l, c) <= 0 or not (l <= min(o, c) <= max(o, c) <= h):
            continue
        if ts not in bars:
            bars[ts] = dict(date=ts, open=o, high=h, low=l, close=c)
    return bars


def hhmmss_to_label(complete_hhmmss):
    t = datetime.strptime(f'{complete_hhmmss:06d}', '%H%M%S') - timedelta(minutes=5)
    return [(t + timedelta(minutes=i)).hour * 10000
            + (t + timedelta(minutes=i)).minute * 100 for i in range(5)]


def aggregate_5min(bars, day):
    """返回 (five, seg_ids, n_data_gaps)。

    seg_ids 与 five 等长：数据缺失组（非午休边界）使后续 bar 段号+1；
    午休不增段。n_data_gaps=非午休缺失组数。
    """
    base = day * 1000000
    out, seg, n_gaps = [], [], 0
    cur_seg = 0
    prev_stop = None
    for stop in FIVE_STOPS:
        comp = [bars.get(base + lab) for lab in hhmmss_to_label(stop)]
        if any(c is None for c in comp):
            if prev_stop is not None and (prev_stop, stop) != LUNCH_EDGE:
                cur_seg += 1
                n_gaps += 1
            prev_stop = stop
            continue
        out.append(dict(t=stop, open=comp[0]['open'], close=comp[-1]['close'],
                        high=max(c['high'] for c in comp),
                        low=min(c['low'] for c in comp)))
        seg.append(cur_seg)
        prev_stop = stop
    return out, seg, n_gaps


def rsi14_by_segment(closes, seg_ids):
    """分段 Wilder RSI(14)：每段独立预热；跨段不连算（午休同段则连算）。"""
    out = [None] * len(closes)
    n = len(closes)
    i = 0
    while i < n:
        j = i
        while j + 1 < n and seg_ids[j + 1] == seg_ids[i]:
            j += 1
        seg_closes = closes[i:j + 1]
        m = len(seg_closes)
        if m > RSI_N:
            gains = [max(seg_closes[k] - seg_closes[k - 1], 0.0)
                     for k in range(1, m)]
            losses = [max(seg_closes[k - 1] - seg_closes[k], 0.0)
                      for k in range(1, m)]
            ag = sum(gains[:RSI_N]) / RSI_N
            al = sum(losses[:RSI_N]) / RSI_N
            out[i + RSI_N] = 100.0 if al == 0 else 100 - 100 / (1 + ag / al)
            for k in range(RSI_N + 1, m):
                ag = (ag * (RSI_N - 1) + gains[k - 1]) / RSI_N
                al = (al * (RSI_N - 1) + losses[k - 1]) / RSI_N
                out[i + k] = 100.0 if al == 0 else 100 - 100 / (1 + ag / al)
        i = j + 1
    return out


def classify_arrival(opening, day_high, f):
    if opening < f:
        return 'REBOUND_TOUCH' if day_high >= f else 'NEVER_ABOVE_F'
    return 'OPEN_AT_OR_ABOVE_F'


def replay(five, seg_ids, f, atr):
    """触发回放核心（纯函数，V3 分段语义）。

    段内逻辑同 V2：基准排除当前根；背离事件=严格low新低+RSI高于锚；
    失效=LOWER_LOW。段边界（数据缺失）即状态边界：跨段无激活背离、
    锚与比较基准不跨段。若段边界时存在激活事件，记 VOID/DATA_GAP。
    """
    lows = [b['low'] for b in five]
    closes = [b['close'] for b in five]
    opens = [b['open'] for b in five]
    rsis = rsi14_by_segment(closes, seg_ids)
    band = f + W_BAND * atr
    n = len(five)
    flags = dict(approach=False, pierced=False, stop_seen=False,
                 divergence_seen=False, rsi_ready=False,
                 div_formed_any=False, div_voided_any=False, data_gaps=0)
    t_a = next((i for i in range(n) if lows[i] <= band), None)
    if t_a is None:
        return dict(trigger=None, reason='NO_APPROACH', path=None, flags=flags)
    flags['approach'] = True
    t_p = next((i for i in range(n) if lows[i] <= f), None)
    flags['pierced'] = t_p is not None
    if any(r is not None for r in rsis):
        flags['rsi_ready'] = True
    seg_start = [0] * n
    for i in range(1, n):
        seg_start[i] = i if seg_ids[i] != seg_ids[i - 1] else seg_start[i - 1]
    flags['data_gaps'] = len({s for s in seg_ids}) - 1
    div = None
    div_hist = []
    stop_idx = []
    for t in range(t_a, n):
        ss = max(t_a, seg_start[t])
        # 数据缺槽切段：跨段时激活中的背离作废（缺失区间不能默认未发生更低价格）
        if t > 0 and seg_ids[t] != seg_ids[t - 1] and div is not None:
            div_hist.append(dict(kind='VOID', bar=t, t=five[t]['t'],
                                 reason='DATA_GAP',
                                 event_low=div['event_low']))
            div = None
            flags['div_voided_any'] = True
        if t > ss and rsis[t] is not None:
            i1 = min(range(ss, t), key=lambda i: lows[i])
            if rsis[i1] is not None:
                if div is not None and lows[t] < div['event_low']:
                    div_hist.append(dict(kind='VOID', bar=t, t=five[t]['t'],
                                         reason='LOWER_LOW',
                                         event_low=div['event_low'],
                                         new_low=round(lows[t], 4)))
                    div = None
                    flags['div_voided_any'] = True
                if div is None and lows[t] < lows[i1] and rsis[t] > rsis[i1]:
                    div = dict(anchor_bar=i1, anchor_t=five[i1]['t'],
                               anchor_low=round(lows[i1], 4),
                               anchor_rsi=round(rsis[i1], 2),
                               event_bar=t, event_t=five[t]['t'],
                               event_low=round(lows[t], 4),
                               event_rsi=round(rsis[t], 2))
                    flags['divergence_seen'] = True
                    flags['div_formed_any'] = True
                    div_hist.append(dict(kind='FORM', bar=t, t=five[t]['t'],
                                         anchor_bar=i1,
                                         event_low=div['event_low'],
                                         event_rsi=div['event_rsi']))
        prior_seg = min(lows[ss:t]) if t > ss else None
        if t_p is not None and t > t_p and t_p >= ss:
            base = min(lows[t_p:t])     # 排除当前根且不跨段
            stop_ok = lows[t] >= base and closes[t] >= f
            path = 'STOP_AFTER_PIERCE'
        elif t_p is None or t <= t_p or t_p < ss:
            if prior_seg is None or prior_seg > band:
                continue
            base = prior_seg
            stop_ok = (lows[t] >= base and closes[t] > opens[t]
                       and closes[t] >= closes[t - 1])
            path = 'HOLD_ABOVE'
        else:
            continue
        if not stop_ok:
            continue
        flags['stop_seen'] = True
        stop_idx.append(t)
        if div is not None and div['event_bar'] <= t:
            rec = dict(trigger=t, reason='TRIGGER', path=path, flags=flags,
                       trigger_close=closes[t], trigger_t=five[t]['t'],
                       divergence=dict(div))
            rec['void_history'] = div_hist
            return rec
    reason = 'NO_STOP'
    if flags['stop_seen']:
        stopped_with_rsi = any(rsis[i] is not None for i in stop_idx)
        if not flags['div_formed_any']:
            reason = 'NOT_FORMED' if stopped_with_rsi else 'RSI_NOT_READY'
        elif flags['div_voided_any']:
            reason = 'VOIDED'
        else:
            reason = 'NOT_FORMED'
    return dict(trigger=None, reason=reason, path=None, flags=flags,
                void_history=div_hist)


def post_trigger(five, t, atr):
    if t + 1 >= len(five):
        return dict(mfe_atr=None, mae_atr=None, close_chg_atr=None,
                    bars_after=0, missing='NO_BARS_AFTER_TRIGGER')
    seg = five[t + 1:]
    tc = five[t]['close']
    mfe = max(b['high'] for b in seg) - tc
    mae = min(b['low'] for b in seg) - tc
    day_close = five[-1]['close']
    return dict(mfe_atr=round(mfe / atr, 4), mae_atr=round(mae / atr, 4),
                close_chg_atr=round((day_close - tc) / atr, 4),
                bars_after=len(seg))


def main():
    with zipfile.ZipFile(HERE / 'real_position_side_tables_v1v2.zip') as z:
        side = json.loads(z.read('real_position_side_table_v2.json'))
    ok = [r for r in side['rows'] if r['status'] == 'OK']

    results, isolated = [], []
    for r in ok:
        lc = r.get('label_check') or {}
        if not (lc.get('high') and lc.get('low')):
            isolated.append(dict(code=r['code'], E=r['E'],
                                 label_check=lc, status='ISOLATED_SCALE_FIELD'))
            continue
        try:
            bars = to_bars(sdb(f"分钟k:{r['code']}:{r['E']}*"))
        except Exception as exc:
            results.append(dict(code=r['code'], E=r['E'],
                                status='FETCH_FAIL', err=str(exc)[:60]))
            continue
        day = r['E']
        day_bars = [b for t, b in sorted(bars.items()) if t // 1000000 == day]
        if len(day_bars) < 200:
            results.append(dict(code=r['code'], E=r['E'], status='BAR_SHORT'))
            continue
        opening = day_bars[0]['open']
        day_high = max(b['high'] for b in day_bars)
        f = r['F']
        atr = r['atr_ref']
        cls = classify_arrival(opening, day_high, f)
        rec = dict(code=r['code'], E=r['E'], segment=r['segment'],
                   setup=r['setup'], arrival_class=cls, F=f, atr_ref=atr)
        if cls != 'OPEN_AT_OR_ABOVE_F':
            rec['status'] = 'SKIPPED_NON_DOWNSIDE_PATH'
            results.append(rec)
            continue
        five, seg_ids, n_gaps = aggregate_5min(bars, day)
        rec['n_five'] = len(five)
        rec['n_data_gaps'] = n_gaps
        rep = replay(five, seg_ids, f, atr)
        rec.update(reason=rep['reason'], path=rep['path'], flags=rep['flags'])
        if rep['trigger'] is not None:
            rec['status'] = 'TRIGGERED'
            rec['trigger_t'] = rep['trigger_t']
            rec['trigger_close'] = rep['trigger_close']
            rec['divergence'] = rep['divergence']
            rec['void_history'] = rep.get('void_history', [])
            rec.update(post_trigger(five, rep['trigger'], atr))
        else:
            rec['status'] = 'NOT_TRIGGERED'
            if rep.get('void_history'):
                rec['void_history'] = rep['void_history']
        results.append(rec)
        time.sleep(0.08)

    out = dict(schema='PENTACLES_TRIGGER_REPLAY_V3',
               params=dict(w_band=W_BAND, rsi_n=RSI_N,
                           confirm='completed_5min_bar_close',
                           no_time_cutoff=True, no_lunch_stitch=True,
                           data_gap_splits_segment=True,
                           lunch_keeps_state=True,
                           divergence_event_requires_positive_close=False,
                           post_trigger_from='NEXT_BAR'),
               divergence_spec_status='首版研究定义（用户认可方向，待最终冻结）',
               n_ok=len(ok), n_isolated=len(isolated),
               isolated=isolated, results=results)
    p = HERE / 'trigger_replay_v3.json'
    p.write_text(json.dumps(out, ensure_ascii=False), encoding='utf-8')
    from collections import Counter
    print('isolated:', len(isolated))
    print('arrival:', dict(Counter(x.get('arrival_class', x['status'])
                                   for x in results)))
    trig = [x for x in results if x.get('status') == 'TRIGGERED']
    print('triggered:', len(trig))
    print('reasons:', dict(Counter(x.get('reason') for x in results)))
    print('data_gaps分布:', dict(Counter(x.get('n_data_gaps', 0)
                                         for x in results)))
    print('path:', dict(Counter(x.get('path') for x in trig)))
    for t in trig:
        d = t['divergence']
        print('TRIG', t['code'], t['E'], '触发t=', t['trigger_t'],
              '事件t=', d['event_t'], '锚t=', d['anchor_t'],
              'chg=', t['close_chg_atr'])
    print(f'bytes={p.stat().st_size}')


if __name__ == '__main__':
    main()