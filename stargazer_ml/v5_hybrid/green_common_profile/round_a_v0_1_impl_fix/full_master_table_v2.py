"""full_master_table_v2.py — 全量母表 V2（用户 2026-10-19 审计修正）。

相对 V1 的修正（V1 产物保留作偏差记录）：
1. 周期锚点状态机按确认规则重写：
   - 上端收缩取**起点阳线收盘价**（V1 误用最高价）；
   - 三日下降中须含**单日**跌幅≥1%（V1 误为三日累计>1%）；
   - 下端收缩：7连阳确认后，锚点取下端=**上涨前一日收盘**（V1 误取区间最低价）。
2. F618 参考日订正：P日收盘制定计划、次日(E=P+1)执行的"前一日"=**P日本身**
   （V1 误用 P−1）；ATR 基准=P−4..P 五日（含P，不含E）。与盘中回放口径一致
   （盘中昨日=E−1=P）。
3. 准入漏洞修复：
   - adj_cache 无该股票（MISSING）≠ 无事件：38只/57条 V1 误放行 → 一律
     ISOLATED(CACHE_MISSING)；EMPTY(在缓存但空链) 与 MISSING 分列；
   - 日线重复键冲突检查（同 code-date 不同值 → 该码全部 REJECT
     DAILY_DUP_CONFLICT；P0 审计称全局0重复，仍保留防线）；
   - 标签窗口**逐日在场**核验（P 与后7个交易日每日都有日线），不再只查首尾。
4. golden_dist 基于修正后锚点与 P 日收盘。
产物：full_master_table_v2.parquet + meta（V1 不覆盖）。
"""
from __future__ import annotations

import glob
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd
import pyarrow.dataset as ds

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / '../../step9_preflight'))
from trading_calendar import load_calendar  # noqa: E402

CACHE = json.loads((HERE / '../../opportunity_v2/gate1c/feb2026_adjustment_cache_v1'
                    '/adj_cache.json').read_bytes())
DAILY_GLOB = 'D:/a股数据/data_daily_parts/part_*.parquet'


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def build_anchor_features(daily):
    """锚点状态机 V2（用户确认规则）：

    - 7日初始化：前7日 high max / low min；
    - 上端收缩：起点阳线(close>open)后连续3日收盘下降，且**其中至少一个单日**
      跌幅≥1% → anchor.hi = min(hi, **起点阳线收盘**)；
    - 下端收缩：连续7日收盘上涨确认 → anchor.lo = max(lo, **上涨前一日收盘**)；
    - 收盘单端突破：close>hi→hi=close；close<lo→lo=close。
    """
    dates = sorted(daily)
    hi = lo = None
    init_n = 7
    rows = []
    for i, d in enumerate(dates):
        o, h, l, c = (daily[d]['open'], daily[d]['high'],
                      daily[d]['low'], daily[d]['close'])
        if i + 1 >= init_n and hi is None:
            win = dates[i + 1 - init_n:i + 1]
            hi = max(daily[x]['high'] for x in win)
            lo = min(daily[x]['low'] for x in win)
        if hi is not None:
            if i >= 3:
                d1, d2, d3 = dates[i - 3], dates[i - 2], dates[i - 1]
                c0, c1, c2, c3 = (daily[d1]['close'], daily[d2]['close'],
                                  daily[d3]['close'], c)
                o0 = daily[d1]['open']
                single_drops = [c0 - c1, c1 - c2, c2 - c3]
                if (c1 < c0 and c2 < c1 and c3 < c2 and c0 > o0
                        and max(single_drops) >= 0.01 * c0):
                    hi = min(hi, c0)          # 起点阳线**收盘价**
            if i >= 7:
                run = dates[i - 6:i + 1]
                if all(daily[run[k]]['close'] > daily[run[k - 1]]['close']
                       for k in range(1, 7)):
                    prev_close = daily[dates[i - 7]]['close']
                    lo = max(lo, prev_close)   # 上涨前一日收盘
            if c > hi:
                hi = c
            if c < lo:
                lo = c
        wave = ((c - lo) / (hi - lo)) if (hi and lo and hi > lo) else None
        rows.append(dict(date=d, anchor_hi=hi, anchor_lo=lo, close=c,
                         high=h, low=l, wave_pos=wave))
    return pd.DataFrame(rows)


def main():
    a = pd.read_parquet(HERE / 'pool_a_sdp_members.parquet')
    b = pd.read_parquet(HERE / 'pool_b_lvb_members.parquet')
    a['pool'] = 'A_sdp'
    b['pool'] = 'B_lvb'
    pools = pd.concat([a, b], ignore_index=True, sort=False)
    pools['code'] = pools['code'].astype(str).str.zfill(6)
    pools = pools.drop_duplicates(['code', 'date'])
    codes = sorted(pools['code'].unique())
    print('candidates:', len(pools), 'codes:', len(codes))

    files = sorted(glob.glob(DAILY_GLOB))
    tbl = ds.dataset(files, format='parquet').to_table(
        columns=['code', 'date', 'open', 'high', 'low', 'close'],
        filter=ds.field('code').isin(codes))
    daily = {}
    dup_conflict = set()
    for c, d, o, h, l, cl in zip(tbl['code'].to_pylist(), tbl['date'].to_pylist(),
                                 tbl['open'].to_pylist(), tbl['high'].to_pylist(),
                                 tbl['low'].to_pylist(), tbl['close'].to_pylist()):
        c = str(c).zfill(6)
        rec = dict(open=float(o), high=float(h), low=float(l), close=float(cl))
        if d in daily.get(c, {}):
            if (daily[c][d]['close'] != rec['close']
                    or daily[c][d]['high'] != rec['high']):
                dup_conflict.add(c)
            continue
        daily.setdefault(c, {})[int(d)] = rec
    print('daily codes:', len(daily), 'dup_conflict codes:', len(dup_conflict))

    cal = load_calendar()
    cal_idx = {d: i for i, d in enumerate(cal)}

    feat_by_code = {}
    for code in codes:
        dmap = daily.get(code)
        if not dmap or len(dmap) < 20 or code in dup_conflict:
            feat_by_code[code] = None
            continue
        dates = sorted(dmap)
        trs, a14 = [], {}
        prev = None
        for d in dates:
            r = dmap[d]
            tr = (r['high'] - r['low']) if prev is None else max(
                r['high'] - r['low'], abs(r['high'] - prev), abs(r['low'] - prev))
            trs.append(tr)
            if len(trs) >= 14:
                a14[d] = sum(trs[-14:]) / 14.0
            prev = r['close']
        anchors = build_anchor_features(dmap)
        anchors['atr14'] = anchors['date'].map(a14)
        lvl = anchors['anchor_hi'] - 0.618 * (anchors['anchor_hi'] - anchors['anchor_lo'])
        anchors['golden_dist_618_atr'] = (anchors['close'] - lvl) / anchors['atr14']
        feat_by_code[code] = anchors.set_index('date')

    # 缓存状态：MISSING(不在缓存) 与 EMPTY(在缓存但无事件) 分列
    cache_status = {}
    for c in codes:
        if c not in CACHE:
            cache_status[c] = 'MISSING'
        elif not (CACHE[c].get('events') or []):
            cache_status[c] = 'EMPTY'
        else:
            cache_status[c] = 'OK'

    rows = []
    for r in pools.itertuples(index=False):
        code, p = r.code, int(r.date)
        rec = dict(code=code, P=p, setup=str(r.setup_type), pool=r.pool,
                   feature_ts=f'{p}150000', adj_cache_status=cache_status.get(code))
        for col in pools.columns:
            if col not in ('code', 'date', 'pool'):
                rec[col] = getattr(r, col)
        dmap = daily.get(code) or {}
        fdf = feat_by_code.get(code)
        status, reasons = 'OK', []
        f618 = atr_ref = plan_dist = None
        chg = y = label_end = None
        if code in dup_conflict:
            status, reasons = 'REJECT', ['DAILY_DUP_CONFLICT']
        elif cache_status.get(code) == 'MISSING':
            status, reasons = 'ISOLATED', ['CACHE_MISSING']
        elif p not in cal_idx:
            status, reasons = 'REJECT', ['P_NOT_IN_CALENDAR']
        else:
            pi = cal_idx[p]
            ref = cal[pi]                     # 修正②：F618 参考日 = P 本身
            base5 = cal[pi - 4:pi + 1]        # P-4..P 恰五值（含P，不含E）
            ref_ohlc = dmap.get(ref)
            if ref_ohlc is None or len(base5) < 5:
                status, reasons = 'REJECT', ['MISSING_DAILY']
            else:
                if fdf is not None and p in fdf.index:
                    row = fdf.loc[p]
                    gdist = row.get('golden_dist_618_atr')
                    wave = row.get('wave_pos')
                else:
                    gdist = wave = None
                bvals = []
                if fdf is not None:
                    bvals = [fdf.loc[d, 'atr14'] for d in base5
                             if d in fdf.index and pd.notna(fdf.loc[d, 'atr14'])]
                if len(bvals) == 5:
                    atr_ref = sorted(bvals)[2]
                if atr_ref and atr_ref > 0:
                    f618 = ref_ohlc['high'] - 0.618 * (ref_ohlc['high'] - ref_ohlc['low'])
                    plan_dist = (f618 - dmap[p]['close']) / atr_ref
                else:
                    reasons.append('ATR_BASE_INCOMPLETE')
                rec.update(F618=round(f618, 4) if f618 else None,
                           atr_ref=round(atr_ref, 6) if atr_ref else None,
                           plan_dist_atr=round(plan_dist, 4) if plan_dist is not None else None,
                           golden_dist_618_atr=(round(float(gdist), 4)
                                                if gdist is not None and pd.notna(gdist) else None),
                           golden_wave_pos=(round(float(wave), 4)
                                            if wave is not None and pd.notna(wave) else None))
                later = [d for d in cal if d > p][:7]
                win_days = [p] + later
                if (len(later) < 7
                        or not all(d in dmap for d in win_days)):   # 逐日在场
                    status = 'REJECT' if status == 'OK' else status
                    reasons.append('LABEL_WINDOW_INCOMPLETE')
                else:
                    label_end = later[-1]
                    chg = (dmap[label_end]['close'] / dmap[p]['close']) - 1.0
                    y = 1 if chg > 0 else 0
                    if cache_status.get(code) == 'OK':
                        ev = [d for d, _ in CACHE[code]['events']
                              if p < int(str(d)[:8].replace('-', '')) <= label_end]
                        if ev:
                            status = 'ISOLATED'
                            reasons.append(f'CORP_ACTION_IN_WINDOW:{len(ev)}')
                    # EMPTY：无事件可检，保留 OK 但证据状态已单列 adj_cache_status
        rec.update(label_end_ts=(f'{label_end}150000' if label_end else None),
                   chg7=round(chg, 6) if chg is not None else None,
                   y7=y, admission=status,
                   isolation_reason=';'.join(reasons) if reasons else None)
        rows.append(rec)

    df = pd.DataFrame(rows)
    p = HERE / 'full_master_table_v2.parquet'
    df.to_parquet(p, index=False)
    meta = dict(schema='PENTACLES_FULL_MASTER_V2',
                fixes='锚点:阳线收盘/单日1%/上涨前一日收盘; F618参考日=P; '
                      'CACHE_MISSING准入; 日线重复键冲突防线; 标签窗逐日在场',
                inputs={f: sha(HERE / f) for f in
                        ('pool_a_sdp_members.parquet', 'pool_b_lvb_members.parquet')},
                cache_sha=sha(HERE / '../../opportunity_v2/gate1c'
                              '/feb2026_adjustment_cache_v1/adj_cache.json'),
                n_rows=int(len(df)),
                admission={str(k): int(v) for k, v in
                           df['admission'].value_counts().items()},
                cache_status={str(k): int(v) for k, v in
                              df['adj_cache_status'].value_counts().items()},
                y7_rate=float(df['y7'].mean()))
    (HERE / 'full_master_table_v2.meta.json').write_text(
        json.dumps(meta, ensure_ascii=False, indent=1), encoding='utf-8')
    print(json.dumps({k: v for k, v in meta.items() if k != 'inputs'},
                     ensure_ascii=False))
    print(f'bytes={p.stat().st_size}')


if __name__ == '__main__':
    main()
