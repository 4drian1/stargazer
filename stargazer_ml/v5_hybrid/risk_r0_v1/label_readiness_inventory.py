"""label_readiness_inventory.py — 44 窗标签工程就绪盘点（零网络；用户 2026-10-13 指示）。

每窗盘四项：
1) P 日参考收盘 + 后五日各自 low：来源（canonical daily parts/qfq）、在场性、
   尺度证据状态（内部方向诊断仅方向支持；独立核验未核销）；
2) 缓存及外部全部事件在目标窗口的交集/差集/盲区（表外事件不可证，显式标注）；
3) 日历证据 / 历史资格·时间证据分别列状态（E==P+1交易日、窗未截断、P/E 时点）；
4) 每窗"缺什么才能生成工程标签"。
不生成标签、不置覆盖。产物：batch_d_mainpop_30/label_readiness_v1.json
"""
from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

import pyarrow.dataset as ds

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / '../step9_preflight'))
sys.path.insert(0, str(HERE))
from data_source_provenance import DAILY_PARTS  # noqa: E402

BASE = HERE / 'independent_fetch' / 'batch_d_mainpop_30'
MAP = json.loads((BASE / 'five_day_window_map_v2.json').read_text(encoding='utf-8'))
POP_PATH = HERE / '../green_common_profile/gcp_round_b_small_sample_v1' \
                '/gcp_247_ruleA_trigger_records_v1.json'
POP = json.loads(POP_PATH.read_bytes())
RECS = POP['records'] if 'records' in POP else POP

SCALE_EVIDENCE = ('内部方向诊断(method_transformation_continuity)：12 例中 7 例方向支持，'
                  '部分 raw_dev 6-8%，仅方向、非 RAW 确认；独立价格核验未核销')


def load_prices(codes):
    files = sorted(glob.glob(str(Path(DAILY_PARTS) / 'part_*.parquet')))
    need = sorted({str(c) for c in codes})
    tbl = ds.dataset(files, format='parquet').to_table(
        columns=['code', 'date', 'close', 'low'],
        filter=ds.field('code').isin(need))
    out = {}
    for row in zip(tbl['code'].to_pylist(), tbl['date'].to_pylist(),
                   tbl['close'].to_pylist(), tbl['low'].to_pylist()):
        out.setdefault(str(row[0]).zfill(6), {})[int(row[1])] = dict(
            close=row[2], low=row[3])
    return out


def main():
    codes = [c['code'] for c in MAP['codes']]
    prices = load_prices(codes)
    rec_by_codeE = {(r['code'], int(r['E'])): r for r in RECS if r.get('E')}

    inventory = []
    n_ready_data = 0
    for c in MAP['codes']:
        code = c['code']
        px = prices.get(code, {})
        for w in c['windows']:
            p, win = w['P'], w['window']
            p_row = px.get(p)
            p_close = p_row['close'] if p_row else None
            lows = []
            for wd in win:
                row = px.get(int(wd.replace('-', '')))
                lows.append(dict(day=wd,
                                 low=(row['low'] if row else None),
                                 present=row is not None and row['low'] is not None))
            data_ok = (p_row is not None and p_close is not None
                       and all(l['present'] for l in lows))
            n_ready_data += data_ok

            # 事件交差（已检查记录范围）
            hits = sorted(set(w['cache_events_in_window'])
                          | set(w['external_in_window']))
            evt = dict(
                intersection=hits,
                cache_only=w['cache_events_in_window'],
                external_only=w['external_in_window'],
                blind_spot_note='表外事件（重整/要约/缩股/股改对价/停牌）不可证：'
                                '窗内"无记录"≠"无事件"，须公告线才能核销')

            rec = rec_by_codeE.get((code, w['E']), {})
            cal_ok = (w['checks']['five_trading_days']
                      and w['checks']['E_is_first_trading_day_after_P']
                      and not w['checks']['truncated_at_calendar_end'])
            missing = []
            if not data_ok:
                missing.append('价格数据在场性缺口（P收盘或后五日low缺失）')
            missing.append('独立尺度核验（P日参考收盘+后五日各自low 对独立源）——全窗共同阻断项')
            if w['future_scheduled_in_window']:
                missing.append('窗内存在未来预定事件，待生效后复查')
            inventory.append(dict(
                code=code, E=w['E'], P=p, window=win,
                price_evidence=dict(
                    source='canonical daily parts (qfq parquet)',
                    p_close=p_close, lows=lows, all_present=data_ok,
                    scale_status=SCALE_EVIDENCE),
                event_cross_check=evt,
                calendar_status='PASS（5交易日全/E=P+1交易日/未截断）' if cal_ok
                                else 'CHECK_FAIL',
                eligibility_time_evidence=dict(
                    s1_ts=rec.get('s1_ts'), s2_ts=rec.get('s2_ts'),
                    action_kind=rec.get('action_kind'),
                    mutual_status=rec.get('mutual_status')),
                missing_for_engineering_label=missing))
    n = len(inventory)
    out = dict(
        note='44 窗标签工程就绪盘点（零网络）；不生成标签、不置覆盖。'
             '"窗内无事件"仅指已检查记录（缓存+外部成熟记录），表外盲区不可证。',
        windows_total=n,
        windows_price_data_complete=n_ready_data,
        common_blocker='独立尺度核验未核销（P日参考收盘+后五日low 对独立源）',
        inventory=inventory)
    (BASE / 'label_readiness_v1.json').write_text(
        json.dumps(out, ensure_ascii=False, indent=1), encoding='utf-8')
    print(f'窗总数={n} 价格数据完整={n_ready_data}')
    miss = {}
    for x in inventory:
        for m in x['missing_for_engineering_label']:
            miss[m] = miss.get(m, 0) + 1
    for k, v in miss.items():
        print(f'  缺项[{v}窗]: {k}')
    print('written label_readiness_v1.json')


if __name__ == '__main__':
    main()
