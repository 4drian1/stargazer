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
import math
import sys
from pathlib import Path

import pyarrow.dataset as ds

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / '../step9_preflight'))
sys.path.insert(0, str(HERE))
from data_source_provenance import DAILY_PARTS  # noqa: E402

BASE = HERE / 'independent_fetch' / 'batch_d_mainpop_30'
MAP = json.loads((BASE / 'five_day_window_map_v3.json').read_text(encoding='utf-8'))
POP_PATH = HERE / '../green_common_profile/gcp_round_b_small_sample_v1' \
                '/gcp_247_ruleA_trigger_records_v1.json'
POP = json.loads(POP_PATH.read_bytes())
RECS = POP['records'] if 'records' in POP else POP

SCALE_EVIDENCE = ('内部方向诊断(method_transformation_continuity)：12 例中 7 例方向支持，'
                  '部分 raw_dev 6-8%，仅方向、非 RAW 确认；独立价格核验未核销')


def aggregate_prices(rows):
    """纯函数（可单测）：rows = [(code, date, close, low, source), ...]。
    返回 (out, conflicts)。重复 (code,date) 不静默覆盖：记录来源与冲突值；
    value_mismatch=True 的键须阻断使用（标记为不可用）。"""
    out = {}
    conflicts = {}
    for code, date, cl, lo, src in rows:
        key = (str(code).zfill(6), int(date))
        val = dict(close=cl, low=lo, source=src)
        if key in out:
            prev = out[key]
            mismatch = (prev['close'] != cl or prev['low'] != lo)
            if key not in conflicts:
                conflicts[key] = {'sources': [prev['source']],
                                  'values': [(prev['close'], prev['low'])],
                                  'value_mismatch': False}
            conflicts[key]['sources'].append(src)
            conflicts[key]['values'].append((cl, lo))
            conflicts[key]['value_mismatch'] = conflicts[key]['value_mismatch'] or mismatch
        out[key] = val
    return out, conflicts


def load_prices(codes):
    """读 daily parts，按 part 文件逐源归属；返回 (out, conflicts)。
    out: {(code,date): {close,low,source}}；conflicts: 同键多源记录来源与是否数值冲突；
    value_mismatch 的键须阻断使用（见 evaluate_window_prices）。"""
    files = sorted(glob.glob(str(Path(DAILY_PARTS) / 'part_*.parquet')))
    need = sorted({str(c) for c in codes})
    rows = []
    for fpath in files:
        tbl = ds.dataset([fpath], format='parquet').to_table(
            columns=['code', 'date', 'close', 'low'],
            filter=ds.field('code').isin(need))
        for code, date, cl, lo in zip(tbl['code'].to_pylist(), tbl['date'].to_pylist(),
                                      tbl['close'].to_pylist(), tbl['low'].to_pylist()):
            rows.append((str(code), int(date), cl, lo, Path(fpath).name))
    return aggregate_prices(rows)


def evaluate_window_prices(prices, conflict_keys, code, p, win):
    """返回 (p_close, lows, data_ok, any_conflict)。
    data_ok 排除冲突键（重复键数值不一致时不静默放行）；present 同时要求有限正数。"""
    p_key = (code, int(p))
    p_conflict = p_key in conflict_keys
    p_row = prices.get(p_key)
    p_close = p_row['close'] if p_row else None
    lows = []
    any_conflict = p_conflict
    for wd in win:
        dkey = (code, int(wd.replace('-', '')))
        row = prices.get(dkey)
        conflict = dkey in conflict_keys
        any_conflict = any_conflict or conflict
        present = row is not None and valid_price(row['low'])
        lows.append(dict(day=wd, low=(row['low'] if row else None),
                         present=present, conflict=conflict))
    p_close_ok = p_row is not None and valid_price(p_close) and not p_conflict
    data_ok = p_close_ok and all(l['present'] and not l['conflict'] for l in lows)
    return p_close, lows, data_ok, any_conflict


def valid_price(v):
    """价格须有限且为正（不只非空）；NaN/inf/<=0 均判无效。"""
    return v is not None and isinstance(v, (int, float)) and math.isfinite(v) and v > 0


def main():
    codes = [c['code'] for c in MAP['codes']]
    prices, conflicts = load_prices(codes)
    conflict_keys = {k for k, c in conflicts.items() if c['value_mismatch']}
    n_conflict_keys = len(conflict_keys)
    rec_by_codeE = {(r['code'], int(r['E'])): r for r in RECS if r.get('E')}

    # 全局未关门（适用全部窗；价格比对通过不会自动关闭）
    global_gates = [
        '下一项优先核验任务：独立尺度核验（P日参考收盘+后五日各自low 对独立源，绝对差≤0.01元同口径比对）',
        '公司行为覆盖未核销（缓存 events 不能自证完备；独立台账路径待评估，不自动关闭）',
        '历史到达/资格时刻未知（实时日志缺失，全窗口 UNKNOWN）——需前瞻记录或显式授权范围',
        '数据源口径未确认（当前仅内部方向诊断 RAW_LIKE 方向支持，缺证据，不标 qfq）',
    ]

    inventory = []
    n_ready_data = 0
    for c in MAP['codes']:
        code = c['code']
        for w in c['windows']:
            p, win = w['P'], w['window']
            p_close, lows, data_ok, any_conflict = evaluate_window_prices(
                prices, conflict_keys, code, p, win)
            n_ready_data += data_ok

            # 事件交差：交集 / 缓存−外部 / 外部−缓存（非并集）
            cs = set(w['cache_events_in_window'])
            es = set(w['external_in_window'])
            evt = dict(
                intersection=sorted(cs & es),
                cache_only=sorted(cs - es),
                external_only=sorted(es - cs),
                blind_spot_note='表外事件（重整/要约/缩股/股改对价/停牌）不可证：'
                                '窗内"无记录"≠"无事件"，须公告线才能核销')

            rec = rec_by_codeE.get((code, w['E']), {})
            cal_ok = (w['checks']['five_trading_days']
                      and w['checks']['E_is_first_trading_day_after_P']
                      and not w['checks']['truncated_at_calendar_end'])
            missing = []
            if not data_ok:
                missing.append('价格数据在场性/有限性/正数缺口（P收盘或后五日low 缺失/NaN/非正/重复冲突）')
            if w['future_scheduled_in_window']:
                missing.append('窗内存在未来预定事件，待生效后复查')
            missing.extend(global_gates)   # 全局门计入缺项汇总
            inventory.append(dict(
                code=code, E=w['E'], P=p, window=win,
                price_evidence=dict(
                    source='canonical daily parts (scale UNVERIFIED: 仅内部方向诊断 '
                           'RAW_LIKE_DIRECTION_SUPPORTED，非 qfq 确认；qfq 标签缺证据)',
                    p_close=p_close, lows=lows, all_present=data_ok,
                    conflict_used=any_conflict,
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
             '本盘点仅确认"价格字段在场性/有限性/正数"，不是标签准入；'
             '标签准入另有多道未关门（独立尺度核验/公司行为覆盖/历史时刻/数据源口径）。'
             '"窗内无事件"仅指已检查记录（缓存+外部成熟记录），表外盲区不可证。',
        windows_total=n,
        windows_price_data_complete=n_ready_data,
        next_priority_verification_task='独立尺度核验（P日参考收盘+后五日各自low 对独立源，绝对差≤0.01元同口径比对）——'
                                        '下一项优先核验任务，非唯一阻断项',
        open_gates=global_gates,
        price_key_conflicts=dict(
            blocking_count=n_conflict_keys,
            total_duplicates=len(conflicts),
            blocking_sample=[{'code': k[0], 'date': k[1],
                              'sources': conflicts[k]['sources'],
                              'values': conflicts[k]['values']}
                             for k in list(conflict_keys)[:10]]),
        inventory=inventory)
    (BASE / 'label_readiness_v1.json').write_text(
        json.dumps(out, ensure_ascii=False, indent=1), encoding='utf-8')
    print(f'窗总数={n} 价格数据完整(有限/正数)={n_ready_data} '
          f'重复键总数={len(conflicts)} 其中数值冲突(阻断)={n_conflict_keys}')
    miss = {}
    for x in inventory:
        for m in x['missing_for_engineering_label']:
            miss[m] = miss.get(m, 0) + 1
    for k, v in miss.items():
        print(f'  缺项[{v}窗]: {k}')
    print('written label_readiness_v1.json')


if __name__ == '__main__':
    main()
