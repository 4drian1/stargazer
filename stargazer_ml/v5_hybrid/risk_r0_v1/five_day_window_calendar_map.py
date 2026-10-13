"""five_day_window_calendar_map.py — 冻结日历映射候选五日窗（零网络，用户 2026-10-13 指示）。

撤回 E..E+7 自然日近似口径的正式表述；改用 step9_preflight/trading_calendar
（由 canonical daily parts 推导的统一冻结日历）逐事件映射 P 后五个交易日，
并检查缓存事件/外部命中是否落在窗内。日历证据状态单列。
另按用户指示：判断 2023 特殊事件（600589@2023-12-29）是否位于各标签必要窗口内。
产物：batch_d_mainpop_30/five_day_window_map_v1.json
"""
from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / '../step9_preflight'))
sys.path.insert(0, str(HERE))
from trading_calendar import load_calendar  # noqa: E402

BASE = HERE / 'independent_fetch' / 'batch_d_mainpop_30'
REAN = json.loads((BASE / 'diff_table_reanalysis_v1.json').read_text(encoding='utf-8'))
DESIGN = json.loads((HERE / 'batch_d_design.json').read_text(encoding='utf-8'))
POP = json.loads((HERE / '../green_common_profile/gcp_round_b_small_sample_v1'
                  '/gcp_247_ruleA_trigger_records_v1.json').read_bytes())
RECS = POP['records'] if 'records' in POP else POP

CAL = load_calendar()
CAL_SET = set(CAL)


def next5_after(p_int: int):
    later = [d for d in CAL if d > p_int]
    return later[:5]


def main():
    specs = {s['code']: s for s in DESIGN['code_specs']}
    rean = {r['code']: r for r in REAN['results']}
    out = []
    total_windows = 0
    windows_with_any_event = 0
    special_hits = []

    for code, spec in specs.items():
        code_recs = [r for r in RECS if r['code'] == code and r.get('P')]
        cache_days = {e['date'] for e in spec['cache_events']}
        ext_days = set()
        r = rean.get(code, {})
        for h in r.get('window_dx', {}).get('date_hits', []):
            ext_days.add(h['external']['ex_date'])
        for x in r.get('window_dx', {}).get('not_returned', []):
            cache_days.add(x['cache_date'])

        entries = []
        for rec in code_recs:
            p = int(rec['P'])
            win = next5_after(p)
            if not win:
                continue
            total_windows += 1
            win_str = [f'{d // 10000:04d}-{d % 10000 // 100:02d}-{d % 100:02d}'
                       for d in win]
            hit_cache = sorted(d for d in cache_days if d in win_str)
            hit_ext = sorted(d for d in ext_days if d in win_str)
            if hit_cache or hit_ext:
                windows_with_any_event += 1
                if '2023-12-29' in hit_cache + hit_ext:
                    special_hits.append(dict(code=code, P=rec['P'], E=rec.get('E'),
                                             window=win_str, hits=hit_cache + hit_ext))
            entries.append(dict(E=rec.get('E'), P=p, window=win_str,
                                in_calendar=True,
                                cache_events_in_window=hit_cache,
                                external_hits_in_window=hit_ext))
        out.append(dict(code=code, stratum=spec['stratum'],
                        calendar_evidence='trading_calendar.load_calendar()'
                                          '(canonical daily parts 推导)',
                        windows=entries))

    result = dict(
        note='冻结日历逐事件映射 P 后五个交易日；近似自然日口径正式表述已撤回。'
             '2023-12-29(600589 重整特殊事件)与各标签窗口的关系单列。',
        total_candidate_windows=total_windows,
        windows_with_corporate_action=windows_with_any_event,
        special_event_20231229_in_any_window=special_hits,
        calendar_range=[f'{CAL[0]}', f'{CAL[-1]}'], codes=out)
    (BASE / 'five_day_window_map_v1.json').write_text(
        json.dumps(result, ensure_ascii=False, indent=1), encoding='utf-8')
    print(f'calendar {CAL[0]}..{CAL[-1]} 候选五日窗总数={total_windows} '
          f'窗内有公司行为事件={windows_with_any_event}')
    print('2023-12-29 落入窗口:', special_hits if special_hits else '无')
    print('written five_day_window_map_v1.json')


if __name__ == '__main__':
    main()
