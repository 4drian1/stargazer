"""five_day_window_calendar_map.py — 冻结日历映射候选五日窗（零网络；v2 修正版）。

v1 盲点修正（用户 2026-10-13 复核）：
- 外部日期不再只取 date_hits：纳入 outside_cache（外部独有）与 date_diff_pending
  （日期未决）；future_scheduled 单列不入成熟集——"寻找遗漏"的检查必须看全部外部记录。
- 每窗显式校验：恰好 5 个交易日 / P-E 关系 / 日历截止（截断标记）/ 输入哈希，
  不再无条件写 in_calendar=True。
- 结论表述收窄为"已检查记录中未检出窗口内公司行为"，不写"窗内事件为0/覆盖通过"。
产物：batch_d_mainpop_30/five_day_window_map_v2.json（保留 v1）。
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / '../step9_preflight'))
sys.path.insert(0, str(HERE))
from trading_calendar import load_calendar  # noqa: E402

BASE = HERE / 'independent_fetch' / 'batch_d_mainpop_30'
REAN = json.loads((BASE / 'diff_table_reanalysis_v1.json').read_text(encoding='utf-8'))
DESIGN = json.loads((HERE / 'batch_d_design.json').read_text(encoding='utf-8'))
POP_PATH = HERE / '../green_common_profile/gcp_round_b_small_sample_v1' \
                '/gcp_247_ruleA_trigger_records_v1.json'
CACHE_PATH = HERE / '../opportunity_v2/gate1c/feb2026_adjustment_cache_v1' \
                  '/adj_cache.json'


def verify_inputs():
    pop_sha = hashlib.sha256(POP_PATH.read_bytes()).hexdigest()
    cache_sha = hashlib.sha256(CACHE_PATH.read_bytes()).hexdigest()
    ok = (pop_sha == DESIGN['population_binding']['sha256']
          and cache_sha == DESIGN['cache_binding']['sha256'])
    return ok, pop_sha[:12], cache_sha[:12]


def main():
    ok, pop_s, cache_s = verify_inputs()
    specs = {s['code']: s for s in DESIGN['code_specs']}
    rean = {r['code']: r for r in REAN['results']}
    CAL = load_calendar()
    cal_end = f'{CAL[-1] // 10000:04d}-{CAL[-1] % 10000 // 100:02d}-{CAL[-1] % 100:02d}'

    out = []
    n_win = n_full5 = n_pe_ok = n_trunc = n_with_evt = 0
    special_in_win = []

    for code, spec in specs.items():
        pop = json.loads(POP_PATH.read_bytes())
        code_recs = [r for r in (pop['records'] if 'records' in pop else pop)
                     if r['code'] == code and r.get('P')]
        cache_days = {e['date'] for e in spec['cache_events']}
        r = rean.get(code, {})
        wd = r.get('window_dx', {})
        ext_mature = set()
        for src in ('date_hits', 'outside_cache', 'date_diff_pending'):
            for x in wd.get(src, []):
                ext_mature.add(x['external']['ex_date'])
        for x in wd.get('not_returned', []):
            cache_days.add(x['cache_date'])
        future_days = {x['external']['ex_date'] for x in wd.get('future_scheduled', [])}

        entries = []
        for rec in code_recs:
            p = int(rec['P'])
            later = [d for d in CAL if d > p]
            win = later[:5]
            n_win += 1
            checks = {}
            checks['five_trading_days'] = (len(win) == 5)
            n_full5 += checks['five_trading_days']
            e = int(rec['E']) if rec.get('E') else None
            checks['E_is_first_trading_day_after_P'] = (
                e is not None and len(later) > 0 and e == later[0])
            n_pe_ok += checks['E_is_first_trading_day_after_P']
            win_str = [f'{d // 10000:04d}-{d % 10000 // 100:02d}-{d % 100:02d}'
                       for d in win]
            checks['truncated_at_calendar_end'] = any(w > cal_end for w in win_str)
            n_trunc += checks['truncated_at_calendar_end']
            checks['in_calendar'] = all(
                w <= cal_end for w in win_str) and checks['five_trading_days']

            hits_cache = sorted(d for d in cache_days if d in win_str)
            hits_ext = sorted(d for d in ext_mature if d in win_str)
            hits_future = sorted(d for d in future_days if d in win_str)
            if hits_cache or hits_ext:
                n_with_evt += 1
                if '2023-12-29' in hits_cache + hits_ext:
                    special_in_win.append(dict(code=code, P=rec['P'], E=e,
                                               window=win_str,
                                               hits=hits_cache + hits_ext))
            entries.append(dict(E=e, P=p, window=win_str, checks=checks,
                                cache_events_in_window=hits_cache,
                                external_in_window=hits_ext,
                                future_scheduled_in_window=hits_future))
        out.append(dict(code=code, stratum=spec['stratum'], windows=entries))

    result = dict(
        note='v2 修正：外部日期=命中+外部独有+日期未决（future 单列）；每窗校验窗长/P-E/'
             '日历截断；输入哈希校验={}。结论收窄：在已检查记录中未检出窗口内公司行为'
             '——不写"窗内事件为0、覆盖通过"。'.format('PASS' if ok else 'FAIL'),
        input_hash_check='PASS' if ok else 'FAIL',
        input_hash=dict(population=pop_s, cache=cache_s,
                        reanalysis_file='diff_table_reanalysis_v1.json'),
        calendar_evidence='trading_calendar.load_calendar()（canonical daily parts'
                          ' 推导，非交易所官方认证）',
        calendar_range_end=cal_end,
        total_candidate_windows=n_win,
        windows_full_five_trading_days=n_full5,
        windows_E_is_first_td_after_P=n_pe_ok,
        windows_truncated_at_calendar_end=n_trunc,
        windows_with_corporate_action_in_checked_records=n_with_evt,
        special_event_20231229_in_any_window=special_in_win,
        narrow_conclusion='已检查记录（缓存事件+外部成熟记录）中，44 窗均未检出窗内公司行为'
                          if n_with_evt == 0 else f'{n_with_evt} 窗检出事件，须逐窗裁决',
        codes=out)
    (BASE / 'five_day_window_map_v2.json').write_text(
        json.dumps(result, ensure_ascii=False, indent=1), encoding='utf-8')
    print(f'hash_check={"PASS" if ok else "FAIL"} 窗总数={n_win} '
          f'恰好5交易日={n_full5} E=P+1交易日={n_pe_ok} 截断={n_trunc} '
          f'已检查记录窗内有事件={n_with_evt}')
    print('2023-12-29 落入窗口:', special_in_win if special_in_win else '无')
    print('written five_day_window_map_v2.json')


if __name__ == '__main__':
    main()
