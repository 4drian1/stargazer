"""reanalyze_batch_d.py — 批次d零网络重判（用户 2026-10-13 三处纠正；新增版本，保留原件）。

纠正点：
1) classify 先按原始参数分类再决定可比性：派>0 一律现金存在（不做无派息一致判定）；
   缺参数（None）记 PARAM_INCOMPLETE，不能当 0；股数倍率接近只作描述不作判据。
2) 外部除权日晚于该码抓取时点 → FUTURE_SCHEDULED，不入成熟事件差集/覆盖评价。
3) 600589 类：无现金送转参数与缓存因子差异 → SHARE_FACTOR_GAP_PENDING（差异待核，
   不足以认定缓存错误）。
另：目标五日窗口（E..E+7 自然日近似口径，非交易所日历认证）与全历史诊断分列报告；
outside_cache=0 仅描述已检查范围。
产物：batch_d_mainpop_30/diff_table_reanalysis_v1.json（不改写 diff_table.json）。
"""
from __future__ import annotations

import json
import os
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sina_div_client import parse_bonus_page

HERE = Path(__file__).resolve().parent
BASE = HERE / 'independent_fetch' / 'batch_d_mainpop_30'
DESIGN = json.loads((HERE / 'batch_d_design.json').read_text(encoding='utf-8'))
SPECS = {s['code']: s for s in DESIGN['code_specs']}
POP = json.loads((HERE / '../green_common_profile/gcp_round_b_small_sample_v1'
                  '/gcp_247_ruleA_trigger_records_v1.json').read_bytes())
RECS = POP['records'] if 'records' in POP else POP


def num_or_none(v):
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def classify_hit(ext, cache_mult):
    """先按原始参数分类，再决定可比性（用户纠正①）。"""
    if ext['table'] == 'allotment':
        return 'TYPE_ALLOTMENT_NOT_COMPARABLE', '配股无进度列按候选保留；因子需配股价与前收（未核销）'
    s, z, p = ext['送'], ext['转'], ext['派']
    if s is None or z is None or p is None:
        return 'PARAM_INCOMPLETE', '缺参数（None 不当 0），不可判定'
    share_factor = 1.0 + (s + z) / 10.0
    near = abs(share_factor - cache_mult) < 0.01
    if p > 0:
        st = ('SHARE_FACTOR_MATCH_CASH_PRESENT' if near
              else 'CASH_PRESENT_FACTOR_NOT_COMPARABLE')
        note = (f'派{p}元现金存在，股数倍率{share_factor:.4f} vs 缓存{cache_mult}'
                f'（接近={near}）仅描述；数值验证需前收口径（未核销）')
        return st, note
    # p == 0：无现金送转，股数倍率可直接比对
    if near:
        return 'SHARE_FACTOR_MATCH_NO_CASH', None
    return ('SHARE_FACTOR_GAP_PENDING',
            f'无现金送转：股数倍率{share_factor:.4f} vs 缓存{cache_mult}，差异待核'
            '（不足以认定缓存错误）')


def main():
    run_dirs = sorted(p for p in BASE.glob('run_*') if p.is_dir())
    assert len(run_dirs) == 1, run_dirs
    run = run_dirs[0]
    fetched = {}
    for line in (run / 'index.jsonl').read_text(encoding='utf-8').splitlines():
        m = json.loads(line)
        fetched[m['code']] = m['fetched_at'][:10]

    results = []
    for f in sorted(run.glob('*.bin')):
        code = os.path.basename(f).split('_')[1].split('.')[0]
        spec = SPECS[code]
        pg = parse_bonus_page(f.read_bytes())
        win = set(spec['window'])
        fetch_day = fetched[code]
        # 该码全部候选 E（来自冻结人口），近似五日窗 = E..E+7 自然日（非认证日历）
        code_Es = sorted({r['E'] for r in RECS if r['code'] == code and r.get('E')})
        e_windows = []
        for e in code_Es:
            d = datetime.strptime(str(e), '%Y%m%d')
            e_windows.append(((d).strftime('%Y-%m-%d'),
                              (d.fromordinal(d.toordinal() + 7)).strftime('%Y-%m-%d')))

        ext_impl = []
        for kind, key_ex in (('bonus', '除权除息日'), ('allotment', '除权日')):
            for r in (pg.get(kind) or []):
                prog = (r.get('进度') or '').strip()
                if kind == 'bonus' and prog != '实施':
                    continue
                ex = str(r.get(key_ex) or '')[:10]
                if ex and ex in win:
                    ext_impl.append(dict(
                        table=kind, ex_date=ex,
                        公告日期=str(r.get('公告日期'))[:10],
                        进度=prog or None,
                        送=num_or_none(r.get('送股(股)')),
                        转=num_or_none(r.get('转增(股)')),
                        派=num_or_none(r.get('派息(税前)(元)')),
                        date_basis=('sina_allotment_no_progress'
                                    if kind == 'allotment' else 'sina_returned_only')))
        cache_days = {e['date']: e['mult'] for e in spec['cache_events']}
        ext_days = {e['ex_date'] for e in ext_impl}

        mature, future = [], []
        for e in ext_impl:
            if e['ex_date'] > fetch_day:
                future.append(dict(external=e, status='FUTURE_SCHEDULED',
                                   note=f'除权日{e["ex_date"]}晚于抓取日{fetch_day}；'
                                        '页面"实施"≠已生效，不入成熟事件差集/覆盖评价'))
            else:
                mature.append(e)

        hits, date_diff, outside = [], [], []
        for e in mature:
            if e['ex_date'] in cache_days:
                st, note = classify_hit(e, cache_days[e['ex_date']])
                hits.append(dict(external=e, status=st, note=note))
            else:
                d = datetime.strptime(e['ex_date'], '%Y-%m-%d')
                near = [c for c in cache_days
                        if abs((datetime.strptime(c, '%Y-%m-%d') - d).days) <= 90]
                if near:
                    date_diff.append(dict(external=e, near_cache_dates=sorted(near),
                                          status='DATE_DIFF_PENDING', note='未裁决'))
                else:
                    outside.append(dict(external=e, status='OUTSIDE_CACHE_CANDIDATE',
                                        note='已检查范围内无缓存事件；不宣称无遗漏'))
        not_returned = [dict(cache_date=c, cache_mult=m,
                             status='NOT_RETURNED_CANDIDATE', note='未裁决')
                        for c, m in sorted(cache_days.items()) if c not in ext_days]

        def in_e_window(day):
            return any(a <= day <= b for a, b in e_windows)

        hits_ew = [h for h in hits if in_e_window(h['external']['ex_date'])]
        hits_hist = [h for h in hits if not in_e_window(h['external']['ex_date'])]
        nr_ew = [x for x in not_returned if in_e_window(x['cache_date'])]

        ex_all = sorted(str(r.get('除权除息日'))[:10] for r in (pg.get('bonus') or [])
                        if str(r.get('除权除息日') or '')[:4].isdigit())
        prog_dist = {}
        for r in (pg.get('bonus') or []):
            p = str(r.get('进度') or 'NULL')
            prog_dist[p] = prog_dist.get(p, 0) + 1
        history = dict(bonus_rows=len(pg.get('bonus') or []),
                       allotment_rows=len(pg.get('allotment') or []),
                       bonus_progress_dist=prog_dist,
                       bonus_ex_date_range=[ex_all[0], ex_all[-1]] if ex_all else None,
                       implemented_rows_outside_window=None)

        results.append(dict(
            code=code, stratum=spec['stratum'], smoke=spec['smoke'], P=spec['P'],
            fetch_day=fetch_day,
            target_windows_approx=dict(codeE_n=len(code_Es), windows=e_windows,
                                       note='E..E+7自然日近似，非交易所日历认证'),
            window_dx=dict(date_hits=hits, hits_in_target_windows=hits_ew,
                           hits_full_history_only=hits_hist,
                           date_diff_pending=date_diff, outside_cache=outside,
                           not_returned=not_returned,
                           not_returned_in_target_windows=nr_ew,
                           future_scheduled=future),
            full_history_dx=None, history=history))
        print(f"{code} [{spec['stratum']}] 成熟窗内={len(mature)} 未来={len(future)} "
              f"命中={len(hits)}(命中在五日窗内={len(hits_ew)}) 日期差={len(date_diff)} "
              f"outside={len(outside)} 未返回={len(not_returned)}(在五日窗内={len(nr_ew)})")

    out = dict(reanalysis_note='零网络重判（用户三处纠正）：参数先分类再可比；'
                               'FUTURE_SCHEDULED 分离；无现金送转差异待核；'
                               '目标五日窗(近似口径)与全历史分列；保留原件不改写。',
               source_run_dir=str(run), results=results)
    (BASE / 'diff_table_reanalysis_v1.json').write_text(
        json.dumps(out, ensure_ascii=False, indent=1), encoding='utf-8')
    print('written diff_table_reanalysis_v1.json')


if __name__ == '__main__':
    main()
