"""breakdown_batch_c.py — 批次②九码逐项零网络拆解（用户 2026-10-13 裁决②）。

每项拆四层：全历史诊断 / 目标窗口记录 / 类型·参数匹配状态 / 未决原因。
输入仅为已留证原始响应（run_*/*.bin）与冻结设计 v2，无新增请求。
EMPTY 结论保持“未检出记录”，不认证无事件；
301682 记整码缓存 MISSING（其外部事件早于样本 P，不称五日窗口内缺失）。
产物：batch_c_mainpop_probe/code_breakdown_v1.json（不改写任何原始产物）。
"""
from __future__ import annotations

import glob
import json
import os
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sina_div_client import parse_bonus_page

HERE = Path(__file__).resolve().parent
DESIGN = json.loads((HERE / 'main_population_sample_design_v2.json').read_text(encoding='utf-8'))
SPECS = {s['code']: s for s in DESIGN['code_specs']}
BASE = HERE / 'independent_fetch' / 'batch_c_mainpop_probe'
RUN = BASE / 'run_20261013T134821_186591_47564'


def flat(rows):
    return [{(k[-1] if isinstance(k, tuple) else k): v for k, v in r.items()}
            for r in rows]


def num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def main():
    breakdown = []
    for f in sorted(glob.glob(str(RUN / '*.bin'))):
        code = os.path.basename(f).split('_')[1].split('.')[0]
        spec = SPECS[code]
        pg = parse_bonus_page(Path(f).read_bytes())
        bonus = flat(pg.get('bonus') or [])
        allot = flat(pg.get('allotment') or [])

        # 1) 全历史诊断
        prog_dist = {}
        for r in bonus:
            p = str(r.get('进度') or 'NULL')
            prog_dist[p] = prog_dist.get(p, 0) + 1
        ex_dates = sorted(str(r.get('除权除息日'))[:10] for r in bonus
                          if str(r.get('除权除息日') or '')[:4].isdigit())
        history = dict(
            bonus_rows=len(bonus), allotment_rows=len(allot),
            bonus_progress_dist=prog_dist,
            bonus_ex_date_range=[ex_dates[0], ex_dates[-1]] if ex_dates else None,
            allotment_has_progress_col=any('进度' in r for r in allot))

        # 2) 目标窗口记录（P±120 ∪ 缓存事件±90，冻结设计 v2）
        win = set(spec['window'])
        win_impl = []
        for kind, rows, key_ex in (('bonus', bonus, '除权除息日'),
                                   ('allotment', allot, '除权日')):
            for r in rows:
                prog = (r.get('进度') or '').strip()
                if kind == 'bonus' and prog != '实施':
                    continue
                ex = str(r.get(key_ex) or '')[:10]
                if ex in win:
                    win_impl.append(dict(table=kind, ex_date=ex,
                                         公告日期=str(r.get('公告日期'))[:10],
                                         进度=prog or None,
                                         送=num(r.get('送股(股)')),
                                         转=num(r.get('转增(股)')),
                                         派=num(r.get('派息(税前)(元)')),
                                         配股=num(r.get('配股方案(每10股配股股数)'))))

        # 3) 类型·参数匹配状态（缓存事件逐条）
        cache_events = spec['cache_events']
        pair_status = []
        used_ext = set()
        for ce in cache_events:
            cd = datetime.strptime(ce['date'], '%Y-%m-%d')
            cands = [(i, e) for i, e in enumerate(win_impl)
                     if i not in used_ext and
                     abs((datetime.strptime(e['ex_date'], '%Y-%m-%d') - cd).days) <= 90]
            if not cands:
                pair_status.append(dict(cache_date=ce['date'], cache_mult=ce['mult'],
                                        status='EXTERNAL_NOT_RETURNED',
                                        unresolved='分红/配股两表均无 ±90 天记录'
                                        if code != '301682' else '整码缓存MISSING，外部有记录：属缓存缺失候选'))
                continue
            i, e = min(cands, key=lambda x: abs((datetime.strptime(x[1]['ex_date'], '%Y-%m-%d') - cd).days))
            used_ext.add(i)
            st = dict(cache_date=ce['date'], cache_mult=ce['mult'],
                      external=e, status=None, factor_check=None)
            if e['table'] == 'bonus':
                simple = 1.0 + ((e['送'] or 0) + (e['转'] or 0)) / 10.0
                st['factor_check'] = dict(
                    simple_factor_no_cash=round(simple, 6),
                    vs_cache_mult=ce['mult'],
                    note='未含派息项；派息需前收口径（未核销），仅并列不作判据')
                if abs(simple - ce['mult']) < 0.01:
                    st['status'] = 'TYPE_PARAMS_CONSISTENT_NO_CASH'
                elif (e['送'] or 0) == 0 and (e['转'] or 0) == 0 and (e['派'] or 0) > 0:
                    # 纯派息：adj 因子 = 1/(1-派/10/前收) > 1，无需前收即可判定
                    # “与口径相容但不可并列验证”，与参数矛盾是两回事
                    st['status'] = 'CASH_ONLY_FACTOR_NOT_COMPARABLE'
                    st['unresolved'] = '纯派息事件；方向相容（mult>1），数值验证需前收口径（未核销）'
                else:
                    st['status'] = 'PARAM_GAP_UNRESOLVED'
                    st['unresolved'] = (f'含送转：简单因子{simple:.4f} vs 缓存{ce["mult"]}；'
                                        f'差额或来自派息项，需前收口径核销（未裁决）')
                    if ce['mult'] == 1.0 and ((e['送'] or 0) + (e['转'] or 0)) > 0:
                        st['unresolved'] += '；缓存 mult=1.0 却有送转，属真异常候选'
            else:
                st['status'] = 'TYPE_ALLOTMENT_FACTOR_NOT_COMPARABLE'
                st['unresolved'] = '配股因子需配股价与前收（均未核销），不并列判据'
            pair_status.append(st)

        outside = [e for i, e in enumerate(win_impl) if i not in used_ext]
        entry = dict(code=code, stratum=spec['stratum'], P=spec['P'],
                     history=history, window_impl=win_impl,
                     pair_status=pair_status, outside_window_impl=outside)
        if code == '301682':
            entry['missing_note'] = ('整码缓存 MISSING；外部 2026-07-13 事件早于样本 P=2026-08-25，'
                                     '不称五日标签窗口内缺失事件')
        breakdown.append(entry)
        n_match = sum(1 for p in pair_status if p['status'] and p['status'].startswith(('TYPE_PARAMS', 'TYPE_ALLOTMENT')))
        n_gap = sum(1 for p in pair_status if p.get('status') == 'PARAM_GAP_UNRESOLVED')
        n_notret = sum(1 for p in pair_status if p['status'] == 'EXTERNAL_NOT_RETURNED')
        print(f"{code} [{spec['stratum']}] 全历史 bonus={history['bonus_rows']} 配股={history['allotment_rows']} "
              f"窗口实施={len(win_impl)} 配对={n_match} 参数差未决={n_gap} 未返回={n_notret} 窗口外剩余={len(outside)}")

    out = dict(note='零网络拆解：全历史诊断/目标窗口记录/类型·参数匹配/未决原因四层分列；'
                    '日期命中≠全要素匹配；EMPTY=未检出记录；301682=整码MISSING。',
               source_run_dir=str(RUN), breakdown=breakdown)
    (BASE / 'code_breakdown_v1.json').write_text(
        json.dumps(out, ensure_ascii=False, indent=1), encoding='utf-8')
    print('written code_breakdown_v1.json')


if __name__ == '__main__':
    main()
