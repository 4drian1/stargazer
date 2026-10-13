"""reanalyze_batch_c.py — 批次②零网络事后复核。

批次②原始 diff_table.json 的过滤逻辑用平键取 进度/除权除息日，
但 parse_bonus_page 对本批 9 个页面均返回元组键 → 全部漏配（impl=0）。
本脚本对已留证的原始响应（run_*/*.bin）做本地重解析，不新增任何请求，
不改写原 diff_table.json，产物为 diff_table_reanalysis_v1.json。
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
    out = []
    for r in rows:
        nr = {}
        for k, v in r.items():
            nr[k[-1] if isinstance(k, tuple) else k] = v
        out.append(nr)
    return out


def main():
    results = []
    for f in sorted(glob.glob(str(RUN / '*.bin'))):
        code = os.path.basename(f).split('_')[1].split('.')[0]
        spec = SPECS[code]
        pg = parse_bonus_page(Path(f).read_bytes())
        win = set(spec['window'])
        impl = []
        for kind, key_ex in (('bonus', '除权除息日'), ('allotment', '除权日')):
            for r in flat(pg.get(kind) or []):
                # 配股表无“进度”列：None 不等于非实施，按候选保留并注明日期依据
                prog = (r.get('进度') or '').strip()
                if kind == 'bonus' and prog != '实施':
                    continue
                ex = str(r.get(key_ex) or '')[:10]
                if ex and ex in win:
                    impl.append(dict(table=kind,
                                     公告日期=str(r.get('公告日期'))[:10],
                                     ex_date=ex,
                                     进度=prog or None,
                                     date_basis=('sina_allotment_no_progress'
                                                 if kind == 'allotment' else 'sina_returned_only'),
                                     params={k: (str(v)[:16] if v is not None else None)
                                             for k, v in r.items()}))
        cache_days = {e['date'] for e in spec['cache_events']}
        ext_days = {e['ex_date'] for e in impl}
        outside = [e for e in impl if not any(
            abs((datetime.strptime(c, '%Y-%m-%d') - datetime.strptime(e['ex_date'], '%Y-%m-%d')).days) <= 90
            for c in cache_days)]
        notret = [c for c in cache_days if not any(
            abs((datetime.strptime(e, '%Y-%m-%d') - datetime.strptime(c, '%Y-%m-%d')).days) <= 90
            for e in ext_days)]
        results.append(dict(code=code, stratum=spec['stratum'], P=spec['P'],
                            ext_impl=impl, outside_cache=outside, not_returned=notret))
        print(f"{code} [{spec['stratum']}] impl={len(impl)} "
              f"outside={len(outside)} not_returned={len(notret)}")
        for e in impl:
            print('   ', e['table'], e['公告日期'], 'ex', e['ex_date'])

    out = dict(
        reanalysis_note='零网络事后复核：原始 diff_table.json 因 parse_bonus_page 对本批页面返回元组键、'
                        '过滤键取不到而全部漏配。本文件为对已留证原始响应的本地重解析，'
                        '无新增请求；不改写原 diff_table.json。'
                        'v2 追加：配股表无进度列，None 不再按非实施剔除，按候选保留并注明日期依据。',
        source_run_dir=str(RUN), results=results)
    (BASE / 'diff_table_reanalysis_v2.json').write_text(
        json.dumps(out, ensure_ascii=False, indent=1), encoding='utf-8')
    print('written diff_table_reanalysis_v2.json')


if __name__ == '__main__':
    main()
