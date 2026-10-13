"""gen_batch_d_design.py — 批次d（主人口30码）设计文件生成，零网络。

代码清单冻结自 next_batch_code_list_draft.json（用户已复验）；
每码规格（最早 P / 全部 Ps / 缓存事件 / 检索窗口）机械导出自冻结人口与
adj_cache（哈希绑定复验）；冒烟码 = 清单前 2（688047、688180），页面复用不另查。
"""
from __future__ import annotations

import json
import hashlib
from datetime import datetime, timedelta
from pathlib import Path

HERE = Path(__file__).resolve().parent
POP = HERE / '../green_common_profile/gcp_round_b_small_sample_v1/gcp_247_ruleA_trigger_records_v1.json'
CACHE = HERE / '../opportunity_v2/gate1c/feb2026_adjustment_cache_v1/adj_cache.json'
LIST = json.loads((HERE / 'next_batch_code_list_draft.json').read_text(encoding='utf-8'))


def fmt(d) -> str:
    return f'{d // 10000:04d}-{d % 10000 // 100:02d}-{d % 100:02d}'


def main():
    pop_sha = hashlib.sha256(POP.read_bytes()).hexdigest()
    cache_sha = hashlib.sha256(CACHE.read_bytes()).hexdigest()
    assert pop_sha == LIST['population_sha'], '人口哈希不一致'
    assert cache_sha == LIST['cache_sha'], '缓存哈希不一致'
    pop = json.loads(POP.read_bytes())
    recs = pop['records'] if 'records' in pop else pop
    cache = json.loads(CACHE.read_bytes())

    codes = (LIST['next_batch_draft']['EMPTY_all']
             + LIST['next_batch_draft']['OK_sample_n23'])
    assert len(codes) == 30 and len(set(codes)) == 30
    smoke = codes[:2]
    assert smoke == ['688047', '688180'], smoke

    specs = []
    for code in codes:
        crecs = [r for r in recs if r['code'] == code]
        ps = sorted({int(r['P']) for r in crecs if r.get('P')})
        events = cache.get(code, {}).get('events') or []
        cache_events = [dict(date=fmt(int(d)), mult=m) for d, m in events]
        win = set()
        p0 = datetime.strptime(fmt(ps[0]), '%Y-%m-%d')
        for i in range(-120, 121):
            win.add((p0 + timedelta(days=i)).strftime('%Y-%m-%d'))
        for e in cache_events:
            d = datetime.strptime(e['date'], '%Y-%m-%d')
            for i in range(-90, 91):
                win.add((d + timedelta(days=i)).strftime('%Y-%m-%d'))
        specs.append(dict(code=code,
                          stratum='EMPTY' if code in LIST['next_batch_draft']['EMPTY_all'] else 'OK',
                          smoke=(code in smoke),
                          P=fmt(ps[0]), all_Ps=[fmt(p) for p in ps],
                          cache_events=cache_events, window=sorted(win)))

    out = dict(version='batch_d_v1.1_frozen',
               frozen_at='2026-10-13',
               matching_rule='精确日期匹配：外部除权日==缓存事件日；±90天仅为检索范围；'
                             '非零日期差单列 date_diff_pending（用户裁决）',
               population_binding=dict(path=str(POP.relative_to(HERE)), sha256=pop_sha),
               cache_binding=dict(path=str(CACHE.relative_to(HERE)), sha256=cache_sha),
               code_list_binding=dict(path='next_batch_code_list_draft.json',
                                      sha256=hashlib.sha256(
                                          (HERE / 'next_batch_code_list_draft.json')
                                          .read_bytes()).hexdigest()),
               smoke_codes=smoke,
               code_specs=specs)
    (HERE / 'batch_d_design.json').write_text(
        json.dumps(out, ensure_ascii=False, indent=1), encoding='utf-8')
    print(f'codes={len(codes)} smoke={smoke} '
          f'EMPTY={sum(1 for s in specs if s["stratum"]=="EMPTY")} '
          f'OK={sum(1 for s in specs if s["stratum"]=="OK")}')
    print('written batch_d_design.json')


if __name__ == '__main__':
    main()
