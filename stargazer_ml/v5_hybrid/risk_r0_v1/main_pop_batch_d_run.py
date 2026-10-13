"""main_pop_batch_d_run.py — 批次d：主人口30码公司行为对拍（用户批准，2026-10-13）。

硬边界（请求单 v1.1 冻结）：总请求≤30、全批零重试、失败/触线即停不替补；
累计响应≤2,621,440 B；新增落盘≤3,145,728 B（含报告/索引/停机留因，预留96KiB）；
仅新浪、新批次目录、不翻页、不切源、不重启续抓。
冒烟=清单前2码（688047/688180），零重试，页面直接复用于正式分析，不做32请求。

比对口径（用户裁决）：精确日期匹配（外部除权日==缓存事件日）；
±90天仅为检索范围；非零日期差单列 date_diff_pending；
日期命中≠类型/实施状态/因子匹配（四态再分）；不置覆盖、不生成标签、不训练。
"""
from __future__ import annotations

import json
import sys
import uuid
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from independent_fetch_client import (ByteBudgetExceeded, ByteCounter,
                                      FetchBudgetExceeded, FetchCounter)
from independent_fetch_run import batch_finalized
from sina_div_client import SinaDivClient, parse_bonus_page

HERE = Path(__file__).resolve().parent
BATCH = HERE / 'independent_fetch' / 'batch_d_mainpop_30'
CAP_R = 30
CAP_B = 2_621_440          # 累计响应（用户签署值）
CAP_DISK = 3_145_728       # 新增落盘（用户签署值）
REPORT_RESERVE = 96 * 1024
DESIGN = json.loads((HERE / 'batch_d_design.json').read_text(encoding='utf-8'))
SPECS = {s['code']: s for s in DESIGN['code_specs']}
CODES = [s['code'] for s in DESIGN['code_specs']]  # 顺序冻结：冒烟2 + 其余28


def num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def classify(ext, cache_mult):
    """命中项的类型/参数状态四态（日期已精确命中，不代表这些已匹配）。"""
    if ext['table'] == 'allotment':
        return 'TYPE_ALLOTMENT_NOT_COMPARABLE', '配股无进度列按候选保留；因子需配股价与前收（未核销）'
    simple = 1.0 + ((ext['送'] or 0) + (ext['转'] or 0)) / 10.0
    if abs(simple - cache_mult) < 0.01:
        return 'TYPE_PARAMS_CONSISTENT_NO_CASH', None
    if (ext['送'] or 0) == 0 and (ext['转'] or 0) == 0 and (ext['派'] or 0) > 0:
        return 'CASH_ONLY_FACTOR_NOT_COMPARABLE', '纯派息；方向相容(mult>1)，数值验证需前收口径（未核销）'
    return 'MIXED_PARAM_PENDING', ('送转+派混合：简单因子%.4f vs 缓存%s；差额或来自派息项，'
                                   '本批不核销，后置另案' % (simple, cache_mult))


def analyze(code, pg, spec):
    """精确日期匹配 + 检索范围关联。返回四段报告结构。"""
    win = set(spec['window'])
    ext_impl = []
    for kind, key_ex in (('bonus', '除权除息日'), ('allotment', '除权日')):
        for r in (pg.get(kind) or []):
            prog = (r.get('进度') or '').strip()
            if kind == 'bonus' and prog != '实施':
                continue
            ex = str(r.get(key_ex) or '')[:10]
            if ex and ex in win:
                ext_impl.append(dict(table=kind, ex_date=ex,
                                     公告日期=str(r.get('公告日期'))[:10],
                                     进度=prog or None,
                                     送=num(r.get('送股(股)')),
                                     转=num(r.get('转增(股)')),
                                     派=num(r.get('派息(税前)(元)')),
                                     date_basis=('sina_allotment_no_progress'
                                                 if kind == 'allotment' else 'sina_returned_only')))
    cache_days = {e['date']: e['mult'] for e in spec['cache_events']}
    ext_days = {e['ex_date'] for e in ext_impl}

    hits, date_diff, outside = [], [], []
    for e in ext_impl:
        if e['ex_date'] in cache_days:
            st, note = classify(e, cache_days[e['ex_date']])
            hits.append(dict(external=e, status=st, note=note))
        else:
            d = datetime.strptime(e['ex_date'], '%Y-%m-%d')
            near = [c for c in cache_days
                    if abs((datetime.strptime(c, '%Y-%m-%d') - d).days) <= 90]
            if near:
                date_diff.append(dict(external=e, near_cache_dates=sorted(near),
                                      status='DATE_DIFF_PENDING',
                                      note='±90检索范围内有缓存事件但日期不精确相等，未裁决'))
            else:
                outside.append(dict(external=e, status='OUTSIDE_CACHE_CANDIDATE',
                                    note='检索范围内无任何缓存事件：缓存漏记候选，未裁决'))
    not_returned = [dict(cache_date=c, cache_mult=m, status='NOT_RETURNED_CANDIDATE',
                         note='缓存事件日无精确同日外部记录（表外事件/停牌/漏抓候选，未裁决）')
                    for c, m in sorted(cache_days.items()) if c not in ext_days]

    ex_all = sorted(str(r.get('除权除息日'))[:10] for r in (pg.get('bonus') or [])
                    if str(r.get('除权除息日') or '')[:4].isdigit())
    history = dict(
        bonus_rows=len(pg.get('bonus') or []),
        allotment_rows=len(pg.get('allotment') or []),
        bonus_progress_dist=_dist(pg.get('bonus') or []),
        bonus_ex_date_range=[ex_all[0], ex_all[-1]] if ex_all else None)
    return dict(code=code, stratum=spec['stratum'], smoke=spec['smoke'], P=spec['P'],
                window_n=len(win), history=history,
                date_hits=hits, date_diff_pending=date_diff,
                outside_cache=outside, not_returned=not_returned)


def _dist(rows):
    d = {}
    for r in rows:
        p = str(r.get('进度') or 'NULL')
        d[p] = d.get(p, 0) + 1
    return d


def main() -> int:
    BATCH.mkdir(parents=True, exist_ok=True)
    if batch_finalized(BATCH):
        raise SystemExit('批次已终结，拒绝再跑')
    ledger = [dict(event='batch_open', batch_id=uuid.uuid4().hex[:12], scope='batch_d_mainpop_30',
                   population_sha=DESIGN['population_binding']['sha256'][:16],
                   cache_sha=DESIGN['cache_binding']['sha256'][:16],
                   code_list_sha=DESIGN['code_list_binding']['sha256'][:16],
                   matching_rule=DESIGN['matching_rule'])]
    req = FetchCounter(CAP_R)
    nb, disk = ByteCounter(CAP_B), ByteCounter(CAP_DISK)
    disk.consume(REPORT_RESERVE)
    client = SinaDivClient(req, nb, timeout_s=10.0, rate_limit_s=1.0,
                           max_retries=0, disk_counter=disk)  # 全批零重试
    stop = 'completed'
    results = []

    for code in CODES:
        spec = SPECS[code]
        try:
            raw = client.fetch_bonus_page(code, archive_dir=str(BATCH))
            pg = parse_bonus_page(raw)
            ledger.append(dict(event='page_ok', code=code, bytes=len(raw)))
        except (FetchBudgetExceeded, ByteBudgetExceeded) as exc:
            stop = f'budget_trip:{type(exc).__name__}'
            ledger.append(dict(event='budget_trip', code=code))
            results.append(dict(code=code, status='BUDGET_TRIP'))
            break
        except Exception as exc:
            stop = f'fetch_fail:{type(exc).__name__}'
            ledger.append(dict(event='fetch_fail', code=code, err=str(exc)[:100]))
            results.append(dict(code=code, status=f'FETCH_FAIL:{type(exc).__name__}'))
            break
        rec = analyze(code, pg, spec)
        rec['status'] = 'OK'
        ledger.append(dict(event='analyzed', code=code,
                           hits=len(rec['date_hits']),
                           date_diff=len(rec['date_diff_pending']),
                           outside=len(rec['outside_cache']),
                           not_returned=len(rec['not_returned'])))
        results.append(rec)

    (BATCH / 'diff_table.json').write_text(
        json.dumps(results, ensure_ascii=False, indent=1), encoding='utf-8')
    final = dict(stop_reason=stop, requests_used=req.used, requests_cap=CAP_R,
                 response_bytes=nb.used, response_bytes_cap=CAP_B,
                 disk_bytes=disk.used, disk_bytes_cap=CAP_DISK,
                 smoke_codes=DESIGN['smoke_codes'],
                 note='批次d单次运行账本；额度不随重启重置；精确日期匹配口径')
    with (BATCH / 'stop_record.json').open('xb') as f:
        f.write(json.dumps(final, ensure_ascii=False, indent=1).encode('utf-8'))
    with (BATCH / 'request_ledger.json').open('xb') as f:
        f.write(json.dumps(ledger, ensure_ascii=False, indent=1).encode('utf-8'))
    write_report(results, final)
    print(json.dumps(final, ensure_ascii=False))
    return 0


def write_report(results, final):
    L = ['# 批次d 主人口30码对拍报告（精确日期匹配口径）', '',
         f"- 账本：请求 {final['requests_used']}/{final['requests_cap']}｜"
         f"响应 {final['response_bytes']} B/{final['response_bytes_cap']}｜"
         f"落盘 {final['disk_bytes']} B/{final['disk_bytes_cap']}",
         f"- 冒烟码（页面复用，未另查）：{final['smoke_codes']}",
         '- 口径：±90天仅为检索范围；日期匹配=精确一致；命中≠类型/状态/因子匹配；'
         '非零日期差单列待裁决', '']
    for r in results:
        if r.get('status') != 'OK':
            L.append(f"## {r['code']} — {r['status']}")
            continue
        L.append(f"## {r['code']} [{r['stratum']}]{'[冒烟]' if r['smoke'] else ''} P={r['P']}")
        h = r['history']
        L.append(f"- 全历史：分红{h['bonus_rows']}行 配股{h['allotment_rows']}行 "
                 f"进度{h['bonus_progress_dist']} 除权范围{h['bonus_ex_date_range']}")
        L.append(f"- 窗口诊断：检索窗{r['window_n']}天")
        L.append(f"- 日期命中：{len(r['date_hits'])}（"
                 + '，'.join(f"{x['status']}" for x in r['date_hits']) + '）')
        for x in r['date_hits']:
            e = x['external']
            L.append(f"    {e['ex_date']} {e['table']} 送{e['送']}转{e['转']}派{e['派']}"
                     + (f"｜{x['note']}" if x.get('note') else ''))
        for x in r['date_diff_pending']:
            e = x['external']
            L.append(f"- 日期差待裁决：外部{e['ex_date']}({e['table']}) vs 缓存{x['near_cache_dates']}")
        for x in r['outside_cache']:
            e = x['external']
            L.append(f"- 缓存外候选：{e['ex_date']} {e['table']} 公告{e['公告日期']}（未裁决）")
        for x in r['not_returned']:
            L.append(f"- 外部未返回：缓存{x['cache_date']} mult={x['cache_mult']}（未裁决）")
    (BATCH / 'report.md').write_text('\n'.join(L), encoding='utf-8')


if __name__ == '__main__':
    raise SystemExit(main())
