"""main_pop_probe_run.py — 批次②：主人口九码双向差集验证（用户批准，9请求/3MiB/3MiB）。

双向先称"缓存外记录/外部未返回记录"，核对类型/状态/生效日/窗口后再裁决。
重试占 9 次额度 → max_retries=0，失败即停；不替补股票。不置覆盖、不生成标签。
"""
from __future__ import annotations

import json
import sys
import uuid
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from independent_fetch_client import (ByteBudgetExceeded, ByteCounter,
                                      FetchBudgetExceeded, FetchCounter)
from independent_fetch_run import batch_finalized
from sina_div_client import SinaDivClient, parse_bonus_page

HERE = Path(__file__).resolve().parent
BATCH = HERE / 'independent_fetch' / 'batch_c_mainpop_probe'
CAP = 9
CAP_B = 3 * 1024 * 1024
DESIGN = json.loads((HERE / 'main_population_sample_design_v2.json').read_text(encoding='utf-8'))
SPECS = {s['code']: s for s in DESIGN['code_specs']}


def main() -> int:
    BATCH.mkdir(parents=True, exist_ok=True)
    if batch_finalized(BATCH):
        raise SystemExit('批次已终结，拒绝再跑')
    ledger = [dict(event='batch_open', batch_id=uuid.uuid4().hex[:12],
                   population_sha=DESIGN['population_binding']['sha256'][:16],
                   cache_sha=DESIGN['cache_binding']['sha256'][:16])]
    req = FetchCounter(CAP)
    nb, disk = ByteCounter(CAP_B), ByteCounter(CAP_B)
    disk.consume(32 * 1024)
    client = SinaDivClient(req, nb, timeout_s=10.0, rate_limit_s=1.0,
                           max_retries=0, disk_counter=disk)  # 重试占额度→0重试
    stop = 'completed'
    results = []

    for code, spec in SPECS.items():
        rec = dict(code=code, stratum=spec['stratum'], P=spec['P'],
                   cache_events=spec['cache_events'], window=spec['window'],
                   external_implemented=[], outside_cache=[], not_returned=[], status='pending')
        try:
            raw = client.fetch_bonus_page(code, archive_dir=str(BATCH))
            pg = parse_bonus_page(raw)
            ledger.append(dict(event='page_ok', code=code, bytes=len(raw)))
        except (FetchBudgetExceeded, ByteBudgetExceeded) as exc:
            stop = f'budget_trip:{type(exc).__name__}'
            ledger.append(dict(event='budget_trip', code=code))
            rec['status'] = 'BUDGET_TRIP'
            results.append(rec)
            break
        except Exception as exc:
            stop = f'fetch_fail:{type(exc).__name__}'
            ledger.append(dict(event='fetch_fail', code=code, err=str(exc)[:100]))
            rec['status'] = f'FETCH_FAIL:{type(exc).__name__}'
            results.append(rec)
            break
        win = set(spec['window'])
        impl = []
        for kind, key_ex in (('bonus', '除权除息日'), ('allotment', '除权日')):
            for r in (pg.get(kind) or []):
                if (r.get('进度') or '').strip() != '实施':
                    continue
                ex = str(r.get(key_ex) or '')[:10]
                if ex and ex in win:
                    impl.append(dict(table=kind, 公告日期=str(r.get('公告日期'))[:10],
                                     ex_date=ex,
                                     params={k: (str(v)[:16] if v is not None else None)
                                             for k, v in r.items()}))
        rec['external_implemented'] = impl
        cache_days = {e['date'] for e in spec['cache_events']}
        ext_days = {e['ex_date'] for e in impl}
        # 缓存外记录：外部实施事件 ±90 天内无缓存事件
        for e in impl:
            d = datetime.strptime(e['ex_date'], '%Y-%m-%d')
            near = any(abs((datetime.strptime(c, '%Y-%m-%d') - d).days) <= 90
                       for c in cache_days)
            if not near:
                rec['outside_cache'].append(e)
        # 外部未返回：缓存事件 ±90 天内无外部实施记录
        for c in cache_days:
            d = datetime.strptime(c, '%Y-%m-%d')
            near = any(abs((datetime.strptime(e['ex_date'], '%Y-%m-%d') - d).days) <= 90
                       for e in ext_days)
            if not near:
                rec['not_returned'].append(c)
        rec['status'] = 'OK'
        ledger.append(dict(event='diff', code=code,
                           ext_impl=len(impl), outside=len(rec['outside_cache']),
                           not_returned=len(rec['not_returned'])))
        results.append(rec)

    (BATCH / 'diff_table.json').write_text(
        json.dumps(results, ensure_ascii=False, indent=1), encoding='utf-8')
    final = dict(stop_reason=stop, requests_used=req.used, requests_cap=CAP,
                 response_bytes=nb.used, response_bytes_cap=CAP_B,
                 disk_bytes=disk.used, disk_bytes_cap=CAP_B,
                 note='批次②单次运行账本；额度不随重启重置')
    with (BATCH / 'stop_record.json').open('xb') as f:
        f.write(json.dumps(final, ensure_ascii=False, indent=1).encode('utf-8'))
    with (BATCH / 'request_ledger.json').open('xb') as f:
        f.write(json.dumps(ledger, ensure_ascii=False, indent=1).encode('utf-8'))
    L = ['# 批次② 主人口九码双向差集报告', '',
         f"- 账本：请求 {req.used}/{CAP}｜响应 {nb.used} B｜落盘 {disk.used} B",
         '- 双向先称"缓存外记录/外部未返回记录"，裁决需另核类型/状态/生效日/窗口', '']
    for r in results:
        L.append(f"## {r['code']} [{r['stratum']}] P={r['P']} — {r['status']}")
        L.append(f"- 外部实施事件(窗内): {len(r['external_implemented'])}｜缓存外记录: {len(r['outside_cache'])}｜外部未返回: {r['not_returned']}")
        for e in r['outside_cache'][:3]:
            L.append(f"  缓存外候选: {e['公告日期']} 除权 {e['ex_date']} {e['table']}")
    (BATCH / 'report.md').write_text('\n'.join(L), encoding='utf-8')
    print(json.dumps(final, ensure_ascii=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
