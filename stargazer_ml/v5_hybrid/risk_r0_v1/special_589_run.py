"""special_589_run.py — 批次e：600589 重整除权原文取证（用户单独批准，2026-10-13）。

范围固定 3 份公告（实施公告/公式专项意见/计算结果专项意见）。
硬边界：≤6 次实际 HTTP 请求、零重试（详情页解析与正文下载全部计数，失败/触线即停）；
响应与新增落盘各 ≤2,097,152 B（含报告/索引/停机留因，预留 96KiB）；
仅巨潮、新批次目录、不切源、不续跑；与 000564 不共享额度（000564 继续挂起）。
从正文核对：公式/参数/计算结果/生效日期，标明页码与原文位置；
取不到或解释不完整保持待核；不改缓存、不泛化。
运行前须通过 test_special_589_precheck.py（零网络）。
"""
from __future__ import annotations

import hashlib
import io
import json
import os
import re
import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cninfo_ann_client import CninfoAnnClient
from independent_fetch_client import (ByteBudgetExceeded, ByteCounter,
                                      FetchBudgetExceeded, FetchCounter)
from independent_fetch_run import batch_finalized
from special_564_run import clean_url, find_adjunct

HERE = Path(__file__).resolve().parent
BATCH = Path(os.environ.get('BATCH_589_DIR', str(
    HERE / 'independent_fetch' / 'batch_e_589_special')))
CAP_R, CAP_B = 6, 2 * 1024 * 1024
REPORT_RESERVE = 96 * 1024

RAW_URLS = [
    dict(id='1218734245', date='2023-12-27',
         title='关于重整计划资本公积金转增股本事项实施的公告',
         url='http://www.cninfo.com.cn/new/disclosure/detail?stockCode=600589'
             '&announcementId=1218734245&orgId=gssh0600589&announcementTime=2023-12-27'),
    dict(id='1218734247', date='2023-12-27',
         title='中金：调整资本公积金转增股本除权参考价格计算公式的专项意见',
         url='http://www.cninfo.com.cn/new/disclosure/detail?stockCode=600589'
             '&announcementId=1218734247&orgId=gssh0600589&announcementTime=2023-12-27'),
    dict(id='1218747449', date='2023-12-28',
         title='中金：调整资本公积金转增股本除权参考价格的计算结果的专项意见',
         url='http://www.cninfo.com.cn/new/disclosure/detail?stockCode=600589'
             '&announcementId=1218747449&orgId=gssh0600589&announcementTime=2023-12-28'),
]
TARGETS = [dict(t, url=clean_url(t['url'])) for t in RAW_URLS]  # 规范化冻结


def extract_fields(pdf: bytes) -> dict:
    """按页抽取：公式 / 参数 / 计算结果 / 生效日期，均带页码与原文位置。"""
    from pypdf import PdfReader
    try:
        pages = [p.extract_text() or '' for p in PdfReader(io.BytesIO(pdf)).pages]
    except Exception as exc:
        return dict(error=f'[PDF_TEXT_FAILED:{type(exc).__name__}]')
    full = '\n'.join(pages)
    f = dict(page_count=len(pages), text_len=len(full))
    locs = []
    pats = [
        ('formula', r'除权[^。；\n]{0,30}(?:参考)?价格[^。；\n]{0,40}公式[^。；\n]{0,200}'),
        ('computed_ref_price', r'除权[^。；\n]{0,30}(?:参考)?价格(?:计算结果)?[为是][:：]?\s*(\d+\.?\d*)\s*元'),
        ('effective_date', r'(\d{4})\s*年\s*(\d{1,2})\s*月\s*(\d{1,2})\s*日[^\n。；]{0,20}(?:生效|除权|实施)'),
        ('ex_date', r'除权日[期为][:：]?(\d{4})\s*年\s*(\d{1,2})\s*月\s*(\d{1,2})\s*日'),
        ('bonus_share_param', r'每\s*10\s*股[^。；\n]{0,30}转增\s*(\d+\.?\d*)\s*股'),
        ('transfer_price', r'受让价格[^。；\n]{0,20}(\d+\.?\d*)\s*元'),
    ]
    for key, pat in pats:
        for pi, text in enumerate(pages):
            for m in re.finditer(pat, text):
                locs.append(dict(field=key, page=pi + 1,
                                 excerpt=m.group(0)[:220],
                                 groups=list(m.groups())))
                break
            else:
                continue
            break
    f['evidence'] = locs
    f['share_param_samples'] = re.findall(r'(\d+\.?\d*)\s*股', full[:4000])[:10]
    return f


def main() -> int:
    BATCH.mkdir(parents=True, exist_ok=True)
    if batch_finalized(BATCH):
        raise SystemExit('批次已终结，拒绝再跑')
    manifest = dict(batch='batch_e_589_special', approved='2026-10-13',
                    cap_requests=CAP_R, cap_bytes=CAP_B, max_retries=0,
                    code_sha256=hashlib.sha256(
                        Path(__file__).read_bytes()).hexdigest(),
                    targets=[dict(id=t['id'], title=t['title'], date=t['date'],
                                  url=t['url'],
                                  url_sha256=hashlib.sha256(t['url'].encode()).hexdigest())
                             for t in TARGETS])
    ledger = [dict(event='batch_open', batch_id=uuid.uuid4().hex[:12],
                   scope='special_589', code_sha256=manifest['code_sha256'])]
    req = FetchCounter(CAP_R)
    nb, disk = ByteCounter(CAP_B), ByteCounter(CAP_B)
    disk.consume(REPORT_RESERVE)
    client = CninfoAnnClient(req, nb, timeout_s=10.0, rate_limit_s=1.0,
                             max_retries=0, disk_counter=disk)
    stop = 'completed'
    results = []

    for item in TARGETS:
        rec = dict(id=item['id'], title=item['title'], date=item['date'],
                   detail_url=item['url'], adjunct=None, pdf_bytes=None,
                   fields={}, status='pending')
        try:
            html = client.fetch_ann_text(item['url'], archive_dir=str(BATCH))
            adj = find_adjunct(html)
            rec['adjunct'] = adj
            ledger.append(dict(event='detail_ok', id=item['id'],
                               adjunct=adj, html_bytes=len(html)))
            if not adj:
                stop = 'no_adjunct'
                rec['status'] = 'NO_ADJUNCT_IN_DETAIL_PAGE'
                results.append(rec)
                break
            pdf = client.fetch_ann_text(adj, archive_dir=str(BATCH))
            rec['pdf_bytes'] = len(pdf)
            f = extract_fields(pdf)
            rec['fields'] = f
            if 'error' in f:
                stop = 'text_extract_failed'
                rec['status'] = 'TEXT_UNOBTAINED'
                results.append(rec)
                break
            rec['status'] = 'TEXT_OBTAINED'
            ledger.append(dict(event='pdf_ok', id=item['id'],
                               pdf_bytes=len(pdf), evidence=len(f.get('evidence', []))))
        except (FetchBudgetExceeded, ByteBudgetExceeded) as exc:
            stop = f'budget_trip:{type(exc).__name__}'
            ledger.append(dict(event='budget_trip', id=item['id']))
            rec['status'] = 'BUDGET_TRIP'
            results.append(rec)
            break
        except Exception as exc:
            stop = f'fetch_fail:{type(exc).__name__}'
            ledger.append(dict(event='fetch_fail', id=item['id'], err=str(exc)[:100]))
            rec['status'] = f'FETCH_FAIL:{type(exc).__name__}'
            results.append(rec)
            break
        results.append(rec)

    (BATCH / 'findings.json').write_text(
        json.dumps(results, ensure_ascii=False, indent=1), encoding='utf-8')
    (BATCH / 'manifest.json').write_text(
        json.dumps(manifest, ensure_ascii=False, indent=1), encoding='utf-8')
    final = dict(stop_reason=stop, requests_used=req.used, requests_cap=CAP_R,
                 response_bytes=nb.used, response_bytes_cap=CAP_B,
                 disk_bytes=disk.used, disk_bytes_cap=CAP_B,
                 note='批次e单次运行账本；与000564不共享额度；额度不随重启重置')
    with (BATCH / 'stop_record.json').open('xb') as f:
        f.write(json.dumps(final, ensure_ascii=False, indent=1).encode('utf-8'))
    with (BATCH / 'request_ledger.json').open('xb') as f:
        f.write(json.dumps(ledger, ensure_ascii=False, indent=1).encode('utf-8'))
    L = ['# 600589 重整除权取证报告', '',
         f"- 账本：请求 {req.used}/{CAP_R}｜响应 {nb.used} B/2MiB｜落盘 {disk.used} B/2MiB",
         '- 只核对：公式/参数/计算结果/生效日期（带页码与原文位置）；不改缓存、不泛化', '']
    for r in results:
        L.append(f"## {r['date']} {r['title'][:40]} — {r['status']}")
        for ev in r.get('fields', {}).get('evidence', []):
            L.append(f"- p{ev['page']} [{ev['field']}] {ev['excerpt']}"
                     + (f"｜捕获:{ev['groups']}" if ev['groups'] else ''))
    (BATCH / 'report.md').write_text('\n'.join(L), encoding='utf-8')
    print(json.dumps(final, ensure_ascii=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
