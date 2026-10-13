"""special_564_run.py — 批次①：000564 重整除权原文取证（用户批准，≤4 请求/2MiB/2MiB）。

预算现实：详情页解析与 PDF 下载各计 1 请求 → 4 次额度只够 2 篇（详情页+PDF）。
取价值最高的 2 篇：计算结果法律意见书 + 实施公告；另 2 篇如实报告"未取得正文"。
标题/网页壳/空响应 ≠ 正文证据。只核对调整公式/参数/生效日/缓存乘子解释；不改缓存。
"""
from __future__ import annotations

import json
import re
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cninfo_ann_client import CninfoAnnClient
from independent_fetch_client import (ByteBudgetExceeded, ByteCounter,
                                      FetchBudgetExceeded, FetchCounter)
from independent_fetch_run import batch_finalized

HERE = Path(__file__).resolve().parent
import os as _os
BATCH = Path(_os.environ.get('BATCH_564_DIR', str(HERE / 'independent_fetch' / 'batch_564_special_v2')))
CAP_R, CAP_B = 4, 2 * 1024 * 1024
REPORT_RESERVE = 16 * 1024
LINKS = json.loads((HERE / 'independent_fetch' / 'batch_b_dividend_probe'
                    / 'special_action_564_links.json').read_text(encoding='utf-8'))
# 优先级：计算结果法律意见书(#2) → 实施公告(#1)
TARGETS = [LINKS[1], LINKS[0]]


def extract_pdf_text(pdf: bytes) -> str:
    try:
        from pypdf import PdfReader
        return '\n'.join((p.extract_text() or '') for p in PdfReader(io.BytesIO(pdf)).pages)
    except Exception as exc:
        return f'[PDF_TEXT_FAILED:{type(exc).__name__}]'


import io


def clean_url(url: str) -> str:
    """冻结清单中的 URL 可能含字面空格（如 announcementTime=2021-12-27 00:00:00）。

    http.client 拒绝含控制/空白字符的 URL（批次①v2 烧毁根因）。统一转 %20。
    """
    return url.replace(' ', '%20')


def find_adjunct(html: bytes) -> str | None:
    m = re.search(rb'(finalpage/[^"\']+?\.PDF)', html, re.I)
    return m.group(1).decode() if m else None


def main() -> int:
    BATCH.mkdir(parents=True, exist_ok=True)
    if batch_finalized(BATCH):
        raise SystemExit('批次已终结，拒绝再跑')
    ledger = [dict(event='batch_open', batch_id=uuid.uuid4().hex[:12], scope='special_564')]
    req = FetchCounter(CAP_R)
    nb, disk = ByteCounter(CAP_B), ByteCounter(CAP_B)
    disk.consume(REPORT_RESERVE)
    client = CninfoAnnClient(req, nb, timeout_s=10.0, rate_limit_s=1.0,
                             max_retries=1, disk_counter=disk)
    stop = 'completed'
    results = []

    for item in TARGETS:
        rec = dict(title=item['title'], date=item['date'], detail_url=item['url'],
                   adjunct=None, pdf_bytes=None, text=None, fields={}, status='pending')
        # 批次①v2 烧毁根因：冻结清单里的 URL 含字面空格（announcementTime=... 00:00:00），
        # http.client 拒绝含控制字符的 URL。统一清洗为 %20，清洗本身零网络。
        url = clean_url(item['url'])
        try:
            html = client.fetch_ann_text(url, archive_dir=str(BATCH))
            adj = find_adjunct(html)
            rec['adjunct'] = adj
            ledger.append(dict(event='detail_ok', title=item['title'][:30],
                               adjunct=adj, html_bytes=len(html)))
            if not adj:
                rec['status'] = 'NO_ADJUNCT_IN_DETAIL_PAGE'
                results.append(rec)
                continue
            pdf = client.fetch_ann_text(adj, archive_dir=str(BATCH))
            rec['pdf_bytes'] = len(pdf)
            text = extract_pdf_text(pdf)
            rec['text'] = text[:4000]
            if text.startswith('[PDF_TEXT_FAILED') or len(text.strip()) < 50:
                rec['status'] = 'TEXT_UNOBTAINED'
                results.append(rec)
                continue
            # 核销字段：调整公式 / 参数 / 计算结果数值 / 生效日期
            f = {}
            m = re.search(r'除权参考价格[^。]{0,120}公式[^。]{0,200}', text)
            f['formula_excerpt'] = m.group(0)[:300] if m else None
            m = re.search(r'除权\(息\)参考价格[为是][:：]?\s*(\d+\.?\d*)\s*元', text)
            f['computed_ref_price'] = m.group(1) if m else None
            for pat, key in [(r'(\d{4})\s*年\s*(\d{1,2})\s*月\s*(\d{1,2})\s*日[^\n]{0,30}生效', 'effective_date'),
                             (r'生效日期[为是][:：]?(\d{4})\s*年\s*(\d{1,2})\s*月\s*(\d{1,2})\s*日', 'effective_date2'),
                             (r'除权日[期为][:：]?(\d{4})\s*年\s*(\d{1,2})\s*月\s*(\d{1,2})\s*日', 'ex_date')]:
                mm = re.search(pat, text)
                if mm:
                    f[key] = f'{mm.group(1)}-{int(mm.group(2)):02d}-{int(mm.group(3)):02d}'
            nums = re.findall(r'(\d+\.?\d*)\s*股', text[:3000])
            f['share_param_samples'] = nums[:10]
            rec['fields'] = f
            rec['status'] = 'TEXT_OBTAINED'
            ledger.append(dict(event='pdf_ok', title=item['title'][:30],
                               pdf_bytes=len(pdf), text_len=len(text)))
        except (FetchBudgetExceeded, ByteBudgetExceeded) as exc:
            stop = f'budget_trip:{type(exc).__name__}'
            ledger.append(dict(event='budget_trip', title=item['title'][:30]))
            rec['status'] = f'BUDGET_TRIP'
            results.append(rec)
            break
        except Exception as exc:
            # 失败即停（用户 2026-10-13 裁决）：首篇有限重试后仍失败 → 终局记为
            # fetch_fail，不再请求下一篇，绝不能写成 completed。
            stop = f'fetch_fail:{type(exc).__name__}'
            ledger.append(dict(event='fetch_fail', title=item['title'][:30],
                               err=str(exc)[:100]))
            rec['status'] = f'FETCH_FAIL:{type(exc).__name__}'
            results.append(rec)
            break
        results.append(rec)
        if stop != 'completed':
            break

    # 未取的两篇如实登记
    for item in LINKS[2:]:
        results.append(dict(title=item['title'], date=item['date'],
                            detail_url=item['url'], status='NOT_FETCHED_BUDGET',
                            note='4 请求额度内未取得正文（详情页+PDF 各计 1 请求）'))

    (BATCH / 'findings.json').write_text(
        json.dumps(results, ensure_ascii=False, indent=1), encoding='utf-8')
    final = dict(stop_reason=stop, requests_used=req.used, requests_cap=CAP_R,
                 response_bytes=nb.used, response_bytes_cap=CAP_B,
                 disk_bytes=disk.used, disk_bytes_cap=CAP_B,
                 note='批次①单次运行账本；额度不随重启重置')
    with (BATCH / 'stop_record.json').open('xb') as f:
        f.write(json.dumps(final, ensure_ascii=False, indent=1).encode('utf-8'))
    with (BATCH / 'request_ledger.json').open('xb') as f:
        f.write(json.dumps(ledger, ensure_ascii=False, indent=1).encode('utf-8'))
    L = ['# 000564 特殊事件取证报告', '',
         f"- 账本：请求 {req.used}/{CAP_R}｜响应 {nb.used} B/2 MiB｜落盘 {disk.used} B/2 MiB",
         '- 只核对：调整公式/参数/生效日期/缓存乘子解释；不改缓存、不泛化重整规则', '']
    for r in results:
        L.append(f"## {r['date']} {r['title'][:40]} — {r['status']}")
        if r.get('fields'):
            for k, v in r['fields'].items():
                L.append(f'- {k}: {v}')
        if r.get('text'):
            L.append(f"  正文摘录：{r['text'][:300]}")
    (BATCH / 'report.md').write_text('\n'.join(L), encoding='utf-8')
    print(json.dumps(final, ensure_ascii=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
