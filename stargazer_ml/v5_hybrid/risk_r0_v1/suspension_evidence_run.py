"""suspension_evidence_run.py — 批次 a：600733/300586 2018 停复牌公告取证（用户 #18.7 批准）。

边界：≤6 HTTP 请求 / ≤1 MiB 响应 / ≤1 MiB 落盘（全部计入：列表、原文、重试）；
新批次目录+共享计数器+原始留证；失败/触线即停；不续跑、不扩围。
未找到或预算不足 → 报告"未取得充分证据"，不认定"没有停牌"。
确认须摘录原文实际停复牌日期+股票身份+链接；公告发布日期≠生效日。
即使确认停牌也不自动恢复风险标签。
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
BATCH = HERE / 'independent_fetch' / 'batch_a_suspension_20181013'
CAP_R, CAP_B = 6, 1024 * 1024
KEYWORDS = ('停牌', '复牌', '暂停上市', '恢复上市', '撤销退市风险')
REPORT_RESERVE = 16 * 1024


def extract_pdf_text(pdf_bytes: bytes) -> str:
    try:
        from pypdf import PdfReader
        import io
        return '\n'.join((p.extract_text() or '') for p in PdfReader(io.BytesIO(pdf_bytes)).pages)
    except Exception as exc:
        return f'[PDF_TEXT_EXTRACTION_FAILED:{type(exc).__name__}]'


def find_dates(text: str) -> dict:
    out = {}
    for pat, key in [(r'(\d{4})\s*年\s*(\d{1,2})\s*月\s*(\d{1,2})\s*日[^\n]{0,20}停牌', 'suspension'),
                     (r'(\d{4})\s*年\s*(\d{1,2})\s*月\s*(\d{1,2})\s*日[^\n]{0,20}复牌', 'resumption'),
                     (r'停牌[^\n]{0,30}?(\d{4})\s*年\s*(\d{1,2})\s*月\s*(\d{1,2})\s*日', 'suspension2'),
                     (r'复牌[^\n]{0,30}?(\d{4})\s*年\s*(\d{1,2})\s*月\s*(\d{1,2})\s*日', 'resumption2')]:
        m = re.search(pat, text)
        if m:
            out[key] = f'{m.group(1)}-{int(m.group(2)):02d}-{int(m.group(3)):02d}'
    return out


def main() -> int:
    BATCH.mkdir(parents=True, exist_ok=True)
    if batch_finalized(BATCH):
        raise SystemExit('批次已终结，拒绝再跑（获书面授权后移走 stop_record.json 留痕）')
    ledger = [dict(event='batch_open', batch_id=uuid.uuid4().hex[:12], scope='a_suspension_only')]
    req, nb, disk = FetchCounter(CAP_R), ByteCounter(CAP_B), ByteCounter(CAP_B)
    disk.consume(REPORT_RESERVE)
    client = CninfoAnnClient(req, nb, timeout_s=10.0, rate_limit_s=1.0,
                             max_retries=1, disk_counter=disk)
    stop = 'completed'
    findings = {}

    for code in ('600733', '300586'):
        rec = dict(code=code, list_total=None, candidates=[], fetched_texts=[],
                   evidence='未取得充分证据')
        try:
            out = client.query_ann_list(code, '2018-01-01~2018-12-31',
                                        page_size=30, archive_dir=str(BATCH))
            anns = out.get('announcements') or []
            rec['list_total'] = out.get('totalAnnouncement')
            hits = []
            for a in anns:
                title = re.sub(r'<[^>]+>', '', str(a.get('announcementTitle', '')))
                if any(k in title for k in KEYWORDS):
                    hits.append(dict(title=title,
                                     pub_ms=a.get('announcementTime'),
                                     url=a.get('adjunctUrl')))
            rec['candidates'] = hits
            ledger.append(dict(event='list_ok', code=code, total=rec['list_total'],
                               keyword_hits=len(hits)))
        except (FetchBudgetExceeded, ByteBudgetExceeded) as exc:
            stop = f'budget_trip:{type(exc).__name__}'
            ledger.append(dict(event='budget_trip', code=code))
            findings[code] = rec
            break
        except Exception as exc:
            stop = f'list_fail:{type(exc).__name__}'
            ledger.append(dict(event='list_fail', code=code, err=str(exc)[:120]))
            findings[code] = rec
            break
        # 预算内取最相关原文：复牌类公告信息密度最高（含停牌起始+复牌日）
        hits_sorted = sorted(hits, key=lambda h: ('复牌' not in h['title'] and '恢复上市' not in h['title'],
                                                  -(h['pub_ms'] or 0)))
        for h in hits_sorted[:1]:
            try:
                pdf = client.fetch_ann_text(h['url'], archive_dir=str(BATCH))
                text = extract_pdf_text(pdf)
                dates = find_dates(text)
                rec['fetched_texts'].append(dict(
                    title=h['title'], url=h['url'], bytes=len(pdf),
                    text_head=text[:600], dates=dates))
                ledger.append(dict(event='text_ok', code=code, title=h['title'][:40]))
            except (FetchBudgetExceeded, ByteBudgetExceeded) as exc:
                stop = f'budget_trip:{type(exc).__name__}'
                ledger.append(dict(event='budget_trip_text', code=code))
                break
            except Exception as exc:
                ledger.append(dict(event='text_fail', code=code, err=str(exc)[:80]))
        if stop != 'completed':
            break
        if rec['candidates'] and rec['fetched_texts']:
            rec['evidence'] = 'candidates_and_text_obtained'
        elif rec['candidates']:
            rec['evidence'] = 'candidates_only_text_failed'
        findings[code] = rec

    (BATCH / 'findings.json').write_text(
        json.dumps(findings, ensure_ascii=False, indent=1), encoding='utf-8')
    final = dict(stop_reason=stop, requests_used=req.used, requests_cap=CAP_R,
                 response_bytes=nb.used, response_bytes_cap=CAP_B,
                 disk_bytes=disk.used, disk_bytes_cap=CAP_B,
                 note='批次 a 单次运行账本；额度不随重启重置')
    with (BATCH / 'stop_record.json').open('xb') as f:
        f.write(json.dumps(final, ensure_ascii=False, indent=1).encode('utf-8'))
    with (BATCH / 'request_ledger.json').open('xb') as f:
        f.write(json.dumps(ledger, ensure_ascii=False, indent=1).encode('utf-8'))
    L = ['# 停复牌证据批次 a 报告（2018 两例）', '',
         f"- 账本：请求 {final['requests_used']}/{CAP_R}｜响应 {nb.used} B/1 MiB｜落盘 {disk.used} B/1 MiB（含 16KB 报告预留）",
         '- 口径：公告发布日期≠生效日；确认以原文实际停/复牌日期为准；未取得=不认定无停牌', '']
    for code, r in findings.items():
        L.append(f"## {code}：{r['evidence']}（列表 {r['list_total']} 条，命中 {len(r['candidates'])}）")
        for c in r['candidates'][:5]:
            L.append(f"- {c['title']}（发布 {datetime.fromtimestamp((c['pub_ms'] or 0)/1000).date() if c['pub_ms'] else '—'}）{c['url']}")
        for t in r['fetched_texts']:
            L.append(f"  原文摘录（{t['bytes']}B）：{t['text_head'][:200]}")
            L.append(f"  日期识别：{t['dates']}")
        L.append('')
    (BATCH / 'report.md').write_text('\n'.join(L), encoding='utf-8')
    print(json.dumps(final, ensure_ascii=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
