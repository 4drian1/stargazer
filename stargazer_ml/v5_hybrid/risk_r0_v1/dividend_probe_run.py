"""dividend_probe_run.py — 批次 b：新浪分红配股接口可行性探查（用户 #18.9 批准）。

更正并入：①单页双表（分红+配股）一次请求；②生效日一致性只认原文/价格缺口佐证，
   非零差单列、缺依据记 DATE_UNVERIFIED，无"相容"判定；③对照分级——
   cache 锚定/空窗口只用于探查，不认证覆盖完整。
边界：4 码 5 案例；≤18 HTTP 请求；响应/落盘各 ≤3 MiB（含 32KB 报告预留）；
仅新浪；失败/触线即停；旧批次不重跑。不生成标签、不置覆盖、不训练。
"""
from __future__ import annotations

import json
import sys
import urllib.parse
import urllib.request
import uuid
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from independent_fetch_client import (ByteBudgetExceeded, ByteCounter,
                                      FetchBudgetExceeded)
from independent_fetch_run import batch_finalized
from independent_fetch_client import FetchCounter
from sina_div_client import SinaDivClient, parse_bonus_page

HERE = Path(__file__).resolve().parent
BATCH = HERE / 'independent_fetch' / 'batch_b_dividend_probe'
CAP_R, CAP_B = 18, 3 * 1024 * 1024
REPORT_RESERVE = 32 * 1024

OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def sdb_daily(code6: str) -> dict:
    """本地 stockdb 日线（零网络，不占外部请求预算）——价格缺口佐证用。"""
    import msgpack
    url = ('http://127.0.0.1:7899/?cmd=get&t='
           + urllib.parse.quote(f'日k:{code6}:*'))
    req = urllib.request.Request(url, headers={'Accept': 'application/x-msgpack'})
    d = msgpack.unpackb(OPENER.open(req, timeout=30).read(), raw=False,
                        strict_map_key=False)
    rows = (d.get('data') if isinstance(d, dict) else d) or []
    out = {}
    for r in rows:
        dd = r[1] if isinstance(r, list) else r
        out[int(dd['date'])] = dd
    return out


def price_gap_check(daily: dict, ex_date_str: str) -> dict:
    """除权日价格缺口佐证：ex_date 当日 pct_chg 与 (close/prev_close-1) 的差异=缺口。"""
    try:
        d = int(ex_date_str.replace('-', ''))
    except Exception:
        return dict(status='BAD_DATE')
    if d not in daily:
        return dict(status='NO_BAR')
    days = sorted(daily)
    i = days.index(d)
    if i == 0:
        return dict(status='NO_PREV')
    prev_close = float(daily[days[i - 1]]['close'])
    pre_close = float(daily[d].get('pre_close') or prev_close)
    o = float(daily[d]['open'])
    gap = pre_close / prev_close - 1          # 除权参考/昨收 − 1 ≈ 0 若非除权日
    return dict(status='OK', ex_date=ex_date_str,
                prev_close=round(prev_close, 3), pre_close=round(pre_close, 3),
                open=round(o, 3), implied_adj=round(gap, 4))


def main() -> int:
    BATCH.mkdir(parents=True, exist_ok=True)
    if batch_finalized(BATCH):
        raise SystemExit('批次已终结，拒绝再跑')
    ledger = [dict(event='batch_open', batch_id=uuid.uuid4().hex[:12], scope='b_dividend_probe')]
    req = FetchCounter(CAP_R)
    nb, disk = ByteCounter(CAP_B), ByteCounter(CAP_B)
    disk.consume(REPORT_RESERVE)
    client = SinaDivClient(req, nb, timeout_s=10.0, rate_limit_s=1.0,
                           max_retries=2, disk_counter=disk)
    stop = 'completed'
    pages = {}
    mechanics = {}

    codes = ['000001', '000049', '002594', '000564']   # 阶段1=000001，其余阶段2
    for code in codes:
        try:
            raw = client.fetch_bonus_page(code, archive_dir=str(BATCH))
            pages[code] = parse_bonus_page(raw)
            ledger.append(dict(event='page_ok', code=code,
                               bytes=len(raw),
                               tables=pages[code]['tables_found']))
        except (FetchBudgetExceeded, ByteBudgetExceeded) as exc:
            stop = f'budget_trip:{type(exc).__name__}'
            ledger.append(dict(event='budget_trip', code=code))
            break
        except Exception as exc:
            stop = f'page_fail:{type(exc).__name__}'
            ledger.append(dict(event='page_fail', code=code, err=str(exc)[:120]))
            break

    if stop == 'completed':
        m = pages['000001']
        bonus = m.get('bonus') or []
        mech = dict(
            bonus_headers=[h for h in m['headers'] if any('除权' in x for x in h)],
            allotment_headers=[h for h in m['headers'] if any('配股' in x for x in h)],
            field_semantics='表头原样保留；生效日=含"除权"的日期列（除权除息日/配股除权日）',
            status_field='实施状态字段：' + ('存在' if any('进度' in '|'.join(h) or '状态' in '|'.join(h)
                                          for h in m['headers']) else '未见（如实登记缺失）'),
            units='派息(元)／送股、转增(股)——以表头括注为准；配股比例、配股价格以表头为准',
            history_range=str(min((str(r.get('公告日期', '9999'))[:4] for r in bonus),
                                  default='无记录')),
            pagination='单页全历史（页面无翻页控件参数；tables_found=%d）' % m['tables_found'],
            status_note='仅按表头判读，未做语义重命名',
        )
        mechanics = mech
        ledger.append(dict(event='mechanics', **{k: str(v)[:80] for k, v in mech.items()}))

        controls = []
        anchors = {
            '000001': [('2025-06-05', 'cash_div', 'C2 正对照：2024年度权益分派（本地公告 2025-06-05）')],
            '000049': [('2023-11-27', 'allotment', 'C1 正对照：配股发行公告（本地 2023-11-27）'),
                       ('2023-12-08', 'allotment', 'C1b 正对照：配股结果公告（本地 2023-12-08）')],
            '002594': [('2025-07-29', 'probe_only', 'C3 cache 锚定（mult 3.036，类型待证；不作覆盖认证）')],
            '000564': [('2021-12-31', 'probe_only', 'C4 cache 锚定（mult 2.378，跨年；不作覆盖认证）')],
        }
        daily_cache = {}
        for code, cases in anchors.items():
            pg = pages.get(code) or {}
            for want_date, kind, note in cases:
                table = (pg.get('allotment') or []) if kind == 'allotment' else (pg.get('bonus') or [])
                hit = next((r for r in table if str(r.get('公告日期', ''))[:10] == want_date), None)
                rec = dict(code=code, want=want_date, kind=kind, note=note,
                           found=hit is not None)
                if hit:
                    ex = None
                    for k, v in hit.items():
                        if '除权' in str(k) and v:
                            ex = str(v)[:10]
                    rec['record'] = {k: (str(v)[:20] if v is not None else None)
                                     for k, v in hit.items()}
                    rec['ex_date'] = ex
                    if ex:
                        if code not in daily_cache:
                            daily_cache[code] = sdb_daily(code)
                        gap = price_gap_check(daily_cache[code], ex)
                        rec['gap_evidence'] = gap
                        # 生效日一致性：只认价格缺口佐证；缺佐证=DATE_UNVERIFIED
                        if gap.get('status') == 'OK':
                            adj = abs(gap['implied_adj'])
                            rec['ex_date_status'] = ('EX_DATE_PRICE_GAP_CORROBORATED'
                                                     if adj > 0.005 else
                                                     'DATE_UNVERIFIED(no_gap_seen)')
                        else:
                            rec['ex_date_status'] = f'DATE_UNVERIFIED({gap.get("status")})'
                    else:
                        rec['ex_date_status'] = 'DATE_UNVERIFIED(no_ex_field)'
                controls.append(rec)
                ledger.append(dict(event='control', code=code, want=want_date,
                                   found=rec['found'],
                                   ex_status=rec.get('ex_date_status', 'n/a')))
        # C5 负对照窗口
        bonus1 = pages['000001'].get('bonus') or []
        in_win = [r for r in bonus1
                  if '2025-02-01' <= str(r.get('除权除息日', ''))[:10] <= '2025-04-30']
        controls.append(dict(code='000001', kind='negative_window',
                             window='2025-02-01~2025-04-30',
                             interface_records_in_window=len(in_win),
                             local_corpus_records=0,
                             note='接口与本地语料均无记录（不作"确认无事件"表述）'))
        (BATCH / 'controls.json').write_text(
            json.dumps(controls, ensure_ascii=False, indent=1), encoding='utf-8')
        (BATCH / 'mechanics.json').write_text(
            json.dumps(mechanics, ensure_ascii=False, indent=1), encoding='utf-8')

    final = dict(stop_reason=stop, requests_used=req.used, requests_cap=CAP_R,
                 response_bytes=nb.used, response_bytes_cap=CAP_B,
                 disk_bytes=disk.used, disk_bytes_cap=CAP_B,
                 note='批次 b 单次运行账本；额度不随重启重置')
    with (BATCH / 'stop_record.json').open('xb') as f:
        f.write(json.dumps(final, ensure_ascii=False, indent=1).encode('utf-8'))
    with (BATCH / 'request_ledger.json').open('xb') as f:
        f.write(json.dumps(ledger, ensure_ascii=False, indent=1).encode('utf-8'))
    L = ['# 批次 b：新浪分红配股接口探查报告', '',
         f"- 账本：请求 {req.used}/{CAP_R}｜响应 {nb.used} B/3 MiB｜落盘 {disk.used} B/3 MiB（含 32KB 预留）",
         '- 本批=接口可行性探查：不生成标签、不置覆盖、不训练', '']
    if mechanics:
        L.append('## 接口机制（阶段 1：000001）')
        for k, v in mechanics.items():
            L.append(f'- {k}：{v}')
    if (BATCH / 'controls.json').exists():
        L.append('\n## 对照结果')
        for c in json.loads((BATCH / 'controls.json').read_text(encoding='utf-8')):
            L.append(f"- {c.get('note', c.get('kind'))}: found={c.get('found', c.get('interface_records_in_window'))} "
                     f"ex={c.get('ex_date', '—')} status={c.get('ex_date_status', '')} "
                     f"gap={c.get('gap_evidence', {})}")
    (BATCH / 'report.md').write_text('\n'.join(L), encoding='utf-8')
    print(json.dumps(final, ensure_ascii=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
