"""independent_fetch_run.py — 一次性有界独立取数执行（INDEPENDENT_FETCH 2026-10-10 用户批准）。

边界（批准原文）：仅腾讯；2 次零重试冒烟→冻结案例；北交所 2 例登记覆盖缺口不请求；
全批共享计数器 ≤25 请求/≤5 MiB 响应，单次运行不重置；留证目录+磁盘计数器必传，
本批新增落盘 ≤5 MiB（含原始/元数据/索引/报告/停机留因，预留 256KB 报告空间）；
限速 1/s、超时 10s、重试≤3；冻结容差 |diff| ≤ 0.01 元逐点；尺度与公司行为覆盖分别报告；
不扩 1,013、不生成标签、不训练。产物：请求账本/实际字节/原始哈希/逐点匹配差异缺失表。
"""
from __future__ import annotations

import bisect
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from independent_fetch_client import (ByteBudgetExceeded, FetchBudgetExceeded,
                                      ByteCounter, FetchCounter,
                                      TencentDailyClient)

BATCH = HERE / 'independent_fetch' / 'batch_20261010'
PARTS = Path(r'D:/a股数据/data_daily_parts')
REQ_CAP = 25
BYTE_CAP = 5 * 1024 * 1024          # 5 MiB 响应
DISK_CAP = 5 * 1024 * 1024          # 5 MiB 落盘（含报告预留）
REPORT_RESERVE = 256 * 1024
TOL = 0.01                           # 冻结容差：绝对差 ≤ 0.01 元
BSE = {'920729', '920808'}           # 北交所覆盖缺口：不请求


def local_calendar() -> list:
    import glob
    days = set()
    for f in glob.glob(str(PARTS / 'part_*.parquet')):
        days.update(int(x) for x in pd.read_parquet(f, columns=['date'])['date'].unique())
    return sorted(days)


def local_window(code6: str, days: list, p: int, w_end: int) -> dict:
    win = [d for d in days if p <= d <= w_end]
    if not win:
        return {}
    frames = []
    for part in sorted(PARTS.glob('part_*.parquet')):
        df = pd.read_parquet(part, columns=['date', 'code', 'close', 'low'],
                             filters=[('date', '>=', win[0]), ('date', '<=', win[-1])])
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    df['code'] = df['code'].astype(str).str.zfill(6)
    g = df[df['code'] == code6].drop_duplicates('date').sort_values('date')
    return {int(r.date): (float(r.close), float(r.low)) for r in g.itertuples()}


def tencent_rows_to_map(rows: list) -> dict:
    """腾讯 day 行序=[date, open, close, high, low, info]——字段序与常规 OHLC 不同，固定映射。"""
    out = {}
    for r in rows:
        d = int(str(r[0]).replace('-', ''))
        out[d] = dict(open=float(r[1]), close=float(r[2]),
                      high=float(r[3]), low=float(r[4]))
    return out


def main() -> int:
    BATCH.mkdir(parents=True, exist_ok=True)
    # 批次最终性守卫（REV5 治理修复）：已终结批次拒绝再次运行——
    # 停止线被触发后，任何续跑须用户书面授权（不得靠重建计数器"重启续抓"）。
    # 授权续跑方式：用户批准后删除/改名 stop_record.json（留痕），或指定新批次目录。
    if (BATCH / 'stop_record.json').exists():
        raise SystemExit(
            f'批次已终结（{BATCH}/stop_record.json 存在）。'
            '停止线触发后续跑须用户书面授权：获授权后移走 stop_record.json 留痕，'
            '或改用新批次目录。额度不随重启重置。')
    import uuid
    batch_id = uuid.uuid4().hex[:12]
    ledger = [dict(event='batch_open', batch_id=batch_id,
                   note='本批计数器自本进程创建；重启=新进程=须重新授权')]
    stop_reason = 'completed'
    req = FetchCounter(REQ_CAP)
    nbytes = ByteCounter(BYTE_CAP)
    disk = ByteCounter(DISK_CAP)
    disk.consume(REPORT_RESERVE)          # 提前预留报告空间（计入 5 MiB）
    client = TencentDailyClient(req, nbytes, rate_limit_s=1.0, timeout_s=10.0,
                                max_retries=3, disk_counter=disk)
    cases = json.loads((HERE / 'bounded_validation_cases.json').read_text(encoding='utf-8'))
    cal = local_calendar()

    def log(event, **kw):
        entry = dict(ts=datetime.now(timezone.utc).isoformat(timespec='seconds'),
                     event=event, **kw)
        ledger.append(entry)
        print(json.dumps(entry, ensure_ascii=False)[:200])

    # ---- 冒烟：2 次零重试（1 主板 +1 创业板），同时字段序校验 ----
    smoke = [('600000', 'sh600000'), ('300750', 'sz300750')]
    smoke_maps = {}
    for code6, expect_symbol in smoke:
        try:
            rows = client.fetch_raw_daily(code6, '20261001', '20261010',
                                          archive_dir=str(BATCH), smoke=True)
            m = tencent_rows_to_map(rows)
            smoke_maps[code6] = m
            log('smoke_ok', code=code6, rows=len(rows))
        except Exception as exc:
            stop_reason = f'smoke_failed:{type(exc).__name__}'
            log('smoke_fail', code=code6, err=str(exc)[:120])
    # 字段序验证：与本地 1008 收盘比对（容忍 ±0.01）
    field_ok = True
    loc = local_window('600000', cal, 20261008, 20261008)
    if 20261008 in smoke_maps.get('600000', {}) and 20261008 in loc:
        d = abs(smoke_maps['600000'][20261008]['close'] - loc[20261008][0])
        if d > TOL:
            field_ok = False
            stop_reason = f'smoke_field_order_suspect:diff={d}'
            log('smoke_field_check_fail', diff=d)
    else:
        stop_reason = 'smoke_window_missing'
    log('smoke_field_check', ok=field_ok)
    if stop_reason != 'completed':
        finalize(ledger, req, nbytes, disk, stop_reason)
        return 2

    # ---- 冻结案例取数（北交所登记缺口，不请求） ----
    case_list = []
    for c in cases['cross_shard']:
        case_list.append(dict(code=str(c['code']).zfill(6), P=int(c['P']),
                              W_end=int(c['W_end']), kind='cross_shard'))
    for c in cases['corp_action_fixed']:
        case_list.append(dict(code=str(c['code']).zfill(6), P=int(c['event_date']),
                              kind='corp_action'))
    # corp W_end 按冻结日历（本地日线分片日历）P+5 交易日
    for c in case_list:
        if 'W_end' not in c:
            i = bisect.bisect_right(cal, c['P'])
            c['W_end'] = cal[i + 4] if i + 4 < len(cal) else None

    remote_maps = {}
    for c in case_list:
        if c['code'] in BSE:
            log('coverage_gap_bse', code=c['code'], P=c['P'])
            c['remote'] = 'SOURCE_COVERAGE_GAP'
            continue
        if c['W_end'] is None:
            c['remote'] = 'W_END_UNAVAILABLE'
            continue
        start = (datetime.strptime(str(c['P']), '%Y%m%d') - timedelta(days=9)) \
            .strftime('%Y%m%d')
        try:
            rows = client.fetch_raw_daily(c['code'], start, str(c['W_end']),
                                          archive_dir=str(BATCH))
            remote_maps[(c['code'], c['P'])] = tencent_rows_to_map(rows)
            c['remote'] = 'ok'
            log('case_ok', code=c['code'], P=c['P'], W_end=c['W_end'])
        except (ByteBudgetExceeded, FetchBudgetExceeded) as exc:
            stop_reason = f'budget_trip:{type(exc).__name__}'
            log('budget_trip', code=c['code'], err=str(exc)[:100])
            c['remote'] = f'budget_trip'
            break
        except Exception as exc:
            stop_reason = f'fetch_fail:{type(exc).__name__}'
            log('case_fail', code=c['code'], err=str(exc)[:100])
            c['remote'] = f'error:{type(exc).__name__}'

    # ---- 逐点比对（|diff| ≤ 0.01 元；逐日 close@P + low@P+1..P+5）----
    points = []
    for c in case_list:
        rec = dict(code=c['code'], P=c['P'], W_end=c['W_end'], kind=c['kind'],
                   remote=c['remote'])
        if c['remote'] != 'ok':
            rec['verdict'] = c['remote']
            points.append(rec)
            continue
        loc = local_window(c['code'], cal, c['P'], c['W_end'])
        rem = remote_maps[(c['code'], c['P'])]
        win = [d for d in cal if c['P'] <= d <= c['W_end']]
        rec['days'] = []
        for d in win:
            l = loc.get(d)
            r = rem.get(d)
            for field, li, ri in (('close', 0 if l else None, 'close'),
                                  ('low', 1 if l else None, 'low')):
                lv = l[li] if l else None
                rv = r[ri] if r else None
                if lv is None and rv is None:
                    verdict = 'MISS_BOTH'
                elif lv is None:
                    verdict = 'MISS_LOCAL'
                elif rv is None:
                    verdict = 'MISS_REMOTE'
                elif abs(lv - rv) <= TOL:
                    verdict = 'MATCH'
                else:
                    verdict = 'DIFF'
                rec['days'].append(dict(date=d, field=field, local=lv, remote=rv,
                                        diff=(round(lv - rv, 4)
                                              if lv is not None and rv is not None else None),
                                        verdict=verdict))
        points.append(rec)
        log('compared', code=c['code'], P=c['P'])

    (BATCH / 'match_table.json').write_text(
        json.dumps(points, ensure_ascii=False, indent=1), encoding='utf-8')
    finalize(ledger, req, nbytes, disk, stop_reason, points)
    return 0


def finalize(ledger, req, nbytes, disk, stop_reason, points=None):
    final = dict(stop_reason=stop_reason,
                 requests_used=req.used, requests_cap=req.hard_cap,
                 response_bytes=nbytes.used, response_bytes_cap=nbytes.hard_cap_bytes,
                 disk_bytes=disk.used, disk_bytes_cap=disk.hard_cap_bytes,
                 note='单次运行账本；额度不随重启重置，本批终结')
    with (BATCH / 'stop_record.json').open('xb') as f:      # 独占写：终局不可覆盖
        f.write(json.dumps(final, ensure_ascii=False, indent=1))
    with (BATCH / 'request_ledger.json').open('xb') as f:   # 独占写：账本不可覆盖
        f.write(json.dumps(ledger, ensure_ascii=False, indent=1))
    if points is not None:
        write_report(points, final)
    print(json.dumps(final, ensure_ascii=False))


def write_report(points, final):
    L = ['# 独立价格比对报告（batch_20261010）', '',
         f"- 账本：请求 {final['requests_used']}/{final['requests_cap']}｜"
         f"响应 {final['response_bytes']} B/{final['response_bytes_cap']} B｜"
         f"落盘 {final['disk_bytes']} B/{final['disk_bytes_cap']} B（含 256KB 报告预留）",
         f"- 容差（冻结）：|diff| ≤ 0.01 元逐点；北交所 2 例 SOURCE_COVERAGE_GAP 未请求",
         '- 尺度与公司行为覆盖**分别**裁决；不覆盖旧数据；不扩 1,013；不生成标签', '']
    cs = [p for p in points if p['kind'] == 'cross_shard']
    ca = [p for p in points if p['kind'] == 'corp_action']
    for title, group in (('## 尺度核销（跨分片 6 例）', cs),
                         ('## 公司行为覆盖（12 例，ENGINEERING_CASE）', ca)):
        L.append(title)
        L.append('| code | P | remote | MATCH/DIFF/MISS |')
        L.append('|---|---|---|---|')
        for p in group:
            if p.get('days'):
                from collections import Counter
                cnt = Counter(d['verdict'] for d in p['days'])
                s = '/'.join(f'{k}:{v}' for k, v in sorted(cnt.items()))
            else:
                s = p.get('verdict', p.get('remote', ''))
            L.append(f"| {p['code']} | {p['P']} | {p.get('remote','')} | {s} |")
        L.append('')
    (BATCH / 'report.md').write_text('\n'.join(L), encoding='utf-8')


if __name__ == '__main__':
    raise SystemExit(main())
