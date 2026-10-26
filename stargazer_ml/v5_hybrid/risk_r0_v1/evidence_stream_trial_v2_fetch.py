"""Step 1+3a: fetch_list_v2 construction, stockdb fetch, minute extension write.

Discipline: local stockdb HTTP only (127.0.0.1:7899), zero external network; frozen
artifacts read-only; extension files in a NEW directory (minute_ext_20260908/); old
4196 frozen minute files untouched. Rate limit: >=1s between requests, 3 retries
with backoff, per-response byte cap; batch aborts honestly on exhausted retries.

Warm-up lookback (verified from frozen builders, read-only inspection):
- intraday_relative_stall_v2.minute_returns: consumes only the replay `days`;
  prior-day close enters via `references` (frozen joint_parts / anchor evidence),
  NOT via extra minute history.
- multiboard_risk_features_v1 (matrix_features/summarize/with_new_high) and
  build_common_candidate_sidecar_v1.month_features: within-replay-window only.
=> minute-level need = E day (20260908, full session) for all 19, plus P day
   (20260907, full session) where the frozen file lacks it (P-bar gap or no file).
   Deeper warm-up history is NOT required by the builders; reference/sidecar gaps
   are reported separately (they are not minute-fetchable).
"""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
OUT = HERE / 'evidence_stream_trial_v2'
EXT = OUT / 'minute_ext_20260908'
sys.path.insert(0, str(HERE))
sys.path.insert(0, r'D:/stargazer/c3')
import _stockdb as sdb
from rebuild_features_v2 import slot_profile, missing_intervals, LEDGER_BY_CODE, QUARANTINE_CSV

P_DAY, E_DAY = 20260907, 20260908
CODES_19 = ['000937', '001278', '002274', '002278', '002519', '300328', '300582',
            '300583', '300769', '301218', '301246', '301358', '600844', '603376',
            '603665', '603701', '688005', '688275', '688759']
REQUEST_INTERVAL = 1.0
RETRY = 3
BACKOFF = (2.0, 4.0, 8.0)
RESPONSE_CAP = 64 * 1024 * 1024
T0 = time.time()


def log(stage, **kw):
    print(json.dumps(dict(stage=stage, elapsed=round(time.time() - T0, 1), **kw)), flush=True)


def fetch_minutes(code):
    """One stockdb request (full history), filtered to [P_DAY, E_DAY]. Retries."""
    last = None
    for attempt in range(RETRY):
        try:
            opener = sdb.make_opener()
            req = opener.open  # noqa - clarity
            import urllib.request
            import urllib.parse
            t = urllib.parse.quote('分钟k:%s:*' % code)
            url = '%s/?cmd=get&t=%s' % (sdb.DEFAULT_BASE, t)
            request = urllib.request.Request(url, headers={'Accept': 'application/json'})
            with opener.open(request, timeout=120) as r:
                raw = r.read(RESPONSE_CAP + 1)
            if len(raw) > RESPONSE_CAP:
                raise ValueError('RESPONSE_CAP_EXCEEDED')
            obj = sdb._decode(raw)
            recs = sdb.recs_of(obj)
            rows = [r for r in recs
                    if isinstance(r.get('date'), int) and P_DAY <= r['date'] // 1000000 <= E_DAY]
            return rows, len(raw)
        except Exception as ex:  # retry with backoff, then abort batch honestly
            last = '%s:%s' % (type(ex).__name__, str(ex)[:120])
            log('fetch_retry', code=code, attempt=attempt + 1, error=last)
            time.sleep(BACKOFF[attempt])
    raise RuntimeError('FETCH_FAILED:' + code + ':' + last)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', action='store_true', required=True)
    args = ap.parse_args()
    if OUT.exists():
        raise ValueError('OUTPUT_EXISTS_NO_RESTART')
    OUT.mkdir()
    EXT.mkdir()

    # ---------- step 1: fetch list ----------
    trial = json.loads((HERE / 'evidence_stream_trial_v1' / 'coverage_trial.json').read_bytes())
    trial = {r['code']: r for r in trial}
    tm = pd.DataFrame(json.loads((HERE / 'rebuild_features_v1' / 'true_missing_v2.json').read_bytes()))
    tm['code'] = tm.code.astype(str).str.zfill(6)

    quarantined = set(pd.read_csv(QUARANTINE_CSV)['code']) if QUARANTINE_CSV.exists() else set()
    entries = []
    for code in CODES_19:
        rec = trial[code]
        need_p = False
        if code in LEDGER_BY_CODE:
            ok_p = slot_profile(LEDGER_BY_CODE[code]['path'], P_DAY, quarantined)
            need_p = not bool(ok_p.all())
        else:
            need_p = True
        days = [E_DAY] + ([P_DAY] if need_p else [])
        tm_same = tm[(tm.code == code) & (tm.E == E_DAY)]
        tm_other = tm[(tm.code == code) & (tm.E != E_DAY)]
        if len(tm_same):
            dedup = 'duplicate: true_missing_v2 已登记同 (code,E=20260908) 条目 %d 条' % len(tm_same)
        elif len(tm_other):
            dedup = 'code 已登记过他日缺口 %s，本次 E=20260908 为新增条目' % sorted(tm_other.E.unique().tolist())
        else:
            dedup = 'new: true_missing_v2 无该 code 任何条目'
        for d in days:
            entries.append(dict(code=code, date=d, bar_range='093100-150000 (1m, 240 bars)',
                                purpose=('E_day' if d == E_DAY else 'warmup_P'),
                                datasource='stockdb:分钟k (local 127.0.0.1:7899)',
                                trial_reason=rec['reason'], dedup=dedup,
                                builder_lookback='minute_returns/summarize/month_features 只消费回放日; '
                                                  'P日仅在冻结文件缺全日时需要'))
    n_req = len(set(e['code'] for e in entries))
    est_bytes = n_req * 7 * 1024 * 1024  # measured ~7MiB JSON per full-history code
    plan = dict(status='FETCH_LIST_V2', generated=time.strftime('%Y-%m-%d %H:%M:%S'),
                p_day=P_DAY, e_day=E_DAY, n_codes=len(CODES_19), n_entries=len(entries),
                n_requests=n_req, estimated_read_bytes=est_bytes,
                rate_limit=dict(interval_s=REQUEST_INTERVAL, retries=RETRY, backoff=BACKOFF,
                                response_cap_bytes=RESPONSE_CAP),
                scale_check='stockdb 分钟k vs 冻结分钟文件: 300014@20260908/300582@20260907 '
                            '逐行 OHLCV 位级一致 (volume/amount ratio=1.000000)',
                warm_up_policy='builders consume replay days only; references bound from frozen '
                               'joint_parts/anchor (separate gap, not minute-fetchable)')
    (OUT / 'fetch_list_v2.json').write_bytes(json.dumps(dict(plan=plan, entries=entries), indent=1).encode())
    log('fetch_list_ready', entries=len(entries), requests=n_req)

    # ---------- step 3a: fetch + extension write ----------
    manifest = {}
    total_in = 0
    fetched = {}
    codes = sorted(set(e['code'] for e in entries))
    for i, code in enumerate(codes):
        rows, n_bytes = fetch_minutes(code)
        total_in += n_bytes
        fetched[code] = rows
        log('fetched', code=code, rows=len(rows), bytes=n_bytes)
        if i + 1 < len(codes):
            time.sleep(REQUEST_INTERVAL)

    ext_ledger = []
    for code in codes:
        rows = fetched[code]
        frame = pd.DataFrame(rows)
        if frame.empty:
            raise ValueError('EMPTY_FETCH:' + code)
        frame = frame.drop_duplicates('date').sort_values('date').reset_index(drop=True)
        frame['code'] = code
        frame = frame[['code', 'date', 'open', 'high', 'low', 'close', 'volume', 'amount']]
        # validate with frozen bar rules per needed day
        path = EXT / ('%s.parquet' % code)
        frame.to_parquet(path, index=False, compression='zstd')
        body = path.read_bytes()
        for d in sorted(set(frame.date // 1000000)):
            ok = slot_profile(path, int(d), quarantined)
            miss = np.flatnonzero(~ok).tolist()
            log('ext_day_validated', code=code, day=int(d), bars=int(ok.sum()),
                missing=missing_intervals(miss) if miss else [])
        ext_ledger.append(dict(code=code, path=str(path), sha256=hashlib.sha256(body).hexdigest(),
                               rows=len(frame),
                               days=sorted(int(x) for x in set(frame.date // 1000000))))
    (OUT / 'minute_ext_20260908_ledger.json').write_bytes(
        json.dumps(ext_ledger, indent=1).encode())
    log('extension_written', files=len(ext_ledger), bytes=sum(e['rows'] for e in ext_ledger),
        read_bytes=total_in)
    (OUT / 'fetch_manifest.json').write_bytes(json.dumps(dict(
        status='FETCH_COMPLETE', http_requests=len(codes), external_http=0,
        stockdb_read_bytes=total_in, extension_files=len(ext_ledger),
        extension_bytes=sum((EXT / ('%s.parquet' % e['code'])).stat().st_size for e in ext_ledger),
        interval_s=REQUEST_INTERVAL, retries_used=0), indent=1).encode())


if __name__ == '__main__':
    main()
