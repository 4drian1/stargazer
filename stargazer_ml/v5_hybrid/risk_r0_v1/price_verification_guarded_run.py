"""44-window lifecycle guards. No network until a separately signed fetch record passes."""
import bisect
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path


def encoded(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False, indent=1).encode('utf-8')


def sha(file):
    value = hashlib.sha256()
    with Path(file).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(chunk)
    return value.hexdigest()


def materialize_points(mapping, calendar):
    if mapping.get('input_hash_check') != 'PASS':
        raise ValueError('MAP_HASH_CHECK_NOT_PASS')
    if calendar != sorted(set(calendar)):
        raise ValueError('CALENDAR_INVALID')
    points, windows, seen = set(), [], set()
    for stock in mapping['codes']:
        code = stock['code']
        if not isinstance(code, str) or len(code) != 6 or not code.isdigit():
            raise ValueError('CODE_INVALID')
        for window in stock['windows']:
            p, e = window['P'], window['E']
            dates = [int(d.replace('-', '')) for d in window['window']]
            i = bisect.bisect_right(calendar, p)
            if p not in calendar or len(dates) != 5 or dates != calendar[i:i+5] or e != dates[0]:
                raise ValueError('WINDOW_CALENDAR_MISMATCH')
            checks = window['checks']
            if not checks.get('five_trading_days') or not checks.get('E_is_first_trading_day_after_P') or checks.get('truncated_at_calendar_end'):
                raise ValueError('WINDOW_CHECK_FAIL')
            if (code, p) in seen:
                raise ValueError('DUPLICATE_WINDOW')
            seen.add((code, p))
            windows.append({'code':code, 'P':p, 'E':e, 'window':window['window']})
            points.add((code, p, 'close'))
            points.update((code, d, 'low') for d in dates)
    return sorted(windows, key=lambda w:(w['code'],w['P'])), [
        {'code':code,'date':date,'field':field} for code,date,field in sorted(points)]


def verify_bundle(manifest_path, api):
    path = Path(manifest_path)
    manifest = json.loads(path.read_bytes())
    if manifest.get('schema') != 'PRICE_VERIFICATION_MANIFEST_V2':
        raise ValueError('MANIFEST_SCHEMA')
    if manifest.get('limits') != {'requests':api.REQUEST_CAP, 'response_bytes':api.BYTE_CAP,
                                  'disk_bytes':api.BYTE_CAP, 'report_reserve_bytes':api.REPORT_RESERVE, 'retries':0}:
        raise ValueError('FROZEN_LIMITS_MISMATCH')
    inputs = manifest['inputs']
    roles = [item['role'] for item in inputs]
    required = {'map','points','population','cache','reanalysis','design','calendar_code',
                'provenance_code','runner_code','guard_code','client_code','freeze_code'}
    if not required.issubset(roles) or len(roles) != len(set(roles)):
        raise ValueError('MANIFEST_INPUTS_INCOMPLETE')
    by_role = {}
    for item in inputs:
        file = Path(item['path'])
        if file.stat().st_size != item['bytes'] or sha(file) != item['sha256']:
            raise ValueError('INPUT_HASH_MISMATCH:' + item['role'])
        by_role[item['role']] = file
    daily = [item['path'] for item in inputs if item['role'].startswith('daily:')]
    if not daily or sorted(str(p) for p in Path(daily[0]).parent.glob('part_*.parquet')) != sorted(daily):
        raise ValueError('DAILY_FILE_SET_CHANGED')
    # Bind the exact running implementation, not an arbitrary file bearing a matching hash.
    expected_codes = {'runner_code':api.HERE / 'price_verification_batch_44win.py',
                      'guard_code':api.HERE / 'price_verification_guarded_run.py',
                      'client_code':api.HERE / 'independent_fetch_client.py',
                      'freeze_code':api.HERE / 'freeze_comparison_points_44win.py'}
    if any(by_role[role].resolve() != file.resolve() for role,file in expected_codes.items()):
        raise ValueError('RUNNING_CODE_NOT_BOUND')
    design, mapping = json.loads(by_role['design'].read_bytes()), json.loads(by_role['map'].read_bytes())
    if sha(by_role['population']) != design['population_binding']['sha256'] or sha(by_role['cache']) != design['cache_binding']['sha256']:
        raise ValueError('POPULATION_CACHE_BINDING_MISMATCH')
    if not sha(by_role['population']).startswith(mapping['input_hash']['population']) or not sha(by_role['cache']).startswith(mapping['input_hash']['cache']):
        raise ValueError('MAP_BINDING_MISMATCH')
    windows, points = materialize_points(mapping, manifest['calendar_dates'])
    frozen = json.loads(by_role['points'].read_bytes())
    if frozen['points'] != points or frozen['windows'] != windows or frozen['map_sha256'] != sha(by_role['map']):
        raise ValueError('FROZEN_POINT_SET_MISMATCH')
    if (len(windows),len(points),len({p['code'] for p in points})) != (44,225,30):
        raise ValueError('FIXED_POPULATION_COUNTS_MISMATCH')
    if any(p['code'] in api.BSE_GAP for p in points):
        raise ValueError('UNAPPROVED_SOURCE_COVERAGE')
    local = api.load_local_prices(points, daily)
    # Detect refreshes while prices were being read, before the first request.
    if any(sha(item['path']) != item['sha256'] for item in inputs):
        raise ValueError('INPUT_CHANGED_DURING_PREFLIGHT')
    return manifest, windows, points, local


def authorize(auth_path, expected_sha, output, manifest_path, api):
    if not auth_path or not expected_sha:
        raise ValueError('INDEPENDENT_FETCH_AUTHORIZATION_REQUIRED')
    if sha(auth_path) != expected_sha:
        raise ValueError('AUTHORIZATION_HASH_MISMATCH')
    auth = json.loads(Path(auth_path).read_bytes())
    if auth.get('status') != 'SIGNED' or auth.get('scope') != 'INDEPENDENT_FETCH':
        raise ValueError('INDEPENDENT_FETCH_AUTHORIZATION_REQUIRED')
    if not auth.get('approved_by') or not auth.get('approval_reference'):
        raise ValueError('AUTHORIZATION_IDENTITY_MISSING')
    now = datetime.now(timezone.utc)
    signed, expires = (datetime.fromisoformat(auth[key]) for key in ('authorized_at','expires_at'))
    if signed.tzinfo is None or expires.tzinfo is None or not signed <= now < expires:
        raise ValueError('AUTHORIZATION_TIME_INVALID')
    output = Path(output).resolve()
    allowed = (api.HERE / 'independent_fetch').resolve()
    if output.parent != allowed or output.name != auth.get('batch_id') or output == api.BATCH.resolve():
        raise ValueError('OUTPUT_SCOPE_MISMATCH')
    if output != Path(auth['output_dir']).resolve() or auth.get('manifest_sha256') != sha(manifest_path):
        raise ValueError('AUTHORIZATION_BINDING_MISMATCH')
    if auth.get('limits') != {'requests':api.REQUEST_CAP,'response_bytes':api.BYTE_CAP,'disk_bytes':api.BYTE_CAP}:
        raise ValueError('AUTHORIZATION_LIMITS_MISMATCH')
    # An interrupted or malformed terminal batch is also consumed, not resumable.
    if output.exists():
        raise ValueError('BATCH_ALREADY_CLAIMED_NO_RESTART')
    return auth


def execute(output, auth_path, auth_sha, manifest_path, api, client_factory=None):
    output = Path(output).resolve()
    auth = authorize(auth_path, auth_sha, output, manifest_path, api)
    manifest, windows, points, local = verify_bundle(manifest_path, api)
    plan = api.plan_from_points(points)
    smoke = [next(c for c in sorted(plan) if c.startswith('6')),
             next(c for c in sorted(plan) if not c.startswith('6'))]
    req, response = api.FetchCounter(api.REQUEST_CAP), api.ByteCounter(api.BYTE_CAP)
    disk = api.ByteCounter(api.BYTE_CAP - api.REPORT_RESERVE)
    opened = encoded({'authorization_sha256':auth_sha, 'manifest_sha256':sha(manifest_path),
                      'batch_id':auth['batch_id'], 'note':'独占批次；崩溃后也不得重建额度续跑'})
    disk.consume(len(opened))  # Before mkdir/write.
    output.mkdir(exist_ok=False)
    with (output / 'batch_open.json').open('xb') as stream:
        stream.write(opened)
    ledger, remote, reason = [], {}, 'completed'
    client = None
    try:
        factory = client_factory or api.TencentDailyClient
        client = factory(req, response, disk_counter=disk, max_retries=0, rate_limit_s=1.0, timeout_s=10.0)
        # Smoke responses cover their full approved intervals and are reused: 30 total, not 32.
        for code in smoke + [c for c in sorted(plan) if c not in smoke]:
            item = {'code':code,'smoke':code in smoke,'requests_before':req.used}
            ledger.append(item)
            rng = plan[code]
            rows = client.fetch_raw_daily(code,rng['start'],rng['end'],archive_dir=str(output),smoke=code in smoke)
            parsed = api.tencent_rows_to_map(rows)
            start, end = int(rng['start']), int(rng['end'])
            if any(not start <= day <= end for day in parsed):
                raise ValueError('REMOTE_DATE_OUTSIDE_APPROVED_RANGE')
            remote[code] = parsed
            if code in smoke:
                for point in (p for p in points if p['code'] == code):
                    value = parsed.get(point['date'],{}).get(point['field'])
                    verdict, _ = api.compare_point(local[(code,point['date'])][point['field']],value)
                    if verdict != 'MATCH':
                        raise ValueError('SMOKE_REQUIRED_PRICE_NOT_MATCH:' + verdict)
            item.update(outcome='ok',requests_after=req.used, rows=len(rows))
        if any(sha(item['path']) != item['sha256'] for item in manifest['inputs']):
            raise ValueError('INPUT_CHANGED_DURING_FETCH')
    except (Exception, KeyboardInterrupt) as exc:
        reason = 'failed:' + type(exc).__name__
        if ledger:
            ledger[-1].update(outcome='failed', exception_type=type(exc).__name__,
                              requests_after=req.used, reason=str(exc)[:160])
    matches = []
    for point in points:
        code,date,field = point['code'],point['date'],point['field']
        lv = local[(code,date)][field]
        rv = remote.get(code,{}).get(date,{}).get(field)
        verdict,diff = api.compare_point(lv,rv)
        if code not in remote:
            verdict = 'NOT_VERIFIED_BATCH_ABORTED'
        matches.append(dict(point,local=lv,remote=rv,verdict=verdict,diff=diff))
    per_point = {(p['code'],p['date'],p['field']):p['verdict'] for p in matches}
    window_report = []
    for window in windows:
        code,p,days = window['code'],window['P'],window['window']
        keys = [(code,p,'close')] + [(code,int(day.replace('-','')),'low') for day in days]
        window_report.append(dict(window, all_six_prices_match=all(per_point[k] == 'MATCH' for k in keys),
            boundary_diagnostic=api.boundary_sensitivity(local,code,p,days),
            label_eligible=False,training_eligible=False,
            remaining_gates={'company_action_coverage':'OPEN','historical_availability':'UNKNOWN',
                             'source_scale_semantics':'UNVERIFIED','calendar_source':'SEPARATE_GATE'}))
    actual_response = getattr(client,'received_total',response.used) if client else 0
    stop = {'stop_reason':reason,'requests_used':req.used,'requests_cap':api.REQUEST_CAP,
            'response_body_bytes':actual_response,'response_budget_accounted_bytes':response.used,
            'response_bytes_cap':api.BYTE_CAP,'disk_bytes_cap':api.BYTE_CAP,
            'report_reserve_bytes':api.REPORT_RESERVE,'evidence_counter_bytes':disk.used,
            'counts':dict(Counter(p['verdict'] for p in matches)),
            'authorization_sha256':auth_sha,'manifest_sha256':sha(manifest_path),
            'label_build_authorized':False,'fit_authorized':False,
            'failed_response':getattr(client,'last_failed_response',None),
            'note':'价格一致性≠标签准入；批次终结，停止后不续跑'}
    reports = {'price_match_table_44win.json':encoded(matches),
               'window_price_report_44win.json':encoded(window_report),
               'request_ledger.json':encoded(ledger),'frozen_manifest.json':encoded(manifest),
               'authorization_record.json':Path(auth_path).read_bytes()}
    current = sum(p.stat().st_size for p in output.rglob('*') if p.is_file())
    for _ in range(5):
        final = encoded(stop)
        stop['directory_actual_bytes'] = current + sum(map(len,reports.values())) + len(final)
    reports['stop_record.json'] = encoded(stop)
    if sum(map(len,reports.values())) > api.REPORT_RESERVE or current + sum(map(len,reports.values())) > api.BYTE_CAP:
        # Reserved tiny terminal marker still blocks any restart, without writing oversized reports.
        reports = {'stop_record.json':encoded({'stop_reason':'FINAL_REPORT_BUDGET_STOP',
                     'requests_used':req.used,'response_body_bytes':actual_response,
                     'reports_written':False,'note':'旧证据保留，本批不得续跑'})}
        stop = json.loads(reports['stop_record.json'])
    if current + sum(map(len,reports.values())) > api.BYTE_CAP:
        raise ValueError('TERMINAL_RESERVE_EXHAUSTED_BATCH_OPEN_STILL_BLOCKS_RESTART')
    for name,raw in reports.items():
        with (output / name).open('xb') as stream:
            stream.write(raw)
    return stop
