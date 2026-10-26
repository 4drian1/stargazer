"""REV3 response-superset adaptation. REV2 sources and terminal evidence stay unchanged."""
from collections import Counter
import json
from pathlib import Path

from price_verification_guarded_run import (
    encoded, sha, authorize, verify_bundle as verify_v2,
)

POLICY = 'ARCHIVE_ALL_VALIDATE_ALL_COMPARE_FROZEN_REQUIRED_ONLY_V1'


def select_required(parsed, code, points, request_range, raw_rows):
    required = [point for point in points if point['code'] == code]
    start, end = int(request_range['start']), int(request_range['end'])
    if not required or any(not start <= point['date'] <= end for point in required):
        raise ValueError('FROZEN_POINT_OUTSIDE_REQUEST_RANGE')
    dates = {point['date'] for point in required}
    selected = {date:parsed[date] for date in dates if date in parsed}
    missing = [dict(point) for point in required
               if point['field'] not in selected.get(point['date'], {})]
    before = [date for date in parsed if date < start]
    after = [date for date in parsed if date > end]
    return selected, {
        'policy':POLICY, 'raw_rows':raw_rows, 'validated_unique_dates':len(parsed),
        'returned_first_date':min(parsed) if parsed else None,
        'returned_last_date':max(parsed) if parsed else None,
        'requested_start':start, 'requested_end':end,
        'rows_before_start':len(before), 'rows_after_end':len(after),
        'outside_first_date':min(before+after) if before or after else None,
        'outside_last_date':max(before+after) if before or after else None,
        'required_point_count':len(required), 'required_date_count':len(dates),
        'selected_date_count':len(selected),
        'excluded_from_comparison_date_count':len(parsed)-len(selected),
        'missing_required_points':missing,
        'note':'完整原文保留；额外历史仅审计，不进入必要点比较或训练人口',
    }


def verify_bundle(manifest_path, api):
    metadata = json.loads(Path(manifest_path).read_bytes())
    if metadata.get('runtime_revision') != 3 or metadata.get('response_scope_policy') != POLICY:
        raise ValueError('REV3_POLICY_OR_VERSION_MISMATCH')
    expected = {
        'rev3_runner_code':api.HERE / 'price_verification_batch_44win_v3.py',
        'rev3_guard_code':api.HERE / 'price_verification_guarded_run_v3.py',
        'rev3_freeze_code':api.HERE / 'freeze_price_verification_v3.py',
        'rev3_request_order':api.HERE / 'price_verification_request_order_44win_REV3.md',
        'parent_manifest':api.BATCH / 'price_verification_manifest_v2.json',
    }
    by_role = {item['role']:item for item in metadata.get('inputs', [])}
    if any(role not in by_role or Path(by_role[role]['path']).resolve() != path.resolve()
           for role,path in expected.items()):
        raise ValueError('REV3_IMPLEMENTATION_NOT_BOUND')
    return verify_v2(manifest_path, api)


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
            selected, scope = select_required(parsed, code, points, rng, len(rows))
            item['response_scope'] = scope
            remote[code] = selected
            if scope['missing_required_points']:
                raise ValueError('REMOTE_REQUIRED_POINT_MISSING')
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
