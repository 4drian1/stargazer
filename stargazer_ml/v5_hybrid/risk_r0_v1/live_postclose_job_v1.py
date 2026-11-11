"""Daily post-close scoring job; local data only, frozen weights, no LLM."""
import os
for _thread_env in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[_thread_env]='1'
import argparse
from dataclasses import asdict
from datetime import datetime
import hashlib
import gzip
import json
from pathlib import Path
import sys
import time

import numpy as np

from live_postclose_source_v1 import atomic_json, digest, produce_live, now_stamp

DESKTOP = Path(__file__).resolve().parents[3] / 'stargazer_desktop'
sys.path.insert(0, str(DESKTOP))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent/'green_common_profile'))


def number(value):
    try:
        value = float(value)
    except (ValueError, TypeError):
        return None
    return value if np.isfinite(value) else None


def verify_prepared_source_versions(capture):
    """TDX downloads racing a warm cache may not change its input version."""
    provenance=capture.output/'common_context_reuse.json'
    if not provenance.is_file():return
    common=json.loads(provenance.read_bytes())
    request=json.loads((Path(common['prepared_folder'])/'request.json').read_bytes())
    for code,source in request['files'].items():
        st=Path(source['path']).stat()
        if (st.st_size,st.st_mtime_ns)!=(source['size'],source['mtime_ns']):
            raise ValueError('SOURCE_VERSION_CHANGED_DURING_SCORING:'+code)


def pack_from_row(row, selection, amplitude, snapshot, missing, run_name, selection_record=None):
    from oracles_v1 import EvidencePack, tarot_candidate, pyxis_context
    cutoff = str(snapshot.cutoff)
    code = selection['code']
    market = {'index': 'sse', 'strength': number(row.get('sse_15m_day_return_prev_close')),
              'cyb_strength': number(row.get('cyb_15m_day_return_prev_close')),
              'golden_position': number(row.get('new_sse_golden_position'))}
    boards = [{'scope': 'candidate_affiliations_loo_observed_partial',
               'strength': number(row.get('context_sector_mean_return_loo_raw')),
               'coverage': number(row.get('context_sector_mean_coverage_loo'))}]
    leaders = [{'scope': 'observed_current_leader_loo',
                'strength': number(row.get('context_current_leader_return_loo_raw')),
                'return_change': number(row.get('context_current_leader_return_change_loo_raw'))}]
    required_missing = []
    if market['strength'] is None:
        required_missing.append('market_state')
    if boards[0]['strength'] is None:
        required_missing.append('board_states')
    if leaders[0]['strength'] is None:
        required_missing.append('leader_states')
    score = (selection_record or {}).get('selection_score')
    golden = (selection_record or {}).get('golden_plan') or {}
    if selection.get('selection_model'):
        score=selection['selection_score']
        golden={'F618':selection['entry_rule']['f618'],'plan_dist_atr':selection['distance_atr']}
    market['model_inputs']={k:number(row[k]) for k in row.keys()
        if k.startswith(('sse_','cyb_')) and any(t in k for t in ('return','rsi','momentum','anchor_position'))
        and not any(t in k for t in ('asof_day','reference_close_day'))}
    context_values={k:number(row[k]) for k in row.keys() if k.startswith('context_')}
    candidate = tarot_candidate(code, p_close_score=score,
        golden_zone=number(selection.get('distance_atr')), f618=golden.get('F618'),
        plan_dist_atr=golden.get('plan_dist_atr'), atr_ref=selection.get('atr_ref'),
        downside_state={'approach': None, 'stop': None, 'divergence': None}, trigger=None,
        evidence_refs={'selection': f'snapshot:{cutoff[:8]}:{code}', 'risk': run_name})
    context = pyxis_context(version='232主版', amp_232=amplitude[0], amp_266=amplitude[1],
        market_state=market, board_states=boards, leader_states=leaders,
        missing=required_missing, asof_ts=cutoff)
    context.update(feature_missing=missing, membership_scope='ARCHIVED_MEMBERSHIP_SNAPSHOT_OBSERVED_NOW',
                   raw_context_values=context_values,stock_close=number(row.get('stock_close')),
                   stock_volume_ratio=number(row.get('stock_volume_ratio_previous_session_raw')),
                   comparison_board_count=len(snapshot.boards),
                   bound_stock_count=len(snapshot.stock_minutes),
                   declared_stock_count=len({c for m in snapshot.boards.values() for c in m}),
                   source_audit=run_name+'/source_manifest.json',
                   asof_ts=cutoff, observed_at=snapshot.observed_at,
                   status='PARTIAL_INPUT_DISCLOSED',
                   trigger_state='NOT_COMPUTED_NOT_A_BUY_SIGNAL')
    from golden_ladder_v3 import index_map
    if getattr(snapshot,'_execution_map_cutoff',None)!=snapshot.cutoff:
        snapshot._execution_maps={name:index_map(frame,snapshot.cutoff) for symbol,name in
            [('000001.SH','sse'),('399006.SZ','cyb')] if (frame:=(getattr(snapshot,'index_5m',{}) or {}).get(symbol)) is not None}
        snapshot._execution_map_cutoff=snapshot.cutoff
    context['execution_index_maps']=snapshot._execution_maps
    from market_price_evidence_v1 import for_snapshot
    context['market_price_evidence']=for_snapshot(snapshot)
    context['market_repair_research']={'qualification':'39维已获用户授权试运行，原验收失败及指数量能未核销保留；trial意见与确认事实分开'}
    return EvidencePack(evidence_id=f'pyxis-close:{cutoff[:8]}:{code}', asof_ts=cutoff,
        market_date=cutoff[:8], query_ts=str(now_stamp()), candidates=[candidate],
        risk_context=context, risk_missing=required_missing, corvus_gap=True,
        notes='POSTCLOSE_LOCAL_RUNTIME; partial archived membership; amplitude not probability; no auto orders').bind()


def run(snapshot_file, runtime, producer=produce_live, *, index_source=None):
    import msvcrt
    runtime.mkdir(parents=True, exist_ok=True)
    with (runtime/'worker.lock').open('a+b') as lock:
        if lock.tell() == 0:
            lock.write(b'0')
            lock.flush()
        lock.seek(0)
        try:
            msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
        except OSError:
            return 3
        raw_snapshot = snapshot_file.read_bytes()
        value = json.loads(raw_snapshot)
        day = int(value['data_date'])
        if getattr(producer,'native_tdx',False):
            from accept_live_postclose_v1 import plan_calendar
            plan_calendar(value)
            if day>now_stamp()//1000000 or (day==now_stamp()//1000000 and now_stamp()%1000000<150000):
                raise ValueError('POSTCLOSE_CLOSE_NOT_AVAILABLE')
            if now_stamp()//1000000>int(value['monitor_date']):
                raise ValueError('POSTCLOSE_PLAN_EXPIRED')
        elif index_source:
            from postclose_index_source import job_due, load
            if not job_due(value, datetime.strptime(str(now_stamp()), '%Y%m%d%H%M%S')):
                raise ValueError('POSTCLOSE_ACTIVE_PLAN_DATE_REQUIRED')
            load(index_source[0], index_source[1], day)
            from accept_live_postclose_v1 import plan_calendar
            plan_calendar(value)  # Reject an unbound or nonconsecutive calendar before any batch reads.
        elif day != now_stamp()//1000000 or now_stamp()%1000000 < 150000:
            raise ValueError('POSTCLOSE_CURRENT_DAY_REQUIRED')
        identity = digest(raw_snapshot)
        output = runtime / f'run_{day}_{datetime.now():%H%M%S_%f}'
        output.mkdir(exist_ok=False)
        with (output/'input_snapshot.json').open('xb') as stream:
            stream.write(raw_snapshot)
        last_progress = {}
        def progress(**values):
            last_progress.clear()
            last_progress.update(values)
            state = dict(status='RUNNING', data_date=day, snapshot_sha256=identity,
                         run_dir=str(output), updated_at=now_stamp(), **values)
            atomic_json(runtime/'state.json', state)
            print(json.dumps({k:v for k,v in state.items() if k != 'snapshot_sha256'}), flush=True)
        try:
            from joint_shadow_service_v1 import verify_fingerprints
            reference = json.loads(gzip.decompress(Path(__file__).with_name('live_monitor_fingerprints_20261110_v206_flow.json.gz').read_bytes()))
            mismatch, ok = verify_fingerprints(reference)
            if not ok:
                raise ValueError('FROZEN_RUNTIME_FINGERPRINT_MISMATCH:'+','.join(mismatch))
            rows = value['rows']
            codes = [str(r['code']).zfill(6) for r in rows]
            if not 0 < len(codes) <= 256 or len(set(codes)) != len(codes):
                raise ValueError('CANDIDATE_SIZE_OR_DUPLICATES')
            progress(stage='SOURCE_PREFLIGHT', done=0, total=len(codes))
            snapshot, capture, source_failures = producer(day, codes, output, progress,
                **({'index_source':index_source} if index_source else {}))
            verify_prepared_source_versions(capture)
            from joint_shadow_context_v1 import build_features
            import rebuild_features_v1 as frozen
            universe = {c for m in snapshot.boards.values() for c in m}
            candidates = [dict(code=c, P=snapshot.days[-2], E=day, cutoff_ts=snapshot.cutoff)
                          for c in codes if c in universe and c in snapshot.stock_minutes]
            if not candidates:
                raise ValueError('NO_CANDIDATE_MINUTE_INPUT')
            progress(stage='REBUILDING_FROZEN_FEATURES', done=0, total=len(candidates))
            from shared_risk_context_v1 import SharedPreparedContext
            feature_cache = SharedPreparedContext(snapshot, capture)
            from postclose_feature_cache_v1 import build as build_cached_features
            frame,x232,x266,audit=build_cached_features(snapshot,candidates,capture,feature_cache,build_features)
            feature_cache.commit()
            capture.save('feature_cache_audit.json',json.dumps(feature_cache.audit).encode())
            capture.save('quarantined_bars.json',json.dumps(
                snapshot.stock_minutes.quarantined,ensure_ascii=False).encode())
            predictions = [frozen.load_booster(folder).predict(frozen.native_missing_view(x, list(x)),num_threads=1)
                           for folder, x in [(frozen.MODEL_232,x232),(frozen.MODEL_266,x266)]]
            verify_prepared_source_versions(capture)
            progress(stage='PACKING_AND_VALIDATING', done=len(candidates), total=len(candidates))
            # Selection's own P-day reference remains separate from risk's prior-session key.
            from joint_shadow_worker_v1 import selection_scores
            selection_rows = ([] if value.get('rule_version')=='TAROT_A_SECOND_CASH_V2' else
                              selection_scores([dict(code=c,P=day,E=int(value['monitor_date'])) for c in codes]))
            selection_by_code = {r['code']:r for r in selection_rows}
            from jev_adapter_v1 import _risk_evidence_incomplete
            packs, results = [], []
            source_rows = {r['code']: r for r in rows}
            required = ['stock_day_return_raw','context_sector_mean_return_loo_raw',
                        'sse_15m_day_return_prev_close','cyb_15m_day_return_prev_close','new_sse_golden_position']
            for i, candidate in enumerate(candidates):
                code = candidate['code']
                missing_required = [k for k in required if number(frame.iloc[i].get(k)) is None]
                missing = {'232':list(x232.columns[x232.iloc[i].isna()]),
                           '266':list(x266.columns[x266.iloc[i].isna()])}
                amplitudes = [number(p[i]) for p in predictions]
                if missing_required or None in amplitudes:
                    results.append(dict(code=code,status='NOT_COMPUTABLE',reasons=missing_required or ['NONFINITE_MODEL']))
                    continue
                pack = pack_from_row(frame.iloc[i], source_rows[code],
                    [float(np.clip(p,0,1)) for p in amplitudes], snapshot, missing, str(output),
                    selection_by_code.get(code))
                reasons = _risk_evidence_incomplete(pack)
                results.append(dict(code=code,status='AVAILABLE' if not reasons else 'INCOMPLETE_CONTEXT',
                                    reasons=reasons, feature_missing={k:len(v) for k,v in missing.items()}))
                if not reasons:
                    packs.append(asdict(pack))
            seen = {r['code'] for r in results}
            for code in codes:
                if code not in seen:
                    results.append(dict(code=code,status='NOT_COMPUTABLE',
                        reasons=[source_failures.get(code,'CANDIDATE_MEMBERSHIP_OR_MINUTE_MISSING')]))
            for name, x in [('inputs_232.parquet',x232),('inputs_266.parquet',x266),('source_features.parquet',frame)]:
                import io
                sink = io.BytesIO()
                x.to_parquet(sink,index=False,compression='zstd')
                capture.save(name,sink.getvalue())
            body = b''.join((json.dumps(p,ensure_ascii=False,allow_nan=False)+'\n').encode() for p in packs)
            capture.save('packs.jsonl',body)
            report = dict(status='COMPLETE' if packs else 'NO_SCORABLE_PACKS',data_date=day,
                snapshot_sha256=identity,candidate_count=len(codes),available_count=len(packs),
                rows=results,run_dir=str(output),packs_path=str(output/'packs.jsonl'),
                packs_sha256=digest(body),finished_at=now_stamp(),requests=capture.requests,
                read_bytes=capture.read_bytes,disk_bytes=capture.disk_bytes,
                read_bytes_basis=('LOCAL_CAPTURE_COUNTER_ONLY; native preparation source reads are in the separate worker'
                    if getattr(snapshot,'_source_contract',None)=='TDX_NATIVE_POSTCLOSE_V1' else 'LOCAL_CAPTURE_COUNTER'),
                fit=False,llm_calls=0,scope=('PARTIAL_ARCHIVED_MEMBERSHIP_POSTCLOSE_OBSERVED_LATER'
                    if snapshot.membership_scope=='POSTCLOSE_CAPTURE_OBSERVED_LATER'
                    else 'PARTIAL_ARCHIVED_MEMBERSHIP_LOCAL_FORWARD_OBSERVATION'),
                feature_cutoff=snapshot.cutoff,observed_at=snapshot.observed_at,
                source_contract=getattr(snapshot,'_source_contract','STOCKDB_MINUTES_V1'),
                source_identity=getattr(snapshot,'_source_identity',None),
                code_sha256={str(Path(__file__)):digest(Path(__file__).read_bytes()),
                    str(Path(__file__).with_name('live_postclose_source_v1.py')):
                        digest(Path(__file__).with_name('live_postclose_source_v1.py').read_bytes())})
            capture.save('terminal.json',json.dumps(report,ensure_ascii=False,allow_nan=False).encode())
            atomic_json(runtime/'state.json',report)
            return 0 if packs else 2
        except Exception as exc:
            from risk_source_recovery import transient_error
            retryable = transient_error(exc)
            failure = dict(status='WAIT_SOURCE' if retryable else 'FAILED',source_retryable=retryable,
                last_reported_progress=dict(last_progress),data_date=day,snapshot_sha256=identity,
                run_dir=str(output),reason=type(exc).__name__+':'+str(exc),finished_at=now_stamp(),
                fit=False,llm_calls=0)
            atomic_json(output/'terminal.json',failure)
            atomic_json(runtime/'state.json',failure)
            print(json.dumps(failure),flush=True)
            return 2


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--snapshot',type=Path,required=True)
    parser.add_argument('--runtime',type=Path,required=True)
    parser.add_argument('--reuse-capture',type=Path)
    parser.add_argument('--tdx-native',action='store_true')
    parser.add_argument('--index-source',type=Path)
    parser.add_argument('--index-source-sha')
    args=parser.parse_args()
    if bool(args.index_source) != bool(args.index_source_sha) or (args.index_source and args.reuse_capture):
        parser.error('index-source requires its SHA and cannot combine with reuse-capture')
    if args.tdx_native:
        if args.reuse_capture:parser.error('tdx-native cannot combine with reuse-capture')
        from postclose_shared_source_v1 import produce
    elif args.reuse_capture:
        from live_postclose_replay_v1 import captured_producer
        produce=captured_producer(args.reuse_capture)
    else:
        from shared_risk_context_v1 import produce_shared
        produce=produce_shared
    raise SystemExit(run(args.snapshot,args.runtime,producer=produce,
        index_source=(args.index_source,args.index_source_sha) if args.index_source else None))
