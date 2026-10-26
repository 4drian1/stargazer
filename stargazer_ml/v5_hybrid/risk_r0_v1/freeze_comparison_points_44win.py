"""Freeze REV2 inputs and points without replacing the legacy REV1 evidence."""
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'step9_preflight'))
BATCH = HERE / 'independent_fetch/batch_d_mainpop_30'
MAP_PATH = BATCH / 'five_day_window_map_v3.json'


def main():
    import pyarrow.dataset as ds
    from data_source_provenance import DAILY_PARTS
    import price_verification_batch_44win as api
    from price_verification_guarded_run import encoded, sha, materialize_points
    point_file = BATCH / 'price_comparison_points_44win_v2.json'
    manifest_file = BATCH / 'price_verification_manifest_v2.json'
    if point_file.exists() or manifest_file.exists():
        raise ValueError('FROZEN_VERSION_EXISTS_NO_OVERWRITE')
    files = sorted(Path(DAILY_PARTS).resolve().glob('part_*.parquet'))
    if not files:
        raise ValueError('DAILY_FILES_MISSING')
    calendar = sorted(ds.dataset([str(p) for p in files],format='parquet')
                      .to_table(columns=['date'])['date'].unique().to_pylist())
    mapping = json.loads(MAP_PATH.read_bytes())
    windows, points = materialize_points(mapping, calendar)
    if (len(windows),len(points),len({p['code'] for p in points})) != (44,225,30):
        raise ValueError('FIXED_COUNTS_MISMATCH')
    point_value = {'schema':'PRICE_COMPARISON_POINTS_V2','windows':windows,'points':points,
                   'windows_total':44,'references_total':264,'unique_points_total':225,'stocks_total':30,
                   'map_sha256':sha(MAP_PATH)}
    old_points = json.loads((BATCH / 'price_comparison_points_44win.json').read_bytes())
    if sorted(old_points['points'],key=lambda p:(p['code'],p['date'],p['field'])) != points:
        raise ValueError('LEGACY_POINT_SET_DIFF_REQUIRES_REVIEW')
    assets = {
        'map':MAP_PATH, 'points':point_file,
        'population':HERE.parent / 'green_common_profile/gcp_round_b_small_sample_v1/gcp_247_ruleA_trigger_records_v1.json',
        'cache':HERE.parent / 'opportunity_v2/gate1c/feb2026_adjustment_cache_v1/adj_cache.json',
        'reanalysis':BATCH / 'diff_table_reanalysis_v1.json',
        'design':HERE / 'batch_d_design.json',
        'calendar_code':HERE.parent / 'step9_preflight/trading_calendar.py',
        'provenance_code':HERE.parent / 'step9_preflight/data_source_provenance.py',
        'runner_code':HERE / 'price_verification_batch_44win.py',
        'guard_code':HERE / 'price_verification_guarded_run.py',
        'client_code':HERE / 'independent_fetch_client.py',
        'freeze_code':Path(__file__).resolve(),
        'request_order':HERE / 'price_verification_request_order_44win.md'}
    assets.update({'daily:'+p.name:p for p in files})
    point_bytes = encoded(point_value)
    inputs = []
    import hashlib
    for role,path in assets.items():
        path = path.resolve()
        inputs.append({'role':role,'path':str(path),
                       'bytes':len(point_bytes) if role == 'points' else path.stat().st_size,
                       'sha256':hashlib.sha256(point_bytes).hexdigest() if role == 'points' else sha(path)})
    manifest = {'schema':'PRICE_VERIFICATION_MANIFEST_V2','inputs':inputs,'calendar_dates':calendar,
                'limits':{'requests':api.REQUEST_CAP,'response_bytes':api.BYTE_CAP,
                          'disk_bytes':api.BYTE_CAP,'report_reserve_bytes':api.REPORT_RESERVE,'retries':0},
                'source':'Tencent RAW day; external cross-check, independence unverified',
                'calendar_semantics':'daily date union; mapping only, independent source gate OPEN',
                'network_authorized':False,'label_build_authorized':False,'fit_authorized':False}
    manifest_bytes = encoded(manifest)
    if len(point_bytes) + len(manifest_bytes) > 1024 * 1024:
        raise ValueError('ZERO_NETWORK_ARTIFACT_BUDGET_STOP')
    # Check original parent bindings before publishing the new artifacts.
    design = json.loads(assets['design'].read_bytes())
    if sha(assets['population']) != design['population_binding']['sha256'] or sha(assets['cache']) != design['cache_binding']['sha256']:
        raise ValueError('PARENT_INPUT_BINDING_MISMATCH')
    for path,raw in ((point_file,point_bytes),(manifest_file,manifest_bytes)):
        with path.open('xb') as stream:
            stream.write(raw)
    print(json.dumps({'windows':44,'references':264,'unique_points':225,'stocks':30,
                      'input_files':len(inputs),'new_bytes':len(point_bytes)+len(manifest_bytes),
                      'manifest_sha256':sha(manifest_file),'status':'FROZEN_NOT_AUTHORIZED'}))


if __name__ == '__main__':
    main()
