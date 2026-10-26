"""Verify the delivered historical preparation package; no training/network side effects."""
import argparse
import json
from pathlib import Path
import sys

from price_verification_guarded_run import sha,encoded
from prepare_historical_risk_v1 import HERE,OUT,bytes_used

PACKAGE = OUT/'preparation_manifest_final.json'


def paths():
    return [OUT/'manifest.json',OUT/'population_inventory.json',OUT/'training_gate.json',
            HERE/'historical_risk_core_v1.py',HERE/'prepare_historical_risk_v1.py',
            HERE/'test_historical_risk_v1.py',Path(__file__),
            HERE/'contracts/HISTORICAL_STOCK_DAY_V1.md',
            HERE/'contracts/historical_stock_day_experiment_v1.json',
            HERE/'contracts/historical_stock_day_features_v1.json',
            Path('D:/a股数据/free_stockdb_loader.py'),
            Path('D:/a股数据/build_increment.py'),
            Path('D:/a股数据/stockdb/pybao/stock_sdk.py')]


def check():
    package=json.loads(PACKAGE.read_bytes())
    original=json.loads((OUT/'manifest.json').read_bytes())
    for item in package['files']+original['inputs']:
        p=Path(item['path'])
        if p.stat().st_size!=item['bytes'] or sha(p)!=item['sha256']:
            raise ValueError('INPUT_DRIFT:'+str(p))
    files={str(Path(x['path']).resolve()) for x in original['inputs']}
    parent=Path(original['inputs'][0]['path']).parent
    if files!={str(p.resolve()) for p in parent.glob('part_*.parquet')}:
        raise ValueError('DAILY_FILE_SET_CHANGED')
    report=json.loads((OUT/'population_inventory.json').read_bytes())
    gate=json.loads((OUT/'training_gate.json').read_bytes())
    config=json.loads((HERE/'contracts/historical_stock_day_experiment_v1.json').read_bytes())
    if config['fit_authorized'] or report['labels_generated'] or gate['training_ready']:
        raise ValueError('PREPARATION_STATE_INCONSISTENT')
    return dict(package_hashes='PASS',historical_stock_day_scope=True,
                actual_arrival_log_is_blocker=False,
                physical_rows=report['physical_rows'],stock_codes=report['in_scope_codes'],
                structural_train_after_purge=report['segments']['train']['structural_after_purge'],
                P1='SPEC_AND_CORE_TESTED_REAL_LABELS_PENDING_PRICE_EVIDENCE',
                P2='FEATURE_FUNCTIONS_TESTED_MATRIX_PENDING_SOURCE_ADMISSION',
                P3='SPLIT_AND_CONFIG_DEFINED_SAMPLE_GATE_NOT_EVALUABLE',
                training_ready=False,fit_authorized=False,
                blockers=['historical_price_scale_and_corporate_actions',
                          'historical_security_coverage_scope','market_index_source',
                          'materialized_labels_features_and_class_counts'])


def freeze():
    if PACKAGE.exists(): raise ValueError('FINAL_MANIFEST_ALREADY_EXISTS')
    data=encoded(dict(files=[dict(path=str(p.resolve()),bytes=p.stat().st_size,sha256=sha(p))
                             for p in paths()],
                      note='Initial inventory manifest retains the 15-test core version. Final feature core adds market calculation and has 17 tests; structural inventory logic unchanged.'))
    if bytes_used()+len(data)>128*1024**2: raise ValueError('BUDGET')
    with PACKAGE.open('xb') as f: f.write(data)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--freeze',action='store_true')
    args=parser.parse_args()
    if args.freeze: freeze()
    try:
        result=check()
        print(json.dumps(result,ensure_ascii=False))
        sys.exit(0 if result['training_ready'] else 2)
    except Exception as exc:
        print(json.dumps(dict(training_ready=False,error=str(exc)),ensure_ascii=False))
        sys.exit(1)
