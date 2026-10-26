"""Read local parquet metadata and installed interface source; zero HTTP and no CSV."""
import ast
from collections import defaultdict
import hashlib
import json
from pathlib import Path

import pyarrow.parquet as pq
from research_budget_v2 import TOTAL_CAP, bytes_used

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / 'sector_leader_preparation_20261017_v1.json'
DAILY = ROOT.parents[1] / 'pullback_train/sector_daily.parquet'
PIT = ROOT.parent / 'opportunity_v2/o5_sector_lag/pit_membership.parquet'
INTERFACE = Path(r'D:\stargazer\_dev_venv\Lib\site-packages\akshare\stock\stock_board_concept_em.py')
SNAPSHOT = ROOT / 'sector_concept_prep/energy_storage_BK0989_members_v3_verified.json'


def run():
    if OUTPUT.exists():
        raise RuntimeError('OUTPUT_EXISTS_NO_OVERWRITE')
    paths = [DAILY, PIT, INTERFACE, SNAPSHOT, Path(__file__),
             ROOT / 'intraday_sector_leaders_v1.py', ROOT / 'test_intraday_sector_leaders_v1.py']
    hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    groups = defaultdict(list)
    for row in pq.read_table(DAILY, columns=['sector', 'em_name', 'bk_code', 'src', 'date']).to_pylist():
        groups[row['sector'], row['em_name'], row['bk_code'], row['src']].append(int(row['date']))
    candidates = [dict(sector=k[0], em_name=k[1], bk_code=k[2], src=k[3], rows=len(v),
                       start=min(v), end=max(v), formal_ranking_universe=False)
                  for k, v in sorted(groups.items())]
    tree = ast.parse(INTERFACE.read_text(encoding='utf-8'))
    functions = {node.name: [arg.arg for arg in node.args.args] for node in tree.body
                 if isinstance(node, ast.FunctionDef) and node.name.startswith('stock_board_concept_')}
    snap = json.loads(SNAPSHOT.read_bytes())
    result = dict(status='ENGINEERING_ONLY_NOT_TRAINING_READY', inputs=hashes,
                  sector_daily_candidates=candidates, installed_interface_functions=functions,
                  installed_consumer_has_historical_date_argument=any(
                      'date' in arg for arg in functions['stock_board_concept_cons_em']),
                  scope='Only inspected installed function signatures; no vendor service availability conclusion',
                  current_membership=dict(board=snap['bk_code'], fetched_at=snap['fetched_at'],
                                          rows=len(snap['members']), historical_coverage_verified=False),
                  other_taxonomy=dict(path=str(PIT), columns=pq.read_schema(PIT).names,
                                      cannot_replace_em_concept=True),
                  ranking_universe_status='DRAFT_CANDIDATES_NOT_ALL_MARKET',
                  unresolved=['Same-taxonomy historical membership and known-at evidence',
                              'Frozen comparable board universe and historical coverage',
                              'Tie policy and cross-day retention duration',
                              'Verified source clock, price scale and day-return reference'],
                  http_requests=0, training_ready=False)
    body = json.dumps(result, ensure_ascii=False, allow_nan=False, indent=2).encode('utf-8')
    if len(body) > 1024**2 or bytes_used() + len(body) > TOTAL_CAP:
        raise RuntimeError('BUDGET_STOP')
    with OUTPUT.open('xb') as f:
        f.write(body)
    print(json.dumps({'output': str(OUTPUT), 'bytes': len(body),
                      'candidates': candidates, 'functions': functions}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    run()
