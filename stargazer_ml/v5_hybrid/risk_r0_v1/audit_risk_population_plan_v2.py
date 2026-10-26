"""Independent arithmetic/sample readback. No fetch, labels, modelling or file overwrite."""
from collections import Counter
import gzip
from hashlib import sha256
import json
from pathlib import Path

from price_verification_guarded_run import encoded,sha
from research_budget_v2 import bytes_used,TOTAL_CAP
from risk_fit_eval_v1 import board


def run():
    root=Path(__file__).resolve().parent
    out=root/'population_expansion_plan_acceptance_v2.json'
    if out.exists():
        raise ValueError('AUDIT_EXISTS_NO_OVERWRITE')
    folder=root/'risk_population_expansion_plan_v2_20261015'
    stop=json.loads((folder/'stop_record.json').read_bytes())
    assert all(sha(folder/r['name'])==r['sha256'] for r in stop['files'])
    refs=json.loads((folder/'input_manifest.json').read_bytes())['inputs']
    assert all(sha(r['path'])==r['sha256'] for r in refs)
    universe=json.loads(gzip.decompress((folder/'visible_universe.json.gz').read_bytes()))
    summary=json.loads((folder/'summary.json').read_bytes())
    selected=json.loads((folder/'selected_codes.json').read_bytes())
    parent=universe['universe_codes']
    counts=Counter(board(c) for c in parent)
    target=400
    quotient={b:(target*n)//len(parent) for b,n in counts.items()}
    left=target-sum(quotient.values())
    order=sorted(counts,key=lambda b:(-((target*counts[b])%len(parent)),b))
    for b in order[:left]:
        quotient[b]+=1
    expected=[]
    for b,k in quotient.items():
        ranked=sorted([c for c in parent if board(c)==b],key=lambda c:(sha256(('R0_EXPANSION_V1|'+c).encode()).hexdigest(),c))
        expected+=ranked[:k]
    assert sorted(expected)==[r['code'] for r in selected]
    assert quotient==summary['selected_board_counts'] and dict(counts)==summary['universe_board_counts']
    assert len(expected)==len(set(expected))==target
    assert all(any(int(y)>=2019 and int(y)<=2025 and n>0 for y,n in universe['by_code'][c]['year_rows'].items()) for c in parent)
    assert sum(sum(r['year_rows'].values()) for r in universe['by_code'].values())+sum(universe['excluded_prefix_rows'].values())==summary['source_physical_rows']
    for r in selected:
        assert all(r[k]==v for k,v in universe['by_code'][r['code']].items())
        assert r['training_eligible'] is False and r['source_admission']=='NOT_ASSESSED_NO_NEW_LABELS'
        assert all(sha(x['path'])==x['sha256'] for x in r['external_refs']+r['native_refs'])
    old=json.loads((root/'risk_population_expansion_plan_v1_20261015/selected_codes.json').read_bytes())
    assert [r['code'] for r in old]==[r['code'] for r in selected]
    pilot_path=root/'contracts/r0_expansion_first30_request_draft_v1.json'
    pilot=json.loads(pilot_path.read_bytes())
    assert all(sha(pilot[k]['path'])==pilot[k]['sha256'] for k in ('parent_selected','parent_summary'))
    missing=sorted([r for r in selected if not r['archived_external_page_available']],key=lambda r:(r['selection_hash'],r['code']))
    assert [r['code'] for r in pilot['codes']]==[r['code'] for r in missing[:30]]
    assert pilot['external_requests_cap']==30 and pilot['smoke_requests_included'] is True
    assert pilot['fetch_authorized'] is False and pilot['fit_authorized'] is False and pilot['retries']==0
    result=dict(status='MECHANICAL_PLAN_READBACK_PASS_NOT_DATA_ADMISSION',frozen_input_hashes_verified=len(refs),
        universe_codes=len(parent),selected_codes=target,selected_code_list_equal_v1=True,
        quota_and_hash_selection_independently_reproduced=True,all_selected_evidence_admission_pending=True,
        pilot_code_list_verified=30,fetch_authorized=False,fit_authorized=False,network_requests=0,
        registered_actual_bytes=bytes_used(),ignored_new_files=summary['ignored_new_files'],
        plan_manifest_sha256=sha(folder/'input_manifest.json'),plan_stop_sha256=sha(folder/'stop_record.json'),
        pilot_draft_sha256=sha(pilot_path),audit_code_sha256=sha(__file__))
    data=encoded(result)
    assert len(data)<1024**2 and bytes_used()+len(data)<=TOTAL_CAP
    with out.open('xb') as f:
        f.write(data)
    print(json.dumps(result,ensure_ascii=False))


if __name__=='__main__':
    run()
