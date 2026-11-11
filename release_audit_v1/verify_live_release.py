"""Read-only release gate: both manifests and every production reference must agree."""
import gzip
import hashlib
import json
from pathlib import Path
import re
import ast

REQUIRED_WORKERS = ('assistant_graph_worker.py', 'minute_history_v1.py',
                    'on_demand_model_worker.py', 'golden_stock_analysis.py',
                    'minute_plan_review.py', 'sector_rule_scan.py', 'strong_sector_watch_v1.py',
                    'persistent_records_v1.py')


def verify_build_spec(root):
    """Check the maintained whitelist without executing a PyInstaller build."""
    errors = []
    try:
        tree = ast.parse((Path(root)/'stargazer_desktop/Stargazer_demo_safe.spec').read_text(encoding='utf8'))
        entries = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'datas' for t in node.targets):
                for item in node.value.elts:
                    try: entries.append(ast.literal_eval(item))
                    except (ValueError, TypeError): pass
        for name in REQUIRED_WORKERS:
            if (name, '.') not in entries: errors.append('BUILD_WORKER_MISSING:'+name)
        for source, target in entries:
            if source.replace('\\', '/').rstrip('/') in ('config.yaml', 'data'):
                errors.append('BUILD_PRIVATE_INPUT:'+source)
        excludes = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.keyword) and node.arg == 'excludes':
                excludes.update(ast.literal_eval(node.value))
        if not {'torch', 'transformers', 'langgraph'} <= excludes:
            errors.append('BUILD_HEAVY_EXCLUDES_MISSING')
    except (OSError, SyntaxError, ValueError, TypeError, AttributeError) as exc:
        errors.append('BUILD_SPEC_INVALID:'+type(exc).__name__)
    return errors


REFERENCES=(
    'stargazer_desktop/golden_stock_analysis.py',
    'stargazer_desktop/on_demand_model_worker.py',
    'stargazer_desktop/sector_rule_scan.py',
    'stargazer_ml/v5_hybrid/green_common_profile/market_price_evidence_v1.py',
    'stargazer_ml/v5_hybrid/risk_r0_v1/tarot_intraday_risk_v2.py',
    'stargazer_ml/v5_hybrid/risk_r0_v1/live_postclose_job_v1.py',
    'stargazer_ml/v5_hybrid/risk_r0_v1/accept_live_postclose_v1.py')


def verify_release(root):
    root=Path(root);errors=verify_build_spec(root);manifests=[];runtime_names=set()
    for name in REFERENCES:
        names=set(re.findall(r'live_monitor_fingerprints_\d+_v\d+_flow\.json\.gz',
                             (root/name).read_text(encoding='utf8')))
        if len(names)!=1:errors.append('RUNTIME_REFERENCE_INVALID:'+name)
        runtime_names.update(names)
    if len(runtime_names)!=1:errors.append('RUNTIME_REFERENCES_DIVERGE')
    gcp=root/'stargazer_ml/v5_hybrid/green_common_profile'
    tarot_names=set(re.findall(r'tarot_live_manifest_v\d+_\d+_rel\.json\.gz',
                              (gcp/'tarot_live_v2.py').read_text(encoding='utf8')))
    if len(tarot_names)!=1:errors.append('TAROT_REFERENCE_INVALID')
    paths=[root/'stargazer_ml/v5_hybrid/risk_r0_v1'/n for n in sorted(runtime_names)]
    paths += [gcp/n for n in sorted(tarot_names)]
    for path in paths:
        try:
            value=json.loads(gzip.decompress(path.read_bytes()))
            items=value.get('frozen_inputs',value)
            if not items:errors.append('EMPTY_MANIFEST:'+path.name)
            mismatch=[]
            for key,expected in items.items():
                target=Path(key)
                if not target.is_absolute():target=root/target
                if not target.is_file() or hashlib.sha256(target.read_bytes()).hexdigest()!=expected:
                    mismatch.append(key)
            errors += [path.name+':'+key for key in mismatch]
            if 'stargazer_desktop/index_outlook_v1.py' not in items:
                errors.append('UNBOUND_INDEX_REPORT:'+path.name)
            manifests.append(dict(file=path.name,count=len(items),mismatch=mismatch))
        except (OSError,ValueError,TypeError) as exc:errors.append(path.name+':'+str(exc))
    return dict(ok=not errors,manifests=manifests,errors=errors)


if __name__=='__main__':
    result=verify_release(Path(__file__).resolve().parents[1])
    print(json.dumps(result,ensure_ascii=False,indent=2))
    raise SystemExit(0 if result['ok'] else 1)
