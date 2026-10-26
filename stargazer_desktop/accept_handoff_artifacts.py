"""Independent readback of final sidecars, weights, packs and packaged resources."""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PyInstaller.archive.readers import CArchiveReader

HERE = Path(__file__).resolve().parent
RISK = HERE.parent / 'stargazer_ml' / 'v5_hybrid' / 'risk_r0_v1'
SOURCE = RISK / 'sidecar_rebuild_v5'
sys.path.insert(0, str(RISK))
import rebuild_features_v1 as rebuild
import jev_adapter_v1 as jev
from oracles_v1 import EvidencePack


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    stop = json.loads((SOURCE / 'stop_record.json').read_bytes())
    assert stop['scoreable'] == '20/20'
    assert all(sha(SOURCE / name) == digest for name, digest in stop['manifest'].items())
    plan = json.loads((SOURCE / 'plan.json').read_bytes())
    assert all(sha(path) == digest for path, digest in plan['inputs'].items())
    ext = RISK / 'evidence_stream_trial_v2' / 'minute_ext_20260908_ledger.json'
    assert all(sha(row['path']) == row['sha256'] for row in json.loads(ext.read_bytes()))
    scored = json.loads((SOURCE / 'scored_20.json').read_bytes())['scored']
    _, fields232, fields266, _ = rebuild.load_spec()
    deltas = {}
    missing = {}
    for version, fields, model in [(232, fields232, rebuild.MODEL_232), (266, fields266, rebuild.MODEL_266)]:
        frame = pd.read_parquet(SOURCE / f'model_inputs_{version}.parquet')
        assert list(frame) == fields and len(frame) == 20
        pred = np.clip(rebuild.load_booster(model).predict(rebuild.native_missing_view(frame, fields)), 0, 1)
        expected = [row[f'amp_{version}'] for row in scored]
        np.testing.assert_array_equal(pred, expected)
        deltas[str(version)] = float(np.max(np.abs(pred - expected)))
        missing[str(version)] = {row['code']: [name for name in fields if pd.isna(frame.iloc[i][name])]
                                 for i, row in enumerate(scored)}
    rows = [json.loads(line) for line in (SOURCE / 'shadow_packs_20260908.jsonl').read_text(encoding='utf-8').splitlines()]
    assert len(rows) == len({row['code'] for row in rows}) == 20
    fields = ('evidence_id', 'asof_ts', 'market_date', 'query_ts', 'candidates',
              'risk_context', 'risk_missing', 'corvus_gap', 'confirmed_plan', 'notes')
    for row in rows:
        pack = EvidencePack(**{key: row[key] for key in fields}).bind()
        assert pack.pack_hash == row['pack_hash'] and not jev._risk_evidence_incomplete(pack)
        assert pack.risk_context['feature_missing'] == missing['232'][row['code']]
    exe = HERE / 'handoff_build_20261021' / 'Stargazer.exe'
    archive = CArchiveReader(str(exe))
    for name in ('web/app.js', 'web/pentacles.js', 'web/index.html'):
        assert hashlib.sha256(archive.extract(name.replace('/', '\\'))).hexdigest() == sha(HERE / name)
    assert not any('.secrets' in name or name == 'config.yaml' for name in archive.toc)
    result = dict(status='PASS', source=str(SOURCE), packs=20, model_readback_delta=deltas,
                  missing_fields=missing, source_hashes_verified=True,
                  bundled_frontend_matches=True, secrets_not_bundled=True, exe_sha256=sha(exe),
                  realtime_ready=False, paid_calls=0, fit=False)
    out = HERE / 'handoff_acceptance_20261021' / 'artifact_acceptance.json'
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({key: value for key, value in result.items() if key != 'missing_fields'}))


if __name__ == '__main__':
    main()
