"""Three display adapters reuse SnapshotV2; no fusion or model promotion."""
from dataclasses import asdict
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import sys

try:
    from snapshot_contracts_v2 import SnapshotHeaderV2, SnapshotV2
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent /
                          'stargazer_ml/v5_hybrid/opportunity_v2/contracts'))
    from snapshot_contracts_v2 import SnapshotHeaderV2, SnapshotV2


def slots(value):
    now = datetime.now(timezone.utc)
    identity = hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                        allow_nan=False).encode()).hexdigest()
    result = {}
    for slot, kind, reason in [('selection', 'opportunity', 'RULE_OBSERVATION_NOT_VALIDATED_PROFIT'),
                               ('information', 'context', 'INFORMATION_MODEL_NOT_CONNECTED'),
                               ('risk', 'risk', 'RISK_MODEL_NOT_READY')]:
        header = SnapshotHeaderV2(
            snapshot_id=f'{slot}-{identity}', schema_version=f'{kind}.v2',
            frozen_artifact_ref=(f'rule_snapshot:PAGE_OF_PENTACLES_DEMO_V1:{identity}' if slot == 'selection'
                                 else f'unavailable:unregistered:{identity}'),
            symbol='CANDIDATE_POOL', asof_ts=now.isoformat(), available_at=now.isoformat(),
            created_ts=now.isoformat(), expires_at=(now + timedelta(minutes=5)).isoformat(),
            quality_state='DEGRADED' if slot == 'selection' and value.get('schema') == 'PAGE_OF_PENTACLES_DEMO_V1' else 'UNKNOWN',
            data_scope='DISPLAY_ONLY', missing_reason=reason, input_refs=(identity,))
        payload = ({'data_date': value.get('data_date'), 'monitor_date': value.get('monitor_date'),
                    'candidate_count': len(value.get('rows', [])), 'snapshot_hash': identity,
                    'timestamp_semantics': 'UI读取时刻，不是历史源到达时刻',
                    'source_schema': value.get('schema')} if slot == 'selection' else {})
        result[slot] = asdict(SnapshotV2(kind, header, payload))
    return {'shell_version': 'stable-shell-v1', 'name': '稳定应用外壳',
            'models': result, 'interaction_enabled': False, 'trading_authority': False}
