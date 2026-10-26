"""Small additive budget for the common-candidate study; frozen v2 stays intact."""
import hashlib
import json
from pathlib import Path

from research_budget_v2 import ROOTS, TOTAL_CAP as PARENT_CAP, bytes_used

ADDENDUM = Path(__file__).parent / 'contracts/research_storage_common_32mib_20261019.json'
config = json.loads(ADDENDUM.read_bytes())
if (config.get('schema') != 'REGISTERED_COMMON_RESEARCH_BUDGET_V3'
        or config.get('status') != 'CHAT_APPROVED_BOUNDED_INCREMENT'
        or config.get('parent_total_bytes') != PARENT_CAP
        or config.get('increment_bytes') != 32 * 1024**2
        or config.get('total_bytes') != PARENT_CAP + 32 * 1024**2
        or config.get('reset_existing_usage') is not False
        or any(config.get(k) is not False for k in
               ('delete_evidence', 'fit_authorized', 'http_authorized', 'deployment_authorized'))
        or {Path(p).resolve() for p in config['roots']} != {p.resolve() for p in ROOTS}):
    raise ValueError('INVALID_COMMON_BUDGET_ADDENDUM')
TOTAL_CAP = config['total_bytes']


class EvidenceWriter:
    def __init__(self, root, cap=28*1024**2, reserve=128*1024):
        self.root, self.cap, self.reserve = Path(root), cap, reserve
        if not any(self.root.resolve().is_relative_to(p.resolve()) for p in ROOTS):
            raise ValueError('OUTPUT_OUTSIDE_REGISTERED_ROOTS')
        if not 0 < reserve < cap <= config['sidecar_batch_cap_bytes']:
            raise ValueError('INVALID_BATCH_BUDGET')
        if bytes_used() + cap > TOTAL_CAP:
            raise ValueError('COMMON_RESERVATION_UNAVAILABLE')
        self.root.mkdir(exist_ok=False)
        self.used, self.outputs = 0, {}

    def write(self, name, body, terminal=False):
        reserve = 0 if terminal else self.reserve
        if Path(name).name != name or name in self.outputs:
            raise ValueError('INVALID_EVIDENCE_NAME')
        if self.used + len(body) + reserve > self.cap or bytes_used() + len(body) + reserve > TOTAL_CAP:
            raise ValueError('COMMON_OUTPUT_BUDGET_STOP')
        with (self.root / name).open('xb') as stream:
            stream.write(body)
        self.used += len(body)
        self.outputs[name] = dict(bytes=len(body), sha256=hashlib.sha256(body).hexdigest())

    def json(self, name, value, terminal=False):
        self.write(name, json.dumps(value, ensure_ascii=False, allow_nan=False,
                                   separators=(',', ':')).encode('utf-8'), terminal)
