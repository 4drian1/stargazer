"""Effective four-root 4GiB addendum; original frozen contracts remain intact."""
from pathlib import Path
import json
from prepare_historical_risk_v1 import ROOTS,bytes_used

ADDENDUM=Path(__file__).parent/'contracts/research_storage_4gib_20261015.json'
config=json.loads(ADDENDUM.read_bytes())
if (config.get('schema')!='REGISTERED_RESEARCH_BUDGET_V2' or config.get('status')!='CHAT_APPROVED'
        or config.get('total_bytes')!=4*1024**3 or config.get('reset_existing_usage') is not False
        or {Path(p).resolve() for p in config['roots']}!={p.resolve() for p in ROOTS}):
    raise ValueError('INVALID_RESEARCH_BUDGET_ADDENDUM')
TOTAL_CAP=config['total_bytes']
