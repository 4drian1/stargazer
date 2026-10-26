"""Full THS-concept scope with forward-only catalog versions; no market IO."""
from copy import deepcopy
import hashlib
import json

from intraday_sector_leaders_v1 import advance as fixed_advance


def catalog_identity(universe):
    fields = {k: universe.get(k) for k in ('id', 'taxonomy', 'reference_rule', 'known_at')}
    fields['boards'] = sorted(universe.get('boards', []))
    return hashlib.sha256(json.dumps(fields, sort_keys=True, allow_nan=False).encode()).hexdigest()


def advance(previous, decision, cutoff, universe, boards, membership, stocks):
    if universe.get('taxonomy') != 'THS_CONCEPT':
        raise ValueError('FULL_THS_CONCEPT_SCOPE_REQUIRED')
    identity = catalog_identity(universe)
    forwarded = deepcopy(previous)
    changed = bool(previous and previous['universe_id'] != universe['id'])
    if previous:
        if changed:
            if universe.get('parent_id') != previous['universe_id']:
                raise ValueError('CATALOG_PARENT_MISMATCH')
            # Change the current reference only; never mutate the historical state.
            forwarded['universe_id'] = universe['id']
        elif previous['catalog_identity'] != identity:
            raise ValueError('CATALOG_CHANGED_WITHOUT_NEW_VERSION')
    result = fixed_advance(forwarded, decision, cutoff, universe, boards, membership, stocks)
    history = deepcopy(previous['catalog_history']) if previous else []
    if not previous or changed:
        history.append(dict(id=universe['id'], known_at=universe['known_at'], activated_at=decision))
    result.update(catalog_identity=identity, catalog_history=history, scope='FULL_THS_CONCEPT',
                  catalog_change_observed=changed)
    return result
