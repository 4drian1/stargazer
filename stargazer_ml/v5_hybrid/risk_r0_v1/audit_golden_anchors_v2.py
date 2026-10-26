"""Read-only hash and event-invariant audit; no replay output modifications."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def run():
    folder = ROOT / 'golden_anchor_replay_20261017_v2'
    manifest = json.loads((folder / 'manifest.json').read_bytes())
    body = (folder / 'replay.json').read_bytes()
    assert hashlib.sha256(body).hexdigest() == manifest['result_sha256']
    for path, expected in {**manifest['inputs'], **manifest['protected']}.items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == expected
    states = json.loads(body)['states']
    assert len(states) == 442
    for old, new in zip(states, states[1:]):
        assert old['asof_day'] < new['asof_day']
        assert new['low'] < new['high']
        assert new['version'] == old['version'] + bool(new['events'])
        price = new['previous_bar']['close']
        if 'EXPAND_HIGH' in new['events']:
            assert price > old['high'] and new['high'] == price
        if 'EXPAND_LOW' in new['events']:
            assert price < old['low'] and new['low'] == price
        if 'SHRINK_HIGH' in new['events']:
            assert new['run_direction'] == -1 and new['run_count'] >= 3
            assert new['run_has_one_percent_drop']
            assert new['high'] == new['run_origin']['close'] < old['high']
            assert new['high_day'] == new['run_origin']['day']
        if 'SHRINK_LOW' in new['events']:
            assert new['run_direction'] == 1 and new['run_count'] >= 7
            assert new['low'] == new['run_origin']['close'] > old['low']
            assert new['low_day'] == new['run_origin']['day']
        assert not new['training_ready'] and new['actual_known_at'] is None
    print('PASS: result/input/protected hashes; 441 transitions and asymmetric event invariants; 0 HTTP / 0 writes')


if __name__ == '__main__':
    run()
