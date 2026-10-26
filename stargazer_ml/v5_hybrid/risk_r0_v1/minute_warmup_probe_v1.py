"""Local fixed-six single-clean-source warmup diagnostics, not live feature admission."""
from collections import Counter
from copy import deepcopy
import json
from pathlib import Path

import pyarrow.parquet as pq

from minute_risk_features_v1 import aggregate, features, parse_ts, source_end
from price_verification_guarded_run import sha
from research_budget_v2 import bytes_used, TOTAL_CAP

HERE = Path(__file__).resolve().parent
HYBRID = HERE.parent
CACHE = HYBRID.parent / 'pullback_train'
OUT = HERE / 'minute_warmup_probe_20261016_v1'
CAP = 2 * 1024**2


def timestamp_profile(rows, day):
    group = [r for r in rows if r['date'] // 1000000 == day]
    keys = [r['date'] for r in group]
    return dict(rows=len(group), unique_timestamps=len(set(keys)),
        duplicate_timestamps=len(keys) - len(set(keys)),
        missing_boundary_hhmm=[hhmm for hhmm in (930, 1130, 1300, 1500)
                               if day * 1000000 + hhmm * 100 not in keys])


def compare_sources(left, right):
    def index(rows):
        result = {}
        for r in rows:
            result.setdefault((r['code'], r['date']), set()).add(
                tuple(r[k] for k in ('open', 'high', 'low', 'close', 'volume')))
        return result
    a, b = index(left), index(right)
    common = a.keys() & b.keys()
    return dict(shared=len(common), left_only=len(a.keys() - b.keys()), right_only=len(b.keys() - a.keys()),
        price_different=sum({v[:4] for v in a[k]} != {v[:4] for v in b[k]} for k in common),
        volume_different=sum({v[4] for v in a[k]} != {v[4] for v in b[k]} for k in common),
        conflicting_left_keys=sum(len(v) > 1 for v in a.values()),
        conflicting_right_keys=sum(len(v) > 1 for v in b.values()))


def rebuild(rows, code, previous, day, semantics):
    bars, issues = aggregate(rows, code, [previous, day], semantics)
    fields = features(bars)
    checks = []
    for i in range(16, 32):
        end = parse_ts(bars[i]['bar_end'])
        prefix = [r for r in rows if source_end(r, semantics) <= end]
        pb, _ = aggregate(prefix, code, [previous, day], semantics)
        if pb[:i+1] != bars[:i+1] or features(pb)[:i+1] != fields[:i+1]:
            raise ValueError('PREFIX_MISMATCH')
        changed = deepcopy(rows)
        count = 0
        for r in changed:
            if source_end(r, semantics) > end:
                count += 1
                for k in ('open', 'high', 'low', 'close'):
                    r[k] *= 2
                r['volume'] *= 3
        fb, _ = aggregate(changed, code, [previous, day], semantics)
        if fb[:i+1] != bars[:i+1] or features(fb)[:i+1] != fields[:i+1]:
            raise ValueError('PERTURBATION_MISMATCH')
        checks.append(dict(bar_end=bars[i]['bar_end'], prefix_equal=True, mutated_rows=count,
            future_test='PASS' if count else 'NOT_TESTED_NO_FUTURE'))
    # Independent recursive arithmetic on complete consecutive bars, not features().
    previous_close = None
    ups, downs, gain, loss, independent = [], [], None, None, []
    for bar in bars:
        if bar['status'] == 'INCOMPLETE':
            previous_close, ups, downs, gain, loss = None, [], [], None, None
            independent.append(None)
            continue
        if previous_close is not None:
            diff = bar['close'] - previous_close
            if gain is None:
                ups.append(max(diff, 0)); downs.append(max(-diff, 0))
                if len(ups) == 14:
                    gain, loss = sum(ups) / 14, sum(downs) / 14
            else:
                gain, loss = (13 * gain + max(diff, 0)) / 14, (13 * loss + max(-diff, 0)) / 14
        value = None if gain is None else (50 if gain == loss == 0 else
                100 if loss == 0 else 100 - 100 / (1 + gain / loss))
        independent.append(value)
        previous_close = bar['close']
    for value, field in zip(independent, fields):
        if (value is None) != (field['rsi14'] is None) or (value is not None and abs(value - field['rsi14']) > 1e-10):
            raise ValueError('INDEPENDENT_RSI_MISMATCH')
    return dict(assumption=semantics, source_clock_certified=False, available_at=None,
        p_bar_status=dict(Counter(b['status'] for b in bars[:16])),
        e_bar_status=dict(Counter(b['status'] for b in bars[16:])), issues=issues, checks=checks,
        e_rsi_nonnull=sum(f['rsi14'] is not None for f in fields[16:]),
        e_same_slot_volume_nonnull=sum(f['volume_ratio_previous_session_same_slot'] is not None for f in fields[16:]),
        independent_nonnull_rsi_checked=sum(v is not None for v in independent),
        e_day=[dict(bar=b, fields=f) for b, f in zip(bars[16:], fields[16:])])


def run():
    if OUT.exists():
        raise ValueError('OUTPUT_EXISTS_NO_RESTART')
    if bytes_used() + CAP > TOTAL_CAP:
        raise ValueError('REGISTERED_BUDGET')
    refs = {}

    def track(path):
        path = Path(path).resolve()
        refs[str(path)] = sha(path)
        return path

    for name in ('minute_warmup_probe_v1.py', 'test_minute_warmup_probe_v1.py',
                 'minute_risk_features_v1.py', 'research_budget_v2.py'):
        track(HERE / name)
    old_path = track(HERE / 'intraday_cache_probe_20261016_v1/diagnostics.json')
    old = json.loads(old_path.read_bytes())
    old_manifest = json.loads(track(old_path.with_name('manifest.json')).read_bytes())
    for ref in old_manifest['inputs']:
        if sha(Path(ref['path'])) != ref['sha256']:
            raise ValueError('OLD_FROZEN_INPUT_DRIFT')
    calendar = json.loads(track(HERE / 'historical_stock_day_preparation_v1/manifest.json').read_bytes())['calendar_dates']
    source_docs = [HYBRID / 'green_common_profile/contracts/car_v0_1_run_sheet_draft.md',
        HYBRID / 'opportunity_v2/gate1c/C0_MINUTE_INVENTORY.md',
        HYBRID / 'green_v2/shadow_bypass/source_semantics_evidence.md',
        HYBRID / 'green_v2/shadow_bypass/source_semantics_evidence.json',
        CACHE / 'fetch_minute_2025.py', CACHE / 'fetch_minute_2026.py',
        Path('D:/a股数据/free-stockdb-main/pybao/stock_sdk.py')]
    for path in source_docs:
        track(path)
    cases = []
    for case in old['results']:
        code, previous, day = case['code'], case['P'], case['E']
        if calendar[calendar.index(previous) + 1] != day:
            raise ValueError('P_E_RELATION')
        clean = CACHE / f'minute_cache_clean/{code}.parquet'
        item = dict(code=code, P=previous, E=day, source=str(clean),
            source_exists=clean.exists(), training_eligible=False, live_feature_admitted=False)
        if not clean.exists():
            item.update(status='NO_SINGLE_SOURCE_FILE', scenarios=[])
        else:
            rows = [r for r in pq.read_table(track(clean)).to_pylist()
                    if r['date'] // 1000000 in (previous, day)]
            item.update(status='DIAGNOSTIC_ONLY', p_profile=timestamp_profile(rows, previous),
                e_profile=timestamp_profile(rows, day),
                volume_repaired_rows=sum(r.get('volume_repaired') is True for r in rows),
                scenarios=[rebuild(rows, code, previous, day, s) for s in ('START', 'END')])
            raw = CACHE / f'minute_cache/{code}.parquet'
            if raw.exists():
                raw_rows = [r for r in pq.read_table(track(raw)).to_pylist()
                            if r['date'] // 1000000 in (previous, day)]
                item['raw_clean_comparison'] = compare_sources(raw_rows, rows)
            original = HYBRID / f'green_common_profile/gcp_round_b_small_sample_v1/minute_ext/{code}.parquet'
            original_rows = pq.read_table(track(original)).to_pylist()
            item['old_primary_comparison'] = compare_sources(original_rows, rows)
        cases.append(item)
    target_days = sorted({c['E'] for c in cases})
    panel_inventory = []
    panel_root = HYBRID / 'opportunity_v2/gate1c/panels'
    for path in sorted(panel_root.glob('sector_sametime_proxy_*.parquet')):
        dates = sorted(set(pq.read_table(track(path), columns=['trade_date']).column(0).to_pylist()))
        panel_inventory.append(dict(path=str(path), dates=dates, target_date_overlap=sorted(set(dates) & set(target_days))))
    report = dict(cases=cases, fixed_population=6, present_source_files=sum(c['source_exists'] for c in cases),
        clock_verdict='UNVERIFIED_FOR_202501_SOURCE',
        clock_reasons=['CAR_START_IS_ENGINEERING_CONVENTION_NOT_CACHE_PROVENANCE',
            'GATE1C_EXECUTION_GRID_AND_SDK_END_GROUPING_NOT_PHYSICAL_INTERVAL_PROOF',
            'SEPTEMBER_VENDOR_COMPARISON_NOT_JANUARY_SOURCE_BINDING',
            '1500_PHYSICAL_INTERVAL_UNRESOLVED'],
        source_documents=[str(p) for p in source_docs], sector_panel_inventory=panel_inventory,
        sector_inventory_scope=str(panel_root), missing_target_dates=target_days,
        sector_known_at_verified=False, sse_15m_bound=False, prior_leader_list_bound=False,
        actual_arrival_replayed=False, fit_ready=False, no_fetch=True)
    for path, digest in refs.items():
        if sha(Path(path)) != digest:
            raise ValueError('INPUT_CHANGED_DURING_RUN')
    encode = lambda value: (json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2) + '\n').encode('utf8')
    bodies = {'diagnostics.json': encode(report)}
    import hashlib
    bodies['manifest.json'] = encode(dict(inputs=[dict(path=p, sha256=h) for p, h in sorted(refs.items())],
        outputs=[dict(path=name, sha256=hashlib.sha256(body).hexdigest(), bytes=len(body)) for name, body in bodies.items()]))
    total = sum(map(len, bodies.values()))
    if total > CAP or bytes_used() + total > TOTAL_CAP:
        raise ValueError('WRITE_BUDGET')
    OUT.mkdir(exist_ok=False)
    for name, body in bodies.items():
        with (OUT / name).open('xb') as stream:
            stream.write(body)
    print(json.dumps(dict(output=str(OUT), bytes=total, present=report['present_source_files'], fit_ready=False)))


if __name__ == '__main__':
    run()
