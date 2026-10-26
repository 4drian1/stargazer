"""Two-stage market candidate evidence; never a signed training entrypoint."""
from pathlib import Path
import hashlib
import io
import json
import numpy as np
import pandas as pd
from left_side_candidates_v1 import candidates
from tdx_index_acceptance_v1 import grid
from research_budget_v2 import bytes_used, TOTAL_CAP

HERE = Path(__file__).parent
OUT = HERE / 'left_side_market_20261017_v1'
PRICE = HERE / 'local_market_context_20261017_v1'
MOMENTUM = HERE / 'market_momentum_proposal_20261017_v1'
TIME = 'decision_ts_assumed_end'
KEYS = ['symbol', 'frequency_minutes', TIME]
CAP, RESERVE = 16 * 1024**2, 1024**2


def digest(body):
    return hashlib.sha256(body).hexdigest()


def build(price, momentum):
    if price.duplicated(KEYS).any() or momentum.duplicated(KEYS).any():
        raise ValueError('DUPLICATE_MARKET_KEY')
    if not price[KEYS].sort_values(KEYS).reset_index(drop=True).equals(momentum[KEYS].sort_values(KEYS).reset_index(drop=True)):
        raise ValueError('PRICE_INDICATOR_CUTOFF_MISMATCH')
    wanted = [c for c in momentum if c.startswith(('intraday_', 'provisional_daily_'))]
    source = price.merge(momentum[KEYS + wanted], on=KEYS, how='left', validate='one_to_one')
    intraday, daily = [], []
    for symbol, rows in source.groupby('symbol', sort=True):
        days = sorted(rows.day.unique())
        five = rows[rows.frequency_minutes == 5].set_index(TIME)
        day_records = []
        for day in days:
            times = grid(int(day), 5)
            complete = all(t in five.index and bool(five.loc[t, 'source_complete']) and pd.notna(five.loc[t, 'close']) for t in times)
            final = five.loc[times[-1]] if times[-1] in five.index else None
            row = dict(day=int(day), close=float(final.close) if complete else np.nan,
                       emitted=final is not None)
            for metric in ['rsi14', 'macd_dif', 'macd_hist', 'bbi']:
                row[metric] = final['provisional_daily_' + metric] if complete else np.nan
            day_records.append(row)
        day_frame = pd.DataFrame(day_records)
        day_features = candidates(day_frame)
        for i, row in day_frame.iterrows():
            if row.emitted:
                item = dict(symbol=symbol, frequency_minutes=1440, day=int(row.day),
                            decision_ts_assumed_end=int(row.day) * 1000000 + 150000,
                            stage='POSTCLOSE_DAILY_NEXT_SESSION_PLAN_EVIDENCE')
                item.update(day_features.iloc[i].to_dict())
                daily.append(item)
        for step, frame in rows.groupby('frequency_minutes', sort=True):
            if step not in (5, 15):
                raise ValueError('INVALID_FREQUENCY')
            indexed = frame.set_index(TIME)
            times = [t for day in days for t in grid(int(day), int(step))]
            aligned = indexed.reindex(times)
            data = pd.DataFrame({'close': aligned.close.where(aligned.source_complete.eq(True))})
            for metric in ['rsi14', 'macd_dif', 'macd_hist', 'bbi']:
                data[metric] = aligned['intraday_' + metric]
            derived = candidates(data)
            derived.insert(0, TIME, times)
            derived = derived[derived[TIME].isin(indexed.index)].copy()
            if len(derived) != len(indexed):
                raise ValueError('OFF_GRID_SOURCE')
            derived.insert(0, 'frequency_minutes', int(step))
            derived.insert(0, 'symbol', symbol)
            derived['stage'] = 'INTRADAY_LEFT_SIDE_WARNING'
            intraday.append(derived)
    intraday = pd.concat(intraday, ignore_index=True).sort_values(KEYS).reset_index(drop=True)
    feature_names = [c for c in intraday if c.startswith('lag')]
    daily = pd.DataFrame(daily, columns=['symbol', 'frequency_minutes', 'day', TIME, 'stage', *feature_names])
    for name in ['frequency_minutes', 'day', TIME]:
        daily[name] = daily[name].astype('int64')
    daily = daily.sort_values(['symbol', TIME]).reset_index(drop=True)
    for result in [intraday, daily]:
        result['signal_status'] = 'LEFT_SIDE_CANDIDATE_NOT_CONFIRMED_REVERSAL'
        result['right_confirmation_bars'] = 0
        result['training_admitted'] = False
        result['actual_available_at'] = None
        for name in ['symbol', 'stage', 'signal_status']:
            result[name] = result[name].astype(object)
        for name in feature_names:
            result[name] = result[name].astype(float)
        if np.isinf(result[feature_names].to_numpy()).any():
            raise ValueError('INFINITE_OUTPUT')
    return intraday, daily


def run():
    if OUT.exists():
        raise ValueError('OUTPUT_EXISTS_NO_OVERWRITE')
    refs, frames = {}, []
    for root, name in [(PRICE, 'market_price_features.parquet'), (MOMENTUM, 'market_momentum.parquet')]:
        manifest_path = root / 'manifest.json'
        m = json.loads(manifest_path.read_bytes())
        body = (root / name).read_bytes()
        if digest(body) != m['output']['sha256']:
            raise ValueError('ACCEPTED_INPUT_DRIFT')
        refs[str(manifest_path)] = digest(manifest_path.read_bytes())
        refs[str(root/name)] = digest(body)
        frames.append(pd.read_parquet(io.BytesIO(body)))
    for name in ['left_side_candidates_v1.py', 'materialize_left_side_candidates_v1.py',
                 'test_left_side_candidates_v1.py', 'tdx_index_acceptance_v1.py',
                 'contracts/LEFT_SIDE_TWO_STAGE_RULES_V1.json']:
        path = HERE / name
        refs[str(path)] = digest(path.read_bytes())
    price, momentum = frames
    intraday, daily = build(price, momentum)
    checks = []
    for symbol in sorted(price.symbol.unique()):
        p, m = price[price.symbol == symbol], momentum[momentum.symbol == symbol]
        times = sorted(p[TIME].unique())
        for offset in [80, len(times)//2, len(times)-2]:
            cutoff = int(times[offset])
            before = build(p[p[TIME] <= cutoff], m[m[TIME] <= cutoff])
            changed_p, changed_m = p.copy(), m.copy()
            changed_p.loc[changed_p[TIME] > cutoff, 'close'] *= 1.3
            numeric = [c for c in changed_m if c.startswith(('intraday_', 'provisional_daily_'))]
            changed_m.loc[changed_m[TIME] > cutoff, numeric] *= 1.05
            for name in ['intraday_rsi14', 'provisional_daily_rsi14']:
                changed_m[name] = changed_m[name].clip(0, 100)
            changed = build(changed_p, changed_m)
            for full, prefix, altered in zip([intraday, daily], before, changed):
                expected = full[(full.symbol == symbol) & (full[TIME] <= cutoff)].reset_index(drop=True)
                pd.testing.assert_frame_equal(prefix, expected)
                pd.testing.assert_frame_equal(altered[altered[TIME] <= cutoff].reset_index(drop=True), expected)
            checks.append(dict(symbol=symbol, cutoff=cutoff, prefix='PASS', future='PASS'))
    payloads, summary = {}, {}
    for stage, frame in [('intraday', intraday), ('postclose_daily', daily)]:
        sink = io.BytesIO()
        frame.to_parquet(sink, index=False, compression='zstd')
        payloads[stage+'.parquet'] = sink.getvalue()
        features = [c for c in frame if c.startswith('lag')]
        summary[stage] = dict(rows=len(frame), candidate_fields=len(features),
            counts={c:dict(nonnull=int(frame[c].notna().sum()), positives=int((frame[c] == 1).sum()))
                    for c in features if c.endswith('_candidate')})
    manifest = dict(status='LEFT_SIDE_TWO_STAGE_MARKET_EVIDENCE_NOT_FIT_READY', inputs=refs,
        summary=summary, acceptance_cases=checks, old_training_inputs_unchanged=True,
        parameters_status='DAILY_1_2_3_APPROVED_INTRADAY_1_2_3_ENGINEERING_PROPOSAL',
        price_basis='SAME_CUTOFF_CLOSE_ENDPOINTS', right_confirmation_bars=0,
        label_contract_changed=False, postclose_stock_population_built=False,
        fit_run=False, training_ready=False, http_requests=0,
        outputs={n:dict(bytes=len(b), sha256=digest(b)) for n,b in payloads.items()})
    payloads['manifest.json'] = json.dumps(manifest, ensure_ascii=False, allow_nan=False, indent=2).encode()
    if sum(map(len,payloads.values()))+RESERVE>CAP or bytes_used()+sum(map(len,payloads.values()))+RESERVE>TOTAL_CAP:
        raise ValueError('RESEARCH_BUDGET_STOP')
    if any(digest(Path(p).read_bytes()) != sha for p,sha in refs.items()):
        raise ValueError('INPUT_CHANGED_DURING_BUILD')
    OUT.mkdir(exist_ok=False)
    for name, body in payloads.items():
        with (OUT/name).open('xb') as stream:
            stream.write(body)
    for name, expected in [('intraday', intraday), ('postclose_daily', daily)]:
        body = (OUT/(name+'.parquet')).read_bytes()
        assert digest(body) == manifest['outputs'][name+'.parquet']['sha256']
        pd.testing.assert_frame_equal(pd.read_parquet(io.BytesIO(body)), expected)
    acceptance = json.dumps(dict(status='PASS_TWO_STAGE_CAUSAL_EVIDENCE_ONLY',
        manifest_sha256=digest(payloads['manifest.json']), actual_checks=checks,
        intraday_rows=len(intraday), daily_rows=len(daily), training_ready=False), indent=2).encode()
    if bytes_used()+len(acceptance)>TOTAL_CAP:
        raise ValueError('RESEARCH_BUDGET_STOP')
    with (OUT/'readback_acceptance.json').open('xb') as stream:
        stream.write(acceptance)
    print(json.dumps(dict(status=manifest['status'], intraday_rows=len(intraday), daily_rows=len(daily),
                         bytes=sum(map(len,payloads.values()))+len(acceptance), actual_checks=len(checks))))


if __name__ == '__main__':
    run()
