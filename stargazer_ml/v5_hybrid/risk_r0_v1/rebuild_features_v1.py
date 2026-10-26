"""Thin orchestrator: rebuild 232/266 frozen-framework features for verified-OK stargazer candidates.

New file; consumes frozen artifacts read-only. Zero network, zero fits.
Pipeline:
  1) arrays() deterministic replay (frozen builder, candidate days only) -> r5/r15 returns matrix
  2) candidate-row source frame: frozen joins (shared_<month>, market tables, joint_parts,
     context sidecar, nh sidecars) + recomputed own/nh where frozen rows are absent
     + frozen overlay attach() for repaired (ar_) semantics
  3) replay acceptance on >=20 frozen test-segment keys vs predictions.parquet (both models)
  4) inference for the 228 verified-OK candidate records, per-record isolation
"""
import argparse
import gzip
import hashlib
import io
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

HERE = Path(__file__).resolve().parent
TIME = 'decision_ts_assumed_end'
OUT = HERE / 'rebuild_features_v1'
TMP_OK = HERE / '_tmp_coverage_assess' / 'per_record_695.csv'
SEG_EXTENSION = 'EXTENSION_RESEARCH'
TOLERANCE_ABS = 1e-4
TOLERANCE_REL = 1e-3
N_REPLAY_SAMPLES = 24

MODEL_232 = HERE / 'auxiliary_120m_fixed_fit_20261018_v1'
MODEL_266 = HERE / 'common_candidate_fixed_fit_20261019_v1'
MULTIBOARD = HERE / 'multiboard_full_features_20261018_v1'
REPAIR = HERE / 'multiboard_observation_repair_20261018_v1'
CONDITIONAL = HERE / 'conditional_board_observations_20261018_v1'
JOINT = HERE / 'joint_risk_features_20261017_v3'
CONTEXT_LOO = HERE / 'joint_risk_context_loo_20261017_v3'
MOMENTUM = HERE / 'joint_momentum_review_20261017_v1' / 'market_features_by_cutoff.parquet'
SHARED_MARKET = JOINT / 'shared_market_and_leaders.parquet'
LEDGER = HERE / 'relative_stall_5m_20261018_v1' / 'raw_input_ledger.json'
QUARANTINE = HERE.parents[1] / 'pullback_train' / 'quarantine_2026.csv'
CATALOG = HERE / 'ths_full_concept_scope_20261017_v1.json'
OVERLAY = CONDITIONAL / 'observation_differences.parquet'
PREPROC_CONFIG = HERE / 'direct_warning_comparison_preparation_20261018_v1' / 'config.json'
MARKET_FEATURES = HERE / 'local_market_context_20261017_v1' / 'market_price_features.parquet'
CANDIDATE_REPLAY = HERE.parent / 'green_common_profile' / 'round_a_v0_1_impl_fix' / 'full_intraday_replay_v2.parquet'
MASTER_TABLE = HERE.parent / 'green_common_profile' / 'round_a_v0_1_impl_fix' / 'full_master_table_v2.parquet'
GOLDEN_STATES = HERE / 'golden_anchor_replay_20261017_v2' / 'replay.json'


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1024 ** 2), b''):
            h.update(block)
    return h.hexdigest()


def log(stage, **kwargs):
    print(json.dumps(dict(stage=stage, elapsed=round(time.time() - T0, 1), **kwargs)), flush=True)


T0 = time.time()


def load_spec():
    config = json.loads(PREPROC_CONFIG.read_bytes())
    mapping = config['logical_to_source']
    fields232 = json.loads((MODEL_232 / 'lightgbm_preprocessing.json').read_bytes())['columns']
    fields266 = json.loads((MODEL_266 / 'lightgbm_preprocessing.json').read_bytes())['columns']
    assert config['arms']['C1'] == fields232, 'FROZEN_FIELD_LIST_DRIFT_232'
    assert fields266[:232] == fields232, 'FROZEN_FIELD_LIST_DRIFT_266'
    extra = fields266[232:]
    assert len(extra) == 34, 'FROZEN_FIELD_LIST_DRIFT_266_COUNT'
    return mapping, fields232, fields266, extra


def load_booster(model_dir):
    import lightgbm as lgb
    text = gzip.decompress((model_dir / 'lightgbm_model.txt.gz').read_bytes()).decode('utf-8')
    return lgb.Booster(model_str=text)


def native_missing_view(frame, fields):
    values = frame[fields].to_numpy(dtype=float)
    if np.isinf(values).any():
        raise ValueError('INFINITE_FEATURE_VALUE')
    return np.c_[values, np.isnan(values).astype(float)]


# ---------------------------------------------------------------- replay core
def replay(days):
    """Deterministic replay over candidate days via the frozen arrays() builder."""
    from build_continuous_motion_population_v1 import arrays
    log('replay_start', days=len(days))
    codes, members, native, report = arrays(days)
    log('replay_done', codes=len(codes), boards=len(members),
        raw_files=report.get('raw_files'), compared=report.get('original_15m_cells_replayed'))
    return codes, members, native


# ---------------------------------------------------------------- frame build
def build_candidate_frame(keys, days, codes, members, native, mapping, diff):
    """keys: DataFrame[code, TIME]. Returns source-level frame with overlay applied."""
    import pyarrow as pa
    from read_conditional_board_observations_v1 import attach
    from joint_fit_preparation_v1 import derive

    frame = keys.copy().reset_index(drop=True)
    frame['cutoff_index'] = frame[TIME].map(TS_TO_CI)
    if frame.cutoff_index.isna().any():
        raise ValueError('CUTOFF_INDEX_MISS')
    frame['cutoff_index'] = frame.cutoff_index.astype(np.int64)
    frame['segment'] = SEG_EXTENSION
    needed_ci = set(frame.cutoff_index)
    needed_ts = set(frame[TIME])

    # --- shared_<month>: mb5_/mb15_ + golden (full cutoff grid) ---
    report = json.loads((MULTIBOARD / 'report.json').read_bytes())
    spec = json.loads((MULTIBOARD / 'feature_spec.json').read_bytes())
    shared_cols = spec['shared']
    key_months = set((keys[TIME] // 100000000).astype(int))
    parts = [p for p in report['parts'] if int(p['month']) in key_months]
    shared_frames = []
    for part in parts:
        t = pq.read_table(MULTIBOARD / part['shared'], columns=[TIME] + shared_cols).to_pandas()
        shared_frames.append(t)
    shared_all = pd.concat(shared_frames, ignore_index=True)
    before = len(frame)
    frame = frame.merge(shared_all, on=TIME, how='left', validate='many_to_one', sort=False)
    log('joined_shared_months', rows=len(frame), months=[p['month'] for p in parts])

    # --- shared_market_and_leaders: market identity fields per cutoff ---
    tf_cols = [c for c in SM_ALL if c.startswith('sse_15m_time_fraction_distance_')]
    sm = pq.read_table(SHARED_MARKET, columns=['cutoff_index'] + SM_COLS + tf_cols).to_pandas()
    frame = frame.merge(sm, on='cutoff_index', how='left', validate='many_to_one', sort=False)

    # --- momentum 54 fields per cutoff ---
    mom = pd.read_parquet(MOMENTUM, columns=[TIME] + MOM_COLS)
    frame = frame.merge(mom, on=TIME, how='left', validate='many_to_one', sort=False)

    # --- joint_parts rows: stock_* + time_fraction columns for needed cutoffs ---
    jp_rows = []
    jmanifest = json.loads((JOINT / 'manifest.json').read_bytes())
    for item in jmanifest['joint_parts']:
        t = pq.read_table(JOINT / item['path'],
                          columns=['code', 'cutoff_index'] + JP_COLS,
                          filters=[('cutoff_index', 'in', sorted(needed_ci))]).to_pandas()
        if len(t):
            jp_rows.append(t)
    jp = pd.concat(jp_rows, ignore_index=True)
    jp['code'] = jp.code.astype(str)
    frame = frame.merge(jp, on=['code', 'cutoff_index'], how='left', validate='one_to_one', sort=False)
    log('joined_joint_parts', rows=len(jp), matched=int(frame.stock_day_return_raw.notna().sum()))

    # --- context LOO sidecar ---
    cmanifest = json.loads((CONTEXT_LOO / 'manifest.json').read_bytes())
    ctx_rows = []
    for item in cmanifest['parts']:
        t = pq.read_table(CONTEXT_LOO / item['context'],
                          columns=['code', 'cutoff_index'] + CTX_COLS,
                          filters=[('cutoff_index', 'in', sorted(needed_ci))]).to_pandas()
        if len(t):
            ctx_rows.append(t)
    ctx = pd.concat(ctx_rows, ignore_index=True)
    ctx['code'] = ctx.code.astype(str)
    frame = frame.merge(ctx, on=['code', 'cutoff_index'], how='left', validate='one_to_one', sort=False)

    # --- nh sidecars (old semantics; may be absent for non-cohort keys -> recomputed below) ---
    nh_shared = pd.read_parquet(REPAIR / 'shared.parquet')
    nh_own = pd.read_parquet(REPAIR / 'own.parquet')
    frame = frame.merge(nh_shared, on=TIME, how='left', validate='many_to_one', sort=False)
    frame = frame.merge(nh_own[['code', TIME] + NH_OWN_COLS], on=['code', TIME],
                        how='left', validate='many_to_one', sort=False)

    # --- own_<month> cohort rows where present (frozen truth for cohort keys) ---
    own_cols = spec['own']
    own_frames = []
    for part in parts:
        t = pq.read_table(MULTIBOARD / part['own'],
                          columns=['code', TIME] + own_cols).to_pandas()
        t['code'] = t.code.astype(str)
        own_frames.append(t)
    own_all = pd.concat(own_frames, ignore_index=True)
    frame = frame.merge(own_all, on=['code', TIME], how='left', validate='one_to_one', sort=False,
                        suffixes=('', '_frozenown'))
    own_frozen_hit = int(frame.own5_mean.notna().sum())
    log('joined_frozen_own', hits=own_frozen_hit)

    # --- recompute own/nh from replay for rows missing frozen values ---
    frame = recompute_own(frame, days, codes, members, native, own_cols)
    log('own_recomputed', filled=int(frame.own5_mean.notna().sum()) - own_frozen_hit)

    # --- derived fields (frozen derive()) ---
    frame = derive(frame)

    # --- repaired-semantics overlay (frozen attach()) ---
    frame = attach(frame, diff)
    return frame


def recompute_own(frame, days, codes, members, native, own_cols):
    from build_multiboard_full_features_v1 import market_arrays
    from conditional_board_patch_v1 import with_new_high
    from multiboard_risk_features_v1 import matrix_features, summarize
    ci = {c: i for i, c in enumerate(codes)}
    di = {d: i for i, d in enumerate(days)}
    market = pd.read_parquet(MARKET_FEATURES)
    missing = frame.own5_mean.isna() | frame.own15_mean.isna()
    todo = frame.loc[missing, ['code', TIME, 'cutoff_index']]
    for month in sorted({d // 100 for d in days}):
        md = [d for d in days if d // 100 == month]
        a, b = di[md[0]], di[md[-1]] + 1
        sub = todo[todo[TIME] // 100000000 == month]
        if sub.empty:
            continue
        for frequency in (5, 15):
            width = 240 // frequency
            grid, markets = market_arrays(market, md, frequency)
            positions = {t: i for i, t in enumerate(grid)}
            returns = native[frequency][:, a * width:b * width]
            for code, group in sub.groupby('code', sort=False):
                group = group[group[TIME].map(positions).notna()]
                if group.empty:
                    continue
                idx = ci.get(code)
                if idx is None:
                    continue
                related = [m for m in members if idx in m]
                if not related:
                    continue
                small = matrix_features(returns, related, frequency, exclude_index=idx)
                own = with_new_high(summarize(small, markets, frequency, f'own{frequency}_', own=True),
                                    markets, frequency, f'own{frequency}_')
                at = np.array([positions[int(t)] for t in group[TIME]], dtype=int)
                for name in own_cols:
                    if name.startswith(f'own{frequency}_'):
                        frame.loc[group.index, name] = own[name][at]
        log('own_recompute_month', month=month, rows=len(sub))
    return frame


def resolve_logical(frame, mapping, fields):
    data = {}
    unresolved = []
    for f in fields:
        src = mapping[f]
        if src in frame:
            data[f] = frame[src].to_numpy(dtype=float)
        else:
            unresolved.append((f, src))
    if unresolved:
        raise ValueError('UNRESOLVED_SOURCE_COLUMNS:' + json.dumps(unresolved[:20]))
    return pd.DataFrame(data, index=frame.index)


# ---------------------------------------------------------------- main flows
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', action='store_true', required=True)
    args = ap.parse_args()
    if OUT.exists():
        raise ValueError('OUTPUT_EXISTS_NO_RESTART')
    OUT.mkdir()
    writer = {}
    def put(name, body, terminal=False):
        with (OUT / name).open('xb') as stream:
            stream.write(body)
        writer[name] = dict(bytes=len(body), sha256=hashlib.sha256(body).hexdigest())

    mapping, fields232, fields266, extra266 = load_spec()
    put('plan.json', json.dumps(dict(
        status='REBUILD_EXTENSION_RESEARCH_NOT_FIT', generated=time.strftime('%Y-%m-%d %H:%M:%S'),
        inputs={str(p): sha(p) for p in [MODEL_232 / 'lightgbm_model.txt.gz', MODEL_232 / 'lightgbm_preprocessing.json',
                  MODEL_232 / 'manifest.json', MODEL_266 / 'lightgbm_model.txt.gz',
                  MODEL_266 / 'lightgbm_preprocessing.json', MODEL_266 / 'manifest.json',
                  PREPROC_CONFIG, MULTIBOARD / 'feature_spec.json', MULTIBOARD / 'report.json',
                  REPAIR / 'shared.parquet', REPAIR / 'own.parquet', OVERLAY, SHARED_MARKET, MOMENTUM,
                  LEDGER, QUARANTINE, CATALOG, MARKET_FEATURES, GOLDEN_STATES]},
        tolerance=dict(abs=TOLERANCE_ABS, rel=TOLERANCE_REL, rule='wider'),
        n_replay_samples=N_REPLAY_SAMPLES, segment=SEG_EXTENSION,
        frozen_builders_unmodified=True, http_requests=0, fit=False), indent=1).encode())

    # keys
    ok = pd.read_csv(TMP_OK, dtype={'code': str})
    ok = ok[ok.reason == 'OK'].copy()
    ok['code'] = ok.code.str.zfill(6)
    ok = ok.rename(columns={'cutoff_ts': TIME})
    ok[TIME] = ok[TIME].astype(np.int64)
    if ok.duplicated(['code', TIME]).any():
        raise ValueError('DUPLICATE_CANDIDATE_KEY')
    days = sorted(set((ok[TIME] // 1000000).astype(int)))
    log('keys_loaded', records=len(ok), days=len(days))

    # global lookups
    global TS_TO_CI, SM_COLS, MOM_COLS, JP_COLS, CTX_COLS, NH_OWN_COLS, SM_ALL
    SM_ALL = pq.read_table(SHARED_MARKET).column_names
    mom_all = set(pd.read_parquet(MOMENTUM).columns)
    mapping_sources = set(mapping.values())
    base_sources = {s for s in mapping_sources if not s.startswith('ar_')}
    sm_cols = [c for c in SM_ALL if c in base_sources]
    derived = {'time_node_nearest_abs_distance', 'session_is_pm', 'context_sector_minus_sse_return_raw',
               'context_stock_minus_sector_return_raw', 'context_stock_minus_leader_return_raw'}
    mom_cols = sorted((set(base_sources) & mom_all) - set(sm_cols) - derived)
    ctx_cols = sorted({s for s in base_sources if s.startswith('context_')} - derived)
    jp_cols = sorted({s for s in base_sources if s.startswith('stock_')})
    nh_own_cols = [c for c in pq.read_table(REPAIR / 'own.parquet').column_names if c.startswith('nh_own')]
    golden_cols = sorted({s for s in base_sources if s.startswith('new_sse_golden')})
    shared_cols = sorted({s for s in base_sources if s.startswith(('mb5_', 'mb15_'))} | set(golden_cols))
    own_cols = sorted({s for s in base_sources if s.startswith(('own5_', 'own15_'))})
    if not shared_cols or not own_cols:
        raise ValueError('COLUMN_RESOLUTION_EMPTY')
    TS_TO_CI = dict(zip(pd.read_parquet(SHARED_MARKET, columns=[TIME, 'cutoff_index'])[TIME].astype(np.int64),
                        pd.read_parquet(SHARED_MARKET, columns=[TIME, 'cutoff_index']).cutoff_index))
    SM_COLS, MOM_COLS, JP_COLS, CTX_COLS, NH_OWN_COLS = sm_cols, mom_cols, jp_cols, ctx_cols, nh_own_cols
    covered = set(sm_cols) | set(mom_cols) | set(ctx_cols) | set(jp_cols) | set(nh_own_cols) | set(shared_cols) \
        | set(own_cols) | {s[3:] for s in mapping_sources if s.startswith('ar_')} | derived
    uncovered = sorted(s for s in base_sources if s not in covered)
    if uncovered:
        raise ValueError('UNCOVERED_SOURCE_COLUMNS:' + json.dumps(uncovered))
    log('column_resolution', sm=len(sm_cols), mom=len(mom_cols), jp=len(jp_cols), ctx=len(ctx_cols),
        nh_own=len(nh_own_cols), shared=len(shared_cols), own=len(own_cols))

    diff = pd.read_parquet(OVERLAY)

    # replay acceptance samples chosen before replay so their days are covered
    pred232 = pd.read_parquet(MODEL_232 / 'predictions.parquet')
    test = pred232[pred232.segment == 'test'].reset_index(drop=True)
    pick = np.linspace(0, len(test) - 1, N_REPLAY_SAMPLES).astype(int)
    sample = test.iloc[pick].reset_index(drop=True)
    sample_keys = sample[['code', TIME]].copy()
    sample_keys['code'] = sample_keys.code.astype(str).str.zfill(6)
    days = sorted(set(days) | set((sample_keys[TIME] // 1000000).astype(int)))
    log('days_extended_for_samples', days=len(days))

    codes, members, native = replay(days)

    booster232 = load_booster(MODEL_232)
    booster266 = load_booster(MODEL_266)

    # ---------- replay acceptance on frozen test-segment keys ----------
    sample_frame = build_candidate_frame(sample_keys, days, codes, members, native, mapping, diff)
    sample232 = resolve_logical(sample_frame, mapping, fields232)
    x232 = native_missing_view(sample232, fields232)
    got232 = booster232.predict(x232)
    frozen232 = sample.lightgbm_raw.to_numpy(dtype=float)
    delta232 = np.abs(got232 - frozen232)
    tol232 = np.maximum(TOLERANCE_ABS, TOLERANCE_REL * np.abs(frozen232))
    log('replay_232', max_delta=float(delta232.max()), pass_=bool((delta232 <= tol232).all()))

    # 266 needs ccm on sample keys too
    sample_frame266 = build_266(sample_frame, sample_keys, days, codes, members, native, extra266)
    sample266 = booster266.predict(native_missing_view(sample_frame266, fields266))
    pred266 = pd.read_parquet(MODEL_266 / 'predictions.parquet')
    frozen266_map = {(str(c).zfill(6), int(t)): float(v) for c, t, v in
                     zip(pred266.code, pred266[TIME], pred266.lightgbm_raw)}
    frozen266 = np.array([frozen266_map[(c, int(t))] for c, t in zip(sample_keys.code, sample_keys[TIME])])
    delta266 = np.abs(sample266 - frozen266)
    tol266 = np.maximum(TOLERANCE_ABS, TOLERANCE_REL * np.abs(frozen266))
    log('replay_266', max_delta=float(delta266.max()), pass_=bool((delta266 <= tol266).all()))

    # feature identity vs frozen arm reader
    from prepare_direct_warning_comparison_v1 import load_arm
    arm = load_arm('test', 'C1')
    arm_map = {(str(c).zfill(6), int(t)): r for c, t, r in
               zip(arm.code, arm[TIME], arm[fields232].itertuples(index=False, name=None))}
    arm_rows = np.array([arm_map[(c, int(t))] for c, t in zip(sample_keys.code, sample_keys[TIME])], dtype=float)
    s232 = sample232[fields232].to_numpy(dtype=float)
    both_nan = np.isnan(s232) & np.isnan(arm_rows)
    feat_delta = np.abs(s232 - arm_rows)
    feat_delta[both_nan] = 0.0
    feat_max = float(np.nanmax(feat_delta))
    log('feature_identity_vs_frozen_arm', max_delta=feat_max, pass_=bool(feat_max <= 1e-4))

    acceptance = dict(
        rule='pass iff all three: |delta|<=max(1e-4,1e-3*|frozen|) for both models, and feature max delta<=1e-4',
        samples=N_REPLAY_SAMPLES,
        model232=dict(max_abs_delta=float(delta232.max()),
                      all_pass=bool((delta232 <= tol232).all()),
                      per_sample=[dict(code=str(c), ts=int(t), rebuilt=float(a), frozen=float(f), delta=float(d))
                                  for c, t, a, f, d in zip(sample_keys.code, sample_keys[TIME], got232, frozen232, delta232)]),
        model266=dict(max_abs_delta=float(delta266.max()), all_pass=bool((delta266 <= tol266).all())),
        feature_identity=dict(max_abs_delta=feat_max, pass_=bool(feat_max <= 1e-4)),
        passed=bool((delta232 <= tol232).all() and (delta266 <= tol266).all() and feat_max <= 1e-4))
    put('replay_acceptance.json', json.dumps(acceptance, indent=1).encode())
    if not acceptance['passed']:
        put('stop_record.json', json.dumps(dict(status='FAILED_REPLAY_ACCEPTANCE')).encode(), terminal=True)
        raise ValueError('REPLAY_ACCEPTANCE_FAILED')

    # ---------- candidate inference ----------
    cand_frame = build_candidate_frame(ok[['code', TIME]], days, codes, members, native, mapping, diff)
    cand232 = resolve_logical(cand_frame, mapping, fields232)
    frame266 = build_266(cand_frame, ok[['code', TIME]], days, codes, members, native, extra266)
    raw232 = cand232[fields232].to_numpy(dtype=float)
    raw266 = frame266[fields266].to_numpy(dtype=float)
    inf_rows = sorted(set(np.flatnonzero(np.isinf(raw232).any(axis=1)).tolist())
                      | set(np.flatnonzero(np.isinf(raw266).any(axis=1)).tolist()))
    for frame, fields in ((cand232, fields232), (frame266, fields266)):
        values = frame[fields].to_numpy(dtype=float).copy()
        values[np.isinf(values)] = np.nan
        frame[fields] = values
    x_c232 = native_missing_view(cand232, fields232)
    amp232 = booster232.predict(x_c232)
    x_c266 = native_missing_view(frame266, fields266)
    amp266 = booster266.predict(x_c266)
    log('candidate_inference_done', rows=len(ok), inf_rows=len(inf_rows))

    write_outputs(ok, cand_frame, cand232, amp232, amp266, frame266, fields232, fields266, inf_rows, writer, put)
    put('stop_record.json', json.dumps(dict(status='COMPLETE_NOT_FIT', http_requests=0,
        manifest={k: v['sha256'] for k, v in writer.items()}), indent=1).encode(), terminal=True)
    log('done')


def build_266(frame, keys, days, codes, members, native, extra266):
    """Attach the 34 ccm_v1 sidecar fields; returns merged source-level frame."""
    from build_common_candidate_sidecar_v1 import month_features
    from build_multiboard_full_features_v1 import market_arrays
    import common_candidate_aggregation_v1 as aggregation
    population = keys.copy().reset_index(drop=True)
    population['segment'] = SEG_EXTENSION
    result = population.copy()
    di = {d: i for i, d in enumerate(days)}
    market = pd.read_parquet(MARKET_FEATURES)
    for frequency in (5, 15):
        width = 240 // frequency
        for month in sorted({d // 100 for d in days}):
            md = [d for d in days if d // 100 == month]
            sub = population[population[TIME].floordiv(100000000).eq(month)]
            if sub.empty:
                continue
            clock, markets = market_arrays(market, md, frequency)
            values, audit, _ = month_features(
                sub, codes, members, native[frequency][:, di[md[0]] * width:(di[md[-1]] + 1) * width],
                markets, clock, frequency)
            for column in values.columns.difference(aggregation.KEYS):
                result.loc[sub.index, column] = values[column].to_numpy()
    missing = [f for f in extra266 if f not in result]
    if missing:
        raise ValueError('CCM_FIELD_MISSING:' + json.dumps(missing))
    return pd.concat([frame.reset_index(drop=True),
                      result[extra266].reset_index(drop=True)], axis=1)


def write_outputs(ok, cand_frame, cand232, amp232, amp266, frame266, fields232, fields266, inf_rows, writer, put):
    ok = ok.reset_index(drop=True)
    iso = []
    v232 = cand232[fields232].to_numpy(dtype=float)
    v266 = frame266[fields266].to_numpy(dtype=float)
    inf_set = set(inf_rows)
    for i in range(len(ok)):
        reasons = []
        if i in inf_set:
            reasons.append('infinite_feature_value')
        if np.isnan(v232[i]).all():
            reasons.append('all_232_missing')
        missing_sources = []
        for col, label in [('stock_day_return_raw', 'joint_parts'), ('context_sector_mean_return_loo_raw', 'context_loo')]:
            if col in cand_frame and pd.isna(cand_frame.loc[i, col]):
                missing_sources.append(label)
        if missing_sources:
            reasons.append('source_join_missing:' + ','.join(missing_sources))
        iso.append(dict(code=ok.loc[i, 'code'], ts=int(ok.loc[i, TIME]), isolated=bool(reasons),
                        reasons=reasons, n_missing_232=int(np.isnan(v232[i]).sum()),
                        n_missing_266=int(np.isnan(v266[i]).sum())))
    iso_frame = pd.DataFrame(iso)
    n_iso = int(iso_frame.isolated.sum())
    log('isolation', isolated=n_iso, of=len(ok))

    scores = ok[['code', 'E', 'status', TIME]].copy()
    scores['code'] = scores.code.str.zfill(6)
    scores['amp_232_raw'] = amp232
    scores['amp_266_raw'] = amp266
    scores['amp_232'] = np.clip(amp232, 0, 1)
    scores['amp_266'] = np.clip(amp266, 0, 1)
    scores['n_missing_232'] = iso_frame.n_missing_232
    scores['n_missing_266'] = iso_frame.n_missing_266
    scores['isolated'] = iso_frame.isolated
    scores['isolation_reason'] = iso_frame.reasons.map(lambda r: ';'.join(r) if r else '')
    scores['scope'] = SEG_EXTENSION
    scores['data_cutoff'] = 20260908

    replay = pd.read_parquet(CANDIDATE_REPLAY)
    replay['code'] = replay.code.astype(str).str.zfill(6)
    sel = pd.read_parquet(MASTER_TABLE)
    sel['code'] = sel.code.astype(str).str.zfill(6)
    scores = scores.merge(replay[['code', 'P', 'E', 'trigger_t', 'trigger_close', 'reason']],
                          on=['code', 'E'], how='left', validate='many_to_one')
    scores = scores.merge(sel[['code', 'P', 'setup', 'pool', 'F618', 'atr_ref', 'plan_dist_atr',
                               'golden_dist_618_atr']], on=['code', 'P'], how='left', validate='many_to_one')
    scores['selection_score_pB2'] = np.nan  # not present in any frozen artifact; see meta note
    scores = scores.rename(columns={'plan_dist_atr': 'plan_dist'})

    evidence = dict(model232_manifest_sha256=sha(MODEL_232 / 'manifest.json'),
                    model266_manifest_sha256=sha(MODEL_266 / 'manifest.json'),
                    coverage_assess_sha256=sha(HERE / 'coverage_assess_v1.json'))
    p_snapshot = scores.drop(columns=['status', 'trigger_t', 'trigger_close', 'reason',
                                      'amp_232_raw', 'amp_266_raw', 'amp_232', 'amp_266'])
    p_snapshot['asof'] = 'P_close'
    e_update = scores.copy()
    e_update['asof'] = 'E_1500'
    p_snapshot['trigger_status'] = np.nan
    p_snapshot['trigger_basis'] = np.nan
    e_update['trigger_status'] = e_update.status
    e_update['trigger_basis'] = e_update.apply(
        lambda r: 'trigger_t=%s trigger_close=%s reason=%s' % (r.trigger_t, r.trigger_close, r.reason), axis=1)

    sink = io.BytesIO(); p_snapshot.to_parquet(sink, index=False, compression='zstd'); put('candidate_scores_p_snapshot.parquet', sink.getvalue())
    sink = io.BytesIO(); e_update.to_parquet(sink, index=False, compression='zstd'); put('candidate_scores_e_update.parquet', sink.getvalue())
    feats = pd.concat([scores[['code', TIME, 'E', 'status']].reset_index(drop=True),
                       cand232.reset_index(drop=True),
                       frame266[fields266[232:]].reset_index(drop=True)], axis=1)
    for column in list(fields232) + list(fields266[232:]):
        feats[column] = feats[column].astype('float32')
    sink = io.BytesIO(); feats.to_parquet(sink, index=False, compression='zstd'); put('candidate_features.parquet', sink.getvalue())
    put('candidate_isolation.json', json.dumps(iso, indent=1).encode())
    meta = dict(scope=SEG_EXTENSION, asof_semantics=dict(
                    p_snapshot='columns knowable at P close; E-day risk amplitudes and trigger columns removed/NaN to avoid look-ahead',
                    e_update='E-day 15:00 assumed-end cutoff; risk amplitudes valid as of 15:00'),
                decision_ts_semantics='all E-day triggers/decisions mapped to the final completed 15m cutoff of E (15:00), identical to phase-1 coverage口径',
                selection_score_pB2='column absent: no frozen artifact in round_a_v0_1_impl_fix carries selection_score_pB2; left NaN pending user-provided source',
                plan_dist='from full_master_table_v2.parquet plan_dist_atr (frozen star-side selection artifact)',
                F618='from full_master_table_v2.parquet (frozen star-side selection artifact)',
                models=dict(model232='auxiliary_120m_fixed_fit_20261018_v1 lightgbm (regression amplitude, clip [0,1] applied to amp_* columns)',
                            model266='common_candidate_fixed_fit_20261019_v1 lightgbm'),
                evidence=evidence, data_cutoff='frozen minute ledger last timestamp 20260908 (444/543 files); per-record file spans see coverage_assess_v1.json',
                isolation_policy='records with infinite values / all-missing 232 / missing source joins are isolated (isolated=True) and must not be used for capability claims',
                replay_acceptance='see replay_acceptance.json',
                bytes_total=sum(v['bytes'] for v in writer.values()))
    put('candidate_scores_meta.json', json.dumps(meta, ensure_ascii=False, indent=1).encode())
    log('outputs_written', bytes=meta['bytes_total'])


if __name__ == '__main__':
    main()
