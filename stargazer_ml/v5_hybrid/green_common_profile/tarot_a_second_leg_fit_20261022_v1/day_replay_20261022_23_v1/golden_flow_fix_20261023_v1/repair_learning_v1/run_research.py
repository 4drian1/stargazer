"""One fixed 2x2 market-repair/MACD diagnostic, strictly separate from production."""
import gzip
import hashlib
import io
import json
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from repair_features import market_panel, GCP, FEATURES

HERE = Path(__file__).resolve().parent
PARENT = HERE.parents[1]
RISK = GCP.parent / 'risk_r0_v1'
ROOT = GCP.parents[2]
sys.path[:0] = [str(RISK), str(PARENT)]
import prepare_auxiliary_fixed_fit_v1 as prep
from auxiliary_regression_eval_v1 import errors, paired_errors
from rebuild_features_v1 import load_booster, native_missing_view, MODEL_232
from risk_cached import peak

TIME = 'decision_ts_assumed_end'
TARGET = 'max_adverse_drop_120m'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def put(name, value):
    body = value if isinstance(value, bytes) else json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2).encode('utf-8')
    if sum(p.stat().st_size for p in HERE.rglob('*') if p.is_file()) + len(body) > 32 * 1024 ** 2:
        raise ValueError('NEW_RESEARCH_32MIB_LIMIT')
    if sum(p.stat().st_size for p in PARENT.rglob('*') if p.is_file()) + len(body) > 128 * 1024 ** 2:
        raise ValueError('PARENT_RESEARCH_128MIB_LIMIT')
    with (HERE / name).open('xb') as stream:
        stream.write(body)


def parquet(frame):
    buffer = io.BytesIO()
    frame.to_parquet(buffer, index=False, compression='zstd')
    return buffer.getvalue()


def log(stage, **kwargs):
    print(json.dumps(dict(stage=stage, **kwargs)), flush=True)


def main():
    config = json.loads((prep.OUT / 'config.json').read_bytes())
    fields = config['fields']
    remove = [n for n in fields if 'macd' in n and ('_5m_' in n or '_15m_' in n)]
    assert len(remove) == 12
    protected = [MODEL_232 / 'lightgbm_model.txt.gz', MODEL_232 / 'lightgbm_preprocessing.json',
                 GCP / 'tarot_live_manifest_v4_20261023_flow.json',
                 RISK / 'live_monitor_fingerprints_20261023_v4_flow.json',
                 ROOT / 'stargazer_desktop/dist/Stargazer.exe',
                 ROOT / 'stargazer_desktop/dist/data/pentacles_demo/assistant_plan.json']
    protected = [p for p in protected if p.exists()]
    bound = [prep.OUT / 'config.json', prep.auxiliary.OUT / 'labels.parquet',
             MODEL_232 / 'predictions.parquet', Path(__file__), HERE / 'repair_features.py',
             HERE / 'test_repair_features.py']
    for n in ['sse', 'cyb']:
        bound += [RISK / f'index_minutes_v1/{n}_5min.parquet', RISK / f'index_minutes_v1/{n}_5min.meta.json']
    inputs = {str(p): sha(p) for p in bound + protected}
    put('plan.json', dict(status='CHAT_AUTHORIZED_FIXED_OFFLINE_DIAGNOSTIC',
        user_quotes=['把前面我加的几个用来看修复的指标加进去让模型学会缓慢修复，然后降低短线macd的权重',
                     '记得我们讨论的组合指标和分割线画法，不要跑偏了，让模型学到缓慢修复是买点，拉升之后反而是追涨了'],
        scope='Market repair first; stock-specific new combination not added to regression in this round',
        target=TARGET, target_changed=False, original_config_sha256=sha(prep.OUT / 'config.json'),
        segments=config['segments'], excluded_short_macd_fields=remove,
        arms=dict(A='frozen 232 replay', B='232 plus 44 market repair fields and two joint market states',
                  C='232 minus 12 short MACD', D='B minus same 12 short MACD'),
        fits=3, parameters=config['models']['lightgbm'], early_stop=config['selection'],
        no_parameter_search=True, labels_not_buy_sell=True, test_reused=True,
        example_20261023='known failure case excluded from all fitting and model selection',
        market_time_basis='TDX_5MIN_BAR_END_ASSUMED_NOT_CERTIFIED',
        volume_units_verified=False, money_flow_not_inferred_from_volume=True,
        feature_definition='repair_features.py; P confirmed maps, current completed 5m prefixes, Wilder RSI6/15',
        new_artifact_cap=32 * 1024 ** 2, parent_cap=128 * 1024 ** 2, ram_cap=3 * 1024 ** 3,
        inputs=inputs, http_requests=0, paid_calls=0, deployment=False))
    try:
        # This invokes the original full input/hash checks. Never bypass a failure.
        prep.check()
        parts = prep.load_parts(config)
        base = pd.concat([f.assign(segment=s) for s, f in parts.items()], ignore_index=True)
        assert len(base) == 86319 and not base.duplicated(['code', TIME]).any()
        requested = sorted(set((base[TIME] // 1000000).astype(int)) | {20261023})
        panels = []
        for name in ['sse', 'cyb']:
            log('market_repair_features', index=name, days=len(requested))
            source = RISK / f'index_minutes_v1/{name}_5min.parquet'
            meta = json.loads((RISK / f'index_minutes_v1/{name}_5min.meta.json').read_bytes())
            assert sha(source) == meta['sha256']
            native = pd.read_parquet(source)
            panel, receipts = market_panel(native, name, requested)
            panels.append(panel)
            put(name + '_map_receipts.json', receipts)
            # Full history seed is retained; only the future is truncated in the comparison.
            cutoff = 20261023113000
            prefix, _ = market_panel(native[native.ts <= cutoff], name, [20261023])
            expected = panel[(panel[TIME] // 1000000 == 20261023) & (panel[TIME] <= cutoff)].reset_index(drop=True)
            pd.testing.assert_frame_equal(expected, prefix[prefix[TIME] <= cutoff].reset_index(drop=True))
            peak()
        panel = panels[0].merge(panels[1], on=TIME, validate='one_to_one')
        joint = ['sse_repair_repair_joint', 'cyb_repair_repair_joint']
        known = panel[joint].notna().all(axis=1)
        panel['market_repair_both_joint'] = np.where(known, panel[joint].eq(1).all(axis=1).astype(float), np.nan)
        panel['market_repair_any_joint'] = np.where(known, panel[joint].eq(1).any(axis=1).astype(float), np.nan)
        added = [n for n in panel if n != TIME]
        assert len(added) == len(FEATURES) * 2 + 2 == 46
        put('market_features.parquet', parquet(panel))
        frame = base.merge(panel, on=TIME, how='left', validate='many_to_one', sort=False)
        assert len(frame) == len(base)
        pd.testing.assert_frame_equal(frame[['code', TIME]], base[['code', TIME]])
        assert not frame[added].isna().all().any(), 'ALL_NULL_ADDED_FIELD'
        old = pd.read_parquet(MODEL_232 / 'predictions.parquet')
        old = frame[['code', TIME]].merge(old, on=['code', TIME], validate='one_to_one', sort=False)
        booster = load_booster(MODEL_232)
        original = booster.predict(native_missing_view(frame, fields), num_threads=2)
        np.testing.assert_allclose(original, old.lightgbm_raw.to_numpy(), atol=1e-11, rtol=0)
        arms = dict(A=fields, B=fields + added,
                    C=[n for n in fields if n not in remove],
                    D=[n for n in fields if n not in remove] + added)
        coverage = {}
        for segment, group in frame.groupby('segment', sort=False):
            events = group[group.market_repair_any_joint.eq(1)][TIME].drop_duplicates()
            coverage[segment] = dict(rows=len(group), stocks=group.code.nunique(),
                dates=(group[TIME] // 1000000).nunique(), repair_cutoffs=len(events),
                repair_dates=(events // 1000000).nunique(),
                field_known={n: int(group[n].notna().sum()) for n in added})
        put('preflight.json', dict(status='PASS_FIXED_INPUTS_AND_PREFIX_CHECKS', coverage=coverage,
            original_predictions_reproduced=True, real_prefix_comparisons=2,
            rows=len(frame), columns={k: len(v) for k, v in arms.items()},
            stock_repair_training_added=False, peak_ram=peak()))
        predictions = {'A': original}
        train = frame.segment.eq('train').to_numpy()
        select = frame.segment.eq('select').to_numpy()
        fitted = {}
        for arm in ['B', 'C', 'D']:
            log('fixed_fit', arm=arm, fields=len(arms[arm]))
            names = arms[arm]
            values = native_missing_view(frame, names)
            model = lgb.LGBMRegressor(**config['models']['lightgbm'])
            model.fit(values[train], frame.loc[train, TARGET].to_numpy(),
                      eval_set=[(values[select], frame.loc[select, TARGET].to_numpy())],
                      eval_metric='l2', callbacks=[lgb.early_stopping(50, verbose=False)])
            text = model.booster_.model_to_string(num_iteration=model.best_iteration_)
            saved = lgb.Booster(model_str=text)
            pred = saved.predict(values, num_threads=2)
            np.testing.assert_allclose(pred, model.predict(values), atol=1e-12, rtol=0)
            predictions[arm] = pred
            put(arm + '_model.txt.gz', gzip.compress(text.encode('utf-8'), mtime=0))
            put(arm + '_features.json', dict(columns=names, preprocessing='native NaN plus per-field missing indicators',
                                           best_iteration=int(model.best_iteration_)))
            fitted[arm] = dict(fields=len(names), best_iteration=int(model.best_iteration_), saved_replay=True)
            peak()
        table = frame[['code', TIME, 'segment', TARGET]].copy()
        for arm, pred in predictions.items():
            table[arm + '_raw'] = pred
        put('predictions.parquet', parquet(table))
        threshold = float(frame.loc[train, TARGET].quantile(.9))
        metrics = {}
        for segment, group in frame.groupby('segment', sort=False):
            pos = group.index.to_numpy()
            y = group[TARGET].to_numpy()
            scores = {arm: np.clip(p[pos], 0, 1) for arm, p in predictions.items()}
            item = dict(all={arm: errors(y, p) for arm, p in scores.items()}, subgroups={})
            for key, mask in dict(repair=group.market_repair_any_joint.eq(1).to_numpy(),
                                  nonrepair=group.market_repair_any_joint.eq(0).to_numpy(),
                                  actual_top_decile=y >= threshold).items():
                item['subgroups'][key] = dict(rows=int(mask.sum()),
                    dates=int((group.loc[mask, TIME] // 1000000).nunique()),
                    metrics={arm: errors(y[mask], p[mask]) for arm, p in scores.items()} if mask.any() else {})
            item['paired'] = {left + '_minus_' + right: paired_errors(y, scores[left], scores[right],
                group[TIME].to_numpy() // 1000000, config['evaluation']['uncertainty'])
                for left, right in [('B', 'A'), ('C', 'A'), ('D', 'B'), ('D', 'C')]}
            metrics[segment] = item
        for path, expected in inputs.items():
            assert sha(path) == expected, 'INPUT_CHANGED:' + path
        report = dict(status='COMPLETE_FIXED_RESEARCH_NOT_DEPLOYED', fits=fitted, metrics=metrics,
            coverage=coverage, training_target=TARGET, target_changed=False,
            top_decile_actual_threshold_from_train=threshold,
            known_case_20261023=panel[panel[TIME].eq(20261023113000)].replace({np.nan: None}).to_dict('records'),
            independent_blind_test=False, reused_test_never_selects_model=True,
            stock_repair_training_added=False, pnl_evaluated=False, promotion_passed=False,
            production_protected=True, paid_calls=0, http_requests=0, peak_ram=peak())
        put('report.json', report)
        put('stop_record.json', dict(status=report['status'], report_sha256=sha(HERE / 'report.json')))
        log('complete', status=report['status'], fitted=3, rows=len(frame))
    except Exception as exc:
        put('failed.json', dict(status='FAILED_NO_AUTOMATIC_RESTART', error=f'{type(exc).__name__}: {exc}'))
        raise


if __name__ == '__main__':
    main()
