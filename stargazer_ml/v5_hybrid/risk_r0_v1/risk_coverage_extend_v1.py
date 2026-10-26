"""risk_coverage_extend_v1.py — 冻结风险模型覆盖扩展（用户 2026-10-19 授权方向）。

为星币侍从 A1 评分候选（2025/2026）构建**同定义、同尺度、P收盘可用**的冻结
16 字段风险特征，并用冻结模型（models.joblib，不重训）批量推理。
特征函数/因子构造/市场源全部复用风险线冻结代码（historical_risk_core_v1 /
stockdb_source_v1 / stockdb_adjustment_v2 / market_index_source_v1）。
尺度证据：多数扩展股票仅有 stockdb 原生复权（单源）——scale_evidence 如实标注
STOCKDB_NATIVE_ONLY，coverage_verified=False；不把缺失当低风险。
特征不足记录单列（不补零）。与既有41股预测交集用于接线验收（概率重放比对）。
输出统一结构（用户规范）：候选代码/决策时间/模型版本/选股评分/黄金位置/
触发状态及依据/风险概率/预测期限/特征缺失/适用范围/数据截止/证据引用。
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / 'green_common_profile/round_a_v0_1_impl_fix'))
sys.path.insert(0, str(HERE.parent.parent / 'step9_preflight'))
from historical_risk_core_v1 import stock_features, market_features  # noqa: E402
from stockdb_source_v1 import read_daily_raw, read_adjustment, build_factor_series  # noqa: E402
from market_index_source_v1 import bind_from_experiment_config  # noqa: E402

FIELDS = ['ret_1d', 'ret_5d', 'ret_20d', 'vol_5d', 'vol_20d', 'downside_vol_20d',
          'atr14_pct', 'range_pct_5d', 'drawdown_20d', 'drawdown_60d', 'gap_last',
          'market_ret_5d', 'market_ret_20d', 'market_vol_20d',
          'market_downside_vol_20d', 'market_drawdown_20d']
MODEL_SHA = '547fae9090a40f133cf4b44677ce506af238cb2e301fa2a869032f88c0a0d90d'
MODEL_PATH = HERE / 'historical_risk_fit_v1_run/models.joblib'
CAL = json.loads((HERE / 'historical_stock_day_preparation_v1/manifest.json'
                  ).read_bytes())['calendar_dates']
OUT_DIR = HERE / 'risk_coverage_extend_v1'
CACHE = OUT_DIR / 'code_cache.pkl'
ROUND = HERE.parent / 'green_common_profile/round_a_v0_1_impl_fix'


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    assert sha(MODEL_PATH) == MODEL_SHA, 'models.joblib 哈希漂移，拒绝'
    v4 = json.loads((ROUND / 'joint_two_stage_v4.json').read_text(encoding='utf-8'))
    thresholds = {int(k): v for k, v in v4['thresholds'].items()}

    rep = pd.read_parquet(ROUND / 'full_intraday_replay_v2.parquet')
    master = pd.read_parquet(ROUND / 'full_master_table_v2.parquet')
    pred = pd.read_parquet(ROUND / 'rolling_lr_v2_predictions.parquet')
    pop = rep[rep['status'].isin(
        ['TRIGGERED', 'NOT_TRIGGERED', 'SKIPPED_NON_DOWNSIDE_PATH'])]
    pop = pop[pop['master_admission'] == 'OK']
    pr = pred.set_index(['code', 'P'])
    pop = pop.copy()
    pop['pB2'] = [pr['pB2'].get((c, int(p)), np.nan)
                  for c, p in zip(pop['code'], pop['P'])]
    pop = pop.dropna(subset=['pB2'])
    pop['pyear'] = pop['P'] // 10000
    pop['thr'] = [thresholds[int(y)] for y in pop['pyear']]
    a1 = pop[pop['pB2'] >= pop['thr']].copy()
    a1 = a1[a1['pyear'] >= 2025]          # 主覆盖 2025/2026
    print('A1候选:', len(a1), 'codes:', a1['code'].nunique(), flush=True)

    m = master.set_index(['code', 'P'])
    rep_idx = {(r.code, int(r.P)): r for r in rep.itertuples(index=False)}
    fit_pred = pd.read_parquet(HERE / 'historical_risk_fit_v1_run/predictions.parquet')
    fit_idx = fit_pred.set_index(['code', 'P'])

    models = joblib.load(MODEL_PATH)
    closes = bind_from_experiment_config()

    adj_cache = json.loads((HERE.parent / 'opportunity_v2/gate1c'
                            '/feb2026_adjustment_cache_v1/adj_cache.json').read_bytes())

    OUT_DIR.mkdir(exist_ok=True)
    if CACHE.exists():
        code_cache = joblib.load(CACHE)
        print('code cache reused:', len(code_cache), flush=True)
    else:
        code_cache = {}
    rows, wiring = [], []
    codes = sorted(a1['code'].unique())
    for i, code in enumerate(codes):
        try:
            raw, _d = read_daily_raw(code)
            adj = read_adjustment(code)
        except Exception as exc:
            code_cache[code] = dict(error=str(exc)[:60])
            continue
        bars = {r['date']: r for r in raw}
        factors = build_factor_series(adj, CAL)
        code_cache[code] = dict(bars=bars, factors=factors, n_adj=len(adj))
        time.sleep(0.06)
    joblib.dump(code_cache, CACHE)
    print('codes fetched:', len(code_cache), flush=True)

    for r in a1.itertuples(index=False):
        code, p = r.code, int(r.P)
        cc = code_cache.get(code) or {}
        if 'error' in cc or not cc:
            rows.append(dict(code=code, P=p, pyear=p // 10000,
                             feature_status='FETCH_FAIL',
                             reason=cc.get('error', 'NO_DATA')))
            continue
        feat, fstat = stock_features(cc['bars'], CAL, p, cc['factors'])
        mkt, mstat = market_features(closes, CAL, p)
        def _v(k):
            return feat.get(k) if k in feat else mkt.get(k)
        miss = [k for k in FIELDS
                if _v(k) is None or not np.isfinite(_v(k))]
        if fstat != 'CALCULATED_EVIDENCE_ADMISSION_SEPARATE' or \
           mstat != 'CALCULATED_INDEX_SOURCE_ADMISSION_SEPARATE' or miss:
            rows.append(dict(code=code, P=p, pyear=p // 10000,
                             feature_status='INSUFFICIENT',
                             reason=f'{fstat}|{mstat}|missing={len(miss)}'))
            continue
        x = np.array([[feat[k] if k in feat else mkt[k] for k in FIELDS]])
        lr_raw = float(models['lr'].predict_proba(x)[0, 1])
        gb_raw = float(models['lightgbm'].predict_proba(x)[0, 1])
        lr_cal = float(models['calibrators']['lr'].predict_proba([[lr_raw]])[0, 1])
        gb_cal = float(models['calibrators']['lightgbm'].predict_proba([[gb_raw]])[0, 1])
        mm = m.loc[(code, p)]
        trig = rep_idx.get((code, p))
        rec = dict(
            code=code, P=p, decision_ts=f'{p}150000', pyear=p // 10000,
            selection_score_pB2=round(float(r.pB2), 6),
            threshold_used=round(float(r.thr), 4),
            F618=(None if np.isnan(mm['F618']) else round(float(mm['F618']), 4)),
            plan_dist_atr=(None if np.isnan(mm['plan_dist_atr'])
                           else round(float(mm['plan_dist_atr']), 4)),
            arrival_class=getattr(trig, 'arrival_class', None) if trig is not None else None,
            trigger_status=(getattr(trig, 'status', None) if trig is not None else None),
            trigger_t=(getattr(trig, 'trigger_t', None) if trig is not None else None),
            trigger_basis=(getattr(trig, 'replay_reason', None) if trig is not None else None),
            risk_lr_raw=round(lr_raw, 6), risk_lr_cal=round(lr_cal, 6),
            risk_lgbm_raw=round(gb_raw, 6), risk_lgbm_cal=round(gb_cal, 6),
            risk_horizon='5d_tail', risk_model_sha256=MODEL_SHA[:16],
            feature_status='OK', missing_fields=None,
            scale_evidence='STOCKDB_NATIVE_ONLY', coverage_verified=False,
            adj_event_count=cc['n_adj'],
            adj_cache_event_count=len((adj_cache.get(code) or {}).get('events') or []),
            data_cutoff='20261013',
            evidence_refs=json.dumps(dict(
                models=MODEL_SHA[:16], features='historical_risk_core_v1',
                source='stockdb_source_v1', factors='stockdb_adjustment_v2',
                market='market_index_source_v1', selection='pentacles_v4'),
                ensure_ascii=False))
        rows.append(rec)
        if (code, p) in fit_idx.index:
            fp = fit_idx.loc[(code, p)]
            wiring.append(dict(code=code, P=p,
                               lr_raw_delta=round(lr_raw - float(fp['lr_raw']), 6),
                               gb_raw_delta=round(gb_raw - float(fp['lightgbm_raw']), 6)))

    df = pd.DataFrame(rows)
    out = OUT_DIR / 'risk_scores_a1_v1.parquet'
    df.to_parquet(out, index=False)
    ok = df[df['feature_status'] == 'OK']
    meta = dict(schema='PENTACLES_RISK_COVERAGE_V1',
                model_sha256=MODEL_SHA,
                n_candidates=int(len(a1)),
                n_scored=int(len(ok)),
                n_insufficient=int((df['feature_status'] == 'INSUFFICIENT').sum()),
                n_fetch_fail=int((df['feature_status'] == 'FETCH_FAIL').sum()),
                by_year={str(k): int(v) for k, v in
                         ok.groupby('pyear').size().items()},
                by_year_insufficient={str(k): int(v) for k, v in
                                      df[df['feature_status'] != 'OK'].groupby('pyear').size().items()},
                n_unique_codes=int(a1['code'].nunique()),
                scope_disclosure='训练41股之外的股票为单源复权扩展应用，'
                                 'coverage_verified=False',
                wiring_checks=wiring,
                wiring_max_abs_delta=(max(max(abs(w['lr_raw_delta']),
                                              abs(w['gb_raw_delta']))
                                          for w in wiring) if wiring else None))
    (OUT_DIR / 'risk_scores_a1_v1.meta.json').write_text(
        json.dumps(meta, ensure_ascii=False, indent=1), encoding='utf-8')
    print(json.dumps({k: v for k, v in meta.items() if k != 'wiring_checks'},
                     ensure_ascii=False)[:800])
    print(f'bytes={out.stat().st_size}')


if __name__ == '__main__':
    main()
