"""Exploratory read-only diagnostics and fold inventory. No model loading, fitting or network."""
from collections import Counter
import json
import math
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from sklearn.metrics import roc_auc_score

from historical_risk_core_v1 import STOCK_FIELDS, MARKET_FIELDS, segment
from price_verification_guarded_run import encoded, sha
from research_budget_v2 import bytes_used, TOTAL_CAP
from risk_fit_eval_v1 import board

HERE = Path(__file__).resolve().parent
RUN = HERE / 'historical_risk_fit_v1_run'
OUT = HERE / 'risk_failure_diagnostics_v1_20261015'
CAP = 2 * 1024**2
SCOPE = 'CONDITIONAL_RESEARCH_DOUBLE_SOURCE_ORDINARY_ACTIONS'
FIELDS = STOCK_FIELDS + MARKET_FIELDS


def require(value, message):
    if not value:
        raise ValueError(message)


def base_valid(row):
    return (row.get('admission_scope') == SCOPE and row.get('coverage_verified') is False
            and row.get('label_status') == 'LABEL_COMPLETE' and type(row.get('y_tail_5d')) is int
            and row['y_tail_5d'] in (0, 1) and type(row.get('mae_5d')) in (int, float)
            and math.isfinite(row['mae_5d']) and all(type(row['features'].get(f)) in (int, float)
            and math.isfinite(row['features'][f]) for f in FIELDS))


def fold_inventory(rows, fold):
    periods = ('train', 'select', 'calibration', 'test')
    out = {}
    for j, name in enumerate(periods):
        start, end = fold[name]
        boundary = fold[periods[j+1]][0] if j < 3 else (end//10000+1)*10000+101
        population = [r for r in rows if start <= r['P'] <= end]
        source_ok = [r for r in population if base_valid(r)]
        selected = [r for r in source_ok if r['P'] < r['label_end_date'] < boundary]
        recovered = [r for r in selected if r['training_eligible'] is not True]
        require(all(not segment(r['P'], r['label_end_date'])[1] for r in recovered),
                'NON_TEMPORAL_QUARANTINE_REVIVED')
        n, positive = len(selected), sum(r['y_tail_5d'] for r in selected)
        dates = len({r['P'] for r in selected})
        minimum = (2000, 200, 250) if name == 'train' else (500, 50, 60)
        out[name] = dict(population_rows=len(population), base_valid_rows=len(source_ok),
            boundary_purged=len(source_ok)-n, n=n, positive=positive, nonpositive=n-positive,
            dates=dates, unique_codes=len({r['code'] for r in selected}),
            old_admitted_false_but_this_fold_eligible=len(recovered),
            pass_gate=n >= minimum[0] and min(positive, n-positive) >= minimum[1] and dates >= minimum[2])
    return dict(id=fold['id'], mode=fold['mode'], segments=out,
                sample_gate_passed=all(s['pass_gate'] for s in out.values()), fit_authorized=False)


def market_thresholds(rows):
    dates = {}
    for r in rows:
        if r['segment'] != 'train':
            continue
        value = r['features']['market_vol_20d']
        require(r['P'] not in dates or dates[r['P']] == value, 'MARKET_CONTEXT_CONFLICT')
        dates[r['P']] = value
    require(len(dates) >= 3, 'TRAIN_DATES_REQUIRED')
    return [float(x) for x in np.quantile(list(dates.values()), [1/3, 2/3], method='linear')]


def regime(value, thresholds):
    return 'LOW' if value <= thresholds[0] else 'MID' if value <= thresholds[1] else 'HIGH'


def auc(y, score):
    return float(roc_auc_score(y, score)) if len(set(y)) == 2 else None


def tail_stats(rows):
    if not rows:
        return dict(n=0, tail_count=0, tail_rate=None, mean_mae=None, worst5_mae_mean=None)
    values = np.array([r['mae_5d'] for r in rows])
    q = np.quantile(values, .05)
    return dict(n=len(rows), tail_count=sum(r['y_tail_5d'] for r in rows),
                tail_rate=float(np.mean([r['y_tail_5d'] for r in rows])), mean_mae=float(values.mean()),
                worst5_mae_mean=float(values[values <= q].mean()) if len(rows) >= 20 else None)


def group_diagnostics(rows):
    days = {}
    for r in rows:
        days.setdefault(r['P'], []).append(r)
    truth = np.array([r['y_tail_5d'] for r in rows])
    probability = {}
    for model in ('lr_raw', 'lr', 'lightgbm_raw', 'lightgbm'):
        p = np.array([r[model] for r in rows])
        bins = []
        for i in range(10):
            mask = (p >= i/10) & (p < (i+1)/10 if i < 9 else p <= 1)
            bins.append(dict(lower=i/10, upper=(i+1)/10, n=int(mask.sum()),
                mean_p=float(p[mask].mean()) if mask.any() else None,
                tail_rate=float(truth[mask].mean()) if mask.any() else None))
        probability[model] = dict(mean_p=float(p.mean()), observed=float(truth.mean()),
            underprediction=float(truth.mean()-p.mean()), brier=float(np.mean((p-truth)**2)),
            pooled_auc=auc(truth.tolist(), p), reliability=bins)
    ranking = {}
    for model in ('atr_rule', 'lr', 'lightgbm'):
        key = 'atr_score' if model == 'atr_rule' else model
        weighted, pairs, valid_days = 0., 0, 0
        for day in days.values():
            y = [r['y_tail_5d'] for r in day]
            a = auc(y, [r[key] for r in day])
            if a is not None:
                weight = sum(y)*(len(y)-sum(y))
                weighted += a*weight
                pairs += weight
                valid_days += 1
        ranking[model] = dict(same_day_pair_weighted_auc=weighted/pairs if pairs else None,
                             dates_with_both_classes=valid_days, opposite_class_pairs=pairs)
    arms = {m: tail_stats([r for r in rows if r[m+'_retained_primary']])
            for m in ('atr_rule', 'lr', 'lightgbm')}
    paired = all(len({sum(r[m+'_retained_primary'] for r in day) for m in arms}) == 1
                 for day in days.values())
    swap = {}
    for name, a, b in (('gb_only', 'lightgbm', 'atr_rule'), ('atr_only', 'atr_rule', 'lightgbm')):
        selected = [r for r in rows if r[a+'_retained_primary'] and not r[b+'_retained_primary']]
        swap[name] = dict(tail_stats(selected), worst_examples=[dict(code=r['code'], P=r['P'],
            mae_5d=r['mae_5d'], y_tail_5d=r['y_tail_5d']) for r in sorted(selected,
                key=lambda r: (r['mae_5d'], r['P'], r['code']))[:8]])
    return dict(n=len(rows), dates=len(days), unique_codes=len({r['code'] for r in rows}),
        unfiltered=tail_stats(rows), probability=probability, ranking=ranking,
        retained=arms, same_day_retained_counts_paired=paired,
        note='Original full-cohort masks inherited, never reranked within subgroup; board exposure may differ.',
        gb_atr_swap_sets=swap)


def feature_drift(train, target):
    out = {}
    for field in FIELDS:
        ref = np.array([r['features'][field] for r in train])
        values = np.array([r['features'][field] for r in target])
        sd = float(ref.std(ddof=0))
        constant = np.ptp(ref) == 0
        edges = np.unique(np.quantile(ref, np.linspace(0, 1, 11)[1:-1], method='linear'))
        if constant:
            psi = None
        else:
            bins = np.r_[-np.inf, edges, np.inf]
            a = np.histogram(ref, bins=bins)[0] + .5
            b = np.histogram(values, bins=bins)[0] + .5
            a, b = a/a.sum(), b/b.sum()
            psi = float(np.sum((b-a)*np.log(b/a)))
        out[field] = dict(train_mean=float(ref.mean()), target_mean=float(values.mean()),
                         standardized_mean_shift=float((values.mean()-ref.mean())/sd) if not constant else None,
                         psi=psi)
    return dict(fields=out, method='Stock-day weighted; train quantile bins, +0.5 counts; exploratory, no automatic PSI gate.')


def run():
    require(not OUT.exists(), 'OUTPUT_EXISTS_NO_OVERWRITE')
    manifest_path = HERE/'contracts/restricted_risk_fit_manifest_v1.json'
    manifest = json.loads(manifest_path.read_bytes())
    require(all(sha(r['path']) == r['sha256'] for r in manifest['inputs']), 'FROZEN_SOURCE_DRIFT')
    stop_path = RUN/'stop_record.json'
    stop = json.loads(stop_path.read_bytes())
    require(stop['status'] == 'completed', 'FIT_NOT_COMPLETE')
    for r in stop['files']:
        require(sha(RUN/r['name']) == r['sha256'], 'FIT_OUTPUT_DRIFT')
    acceptance_path = HERE/'archived_corpus_v4_acceptance_20261015.json'
    acceptance = json.loads(acceptance_path.read_bytes())
    matrix = [r for part in acceptance['matrix_files'] for r in pq.read_table(part['path']).to_pylist()]
    source = {(r['code'], r['P']): r for r in matrix}
    require(len(source) == len(matrix), 'MATRIX_DUPLICATES')
    predictions = pq.read_table(RUN/'predictions.parquet').to_pylist()
    require(len({(r['code'], r['P']) for r in predictions}) == len(predictions), 'PREDICTION_DUPLICATES')
    require({(r['code'], r['P']) for r in predictions} == {k for k, r in source.items() if r['training_eligible']},
            'PREDICTION_POPULATION_MISMATCH')
    for r in predictions:
        original = source[r['code'], r['P']]
        require(all(r[k] == original[k] for k in ('segment', 'y_tail_5d', 'mae_5d', 'label_end_date')), 'LABEL_DRIFT')
        r['features'] = original['features']
    thresholds = market_thresholds(predictions)
    groups = {}
    for name in ('train', 'select', 'calibration', 'test', 'description'):
        selected = [r for r in predictions if r['segment'] == name]
        groups[name] = dict(overall=group_diagnostics(selected))
        if name in ('test', 'description'):
            by_month, by_board, by_regime = {}, {}, {}
            for r in selected:
                by_month.setdefault(str(r['P']//100), []).append(r)
                by_board.setdefault(board(r['code']), []).append(r)
                by_regime.setdefault(regime(r['features']['market_vol_20d'], thresholds), []).append(r)
            groups[name].update(months={k: group_diagnostics(v) for k, v in sorted(by_month.items())},
                boards={k: group_diagnostics(v) for k, v in sorted(by_board.items())},
                regimes={k: group_diagnostics(v) for k, v in sorted(by_regime.items())})
    train = [r for r in predictions if r['segment'] == 'train']
    drifts = {s: feature_drift(train, [r for r in predictions if r['segment'] == s])
              for s in ('test', 'description')}
    fold_path = HERE/'contracts/r0_crossyear_folds_draft_v1.json'
    fold_config = json.loads(fold_path.read_bytes())
    inventory = [fold_inventory(matrix, fold) for fold in fold_config['folds']]
    original_gate = acceptance['gate']['segments']
    require(all({k: inventory[2]['segments'][s][k] for k in ('n', 'positive', 'nonpositive', 'dates', 'pass_gate')}
                == original_gate[s] for s in ('train', 'select', 'calibration', 'test')), 'F3_REUSE_COUNT_MISMATCH')
    result = dict(status='READONLY_DIAGNOSTICS_COMPLETED', exploratory_after_label_inspection=True,
        fit_refit_performed=False, network_requests=0, deployment_authorized=False,
        market_vol_regime_cutoffs=thresholds, regime_cutoff_population='Unique train P dates, equally weighted',
        groups=groups, feature_drift=drifts,
        note='Probability level, pooled vs same-day ranking and deep MAE separated; no causal attribution or new Gate.')
    fold_report = dict(status='DRAFT_FOLD_INVENTORY_COMPLETED', folds=inventory,
        all_folds_sample_gate_passed=all(f['sample_gate_passed'] for f in inventory),
        fit_authorized=False, new_blind_test=False, original_time_purge_reassessed=True,
        note='F3 reuses completed fit only; F1/F2 require separate FIT authorization, not automatically executed.')
    paths = {manifest_path, stop_path, acceptance_path, RUN/'report.json', RUN/'predictions.parquet',
             Path(__file__), HERE/'test_risk_failure_diagnostics_v1.py', fold_path,
             HERE/'contracts/R0_CROSSYEAR_VALIDATION_DRAFT_V1.md'}
    paths.update(Path(r['path']) for r in manifest['inputs'])
    refs = [dict(path=str(p), sha256=sha(p)) for p in sorted(paths)]
    overview = ['# R0固定预测只读诊断与跨年盘点', '',
        '探索性诊断；未拟合/未取数/未改监控。详细分月、交易板块与市场波动分组见diagnostics.json。', '',
        '| 段 | 实际尾部率 | GB平均概率 | GB整体AUC | GB同日AUC | ATR同日AUC |',
        '|---|---:|---:|---:|---:|---:|']
    for name in ('test', 'description'):
        s = groups[name]['overall']; p = s['probability']['lightgbm']
        overview.append(f"| {name} | {p['observed']:.2%} | {p['mean_p']:.2%} | {p['pooled_auc']:.4f} | "
            f"{s['ranking']['lightgbm']['same_day_pair_weighted_auc']:.4f} | {s['ranking']['atr_rule']['same_day_pair_weighted_auc']:.4f} |")
    overview += ['', '## 跨年数据门（草案，不是新FIT授权）', '',
        '| 折 | train数量/阳性 | select数量/阳性 | calibration数量/阳性 | test数量/阳性 | 原样本门 |',
        '|---|---:|---:|---:|---:|---|']
    for f in inventory:
        parts = [f"{f['segments'][s]['n']}/{f['segments'][s]['positive']}" for s in ('train','select','calibration','test')]
        overview.append('| '+f['id']+' | '+' | '.join(parts)+' | '+('PASS' if f['sample_gate_passed'] else 'FAIL')+' |')
    overview += ['', '分组继承原整日保留标志，板块子组保留数量可不同，不能当同覆盖增量。',
        '同日AUC按相反标签股票对数量加权；无正负两类的日期不计AUC，但仍进入其他统计。',
        'PSI和分组差异只是描述，不能证明2026变化原因，也不用于选新模型/阈值。',
        '校准概率在更早train/select上的结果为回看诊断，不是当时可用的OOT。',
        '最差5%MAE在子组保留样本不足20时为NULL。读回诊断不重新生成任何标签。']
    contents = {'diagnostics.json': encoded(result), 'fold_inventory.json': encoded(fold_report),
                'report.md': ('\n'.join(overview)+'\n').encode('utf-8'),
                'input_manifest.json': encoded(dict(inputs=refs, scope='READONLY_DIAGNOSTICS_NO_FIT'))}
    terminal = encoded(dict(status='completed', fit_performed=False, network_requests=0,
        files=[dict(name=n, bytes=len(v), sha256=__import__('hashlib').sha256(v).hexdigest()) for n,v in contents.items()]))
    require(sum(map(len, contents.values()))+len(terminal) <= CAP and
            bytes_used()+sum(map(len, contents.values()))+len(terminal) <= TOTAL_CAP, 'DIAGNOSTIC_BUDGET')
    require(all(sha(r['path']) == r['sha256'] for r in refs), 'INPUT_CHANGED_DURING_DIAGNOSIS')
    OUT.mkdir(exist_ok=False)
    for name, data in contents.items():
        with (OUT/name).open('xb') as f:
            f.write(data)
    with (OUT/'stop_record.json').open('xb') as f:
        f.write(terminal)
    print(json.dumps(dict(output=str(OUT), all_folds_sample_gate_passed=fold_report['all_folds_sample_gate_passed'],
                          total_bytes=sum(map(len, contents.values()))+len(terminal), fit_performed=False), ensure_ascii=False))


if __name__ == '__main__':
    run()
