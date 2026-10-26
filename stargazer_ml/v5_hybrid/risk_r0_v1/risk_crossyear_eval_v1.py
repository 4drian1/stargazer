"""Cross-year additions to the frozen evaluator; price excursion is not PnL."""
import numpy as np

from risk_fit_eval_v1 import block_indices, evaluate_segment, interval, keep_mask
from risk_failure_diagnostics_v1 import group_diagnostics


def deep_tail(values):
    values = np.asarray(values, dtype=float)
    if not len(values) or not np.isfinite(values).all():
        raise ValueError('INVALID_DEEP_TAIL_VALUES')
    return float(values[values <= np.quantile(values, .05)].mean())


def paired_additions(rows, probabilities, config):
    dates = sorted({r['P'] for r in rows})
    by_day = {d: [] for d in dates}
    for i, row in enumerate(rows):
        by_day[row['P']].append(i)
    indices = [np.array(by_day[d]) for d in dates]
    scores = dict(atr_rule=np.array([r['features']['atr14_pct'] for r in rows]),
                  lr=np.asarray(probabilities['lr']), lightgbm=np.asarray(probabilities['lightgbm']))
    masks = {m: keep_mask(rows, v, config['primary_rejection_fraction']) for m, v in scores.items()}
    if any(len({int(mask[ix].sum()) for mask in masks.values()}) != 1 for ix in indices):
        raise ValueError('UNPAIRED_DAILY_COUNTS')
    y = np.array([r['y_tail_5d'] for r in rows])
    mae = np.array([r['mae_5d'] for r in rows])
    blocks = block_indices(len(dates), config['bootstrap_reps'], config['block_dates'], config['seed'])
    # Resample the SAME whole-date blocks for every arm; repeat the raw MAE observations.
    # Recompute each arm's 5% quantile each time, including ties (original evaluator convention).
    tail_draws, deep_draws = [], {b: [] for b in ('atr_rule', 'lr')}
    for draw in blocks:
        ix = np.concatenate([indices[d] for d in draw])
        kept = {m: ix[mask[ix]] for m, mask in masks.items()}
        tail_draws.append(float(y[kept['lightgbm']].mean()-y[kept['lr']].mean()))
        gb = deep_tail(mae[kept['lightgbm']])
        for baseline in deep_draws:
            deep_draws[baseline].append(gb-deep_tail(mae[kept[baseline]]))
    tail = interval(y[masks['lightgbm']].mean()-y[masks['lr']].mean(), tail_draws)
    deep = {b: interval(deep_tail(mae[masks['lightgbm']])-deep_tail(mae[masks[b]]), draws)
            for b, draws in deep_draws.items()}
    return dict(lightgbm_minus_lr_tail_rate=tail,
        worst5pct_mae_differences={f'lightgbm_minus_{b}': v for b, v in deep.items()},
        sign='Tail-rate delta: negative better. MAE delta: negative WORSE, positive less adverse.',
        method='Same 10-P-date moving blocks; raw retained MAE quantile recomputed; ties included.')


def evaluate_test(rows, probabilities, constant, config):
    result = evaluate_segment(rows, probabilities, constant, config)
    result['paired_additions'] = paired_additions(rows, probabilities, config)
    # More-negative MAE is worse. Thus GB-minus-baseline CI entirely BELOW zero is deterioration.
    result['paired_additions']['deep_tail_red_flag'] = any(
        v['ci95'][1] < 0 for v in result['paired_additions']['worst5pct_mae_differences'].values())
    masks = {m: keep_mask(rows, scores, config['primary_rejection_fraction']) for m, scores in dict(
        lr=probabilities['lr'], lightgbm=probabilities['lightgbm'],
        atr_rule=np.array([r['features']['atr14_pct'] for r in rows])).items()}
    augmented = [dict(r, atr_score=r['features']['atr14_pct'],
        **{k: float(v[i]) for k, v in probabilities.items()},
        **{k+'_retained_primary': bool(v[i]) for k, v in masks.items()}) for i, r in enumerate(rows)]
    result['same_day_ranking'] = group_diagnostics(augmented)['ranking']
    result['role'] = 'HISTORICAL_RESEARCH_TEST_NOT_NEW_BLIND_TEST'
    return result


def consistency_flags(reports):
    if len(reports) != 3:
        raise ValueError('ALL_THREE_FOLDS_REQUIRED')
    pairs = [r['filtering'][0]['paired_tail_rate_differences'] for r in reports]
    keys = ('lightgbm_minus_atr_rule', 'lightgbm_minus_random_expected')
    tail = (sum(all(p[k]['ci95'][1] < 0 for k in keys) for p in pairs) >= 2
            and all(p[k]['mean'] <= 0 for p in pairs for k in keys))
    probability = sum(r['probability']['lightgbm']['brier_minus_constant']['ci95'][1] < 0
        and r['probability']['lightgbm']['log_loss'] <= r['probability']['constant']['log_loss']
        for r in reports) >= 2
    red_flags = [i+1 for i, r in enumerate(reports) if r['paired_additions']['deep_tail_red_flag']]
    return dict(tail_consistency=tail, probability_consistency=probability,
        deep_tail_red_flag_folds=red_flags, research_consistency=tail and probability and not red_flags,
        production_gate_passed=False, deployment_authorized=False)
