"""Fixed C1 minus C0 probability diagnostics; no trading or promotion gate."""
import numpy as np
from sklearn.metrics import roc_auc_score

from multiboard_fit_evaluation_v1 import probabilities, paired_loss, filtering
from prepare_multiboard_comparison_v1 import TIME, neutral_keep
from risk_fit_eval_v1 import block_indices


def paired_filter(frame, left, right, policy):
    masks = [neutral_keep(frame.code.to_numpy(), frame[TIME].to_numpy(), p,
             policy['rejection_fraction'], policy['minimum_cross_section']) for p in [left, right]]
    (a, support), (b, other) = masks
    if not np.array_equal(support, other):
        raise ValueError('UNPAIRED_SUPPORT')
    a, b = a & support, b & support
    dates = frame[TIME].to_numpy() // 1000000
    days, ids = np.unique(dates[support], return_inverse=True)
    denominator = np.bincount(ids, weights=a[support], minlength=len(days))
    other_denominator = np.bincount(ids, weights=b[support], minlength=len(days))
    np.testing.assert_array_equal(denominator, other_denominator)
    result = dict(support_rows=int(support.sum()), support_dates=len(days), retained_rows=int(a.sum()))
    if not a.any():
        return dict(result, status='NOT_TESTED_NO_SUPPORTED_CUTOFFS')
    y = frame.y_tail_1200m.to_numpy()
    delta = np.bincount(ids, weights=y[support]*(a[support].astype(int)-b[support].astype(int)), minlength=len(days))
    result['left_minus_right'] = float(delta.sum()/denominator.sum())
    u = policy['uncertainty']
    if len(days) < u['block_dates']:
        return dict(result, status='NOT_TESTED_TOO_FEW_DATES')
    draws = block_indices(len(days), u['reps'], u['block_dates'], u['seed'])
    ratios = delta[draws].sum(1)/denominator[draws].sum(1)
    return dict(result, status='PAIRED_OBSERVED_DATE_MOVING_BLOCK',
                ci95=np.quantile(ratios, [.025, .975]).tolist(), policy=u)


def cutoff_ranking(frame, p):
    numerator, pairs, usable = 0., 0, 0
    for indices in frame.groupby(TIME, sort=True).indices.values():
        y = frame.y_tail_1200m.to_numpy()[indices]
        count = int(y.sum())*int(len(y)-y.sum())
        if count:
            numerator += roc_auc_score(y, np.asarray(p)[indices])*count
            pairs += count
            usable += 1
    return dict(pair_weighted_auc=numerator/pairs if pairs else None,
                mixed_label_cutoffs=usable, positive_negative_pairs=pairs,
                total_cutoffs=int(frame[TIME].nunique()), independent_observations=False)


def evaluate(frame, scores, train_rate, config):
    y = frame.y_tail_1200m.to_numpy()
    dates = frame[TIME].to_numpy()//1000000
    policy = config['evaluation']
    all_scores = dict(scores, train_constant=np.full(len(frame), train_rate))
    pairs, tails = {}, {}
    for algorithm in ['lr', 'lightgbm']:
        for state in ['raw', 'calibrated']:
            a, b = [f'{arm}_{algorithm}_{state}' for arm in ['C1', 'C0']]
            pairs[a+'_minus_'+b] = paired_loss(y, scores[a], scores[b], dates, policy['uncertainty'])
            tails[a+'_minus_'+b] = paired_filter(frame, scores[a], scores[b], policy)
    simple = -frame.stock_return15m_raw.to_numpy(dtype=float)
    known = np.isfinite(simple)
    monthly = {}
    for month in np.unique(dates//100):
        m = dates//100 == month
        monthly[str(month)] = dict(rows=int(m.sum()), dates=int(np.unique(dates[m]).size),
            predictions={k: probabilities(y[m], v[m]) for k, v in all_scores.items()})
    return dict(rows=len(frame), probability_metrics={k: probabilities(y, v) for k, v in all_scores.items()},
        monthly=monthly, paired_loss_differences=pairs, paired_filter_differences=tails,
        same_cutoff_ranking={k: cutoff_ranking(frame, v) for k, v in scores.items()},
        model_vs_random_filter=filtering(frame, scores, policy),
        rule_comparison=filtering(frame.loc[known].reset_index(drop=True),
            dict({k: v[known] for k,v in scores.items()}, simple_negative_return15m=simple[known]), policy) if known.any() else None,
        simple_rule_missing_rows=int((~known).sum()), pnl_evaluated=False,
        classification_accuracy_not_used=True, independent_blind_test=False)
