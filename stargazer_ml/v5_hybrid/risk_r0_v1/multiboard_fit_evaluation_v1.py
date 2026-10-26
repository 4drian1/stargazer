"""Paired probability and equal-retention risk diagnostics, never trading PnL."""
import numpy as np
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score, average_precision_score

from prepare_multiboard_comparison_v1 import TIME, neutral_keep
from risk_fit_eval_v1 import block_indices


def probabilities(y, p):
    y, p = np.asarray(y), np.asarray(p, dtype=float)
    if not len(y) or len(y) != len(p) or not np.isin(y, [0, 1]).all() or not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError('INVALID_PROBABILITY_OR_TARGET')
    q = np.clip(p, 1e-12, 1 - 1e-12)
    bins = []
    for i in range(10):
        mask = (p >= i / 10) & ((p < (i + 1) / 10) if i < 9 else (p <= 1))
        bins.append(dict(lower=i / 10, rows=int(mask.sum()),
                         predicted=float(p[mask].mean()) if mask.any() else None,
                         actual=float(y[mask].mean()) if mask.any() else None))
    return dict(brier=float(brier_score_loss(y, p)), log_loss=float(log_loss(y, q, labels=[0, 1])),
                roc_auc=float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else None,
                pr_auc_average_precision=float(average_precision_score(y, p)) if y.sum() else None,
                predicted_risk=float(p.mean()), actual_risk=float(y.mean()),
                actual_minus_predicted=float(y.mean() - p.mean()), reliability_bins=bins)


def paired_loss(y, left, right, dates, policy):
    """left minus right; negative favors left. Dates are the clustering unit."""
    y, dates = np.asarray(y), np.asarray(dates)
    ordered, ids = np.unique(dates, return_inverse=True)
    if len(ordered) < policy['block_dates']:
        return dict(status='NOT_TESTED_TOO_FEW_DATES', dates=len(ordered))
    draws = block_indices(len(ordered), policy['reps'], policy['block_dates'], policy['seed'])
    counts = np.bincount(ids).astype(float)
    left, right = np.clip(left, 1e-12, 1-1e-12), np.clip(right, 1e-12, 1-1e-12)
    differences = {'brier': (left-y)**2 - (right-y)**2,
        'log_loss': -y*np.log(left)-(1-y)*np.log1p(-left) + y*np.log(right)+(1-y)*np.log1p(-right)}
    result = dict(status='PAIRED_OBSERVED_DATE_MOVING_BLOCK', dates=len(ordered), policy=policy)
    for name, difference in differences.items():
        sums = np.bincount(ids, weights=difference, minlength=len(ordered))
        estimate = sums[draws].sum(axis=1) / counts[draws].sum(axis=1)
        result[name] = dict(left_minus_right=float(difference.mean()), ci95=np.quantile(estimate, [.025, .975]).tolist())
    return result


def filtering(frame, scores, policy):
    if not scores:
        raise ValueError('EMPTY_SCORE_COMPARISON')
    y = frame.y_tail_1200m.to_numpy()
    codes, times = frame.code.to_numpy(), frame[TIME].to_numpy()
    masks, support = {}, None
    for name, values in scores.items():
        keep, comparable = neutral_keep(codes, times, values, policy['rejection_fraction'], policy['minimum_cross_section'])
        if support is not None and not np.array_equal(support, comparable):
            raise ValueError('UNPAIRED_FILTER_SUPPORT')
        support = comparable
        masks[name] = keep & support
    retained = {name: int(mask.sum()) for name, mask in masks.items()}
    if len(set(retained.values())) != 1:
        raise ValueError('UNPAIRED_RETAINED_COUNTS')
    output = dict(rows=len(frame), comparable_rows=int(support.sum()),
                  excluded_tiny_rows=int((~support).sum()), arms={}, random_expected_tail_rate=None,
                  paired_tail_rate_differences={})
    cutoffs, ids = np.unique(times[support], return_inverse=True)
    count = np.bincount(ids)
    positives = np.bincount(ids, weights=y[support])
    kept = count - np.floor(policy['rejection_fraction'] * count)
    if kept.sum():
        output['random_expected_tail_rate'] = float((positives * kept / count).sum() / kept.sum())
    date_ids = None
    draws = None
    if support.any():
        days = frame[TIME].to_numpy()[support] // 1000000
        unique_days, date_ids = np.unique(days, return_inverse=True)
        u = policy['uncertainty']
        if len(unique_days) >= u['block_dates']:
            draws = block_indices(len(unique_days), u['reps'], u['block_dates'], u['seed'])
    daily = {}
    for name, mask in masks.items():
        point = float(y[mask].mean()) if mask.any() else None
        item = dict(retained_rows=int(mask.sum()), retained_tail_events=int(y[mask].sum()), tail_rate=point)
        if draws is not None:
            member = mask[support]
            sums = np.bincount(date_ids, weights=y[support] * member)
            denominators = np.bincount(date_ids, weights=member)
            daily[name] = sums
            # Every supported date has at least one nonempty retained cross section.
            ratios = sums[draws].sum(1) / denominators[draws].sum(1)
            item['tail_rate_ci95'] = np.quantile(ratios, [.025, .975]).tolist()
        output['arms'][name] = item
    if draws is not None:
        daily['random_expected'] = np.bincount(np.searchsorted(unique_days, cutoffs // 1000000),
            weights=positives * kept / count, minlength=len(unique_days))
        pairs = [(name, 'random_expected') for name in scores]
        for algorithm in ['lr', 'lightgbm']:
            for state in ['raw', 'calibrated']:
                for left, right in [('M1', 'B1'), ('B1', 'B0')]:
                    a, b = f'{left}_{algorithm}_{state}', f'{right}_{algorithm}_{state}'
                    if a in daily and b in daily:
                        pairs.append((a, b))
        for a, b in pairs:
            delta = daily[a] - daily[b]
            draws_delta = delta[draws].sum(1) / denominators[draws].sum(1)
            output['paired_tail_rate_differences'][a + '_minus_' + b] = dict(
                left_minus_right=float(delta.sum() / denominators.sum()),
                ci95=np.quantile(draws_delta, [.025, .975]).tolist())
    return output


def evaluate(frame, scores, train_rate, config):
    y = frame.y_tail_1200m.to_numpy()
    days = frame[TIME].to_numpy() // 1000000
    all_scores = dict(scores, train_constant=np.full(len(frame), train_rate))
    metrics = {name: probabilities(y, values) for name, values in all_scores.items()}
    months = {}
    for month in np.unique(days // 100):
        mask = days // 100 == month
        months[str(month)] = dict(rows=int(mask.sum()), dates=int(np.unique(days[mask]).size),
            stocks=int(frame.loc[mask, 'code'].nunique()),
            predictions={name: probabilities(y[mask], values[mask]) for name, values in all_scores.items()})
    pairs = {}
    for algorithm in ['lr', 'lightgbm']:
        for state in ['raw', 'calibrated']:
            for left, right in [('M1', 'B1'), ('B1', 'B0')]:
                a, b = f'{left}_{algorithm}_{state}', f'{right}_{algorithm}_{state}'
                pairs[a + '_minus_' + b] = paired_loss(y, scores[a], scores[b], days, config['evaluation']['uncertainty'])
    policy = config['evaluation']
    model_filter = filtering(frame, scores, policy)
    simple = -frame.stock_return15m_raw.to_numpy(dtype=float)
    known = np.isfinite(simple)
    rule_filter = filtering(frame.loc[known].reset_index(drop=True),
        dict({k: v[known] for k, v in scores.items()}, simple_negative_return15m=simple[known]), policy) if known.any() else None
    return dict(rows=len(frame), probability_metrics=metrics, paired_loss_differences=pairs,
                monthly=months, model_vs_random_filter=model_filter,
                rule_comparison=rule_filter, simple_rule_missing_rows=int((~known).sum()),
                alert_threshold=None, classification_accuracy_not_used=True, pnl_evaluated=False)
