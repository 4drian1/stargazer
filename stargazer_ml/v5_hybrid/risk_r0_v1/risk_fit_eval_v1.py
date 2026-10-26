"""Fixed daily-cohort comparisons; MAE is adverse price excursion, never trading PnL."""
import math
import numpy as np
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score, average_precision_score


def probability_metrics(y, p):
    y, p = np.asarray(y), np.asarray(p, dtype=float)
    if len(y) == 0 or len(y) != len(p) or not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError('INVALID_PROBABILITY_VECTOR')
    p = np.clip(p, 1e-12, 1 - 1e-12)
    bins = []
    for i in range(10):
        mask = (p >= i / 10) & (p < (i + 1) / 10)
        bins.append(dict(lower=i/10, upper=(i+1)/10, n=int(mask.sum()),
                         mean_probability=float(p[mask].mean()) if mask.any() else None,
                         observed_tail_rate=float(y[mask].mean()) if mask.any() else None))
    return dict(brier=float(brier_score_loss(y, p)), log_loss=float(log_loss(y, p, labels=[0, 1])),
                roc_auc=float(roc_auc_score(y, p)) if len(set(y)) == 2 else None,
                average_precision=float(average_precision_score(y, p)) if y.sum() else None,
                reliability_bins=bins)


def keep_mask(rows, scores, fraction):
    if not 0 <= fraction < 1 or len(rows) != len(scores) or not np.isfinite(scores).all():
        raise ValueError('INVALID_FILTER_INPUT')
    days = {}
    for i, row in enumerate(rows):
        days.setdefault(row['P'], []).append(i)
    keep = np.ones(len(rows), dtype=bool)
    for indices in days.values():
        reject = math.floor(len(indices) * fraction)
        ranked = sorted(indices, key=lambda i: (-float(scores[i]), rows[i]['code']))
        keep[ranked[:reject]] = False
    return keep


def block_indices(n, reps, block, seed):
    if n < block or block < 1 or reps < 1:
        raise ValueError('INSUFFICIENT_DATES_FOR_BLOCK_BOOTSTRAP')
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, n - block + 1, size=(reps, math.ceil(n / block)))
    return (starts[:, :, None] + np.arange(block)).reshape(reps, -1)[:, :n]


def interval(point, draws):
    if not np.isfinite(point) or not np.isfinite(draws).all():
        raise ValueError('NONFINITE_EVALUATION')
    return dict(mean=float(point), ci95=[float(x) for x in np.quantile(draws, [.025, .975])])


def risk_groups(rows, scores):
    order = sorted(range(len(rows)), key=lambda i: (float(scores[i]), rows[i]['P'], rows[i]['code']))
    result = []
    for group in np.array_split(order, 10):
        if len(group) == 0:
            continue
        result.append(dict(n=len(group), mean_score=float(np.mean(np.asarray(scores)[group])),
                           tail_rate=float(np.mean([rows[i]['y_tail_5d'] for i in group]))))
    return dict(groups_low_to_high=result, distinct_scores=len(set(float(v) for v in scores)),
                note='Equal-count groups; tied scores may span groups and imply no risk separation.')


def board(code):
    return ('STAR' if code.startswith('688') else 'GEM' if code.startswith(('300', '301'))
            else 'SH_MAIN' if code.startswith('6') else 'SZ_MAIN')


def evaluate_segment(rows, probabilities, constant_probability, config):
    dates = sorted({r['P'] for r in rows})
    di = {d: i for i, d in enumerate(dates)}
    date_ids = np.array([di[r['P']] for r in rows])
    counts = np.bincount(date_ids, minlength=len(dates)).astype(float)
    y = np.array([r['y_tail_5d'] for r in rows])
    mae = np.array([r['mae_5d'] for r in rows])
    index = block_indices(len(dates), config['bootstrap_reps'], config['block_dates'], config['seed'])
    constant = np.full(len(rows), constant_probability)
    metrics = {'constant': probability_metrics(y, constant)}
    for name, values in probabilities.items():
        metrics[name] = probability_metrics(y, values)
        loss_diff = (np.asarray(values) - y)**2 - (constant - y)**2
        sums = np.bincount(date_ids, weights=loss_diff, minlength=len(dates))
        metrics[name]['brier_minus_constant'] = interval(loss_diff.mean(), sums[index].sum(1) / counts[index].sum(1))
    scores = dict(atr_rule=np.array([r['features']['atr14_pct'] for r in rows]),
                  lr=probabilities['lr'], lightgbm=probabilities['lightgbm'])
    summaries = []
    fractions = [config['primary_rejection_fraction']] + config['descriptive_rejection_fractions']
    for fraction in fractions:
        masks = {name: keep_mask(rows, s, fraction) for name, s in scores.items()}
        kept = np.bincount(date_ids, weights=masks['atr_rule'], minlength=len(dates))
        expected = np.bincount(date_ids, weights=y, minlength=len(dates)) * kept / counts
        daily = {'random_expected': expected}
        arm_stats = {}
        for name, mask in masks.items():
            require_counts = np.bincount(date_ids, weights=mask, minlength=len(dates))
            if not np.array_equal(kept, require_counts):
                raise ValueError('UNPAIRED_RETAINED_COUNTS')
            sums = np.bincount(date_ids, weights=y * mask, minlength=len(dates))
            daily[name] = sums
            retained_mae = mae[mask]
            threshold = np.quantile(retained_mae, .05)
            exposure = {b: sum(bool(mask[i]) and board(r['code']) == b for i, r in enumerate(rows))
                        for b in ('SH_MAIN', 'SZ_MAIN', 'GEM', 'STAR')}
            arm_stats[name] = dict(retained=int(mask.sum()), rejected=int((~mask).sum()),
                                   tail_rate=float(y[mask].mean()), mean_mae=float(retained_mae.mean()),
                                   q05_mae=float(threshold),
                                   worst5pct_mae_mean=float(retained_mae[retained_mae <= threshold].mean()),
                                   board_retained_counts=exposure)
        pairs = {}
        for name in ('lr', 'lightgbm'):
            for baseline in ('atr_rule', 'random_expected'):
                delta = daily[name] - daily[baseline]
                pairs[name + '_minus_' + baseline] = interval(delta.sum() / kept.sum(),
                    delta[index].sum(1) / kept[index].sum(1))
        summaries.append(dict(rejection_fraction=fraction,
                              role='primary' if fraction == config['primary_rejection_fraction'] else 'descriptive',
                              arms=arm_stats, random_expected_tail_rate=float(expected.sum() / kept.sum()),
                              random_expected_mean_mae=float((np.bincount(date_ids, weights=mae,
                                  minlength=len(dates)) * kept/counts).sum() / kept.sum()),
                              paired_tail_rate_differences=pairs,
                              daily_sufficient_statistics=dict(dates=dates, population_counts=counts.tolist(),
                                  retained_counts=kept.tolist(), tail_counts={k: v.tolist() for k, v in daily.items()})))
    return dict(n=len(rows), dates=len(dates), unfiltered_tail_rate=float(y.mean()),
                unfiltered_mean_mae=float(mae.mean()), probability=metrics, filtering=summaries,
                risk_groups={name: risk_groups(rows, score) for name, score in scores.items()},
                bootstrap='Paired moving blocks of whole P-date cohorts, no independent stock-day resampling.')
