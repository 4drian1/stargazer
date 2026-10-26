"""risk_joint_eval_v1.py — 冻结风险模型离线联合对照（用户 2026-10-19 授权）。

人口=A1评分候选（risk_scores_a1_v1，2,106条，2025主分析/2026单列）。
固定比较（同日同保留数量，避免"少留股票"被误认为能力）：
  A1        星币侍从单独（评分集，无过滤）
  A1+risk   每日按 risk_lgbm_cal 降序剔除 floor(n×20%)
  A1+random 同保留数量随机剔除（seed固定）
  A1+atr    同保留数量，按 ATR14%（ frozen 特征字段）降序剔除
结局：七日变化(chg7)+不利下探(mae7严格窗, P后七交易日, 用缓存bars)•胜率•
被剔除中的上涨机会(excluded chg7>0占比)。
不确定性：完整日历连续7日块（block>=标签窗），90%暂定区间。
五日风险目标：扩展股票尺度证据单源，label_at 拒绝出标签→如实披露改用
七日结局评价；期限差异(5日尾部 vs 7日变化)不抹平。
缺失风险分数不当低风险（本批2,106条全有分数；34条不足单列不进入）。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'step9_preflight'))
from trading_calendar import load_calendar  # noqa: E402

CAL = load_calendar()
BOOT, SEED = 5000, 20261019
ROUND = HERE.parent / 'green_common_profile/round_a_v0_1_impl_fix'


def cal_blocks(day_index, origin):
    pos = {d: i for i, d in enumerate(CAL)}
    o = pos[int(origin)]
    return {d: (pos[int(d)] - o) // 7 for d in day_index}


def boot(day_diffs: pd.Series, origin, rng):
    arr = day_diffs.values
    if len(arr) < 5:
        return None
    seg = cal_blocks(list(day_diffs.index), origin)
    seg_ids = np.array(sorted(set(seg.values())))
    seg_of = np.array([seg[d] for d in day_diffs.index])
    days_of_seg = {si: np.where(seg_of == si)[0] for si in seg_ids}
    reps = np.empty(BOOT)
    for k in range(BOOT):
        pick = seg_ids[rng.integers(0, len(seg_ids), len(seg_ids))]
        idx = np.concatenate([days_of_seg[si] for si in pick])
        reps[k] = arr[idx].mean()
    return dict(point_pp=round(float(arr.mean()) * 100, 3),
                p5_pp=round(float(np.percentile(reps, 5)) * 100, 3),
                p95_pp=round(float(np.percentile(reps, 95)) * 100, 3),
                n_days=int(len(arr)), n_blocks=int(len(seg_ids)),
                ci='90%暂定;块=7日历交易日;块长=标签窗不消除边界依赖')


def main():
    scores = pd.read_parquet(HERE / 'risk_coverage_extend_v1/risk_scores_a1_v1.parquet')
    scores = scores[scores['feature_status'] == 'OK'].copy()
    master = pd.read_parquet(ROUND / 'full_master_table_v2.parquet')
    m = master.set_index(['code', 'P'])
    scores['chg7'] = [m.loc[(c, int(p)), 'chg7']
                      for c, p in zip(scores['code'], scores['P'])]
    scores['label_end'] = [int(str(m.loc[(c, int(p)), 'label_end_ts'])[:8])
                           for c, p in zip(scores['code'], scores['P'])]
    cache = joblib.load(HERE / 'risk_coverage_extend_v1/code_cache.pkl')

    # mae7 严格窗 + ATR14% 排序字段（用缓存 bars，零查询）
    mae, atr = [], []
    for r in scores.itertuples(index=False):
        bars = cache.get(r.code, {}).get('bars', {})
        days = [d for d in CAL if int(r.P) < d <= int(r.label_end)]
        pc = bars.get(int(r.P), {}).get('close')
        ls = [bars.get(d, {}).get('low') for d in days]
        mae.append(round(min(ls) / pc - 1.0, 6)
                   if len(days) == 7 and pc and all(x and x > 0 for x in ls)
                   else None)
        idx = CAL.index(int(r.P))
        h61 = [bars.get(d) for d in CAL[max(0, idx - 60):idx + 1]]
        if len(h61) == 61 and all(h61) and pc:
            trs = [max(b['high'] - b['low'], abs(b['high'] - h61[i - 1]['close']),
                       abs(b['low'] - h61[i - 1]['close']))
                   for i, b in enumerate(h61) if i > 0]
            atr.append(sum(trs[-14:]) / 14.0 / pc)
        else:
            atr.append(None)
    scores['mae7'] = mae
    scores['atr14_pct'] = atr
    scores = scores.dropna(subset=['mae7', 'atr14_pct', 'chg7']).copy()
    rng = np.random.default_rng(SEED)
    origin = int(sorted(scores['P'])[0])
    print('联合评价人口:', len(scores), flush=True)

    def filter_mask(g, key, n_drop):
        if n_drop <= 0:
            return pd.Series(False, index=g.index)
        if key == 'random':
            r = np.random.default_rng(SEED + int(g['P'].iloc[0]))
            drop = r.choice(g.index, size=n_drop, replace=False)
            return pd.Series(g.index.isin(drop), index=g.index)
        worst = g.sort_values(key, ascending=False).head(n_drop)
        return pd.Series(g.index.isin(worst.index), index=g.index)

    arms_rows = {}
    for year, gy in scores.groupby(scores['P'] // 10000):
        res = {}
        kept = {}
        for name in ('A1', 'A1_risk', 'A1_random', 'A1_atr'):
            res[name] = dict(n=0, chg_mean=0.0, mae_mean=0.0, win=0.0, excl_up=0.0)
        for p, gd in gy.groupby('P'):
            n = len(gd)
            n_drop = int(n * 0.2)
            masks = dict(A1=pd.Series(False, index=gd.index),
                         A1_risk=filter_mask(gd, 'risk_lgbm_cal', n_drop),
                         A1_random=filter_mask(gd, 'random', n_drop),
                         A1_atr=filter_mask(gd, 'atr14_pct', n_drop))
            for name, mask in masks.items():
                keep = gd[~mask]
                res[name]['n'] += len(keep)
                res[name]['chg_mean'] += keep['chg7'].sum()
                res[name]['mae_mean'] += keep['mae7'].sum()
                res[name]['win'] += (keep['chg7'] > 0).sum()
                if len(gd[mask]):
                    res[name]['excl_up'] += (gd.loc[mask, 'chg7'] > 0).sum()
                    res[name]['excl_n'] = res[name].get('excl_n', 0) + int(mask.sum())
                    res[name]['excl_up_total'] = res[name].get('excl_up_total', 0) + int(
                        (gd['chg7'] > 0).sum())
                    res.setdefault('__excl_chg', {}).setdefault(name, []).extend(
                        gd.loc[mask, 'chg7'].tolist())
                kept.setdefault(name, {})[p] = keep
        for name in res:
            if name == '__excl_chg':
                continue
            r = res[name]
            r['chg_mean'] = round(float(r['chg_mean'] / r['n']) * 100, 3)
            r['mae_mean'] = round(float(r['mae_mean'] / r['n']) * 100, 3)
            r['win'] = round(float(r['win'] / r['n']), 4)
            if name != 'A1' and r.get('excl_n'):
                r['excl_up_share'] = round(r['excl_up'] / r['excl_n'], 4)
                r['false_removal_rate'] = round(
                    r['excl_up'] / max(1, r['excl_up_total']), 4)
                ec = res['__excl_chg'].get(name, [])
                r['excl_chg_mean_pp'] = round(float(np.mean(ec)) * 100, 3)
                r['excl_chg_median_pp'] = round(float(np.median(ec)) * 100, 3)
            r.pop('excl_n', None)
            r.pop('excl_up', None)
            r.pop('excl_up_total', None)
        excl_store = res.pop('__excl_chg', {})
        arms_rows[int(year)] = dict(stats=res, kept=kept)

    out = dict(schema='PENTACLES_RISK_JOINT_V1_1',
               population=int(len(scores)),
               horizon_note='七日结局评价五日风险过滤;期限差异不抹平', benchmark_note='diffs_vs_A1=对精确随机期望之差(同保留随机保留均值期望=当日A1均值);A1_random单次蒙特卡洛仅sanity',
               by_year={})
    for year, pack in arms_rows.items():
        stats, kept = pack['stats'], pack['kept']
        yday = {}
        for name in ('A1_risk', 'A1_random', 'A1_atr'):
            da = {p: k['chg7'].mean() for p, k in kept['A1'].items()
                  if p in kept[name]}
            db = {p: k['chg7'].mean() for p, k in kept[name].items()}
            common = sorted(set(da) & set(db))
            d_chg = pd.Series({p: db[p] - da[p] for p in common})
            da_m = {p: k['mae7'].mean() for p, k in kept['A1'].items()
                    if p in kept[name]}
            db_m = {p: k['mae7'].mean() for p, k in kept[name].items()}
            d_mae = pd.Series({p: db_m[p] - da_m[p] for p in common})
            yday[name] = dict(
                chg7=boot(d_chg, origin, rng), mae7=boot(d_mae, origin, rng))
        paired = {}
        for a, b in (('A1_risk', 'A1_random'), ('A1_risk', 'A1_atr')):
            da_c = {pp: k['chg7'].mean() for pp, k in kept[a].items()}
            db_c = {pp: k['chg7'].mean() for pp, k in kept[b].items()}
            da_m = {pp: k['mae7'].mean() for pp, k in kept[a].items()}
            db_m = {pp: k['mae7'].mean() for pp, k in kept[b].items()}
            common = sorted(set(da_c) & set(db_c))
            paired[a + '_minus_' + b] = dict(
                chg7=boot(pd.Series({pp: da_c[pp] - db_c[pp] for pp in common}),
                          origin, rng),
                mae7=boot(pd.Series({pp: da_m[pp] - db_m[pp] for pp in common}),
                          origin, rng))
        out['by_year'][str(year)] = dict(stats=stats, diffs_vs_A1=yday,
                                         paired_filter_diffs=paired)
    pout = HERE / 'risk_joint_eval_v1.json'
    pout.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding='utf-8')
    print(json.dumps(out, ensure_ascii=False)[:2200])


if __name__ == '__main__':
    main()
