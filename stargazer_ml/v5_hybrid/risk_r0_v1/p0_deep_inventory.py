"""Local P0 audit only: no labels, models, network, or source modifications."""
from collections import Counter
from datetime import datetime, timezone, timedelta
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

BASE = Path(__file__).resolve().parent.parent
OUT = Path(__file__).resolve().parent
DAILY = Path('D:/a股数据/data_daily_parts')
ROUND = BASE / 'green_common_profile/gcp_round_b_small_sample_v1'
CHAIN = BASE / 'opportunity_v2/gate1c/feb2026_adjustment_cache_v1/adj_cache.json'
SECTOR = BASE / 'opportunity_v2/o5_sector_lag/pit_membership.parquet'
COLS = ['code', 'date', 'open', 'high', 'low', 'close', 'pre_close', 'volume', 'amount', 'is_st']


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def fingerprint(path):
    return {'path': str(path), 'bytes': path.stat().st_size, 'sha256': sha(path)}


def normalize(frame):
    frame = frame.copy()
    codes = frame.code.astype(str)
    if not codes.str.fullmatch(r'\d{6}').all():
        raise ValueError('Invalid code; no silent normalization')
    dates = pd.to_numeric(frame.date, errors='raise')
    if (dates != np.floor(dates)).any():
        raise ValueError('Fractional date')
    frame['date'] = dates.astype('int32')
    pd.to_datetime(frame.date.astype(str), format='%Y%m%d', errors='raise')
    frame['code'] = codes.astype('int32')
    return frame


def keys(frame):
    return frame.code.to_numpy(dtype='int64') * 100000000 + frame.date.to_numpy(dtype='int64')


def board(code):
    code = str(code).zfill(6)
    if code.startswith(('688', '689')):
        return 'STAR_PREFIX'
    if code.startswith(('300', '301')):
        return 'GEM_PREFIX'
    if code.startswith(('600', '601', '603', '605')):
        return 'SH_MAIN_PREFIX'
    if code.startswith(('000', '001', '002', '003')):
        return 'SZ_MAIN_PREFIX'
    if code.startswith(('4', '8', '92')):
        return 'BSE_PREFIX'
    return 'OTHER_OR_NON_STOCK'


def forward_coverage(candidates, daily_keys, calendar):
    dates = candidates.date.to_numpy(dtype='int64')
    codes = candidates.code.to_numpy(dtype='int64')
    pos = np.searchsorted(calendar, dates)
    in_calendar = (pos < len(calendar)) & (calendar[np.minimum(pos, len(calendar)-1)] == dates)
    mature = in_calendar & (pos + 5 < len(calendar))
    count = np.zeros(len(candidates), dtype='int8')
    for h in range(1, 6):
        future = calendar[np.minimum(pos + h, len(calendar)-1)]
        query = codes * 100000000 + future
        at = np.searchsorted(daily_keys, query)
        hit = (at < len(daily_keys)) & (daily_keys[np.minimum(at, len(daily_keys)-1)] == query)
        count += (hit & mature).astype('int8')
    return {'rows': len(candidates), 'anchor_date_not_observed': int((~in_calendar).sum()),
            'calendar_tail': int((in_calendar & ~mature).sum()),
            'all_next5_rows_present': int((mature & (count == 5)).sum()),
            'mature_but_missing_stock_rows': int((mature & (count != 5)).sum()),
            'meaning': 'File-row coverage only, NOT label readiness; observed calendar is not exchange-certified'}


def candidate_report(path, panel, calendar, chains, membership, manifest):
    manifest.append(fingerprint(path))
    if path.suffix == '.json':
        records = json.loads(path.read_text(encoding='utf-8'))['events']
        frame = pd.DataFrame(records).rename(columns={'E': 'date'})[['code', 'date', 'branch_at_decision']]
        qualifiers = {'branch_counts': frame.branch_at_decision.value_counts().to_dict(),
                      'population': 'Full 1013 box population, not only 124 admitted labels'}
    else:
        names = pq.ParquetFile(path).schema_arrow.names
        cols = [c for c in ['code', 'date', 'mode_id', 'rule_version', 'candidate_generated_asof',
                           'candidate_contract_version', 'source_data_snapshot_id'] if c in names]
        frame = pd.read_parquet(path, columns=cols)
        qualifiers = {'population': 'Existing source rows; NOT adopted as R0 population',
                      'schema': names, 'asof_columns_present': [c for c in cols if c not in ['code', 'date']]}
        if 'mode_id' in frame:
            qualifiers['mode_counts'] = frame.mode_id.value_counts().to_dict()
        for col in ['rule_version', 'candidate_contract_version', 'source_data_snapshot_id']:
            if col in frame:
                qualifiers[col] = frame[col].value_counts(dropna=False).to_dict()
        qualifiers['available_at_certification'] = 'NOT_TESTED; retrospective asof value is not capture-time proof'
    frame = normalize(frame)
    dk = np.sort(keys(panel))
    unique = frame[['code', 'date']].drop_duplicates()
    grouped = []
    # Each source is reported independently; never union the research populations.
    for year, part in unique.groupby(unique.date // 10000):
        stat = forward_coverage(part, dk, calendar)
        stat.update(year=int(year), stocks=int(part.code.nunique()), dates=int(part.date.nunique()),
                    board_prefix_counts=dict(Counter(board(c) for c in part.code)),
                    chain_status_counts=dict(Counter(chains.get(str(c).zfill(6), {}).get('status', 'UNKNOWN') for c in part.code)))
        lookup = panel.set_index(['code', 'date'])[['pre_close', 'is_st']].reindex(
            pd.MultiIndex.from_frame(part[['code', 'date']]))
        stat['anchor_row_missing'] = int(lookup.pre_close.isna().sum())
        stat['pre_close_nonpositive_or_missing'] = int((lookup.pre_close.isna() | (lookup.pre_close <= 0)).sum())
        stat['st_value_unknown'] = int((~lookup.is_st.isin([0, 1])).sum())
        mapped = 0
        ambiguous = 0
        for code, chunk in part.groupby('code'):
            intervals = membership.get(str(code).zfill(6), [])
            n = np.zeros(len(chunk), dtype='int16')
            ds = chunk.date.to_numpy()
            for start, end in intervals:
                n += ((ds >= start) & (ds < end)).astype('int16')
            mapped += int((n == 1).sum())
            ambiguous += int((n > 1).sum())
        stat['sector_effective_interval_single_match'] = mapped
        stat['sector_effective_interval_ambiguous'] = ambiguous
        stat['sector_available_at_certified'] = None
        stat['listing_date_certified'] = None
        grouped.append(stat)
    return {'input': str(path), 'physical_rows': len(frame), 'unique_code_date': len(unique),
            'duplicate_code_date_excess': len(frame)-len(unique), **qualifiers, 'by_year': grouped}


def main():
    outputs = [OUT / 'risk_population_inventory_v2.json', OUT / 'P0_deep_inventory_report_v2.md']
    if any(p.exists() for p in outputs):
        raise FileExistsError('Immutable audit outputs already exist; use a new version')
    manifest, file_reports, frames = [], [], []
    for path in sorted(DAILY.glob('*.parquet')):
        before = fingerprint(path)
        frame = normalize(pd.read_parquet(path, columns=COLS))
        numeric = frame[COLS[2:]].to_numpy(dtype='float64')
        report = {'file': path.name, 'rows': len(frame), 'date_min': int(frame.date.min()),
                  'date_max': int(frame.date.max()), 'stocks': int(frame.code.nunique()),
                  'duplicate_key_excess': int(frame.duplicated(['code', 'date']).sum()),
                  'null_by_field': frame[COLS[2:]].isna().sum().astype(int).to_dict(),
                  'infinity_by_field': dict(zip(COLS[2:], np.isinf(numeric).sum(axis=0).astype(int).tolist())),
                  'nonpositive_ohlc_rows': int((frame[['open', 'high', 'low', 'close']] <= 0).any(axis=1).sum()),
                  'negative_volume_or_amount': int(((frame.volume < 0) | (frame.amount < 0)).sum()),
                  'zero_volume_rows': int((frame.volume == 0).sum()),
                  'ohlc_order_invalid': int(((frame.low > frame[['open', 'close']].min(axis=1)) |
                                            (frame.high < frame[['open', 'close']].max(axis=1)) |
                                            (frame.low > frame.high)).sum())}
        if sha(path) != before['sha256']:
            raise RuntimeError(f'Source changed during read: {path}')
        manifest.append(before)
        frame['_file'] = path.name
        frames.append(frame)
        file_reports.append(report)
        print(f"daily {len(file_reports)}: {path.name} {len(frame)} rows", flush=True)
    panel = pd.concat(frames, ignore_index=True)
    del frames
    dup = panel[panel.duplicated(['code', 'date'], keep=False)]
    conflicting = dup.groupby(['code', 'date'])[COLS[2:]].nunique(dropna=False).gt(1).any(axis=1)
    duplicate_summary = {'duplicate_keys': int(len(conflicting)), 'duplicate_rows': len(dup),
                         'conflicting_numeric_keys': int(conflicting.sum()),
                         'identical_numeric_keys': int((~conflicting).sum()),
                         'examples': dup.head(20).replace({np.nan: None}).to_dict('records')}
    # Conflicting duplicates are excluded only from coverage lookup, not repaired or erased.
    badkeys = set(conflicting[conflicting].index)
    if badkeys:
        index = pd.MultiIndex.from_frame(panel[['code', 'date']])
        panel = panel[~index.isin(badkeys)]
    panel = panel.drop_duplicates(['code', 'date']).sort_values(['code', 'date']).reset_index(drop=True)
    prior = panel.groupby('code', sort=False).close.shift(1)
    comparable = prior.notna() & np.isfinite(panel.pre_close) & (panel.pre_close > 0)
    mismatch = comparable & ((panel.pre_close-prior).abs() > np.maximum(0.01, prior.abs()*0.001))
    prior_admission = ROUND / 'r1_label_admission_v0_1_2.json'
    manifest.append(fingerprint(prior_admission))
    old = json.loads(prior_admission.read_text(encoding='utf-8'))
    known = {int(r['code']) for r in old['track_decisions'] if r['verdict'] == 'ADJ_EVIDENCE_CONFLICT'}
    scale = {'comparison': 'pre_close vs previous available daily close; tolerance max(0.01, abs(prev)*0.001)',
             'comparable_rows': int(comparable.sum()), 'mismatch_rows': int(mismatch.sum()),
             'mismatch_codes': int(panel.loc[mismatch, 'code'].nunique()),
             'old_conflict_codes': len(known),
             'old_conflict_codes_still_mismatching': sorted(str(c).zfill(6) for c in known & set(panel.loc[mismatch, 'code'])),
             'examples': panel.loc[mismatch, COLS[:2]+['close', 'pre_close']].head(20).to_dict('records'),
             'raw_scale_verdict': 'NOT_CERTIFIED: this comparison cannot distinguish adjustment/company actions/source holes; no RAW claim'}
    manifest.extend([fingerprint(CHAIN), fingerprint(SECTOR)])
    chains = json.loads(CHAIN.read_text(encoding='utf-8'))
    sector = pd.read_parquet(SECTOR)
    membership = {}
    for code, group in sector.groupby('symbol'):
        intervals = []
        for row in group.itertuples():
            start = int(str(row.effective_from).replace('-', '')[:8])
            end = int(str(row.effective_to).replace('-', '')[:8])
            intervals.append((start, end))
        membership[str(code).zfill(6)] = intervals
    calendar = np.sort(panel.date.unique())
    sources = [BASE/'v5_6_green/green_timing_dataset.parquet', BASE/'green_watch_v0/pool_daily_v11.parquet',
               ROUND/'r1_branch_at_decision_v0_1_2.json']
    candidates = []
    for path in sources:
        print(f'candidate audit: {path.name}', flush=True)
        candidates.append(candidate_report(path, panel, calendar, chains, membership, manifest))
    for rel in ['green_exit_v1/step6_rescue/r0_retrain.py', 'green_exit_v1/step6_rescue/common.py',
                'opportunity_v2/o5_sector_lag/etl_pit_membership.py', 'risk_r0_v1/R0_V1_DEVELOPMENT.md',
                'risk_r0_v1/P0_readonly_inventory_tasklist_v1.md']:
        manifest.append(fingerprint(BASE/rel))
    manifest.append(fingerprint(Path(__file__)))
    for item in manifest:
        if sha(Path(item['path'])) != item['sha256']:
            raise RuntimeError('Input mutated; reports not published')
    report = {'status': 'P0_AUDITED_WITH_BLOCKERS_NOT_TRAINING_READY',
              'generated_at': datetime.now(timezone(timedelta(hours=8))).isoformat(),
              'network_requests': 0, 'labels_created': False, 'models_fitted': False,
              'input_manifest': manifest, 'daily_files': file_reports,
              'daily_totals': {'files': len(file_reports), 'bytes': sum(r['bytes'] for r in manifest[:len(file_reports)]),
                               'physical_rows': sum(r['rows'] for r in file_reports),
                               'lookup_unique_rows': len(panel), 'observed_calendar_days': len(calendar),
                               'date_min': int(calendar.min()), 'date_max': int(calendar.max()),
                               'by_year_rows': {str(k): int(v) for k,v in (panel.date//10000).value_counts().sort_index().items()}},
              'duplicate_keys': duplicate_summary, 'scale_audit': scale, 'candidate_sources': candidates,
              'metadata': {'sector_source_rows': len(sector), 'sector_columns': list(sector.columns),
                           'sector_available_at': 'NOT_CERTIFIED: effective dates/update_time are not historical publication timestamps',
                           'listing_date': 'PENDING_SOURCE: absent from daily and selected candidate sources; first seen is not listing date',
                           'st': 'Daily 0/1 availability audited; capture-time/PIT certification remains pending',
                           'chain': 'OK/EMPTY/UNKNOWN reachability only; not corporate-action/window/RAW certification'},
              'old_r0': {'train_end': 20231220, 'dev_window': [20240101, 20241231],
                         'labels': 'X1 STOP / daily close loss3,loss5 / X1_RF failure / MAE thresholds; NOT new P_tail_5d',
                         'cost': 'gross_return - 0.0031; NOT current cash-ledger execution label',
                         'source_filter': 'mode_id=G0, further target joins; not adopted as R0-V1 population'},
              'blockers': ['P1 population/decision/ref/threshold contract not signed',
                           'RAW scale/duplicate conflict policy not certified',
                           'Exchange calendar/sector historical availability/listing provenance not certified',
                           'R0 resource contract before mother table/models not approved',
                           'Training/collection/shadow not authorized']}
    data = (json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)+'\n').encode('utf-8')
    markdown = ('# R0-V1 深度 P0 只读盘点 v2\n\n'
                '状态：P0_AUDITED_WITH_BLOCKERS_NOT_TRAINING_READY。零网络、未构建标签、未拟合。\n\n'
                f"日线：{len(file_reports)} 文件，{report['daily_totals']['physical_rows']:,} 行；实际日期 {calendar.min()}–{calendar.max()}。\n\n"
                f"重复键 {len(conflicting):,} 个，其中数值冲突 {int(conflicting.sum()):,} 个；冲突不修复，只在覆盖查询中隔离。\n\n"
                f"前收与前一条日线收盘差异：{int(mismatch.sum()):,} 行 / {scale['mismatch_codes']} 码；不是公司行为或 RAW 尺度判定。\n\n"
                '三条候选线独立统计，不混合分母；逐年覆盖见 JSON。next5 只证明本地观测日历行存在，不是标签有效。\n\n'
                '行业区间命中≠历史可知；上市日期缺证据；ST 列非空≠ PIT 已验；链缓存存在≠公司行为核销。\n\n'
                '建议先定日频价格观察版 P1；执行版仍需分钟/元数据/合法交易链验收。新人口与阈值未签，不产生 Y_tail。\n\n'
                '输出增量封顶 1 MiB，仅本地报告，不新增输入副本，不授予独立 128 MiB 资源额度。\n').encode('utf-8')
    if len(data)+len(markdown) > 1024*1024:
        raise RuntimeError('P0 report output cap exceeded; nothing published')
    outputs[0].write_bytes(data)
    outputs[1].write_bytes(markdown)
    print(json.dumps({'status': report['status'], 'daily': report['daily_totals'],
                      'duplicate_keys': len(conflicting), 'report_bytes': len(data)+len(markdown)}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
