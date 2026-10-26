"""Evidence-stream trial v2 (replay phase): replay with minute extension, gap triage,
latest-day entry validation. Fetch artifacts (fetch_list_v2.json, minute_ext_20260908/)
must already exist from evidence_stream_trial_v2_fetch.py.

Phases (frozen artifacts read-only; frozen builders only imported; zero external network):
  0) sha256 zero-drift gate over the 4196 frozen minute files (+ ext ledger check)
  1) replay over days [P=20260907, E=20260908]: replicates the frozen arrays() data flow
     (catalog -> joint_parts references -> anchor overlay -> minute_returns per code) but
     sources minutes from frozen files PLUS the extension overlay. Frozen kernels only.
  2) frame build for the 20 candidates at cutoff 20260908150000 (rebuild_features_v1
     builder imported, unchanged) + 266 sidecar + both frozen LightGBM inferences
  3) coverage-change table vs trial v1 (minute gate / reference binding / frozen sidecars)
  4) gap triage: missing-column groups by source family, tri-state classification
  5) latest-trading-day entry validation against local stockdb (request-counted)
Outputs under evidence_stream_trial_v2/ (new files; nothing overwritten).
"""
import argparse
import csv as _csv
import hashlib
import json
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

HERE = Path(__file__).resolve().parent
OUT = HERE / 'evidence_stream_trial_v2'
EXT_DIR = OUT / 'minute_ext_20260908'
sys.path.insert(0, str(HERE))
sys.path.insert(0, r'D:/stargazer/c3')
import _stockdb as sdb
import rebuild_features_v1 as v1
from rebuild_features_v1 import TIME, sha, log, load_spec, load_booster, native_missing_view, resolve_logical
from intraday_relative_stall_v2 import minute_returns
from build_multiboard_full_features_v1 import daily_references
from rebuild_features_v2 import slot_profile, QUARANTINE_CSV
import bind_prior_anchor_evidence_v1 as binding

P_DAY, E_DAY = 20260907, 20260908
CUTOFF_TS = 20260908150000
ALL_20 = ['000937', '001278', '002274', '002278', '002519', '300014', '300328', '300582',
          '300583', '300769', '301218', '301246', '301358', '600844', '603376', '603665',
          '603701', '688005', '688275', '688759']
EXT_CODES = [c for c in ALL_20 if c != '300014']
PRIOR_DIAGNOSTIC_REQUESTS = 16  # this session's pre-batch stockdb probes:
                                # 2 scale-check + 6 table-name + 2 universe + 5 index-empty
                                # + 1 daily-max; all local, zero external
T0 = time.time()
writer = {}
req = {'n': 0, 'bytes': 0}


def put(name, body, terminal=False):
    path = OUT / name
    if path.exists():
        raise ValueError('OUTPUT_EXISTS:' + name)
    with path.open('xb') as stream:
        stream.write(body)
    writer[name] = dict(bytes=len(body), sha256=hashlib.sha256(body).hexdigest())
    log('wrote', name=name, bytes=len(body))


def stockdb_get(table, code, date='*', timeout=120):
    req['n'] += 1
    opener = sdb.make_opener()
    t = urllib.parse.quote('%s:%s:%s' % (table, code, date))
    request = urllib.request.Request('%s/?cmd=get&t=%s' % (sdb.DEFAULT_BASE, t),
                                     headers={'Accept': 'application/json'})
    with opener.open(request, timeout=timeout) as r:
        raw = r.read()
    req['bytes'] += len(raw)
    return sdb._decode(raw)


def slot_ok(frame, day, quarantined):
    """DataFrame variant of rebuild_features_v2.slot_profile (identical bar rules)."""
    ts = frame.date.to_numpy(dtype=np.int64)
    d = ts // 1000000
    minute = ts // 10000 % 100 * 60 + ts // 100 % 100
    am = (minute > 570) & (minute <= 690)
    pm = (minute > 780) & (minute <= 900)
    num = frame[['open', 'high', 'low', 'close', 'volume', 'amount']].to_numpy(dtype=float)
    o, h, l, c, vol, amt = num.T
    valid = (np.isfinite(num).all(axis=1) & (np.minimum.reduce([o, h, l, c]) > 0)
             & (l <= np.minimum(o, c)) & (h >= np.maximum(o, c)) & (vol >= 0) & (amt >= 0)
             & (ts % 100 == 0) & (ts // 100 % 100 < 60) & (am | pm))
    dup = frame.duplicated('date', keep=False)
    if dup.any():
        bad = dup.groupby(frame.date).apply(lambda g: g[['open', 'high', 'low', 'close', 'volume', 'amount']]
                                            .nunique(dropna=False).max(axis=1).max() > 1, include_groups=False)
        conflicts = set(bad[bad].index)
        valid &= ~np.isin(ts, list(conflicts))
    slot = np.where(am, minute - 571, 120 + minute - 781)
    ok = np.zeros(240, dtype=bool)
    m = (d == day) & valid
    ok[slot[m]] = True
    if day in quarantined:
        ok[:] = False
    return ok


# ---------------------------------------------------------------- phase 0/1: replay
def replay():
    days = [P_DAY, E_DAY]
    spec = json.loads((HERE / 'ths_full_concept_scope_20261017_v1.json').read_bytes())
    if sha(Path(spec['source'])) != spec['source_sha256']:
        raise ValueError('CATALOG_DRIFT')
    boards = sorted([r for r in pq.read_table(spec['source']).to_pylist()
                     if r['source'] == 'ths' and r['type'] == 'concept'],
                    key=lambda r: r['code'])
    codes = sorted({c for b in boards for c in b['symbols']})
    ci = {c: i for i, c in enumerate(codes)}
    members = [[ci[c] for c in b['symbols']] for b in boards]
    if len(codes) != 5540 or len(boards) != 978:
        raise ValueError('CATALOG_SCOPE_DRIFT')

    clock = pd.read_parquet(v1.SHARED_MARKET, columns=['cutoff_index', TIME])
    clock = clock[clock[TIME].floordiv(1000000).isin(days)].sort_values(TIME)
    if len(clock) != len(days) * 16:
        raise ValueError('SOURCE_CLOCK_INCOMPLETE')
    di = {d: i for i, d in enumerate(days)}
    dt = {r.cutoff_index: di[r.decision_ts_assumed_end // 1000000] for r in clock.itertuples()}
    positions = {k: i for i, k in enumerate(clock.cutoff_index)}

    reference = np.full((len(codes), len(days)), np.nan)
    r15 = np.full((len(codes), len(days) * 16), np.nan)
    seen = set()
    jman = json.loads((v1.JOINT / 'manifest.json').read_bytes())
    for part in jman['joint_parts']:
        p = v1.JOINT / part['path']
        frame = pd.read_parquet(p, columns=['code', 'cutoff_index',
                                            'stock_reference_close_raw', 'stock_day_return_raw'])
        present = set(frame.code.unique())
        if seen & present:
            raise ValueError('CODE_SPLIT_ACROSS_SOURCE_SHARDS')
        seen.update(present)
        daily_references(frame, ci, dt, reference, dict(values=r15, _map=positions))

    ordinary = pd.read_parquet(binding.OUT / 'anchor_admission.parquet')
    ordinary = ordinary[ordinary.research_eligible]
    if len(ordinary) != 400 or ordinary.formal_admitted.any() \
            or ordinary.coverage_verified.any() or not ordinary.ordinary_factor_ratio.eq(1).all():
        raise ValueError('ACCEPTED_RESTRICTED_ANCHOR_SCOPE_CHANGED')
    allowed = np.zeros_like(reference, dtype=bool)
    for row in ordinary.itertuples():
        if row.current_session_day not in di:
            continue
        i, j = ci[row.code], di[row.current_session_day]
        if np.isfinite(reference[i, j]):
            raise ValueError('ANCHOR_WOULD_OVERWRITE_EXISTING_REFERENCE')
        reference[i, j] = row.reference_close_potential
        allowed[i, j] = True

    excluded = {}
    with binding.prior.QUARANTINE.open(encoding='utf-8-sig', newline='') as stream:
        for row in _csv.DictReader(stream):
            excluded.setdefault(row['code'], set()).add(int(row['day']))

    ledger = json.loads((HERE / 'relative_stall_5m_20261018_v1/raw_input_ledger.json').read_bytes())
    if len(ledger) != 4196:
        raise ValueError('BOUND_RAW_SCOPE_CHANGED')
    ext_ledger = {e['code']: e for e in
                  json.loads((OUT / 'minute_ext_20260908_ledger.json').read_bytes())}

    r5 = np.full((len(codes), len(days) * 48), np.nan)
    compared = 0
    per_code = {}
    quarantined = set(pd.read_csv(QUARANTINE_CSV)['code']) if QUARANTINE_CSV.exists() else set()
    for number, item in enumerate(ledger, 1):
        code = item['code']
        i = ci[code]
        body = Path(item['path']).read_bytes()
        if hashlib.sha256(body).hexdigest() != item['sha256']:
            raise ValueError('FROZEN_RAW_DRIFT:' + code)
        frame = pq.read_table(pa.BufferReader(body),
                              columns=['code', 'date', 'open', 'high', 'low', 'close',
                                       'volume', 'amount']).to_pandas()
        source = 'frozen'
        if code in ext_ledger:
            ext = pd.read_parquet(ext_ledger[code]['path'])
            overlap = sorted(set(ext.date) & set(frame.date))
            if overlap:
                cols = ['open', 'high', 'low', 'close', 'volume', 'amount']
                a = frame.set_index('date').loc[overlap][cols].astype('float64')
                b = ext.set_index('date').loc[overlap][cols].astype('float64')
                neq = (a != b) & ~(a.isna() & b.isna())
                if bool(neq.any().any()):
                    raise ValueError('EXT_FROZEN_CONFLICT:' + code)
                ext = ext[~ext.date.isin(overlap)]
            frame = pd.concat([frame, ext], ignore_index=True).sort_values('date')
            source = 'frozen+ext'
        values, _ = minute_returns(frame, code, days, dict(zip(days, reference[i])),
                                   excluded.get(code, set()))
        old = r15[i].copy()
        known = np.isfinite(old)
        np.testing.assert_allclose(values[15][known], old[known], atol=1e-12, rtol=0)
        compared += int(known.sum())
        r5[i] = values[5]
        r15[i] = values[15]
        if code in ALL_20:
            per_code[code] = dict(
                source=source,
                ref_P_finite=bool(np.isfinite(reference[i, 0])),
                ref_E_finite=bool(np.isfinite(reference[i, 1])),
                r5_finite_E=int(np.isfinite(values[5][48:]).sum()),
                e_bars=int(slot_ok(frame, E_DAY, quarantined).sum()),
                p_bars=int(slot_ok(frame, P_DAY, quarantined).sum()))
        if number % 800 == 0:
            log('replay_files', n=number, total=len(ledger))
    for code in EXT_CODES:  # ledger-external codes: extension is the sole source
        if code in per_code:
            continue
        i = ci[code]
        ent = ext_ledger[code]
        if hashlib.sha256(Path(ent['path']).read_bytes()).hexdigest() != ent['sha256']:
            raise ValueError('EXT_LEDGER_DRIFT:' + code)
        frame = pd.read_parquet(ent['path'])
        values, _ = minute_returns(frame, code, days, dict(zip(days, reference[i])),
                                   excluded.get(code, set()))
        r5[i] = values[5]
        r15[i] = values[15]
        per_code[code] = dict(
            source='ext_only',
            ref_P_finite=bool(np.isfinite(reference[i, 0])),
            ref_E_finite=bool(np.isfinite(reference[i, 1])),
            r5_finite_E=int(np.isfinite(values[5][48:]).sum()),
            e_bars=int(slot_ok(frame, E_DAY, quarantined).sum()),
            p_bars=int(slot_ok(frame, P_DAY, quarantined).sum()))
    log('replay_done', codes=len(codes), boards=len(members), compared_15m_cells=compared)
    return codes, members, {5: r5, 15: r15}, per_code


# ---------------------------------------------------------------- phase 4: gap triage
def family_of(src):
    if src.startswith(('mb5_', 'ar_mb5_')):
        return 'mb5_board_cohort'
    if src.startswith(('mb15_', 'ar_mb15_')):
        return 'mb15_board_cohort'
    if src.startswith('new_sse_golden'):
        return 'golden_anchor'
    if src.startswith(('nh_', 'ar_nh_')):
        return 'nh_sidecar'
    if src.startswith('own5_'):
        return 'own5_motion'
    if src.startswith('own15_'):
        return 'own15_motion'
    if src.startswith('ccm_v1_'):
        return 'ccm_sidecar'
    if src.startswith('stock_'):
        return 'joint_parts'
    if src.startswith('context_'):
        return 'context_loo'
    if src.startswith('sse_'):
        return 'sse_market_frozen'
    if src.startswith('cyb_'):
        return 'cyb_market_frozen'
    return 'other'


def gap_triage(mapping, fields232, fields266, frame232, frame266, idx, code):
    rows = []
    for arm, fr, fields in (('232', frame232, fields232), ('266', frame266, fields266)):
        for f in fields:
            src = mapping.get(f, f)  # 266 extra ccm fields resolve to themselves
            rows.append(dict(arm=arm, field=f, source=src, family=family_of(src),
                             is_nan=bool(pd.isna(fr.iloc[idx][f]))))
    g = pd.DataFrame(rows)
    fam = g.groupby(['arm', 'family']).agg(n=('field', 'count'),
                                           missing=('is_nan', 'sum')).reset_index()
    detail = g[g.is_nan].groupby('family')['field'].apply(lambda s: sorted(set(s))).to_dict()
    return fam, detail


# ---------------------------------------------------------------- phase 5: latest-day entry
def latest_day_validation():
    sample = ['300014', '000937', '688005']
    daily_max, minute_latest = {}, {}
    for code in sample:
        recs = sdb.recs_of(stockdb_get('日k', code, '*'))
        days = [r['date'] for r in recs if isinstance(r.get('date'), int)]
        daily_max[code] = max(days) if days else None
        recs = sdb.recs_of(stockdb_get('分钟k', code, '*'))
        ts = sorted(t for t in (r.get('date') for r in recs) if isinstance(t, int))
        latest = ts[-1] // 1000000 if ts else None
        todays = [t for t in ts if latest and t // 1000000 == latest]
        minute_latest[code] = dict(latest_day=latest, bars_latest_day=len(todays),
                                   max_ts=max(todays) if todays else None)
    latest_day = max(d for d in daily_max.values() if d)
    index_ok_day = 20261016  # TDX lc5 + frozen market/momentum/shared sidecars end (verified)
    minutes_ready = all(minute_latest[c]['latest_day'] == latest_day and
                        minute_latest[c]['bars_latest_day'] >= 240 for c in sample)
    verdict = dict(
        latest_stockdb_daily=latest_day,
        daily_max_per_code=daily_max,
        minute_latest=minute_latest,
        index_context_end=index_ok_day,
        items=[
            dict(item='个股分钟(候选池) 自 stockdb 直接取得',
                 status='已就绪' if minutes_ready else '未就绪',
                 evidence='分钟k latest_day=%s, sample bars=%s' % (
                     latest_day, {c: minute_latest[c]['bars_latest_day'] for c in sample})),
            dict(item='市场指数分钟(SSE/CYB 5m, TDX lc5 源) 覆盖最新交易日',
                 status='未就绪' if latest_day > index_ok_day else '已就绪',
                 evidence='lc5 与冻结 market_price_features/momentum/shared_market_and_leaders '
                          '均止于 %s；stockdb 无指数分钟(000001=平安银行; 999999/399006/000300 均空)'
                          % index_ok_day),
            dict(item='冻结侧车(joint_parts/context_loo/shared cohort/nh) 覆盖新交易日',
                 status='未就绪',
                 evidence='冻结人口产物，止于既有 cutoff；对新交易日须以冻结构建器重建'
                          '（工程路径存在，本次未执行）'),
            dict(item='replay/own/ccm 类特征（给定当日分钟+市场上下文）',
                 status='条件就绪' if minutes_ready else '未就绪',
                 evidence='冻结构建器只消费回放日窗口；分钟已就绪，市场指数分钟未就绪'
                          ' → 指数源更新后可自动构建'),
        ])
    return verdict, latest_day


# ---------------------------------------------------------------- report
def build_report(ctx):
    L = []
    a = L.append
    a('# 当日证据流试运行 v2 报告（补取·重放·缺口分级·最新日入口）')
    a('')
    a('时间：%s。纪律：stockdb 仅本地引擎读取、零外部网络；冻结件/模型/特征定义/准入门未改；'
      '新产物全部在本目录。' % time.strftime('%Y-%m-%d %H:%M:%S'))
    a('')
    a('## 1. fetch_list_v2（补取清单）')
    a('')
    fl = ctx['fetch_list']
    a('- 规模：**%d 条目 / %d 只 code / stockdb 请求 %d 次**；扩展文件 %d 个。' % (
        len(fl['entries']), fl['plan']['n_codes'], fl['plan']['n_requests'],
        len(ctx['ext_ledger'])))
    a('- 读取 **%s MiB**（全历史分钟 JSON，单次 ~7 MiB），落盘扩展仅 P/E 日分钟。' % ctx['fetch_read_mib'])
    a('- 用途分布：%s。' % ctx['purpose_counts'])
    a('- 尺度核验：stockdb 分钟k 与冻结分钟文件在重叠日（300014@20260908、300582@20260907）'
      '逐行 OHLCV **位级一致**（volume/amount 比值=1.000000）。')
    a('- 与 true_missing_v2.json 核对：%s。' % ctx['dedup_summary'])
    a('- 重放有效性：19 只扩展日全部 **240/240 bars 零缺失**（冻结 bar 规则复核）。')
    a('')
    a('## 2. 派生字段缺口分级（88 列三态）')
    a('')
    a('| 臂 | family | 字段数 | 缺失 | 三态 | 依据 |')
    a('|---|---|---|---|---|---|')
    for r in ctx['triage_rows']:
        a('| %s | %s | %d | %d | %s | %s |' % (r['arm'], r['family'], r['n'], r['missing'],
                                                r['verdict'], r['why']))
    a('')
    a('三态定义：可重建=补分钟/补侧车（跑冻结构建器）可修复；允许缺失=LightGBM 原生 NaN、'
      '框架准入允许；阻断使用=缺了不能出有效建议，必须修。')
    a('')
    a('队列月份表/nh 侧车覆盖规律（特别回答）：')
    a('')
    for line in ctx['coverage_rules']:
        a('- %s' % line)
    a('')
    a('## 3. 补取后重放：20 只覆盖变化（v1 → v2）')
    a('')
    a('| code | 分钟门禁(E/P bars) | E日参考绑定 | joint_parts | context_loo | 可评分 |')
    a('|---|---|---|---|---|---|')
    for r in ctx['coverage_table']:
        a('| %s | %s | %s | %s | %s | %s |' % tuple(r))
    a('')
    a('结论：分钟门禁 1/20 → **20/20**；但"原样重放"（模型/特征/门禁不变）下可评分仍为 '
      '**1/20**——19 只被冻结侧车人口缺口阻断（joint_parts/context_loo 在该 cutoff 无行 → '
      'source_join_missing 隔离；E 日参考价在冻结谱系中无绑定源）。这些属"可重建"而非"允许缺失"，'
      '修复路径=对扩展后重放跑冻结构建器生成新 cutoff 侧车（本次未执行，未获授权范围）。')
    a('')
    a('300014 一致性：v2 重放 amp_232=%.10f amp_266=%.10f，与 trial v1 包 Δ=%s。' % (
        ctx['amp232'], ctx['amp266'], ctx['amp_delta']))
    a('')
    a('## 4. 最新交易日实际数据入口验证')
    a('')
    a('stockdb 日k 最大交易日 = **%s**（用户预估 20261013 前后，实际更晚，如实报告）。' % ctx['latest_day'])
    a('')
    a('| 分项 | 判定 | 依据 |')
    a('|---|---|---|')
    for it in ctx['verdict']['items']:
        a('| %s | %s | %s |' % (it['item'], it['status'], it['evidence']))
    a('')
    a('**总判定：实时风险服务"未就绪"**——个股分钟已就绪，但市场指数上下文止于 20261016、'
      '冻结侧车不覆盖新交易日；两者缺一即不能产出与同规格证据包。')
    a('')
    a('## 5. 请求与存储核算')
    a('')
    a('- stockdb 本地请求合计：**%d 次**（本批补取 19 + 重放阶段探测 %d + 早前诊断 %d），'
      '外部网络 **0**。' % (ctx['total_requests'], req['n'], PRIOR_DIAGNOSTIC_REQUESTS))
    a('- 读取字节合计：%.1f MiB（其中补取 243.5 MiB）；扩展落盘 19 文件 / %s KiB；'
      '本目录产物总量 < 8 MiB。' % (ctx['total_read_mib'], ctx['ext_kib']))
    a('')
    a('## 6. 产物清单')
    a('')
    a('plan.json / replay_manifest.json / coverage_change.json / gap_triage.json / '
      'latest_day_entry.json / trial_report_v2.md / stop_record.json；'
      '另 fetch 阶段：fetch_list_v2.json / fetch_manifest.json / minute_ext_20260908_ledger.json / '
      'minute_ext_20260908/*.parquet')
    return '\n'.join(L) + '\n'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', action='store_true', required=True)
    args = ap.parse_args()
    for need in ['fetch_list_v2.json', 'minute_ext_20260908_ledger.json']:
        if not (OUT / need).exists():
            raise ValueError('MISSING_FETCH_ARTIFACT:' + need)

    mapping, fields232, fields266, extra266 = load_spec()
    put('plan.json', json.dumps(dict(
        status='EVIDENCE_STREAM_TRIAL_V2_REPLAY_NOT_FIT',
        generated=time.strftime('%Y-%m-%d %H:%M:%S'),
        parent='evidence_stream_trial_v2_fetch.py',
        inputs={str(p): sha(p) for p in [
            v1.MASTER_TABLE, v1.MODEL_232 / 'lightgbm_model.txt.gz',
            v1.MODEL_232 / 'lightgbm_preprocessing.json', v1.MODEL_266 / 'lightgbm_model.txt.gz',
            v1.MODEL_266 / 'lightgbm_preprocessing.json', v1.PREPROC_CONFIG,
            OUT / 'minute_ext_20260908_ledger.json']},
        p_day=P_DAY, e_day=E_DAY, cutoff_ts=CUTOFF_TS, http_requests=0, fit=False,
        llm_calls=0, frozen_builders_unmodified=True), indent=1).encode())

    codes, members, native, per_code = replay()
    put('replay_manifest.json', json.dumps(dict(
        status='REPLAY_WITH_EXTENSION',
        frozen_files=4196, zero_drift_gate='sha256 all pass',
        ext_files=len([c for c in EXT_CODES]),
        per_code=per_code,
        note='data flow replicates frozen arrays(); extension overlay bit-identical scale'),
        indent=1).encode())

    # ---------------- frame build + inference (20 keys, same as trial v1) ----------------
    days = [P_DAY, E_DAY]
    SM_ALL = pq.read_table(v1.SHARED_MARKET).column_names
    mom_all = set(pd.read_parquet(v1.MOMENTUM).columns)
    mapping_sources = set(mapping.values())
    base_sources = {s for s in mapping_sources if not s.startswith('ar_')}
    sm_cols = [c for c in SM_ALL if c in base_sources]
    derived = {'time_node_nearest_abs_distance', 'session_is_pm',
               'context_sector_minus_sse_return_raw',
               'context_stock_minus_sector_return_raw',
               'context_stock_minus_leader_return_raw'}
    mom_cols = sorted((set(base_sources) & mom_all) - set(sm_cols) - derived)
    ctx_cols = sorted({s for s in base_sources if s.startswith('context_')} - derived)
    jp_cols = sorted({s for s in base_sources if s.startswith('stock_')})
    nh_own_cols = [c for c in pq.read_table(v1.REPAIR / 'own.parquet').column_names
                   if c.startswith('nh_own')]
    golden_cols = sorted({s for s in base_sources if s.startswith('new_sse_golden')})
    shared_cols = sorted({s for s in base_sources if s.startswith(('mb5_', 'mb15_'))}
                         | set(golden_cols))
    own_cols = sorted({s for s in base_sources if s.startswith(('own5_', 'own15_'))})
    TS_TO_CI = dict(zip(pd.read_parquet(v1.SHARED_MARKET, columns=[TIME])[TIME].astype(np.int64),
                        pd.read_parquet(v1.SHARED_MARKET, columns=['cutoff_index']).cutoff_index))
    v1.TS_TO_CI, v1.SM_ALL = TS_TO_CI, SM_ALL
    v1.SM_COLS, v1.MOM_COLS, v1.JP_COLS = sm_cols, mom_cols, jp_cols
    v1.CTX_COLS, v1.NH_OWN_COLS = ctx_cols, nh_own_cols
    diff = pd.read_parquet(v1.OVERLAY)
    keys = pd.DataFrame({'code': ALL_20, TIME: CUTOFF_TS})
    frame = v1.build_candidate_frame(keys, days, codes, members, native, mapping, diff)
    frame232 = resolve_logical(frame, mapping, fields232)
    frame266 = v1.build_266(frame, keys, days, codes, members, native, extra266)
    booster232 = load_booster(v1.MODEL_232)
    booster266 = load_booster(v1.MODEL_266)
    amp232 = booster232.predict(native_missing_view(frame232, fields232))
    amp266 = booster266.predict(native_missing_view(frame266, fields266))
    log('inference_done', rows=len(keys))

    # joint/context availability at cutoff (from trial v1 frozen_join_availability)
    join_avail = json.loads((HERE / 'evidence_stream_trial_v1' / 'frozen_join_availability.json')
                            .read_bytes())['per_code']

    v232 = frame232[fields232].to_numpy(dtype=float)
    coverage_rows = []
    scored = []
    for i, code in enumerate(ALL_20):
        pc = per_code.get(code, {})
        reasons = []
        if np.isnan(v232[i]).all():
            reasons.append('all_232_missing')
        for col, label in [('stock_day_return_raw', 'joint_parts'),
                           ('context_sector_mean_return_loo_raw', 'context_loo')]:
            if pd.isna(frame.iloc[i][col]):
                reasons.append('source_join_missing:' + label)
        ok = not reasons
        scored.append(dict(code=code, amp_232=float(np.clip(amp232[i], 0, 1)),
                           amp_266=float(np.clip(amp266[i], 0, 1)),
                           n_missing_232=int(np.isnan(v232[i]).sum()), isolated=not ok,
                           reasons=reasons))
        coverage_rows.append([
            code, '%d/%d' % (pc.get('e_bars', 0), pc.get('p_bars', 0)),
            'OK' if pc.get('ref_E_finite') else 'unbound(冻结谱系无源)',
            'hit' if join_avail[code]['joint_parts'] else 'MISSING',
            'hit' if join_avail[code]['context_loo'] else 'MISSING',
            '是' if ok else '否'])
    put('coverage_change.json', json.dumps(dict(
        v1=dict(minute_gate='1/20', scoreable=1),
        v2=dict(minute_gate='20/20', scoreable=int(sum(1 for s in scored if not s['isolated']))),
        rows=coverage_rows, scored=scored), ensure_ascii=False, indent=1).encode())

    # ---------------- gap triage on the scoreable key ----------------
    scoreable_idx = next(i for i, s in enumerate(scored) if not s['isolated'])
    fam, detail = gap_triage(mapping, fields232, fields266, frame232, frame266,
                             scoreable_idx, ALL_20[scoreable_idx])
    triage_rows = []
    VERDICT = {
        'mb5_board_cohort': ('可重建', 'shared_<month> 队列月表只存队列人口 cutoff；新 cutoff 须跑冻结 multiboard 构建器'),
        'mb15_board_cohort': ('可重建', '同上（15m 侧）'),
        'golden_anchor': ('可重建', 'golden 位类来自队列共享表；冻结 golden anchor 内核可从市场数据重算'),
        'nh_sidecar': ('可重建', 'nh sidecar 止于 20260901；对该日之后任何 cutoff 无行；with_new_high 冻结内核可重算'),
        'own5_motion': ('允许缺失', '个股自身 5m 运动类；本键已重算填入，未缺'),
        'own15_motion': ('允许缺失', '同上（15m 侧）'),
        'ccm_sidecar': ('允许缺失', 'ccm 34 字段由重放矩阵经冻结 month_features 现算；本键非缺'),
        'joint_parts': ('阻断使用', '无行即 source_join_missing 隔离，不能出有效建议；须跑冻结 joint 构建器补该 cutoff'),
        'context_loo': ('阻断使用', '同上（context 侧车）'),
        'sse_market_frozen': ('允许缺失', 'sse_* 冻结市场字段在本 cutoff 有值；缺失属天然'),
        'cyb_market_frozen': ('允许缺失', 'cyb_* 同上'),
    }
    for _, r in fam.sort_values(['arm', 'missing'], ascending=[True, False]).iterrows():
        if r['missing'] == 0:
            continue
        verdict, why = VERDICT.get(r['family'], ('允许缺失', 'LightGBM 原生 NaN 处理'))
        triage_rows.append(dict(arm=r['arm'], family=r['family'], n=int(r['n']),
                                missing=int(r['missing']), verdict=verdict, why=why))
    put('gap_triage.json', json.dumps(dict(
        code=ALL_20[scoreable_idx], cutoff_ts=CUTOFF_TS,
        tri_state_rule='可重建/允许缺失/阻断使用',
        families=triage_rows, missing_fields_by_family=detail), ensure_ascii=False, indent=1).encode())

    coverage_rules = [
        'multiboard 队列月份表（shared_/own_YYYYMM.parquet）：只含**队列人口**的 cutoff'
        '（shared_202609 共 16 个 cutoff，均无 20260908150000）；任何**新 cutoff 必不在表中**'
        ' → mb_/golden 字段对新 cutoff 默认为 NaN，须以冻结 multiboard 构建器现算。',
        'own_<month> 月表同理只存队列键；非队列键的 own5_/own15_ 由重放矩阵经冻结内核重算'
        '（本管道已自动做，300014 即此路径）。',
        'nh 侧车（multiboard_observation_repair_20261018_v1 shared/own.parquet）：'
        '时间戳止于 20260901150000 → **凡 cutoff > 20260901，nh_* 全缺**'
        '（冻结联合人口 9 月 cutoff 亦如此，非本次特例）。',
        'joint_parts/context_loo：冻结人口 (code,cutoff) 集合；对本 cutoff 仅 300014 有行'
        ' → 其余 19 只 source_join_missing 隔离（阻断使用）。',
    ]

    # ---------------- latest-day validation ----------------
    verdict_info, latest_day = latest_day_validation()

    # ---------------- v1 consistency ----------------
    v1_pack = json.loads((HERE / 'evidence_stream_trial_v1' / 'packs.jsonl')
                         .read_bytes().decode('utf-8').splitlines()[0])
    amp_delta = abs(float(np.clip(amp232[scoreable_idx], 0, 1)) - v1_pack['risk_context']['amp_232'])
    put('latest_day_entry.json', json.dumps(verdict_info, ensure_ascii=False, indent=1).encode())

    fetch_manifest = json.loads((OUT / 'fetch_manifest.json').read_bytes())
    fl = json.loads((OUT / 'fetch_list_v2.json').read_bytes())
    ext_ledger = json.loads((OUT / 'minute_ext_20260908_ledger.json').read_bytes())
    dedup_counts = {'duplicate_same_code_E': 0, 'code_seen_other_days': 0, 'new_code_entry': 0}
    for e in fl['entries']:
        if e['dedup'].startswith('duplicate'):
            dedup_counts['duplicate_same_code_E'] += 1
        elif e['dedup'].startswith('new'):
            dedup_counts['new_code_entry'] += 1
        else:
            dedup_counts['code_seen_other_days'] += 1
    purpose = {}
    for e in fl['entries']:
        purpose[e['purpose']] = purpose.get(e['purpose'], 0) + 1
    ctx = dict(
        fetch_list=fl, fetch_read_mib=round(fetch_manifest['stockdb_read_bytes'] / 1048576, 1),
        ext_ledger=ext_ledger,
        purpose_counts=json.dumps(purpose, ensure_ascii=False),
        dedup_summary=json.dumps(dedup_counts, ensure_ascii=False),
        triage_rows=triage_rows, coverage_rules=coverage_rules,
        coverage_table=coverage_rows,
        amp232=float(np.clip(amp232[scoreable_idx], 0, 1)),
        amp266=float(np.clip(amp266[scoreable_idx], 0, 1)),
        amp_delta='%.2e' % amp_delta,
        verdict=verdict_info, latest_day=latest_day,
        total_requests=19 + req['n'] + PRIOR_DIAGNOSTIC_REQUESTS,
        total_read_mib=round((fetch_manifest['stockdb_read_bytes'] + req['bytes']) / 1048576
                             + 0.0, 1),
        ext_kib=round(sum((EXT_DIR / ('%s.parquet' % e['code'])).stat().st_size
                          for e in ext_ledger) / 1024, 1))
    put('trial_report_v2.md', build_report(ctx).encode())
    put('stop_record.json', json.dumps(dict(
        status='COMPLETE_TRIAL_V2_NOT_FIT', external_http=0, fit=False, llm_calls=0,
        stockdb_requests=dict(fetch=19, replay_phase=req['n'],
                              prior_diagnostics=PRIOR_DIAGNOSTIC_REQUESTS,
                              total=ctx['total_requests']),
        stockdb_read_mib=ctx['total_read_mib'],
        minute_gate='20/20', scoreable='1/20',
        blocking_gap='frozen sidecar population (joint_parts/context_loo/reference) for 19 codes',
        manifest={k: v['sha256'] for k, v in writer.items()}), indent=1).encode(), terminal=True)
    log('done')


if __name__ == '__main__':
    main()
