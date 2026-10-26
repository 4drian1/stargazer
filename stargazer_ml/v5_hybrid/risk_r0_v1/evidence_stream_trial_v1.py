"""Same-day evidence-stream trial: P=20260907 candidates -> scorable evidence packs at E=20260908 15:00.

Thin orchestrator (new file). Frozen artifacts read-only; frozen builders only imported.
Zero network, zero fits, zero LLM calls.
Pipeline:
  1) admission=OK candidates for P=20260907 from frozen full_master_table_v2.parquet
  2) per-code frozen-minute coverage for E=20260908 (full 240 bars through the 15:00 cutoff)
     + P=20260907 full session, via rebuild_features_v2.slot_profile (frozen bar rules);
     codes without coverage are left out with per-record reasons
  3) deterministic replay (frozen arrays()) over [P, E]; rebuild 232/266 features at the
     15:00 cutoff via rebuild_features_v1 frame builder (frozen attach/derive only)
  4) inference with both frozen LightGBM boosters
  5) assemble EvidencePack-shaped JSON per successful candidate, fields aligned with
     stargazer_desktop/oracles_v1.py (EvidencePack / tarot_candidate / pyxis_context);
     downside_state/trigger explicitly null (no intraday replay in this trial);
     market/board/leader states from the frozen replay matrices (shared_market_and_leaders,
     context LOO sidecar) with real values, honestly missing where absent
Outputs: evidence_stream_trial_v1/ (packs.jsonl, trial_report.md, plan/stop/coverage json)
"""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[2] / 'stargazer_desktop'))

import rebuild_features_v1 as v1
from rebuild_features_v1 import (TIME, sha, log, load_spec, load_booster, native_missing_view,
                                 resolve_logical)
from rebuild_features_v2 import slot_profile, missing_intervals, LEDGER_BY_CODE, QUARANTINE_CSV
from oracles_v1 import EvidencePack, tarot_candidate, pyxis_context, PYXIS_FROZEN
import jev_adapter_v1 as jev

OUT = HERE / 'evidence_stream_trial_v1'
P_DAY, E_DAY = 20260907, 20260908
CUTOFF_TS = 20260908150000        # E-day final completed 15m grid cutoff
ASOF = '20260908150000'
MARKET_DATE = '20260908'
MASTER = v1.MASTER_TABLE
PB2_TABLE = MASTER.parent / 'rolling_lr_v2_predictions.parquet'
SEG_EXTENSION = 'EXTENSION_RESEARCH'
T0 = time.time()

writer = {}


def put(name, body, terminal=False):
    path = OUT / name
    if path.exists():
        raise ValueError('OUTPUT_EXISTS:' + name)
    with path.open('xb') as stream:
        stream.write(body)
    writer[name] = dict(bytes=len(body), sha256=hashlib.sha256(body).hexdigest())


def finite_or_none(x):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if np.isfinite(v) else None


def sign_label(x):
    v = finite_or_none(x)
    if v is None:
        return None
    return 'up' if v > 0 else ('down' if v < 0 else 'flat')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', action='store_true', required=True)
    args = ap.parse_args()
    if OUT.exists():
        raise ValueError('OUTPUT_EXISTS_NO_RESTART')
    OUT.mkdir()

    # ---------------- 1) candidates ----------------
    sel = pd.read_parquet(MASTER)
    sel['code'] = sel.code.astype(str).str.zfill(6)
    cand = sel[(sel.P == P_DAY) & (sel.admission == 'OK')].copy()
    cand = cand.sort_values('code').reset_index(drop=True)
    log('candidates_loaded', n=len(cand), p=P_DAY)
    put('candidates_p20260907.json', json.dumps(dict(
        p=P_DAY, e=E_DAY, admission='OK', n=len(cand),
        codes=cand.code.tolist(),
        master_fields_nan=cand[cand.F618.isna()].code.tolist()), indent=1).encode())

    pb = pd.read_parquet(PB2_TABLE)
    pb['code'] = pb.code.astype(str).str.zfill(6)
    pb = pb[pb.fold == 'FINAL']
    cand = cand.merge(pb[['code', 'P', 'pB2']], on=['code', 'P'], how='left', validate='one_to_one')
    log('pb2_bound', hits=int(cand.pB2.notna().sum()), of=len(cand))

    # ---------------- 2) coverage ----------------
    quarantined = set(pd.read_csv(QUARANTINE_CSV)['code']) if QUARANTINE_CSV.exists() else set()
    coverage = []
    for _, row in cand.iterrows():
        code = row.code
        ent = LEDGER_BY_CODE.get(code)
        rec = dict(code=code, P=P_DAY, E=E_DAY, cutoff_ts=CUTOFF_TS)
        if ent is None:
            rec.update(reason='no_minute_file', detail='code absent from frozen minute ledger')
            coverage.append(rec)
            continue
        try:
            ok_e = slot_profile(ent['path'], E_DAY, quarantined)
            ok_p = slot_profile(ent['path'], P_DAY, quarantined)
        except Exception as ex:  # unreadable frozen file: honest reason, no retry
            rec.update(reason='minute_file_unreadable', detail=type(ex).__name__ + ':' + str(ex)[:120])
            coverage.append(rec)
            continue
        e_missing = np.flatnonzero(~ok_e).tolist()
        p_missing = np.flatnonzero(~ok_p).tolist()
        rec.update(E_bars=int(ok_e.sum()), P_bars=int(ok_p.sum()),
                   E_missing_intervals=missing_intervals(e_missing),
                   P_missing_intervals=missing_intervals(p_missing),
                   file_max_day=int(pd.read_parquet(ent['path'], columns=['date']).date.max() // 1000000)
                   if ok_e.any() or ok_p.any() else None)
        if len(e_missing):
            rec.update(reason='e_day_incomplete_bars',
                       detail='missing %d/240 bars through 15:00 cutoff' % len(e_missing))
        elif len(p_missing):
            rec.update(reason='prev_day_incomplete_bars',
                       detail='P day missing %d/240 bars' % len(p_missing))
        else:
            rec.update(reason='OK')
        coverage.append(rec)
    cov = pd.DataFrame(coverage)
    n_ok = int((cov.reason == 'OK').sum())
    log('coverage_assessed', n=len(cov), ok=n_ok,
        reasons=cov.reason.value_counts().to_dict())
    survivors = cov.loc[cov.reason == 'OK', 'code'].tolist()
    put('coverage_trial.json', json.dumps(coverage, indent=1).encode())

    # frozen-sidecar join availability at the cutoff (informational for all 20)
    sm_map = pd.read_parquet(v1.SHARED_MARKET, columns=[TIME, 'cutoff_index'])
    ci = int(sm_map.loc[sm_map[TIME] == CUTOFF_TS, 'cutoff_index'].iloc[0])
    join_hits = {}
    jman = json.loads((v1.JOINT / 'manifest.json').read_bytes())
    jp_codes = set()
    for item in jman['joint_parts']:
        t = pq.read_table(v1.JOINT / item['path'], columns=['code'],
                          filters=[('cutoff_index', '==', ci)]).to_pandas()
        jp_codes |= set(t.code.astype(str).str.zfill(6))
    cman = json.loads((v1.CONTEXT_LOO / 'manifest.json').read_bytes())
    ctx_codes = set()
    for item in cman['parts']:
        t = pq.read_table(v1.CONTEXT_LOO / item['context'], columns=['code'],
                          filters=[('cutoff_index', '==', ci)]).to_pandas()
        ctx_codes |= set(t.code.astype(str).str.zfill(6))
    for code in cand.code:
        join_hits[code] = dict(joint_parts=code in jp_codes, context_loo=code in ctx_codes,
                               shared_market_and_leaders=True)
    put('frozen_join_availability.json', json.dumps(dict(
        cutoff_ts=CUTOFF_TS, cutoff_index=ci, per_code=join_hits), indent=1).encode())

    # ---------------- plan ----------------
    mapping, fields232, fields266, extra266 = load_spec()
    put('plan.json', json.dumps(dict(
        status='EVIDENCE_STREAM_TRIAL_NOT_FIT', generated=time.strftime('%Y-%m-%d %H:%M:%S'),
        inputs={str(p): sha(p) for p in [
            MASTER, PB2_TABLE, v1.MODEL_232 / 'lightgbm_model.txt.gz',
            v1.MODEL_232 / 'lightgbm_preprocessing.json', v1.MODEL_266 / 'lightgbm_model.txt.gz',
            v1.MODEL_266 / 'lightgbm_preprocessing.json', v1.PREPROC_CONFIG,
            v1.MULTIBOARD / 'feature_spec.json', v1.MULTIBOARD / 'report.json',
            v1.REPAIR / 'shared.parquet', v1.REPAIR / 'own.parquet', v1.OVERLAY,
            v1.SHARED_MARKET, v1.MOMENTUM, v1.LEDGER, v1.MARKET_FEATURES]},
        orchestrator='evidence_stream_trial_v1.py (frozen builders imported read-only; '
                     'reuses rebuild_features_v1/v2 kernels)',
        p_day=P_DAY, e_day=E_DAY, cutoff_ts=CUTOFF_TS,
        coverage_rule='E day full 240 valid bars through 15:00 cutoff + P day full session '
                      '(rebuild_features_v2.slot_profile frozen minute-bar rules; quarantine applied)',
        evidence_pack_contract='stargazer_desktop/oracles_v1.py EvidencePack/tarot_candidate/'
                               'pyxis_context (imported and instantiated, not reimplemented)',
        downstream_triggers_null=True, http_requests=0, fit=False, llm_calls=0), indent=1).encode())

    # ---------------- 3/4) replay + rebuild + inference (survivors only) ----------------
    result_rows = []
    packs = []
    if survivors:
        days = [P_DAY, E_DAY]
        codes, members, native = v1.replay(days)

        # global column-resolution setup identical to rebuild_features_v1.main();
        # assigned on the v1 MODULE because its frame builder reads module globals
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
        TS_TO_CI = dict(zip(pd.read_parquet(v1.SHARED_MARKET, columns=[TIME])[TIME]
                            .astype(np.int64),
                            pd.read_parquet(v1.SHARED_MARKET, columns=['cutoff_index'])
                            .cutoff_index))
        v1.TS_TO_CI, v1.SM_ALL = TS_TO_CI, SM_ALL
        v1.SM_COLS, v1.MOM_COLS, v1.JP_COLS = sm_cols, mom_cols, jp_cols
        v1.CTX_COLS, v1.NH_OWN_COLS = ctx_cols, nh_own_cols

        diff = pd.read_parquet(v1.OVERLAY)
        keys = pd.DataFrame({'code': survivors, TIME: CUTOFF_TS})
        frame = v1.build_candidate_frame(keys, days, codes, members, native, mapping, diff)
        frame232 = resolve_logical(frame, mapping, fields232)
        frame266 = v1.build_266(frame, keys, days, codes, members, native, extra266)

        booster232 = load_booster(v1.MODEL_232)
        booster266 = load_booster(v1.MODEL_266)
        amp232_raw = booster232.predict(native_missing_view(frame232, fields232))
        amp266_raw = booster266.predict(native_missing_view(frame266, fields266))

        v232 = frame232[fields232].to_numpy(dtype=float)
        v266 = frame266[fields266].to_numpy(dtype=float)
        sm_row = pd.read_parquet(v1.SHARED_MARKET,
                                 filters=[('cutoff_index', '==', ci)]).iloc[0]
        ctx_row = None
        for item in cman['parts']:
            t = pq.read_table(v1.CONTEXT_LOO / item['context'],
                              filters=[('cutoff_index', '==', ci)]).to_pandas()
            t['code'] = t.code.astype(str).str.zfill(6)
            hit = t[t.code.isin(survivors)]
            if len(hit):
                ctx_row = hit.iloc[0]
                break

        for i, code in enumerate(survivors):
            n_missing_232 = int(np.isnan(v232[i]).sum())
            n_missing_266 = int(np.isnan(v266[i]).sum())
            reasons = []
            if np.isnan(v232[i]).all():
                reasons.append('all_232_missing')
            for col, label in [('stock_day_return_raw', 'joint_parts'),
                               ('context_sector_mean_return_loo_raw', 'context_loo')]:
                if col in frame and pd.isna(frame.iloc[i][col]):
                    reasons.append('source_join_missing:' + label)
            mrow = cand[cand.code == code].iloc[0]
            result_rows.append(dict(
                code=code, P=P_DAY, E=E_DAY, cutoff_ts=CUTOFF_TS,
                amp_232_raw=float(amp232_raw[i]), amp_266_raw=float(amp266_raw[i]),
                amp_232=float(np.clip(amp232_raw[i], 0, 1)),
                amp_266=float(np.clip(amp266_raw[i], 0, 1)),
                n_missing_232=n_missing_232, n_missing_266=n_missing_266,
                isolated=bool(reasons), isolation_reason=';'.join(reasons),
                pB2=finite_or_none(mrow.pB2)))
            if reasons:
                continue

            # ---- risk context groups from the frozen replay matrices ----
            market_state = dict(
                index='sse', session_fraction=finite_or_none(sm_row.sse_15m_session_fraction),
                trend=sign_label(sm_row.sse_15m_day_return_change_adjacent),
                state='day_gain' if finite_or_none(sm_row.sse_15m_day_return_prev_close) and
                     sm_row.sse_15m_day_return_prev_close > 0 else 'day_loss',
                direction=sign_label(sm_row.sse_15m_momentum_change),
                strength=finite_or_none(sm_row.sse_15m_rsi14),
                drawdown_from_day_high=finite_or_none(sm_row.sse_15m_drawdown_from_observed_day_high),
                anchor_position=finite_or_none(sm_row.sse_15m_anchor_position),
                daily_return5_prior=finite_or_none(sm_row.sse_15m_daily_return5_asof_prior_day),
                breadth_observed_stocks=int(sm_row.captured_stock_count_today)
                if finite_or_none(sm_row.captured_stock_count_today) is not None else None,
                cyb_day_return_prev_close=finite_or_none(sm_row.cyb_15m_day_return_prev_close))
            board_states = []
            if ctx_row is not None:
                board_states.append(dict(
                    board='own_sector_loo',
                    strength=finite_or_none(ctx_row.context_sector_mean_return_loo_raw),
                    breadth=finite_or_none(ctx_row.context_sector_breadth_loo),
                    coverage=finite_or_none(ctx_row.context_sector_mean_coverage_loo),
                    direction=sign_label(ctx_row.context_sector_common_change_loo_raw),
                    state='positive' if finite_or_none(ctx_row.context_sector_mean_return_loo_raw)
                          is not None and ctx_row.context_sector_mean_return_loo_raw > 0
                          else 'nonpositive',
                    current_leader_return=finite_or_none(ctx_row.context_current_leader_return_loo_raw),
                    current_leader_rsi14_15m=finite_or_none(ctx_row.context_current_leader_rsi14_15m_raw)))
            leader_states = []
            if finite_or_none(sm_row.leader_observed_stock_return_raw) is not None:
                leader_states.append(dict(
                    code=str(sm_row.leader_observed_stock),
                    board=str(sm_row.leader_observed_board),
                    strength=finite_or_none(sm_row.leader_observed_stock_return_raw),
                    state='up' if sm_row.leader_observed_stock_return_raw > 0 else 'down',
                    full_universe_observed=bool(sm_row.leader_full_universe_observed)
                    if sm_row.leader_full_universe_observed is not None else None))

            risk_missing = []
            for grp, obj in (('market_state', market_state),
                             ('board_states', board_states),
                             ('leader_states', leader_states)):
                if not jev._group_has_state(grp, obj):
                    risk_missing.append(grp)

            risk_context = pyxis_context(
                version='232主版',
                amp_232=float(np.clip(amp232_raw[i], 0, 1)),
                amp_266=float(np.clip(amp266_raw[i], 0, 1)),
                market_state=market_state, board_states=board_states,
                leader_states=leader_states, missing=risk_missing, asof_ts=ASOF)
            risk_context['n_missing_232'] = n_missing_232
            risk_context['n_missing_266'] = n_missing_266
            risk_context['feature_gap_note'] = (
                'cohort shared/own month tables and nh sidecars have no row at this cutoff '
                '(multiboard cohort cutoffs end before / nh sidecars end 20260901); '
                'own5_/own15_ recomputed from frozen replay; mb_/golden/nh_* fields NaN')

            cand_dict = tarot_candidate(
                code,
                p_close_score=finite_or_none(mrow.pB2),
                golden_zone=finite_or_none(mrow.golden_dist_618_atr),
                f618=finite_or_none(mrow.F618),
                plan_dist_atr=finite_or_none(mrow.plan_dist_atr),
                atr_ref=finite_or_none(mrow.atr_ref),
                downside_state=None,      # no intraday replay in this trial -> explicit null
                trigger=None,             # ditto
                evidence_refs=dict(
                    master_table='full_master_table_v2.parquet(code,P)',
                    pB2='rolling_lr_v2_predictions.parquet(fold=FINAL, code,P)',
                    amp='rebuild via evidence_stream_trial_v1.py @ cutoff 20260908150000',
                    market_board_leader='shared_market_and_leaders + context_loo @ cutoff_index '
                                        + str(ci)))
            pack = EvidencePack(
                evidence_id='trial_%s_%d' % (code, E_DAY), asof_ts=ASOF,
                market_date=MARKET_DATE, query_ts=ASOF,
                candidates=[cand_dict], risk_context=risk_context,
                risk_missing=risk_missing,
                notes='post-hoc trial at the 15:00 cutoff; query_ts==asof_ts (no separate '
                      'query clock in replay); downside/trigger null by design').bind()
            packs.append(dict(
                evidence_id=pack.evidence_id, pack_hash=pack.pack_hash,
                code=code, P=P_DAY, E=E_DAY, asof_ts=pack.asof_ts,
                query_ts=pack.query_ts, market_date=pack.market_date,
                candidates=pack.candidates, risk_context=pack.risk_context,
                risk_missing=pack.risk_missing, corvus_gap=pack.corvus_gap,
                confirmed_plan=pack.confirmed_plan, notes=pack.notes))

    # ---------------- 6) cross-check vs rebuild_features_v1 (same code+cutoff) ----------------
    xcheck = []
    if result_rows:
        e1 = pd.read_parquet(v1.OUT / 'candidate_scores_e_update.parquet')
        e1['code'] = e1.code.astype(str).str.zfill(6)
        for r in result_rows:
            hit = e1[(e1.code == r['code']) & (e1[TIME] == r['cutoff_ts'])]
            if len(hit):
                h = hit.iloc[0]
                xcheck.append(dict(code=r['code'], cutoff_ts=r['cutoff_ts'],
                                   trial_amp_232=r['amp_232'], v1_amp_232=float(h.amp_232),
                                   delta_232=abs(r['amp_232'] - float(h.amp_232)),
                                   trial_amp_266=r['amp_266'], v1_amp_266=float(h.amp_266),
                                   delta_266=abs(r['amp_266'] - float(h.amp_266))))
    put('crosscheck_v1.json', json.dumps(dict(
        note='rebuild_features_v1 covers exactly 1 of the 20 (code,20260908150000) keys '
             '(its 695-population); a 3-sample comparison is impossible, the single overlap '
             'must match bit-exactly', pairs=xcheck), indent=1).encode())

    # ---------------- outputs ----------------
    packs_body = ''.join(json.dumps(p, ensure_ascii=False) + '\n' for p in packs)
    put('packs.jsonl', packs_body.encode())
    put('trial_results.json', json.dumps(dict(
        coverage_reasons=cov.reason.value_counts().to_dict(),
        survivors=survivors, scored=result_rows), indent=1).encode())

    report = build_report(cand, coverage, survivors, result_rows, packs, xcheck)
    put('trial_report.md', report.encode())

    put('stop_record.json', json.dumps(dict(
        status='COMPLETE_TRIAL_NOT_FIT', http_requests=0, llm_calls=0, fit=False,
        candidates_total=int(len(cand)), coverage_ok=len(survivors),
        packs_written=len(packs),
        manifest={k: v['sha256'] for k, v in writer.items()}), indent=1).encode(),
        terminal=True)
    log('done', packs=len(packs), survivors=len(survivors))


def build_report(cand, coverage, survivors, result_rows, packs, xcheck):
    cov = pd.DataFrame(coverage)
    lines = []
    a = lines.append
    a('# 当日证据流试运行报告（evidence_stream_trial_v1）')
    a('')
    a('时间：%s。纪律：零新增取数、未改任何冻结件、无重训、无 LLM 调用、'
      '无联合对照；全部产物在本目录（<8MiB）。' % time.strftime('%Y-%m-%d %H:%M:%S'))
    a('试跑日：P=20260907（T 日收盘入名单）→ E=20260908 15:00 cutoff'
      '（冻结分钟账本最后完整日）。')
    a('')
    a('## 1. 候选清单（冻结 full_master_table_v2.parquet，P=20260907，admission=OK）')
    a('')
    a('共 **%d** 只（A_sdp %d / B_lvb %d），全部 admission=OK。' % (
        len(cand), int((cand.pool == 'A_sdp').sum()), int((cand.pool == 'B_lvb').sum())))
    a('')
    a('| code | pool | F618 | atr_ref | plan_dist_atr | pB2(FINAL) |')
    a('|---|---|---|---|---|---|')
    for _, r in cand.iterrows():
        a('| %s | %s | %s | %s | %s | %s |' % (
            r.code, r.pool,
            '%.4f' % r.F618 if pd.notna(r.F618) else 'NaN',
            '%.6f' % r.atr_ref if pd.notna(r.atr_ref) else 'NaN',
            '%.4f' % r.plan_dist_atr if pd.notna(r.plan_dist_atr) else 'NaN',
            '%.6f' % r.pB2 if pd.notna(r.pB2) else 'NaN'))
    nan_fields = cand[cand.F618.isna()].code.tolist()
    if nan_fields:
        a('')
        a('字段缺口：%s 在 master 中 F618/atr_ref/plan_dist_atr 为 NaN（如实保留为 null）。'
          % ', '.join(nan_fields))
    a('')
    a('## 2. 覆盖核验与留因分布（冻结分钟账本，E 日 15:00 cutoff 需全日 240 根 + P 日全日）')
    a('')
    vc = cov.reason.value_counts()
    a('| reason | n |')
    a('|---|---|')
    for k, v in vc.items():
        a('| %s | %d |' % (k, v))
    a('')
    a('逐条明细见 coverage_trial.json。要点：')
    a('')
    for _, r in cov.iterrows():
        if r.reason == 'OK':
            continue
        extra = ''
        fmd = r.file_max_day if pd.notna(r.file_max_day) else None
        if r.reason == 'e_day_incomplete_bars' and fmd is not None:
            extra = '（分钟文件最后交易日 %s）' % int(fmd)
        a('- `%s`：%s — %s%s' % (r.code, r.reason, r.detail, extra))
    a('')
    a('## 3. 重建与推理（仅覆盖通过者；冻结构建器只 import 不改）')
    a('')
    if not result_rows:
        a('**无覆盖通过的候选，管道止步于此（如实报告，不硬凑）。**')
    else:
        a('| code | amp_232 | amp_266 | n_missing_232 | n_missing_266 | isolated |')
        a('|---|---|---|---|---|---|')
        for r in result_rows:
            a('| %s | %.6f | %.6f | %d | %d | %s |' % (
                r['code'], r['amp_232'], r['amp_266'], r['n_missing_232'],
                r['n_missing_266'], r['isolated']))
        a('')
        a('特征缺口（对成功候选同样如实披露）：该 cutoff 不在冻结 multiboard 队列月份表'
          '（shared_202609 仅 16 个队列 cutoff、无本 cutoff；own 月表无本键），'
          'nh sidecar 止于 20260901——故 mb_/golden/nh_* 字段为 NaN，own5_/own15_ 由冻结'
          '重放矩阵经冻结内核重算填入；joint_parts/context_loo 在本 cutoff 仅覆盖 300014。')
    a('')
    a('## 4. 证据包（packs.jsonl，契约对齐 stargazer_desktop/oracles_v1.py）')
    a('')
    a('成功证据包 **%d** 个。逐包字段：EvidencePack 形状'
      '（evidence_id/pack_hash/asof_ts/query_ts/market_date/candidates/risk_context/'
      'risk_missing/corvus_gap），候选段为 tarot_candidate 形状'
      '（p_close_score=pB2、golden_zone=golden_dist_618_atr、f618/plan_dist_atr/atr_ref '
      '来自冻结 master，downside_state 与 trigger 显式 null——本试运行不做盘中回放），'
      'risk_context 为 pyxis_context 形状（version=232主版，模型/预处理哈希经冻结注册表'
      ' stamping），market_state/board_states/leader_states 取冻结重放矩阵真实值。' % len(packs))
    for p in packs:
        rc = p['risk_context']
        a('')
        a('- `%s`：amp_232=%.6f amp_266=%.6f；market_state.trend=%s state=%s '
          'strength=%s；board_states[0].strength=%s breadth=%s；leader_states[0]=%s；'
          'risk_missing=%s；n_missing_232=%s n_missing_266=%s' % (
              p['evidence_id'], rc['amp_232'], rc['amp_266'],
              rc['market_state'].get('trend'), rc['market_state'].get('state'),
              rc['market_state'].get('strength'),
              rc['board_states'][0].get('strength') if rc['board_states'] else None,
              rc['board_states'][0].get('breadth') if rc['board_states'] else None,
              rc['leader_states'][0].get('code') if rc['leader_states'] else None,
              p['risk_missing'] or '[]',
              rc.get('n_missing_232'), rc.get('n_missing_266')))
    a('')
    a('query_ts 口径：重演取 cutoff 自身（query_ts==asof_ts=20260908150000），'
      '报告与 plan.json 中已注明。')
    a('')
    a('## 5. 与既有研究产物交叉核验（rebuild_features_v1）')
    a('')
    if xcheck:
        for x in xcheck:
            a('- `%s @ %s`：trial amp_232=%.6f vs v1=%.6f（Δ=%.2e）；'
              'amp_266=%.6f vs v1=%.6f（Δ=%.2e）→ %s' % (
                  x['code'], x['cutoff_ts'], x['trial_amp_232'], x['v1_amp_232'],
                  x['delta_232'], x['trial_amp_266'], x['v1_amp_266'], x['delta_266'],
                  '一致' if x['delta_232'] == 0 and x['delta_266'] == 0 else '不一致'))
        a('')
        a('注：rebuild_features_v1 的 695 人口中恰含 1 条本日 (code,cutoff) 键'
          '（300014），无法抽样 3 只；该唯一重叠键必须位级一致，已核验如上。')
    else:
        a('无 (code,cutoff) 重叠，无法交叉核验（如实报告）。')
    a('')
    a('## 6. 结论与管道缺口')
    a('')
    a('- 管道端到端**可运行**：候选读取→覆盖门禁→冻结重放→232/266 特征重建→'
      '双模型冻结推理→EvidencePack 组装，全链路零取数、零改冻结件。')
    a('- 当日可评分横截面极薄：20 只 OK 候选中仅 **%d** 只通过 E 日全日分钟覆盖'
      '（19 只留因：多数分钟文件止于 20260907，5 只不在冻结分钟账本）；'
      '且该 cutoff 的 joint_parts/context_loo 冻结侧车只覆盖 300014。'
      '证据流“同刻可评分”在冻结数据下只对这 %d 只成立。' % (len(survivors), len(survivors)))
    a('- 能力声明：本产物为工程试运行（EXTENSION_RESEARCH），不构成任何模型能力结论；'
      '联合对照由主代理另行执行。')
    a('')
    a('## 7. 产物清单')
    a('')
    a('plan.json / candidates_p20260907.json / coverage_trial.json / '
      'frozen_join_availability.json / trial_results.json / crosscheck_v1.json / '
      'packs.jsonl / trial_report.md / stop_record.json')
    return '\n'.join(lines) + '\n'


if __name__ == '__main__':
    main()
