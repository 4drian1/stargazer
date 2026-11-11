"""Pyxis build_features 内存剖面 v2 —— 只读诊断，不修改任何生产代码。

目的: 定位 build_features 中随股票数线性累积的对象。

特点:
  - 真实 Windows 内存 (GetProcessMemoryInfo: WorkingSet / PeakWorkingSet / PrivateUsage)
  - 逐票采样 (每 10 只), 观察是阶梯累积还是锯齿瞬时
  - tracemalloc 快照 diff (每 100 只), 直接指出累积对象
  - 不改冻结代码: 仅 monkey-patch 模块属性, 进程退出即失效

用法:
  python -X utf8 pyxis_mem_profile_v2.py --n 200
  python -X utf8 pyxis_mem_profile_v2.py --n 500 --skip-precheck  (已知全部可计算时)
"""
from __future__ import annotations

import argparse
import ctypes
import gc
import hashlib
import json
import sys
import threading
import time
import tracemalloc
from ctypes import wintypes
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))


class _PMC(ctypes.Structure):
    _fields_ = [
        ('cb', wintypes.DWORD),
        ('PageFaultCount', wintypes.DWORD),
        ('PeakWorkingSetSize', ctypes.c_size_t),
        ('WorkingSetSize', ctypes.c_size_t),
        ('QuotaPeakPagedPoolUsage', ctypes.c_size_t),
        ('QuotaPagedPoolUsage', ctypes.c_size_t),
        ('QuotaPeakNonPagedPoolUsage', ctypes.c_size_t),
        ('QuotaNonPagedPoolUsage', ctypes.c_size_t),
        ('PagefileUsage', ctypes.c_size_t),
        ('PeakPagefileUsage', ctypes.c_size_t),
    ]


_k32 = ctypes.WinDLL('kernel32')
_GetMem = getattr(_k32, 'K32GetProcessMemoryInfo', None)
if _GetMem is None:
    _GetMem = ctypes.WinDLL('psapi').GetProcessMemoryInfo
_GetMem.argtypes = [wintypes.HANDLE, ctypes.POINTER(_PMC), wintypes.DWORD]
_GetMem.restype = wintypes.BOOL


def mem():
    """真实进程内存 (MB)。Windows 专用, 失败返回空 dict。"""
    c = _PMC()
    c.cb = ctypes.sizeof(c)
    if not _GetMem(wintypes.HANDLE(-1), ctypes.byref(c), c.cb):
        return {}
    mb = 1024 * 1024
    return {
        'ws_mb': round(c.WorkingSetSize / mb, 1),
        'peak_ws_mb': round(c.PeakWorkingSetSize / mb, 1),
        'private_mb': round(c.PagefileUsage / mb, 1),
        'peak_private_mb': round(c.PeakPagefileUsage / mb, 1),
    }


def tm_cur_mb():
    return round(tracemalloc.get_traced_memory()[0] / 1024 / 1024, 1)


def top_diff(old, new, limit=8):
    """两次 tracemalloc 快照的净增 top。"""
    diff = new.compare_to(old, 'lineno')
    out = []
    for stat in diff[:limit]:
        if stat.size_diff <= 0:
            continue
        frame = stat.traceback[0]
        out.append({
            'file': Path(frame.filename).name,
            'line': frame.lineno,
            'net_mb': round(stat.size_diff / 1024 / 1024, 2),
            'count': stat.count_diff,
        })
    return out


class StageSampler:
    """50ms 采样的分阶段 WS 峰值记录器。

    enter/exit 用栈支持嵌套: 外层阶段峰值含内层 (inclusive), 内层自身窗口单独记峰值。
    生命周期峰值 (PeakWorkingSetSize) 是进程级单调值, 无法分阶段, 必须靠主动采样。
    """

    def __init__(self, interval_s=0.05):
        self.interval = interval_s
        self._stack = []
        self._peaks = {}
        self._elapsed = {}
        self._lock = threading.Lock()
        self._stop = False
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self):
        while not self._stop:
            with self._lock:
                active = list(self._stack)
            if active:
                m = mem()
                ws = m.get('ws_mb', 0)
                priv = m.get('private_mb', 0)
                with self._lock:
                    for name in active:
                        cur = self._peaks.get(name)
                        if cur is None or ws > cur[0]:
                            self._peaks[name] = (ws, priv)
            time.sleep(self.interval)

    def enter(self, name):
        with self._lock:
            self._stack.append(name)

    def exit(self, name):
        with self._lock:
            if self._stack and self._stack[-1] == name:
                self._stack.pop()
            elif name in self._stack:
                self._stack.remove(name)

    def add_elapsed(self, name, seconds):
        with self._lock:
            self._elapsed[name] = self._elapsed.get(name, 0.0) + seconds

    def report(self):
        with self._lock:
            return {
                name: {'peak_ws_mb': ws, 'peak_private_mb': priv,
                       'elapsed_s': round(self._elapsed.get(name, 0.0), 2)}
                for name, (ws, priv) in sorted(self._peaks.items())
            }

    def stop(self):
        self._stop = True
        self._thread.join(timeout=1.0)


def make_stage(sampler, name, fn, *, hot=False, log=None):
    """包装一个阶段函数。hot=True 用于每票/每板块高频调用: 不 gc 不打印, 只累计。"""
    def wrapped(*a, **kw):
        if not hot:
            gc.collect()
        before = mem()
        t0 = time.time()
        sampler.enter(name)
        try:
            return fn(*a, **kw)
        finally:
            sampler.exit(name)
            sampler.add_elapsed(name, time.time() - t0)
            if not hot:
                gc.collect()
                rec = {'stage': name, 'elapsed_s': round(time.time() - t0, 2),
                       'before_ws_mb': before.get('ws_mb'), 'after': mem()}
                if log is not None:
                    log.append(rec)
                print('[stage]', name, rec, flush=True)
    return wrapped


def install_stage_patches(sampler):
    """对 build_features 各阶段函数打补丁。生产代码零修改, 进程退出即失效。

    build_features 内部用 `from X import Y` 在每次调用时取当前属性, 所以
    patch 源模块属性即可生效。"""
    import joint_shadow_context_v1 as ctx
    import build_multiboard_full_features_v1 as bmff
    import multiboard_risk_features_v1 as mrf
    import conditional_board_patch_v1 as cbp
    import joint_fit_preparation_v1 as jfp
    import rebuild_features_v1 as rf
    import build_common_candidate_sidecar_v1 as bccs
    import materialize_multitimeframe_momentum_v1 as mm
    import join_momentum_review_v1 as jmr

    stages_log = []
    ctx._market = make_stage(sampler, '_market', ctx._market, log=stages_log)
    ctx._context = make_stage(sampler, '_context', ctx._context, log=stages_log)
    # _market 内部子阶段
    mm.build = make_stage(sampler, 'market.momentum_build', mm.build, log=stages_log)
    jmr.project_market = make_stage(sampler, 'market.project_market', jmr.project_market,
                                    log=stages_log)
    # 矩阵循环 (L317-334 内联段): 高频函数统一归入 matrix_loop
    for mod, attr in ((bmff, 'market_arrays'), (bmff, 'golden_fields'),
                      (mrf, 'matrix_features'), (mrf, 'summarize'),
                      (cbp, 'with_new_high')):
        mod.__dict__[attr] = make_stage(sampler, 'matrix_loop', mod.__dict__[attr], hot=True)
    jfp.derive = make_stage(sampler, 'derive', jfp.derive, log=stages_log)
    rf.resolve_logical = make_stage(sampler, 'resolve_logical', rf.resolve_logical,
                                    log=stages_log)
    bccs.month_features = make_stage(sampler, 'month_features', bccs.month_features,
                                     log=stages_log)
    return stages_log


def _sha_array(a, limit=16):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()[:limit]


def save_baseline(ref_dir, frame, x232, x266, preds):
    ref_dir.mkdir(exist_ok=True)
    x232.to_numpy(dtype=np.float64).astype(np.float32).dump(str(ref_dir / 'x232.npy'))
    x266.to_numpy(dtype=np.float64).astype(np.float32).dump(str(ref_dir / 'x266.npy'))
    frame.to_parquet(ref_dir / 'frame.parquet')
    hashes = {'x232_f32': _sha_array(x232.to_numpy(dtype=np.float64).astype(np.float32)),
              'x266_f32': _sha_array(x266.to_numpy(dtype=np.float64).astype(np.float32)),
              'pred': {k: _sha_array(np.asarray(v, dtype=np.float64)) for k, v in preds.items()}}
    (ref_dir / 'hashes.json').write_text(json.dumps(hashes, indent=1), encoding='utf8')
    return hashes


def compare_baseline(ref_dir, frame, x232, x266, preds, stash_dir=None):
    """特征/预测逐值对照。float64→float32 量化: 容忍 <1e-6 相对噪声, NaN 位置必须一致。

    基线 .npy 由 ndarray.dump() 写出 (pickle 容器), np.load 需要 allow_pickle=True;
    文件是本脚本自己写在本目录下的, 可信。stash_dir 先落盘优化侧输出,
    保证对照逻辑自身的缺陷不会丢掉本次运行结果。"""
    res = {}
    if stash_dir is not None:
        stash_dir.mkdir(exist_ok=True)
        np.save(stash_dir / 'x232.npy', x232.to_numpy(dtype=np.float64))
        np.save(stash_dir / 'x266.npy', x266.to_numpy(dtype=np.float64))
        frame.to_parquet(stash_dir / 'frame.parquet')
        (stash_dir / 'pred_hashes.json').write_text(json.dumps(
            {k: _sha_array(np.asarray(v, dtype=np.float64)) for k, v in preds.items()}),
            encoding='utf8')
    res = {}
    x232_ref = np.load(ref_dir / 'x232.npy', allow_pickle=True)
    x266_ref = np.load(ref_dir / 'x266.npy', allow_pickle=True)
    x232_new = x232.to_numpy(dtype=np.float64).astype(np.float32)
    x266_new = x266.to_numpy(dtype=np.float64).astype(np.float32)
    for label, ref, new in (('x232', x232_ref, x232_new), ('x266', x266_ref, x266_new)):
        same_shape = ref.shape == new.shape
        if same_shape:
            mask_ref, mask_new = np.isnan(ref), np.isnan(new)
            nan_ok = bool((mask_ref == mask_new).all())
            val = ref[~mask_ref] - new[~mask_new]
            val_ok = bool(nan_ok and np.all(np.abs(val) <= 1e-6 * np.maximum(1.0, np.abs(ref[~mask_ref]))))
        else:
            nan_ok = val_ok = False
        res[label] = {'shape_ref': list(ref.shape), 'shape_new': list(new.shape),
                      'nan_pattern_equal': nan_ok, 'values_equal_1e-6': val_ok,
                      'max_abs_diff': (float(np.nanmax(np.abs(ref - new)))
                                       if same_shape else None)}
    frame_ref = pd.read_parquet(ref_dir / 'frame.parquet')
    common = [c for c in frame_ref.columns if c in frame.columns]
    only_ref = [c for c in frame_ref.columns if c not in frame.columns]
    only_new = [c for c in frame.columns if c not in frame_ref.columns]
    diff_cols, dtype_changes = [], []
    if list(frame_ref.index) == list(frame.index) or len(frame_ref) == len(frame):
        for c in common:
            a, b = frame_ref[c], frame[c]
            na_a, na_b = pd.isna(a), pd.isna(b)
            if not bool((na_a.values == na_b.values).all()):
                diff_cols.append(c + ':NaN')
                continue
            mask = ~na_a
            if mask.any():
                va = pd.to_numeric(a[mask], errors='coerce')
                vb = pd.to_numeric(b[mask], errors='coerce')
                if va.notna().all() and vb.notna().all():
                    if not np.allclose(va.to_numpy(dtype=float), vb.to_numpy(dtype=float),
                                       rtol=0, atol=1e-9, equal_nan=False):
                        diff_cols.append(c + ':value')
                elif not (a[mask].astype(str).values == b[mask].astype(str).values).all():
                    diff_cols.append(c + ':str')
            if str(a.dtype) != str(b.dtype):
                dtype_changes.append(f'{c}:{a.dtype}->{b.dtype}')
    res['frame'] = {'rows_ref': len(frame_ref), 'rows_new': len(frame),
                    'cols_only_ref': only_ref[:20], 'cols_only_new': only_new[:20],
                    'value_diff_columns': diff_cols[:30], 'diff_count': len(diff_cols),
                    'dtype_changes': dtype_changes[:30],
                    'equal': not diff_cols and not only_ref and not only_new}
    res['pred'] = {}
    for k, ref_sha in json.loads((ref_dir / 'hashes.json').read_text(encoding='utf8'))['pred'].items():
        new_sha = _sha_array(np.asarray(preds[k], dtype=np.float64)) if k in preds else None
        res['pred'][k] = {'ref_sha': ref_sha, 'new_sha': new_sha, 'equal': new_sha == ref_sha}
    res['all_equal'] = bool(res['x232']['values_equal_1e-6'] and res['x266']['values_equal_1e-6']
                            and res['frame']['equal']
                            and all(v['equal'] for v in res['pred'].values()))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=200)
    ap.add_argument('--cutoff', type=int, default=20261016094500)
    ap.add_argument('--full-universe', action='store_true',
                    help='完整成员集合冷启动: 不收缩 boards, 不剔除任何股票')
    ap.add_argument('--skip-precheck', action='store_true',
                    help='跳过异常预检(仅当样本集已知全部可计算时)')
    ap.add_argument('--save-baseline', action='store_true',
                    help='保存 frame/x232/x266/预测 到 _baseline_ref_v1/, 供 --compare 对照')
    ap.add_argument('--compare', action='store_true',
                    help='与 _baseline_ref_v1/ 逐值对照, 输出一致性结论')
    ap.add_argument('--out', default='_mem_profile_v2.json')
    args = ap.parse_args()

    tracemalloc.start(1)
    report = {
        'cutoff': args.cutoff, 'n_requested': args.n,
        'started_at': datetime.now().isoformat(timespec='seconds'),
        'stages': {}, 'samples': [], 'diffs': [],
    }
    t_all = time.time()

    # --- 1. produce snapshot ---
    t0 = time.time()
    from joint_shadow_snapshot_v1 import produce
    snapshot, audit = produce(args.cutoff, historical=True)
    day = args.cutoff // 1000000
    P = snapshot.days[-2]
    report['stages']['produce'] = {
        'elapsed_s': round(time.time() - t0, 1), **mem(), 'tm_cur_mb': tm_cur_mb(),
        'days': len(snapshot.days), 'stock_minutes': len(snapshot.stock_minutes),
        'boards': len(snapshot.boards),
        'index_mem_mb': {k: round(v.memory_usage(deep=True).sum() / 1048576, 2)
                         for k, v in snapshot.index_5m.items()},
    }
    print('[produce]', report['stages']['produce'], flush=True)

    # --- 2. 取候选 (按板块成员顺序, 保证与生产一致) ---
    all_board_codes = sorted({c for m in snapshot.boards.values() for c in m
                              if c in snapshot.stock_minutes})
    n_target = min(args.n, len(all_board_codes))
    scope = snapshot.membership_scope
    research = scope == 'SNAPSHOT_RECONSTRUCTION_RESEARCH'

    if args.full_universe:
        # 完整成员集合: 不收缩 boards, 不剔除股票. 单股异常由 build_features 隔离.
        codes = all_board_codes
        report['stages']['precheck'] = {
            'elapsed_s': 0, 'skipped': True, 'full_universe': True,
            'boards': len(snapshot.boards),
            'members': len({c for m in snapshot.boards.values() for c in m}),
            'bound_members': len(codes),
            'universe_note': 'FULL_MEMBER_SET_NO_FILTERING',
        }
        print('[precheck] skipped (full universe)', flush=True)
    elif args.skip_precheck:
        codes = all_board_codes[:n_target]
    else:
        from joint_shadow_context_v1 import _prefix
        cache_path = HERE / '_bad_codes_cache.json'
        t0 = time.time()
        if cache_path.exists():
            cache = json.loads(cache_path.read_text(encoding='utf8'))
            bad, rows_count = cache['bad'], cache['rows_count']
            good = [c for c in all_board_codes if c not in bad]
            cached = True
        else:
            cached = False
            good, bad, rows_count = [], {}, {}
            # 必须预检全部成员: build_features 遍历 boards 全集, 任一坏股票会崩全批
            for code in all_board_codes:
                try:
                    raw = _prefix(snapshot.stock_minutes[code], args.cutoff,
                                  snapshot.receipts.get(code), stock=True, research=research,
                                  observed_at=snapshot.observed_at)
                    good.append(code)
                    rows_count[code] = len(raw)
                except Exception as exc:
                    s = str(exc)
                    key = next((k for k in ('INVALID_SOURCE_PRICE', 'INVALID_SOURCE_OHLC',
                                            'SOURCE_RECEIPT_MISSING', 'SOURCE_CLOCK_UNBOUND',
                                            'INVALID_SOURCE_ACTIVITY') if k in s),
                               f'{type(exc).__name__}:{s[:40]}')
                    bad[code] = key
            cache_path.write_text(json.dumps({'bad': bad, 'rows_count': rows_count},
                                             ensure_ascii=False), encoding='utf8')
        # 选历史最长的 n 只: 单票瞬时峰值由最长历史决定, 必须优先暴露最坏情况
        codes = sorted(good, key=lambda c: -rows_count.get(c, 0))[:n_target]
        keep = set(codes)
        # 诊断专用: 缩小 boards 以模拟"单股异常隔离"; 空板块必须删除(第245行校验)
        for bk in list(snapshot.boards):
            m = [c for c in snapshot.boards[bk] if c in keep]
            if m:
                snapshot.boards[bk] = m
            else:
                del snapshot.boards[bk]
        gc.collect()
        report['stages']['precheck'] = {
            'elapsed_s': round(time.time() - t0, 1), **mem(), 'tm_cur_mb': tm_cur_mb(),
            'from_cache': cached, 'good': len(good), 'bad': len(bad),
            'bad_dist': {k: sum(1 for v in bad.values() if v == k) for k in set(bad.values())},
            'boards_members_before': len(all_board_codes),
            'boards_members_after': len({c for m in snapshot.boards.values() for c in m}),
            'top_rows_count': {c: rows_count[c] for c in codes[:5]},
            'universe_note': 'DIAGNOSTIC_SUBSET_ONLY_NOT_FINAL_RESULT',
        }
        print('[precheck]', report['stages']['precheck'], flush=True)

    candidates = [{'code': c, 'cutoff_ts': args.cutoff, 'P': P, 'E': day} for c in codes]

    # --- 3. patch stock_features 做逐票采样 (不改源码) + 分阶段采样器 ---
    import build_joint_risk_features_v1 as bjrf
    import joint_shadow_context_v1 as ctx

    sampler = StageSampler()
    stages_log = install_stage_patches(sampler)
    sampler.enter('outer')

    orig_sf = bjrf.stock_features
    state = {'i': 0, 'samples': [], 'diffs': [], 'last_snap': tracemalloc.take_snapshot(),
             'raw_sizes': [], 'out_sizes': []}

    def wrapped(raw, code, days, *a, **kw):
        out = orig_sf(raw, code, days, *a, **kw)
        state['i'] += 1
        i = state['i']
        # 采样点才计算体积: deep=True 在 98568 行 DataFrame 上每只多耗数秒
        if i <= 5 or i % 10 == 0:
            try:
                state['raw_sizes'].append(round(raw.memory_usage(deep=False).sum() / 1048576, 2))
                state['out_sizes'].append(round(out.memory_usage(deep=True).sum() / 1048576, 2))
            except Exception:
                pass
            state['samples'].append({
                'i': i, 'code': code, **mem(), 'tm_cur_mb': tm_cur_mb(),
                'last_raw_mb': state['raw_sizes'][-1] if state['raw_sizes'] else None,
                'last_out_mb': state['out_sizes'][-1] if state['out_sizes'] else None,
            })
        if i % 100 == 0:
            new = tracemalloc.take_snapshot()
            d = top_diff(state['last_snap'], new)
            state['last_snap'] = new
            state['diffs'].append({'i': i, **mem(), 'top_net': d})
            print('[scan] i=%d %s %s' % (i, code, mem()), flush=True)
        return out

    bjrf.stock_features = wrapped
    try:
        t0 = time.time()
        frame, x232, x266, audit_bf = ctx.build_features(snapshot, candidates)
        elapsed = round(time.time() - t0, 1)
        report['stages']['build_features'] = {
            'elapsed_s': elapsed, **mem(), 'tm_cur_mb': tm_cur_mb(),
            'frame_shape': list(frame.shape), 'x232_shape': list(x232.shape),
            'x266_shape': list(x266.shape),
            'frame_mb': round(frame.memory_usage(deep=True).sum() / 1048576, 2),
            'stocks': audit_bf['stocks'], 'boards': audit_bf['boards'],
            'missing_stock_sources': len(audit_bf['missing_stock_sources']),
            'invalid_stock_count': len(audit_bf['invalid_stock_sources']),
            'invalid_stock_reasons': {
                r: sum(1 for x in audit_bf['invalid_stock_sources'] if x['reason'] == r)
                for r in {x['reason'] for x in audit_bf['invalid_stock_sources']}},
            'invalid_stock_sample': audit_bf['invalid_stock_sources'][:10],
            'computable_stock_count': audit_bf['computable_stock_count'],
            'x232_all_nan_rows': int(x232.isna().all(axis=1).sum()),
        }
        print('[build_features]', report['stages']['build_features'], flush=True)
        if args.full_universe:
            t1 = time.time()
            import rebuild_features_v1 as frozen
            x = {'232': x232, '266': x266}
            preds = {}
            try:
                boosters = [frozen.load_booster(frozen.MODEL_232), frozen.load_booster(frozen.MODEL_266)]
                sampler.enter('predict')
                preds = {k: m.predict(frozen.native_missing_view(x[k], list(x[k])))
                         for k, m in zip(x, boosters)}
            except Exception as exc:
                report['stages']['predict'] = {'error': f'{type(exc).__name__}: {str(exc)[:300]}'}
                print('[predict][ERROR]', report['stages']['predict'], flush=True)
            finally:
                sampler.exit('predict')
                gc.collect()
            if preds:
                report['stages']['predict'] = {
                    'elapsed_s': round(time.time() - t1, 1), **mem(),
                    'rows': {k: int(len(v)) for k, v in preds.items()},
                    'nonfinite': {k: int((~np.isfinite(v)).sum()) for k, v in preds.items()},
                    'first5': {k: [round(float(v), 6) for v in values[:5]]
                               for k, values in preds.items()},
                    'order_matches_candidates': bool(
                        len(preds['232']) == len(candidates)),
                }
                print('[predict]', report['stages']['predict'], flush=True)
                if args.save_baseline:
                    sampler.exit('outer')
                    hashes = save_baseline(HERE / '_baseline_ref_v1', frame, x232, x266, preds)
                    report['stages']['baseline_saved'] = {'dir': '_baseline_ref_v1',
                                                          'hashes': hashes}
                    sampler.enter('outer')
                    print('[baseline] saved', hashes, flush=True)
                if args.compare:
                    try:
                        res = compare_baseline(HERE / '_baseline_ref_v1', frame, x232, x266, preds,
                                               stash_dir=HERE / '_optimized_out_v1')
                        report['stages']['baseline_comparison'] = res
                        print('[compare]', json.dumps(res, ensure_ascii=False)[:1200], flush=True)
                    except Exception as exc:
                        # 对照失败不得覆盖 build_features 的成功记录
                        report['stages']['baseline_comparison'] = {
                            'error': f'{type(exc).__name__}: {str(exc)[:300]}'}
                        print('[compare][ERROR]', report['stages']['baseline_comparison'], flush=True)
        # 保留对象, 供观察释放
        del frame, x232, x266
        gc.collect()
        report['stages']['after_del_gc'] = {**mem(), 'tm_cur_mb': tm_cur_mb()}
    except Exception as exc:
        report['stages']['build_features'] = {
            'elapsed_s': round(time.time() - t0, 1), **mem(),
            'error': f'{type(exc).__name__}: {str(exc)[:300]}',
        }
        print('[build_features][ERROR]', report['stages']['build_features'], flush=True)
    finally:
        bjrf.stock_features = orig_sf
        sampler.exit('outer')
        report['stage_peaks'] = sampler.report()
        report['stage_timeline'] = stages_log
        sampler.stop()

    report['samples'] = state['samples']
    report['diffs'] = state['diffs']
    report['raw_size_avg_mb'] = (round(sum(state['raw_sizes']) / len(state['raw_sizes']), 2)
                                 if state['raw_sizes'] else None)
    report['out_size_avg_mb'] = (round(sum(state['out_sizes']) / len(state['out_sizes']), 2)
                                 if state['out_sizes'] else None)
    report['total_elapsed_s'] = round(time.time() - t_all, 1)
    report['finished_at'] = datetime.now().isoformat(timespec='seconds')

    out_path = HERE / args.out
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf8')

    # --- 控制台摘要 ---
    print('\n=== 摘要 ===')
    print('每 10 只采样 (i, ws_mb, peak_ws_mb, private_mb, tm_cur_mb, raw_mb, out_mb):')
    for s in report['samples']:
        print('  %4d | ws=%7s peak=%7s priv=%7s tm=%7s raw=%5s out=%5s'
              % (s['i'], s.get('ws_mb'), s.get('peak_ws_mb'), s.get('private_mb'),
                 s.get('tm_cur_mb'), s.get('last_raw_mb'), s.get('last_out_mb')))
    print('\n每 100 只净增 top:')
    for d in report['diffs']:
        print('  i=%4d ws=%7s peak=%7s' % (d['i'], d.get('ws_mb'), d.get('peak_ws_mb')))
        for t in d['top_net'][:5]:
            print('      +%6.2fMB  %s:%s  (%+d objs)'
                  % (t['net_mb'], t['file'], t['line'], t['count']))
    print('\nraw_size_avg=%s MB  out_size_avg=%s MB  total=%ss'
          % (report['raw_size_avg_mb'], report['out_size_avg_mb'],
             report['total_elapsed_s']))
    print('报告:', out_path)


if __name__ == '__main__':
    main()
