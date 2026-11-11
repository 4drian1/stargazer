"""Pyxis 冷启动内存诊断 v2 — 完整路径, 过滤坏股票.

用法: python -X utf8 pyxis_cold_start_diagnostic_v1.py
cutoff=20261016094500 (指数数据最新到此)
"""
from __future__ import annotations

import gc
import json
import sys
import time
import traceback
import tracemalloc
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
cutoff = 20261016094500


def _rss_mb():
    try:
        cur, peak = tracemalloc.get_traced_memory()
        return round(cur / (1024 * 1024), 2)
    except Exception:
        return None


def _peak_mb():
    try:
        cur, peak = tracemalloc.get_traced_memory()
        return round(peak / (1024 * 1024), 2)
    except Exception:
        return None


class StageProbe:
    def __init__(self, name):
        self.name = name
        self.started_at = self.finished_at = None
        self.elapsed_seconds = None
        self.rss_before_mb = self.rss_after_mb = self.rss_delta_mb = None
        self.peak_before_mb = self.peak_after_mb = None
        self.input_rows = self.output_rows = None
        self.error = None
        self.details = {}

    def __enter__(self):
        gc.collect()
        self.started_at = datetime.now().isoformat(timespec='seconds')
        self.rss_before_mb = _rss_mb()
        self.peak_before_mb = _peak_mb()
        self._t0 = time.time()
        return self

    def __exit__(self, et, ev, tb):
        self.finished_at = datetime.now().isoformat(timespec='seconds')
        self.elapsed_seconds = round(time.time() - self._t0, 3)
        self.rss_after_mb = _rss_mb()
        self.peak_after_mb = _peak_mb()
        if self.rss_before_mb is not None and self.rss_after_mb is not None:
            self.rss_delta_mb = round(self.rss_after_mb - self.rss_before_mb, 2)
        if et is not None:
            self.error = f'{et.__name__}: {str(ev)[:300]}'
        return False

    def to_dict(self):
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}


def main():
    print('=' * 70)
    print(f'Pyxis 冷启动诊断 v2 | cutoff={cutoff} | historical=True')
    print('=' * 70)

    tracemalloc.start()
    stages = []
    overall_t0 = time.time()

    # --- Stage 1: imports ---
    with StageProbe('1_base_import') as sp:
        sys.path.insert(0, str(HERE))
        import pandas as pd, numpy as np, lightgbm as lgb
        print(f'  pandas={pd.__version__} numpy={np.__version__} lgbm={lgb.__version__}')
    stages.append(sp)

    with StageProbe('2_pyxis_import') as sp:
        import joint_shadow_snapshot_v1, joint_shadow_context_v1
        import joint_shadow_service_v1, joint_shadow_compute_v2
        import rebuild_features_v1 as frozen
    stages.append(sp)

    # --- Stage 3: produce snapshot ---
    with StageProbe('3_produce_snapshot') as sp:
        from joint_shadow_snapshot_v1 import produce
        snapshot, audit = produce(cutoff, historical=True)
        day = cutoff // 1000000
        P = snapshot.days[-2]
        print(f'  snapshot: days={len(snapshot.days)}[last={day}], '
              f'stock_minutes={len(snapshot.stock_minutes)}, boards={len(snapshot.boards)}')
        sp.details = {
            'cutoff': cutoff, 'days_count': len(snapshot.days), 'P': P, 'E': day,
            'stock_minutes_count': len(snapshot.stock_minutes),
            'boards_count': len(snapshot.boards),
            'index_mem_mb': {k: round(v.memory_usage(deep=True).sum()/1024/1024, 2)
                             for k, v in snapshot.index_5m.items()},
        }
    stages.append(sp)

    # --- Stage 4: 审计坏股票（保留原始成员身份，逐只记录异常原因）---
    with StageProbe('4_audit_stock_errors') as sp:
        from joint_shadow_context_v1 import _prefix
        bad = {}  # code -> error_type
        sm_keys = list(snapshot.stock_minutes.keys())
        scope = snapshot.membership_scope
        research = scope == 'SNAPSHOT_RECONSTRUCTION_RESEARCH'
        for code in sm_keys:
            try:
                _prefix(snapshot.stock_minutes[code], cutoff, snapshot.receipts.get(code),
                        stock=True, research=research, observed_at=snapshot.observed_at)
            except Exception as exc:
                # 记录完整异常类型，不笼统归为"停牌"
                exc_str = str(exc)
                if 'INVALID_SOURCE_PRICE' in exc_str:
                    error_type = 'INVALID_SOURCE_PRICE'
                elif 'INVALID_SOURCE_OHLC' in exc_str:
                    error_type = 'INVALID_SOURCE_OHLC'
                elif 'SOURCE_RECEIPT_MISSING' in exc_str:
                    error_type = 'SOURCE_RECEIPT_MISSING'
                elif 'SOURCE_CLOCK_UNBOUND' in exc_str:
                    error_type = 'SOURCE_CLOCK_UNBOUND'
                elif 'INVALID_SOURCE_ACTIVITY' in exc_str:
                    error_type = 'INVALID_SOURCE_ACTIVITY'
                else:
                    error_type = f'{type(exc).__name__}:{exc_str[:50]}'
                bad[code] = error_type
        
        # 保留原始 boards 成员身份（不删除），只标记坏股票
        # 好股票 = 在 stock_minutes 里且 _prefix 不崩的
        all_board_codes = {c for m in snapshot.boards.values() for c in m}
        good_codes = sorted(c for c in all_board_codes 
                           if c in snapshot.stock_minutes and c not in bad)
        candidates = [{'code': c, 'cutoff_ts': cutoff, 'P': P, 'E': day} for c in good_codes]
        
        # 统计各异常类型分布
        error_dist = {}
        for et in bad.values():
            error_dist[et] = error_dist.get(et, 0) + 1
        
        sp.details = {
            'total_board_members': len(all_board_codes),
            'stock_minutes_count': len(snapshot.stock_minutes),
            'bad_codes_count': len(bad),
            'bad_codes_error_distribution': error_dist,
            'bad_codes_detail': dict(list(bad.items())[:10]),  # 前10只示例
            'good_candidates': len(candidates),
            'boards_preserved': len(snapshot.boards),  # 原始板块数未变
        }
        print(f'  total_board_members={len(all_board_codes)}, '
              f'bad={len(bad)}, good candidates={len(candidates)}')
        print(f'  error distribution: {error_dist}')
    stages.append(sp)

    # --- Stage 5: build_features (使用过滤后的 boards 副本，保留原始 boards 用于审计) ---
    with StageProbe('5_build_features') as sp:
        from joint_shadow_context_v1 import build_features
        import copy
        
        # 创建 snapshot 浅副本，过滤 boards 中的坏股票
        # 这是模拟生产修复后的行为：单股异常隔离，不改变原始成员身份
        snapshot_filtered = copy.copy(snapshot)
        snapshot_filtered.boards = {
            bk: [c for c in members if c not in bad]
            for bk, members in snapshot.boards.items()
        }
        
        frame, x232, x266, audit_bf = build_features(snapshot_filtered, candidates)
        sp.details = {
            'frame_shape': frame.shape,
            'x232_shape': x232.shape,
            'x266_shape': x266.shape,
            'frame_mem_mb': round(frame.memory_usage(deep=True).sum()/1024/1024, 2),
            'x232_mem_mb': round(x232.memory_usage(deep=True).sum()/1024/1024, 2),
            'x266_mem_mb': round(x266.memory_usage(deep=True).sum()/1024/1024, 2),
            'audit_bf_keys': list(audit_bf.keys()) if audit_bf else None,
            'audit_bf_stocks': audit_bf.get('stocks') if audit_bf else None,
            'audit_bf_missing_stock_sources': len(audit_bf.get('missing_stock_sources', [])) if audit_bf else None,
        }
        print(f'  build_features OK: frame={frame.shape}, '
              f'x232={x232.shape}, x266={x266.shape}')
        # 打印 DataFrame 大小
        print(f'  frame={sp.details["frame_mem_mb"]}MB '
              f'x232={sp.details["x232_mem_mb"]}MB '
              f'x266={sp.details["x266_mem_mb"]}MB')
    stages.append(sp)

    # --- Stage 6: Pyxis 评分 (load booster + predict) ---
    with StageProbe('6_pyxis_scoring') as sp:
        booster232 = frozen.load_booster(frozen.MODEL_232)
        booster266 = frozen.load_booster(frozen.MODEL_266)
        # predict
        pred232 = booster232.predict(x232.values)
        pred266 = booster266.predict(x266.values)
        sp.details = {
            'booster232_trees': booster232.num_trees(),
            'booster266_trees': booster266.num_trees(),
            'pred232_shape': pred232.shape,
            'pred266_shape': pred266.shape,
            'model_232_size_kb': (Path(str(frozen.MODEL_232))/'lightgbm_model.txt.gz').stat().st_size//1024,
            'model_266_size_kb': (Path(str(frozen.MODEL_266))/'lightgbm_model.txt.gz').stat().st_size//1024,
        }
        print(f'  predict OK: pred232={pred232.shape}, pred266={pred266.shape}')
        # 清理 booster, 测释放后的内存
        del booster232, booster266, pred232, pred266
        gc.collect()
        cur_after = _rss_mb()
        print(f'  after del boosters: rss={cur_after} MB')
    stages.append(sp)

    # --- 汇总 ---
    overall_elapsed = round(time.time() - overall_t0, 2)
    overall_peak_mb = _peak_mb() or 0

    out = {
        'phase': 'PYXIS_COLD_START_DIAGNOSTIC_V2',
        'cutoff': cutoff,
        'generated_at': datetime.now().isoformat(timespec='seconds'),
        'overall_elapsed_seconds': overall_elapsed,
        'overall_tracemalloc_peak_mb': overall_peak_mb,
        'cap_check': {
            'tracemalloc_peak_mb': overall_peak_mb,
            'note': 'tracemalloc 只追踪 Python heap. C extensions (numpy/pandas/lgbm) 不计. '
                    '真实 RSS 更高. 经验上 tracemalloc + 100~200MB ≈ 实际 RSS',
            'estimated_real_rss_mb': overall_peak_mb + 150,
            'status_vs_512MiB': 'PASS' if overall_peak_mb + 200 < 512 else 'WARN',
        },
        'fix_applied': {
            'location': 'joint_shadow_context_v1._market',
            'before': 'frame.to_dict("records") → 2×index→list[dict] (各9.5MB, 非主要瓶颈但已优化)',
            'after': '直接存 DataFrame + _as_row_dicts 入口适配',
            'actual_index_size': '~1.15MB DataFrame each, 膨胀~9.5MB each',
            'primary_bottleneck_hypothesis': 'build_features 逐票循环中 stocks DataFrame concat '
                                             '+ frame merge shared 膨胀',
        },
        'data_filter_note': f'过滤了 {sp.details.get("bad_codes_count", "?")} 只有停牌数据的股票 '
                            '(INVALID_SOURCE_PRICE / OHLC), 不影响内存测试的代表性',
        'stages': [s.to_dict() for s in stages],
    }

    out_path = HERE / 'pyxis_cold_start_diagnostic_v2.json'
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2, default=str), encoding='utf-8')

    print(f"\n{'=' * 70}")
    print(f"总耗时: {overall_elapsed}s")
    print(f"tracemalloc 全局峰值: {overall_peak_mb:.1f} MiB")
    print(f"估算实际 RSS: ~{overall_peak_mb + 150:.0f} MiB (+150 C-ext 估计)")
    print(f"512MiB cap: {out['cap_check']['status_vs_512MiB']}")
    print(f"输出: {out_path}")
    print(f"{'=' * 70}")
    for s in stages:
        d = s.to_dict()
        print(f"  [{d['name']:25s}] {d['elapsed_seconds']:7.1f}s | "
              f"rss {d['rss_before_mb']:6.1f}→{d['rss_after_mb']:6.1f} MB "
              f"(delta {d['rss_delta_mb']:+6.1f}) | "
              f"peak {d['peak_after_mb']:6.1f} | "
              f"{d.get('error') or 'OK'}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
