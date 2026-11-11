"""Reuse verified, already-computed day inputs across plans at the same cutoff.

No warmup is shortened: cold values come from PrefixFeatureCache's full-history
kernel. Cached day rows/returns are its output, not newly seeded indicators.
"""
import gzip
import hashlib
import io
import json
from pathlib import Path

import numpy as np
import pandas as pd

from joint_shadow_context_v1 import RiskSnapshot
from live_postclose_source_v1 import LocalCapture, CapturedMinutes, disk_limit
from stock_prefix_cache_v1 import PrefixFeatureCache

HERE = Path(__file__).resolve().parent


def sha(body):
    return hashlib.sha256(body).hexdigest()


def encoded(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()


def kernel():
    # No display/plan IDs. Keep mathematical kernels, adapter, cache and models
    # bound so a runtime or feature change cannot reuse an incompatible result.
    import rebuild_features_v1 as frozen
    names = ['shared_risk_context_v1.py', 'stock_prefix_cache_v1.py',
             'joint_shadow_context_v1.py', 'build_joint_risk_features_v1.py',
             'live_postclose_source_v1.py', 'intraday_relative_stall_v1.py',
             'intraday_relative_stall_v2.py', 'build_multiboard_full_features_v1.py',
             'multiboard_risk_features_v1.py', 'build_joint_candidate_context_v1.py',
             'build_common_candidate_sidecar_v1.py', 'conditional_board_patch_v1.py',
             'joint_fit_preparation_v1.py', 'rebuild_features_v1.py']
    spec = [sha((HERE / n).read_bytes()) for n in names]
    # Read inference artifacts only, never training tables or predictions.
    for folder in (frozen.MODEL_232, frozen.MODEL_266):
        for name in ('lightgbm_model.txt.gz', 'lightgbm_preprocessing.json'):
            p = Path(folder) / name
            spec.append([name, sha(p.read_bytes())])
    spec.append(sha(frozen.PREPROC_CONFIG.read_bytes()))
    return sha(encoded(spec))


def frame_bytes(frame):
    sink = io.BytesIO()
    frame.to_parquet(sink, index=False, compression='zstd')
    return sink.getvalue()


def index_identity(frame):
    columns = ['date', 'open', 'high', 'low', 'close', 'volume', 'amount']
    selected = frame[[c for c in columns if c in frame]].reset_index(drop=True)
    return sha(pd.util.hash_pandas_object(selected, index=False).values.tobytes())


class SharedPreparedContext:
    def __init__(self, snapshot, capture):
        self.snapshot, self.capture = snapshot, capture
        self.version = kernel()
        self.codes = sorted(getattr(snapshot.stock_minutes,'entries',snapshot.stock_minutes))
        self.day_rows, self.day_returns = {}, {}
        self.delegate = None
        self.audit = dict(prepared_hits=0, prepared_misses=0, committed=False)
        self.loaded = getattr(snapshot, '_prepared_context', None)
        if self.loaded is None:
            self.delegate = PrefixFeatureCache(snapshot, capture)

    def stock(self, code, days, cutoff):
        if self.loaded is not None:
            self.audit['prepared_hits'] += 1
            rows, native = self.loaded
            return rows[code].copy(), {f: a[code].copy() for f, a in native.items()}
        rows, native = self.delegate.stock(code, days, cutoff)
        self.day_rows[code] = rows[rows.cutoff_index // 16 == len(days) - 1].copy()
        self.day_returns[code] = {f: native[f][-(240 // f):].copy() for f in (5, 15)}
        self.audit['prepared_misses'] += 1
        return rows, native

    def commit(self):
        if self.loaded is not None or set(self.day_rows) != set(self.codes):
            return
        snapshot, capture = self.snapshot, self.capture
        manifest = capture.output / 'source_manifest.json'
        if not manifest.is_file():
            self.audit['commit_skipped'] = 'NO_CAPTURE_MANIFEST'
            return
        source = capture.read(manifest)
        source_key = sha(source)
        folder = capture.budget_root / 'shared_context_v1' / str(snapshot.cutoff) / (self.version + '_' + source_key[:16])
        if (folder / 'complete.json.gz').exists():
            return
        stocks = pd.concat([self.day_rows[c] for c in self.codes], ignore_index=True)
        native = io.BytesIO()
        np.savez_compressed(native, **{str(f): np.stack([self.day_returns[c][f] for c in self.codes]) for f in (5, 15)})
        bodies = {'stocks.parquet': frame_bytes(stocks), 'native.npz': native.getvalue(), 'source_manifest.json': source}
        for symbol, frame in snapshot.index_5m.items():
            bodies[symbol + '.parquet'] = frame_bytes(frame)
        meta = dict(cutoff=snapshot.cutoff, available_at=snapshot.available_at, days=list(map(int, snapshot.days)),
                    boards=snapshot.boards, board_order=list(snapshot.boards), receipts=snapshot.receipts, membership_known_at=snapshot.membership_known_at,
                    membership_scope=snapshot.membership_scope, observed_at=snapshot.observed_at,
                    codes=self.codes, kernel=self.version, refs={n: sha(b) for n, b in bodies.items()},
                    index_identity={s: index_identity(f) for s, f in snapshot.index_5m.items()})
        packed = gzip.compress(encoded(meta), mtime=0)
        extra = sum(map(len, bodies.values())) + len(packed)
        if capture.disk_bytes + extra + 2 * 1024**2 > capture.disk_cap:
            self.audit['commit_skipped'] = 'RUNTIME_DISK_BUDGET'
            return
        folder.mkdir(parents=True, exist_ok=True)
        # An interrupted context is retained, never treated as complete.
        if any(folder.iterdir()):
            self.audit['commit_skipped'] = 'INCOMPLETE_CONTEXT_RETAINED'
            return
        for name, body in bodies.items():
            with (folder / name).open('xb') as f:
                f.write(body)
        with (folder / 'complete.json.gz').open('xb') as f:
            f.write(packed)
        capture.disk_bytes += extra
        self.audit.update(committed=True, bytes_written=extra)
        if self.delegate is not None:
            self.audit['prefix_cache'] = self.delegate.audit


def prepared_snapshot(runtime, output, cutoff, codes, *, index_source=None):
    """Read immutable inputs once per cutoff; a new cutoff never reuses old data."""
    root = Path(runtime) / 'shared_context_v1' / str(cutoff)
    version = kernel()
    for marker in sorted(root.glob('*/complete.json.gz'), key=lambda p: p.stat().st_mtime_ns, reverse=True):
        try:
            c = LocalCapture(output, budget_root=runtime)
            meta = json.loads(gzip.decompress(c.read(marker)))
            if meta['kernel'] != version or meta['cutoff'] != cutoff or not set(codes) <= set(meta['codes']):
                continue
            # Verify every prepared artifact, not the entire original minute seed.
            bodies = {n: c.read(marker.parent / n, h) for n, h in meta['refs'].items()}
            manifest = json.loads(bodies['source_manifest.json'])
            binding = manifest.get('index_source_binding')
            if index_source and binding != dict(path=str(index_source[0]), sha256=index_source[1]):
                continue
            if not index_source:
                # The public mirror may have advanced. Compare only the actual
                # prefix used by this decision, including late price revisions.
                matches = True
                for symbol, name in [('000001.SH', 'sse'), ('399006.SZ', 'cyb')]:
                    live = pd.read_parquet(HERE / 'index_minutes_v1' / (name + '_5min.parquet')).rename(columns={'ts': 'date', 'raw_volume': 'volume'})
                    live = live[live.date <= cutoff]
                    matches &= index_identity(live) == meta['index_identity'][symbol]
                if not matches:
                    continue
            indices = {s: pd.read_parquet(io.BytesIO(bodies[s + '.parquet'])) for s in ('000001.SH', '399006.SZ')}
            stocks = pd.read_parquet(io.BytesIO(bodies['stocks.parquet']))
            rows = {str(k): v.reset_index(drop=True) for k, v in stocks.groupby('code', sort=False)}
            if not set(rows) <= set(meta['codes']):
                raise ValueError('PREPARED_UNKNOWN_STOCK')
            for code in meta['codes']:
                rows.setdefault(code, stocks.iloc[:0].copy())
            with np.load(io.BytesIO(bodies['native.npz']), allow_pickle=False) as native:
                arrays = {}
                for f in (5, 15):
                    values = native[str(f)]
                    if values.shape != (len(meta['codes']), 240 // f):
                        raise ValueError('PREPARED_NATIVE_SHAPE')
                    arrays[f] = {code: values[i].copy() for i, code in enumerate(meta['codes'])}
            if set(rows) != set(meta['codes']) or any(len(a) != 240 // f for f, values in arrays.items() for a in values.values()):
                raise ValueError('PREPARED_SHAPE')
            c.save('source_manifest.json', bodies['source_manifest.json'])
            if set(meta['board_order']) != set(meta['boards']) or len(meta['board_order']) != len(meta['boards']):
                raise ValueError('PREPARED_BOARD_ORDER')
            boards={b:meta['boards'][b] for b in meta['board_order']}
            snapshot = RiskSnapshot(meta['cutoff'], meta['available_at'], meta['days'],
                CapturedMinutes(manifest['entries'], c, cutoff), indices, boards, meta['receipts'],
                meta['membership_known_at'], meta['membership_scope'], observed_at=meta['observed_at'])
            snapshot._prepared_context = (rows, arrays)
            return snapshot, c, manifest['failures']
        except (ValueError, OSError, KeyError, EOFError):
            # Corrupt/incompatible derived cache is retained; caller captures
            # fresh sources. It is never reported as complete risk evidence.
            continue
    return None


def produce_shared(day, codes, output, progress, *, cutoff=None, index_source=None, feed=None):
    from live_postclose_source_v1 import produce_live
    cutoff = cutoff or int(day) * 1000000 + 150000
    if feed == 'realtime':
        # Production intraday lane (#18.188/W1): realtime current-day minutes +
        # local TDX history + public index mirror; never falls back to 7899.
        # Fresh capture per cutoff: prepared contexts bound to legacy receipts
        # are not reused across feeds.
        from intraday_risk_feed_v1 import RealtimeCapture
        return produce_live(day, codes, output, progress, cutoff=cutoff,
                            index_source=index_source, capture_factory=RealtimeCapture)
    cached = prepared_snapshot(Path(output).parent, output, cutoff, codes, index_source=index_source)
    if cached is not None:
        progress(stage='REUSING_PREPARED_CONTEXT', done=len(codes), total=len(codes))
        return cached
    return produce_live(day, codes, output, progress, cutoff=cutoff, index_source=index_source)
