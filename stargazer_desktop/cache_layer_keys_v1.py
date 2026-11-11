"""Per-layer cache identity (#18.199): data caches bind only the dependencies
that actually change their math. Source-code release fingerprints keep
protecting ALL production code; these keys decide when derived data may be
reused.

Layers:
  math    indicator/feature computation semantics (prefix arrays, stall, joint
          context). UI / prompt / push-text / capture-path changes must NOT
          invalidate these.
  capture data ingestion & parsing semantics (capture classes, TDX reader,
          realtime lane). Changes here invalidate captured bytes' meaning and
          therefore downstream evidence, via the entry's own source shas.
  model   scoring: booster weights, preprocessing, feature order.

A cache entry may also bind ``extra`` (e.g. source byte shas), which is the
fine-grained invalidation path for parsing changes.
"""
import hashlib
import json
from pathlib import Path

RISK = Path(__file__).resolve().parent.parent / 'stargazer_ml/v5_hybrid/risk_r0_v1'

LAYER_FILES = {
    'math': (
        'stock_prefix_cache_v1.py',
        'build_joint_risk_features_v1.py',
        'intraday_relative_stall_v2.py',
        'joint_shadow_context_v1.py',
    ),
    'capture': (
        'live_postclose_source_v1.py',
        'intraday_risk_feed_v1.py',
        'tdx_native_history_v1.py',
    ),
    'model': (
        'rebuild_features_v1.py',
    ),
}


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def layer_key(layer: str, extra=None) -> str:
    """Digest of the layer's dependency files plus caller-bound extras.

    ``extra`` must be JSON-serializable; typical use: source byte signatures,
    calendar version, membership version."""
    if layer not in LAYER_FILES:
        raise ValueError('UNKNOWN_CACHE_LAYER:' + layer)
    body = {name: _sha(RISK / name) for name in LAYER_FILES[layer]}
    body['_layer'] = layer
    body['_extra'] = extra
    return hashlib.sha256(
        json.dumps(body, sort_keys=True, ensure_ascii=False, default=str).encode()).hexdigest()


def math_key(extra=None) -> str:
    return layer_key('math', extra)


def capture_key(extra=None) -> str:
    return layer_key('capture', extra)


def model_key(extra=None):
    """Scoring identity: preprocessing/feature code PLUS the actual model
    artifacts (booster weights, preprocessing config, feature order source).
    Unavailable artifacts are recorded, never silently skipped."""
    artifacts = {}
    try:
        import rebuild_features_v1 as frozen
        from pathlib import Path as _P
        for label, folder in (('model_232', frozen.MODEL_232), ('model_266', frozen.MODEL_266)):
            for name in ('lightgbm_model.txt.gz', 'lightgbm_preprocessing.json'):
                p = _P(folder) / name
                artifacts[label + '/' + name] = _sha(p) if p.is_file() else 'MISSING'
        pc = _P(frozen.PREPROC_CONFIG)
        artifacts['preproc_config'] = _sha(pc) if pc.is_file() else 'MISSING'
    except Exception as exc:  # research chain unavailable in some contexts
        artifacts['unavailable'] = type(exc).__name__
    merged = dict(extra or {})
    merged['_model_artifacts'] = artifacts
    return layer_key('model', merged)
