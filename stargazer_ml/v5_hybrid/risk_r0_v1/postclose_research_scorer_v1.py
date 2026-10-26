"""Read-only D2 research inference. No fitting, fetching, or trading actions."""
import argparse
import hashlib
import json
import math
from datetime import datetime
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
CANDIDATE = 'R0_POSTCLOSE_D2_FIXED_RESEARCH_20261017'
DEFAULT_PACKAGE = HERE / 'postclose_scoring_handoff_20261017_v1/manifest.json'
MAX_INPUT = 4 * 1024**2
MAX_ROWS = 512


def encode(value):
    return json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True,
                      separators=(',', ':')).encode('utf-8')


def sha(body):
    return hashlib.sha256(body).hexdigest()


def strict_json(body):
    def reject(value):
        raise ValueError('NONFINITE_JSON:' + value)
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError('DUPLICATE_JSON_KEY:' + key)
            result[key] = value
        return result
    return json.loads(body, parse_constant=reject, object_pairs_hook=unique)


class ResearchScorer:
    def __init__(self, manifest=DEFAULT_PACKAGE):
        self.manifest_path = Path(manifest)
        self.manifest = strict_json(self.manifest_path.read_bytes())
        if self.manifest['candidate_id'] != CANDIDATE:
            raise ValueError('CANDIDATE_MISMATCH')
        self.verify_artifacts()
        self.fields = self.manifest['fields']
        if len(self.fields) != 131 or len(set(self.fields)) != 131:
            raise ValueError('INVALID_FIELD_REGISTRY')
        if sha(encode(self.fields)) != self.manifest['feature_schema_sha256']:
            raise ValueError('FIELD_SCHEMA_HASH_MISMATCH')
        artifacts = self.manifest['artifacts']
        state = strict_json(Path(artifacts['preprocessing']['path']).read_bytes())
        plan = strict_json(Path(artifacts['plan']['path']).read_bytes())
        if state['fields'] != self.fields or plan['fields'] != self.fields:
            raise ValueError('PREPROCESSING_SCHEMA_MISMATCH')
        contract = strict_json(Path(artifacts['contract']['path']).read_bytes())
        self.families = contract['family_checks']
        if any(f not in self.fields for fs in self.families.values() for f in fs):
            raise ValueError('FAMILY_SCHEMA_MISMATCH')
        self.shift = strict_json(Path(artifacts['report']['path']).read_bytes())['quantile_shift']
        if not math.isfinite(self.shift):
            raise ValueError('INVALID_CALIBRATION_SHIFT')
        import lightgbm as lgb
        self.model = lgb.Booster(model_file=artifacts['model']['path'])
        if self.model.num_feature() != 2 * len(self.fields):
            raise ValueError('MODEL_DIMENSION_MISMATCH')

    def verify_artifacts(self):
        for name, item in self.manifest['artifacts'].items():
            if sha(Path(item['path']).read_bytes()) != item['sha256']:
                raise ValueError('ARTIFACT_HASH_DRIFT:' + name)
        if self.manifest['artifacts']['runtime']['sha256'] != sha(Path(__file__).read_bytes()):
            raise ValueError('RUNTIME_HASH_DRIFT')

    def score(self, payload):
        self.verify_artifacts()
        if len(encode(payload)) > MAX_INPUT:
            raise ValueError('INPUT_BYTE_LIMIT')
        if (payload.get('candidate_id') != CANDIDATE or
                payload.get('feature_schema_sha256') != self.manifest['feature_schema_sha256']):
            raise ValueError('CANDIDATE_OR_SCHEMA_MISMATCH')
        if payload.get('usage_mode') != 'HISTORICAL_RECONSTRUCTION':
            raise ValueError('UNAUTHORIZED_USAGE_MODE')
        rows = payload.get('rows')
        if not isinstance(rows, list) or not 1 <= len(rows) <= MAX_ROWS:
            raise ValueError('ROW_LIMIT_OR_EMPTY')
        keys, results, vectors, positions = set(), [], [], []
        for row in rows:
            if not isinstance(row, dict):
                raise ValueError('INVALID_ROW_SHAPE')
            key = (row.get('code'), row.get('day'))
            if not isinstance(key[0], str) or type(key[1]) is not int:
                raise ValueError('INVALID_KEY_TYPE')
            if key in keys:
                raise ValueError('DUPLICATE_SAMPLE_KEY')
            keys.add(key)
            reasons, missing = [], []
            if key[0] not in self.manifest['research_codes']:
                reasons.append('OUTSIDE_43_CODE_RESEARCH_SCOPE')
            try:
                day = datetime.strptime(str(key[1]), '%Y%m%d')
                if day.strftime('%Y%m%d') != str(key[1]):
                    raise ValueError('INVALID_DAY')
                ts = row['feature_asof_ts']
                if type(ts) is not int or ts != key[1] * 1000000 + 150000:
                    raise ValueError('ASOF_NOT_COMPLETED_DAILY_ENDPOINT')
                # Historical engineering endpoint is not a fabricated arrival timestamp.
                if row.get('decision_available_ts') is not None:
                    raise ValueError('HISTORICAL_ARRIVAL_UNBOUND_EXPECT_NULL')
            except (KeyError, ValueError, TypeError):
                reasons.append('INVALID_OR_UNBOUND_TIME_SEMANTICS')
            if set(row) != {'code', 'day', 'feature_asof_ts', 'decision_available_ts', 'features'}:
                reasons.append('ROW_SCHEMA_MISMATCH')
            features = row.get('features')
            vector = []
            if not isinstance(features, dict) or set(features) != set(self.fields):
                reasons.append('FEATURE_KEYS_MISMATCH')
            else:
                for field in self.fields:
                    value = features[field]
                    if value is None:
                        missing.append(field)
                        vector.append(np.nan)
                    elif isinstance(value, (int, float)) and math.isfinite(value):
                        vector.append(float(value))
                    else:
                        reasons.append('INVALID_FEATURE:' + field)
                for family, fields in self.families.items():
                    available = [features[f] is not None for f in fields]
                    valid = any(available) if family == 'leader_any' else all(available)
                    if not valid:
                        reasons.append('MISSING_REQUIRED_FAMILY:' + family)
            result = dict(code=key[0], day=key[1], feature_asof_ts=row.get('feature_asof_ts'),
                          decision_available_ts=None, status='UNSCORABLE' if reasons else 'RESEARCH_SCORE',
                          q90_next_session_adverse_excursion=None, reasons=reasons, missing_fields=missing)
            results.append(result)
            if not reasons:
                vectors.append(vector)
                positions.append(len(results) - 1)
        if vectors:
            x = np.asarray(vectors, dtype=float)
            # Native NaNs and all 131 missing indicators exactly match D2 training.
            matrix = np.c_[x, np.isnan(x).astype(float)]
            pred = np.clip(self.model.predict(matrix, num_threads=1) + self.shift, 0, 1)
            if not np.isfinite(pred).all():
                raise ValueError('NONFINITE_PREDICTION')
            for index, value in zip(positions, pred):
                results[index]['q90_next_session_adverse_excursion'] = float(value)
        return dict(candidate_id=CANDIDATE, feature_schema_sha256=self.manifest['feature_schema_sha256'],
                    status='RESEARCH_ONLY', provenance_verified=False, independent_validation_passed=False,
                    deployment_authorized=False, input_content_sha256=sha(encode(payload)), rows=results,
                    warning='Content hash is not PIT/source acceptance. Q90 is not probability, PnL, or a trade instruction.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, default=DEFAULT_PACKAGE)
    args = parser.parse_args()
    try:
        with args.input.open('rb') as stream:
            body = stream.read(MAX_INPUT + 1)
        if len(body) > MAX_INPUT:
            raise ValueError('INPUT_BYTE_LIMIT')
        print(json.dumps(ResearchScorer(args.manifest).score(strict_json(body)), ensure_ascii=False, allow_nan=False))
    except (ValueError, KeyError, OSError, TypeError) as exc:
        print(json.dumps(dict(status='REJECTED', error=str(exc)), ensure_ascii=False))
        return 2
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
