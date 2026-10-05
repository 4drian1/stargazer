"""c4_shadow.py — C4 前瞻影子评分（materiality_head_v1 FROZEN）。

边界（C3→C4 / C4_SHADOW.md / scoring_domain.py）：
- 只输出概率/校准状态/缺失警告/evidence_ids，不直接改变交易；不回写成熟标签。
- 评分域（C4_SHADOW_SCORING_DOMAIN_V1）刻意不使用 label_valid /
  corp_action_within_horizon / later_realized_label（标签侧信息）。
- score_event 加载真实模型前，必须先通过 feature_parity_gate（见 artifacts/feature_parity_passed.json）。
"""
from __future__ import annotations

import datetime as dt
import json
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from corvus.materiality_v1.materiality_features_v1 import build_features, MODEL_FEATURES
from corvus.materiality_v1.scoring_domain import is_scoring_eligible, CONTRACT

ART = Path(__file__).resolve().parent / "corvus" / "materiality_v1" / "artifacts"
MATERIALITY_MODEL_VERSION = "1.0.0"  # 与 model_manifest.json 一致；score_event 实际以 manifest 为准


@dataclass
class CorvusInformationSnapshot:
    event_cluster_id: str
    available_at: int
    available_at_precision: str
    event_tier: str
    p_materiality: Optional[float] = None
    materiality_model_version: str = MATERIALITY_MODEL_VERSION
    calibration_status: str = "HISTORICAL_OOT_VALIDATED_RAW_PROBA"
    missing_warnings: list[str] = field(default_factory=list)
    entity_ids: list[str] = field(default_factory=list)
    event_family: Optional[str] = None
    evidence_ids: list[str] = field(default_factory=list)
    feature_snapshot_id: Optional[str] = None
    feature_asof: Optional[int] = None
    created_ts: str = field(
        default_factory=lambda: dt.datetime.now().isoformat(timespec="seconds"))

    def to_record(self) -> dict:
        return asdict(self)


class MaterialityModel:
    """加载冻结模型（3 seed 平均）+ 特征中位数填充；强制 parity gate 先过。"""

    def __init__(self, manifest_path: str = str(ART / "model_manifest.json")):
        self._manifest_path = Path(manifest_path)
        self.manifest = json.loads(self._manifest_path.read_text(encoding="utf-8"))
        self.version = self.manifest["version"]
        medians_path = self._resolve(self.manifest["feature_medians_file"])
        self.medians = {k: v for k, v in json.loads(
            medians_path.read_text(encoding="utf-8")).items()}
        import lightgbm as lgb
        self.boosters = [lgb.Booster(model_file=str(self._resolve(p)))
                         for p in self.manifest["model_files"]]
        flag = ART / "feature_parity_passed.json"
        if not flag.exists():
            raise RuntimeError(
                "Feature Parity Gate 未通过：先运行 "
                "python corvus/materiality_v1/feature_parity_gate.py "
                "（C4 禁止在未验证特征一致性时加载模型）。")
        # MODEL HASH VERIFY：manifest 中登记的 artifact hash 与本地不一致 → 拒绝启动
        self._verify_hashes()

    def _resolve(self, p) -> Path:
        """artifact 路径可移植解析（DEPLOYMENT_PORTABILITY_V1）。

        冻结 manifest 可能在另一台机器生成（绝对路径在本地不存在）：
        1) 登记路径本身存在 → 用之；
        2) 相对路径 → 按 manifest 所在目录解析；
        3) 绝对路径不存在 → 按 manifest 所在目录下同文件名解析。
        都失败 → RuntimeError（模型完整性拒绝启动，与 hash 校验同语义）。
        """
        pp = Path(str(p))
        base = self._manifest_path.resolve().parent
        cands = [pp] if pp.is_absolute() else []
        # feature_contract.json 等登记在模块目录（artifacts 上级）的文件需多一级回退
        cands += [base / pp, base / pp.name, base.parent / pp.name]
        for c in cands:
            if c.exists():
                return c
        raise RuntimeError(f"artifact 不可解析: {p}（试过: {[str(c) for c in cands]}）")

    def _verify_hashes(self):
        import hashlib, json as _json
        from pathlib import Path as _P

        def sha(p):
            h = hashlib.sha256()
            with open(p, "rb") as f:
                for c in iter(lambda: f.read(1 << 20), b""):
                    h.update(c)
            return h.hexdigest()

        mf = self.manifest
        for p in mf["model_files"]:
            b = _P(p).name
            exp = mf.get("model_sha256", {}).get(b)
            if exp and sha(self._resolve(p)) != exp:
                raise RuntimeError(
                    f"MODEL HASH VERIFY FAILED: {b} 与 manifest 不一致，scorer 拒绝启动")
        fm = mf.get("feature_medians_file")
        if fm and mf.get("feature_medians_sha256") and \
           sha(self._resolve(fm)) != mf["feature_medians_sha256"]:
            raise RuntimeError("feature_medians hash 与 manifest 不一致，scorer 拒绝启动")
        fc = mf.get("feature_contract_file")
        if fc and mf.get("feature_contract_sha256") and \
           sha(self._resolve(fc)) != mf["feature_contract_sha256"]:
            raise RuntimeError("feature_contract hash 与 manifest 不一致，scorer 拒绝启动")
        if mf.get("config_sha256"):
            if _json.dumps(mf.get("config"), sort_keys=True).encode() and \
               hashlib.sha256(_json.dumps(mf.get("config"), sort_keys=True).encode()).hexdigest() \
               != mf["config_sha256"]:
                raise RuntimeError("config hash 与 manifest 不一致，scorer 拒绝启动")

    def predict_proba(self, X: pd.DataFrame):
        import numpy as np
        Xf = X[MODEL_FEATURES].fillna(pd.Series(self.medians)).values
        ps = [b.predict(Xf) for b in self.boosters]
        return np.mean(ps, axis=0)


_model_cache: dict = {}


def get_model():
    if "m" not in _model_cache:
        _model_cache["m"] = MaterialityModel()
    return _model_cache["m"]


def score_event(cluster_record: dict, *, out_dir="out", as_of_date=None,
                daily_provider=None, model=None):
    """对单条新事件打分。返回 (snapshot, diagnostics)。

    cluster_record 需含（均可用 available_at 前信息）：
      event_cluster_id / code / title / announce_date / label_start /
      n_announcements_in_cluster / sigma250 / gap_return / volume_shock /
      category / tier / attribution_quality
    """
    ev = pd.DataFrame([cluster_record])
    feat = build_features(ev, out_dir, with_labels=False, daily_provider=daily_provider)
    warnings: list[str] = []
    # 先评估域门禁：即便特征缺失也要显式给出 tier / entity / distribution / future 等拒绝原因
    feat_row = feat.iloc[0] if not feat.empty else None
    eligible, problems = is_scoring_eligible(cluster_record, feat_row, as_of_date=as_of_date)
    if not eligible:
        warnings += problems
        snap = _snapshot(cluster_record, p_materiality=None,
                        calibration_status="DOMAIN_INELIGIBLE",
                        missing_warnings=warnings)
        return snap, {"eligible": False, "problems": problems}
    if feat_row is None:
        # 域门禁通过但行情特征无法构造（如实体无行情源）→ 显式标记，不静默丢
        warnings.append("feature_extraction_failed")
        snap = _snapshot(cluster_record, p_materiality=None,
                        calibration_status="DOMAIN_OR_FEATURE_FAILED",
                        missing_warnings=warnings)
        return snap, {"eligible": False, "problems": warnings}

    model = model or get_model()
    p = model.predict_proba(feat)
    p_val = float(p[0]) if hasattr(p, "__len__") else float(p)
    snap = _snapshot(
        cluster_record, p_materiality=p_val, model_version=model.version,
        entity_ids=[str(cluster_record["code"])] if not pd.isna(cluster_record.get("code")) else [],
        calibration_status="HISTORICAL_OOT_VALIDATED_RAW_PROBA")
    return snap, {"eligible": True, "problems": []}


def _snapshot(cluster_record, *, p_materiality=None, model_version=MATERIALITY_MODEL_VERSION,
              entity_ids=None, calibration_status="HISTORICAL_OOT_VALIDATED_RAW_PROBA",
              missing_warnings=None):
    feature_asof = int(cluster_record.get("label_start") or cluster_record.get("announce_date", 0))
    return CorvusInformationSnapshot(
        event_cluster_id=cluster_record["event_cluster_id"],
        available_at=int(cluster_record.get("announce_date", 0)),
        available_at_precision="DATE_ONLY",
        event_tier=str(cluster_record.get("tier", "P0")),
        p_materiality=p_materiality,
        materiality_model_version=model_version,
        calibration_status=calibration_status,
        missing_warnings=missing_warnings or [],
        entity_ids=entity_ids or [],
        event_family=cluster_record.get("category"),
        evidence_ids=[],
        feature_asof=feature_asof,
    )


def build_snapshot(event_cluster_id, available_at, event_tier, *, entity_ids=None,
                   event_family=None, evidence_ids=None, feature_snapshot_id=None,
                   feature_asof=None, p_materiality=None,
                   calibration_status="HISTORICAL_OOT_VALIDATED_RAW_PROBA",
                   missing_warnings=None):
    return CorvusInformationSnapshot(
        event_cluster_id=event_cluster_id, available_at=available_at,
        available_at_precision="DATE_ONLY", event_tier=event_tier,
        p_materiality=p_materiality, calibration_status=calibration_status,
        missing_warnings=missing_warnings or [], entity_ids=entity_ids or [],
        event_family=event_family, evidence_ids=evidence_ids or [],
        feature_snapshot_id=feature_snapshot_id, feature_asof=feature_asof)
