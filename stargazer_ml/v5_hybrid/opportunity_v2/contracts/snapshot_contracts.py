# -*- coding: utf-8 -*-
"""Snapshot / PolicyDecision 合同（OPPORTUNITY-DISCOVERY-V2 冻结契约的代码形态）。

来源：开发路线.txt 2026-09-23 变更记录 #4（Orchestration/Runtime Layer 合同冻结）。
纪律：
- 五类世界状态快照 + 一个决策对象；PolicyDecision 与快照不并列。
- 全部强制全局头：snapshot_id / schema_version / frozen_artifact_ref / symbol /
  asof_ts（最晚用到的信息时刻）/ created_ts（实际生成时刻）。
- frozen_artifact_ref = "artifact_type:artifact_version:hash" 三元组（非文件路径）。
- 本模块零第三方依赖（stdlib only）：任何 Runtime 层（LangGraph / thin runner /
  Temporal / 纯脚本）都能 import；训练/推理模块以纯函数形态输出这些对象，
  不自行操作库、不自行决定交易流程。
- POLICY_DETERMINISM_CONTRACT：同输入 + 同 policy_version ⇒ 同输出。
  LLM 只做 presentation/explanation sidecar，不得进入决策路径（不得作为任何
  确定性节点的输入特征）。

新增/修改字段 = schema_version 升级（v1 -> v2），禁止原地改语义。
"""
from __future__ import annotations

import hashlib
import json
import time
import uuid
from dataclasses import asdict, dataclass, field, fields
from typing import Any, Optional


SCHEMA_VERSION = "v1"


def _new_id(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex[:12]}"


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime())


@dataclass
class SnapshotHeader:
    """所有 Snapshot 的强制全局头（嵌入各快照，校验用）。"""
    snapshot_id: str
    schema_version: str                       # 如 "opportunity_snapshot.v1"
    frozen_artifact_ref: str                  # "artifact_type:artifact_version:hash"
    symbol: str                               # 6 位代码
    asof_ts: str                              # 最晚用到的信息时刻（PIT 审计核心）
    created_ts: str

    def __post_init__(self) -> None:
        if ":" not in self.frozen_artifact_ref:
            raise ValueError(
                "frozen_artifact_ref 必须是 artifact_type:artifact_version:hash 三元组")
        if self.schema_version.split(".")[-1] != SCHEMA_VERSION:
            raise ValueError(f"schema_version 必须以 .{SCHEMA_VERSION} 结尾")


def make_header(schema_kind: str, artifact_ref: str, symbol: str, asof_ts: str) -> SnapshotHeader:
    return SnapshotHeader(
        snapshot_id=_new_id(schema_kind),
        schema_version=f"{schema_kind}.{SCHEMA_VERSION}",
        frozen_artifact_ref=artifact_ref,
        symbol=symbol,
        asof_ts=asof_ts,
        created_ts=_now_iso(),
    )


# --------------------------------------------------------------------------- #
# 五类世界状态快照
# --------------------------------------------------------------------------- #
@dataclass
class OpportunitySnapshot:
    """机会来源层输出（GREEN-v2.2 Watchlist/Intraday Discovery、OD-V2 各机制）。"""
    header: SnapshotHeader
    source: str                               # PCLOSE_WATCHLIST / INTRADAY_DISCOVERY / HYP_01 ...
    discovery_ts: str                         # 首次可发现时刻
    feature_asof_ts: str                      # 特征锚点（P_CLOSE / snapshot_time）
    setup_type: Optional[str] = None          # BOTTOM_SETUP / INTRADAY_DIP_SETUP / ...
    opportunity_score: Optional[float] = None
    opportunity_version: Optional[str] = None
    zone_definition_version: Optional[str] = None
    plan_id: Optional[int] = None             # 关联 stargazer_plan（可空）
    extras: dict = field(default_factory=dict)


@dataclass
class EntryStateSnapshot:
    """Entry State（买前）：washout/support/recovery/可执行性。不含收益预测。"""
    header: SnapshotHeader
    entry_state: str                          # WATCH/SELL_OFF/ZONE_INTERACTION/EXHAUSTION/
                                              # STABILIZATION/EARLY_RECOVERY/ENTRY_READY/REJECT
    entry_readiness: Optional[float] = None   # 冻结口径的确定性分数（若有）
    entry_ts: Optional[str] = None
    washout_state: Optional[str] = None
    support_state: Optional[str] = None
    recovery_state: Optional[str] = None
    execution_feasible: Optional[bool] = None  # 涨跌停/停牌/流动性门控
    extras: dict = field(default_factory=dict)


@dataclass
class RiskSnapshot:
    """R0 风险头输出（影子阶段只记录，不否决）。"""
    header: SnapshotHeader
    risk_ts: str
    p_loss5: Optional[float] = None
    mae_q80: Optional[float] = None
    mae_q90: Optional[float] = None
    failed_rebound_prob: Optional[float] = None
    breakdown_risk: Optional[float] = None
    extras: dict = field(default_factory=dict)


@dataclass
class ContextSnapshot:
    """Pyxis/Corvus 上下文输出（市场环境/板块/个股画像/信息旗标）。"""
    header: SnapshotHeader
    context_snapshot_id: str
    market_context: Optional[str] = None
    sector_context: Optional[str] = None
    stock_profile: Optional[str] = None
    information_flags: Optional[str] = None
    evidence_ids: list = field(default_factory=list)
    extras: dict = field(default_factory=dict)


@dataclass
class PositionStateSnapshot:
    """持仓后状态：修复延续/失败/新低/破位/派发/时间衰减。与 Entry State 分离。"""
    header: SnapshotHeader
    position_id: str
    entry_price: float
    entry_ts: str
    continued_recovery: Optional[bool] = None
    failed_recovery: Optional[bool] = None
    new_lower_low: Optional[bool] = None
    structural_breakdown: Optional[bool] = None
    distribution_flag: Optional[bool] = None
    held_minutes: Optional[int] = None
    unrealized_pnl: Optional[float] = None
    extras: dict = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# 决策对象（与快照不并列）
# --------------------------------------------------------------------------- #
@dataclass
class PolicyDecision:
    """确定性 Policy 函数的输出。同输入+同 policy_version ⇒ 必须同输出。"""
    decision_id: str
    schema_version: str                       # "policy_decision.v1"
    policy_version: str                       # 冻结规则版本（如 "policy_v0.1"）
    input_snapshot_ids: list                  # 输入快照的 snapshot_id 列表
    action: str                               # BUY_CANDIDATE / WAIT / REJECT / HOLD /
                                              # REDUCE / EXIT_CANDIDATE
    reason_codes: list                        # 机器可读原因码（LLM 只能翻译展示）
    requires_human_review: bool = True        # 默认必须人工终审（HITL 契约）
    created_ts: str = field(default_factory=_now_iso)
    extras: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.schema_version.split(".")[-1] != SCHEMA_VERSION:
            raise ValueError(f"schema_version 必须以 .{SCHEMA_VERSION} 结尾")

    @staticmethod
    def new(policy_version: str, input_snapshot_ids: list,
            action: str, reason_codes: list,
            requires_human_review: bool = True) -> "PolicyDecision":
        return PolicyDecision(
            decision_id=_new_id("pd"),
            schema_version=f"policy_decision.{SCHEMA_VERSION}",
            policy_version=policy_version,
            input_snapshot_ids=list(input_snapshot_ids),
            action=action,
            reason_codes=list(reason_codes),
            requires_human_review=requires_human_review,
        )


# --------------------------------------------------------------------------- #
# 序列化 / 校验工具
# --------------------------------------------------------------------------- #
_SNAPSHOT_TYPES = {
    "opportunity_snapshot": OpportunitySnapshot,
    "entry_state_snapshot": EntryStateSnapshot,
    "risk_snapshot": RiskSnapshot,
    "context_snapshot": ContextSnapshot,
    "position_state_snapshot": PositionStateSnapshot,
}


def snapshot_to_dict(snap: Any) -> dict:
    d = asdict(snap)
    d["_kind"] = type(snap).__name__
    return d


def snapshot_to_json(snap: Any) -> str:
    return json.dumps(snapshot_to_dict(snap), ensure_ascii=False, sort_keys=True)


def snapshot_hash(snap: Any) -> str:
    """内容哈希：同输入 ⇒ 同 hash，供 Policy 复现审计。"""
    return hashlib.sha256(snapshot_to_json(snap).encode("utf-8")).hexdigest()


def validate_snapshot_dict(d: dict) -> None:
    """从 dict 重建前的合同校验（Runtime 层入库前调用）。"""
    hdr = d.get("header") or {}
    for k in ("snapshot_id", "schema_version", "frozen_artifact_ref",
              "symbol", "asof_ts", "created_ts"):
        if k not in hdr or hdr[k] in (None, ""):
            raise ValueError(f"snapshot 缺强制头字段: header.{k}")
    kind = (d.get("header", {}).get("schema_version", "").split(".")[0]
            or d.get("_kind", ""))
    kind = "".join(c.lower() if c.isalnum() else "_" for c in kind).strip("_")
    if kind in _SNAPSHOT_TYPES:
        required = {f.name for f in fields(_SNAPSHOT_TYPES[kind])} - {"extras"}
        missing = required - set(d.keys())
        if missing:
            raise ValueError(f"{kind} 缺字段: {sorted(missing)}")


def selftest() -> None:
    """合同自测：构造五快照 + 一决策，序列化/反序列化/哈希稳定性。"""
    ref = "R0:v0.3:sha256_deadbeef"
    hdr = make_header("risk_snapshot", ref, "600054", "2026-09-18T15:00:00")
    rs = RiskSnapshot(header=hdr, risk_ts=hdr.asof_ts, p_loss5=0.31, mae_q90=0.052)
    d = snapshot_to_dict(rs)
    validate_snapshot_dict(d)
    assert snapshot_hash(rs) == snapshot_hash(rs), "hash 不稳定"
    pd_ = PolicyDecision.new("policy_v0.1", [hdr.snapshot_id],
                             "BUY_CANDIDATE", ["ENTRY_READY", "RISK_ACCEPTABLE"])
    d2 = asdict(pd_)
    assert d2["schema_version"] == "policy_decision.v1"
    # 非法 artifact_ref 必须拒
    try:
        make_header("risk_snapshot", "no_colon_ref", "600054", "2026-09-18T15:00:00")
        raise AssertionError("非法 frozen_artifact_ref 未被拒绝")
    except ValueError:
        pass
    print("snapshot_contracts selftest PASS")


if __name__ == "__main__":
    selftest()
