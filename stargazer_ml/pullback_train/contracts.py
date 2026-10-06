# -*- coding: utf-8 -*-
"""统一结构化输出契约（为后续 LangGraph/外层 LLM 预留，现不引入任何框架）。

原则（用户定）：
- 模型数值输出 → Agent结构化质询 → 固定准入规则 → 人工最终决策；
- 框架只读取本结构，不侵入模型内部；
- 决策引擎独立于 LLM，"是否可买入"不写死在任何链里。
"""
from dataclasses import dataclass, field, asdict
from typing import List


@dataclass
class CandidateAssessment:
    candidate_id: str          # event_id（决策时点冻结的快照键）
    strategy_id: str           # 如 "A/adj_ready"、"B/b0"
    model_version: str         # 如 "a2_lgbm_k1.2@v3.3"
    opportunity_score: float   # 排序分（本链=EV）
    p_target: float
    p_stop: float
    p_timeout: float
    expected_value: float      # 净 EV（扣成本）
    evidence_ids: List[str] = field(default_factory=list)   # 数据快照/证据 id
    risk_flags: List[str] = field(default_factory=list)     # 如 sector_not_confirmed


@dataclass
class GateDecision:
    """固定准入规则的裁决结果（确定性，不经 LLM）。"""
    candidate_id: str
    gate: str                  # G0..G7
    ev_pass: bool
    sector_confirmed: bool
    slot_cap: int
    verdict: str               # BUY_CANDIDATE / WATCH / EMPTY_SLOT / REJECT
