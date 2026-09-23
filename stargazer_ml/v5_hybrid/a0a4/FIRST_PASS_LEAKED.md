# A0A4 FIRST PASS — LEAKED / INVALID

标记时间: 2026-09-19 (红队审计)
状态: A0A4_FIRST_PASS_LEAKED

## 根因
训练 Harness 按 tier 选特征, s0 与 timing 共用同一 `feats`
(a0a4_train.py `train_tier` 在 track 循环外只算一次 `feats`),
且特征行 as-of 锁在 D 日 15:00
(latr_builder.py: `SNAPSHOT_MINUTE = 150000`, `snapshot_time = d_date 150000`),
晚于 S0 锚点(D 日 open) 与 Timing 信号(09:30)/成交(09:31)。
A3/A4 的 D 日盘中特征因此注入 S0 轨(role leakage),
并对 Timing 轨构成前视污染(temporal look-ahead)。

## 以下结果不得作为性能证据 (INVALID_FOR_PERFORMANCE_INFERENCE)
- S0 A3 IC = +0.210
- S0 A4 IC = +0.320
- Timing A3 IC = +0.395
- Timing A4 IC = +0.561
- A4 账本年化 +124%
- v1_rejected_v11_accepted: IC +0.569 / top +4.78%

## 保留策略
第一轮产物全部保留 (a1..a4/, a0/), 仅作审计对照; 不做删除。
最终以 causal rebuild 后的 clean rerun 为准。

## 修复路线 (用户 2026-09-19 批准)
Track-aware Feature Roles
  + Event-driven Timing Samples (FIRST_MODEL_SCORE_REQUEST)
  + Causal snapshot at S
  + next-legal-minute Execution
  + full-row timestamp lineage
  -> 3 新硬 Gate (MODEL_ROLE / TRAINING_TIME_CAUSALITY / TIMING_TRIGGER_SEMANTICS)
  -> 重冻结 (SUPERSEDED 旧 FINAL_FROZEN, 不静默复用)
  -> 从零 clean A0-A4

## S0 正确口径 (修复后)
- S0-A1 = P_CLOSE Core         (有效)
- S0-A2 = P_CLOSE Core + Frozen Zone/Structure (有效)
- S0-A3 = N/A (not eligible for S0 role)
- S0-A4 = N/A (not eligible for S0 role)
