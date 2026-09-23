# Gate 4 Status — SUPERSEDED / INVALIDATED_BY_CONTRACT_GAP

> **本结果不再作为正式 Gate4 通过依据。** 保留原始运行产物用于工程证据，
> 与 `L0_ORIGINAL` 同样遵循「不删除、不覆盖」纪律。

## 原始运行（已 SUPERSEDED）
- `dry_replay.py` 曾报 `PIPELINE_PASS / TIME_TRAVEL_PASS / NO_FUTURE_LEAK / ... = True`
- 40 样本端到端跑通，100 条 trace 已导出

## 为何被 SUPERSEDED（用户判定，2026-09-17）
Gate 4 当时验证的是**简化实现**，非冻结契约真实实现：
1. **候选源 ≠ 冻结契约**：`latr_builder` 自扫疑似 dip 日，而非 GREEN-WATCH-v1 双 Setup 候选流
2. **成交 ≠ 真实成交契约**：`actual_fill = open[D]`，未接 Execution Engine（signal→next legal fill→slippage/fee）
3. **Zone / 状态机不完整**：仅日线支撑 + 日内低点合并，缺完整 Minute State 机（WATCH→ZONE_TOUCH→ARMED→RECLAIM/BREAKDOWN）
4. **TRACE 覆盖不合格**：原 trace 为 `head(100)` 顺序抽，未分层覆盖 BOTTOM/INTRADAY_DIP/ZONE_TOUCH/RECLAIM/复权附近/INFERRED_HALT附近/开盘早期/午后

## 关联 Gate 状态（同次判定）
- Gate 1A = SUSPENDED（源身份分裂，P0 ① 已查清：单一 stockdb HTTP 满量到 20260917，矛盾为 parquet 衍生品 vs HTTP 源真相）
- Gate 3 = INVALIDATED（Manifest 内 DATA_EPOCH 错误，已修正为动态探测）
- A0-A4 = NOT STARTED

## 重新验收条件（见主方案）
Gate 1A 重扫 → Gate 3 重冻 → Gate 4 dry_replay（真实候选源 + 真实成交 + 完整 Zone/State + 分层 Trace）
→ 三项 PASS（PIPELINE / TIME_TRAVEL / TRACE）方算正式 Gate4 通过。
