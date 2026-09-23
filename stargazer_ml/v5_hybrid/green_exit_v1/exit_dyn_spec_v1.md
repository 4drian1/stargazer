# 动态退出标签规格 v1（冻结）

本文件是绿盘低吸「动态退出标签」的不可修改执行契约。所有标签引擎、验收测试、案例复放
都必须严格遵循本文件。任何参数修改需另开 v2 并显式说明差异。

## 0. 统一交易时序（不可修改）
- T 日收盘产生信号（候选在 T 日收盘冻结特征）。
- T+1 按真实开盘（或指定可成交价）买入；A 股 T+1，**买入当天不得卖出**。
- 最早可卖出日 = T+2。
- T+5 仅是最长观察期限，不是默认卖出日。
- 涨跌停、停牌、跳空、除权导致无法成交时，**延迟到下一可成交时点**；若到最后仍无法
  成交且处于跌停，标记 `EXIT_BLOCKED` 并继续持仓（不臆造成交）。
- 除权：`_raw_and_qfq` 的 `factor` 列（`raw_close/qfq_close`）用于识别除权跳变；
  入场日若因除权产生异常跳空，按实际开盘价成交（真实情况），不修正。

## 1. ATR_ref（入场前 5 日 ATR14 中位数）
- `TR_t = max(H_t - L_t, |H_t - C_{t-1}|, |L_t - C_{t-1}|)`
- `ATR14_t = mean(TR over t-13..t)`
- 取入场日(T+1)前 5 个交易日（T-4..T）的 `ATR14` 取**中位数** → `ATR_ref`
  （用中位数避免信号日前后突发大波动放大 ATR）。
- 若历史不足 14 日，`ATR_ref` 退化为可用窗口的 ATR 均值并打标 `atr_ref_short=True`。

## 2. 初始止损（主版本冻结）
```
StopDistance = clip(1.5 * ATR_ref / Entry, 3%, 6%)
StopPrice   = Entry * (1 - StopDistance)
```
- 预注册对照：`1.2×ATR`、`1.8×ATR`（仅记录，不作为线上默认）。
- `2×ATR` 仅保留诊断，不作为默认。

## 3. 移动止盈（两档，必须拆开）

### 第一档：普通反弹保护
- 启动门槛：`PeakRet >= max(2%, 0.8 * ATR_ref / Entry)`
- 启动后，从实时高水位回撤：`Drawdown >= max(1%, 0.45 * ATR_ref / Entry)` → 普通反弹退出。
- 作用：保护小反弹利润。

### 第二档：盈利趋势保护
- 启动门槛：`PeakRet >= max(5%, 1.5 * ATR_ref / Entry)`
- 移动止盈距离：`TrailDistance = max(1.5%, 0.75 * ATR_ref / Entry)`
- `TrailStop = HighWatermark * (1 - TrailDistance)`
- 作用：允许较强走势继续运行，仅回撤超阈值才出场。

高水位（HighWatermark）每次只使用当前及之前 bar，绝不使用未来窗口最大值。

## 4. 反弹失败（REBOUND_FAILURE-v1，独立消融分支）
状态机：
1. 进入受损：`Price <= Entry - 0.5 * ATR_ref`
2. 形成一次有效反弹：`Price - ReboundLow >= 0.3 * ATR_ref`
3. 尚未收复成本，随后从反弹高点再次回落：`ReboundHigh - Price >= 0.2 * ATR_ref` 且 `Price < Entry`
   → 满足后判定反弹失败。
执行：
- 若全过程发生在买入日 T+1：仅记录触发，**T+2 按实际开盘可成交价卖**。
- 若发生在 T+2 以后：按下一分钟 VWAP 或下一可成交价卖。
- 若 T+1 曾下破 0.5ATR，但收盘未收复成本、也未形成明确二次回落：
  标记 `DAMAGED_UNRESOLVED`，T+2 重点观察，**不臆造卖出**。

本分支是实验分支，不与基础动态止盈捆死。比较 `X1`（不含 RF）与 `X1_RF`（含 RF）。

## 5. 退出优先级（冻结）
1. 硬止损
2. 反弹失败
3. 已激活的移动止盈（一档/二档）
4. T+5 时间退出

同根 bar 同时触发止损与止盈时，**止损优先（悲观序）**。

## 6. 分钟精确口径
- T+5 在 14:50 产生时间退出信号，使用 14:51–14:55 VWAP 或收盘集合竞价口径成交。
- 涨跌停锁死不得假设成交。
- 跳空按实际开盘价成交（不按理想止盈价）。

## 7. 双轨历史数据
- 2019–2024：`DAILY_APPROX`（日线保守模拟，明确标记为近似）。
- 2025–2026：`MINUTE_EXACT`（分钟精确模拟）。
- 两类结果**分开报告**，不混算成一条收益曲线。
- 某 code 在 2025–2026 缺分钟缓存时，该 event 回落 `DAILY_APPROX` 并在
  `label_precision` 标注实际精度。

## 8. 四套标签变体
| 变体 | 硬止损 | 反弹失败 | 移动止盈 |
|------|-------|---------|---------|
| X0_CONTROL | ✓ | ✗ | ✗（仅止损+时间退出，对照） |
| X1 | ✓ | ✗ | ✓（一档+二档） |
| X1_RF | ✓ | ✓（全仓卖出） | ✓ |
| X2_STAGED | ✓ | ✓（卖 50%，余仓进入二档移动止盈） | ✓ |

## 9. 每笔标签字段（parquet 列）
```
event_id, code, signal_date, entry_date, entry_time, entry_price,
atr_ref, initial_stop_price, high_watermark, activation_time,
exit_signal_time, exit_time, exit_price, exit_reason, exit_blocked,
gross_return, cost, net_return, holding_minutes, holding_days,
label_precision, strategy_variant
```

## 10. 验收标准（必须全部通过）
- 不读取未来窗口最大值。
- 每次高水位更新只使用当前及之前的 bar。
- T+1 买入当天不得成交卖出。
- 同 bar 止损与止盈冲突时止损优先。
- 跳空按实际开盘成交。
- 涨跌停锁死时不得假设成交。
- 日线与分钟标签严格分开。
- 600313 / 600479 三段分钟复放与 K 线人工观察一致。
- 随机抽取至少 50 笔，逐路径可解释。
- X0/X1/X2/X1_RF 的退出原因与收益能从原始行情重新计算一致。
