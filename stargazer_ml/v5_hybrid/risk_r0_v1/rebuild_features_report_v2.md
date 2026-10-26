# 阶段三报告：pB2 绑定、决策时刻对齐与回收（rebuild_features_v2）

时间：2026-10-19。纪律：零网络取数、未改冻结文件、未重训、缺证据逐条隔离。产物全部在 `risk_r0_v1/rebuild_features_v1/`（v2 后缀，不覆盖 v1；新增约 0.74 MiB）。脚本：`risk_r0_v1/rebuild_features_v2.py`（新文件）。

## 1. 重放验收（v2 重跑，位级一致）

24 个冻结 test 段样本：232 模型 max|Δ|=0.0、266 模型 max|Δ|=0.0、特征矩阵 vs 冻结 load_arm max|Δ|=0.0（`replay_acceptance_v2.json`）。新 cutoff 校验：2 个有冻结参考的重叠键上，修复后参考价直连重算 vs 生产线最终值 max|Δ|=0.0（`new_cutoff_validation_v2.json`）。

## 2. pB2 绑定

- 来源：`rolling_lr_v2_predictions.parquet` **FINAL** fold，按 (code,P) 关联。
- **220/228 匹配**；8 条不匹配（301052×2、605218、301357、301658、600751、603396、605116）——与用户事先清单完全一致，`pB2=NaN`、`pb2_reason=NOT_IN_SCORING_POPULATION(CORP_ACTION_ISOLATED)`，未填零。

## 3. 决策时刻对齐与回收

- 触发记录 cutoff = 不晚于 trigger_t 的最近已完成 15 分截止（权威网格 slots(day,15)，16 个：09:45–11:30、13:15–15:00，bar 完成语义）；非触发维持 E 日 15:00。
- 重新验收口径：E 日截至 cutoff 的 bar + 前一完整交易日（240 根）+ joint_parts 前收。**cutoff 之前缺 bar 仍拒绝**（237/240 规格未放宽）。
- 触发记录重验结果：30 条中 **9 条通过**（8 条原 OK 改到新 cutoff + **回收 1 条**），21 条仍拒绝（20 条 cutoff 前缺 bar、1 条前一日不完整）。无 trigger_t 早于 09:45 的 NO_RISK_SNAPSHOT 案例。
- **回收的 1 条：300639 @ 2026-04-14**，trigger 11:10 → cutoff 11:00；缺的两根 bar 在 11:15/11:30（cutoff 之后），按新口径合格。特征在新 cutoff 下由重放矩阵重算（226 条非冻结 own 重算之一），双模型打分成功（amp_232=1.09%，为 9 触发中最低）。
- 隔离 1 条：002405 @2026-09-18（context_loo 源缺行，同 v1），不计入有效人口。

## 4. 新有效人口（v2）

| 指标 | 数值 |
|---|---|
| 有效打分记录 | **228**（229 构建 − 1 隔离） |
| 触发 | **9**（8 原有 + 1 回收），分布在 8 个 E 日 |
| E 日覆盖 | 46 天（4/5/6 月为主：49/74/99 条，8 月 2、9 月 3） |
| 每 E 日横截面 | 均值 4.96，中位 ~3，≥5 的日 18/46，≥10 仅 8 天 |
| 每 cutoff 横截面 | 53 个不同 cutoff，**触发 cutoff 时刻 8/9 只有 1 只候选**（唯一例外 2026-05-29 15:00 共 12 只） |
| amp_232（clip 后） | mean 1.46%、范围 [1.09%, 3.26%]；触发记录 1.09%–1.55% |
| pB2 | 220 匹配 / 8 留空留因 |
| decision cutoff | 每条记录实际值见 e_update 表（trigger_basis 含 trigger_t/trigger_close/cutoff/reason） |

## 5. 联合对照可执行性判断

**仍不能执行有统计效力的固定联合对照。** 触发仅 9 条、且 8/9 的触发 cutoff 时刻同刻横截面为 1（连"同刻配对比较"都无从做起），E 日横截面中位数个位数。本阶段成果定位：决策时刻对齐后的**描述性评分表**（可用于个案检视与方向性观察），不构成能力结论。若要推进对照，需先解决覆盖（见下）或改用冻结 predictions 原人口。

## 6. 真缺失清单（`true_missing_v2.json`，供最小补取预算）

695 候选中 466 条仍缺，精确到 bar 号位/时间区间：
- **触发未回收 21 条**：20 条 cutoff 前缺 bar（典型：缺 11:30 那根 (113000,113000) ×7——7 月采集缺口的同型缺口；整日缺口 (93100,144500) ×2 等）；1 条前一日不完整（002114）。
- 非触发 445 条：E 日缺席 110、E 日不完整 248（绝大多数为 11:30 单根缺口或文件止于 2026-01/09-07）、前一日不完整 82、quarantine 4、前收缺失 1。
- 最小补取预算方向（若获批取数）：**(a) 50 只文件的 11:30 单根 bar（2026-07 为主）**；(b) 7 只触发股票的整日分钟（文件止于 2026-01，E 日在 4–7 月）；(c) 个别文件尾盘区间。不需求全市场。

## 7. 产物（v2 后缀，合计 ~0.74 MiB）

`pday_snapshot_v2.parquet`（P 日快照：pB2 及 P 可知列，无前视）、`eday_intraday_update_v2.parquet`（实际 decision cutoff + 触发依据 + 双模型幅度）、`candidate_features_v2.parquet`（审计矩阵）、`candidate_isolation_v2.json`、`replay_acceptance_v2.json`、`new_cutoff_validation_v2.json`、`population_stats_v2.json`、`true_missing_v2.json`、`candidate_scores_meta_v2.json`、`plan_v2.json`、`stop_record_v2.json`。
