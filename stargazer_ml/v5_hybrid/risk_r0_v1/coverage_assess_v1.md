# 覆盖核验与重建管线可复用性评估（第一阶段，coverage_assess_v1）

生成时间：见 coverage_assess_v1.json。全程零网络取数、未修改任何冻结文件/模型目录/星币侧产物。

## 1. 冻结分钟账本定位

- 清单：`risk_r0_v1/relative_stall_5m_20261018_v1/raw_input_ledger.json`，**4196 条**，每条 = code + 绝对路径 + sha256 + 字节数。
- 存放目录（按 code 直接查 `<code>.parquet`）：
  - `stargazer_ml/pullback_train/minute_cache_clean/`（3955 条）
  - `green_common_profile/gcp_round_b_small_sample_v1/minute`（1）、`minute_ext`（147）、`minute_car`（93）
- 字段：`code, date(1分钟bar末时间戳), open, close, high, low, volume, amount, volume_scale_factor, volume_repaired`。
- 网格语义：每日 240 根有效 bar（09:31–11:30 + 13:01–15:00，bar 末，09:30 那根不计入）。
- 整体日期跨度 20250102–20260908，但**逐文件差异很大**：543 只交集中 444 只到 2026-09（约 09-07/08），**50 只止于 2026-01**，其余散在 2026-03~08。
- 完整性：543/543 文件 sha256 与账本一致（零漂移）。

## 2. 候选集口径对齐

- 用户的 812 条/639 只/33 触发 = status+E 窗口过滤**不含** admission 条件；加上 `master_admission=='OK'` 后为 787 条/620 只/32 触发（剔除 23 条 ISOLATED 除权 + 2 条 REJECT 标签窗口不全）。JSON 中两组都留档。
- 与用户口径完全一致的主集 = status+E ∩ 4196 账本代码 = **695 条 / 543 只 / 30 触发**（已复核一致）。下表为主集统计。

## 3. 覆盖核验结果（695 条主集，判定口径：E 日 240 根全量 + 前一完整交易日 + joint_parts 有 E 日前收）

| 判定 | 条数 | 说明 |
|---|---|---|
| **全部在场（OK）** | **228** | 其中触发 8/30；223 条同时有完整 120m 标签窗口 |
| E 日 bar 不全 | 262 | 237–239/240 根（均值 238.8）；7 月大量缺第 119 号位（11:30 那根），属近失而非停牌 |
| E 日完全无数据 | 117 | 文件本身止于 E 之前（108/379 的 absent+incomplete 属此类）；9 月下旬 E 基本够不到 |
| 前一日不完整 | 83 | E 日前一交易日无完整会话（多为停牌恢复边界） |
| E 日被 quarantine | 4 | pullback_train/quarantine_2026.csv 排除 |
| joint_parts 缺 E 日前收 | 1 | 唯一非分钟级缺失 |
| 文件 sha 漂移 | 0 | — |

- 月份分布（OK/总数）：202604=49/103，202605=74/148，202606=99/286，**202607=0/103**，202608=2/19，202609=4/36。
- 543 只中仅 **196 只**至少有 1 条 OK 记录。
- 冻结 predictions.parquet 里已覆盖其中 2 条键，可直接 join。
- 预热需求结论：个股侧只需 **E 日全日 + 1 个前序完整交易日**（前收参考 + 前会话量比）；不存在 60/120 交易日的个股预热；市场侧历史（rsi/macd/daily_return20）在 market_price_features.parquet 里全窗口在场。120m 标签需 E 日尾盘 60 分钟 + 次日 09:31–10:30。

**关键区分：543/543 文件在场 ≠ 特征可重建。真正可打分的是 228/695 条；467 条失败是原始数据本身缺 bar/缺日，任何重建都救不回来（零取数纪律下）。**

## 4. 重建管线可复用性评估

### 4.1 冻结构建器是"重放器"，不是通用接口

- `build_multiboard_full_features_v1.run`：钉死原始 cohort 100704 行、钉死账本逐文件哈希、cohort 网格精确匹配（EXACT_COHORT_CUTOFF_GRID_MISMATCH）。
- `build_continuous_motion_population_v1`：钉死 population 86319（POPULATION_DRIFT）、账本长度 4196（BOUND_RAW_SCOPE_CHANGED）、CATALOG 5540/978（CATALOG_DRIFT）、400 受限锚（ANCHOR_ACCEPTANCE_DRIFT / ACCEPTED_RESTRICTED_ANCHOR_SCOPE_CHANGED）。
- 结论：**不能传入任意 code 列表**；且按纪律不得改动这些冻结文件。

### 4.2 但纯计算内核全部可复用

`minute_returns`、`path_arrays`/`board_motion`/`context`（continuous_motion_matrix_v1）、`matrix_features`/`summarize`（multiboard_risk_features_v1）、`common_loo_derive_v1`+`continuous_motion_common_v1`（ccm_v1 34 字段）、`market_arrays` 均为无 IO 纯函数，import 即用。

### 4.3 上下文覆盖（均已有全窗口覆盖，无需重建）

- `market_price_features.parquet`：448 天（20241209–20261016），测试窗 100% source_complete。
- `shared_market_and_leaders.parquet`：448 天 × 每日 16 个 cutoff 完整（含领涨列），任何 15:00 on-grid cutoff 可直接 join。
- `golden_anchor_replay_20261017_v2`：442 个状态至 20261016。
- THS 概念目录：5540 代码/978 板块，hash 钉死但 543 只全部在内且均有板块成员资格。
- `joint_parts` 前收：4196 账本代码全覆盖（4047 只有参考行），543 只的 E 日前收仅缺 1 条。

### 4.4 232/266 字段分级结论

- **唯一需要真"算"的**是板块共享上下文（mb5_/mb15_/nh_mb_*，约 90 字段）所需的 5/15 分钟收益矩阵：该矩阵未持久化（persisted_wide_table=False），需对 4196 个冻结分钟文件做一次确定性重放（分钟级，零取数）。
- **232 字段版：等级 B —— 需新编排脚本（不改冻结文件），无算法改动。** 共享矩阵按月重建一次即可缓存，之后任意候选 (code, 15:00 cutoff) 拼接冻结上下文直接出特征。
- **266 字段版：等级 B —— 同上。** 多出的 34 个 ccm_v1 字段是同矩阵上的纯数组 LOO 推导，增量成本可忽略。
- 真正的瓶颈不是工程量而是**数据在场**：467/695 条在原始分钟数据层即不合格。

## 5. 建议的下一步

1. 第二阶段写一个**新的薄编排脚本**（冻结文件一律不动），只对 228 条 OK 记录（含 8 条触发）出 232/266 特征并跑两版冻结模型打分——纯复用、零取数、当天可完成。
2. 262 条近失记录（237–239 根）是否按放宽口径打分是**规格变更**，需显式批准；默认排除。
3. 117 条缺席 + 83 条前日缺失不要花力气：冻结语料里根本没有那些 bar/日期。
4. 若要联合实验评估预测对实现风险，223 条 OK 记录有完整 120m 标签窗口可用。
