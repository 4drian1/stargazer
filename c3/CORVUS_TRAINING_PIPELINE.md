# Corvus 信息模型 · 训练流程文档（materiality_head_v1 / v1.0.0）

> 本文档描述第一阶段模型 `materiality_head_v1` v1.0.0 的**完整训练链路**：数据来源、标签契约、训练域、特征、模型、验证、artifact 身份与冻结纪律。
> 使用与接入见 `CORVUS_INFO_MODEL_GUIDE.md`；状态登记见 `MATURITY_STATUS.md`。

---

## 0. 一句话结论

模型已**冻结（FROZEN）**：全部合格历史数据（2019–2025 P0）一次性训练，3 个 seed 的 LightGBM 平均，产出不可覆盖的 artifact 身份（manifest + SHA256）。训练结束后**不再回头调参**。第一阶段是「历史建模」的终点，后续进入「前瞻影子运行」。

---

## 1. 数据来源（canonical 主表）

训练与评分**共用同一张表**：`out/event_cluster.parquet`（由 `backfill_pipeline.py` 回填产出）。

- 1 行 / `event_cluster_id`（同票同日多公告合并为簇）。
- 字段包含：行情派生标签（`ret_1d/3d/5d_pc`、`sigma250`、`gap_return`、`volume_shock`、`mfe/mae`、`mat_1d/3d/5d`、`materiality`）、事件元数据（`code/name/title/category/tier/attribution_quality`）、时间语义（`announce_date`、`label_start`、`label_start_basis`）、标签质量（`label_status`）。
- 关键过滤列：`label_valid`、`training_eligible`、`corp_action_within_horizon`、`label_status`，由回填阶段统一写入，训练/C4/Pyxis 不再各自硬编码排除。

> 标签口径细节见仓库 `README.md`（C3 Pack 回填与标签章节）。

---

## 2. 标签契约（materiality）

```
materiality = 任一视野异常
            = (|ret_1d_pc| > 2×σ250) 或 (|ret_3d_pc| > 2×σ250) 或 (|ret_5d_pc| > 2×σ250)
```

- `σ250` = 事件日前 250 交易日收益标准差（前波动基线）。
- 窗口含除权日则对应路径标签为 NULL（避免除权伪影）。
- 时间语义（保守可见性）：`POST_CLOSE_NEXT_DAY` / `WEEKEND_NEXT_DAY` / `RESUME_FIRST_TRADE`；巨潮仅日期粒度（`available_at_precision = DATE_ONLY`），**禁止 `--assume-premarket`**（FROZEN 口径外）。

---

## 3. 训练域（冻结）

```
tier == P0
& label_valid == True
& training_eligible == True
& attribution_quality != HEAVY
& category != DISTRIBUTION
& ~corp_action_within_horizon        # 排除 CORPORATE_ACTION_WINDOW（未来信息）
& cluster-level（按 event_cluster_id 去重）
```

实现见 `train_final.py::TRAIN_DOMAIN` / `main()`。在 eligible 集上 `drop_duplicates("event_cluster_id")` 保证簇级、不重复。

**DISTRIBUTION 审计**：计数审计确认 eligible 集内 `category==DISTRIBUTION` 数量为 0——训练域与线上评分域一致地排除同一切片，且无实际占比，因此**不构成 train-serving mismatch**，无需重训。

---

## 4. 特征（唯一共享实现，52 维，冻结）

所有特征均满足 **available_at 前可见**，只用 `label_start` 及其之前的数据。

实现文件：`corvus/materiality_v1/materiality_features_v1.py` 的 `build_features`——这是 **C3 walk-forward / train_final / C4 在线打分三者唯一共享**的特征代码，任何一侧都不得自行重实现（防 feature drift）。

| 分组 | 数量 | 内容 |
|---|---|---|
| `RAW_NUMERIC` | 10 | `is_st_at_event`, `log_float_mv`, `log_total_mv`, `turnover20`, `vol20`, `dist_high_250`, `gap_return`, `volume_shock`, `n_announcements_in_cluster`, `sigma250` |
| `KEYWORD_FEATURES` (`kw_*`) | 30 | 标题关键词命中权重（来自 `_taxonomy.KEYWORD_MATERIAL_WEIGHT`，顺序固定） |
| `CATEGORY_FEATURES` (`cat_*`) | 12 | **冻结 one-hot over `TOP_CATEGORIES`**（不按输入动态 `value_counts`，否则线上单簇编码与离线错位） |

- 合计 **52 维**，顺序由 `MODEL_FEATURES` 固定，契约见 `feature_contract.json`（`feature_count: 52`）。
- 缺失填充：训练期计算各特征中位数存 `feature_medians.json`，线上用中位数填充（`predict_proba` 内 `fillna(medians)`）。

> 类目冻结是防线上/离线漂移的关键设计：动态 `value_counts()` 做 one-hot 在批量回测看不出问题，进单事件线上评分就很危险。

---

## 5. 模型（LightGBM，3 seed 平均）

```python
FROZEN_CONFIG = {
    "model_family": "LightGBM", "objective": "binary",
    "n_estimators": 300, "learning_rate": 0.05, "num_leaves": 15,
    "min_child_samples": 30, "subsample": 0.8, "colsample_bytree": 0.8,
    "class_weight": "balanced", "seeds": [0, 1, 2],
}
```

- 每个 seed 单独训练一个 `LGBMClassifier`，分别 `save_model` 为 `lgbm_materiality_s{0,1,2}.txt`。
- 线上 `MaterialityModel.predict_proba` 对 3 个 booster 取 **平均概率**（raw LGBM proba，不做 Platt/isotonic 再校准）。
- `calibration_status = HISTORICAL_OOT_VALIDATED_RAW_PROBA`：原始概率在冻结期四折 OOT 已证明可用；**不在冻结后新增校准器**。

---

## 6. 验证（四折 OOT + 部署语义）

### 6.1 历史时间外验证（OOT，配置可用性的证据）

四折（按年）：

| 年份 | ROC | PR-AUC |
|---|---|---|
| 2022 | 0.672 | 0.557 |
| 2023 | 0.701 | 0.522 |
| 2024 | 0.695 | 0.668 |
| 2025 | 0.715 | 0.519 |

OOT 仅作为「配置在历史时间外能工作」的证据；训练后不再回头调模型，也不再用历史 ROC 决定任何东西。

### 6.2 Feature Parity Gate（上线前必过）

`feature_parity_gate.py` 验证：

1. 离线（批量）特征 vs 在线（子集注入）特征：列顺序 / dtype / 缺失语义 / 类别编码一致，数值 `max_abs_diff ≈ 0`；
2. 截断测试：向 `label_start` 之后追加伪造行情，特征向量不变（无未来泄漏）。

通过后写 `artifacts/feature_parity_passed.json`；`MaterialityModel` 加载前强制检查，缺失即拒绝。

### 6.3 部署验收（C4_DEPLOYMENT_ACEPTANCE_V1，6/6）

`deployment_acceptance.py` 验证运行语义：IDEMPOTENCY / APPEND-ONLY / MODEL HASH VERIFY / EVENT-TIME ORDER / RETRY SAFETY / FAILURE EXPLICIT。通过后写 `artifacts/c4_deployment_acceptance_passed.json`。

---

## 7. Artifact 身份（不可覆盖）

`train_final.py` 产出 `artifacts/model_manifest.json`（version **1.0.0**），登记：

- 3 个模型文件 + 各自 **SHA256**（`model_sha256`）；
- `feature_medians.json` + SHA256；
- 训练数据 `event_cluster.parquet` + SHA256；
- `feature_contract.json` + SHA256；
- 配置 `FROZEN_CONFIG` + SHA256；
- `code_commit`、`train_domain`、`n_train`、`positive_rate`、`label_contract`、`time_semantics`。

**身份反查链**：`model ← feature medians ← training data ← feature contract ← config` 全部可 SHA256 反查，满足 Stargazer 后续完整审计要求。

> 实测：`n_train = 148,384` 簇，`positive_rate ≈ 0.361`。

---

## 8. 冻结纪律与如何升级版本

### 8.1 冻结后禁止

- 改 LightGBM 超参 / `TOP_CATEGORIES` / 52 维特征；
- 改标签契约 / 时间语义 / 重新选 threshold；
- 新增 Platt / isotonic 校准器；
- 根据短期表现重训 v1.0.0。

### 8.2 版本升级规则

`1.0.0` 永久对应此模型；重新训练**只能升 `1.0.1` / `1.1.0`，不覆盖**：

- 仅修复/补数据、行为不变 → `1.0.1`；
- 引入新特征/改结构 → `1.1.0`。

新版本独立登记新 manifest（新 SHA256），旧版本 artifact 保留。DB 唯一键 `(event_cluster_id, materiality_model_version)` 天然支持同事件多版本并行 Shadow。

### 8.3 重训命令

```bat
python corvus/materiality_v1/train_final.py
```

> 重训前请确保 `out/event_cluster.parquet` 已按最新回填口径更新，且新 manifest 的 `version` 已按 8.2 规则递增（默认脚本写 `1.0.0`，升级版本时须手动改 `version` 字段以免覆盖）。

---

## 9. 阶段边界

- **历史建模（C3+C4 模型侧）= CLOSED**；`materiality_head_v1 = FROZEN`。
- 下一步是**部署任务**：接实时 ingest → append-only Shadow → 跑 deployment acceptance → `LIVE_SHADOW`。
- 至少经过冻结的 ≥20–40 个交易日 + calibration/drift 闸门后，才有资格进入 `FORWARD_VALIDATED`，届时再讨论 `Corvus C5 → Pyxis`。
