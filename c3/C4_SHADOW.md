# C4 Shadow — materiality_head_v1 前瞻影子评分

## 边界（路线 C3→C4）
C3 负责历史信息影响模型（已 `HISTORICAL_TUNING: CLOSED` / FINAL 历史训练 v1.0.0 完成）。C4 负责**冻结代码后的前瞻影子评分与校准**，只输出概率、校准状态、缺失警告与 `evidence_ids`，**不直接改变交易**。真正晋级证据来自冻结后的新前瞻期；C5 才由 Pyxis 判断风险。

## Head 路由（第一版，不等其他三个 head）
- `materiality_head_v1` = **FROZEN / ACTIVE SHADOW**（模型 v1.0.0）
- `direction_head` = **EXPERIMENTAL / NOT_USED**
- `reflexivity_head` = **UNAVAILABLE**
- `persistence_head` = **UNAVAILABLE**

C4 当前只能回答：“这条信息产生显著市场反应的概率有多大。” 不能翻译成 `p_materiality 高 = 利好 = 应该买`（direction 未证明）。当前合法输出：
```text
p_materiality      = 0.xxx
direction_status   = EXPERIMENTAL_NOT_USED
reflexivity_status = UNAVAILABLE
persistence_status = UNAVAILABLE
```
Corvus 上线 Shadow 后也不为「字段完整」硬塞三个弱/未完成头；这比伪造完整性强得多。

## 组件化结构（防 feature drift 的核心）
```
corvus/materiality_v1/
  feature_contract.json          # 可哈希冻结特征契约（52 维，含 TOP_CATEGORIES）
  materiality_features_v1.py     # 唯一共享特征实现 build_features + MODEL_FEATURES
  scoring_domain.py              # C4_SHADOW_SCORING_DOMAIN_V1 契约 + is_scoring_eligible
  feature_parity_gate.py         # 上线前必过：离线/在线特征一致性 + 截断无泄漏
  train_final.py                 # 全部 eligible P0 2019–2025 训练，持久化 v1.0.0
  c4_shadow.py                   # CorvusInformationSnapshot + score_event（入口）
  artifacts/
    model_manifest.json          # v1.0.0 + 各文件 SHA256 + 训练元数据
    lgbm_materiality_s{0,1,2}.txt
    feature_medians.json         # 线上缺失填充参考
    feature_parity_passed.json   # parity gate 通过凭证（score_event 强制检查）
```
`walk_forward.py` / `c3_smoke_train.py` / `make_stratified_checklist.py` 现在**全部从 `materiality_features_v1` 导入**，不再各自实现特征。

## 评分域契约（C4_SHADOW_SCORING_DOMAIN_V1）
实时打分**只要求**：`event_tier==P0` & `non-HEAVY` & `category!=DISTRIBUTION` & `entity mapping valid` & `available_at 已真实可见` & `feature completeness 满足`。
**禁止使用标签侧字段**：`label_valid` / `corp_action_within_horizon` / `later_realized_label` / `training_eligible`（这些要看未来 T..T+5，线上即未来信息泄漏）。`is_scoring_eligible()` 刻意不检查它们。

## Feature Parity Gate（上线前必过）
随机抽历史 cluster，比较：
- 离线（批量）特征 vs 在线（子集注入）特征：`MODEL_FEATURES` 名称/顺序/dtype/缺失语义/类别编码一致，数值 `max_abs_diff ≈ 0`；
- 截断测试：向 `label_start` 之后追加伪造行情，特征向量不变（无未来泄漏）。
通过后才写 `feature_parity_passed.json`；`score_event` 加载模型前强制检查，缺失即拒绝。

## 每个新事件写一条 `CorvusInformationSnapshot`
```
event_cluster_id
available_at
available_at_precision
event_tier

p_materiality
materiality_model_version

calibration_status
missing_warnings

entity_ids
event_family
evidence_ids

feature_snapshot_id
created_ts
```

## 实时链路
```
新公告 → document ingest → event clustering → entity/taxonomy → P0 eligibility
      → available_at confirmed → materiality_features_v1 → CORVUS_MATERITY_HEAD_V1
      → p_materiality → CorvusInformationSnapshot → SQL append-only
```
标签成熟后走完全独立链（见下文「生产持久化」）：label builder → `c4_shadow_realization`（独立 append）→ 评估。**prediction 表永不 UPDATE；realization 存于独立表**。

## 前瞻期重点监控（不再追 ROC 小数点）
- **Calibration drift**：`p=0.8` 的事件之后实际 materiality 是否接近对应水平。
- **年份/市场状态漂移**：同时记录 base rate（2024/2025 先验差异已明显）。
- **事件族漂移**：哪些 event family 开始失准。
- **数据质量**：实体、时间戳、cluster、缺失、corp action、重复事件。
- 记录 prediction 表：`prediction_ts / model_version / p_materiality`；realization 单独存 `c4_shadow_realization`：`event_cluster_id / materiality_model_version / label_horizon / realized_label / label_ts`。

## 数据纪律（硬性）
prediction 表保存：`prediction_ts` / `model_version` / `p_materiality`（INSERT 后不再改）。realization 存独立表 `c4_shadow_realization`。**禁止标签成熟后回写 prediction 行**（避免 leakage / 自我印证）。

## Calibration 说明
当前 `calibration_status = HISTORICAL_OOT_VALIDATED_RAW_PROBA`（raw LGBM predict_proba，冻结期 OOT 已证明可用）。**不新增 Platt / isotonic calibrator**（否则等于重新打开 Historical Tuning）。真正的校准质量由 C4 新前瞻期验证。

## 部署验收（C4_DEPLOYMENT_ACCEPTANCE_V1, PASS）
不涉及训练，只验证实时运行语义。脚本 `corvus/materiality_v1/deployment_acceptance.py`，通过写 `artifacts/c4_deployment_acceptance_passed.json`（`score_event` 启动前须先过 Feature Parity Gate 与 Model Hash Verify）：

1. **IDEMPOTENCY**：同 `(event_cluster_id, model_version)` 不产生两个不同 prediction。
2. **APPEND-ONLY**：realization 独立 append 到 `c4_shadow_realization`；`c4_shadow_prediction` 永不 UPDATE。
3. **MODEL HASH VERIFY**：manifest 与本地 artifact hash 不一致 → scorer 拒绝启动（含负例）。
4. **EVENT-TIME ORDER**：`prediction_ts >= available_at` 且 `feature_asof <= prediction_ts`。
5. **RETRY SAFETY**：ingest 重试/重启不重复写多条 shadow prediction（unique key upsert）。
6. **FAILURE EXPLICIT**：feature 缺失 / entity 无效 / 非 P0 → 显式 `REJECTED/NOT_SCORED` 原因，不静默丢事件。

## 持久化（DB 唯一键 = (event_cluster_id, materiality_model_version)）
参考实现 `corvus/materiality_v1/shadow_store.py`（`ShadowStore`）：
- 同事件 + 同模型版本 upsert，**不新增重复行**（RETRY SAFETY + IDEMPOTENCY）；同事件+同版本两条不同结果 → 拒绝。
- 同事件允许不同模型版本并行 Shadow（1.0.0→A, 1.1.0→B），互不影响。
- `attach_realized_label` 只写 realization 表；必须先有 prediction row，顺序不可颠倒（APPEND-ONLY）。

## 生产持久化（PostgreSQL 双表不可变）
将「不可变预测」由数据库结构保证，而非仅靠代码纪律：

```sql
-- 预测表：永不 UPDATE，仅 INSERT（同键重复 INSERT 由唯一约束去重/拒绝）
CREATE TABLE c4_shadow_prediction (
    event_cluster_id            TEXT    NOT NULL,
    materiality_model_version   TEXT    NOT NULL,
    prediction_ts               TIMESTAMP NOT NULL,
    p_materiality              DOUBLE PRECISION,
    feature_snapshot_id         TEXT,
    available_at                INT,
    event_tier                  TEXT,
    calibration_status          TEXT,
    missing_warnings            TEXT,
    entity_ids                  TEXT,
    event_family                TEXT,
    evidence_ids                TEXT,
    created_ts                  TIMESTAMP,
    PRIMARY KEY (event_cluster_id, materiality_model_version)
);

-- 实现表：标签成熟后独立 append；绝不回写 prediction 表
CREATE TABLE c4_shadow_realization (
    event_cluster_id            TEXT    NOT NULL,
    materiality_model_version   TEXT    NOT NULL,
    label_horizon               TEXT    NOT NULL,   -- 'T+1' / 'T+3' / 'T+5'
    realized_label              BOOLEAN NOT NULL,
    label_ts                    TIMESTAMP,
    label_version               TEXT,
    PRIMARY KEY (event_cluster_id, materiality_model_version, label_horizon)
);
```

- prediction 表一旦 INSERT 即冻结；任何「修正」都应是新模型版本的 prediction，而非 UPDATE 旧行。
- realization 表按 `(event_cluster_id, model_version, label_horizon)` 多 horizon 独立记录，便于 C4_FORWARD_REPORT 多窗口评估。
- 内存参考实现 `ShadowStore`（`_predictions` / `_realizations` 两表）已对齐此结构。

## 上线后禁止动作（从第一条真实 prediction 落库起）
不因短期结果发生：
- 不改 LightGBM 参数 / `TOP_CATEGORIES` / 52 维特征；
- 不重新选 threshold；不新增 Platt / isotonic calibrator；
- 不根据几天表现重训 v1.0.0；
- 不让 `p_materiality` 进入 GREEN / Pyxis 决策。

实时链异常先看四个工程健康指标（latency / duplicate_retry / not_scored+reason / feature_missing_drift），先查数据链，再查模型。

## 晋级条件（提前冻结 → FORWARD_VALIDATED）
≥20–40 trading days、样本量足够、无严重 feature drift、无 calibration collapse、高分→实际 materiality 基本单调、Brier 未明显恶化、无单一 event family 撑结果。通过后才讨论 `Corvus C5 → Pyxis`。详见 `MATURITY_STATUS.md`。

## 前瞻报告（C4_FORWARD_REPORT）
`T0` 立即存 `p_materiality / model_version / feature_snapshot / prediction_ts / evidence`；`T+1/T+3/T+5` 另表追加 `realized_materiality`。第一版只看 `n_scored / n_labeled / score distribution / realized base rate / ROC / PR-AUC / Brier / calibration bins / ECE / event-family breakdown / missing & reject rate / score drift / 高分失败与低分漏判案例`。**不再用历史 ROC 决定任何东西**。

## 上线状态语义（LIVE_SHADOW ≠ FORWARD_VALIDATED）
第一条真实事件成功落库，只把 `C4 realtime deployment` 从 `PENDING` 升为 **`LIVE_SHADOW`**；模型**不**晋级 `FORWARD_VALIDATED`。`FORWARD_VALIDATED` 仍须等冻结的前瞻观察期（≥20–40 trading days）与 calibration/drift 闸门。Corvus 在 `LIVE_SHADOW` 阶段不能直接干预交易。

## 工程层长期健康监控（非模型闸门）
区分「模型真的漂了」与「实时数据链坏了」：
- `ingest_to_prediction_latency`：公告到落 prediction 的时延（链路健康）。
- `duplicate/retry_rate`：ingest 重试/去重比例（RETRY SAFETY 实际表现）。
- `not_scored_rate + reason distribution`：被拒比例与原因（tier / entity / feature / future）。
- `feature_missing/drift rate`：缺失填充触发率与特征漂移（Feature Parity Gate 持续 PASS 是其硬约束）。

异常先查数据链，再查模型；这些指标不影响模型冻结状态。

## 最薄生产链（C4 段编排）
`corvus/materiality_v1/production_chain.py` 仅编排 C4 拥有的段：
`C4 scoring domain → materiality_features_v1 → model v1.0.0 → CorvusInformationSnapshot → shadow_store (append-only)`。
上游（raw document persist → clustering → entity/taxonomy → available_at）由部署侧负责，不在本模块。编排层不加载/不修改模型，只返回含 `action / reason / p_materiality / latency_ms / missing_warnings` 的结果，供上述四个健康指标计算。
