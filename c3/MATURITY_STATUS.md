# C3 materiality — 冻结状态登记

## 当前登记（冻结升级）
```
CORVUS_MATERIALITY_HEAD_V1

STATUS:
FROZEN

TRAINING_DOMAIN:
P0
label_valid == True
training_eligible == True
non-HEAVY
cluster-level

CANONICAL_DATA:
event_cluster.parquet

EXCLUDED:
CORPORATE_ACTION_WINDOW
invalid / censored labels
non-P0 scoring domain

TIME_SEMANTICS:
POST_CLOSE_NEXT_DAY
WEEKEND_NEXT_DAY
RESUME_FIRST_TRADE

ASSUME_PREMARKET:
FORBIDDEN_IN_FROZEN_V1

OOT:
2022 ROC 0.672 / PR 0.557
2023 ROC 0.701 / PR 0.522
2024 ROC 0.695 / PR 0.668
2025 ROC 0.715 / PR 0.519

HISTORICAL_TUNING:
CLOSED
```

## 正式裁决（C3=CLOSED 收口）
```text
C3                          = CLOSED
C4 MODEL SIDE               = COMPLETE
C4 DEPLOYMENT CONTRACT      = PASS
Historical tuning           = CLOSED
Realtime ingest             = NEXT        ← 部署任务，不再属于历史建模
Pyxis consumption           = LOCKED
Forward validation          = NOT YET
```

## 状态转换语义（关键边界）
```text
C3 historical research
        ↓
materiality_head_v1 = FROZEN ✅
        ↓
C4 shadow implementation
        ↓
deployment semantics 6/6 PASS ✅
        ↓
Realtime ingest integration      ← NOW
        ↓
第一条真实新事件
→ v1.0.0 产生 immutable prediction
→ append-only 落库
→ deployment acceptance 再跑真实链路
        ↓
C4 realtime deployment = LIVE_SHADOW        ← 仅此步
        ↓
Forward evidence accumulating
        ↓
≥20–40 trading days
        ↓
FORWARD_VALIDATED / FAIL
```

**重要**：第一条真实事件成功落库，只能把 `C4 realtime deployment` 从 `PENDING` 升为 `LIVE_SHADOW`，**不能**把模型升成 `FORWARD_VALIDATED`。`FORWARD_VALIDATED` 仍必须等已冻结的前瞻观察期（≥20–40 trading days）与 calibration/drift 闸门。这与开发路线一致：C4 只做影子概率、校准、缺失警告和归因，冻结后的新前瞻期才用于晋级；Corvus 这一阶段不能直接干预交易。

## 最终历史训练（v1.0.0，APPROVED）
- 训练域（冻结）：`P0 & label_valid==True & training_eligible==True & non-HEAVY & cluster-level & category!=DISTRIBUTION & ~corp_action_within_horizon`（CORPORATE_ACTION_WINDOW 排除）。
- 数据：全部合格 2019–2025；`n_train=148,384` 簇，`positive_rate=0.361`。
- 模型：LightGBM（`n_estimators=300, lr=0.05, num_leaves=15, min_child_samples=30, subsample=0.8, colsample_bytree=0.8, class_weight=balanced, seeds=[0,1,2]` 平均）。
- 产出：`corvus/materiality_v1/artifacts/model_manifest.json`（version **1.0.0**）+ `lgbm_materiality_s{0,1,2}.txt` + `feature_medians.json`。
- 登记 SHA256：模型文件 / feature_medians / 训练数据 / feature_contract / config 各一份。`1.0.0` 永久对应此模型；重新训练只能升 `1.0.1 / 1.1.0`，**不覆盖**。
- 四折 OOT（2022–2025）继续作为“配置在历史时间外能工作”的证据；训练后不再回头调参。

## C4 组件化与边界
- 共享特征实现：`corvus/materiality_v1/materiality_features_v1.py`（`build_features` + `MODEL_FEATURES` 52 维，冻结）；`feature_contract.json` 为可哈希契约。
- 评分域契约：`corvus/materiality_v1/scoring_domain.py` → `C4_SHADOW_SCORING_DOMAIN_V1`（P0 & non-HEAVY & mapping valid & available_at 可见 & feature 完整）；**刻意不使用 `label_valid` / `corp_action_within_horizon` / `later_realized_label`**（标签侧信息，仅事后评估用）。
- Feature Parity Gate：`corvus/materiality_v1/feature_parity_gate.py` 必过（列顺序/dtype/缺失/编码一致 + 截断无未来泄漏）；通过后写 `artifacts/feature_parity_passed.json`，`score_event` 加载模型前强制检查。
- `c4_shadow.py`：`CorvusInformationSnapshot` + `score_event`（加载 1.0.0，域门禁，raw LGBM proba，`calibration=HISTORICAL_OOT_VALIDATED_RAW_PROBA`，不回写标签）。

## C4 部署验收（C4_DEPLOYMENT_ACCEPTANCE_V1, PASS）
不涉及训练，仅验证实时运行语义。脚本 `corvus/materiality_v1/deployment_acceptance.py`，通过写 `artifacts/c4_deployment_acceptance_passed.json`。`score_event` 启动前须先过 Feature Parity Gate 与 Model Hash Verify。

| # | 检查 | 结果 |
|---|------|------|
| 1 | IDEMPOTENCY：同 (event_cluster_id, model_version) 不产生两个不同 prediction | PASS |
| 2 | APPEND-ONLY：realization 独立 append 到 `c4_shadow_realization`，prediction 表永不 UPDATE | PASS |
| 3 | MODEL HASH VERIFY：manifest 与本地 artifact hash 不一致 → scorer 拒绝启动（含负例） | PASS |
| 4 | EVENT-TIME ORDER：prediction_ts >= available_at 且 feature_asof <= prediction_ts | PASS |
| 5 | RETRY SAFETY：ingest 重试/重启不重复写多条 shadow prediction（unique key upsert） | PASS |
| 6 | FAILURE EXPLICIT：feature 缺失 / entity 无效 / 非 P0 → 显式 REJECTED/NOT_SCORED，不静默丢 | PASS |

- DB 唯一键：`(event_cluster_id, materiality_model_version)`（prediction 表）。同事件不同模型版本可并行 Shadow（1.0.0→A, 1.1.0→B）；同事件+同版本两条不同结果应拒绝。realization 表唯一键 `(event_cluster_id, materiality_model_version, label_horizon)`，独立 append。
- 参考实现：`corvus/materiality_v1/shadow_store.py`（`ShadowStore`：prediction 表 upsert by 唯一键永不改 + realization 表独立 append，两表物理隔离）。生产层见 `C4_SHADOW.md`「生产持久化」DDL。

### DISTRIBUTION 审计（无 train-serving mismatch）
计数审计：eligible（P0 & label_valid & training_eligible & non-HEAVY & ~corp）簇中 `category==DISTRIBUTION` 数量 = **0**。
即训练域（`category!=DISTRIBUTION`）与线上评分域（`scoring_domain` 检查 `category==DISTRIBUTION`）**一致地排除同一个切片**，且该切片在 eligible 集内本就为空 → 不构成 train-serving domain mismatch，无需重训。`model_manifest.json` 的 `train_domain` 字段已显式声明排除 DISTRIBUTION。

## 冻结依据（收官序列 A/B/C/D 结论）
- **A (canonical 有效性)**：`corp_action_within_horizon == True` → `label_valid=False` / `label_invalid_reason=CORPORATE_ACTION_WINDOW` / `training_eligible=False`（364,305 cluster）。已写回 `backfill_pipeline.py`，训练/C4/Pyxis 统一按 `label_valid` 过滤，不再各自硬编码排除。
- **B (cluster 粒度)**：`event_cluster.parquet`（1,288,240 行）+ `event_document_link.parquet`（4,231,335 行）。模型表不再靠 `drop_duplicates("event_cluster_id")` 兜底；C4 重复实时打分也不会产生多份 prediction。
- **C (UNMAPPED 审计)**：占公告 52.4% / cluster 代表性 44.1%；P0 模型只训 P0，UNMAPPED 强制 tier=P2 → **0% 进入 scoring 域**。内容：ROUTINE_DISCLOSURE 55.3% / TRUE_MAPPING_MISS 35.4% / POTENTIALLY_MATERIAL 8.8%（低于 mapped 的 18.9%）。结论 **tolerated_unknown**，非模型完整性风险。
- **D (available_at 精度)**：巨潮仅日期粒度；canonical 实际执行路径全为 `POST_CLOSE_NEXT_DAY`/`WEEKEND_NEXT_DAY`/`RESUME_FIRST_TRADE`（零 `PREMARKET_OPEN`），即“次日/复牌首个合法时点可见”保守可见性；`--assume-premarket` 禁用于冻结；`available_at_precision=DATE_ONLY`；无分钟级标签。
- **corp 排除前后四折 ROC 变化 ≤0.003** → 预测结构真实，非除权伪影。

## SUPERSEDED / AUDIT_ONLY（保留，不删除）
- `out/events_SUPERSEDED_V2_labeled.parquet` — 修复后标签落盘中间 dump（含 3.28x 公告冗余），已被 `event_cluster.parquet` 取代。
- `out/events_v1_pre_corp_fix.parquet` — 修复前标签备份（cluster 级）。
- `out/walk_forward_pre_fix_results.json` / `.md` — buggy 基线（诊断用，勿用于选模型）。

## 关键产物
- `out/event_cluster.parquet` — canonical 主表（冻结核心）
- `out/event_document_link.parquet` — 公告级文档链接
- `out/walk_forward_final_results.json` / `.md` — canonical FINAL（最后一次历史 walk-forward）
- `out/CORP_FIX_IMPACT_REPORT.json` / `.md`

## Final Gate 评估（冻结门槛）
ROC>0.5 ✓ ｜ PR-AUC > 当年 prevalence ✓ ｜ decile 单调 ✓ ｜ cal_slope≈0.9 ✓ ｜ 跨 4 折稳定 ✓ ｜ mapped-only（P0）即当前 OOT，UNMAPPED 不进入 ✓

→ 满足显式冻结门槛。`HISTORICAL_TUNING: CLOSED`；进入 C4 前瞻 Shadow，无需再追 ROC 0.72→0.75。

## C4 Shadow（已解锁）
见 `C4_SHADOW.md`。C4 只做冻结后的前瞻影子评分、校准与漂移监控，不改买卖。`direction / reflexivity / persistence` 保持隔离研究，Corvus 上线 Shadow 后也**不**为「字段完整」硬塞三个弱/未完成头。当前合法输出：
```text
p_materiality      = 0.xxx
direction_status   = EXPERIMENTAL_NOT_USED
reflexivity_status = UNAVAILABLE
persistence_status = UNAVAILABLE
```
这比伪造完整性强得多。

## C4 晋级条件（提前冻结，FORWARD_VALIDATED）
从 `FROZEN` → `FORWARD_VALIDATED` 的最低门槛（先把底线定死）：
- 最低前瞻观察：≥ 20–40 trading days，且样本量足够；
- 无严重 feature drift（Feature Parity Gate 持续 PASS）；
- 无 calibration collapse（ECE 未爆）；
- 高分 → 实际 materiality 保持基本单调；
- Brier 未明显恶化；
- 没有某单一 event family 撑起全部结果。

通过后才开始讨论 `Corvus C5 → Pyxis`，而非让 `p_materiality=0.9` 直接影响买卖。这与路线“先影子运行、连续记录至少 20–40 个交易日”的上线纪律一致。

## C4 前瞻报告（C4_FORWARD_REPORT，第一版只看这些）
对每条前瞻事件：`T0` 立即保存 `p_materiality / model_version / feature_snapshot / prediction_ts / evidence`；`T+1/T+3/T+5` 标签成熟后另表追加 `realized_materiality`。周期生成报告，第一版只看：
- `n_scored` / `n_labeled` / score distribution / realized base rate；
- ROC / PR-AUC / Brier；
- calibration bins / ECE；
- event-family breakdown / missing & reject rate / score drift；
- 高分失败案例 + 低分漏判案例。

**纪律：不再用历史 ROC 决定任何东西**；研究对象已切换为 2026-09-29 之后的新事件，每天看到结果不调模型。

## 实时 ingest → C4 上线判定
第一步（部署链路验收，非历史建模）是接实时事件流 → append-only Shadow → 跑 `C4_DEPLOYMENT_ACCEPTANCE_V1`。一旦第一条真实新事件通过 v1.0.0 留下**不可变** prediction，`C4 realtime deployment` 从 `PENDING` 升为 **`LIVE_SHADOW`**（仅此步，模型不晋级）；随后 deployment acceptance 用真实链路再跑一遍。从那一刻起，新数据才是下一次晋级最有价值的证据。

## 工程层长期健康监控（C4 上线后，非模型闸门）
用于区分「模型真的漂了」与「实时数据链坏了」，否则看到 calibration 恶化易误判原因：
- `ingest_to_prediction_latency`：从公告到落 prediction 的时延（链路健康）。
- `duplicate/retry_rate`：ingest 重试/去重比例（RETRY SAFETY 实际表现）。
- `not_scored_rate + reason distribution`：被拒比例与原因（tier / entity / feature / future）。
- `feature_missing/drift rate`：缺失填充触发率与特征漂移（Feature Parity Gate 持续 PASS 是其硬约束）。

这些指标**不影响模型冻结状态**；异常先查数据链，再查模型。
