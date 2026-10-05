# Corvus 信息模型 · 使用与说明文档（materiality_head_v1 第一阶段）

> 适用版本：`CORVUS_MATERIALITY_HEAD_V1` / 模型 `v1.0.0`（FROZEN）
> 配套冻结登记见 `MATURITY_STATUS.md`；前瞻影子评分细节见 `C4_SHADOW.md`。
> 本文档面向「如何接入与使用」，训练流程见 `CORVUS_TRAINING_PIPELINE.md`。

---

## 1. 这个模型是什么

Corvus 是 Stargazer 体系里负责**信息影响**的模块。`materiality_head_v1`（重大性头）回答一个**单一、保守**的问题：

> **这条公告/事件，是否可能产生显著的市场反应（materiality）？**

它**只输出一个概率** `p_materiality ∈ [0,1]`，附带校准状态、缺失告警与证据 id。它**不能**回答：

- 方向（利好还是利空）→ 由 `direction_head` 负责（当前 `EXPERIMENTAL_NOT_USED`）；
- 反射性 / 持续性 → 由 `reflexivity_head` / `persistence_head` 负责（当前 `UNAVAILABLE`）。

因此 **`p_materiality` 高 ≠ 应该买入**。第一阶段它只做**前瞻影子评分（shadow）**，不直接改变任何交易决策。

### 四个头的阶段状态

| Head | 状态 | 说明 |
|---|---|---|
| `materiality_head_v1` | **FROZEN / ACTIVE SHADOW** | 模型 v1.0.0，已上线影子运行 |
| `direction_head` | `EXPERIMENTAL_NOT_USED` | 未完成，不参与 |
| `reflexivity_head` | `UNAVAILABLE` | 无实现，不参与 |
| `persistence_head` | `UNAVAILABLE` | 无实现，不参与 |

合法输出示例：

```text
p_materiality      = 0.xxx
direction_status   = EXPERIMENTAL_NOT_USED
reflexivity_status = UNAVAILABLE
persistence_status = UNAVAILABLE
```

---

## 2. 架构与组件

```
实时公告
  ↓  raw document persist
  ↓  event clustering（同票同日合并为 event_cluster_id）
  ↓  entity / taxonomy（code 映射、category 路由）
  ↓  available_at（公告日；DATE_ONLY 粒度）
  ↓
production_chain.py  ── 仅编排 C4 拥有的段 ──
  ↓
scoring_domain  (C4_SHADOW_SCORING_DOMAIN_V1 门禁)
  ↓
materiality_features_v1  (build_features, 52 维, 唯一共享实现)
  ↓
model v1.0.0  (3× LightGBM 平均, 冻结)
  ↓
CorvusInformationSnapshot  (p_materiality + 元数据)
  ↓
shadow_store  (双表不可变持久化)
  ↓
c4_shadow_prediction  ──  永不 UPDATE
c4_shadow_realization ──  标签成熟后独立 append
```

核心代码位置：`corvus/materiality_v1/`

| 文件 | 职责 |
|---|---|
| `materiality_features_v1.py` | **唯一**特征实现 `build_features` + 冻结 `MODEL_FEATURES`(52) / `TOP_CATEGORIES` |
| `scoring_domain.py` | `C4_SHADOW_SCORING_DOMAIN_V1` 契约与门禁 `is_scoring_eligible`（刻意不用标签侧字段） |
| `feature_parity_gate.py` | 上线前必过：离线/在线特征一致性 + 截断无未来泄漏 |
| `train_final.py` | 冻结历史训练（产出 v1.0.0 artifact） |
| `c4_shadow.py` | `CorvusInformationSnapshot` + `score_event` + `MaterialityModel`（启动强校验 parity + hash） |
| `shadow_store.py` | 双表不可变参考实现 `ShadowStore` |
| `production_chain.py` | 最薄生产链编排（只编排上面 C4 段，不加载/不修改模型） |
| `deployment_acceptance.py` | `C4_DEPLOYMENT_ACCEPTANCE_V1` 6 项运行语义验收 |
| `artifacts/` | 模型 + 中位数 + manifest + parity/acceptance 凭证 |

---

## 3. 快速使用

> 运行环境：`.c3venv\Scripts\python.exe`（或 `prepare_env.bat` 选定的解释器）。
> 依赖：`out/event_cluster.parquet`（canonical 主表）已在训练/回填阶段产出。

### 3.1 给单条事件打分

`cluster_record` 需包含（均可由 available_at 前信息得到）：
`event_cluster_id / code / title / announce_date / label_start / n_announcements_in_cluster / sigma250 / gap_return / volume_shock / category / tier / attribution_quality`。

```python
import sys; sys.path.insert(0, ".")
from c4_shadow import score_event

rec = {"event_cluster_id": "000049a94269", "code": "000049", "title": "关于回购股份的公告",
       "announce_date": 20260923, "label_start": 20260926, "tier": "P0",
       "category": "BUYBACK", "attribution_quality": "CLEAN",
       "n_announcements_in_cluster": 1, "sigma250": 0.03,
       "gap_return": 0.0, "volume_shock": 1.2}

snap, diag = score_event(rec, as_of_date=20260929)
print(snap.p_materiality, snap.calibration_status)   # 例如 0.3694 / HISTORICAL_OOT_VALIDATED_RAW_PROBA
print(diag)   # {'eligible': True, 'problems': []}
```

- 域门禁不通过时 `eligible=False`，`p_materiality=None`，`calibration_status` 为 `DOMAIN_INELIGIBLE` 或 `DOMAIN_OR_FEATURE_FAILED`，并给出 `problems`（如 `tier!=P0` / `entity_unmapped` / `category==DISTRIBUTION` / `available_at_in_future` / `feature_incomplete`）——**绝不静默丢事件**。

### 3.2 走最薄生产链（含持久化）

```python
from corvus.materiality_v1.production_chain import run_shadow_once
from corvus.materiality_v1.shadow_store import ShadowStore

store = ShadowStore()
r = run_shadow_once(rec, store=store, as_of_date=20260929)
print(r["action"], r["p_materiality"], len(store))   # SCORED 0.3694 1
# 重试/重启同一事件 → 仍只有 1 行（RETRY SAFETY / IDEMPOTENCY）
run_shadow_once(rec, store=store, as_of_date=20260929)
print(len(store))   # 1
```

返回值 `r` 含 `action / eligible / reason / p_materiality / model_version / feature_asof / missing_warnings / latency_ms / record`——可直接喂给下面第 6 节的四个工程健康指标。

### 3.3 标签成熟后追加 realization（独立表）

```python
store.attach_realized_label("000049a94269", "1.0.0", "T+5", realized_label=True)
# 写入 c4_shadow_realization；prediction 表行不变（APPEND-ONLY）
```

### 3.4 上线前必跑的两个校验

```bat
REM 1) 特征一致性 + 截断无泄漏（写 artifacts/feature_parity_passed.json）
python corvus/materiality_v1/feature_parity_gate.py

REM 2) 运行语义验收 6/6（写 artifacts/c4_deployment_acceptance_passed.json）
python corvus/materiality_v1/deployment_acceptance.py
```

`score_event` / `MaterialityModel` 加载前**强制**要求 parity 凭证存在，且 **manifest 登记的 artifact SHA256 与本地文件一致**，否则拒绝启动（`MODEL HASH VERIFY`）。

---

## 4. 评分域边界（重要）

`C4_SHADOW_SCORING_DOMAIN_V1` 实时打分**只要求**：

- `event_tier == P0`
- `non-HEAVY`（attribution_quality != HEAVY）
- `category != DISTRIBUTION`
- entity 映射有效（code 非 NaN）
- `available_at` 已真实可见（announce_date <= as_of_date）
- 特征完整性满足（MODEL_FEATURES 缺失比例 ≤ 0.5）

**禁止在打分时使用任何标签侧字段**（这些要看未来 T+1..T+5，线上即未来泄漏）：

```
label_valid / corp_action_within_horizon / later_realized_label / training_eligible
```

训练域与线上评分域**一致地**排除了 `DISTRIBUTION`（计数审计该切片在 eligible 集内本就为空），不存在 train-serving mismatch。

---

## 5. 持久化：双表不可变（生产层）

将「不可变预测」由**数据库结构**保证，而非仅靠代码纪律：

| 表 | 主键 | 写入语义 |
|---|---|---|
| `c4_shadow_prediction` | `(event_cluster_id, materiality_model_version)` | 仅 INSERT；**永不 UPDATE** |
| `c4_shadow_realization` | `(event_cluster_id, materiality_model_version, label_horizon)` | 标签成熟后**独立 append** |

- 同一事件+同版本两条不同结果 → 拒绝（IDEMPOTENCY）。
- 同一事件允许不同模型版本并行 Shadow（1.0.0→A, 1.1.0→B）。
- 必须先有 prediction 行，才能 attach realization（顺序约束）。

内存参考实现 `ShadowStore`（`_predictions` / `_realizations` 两表）已对齐此结构；完整 PostgreSQL DDL 见 `C4_SHADOW.md`「生产持久化」。

---

## 6. 上线后长期监控（工程健康指标，非模型闸门）

用于区分「模型真的漂了」与「实时数据链坏了」：

| 指标 | 含义 |
|---|---|
| `ingest_to_prediction_latency` | 公告到落 prediction 的时延（链路健康） |
| `duplicate/retry_rate` | ingest 重试/去重比例（RETRY SAFETY 实际表现） |
| `not_scored_rate + reason distribution` | 被拒比例与原因（tier / entity / feature / future） |
| `feature_missing/drift rate` | 缺失填充触发率与特征漂移（Feature Parity Gate 持续 PASS 为其硬约束） |

这些**不影响模型冻结状态**；异常先查数据链，再查模型。

### 前瞻报告（C4_FORWARD_REPORT）

`T0` 立即存 `p_materiality / model_version / feature_snapshot / prediction_ts / evidence`；`T+1/T+3/T+5` 另表追加 `realized_materiality`。第一版只看：
`n_scored / n_labeled / score distribution / realized base rate / ROC / PR-AUC / Brier / calibration bins / ECE / event-family breakdown / missing & reject rate / score drift / 高分失败与低分漏判案例`。

**不再用历史 ROC 决定任何东西。** 研究对象已切换为 2026-09-29 之后的新事件。

---

## 7. 上线后禁止动作（从第一条真实 prediction 落库起）

不因短期结果发生：

- 不改 LightGBM 参数 / `TOP_CATEGORIES` / 52 维特征；
- 不重新选 threshold；不新增 Platt / isotonic calibrator；
- 不根据几天表现重训 v1.0.0；
- 不让 `p_materiality` 进入 GREEN / Pyxis 决策。

---

## 8. 状态边界（何时晋级）

- 第一条**真实、非 fixture**事件完成 `ingest → cluster/entity/available_at → v1.0.0 → immutable prediction → DB`，并在真实部署链上重跑 acceptance → `C4 realtime deployment` 升为 **`LIVE_SHADOW`**（**不必等 T+5 标签成熟**）。
- 之后持续积累前瞻数据；至少经过冻结的 **≥20–40 个交易日** 且满足 calibration/drift 闸门后，才有资格进入 **`FORWARD_VALIDATED`**，届时才讨论 `Corvus C5 → Pyxis`。

> 完整状态机与裁决见 `MATURITY_STATUS.md`。
