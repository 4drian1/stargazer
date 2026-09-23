# Corvus 事实层规格 v1.2（2026-09-13 冻结）

> `effective_date`、`amount` 均属于证据约束型低信任字段。只有能够在原文中定位到直接证据、
> 且未经过 LLM 推断、补全或计算时，方可进入事件影响模型。容器文档中的字段默认不参与建模，
> 直到识别出具体底层事件。

## 五条不变量（架构级，不可绕过）

1. **未映射不等于中性**：映射失败 → `materiality_tier='UNKNOWN'` + `mapping_status='UNMAPPED_CONTEXT'`
   + `model_eligible=0`，进观察池（每周主题聚类、10–20% 人工复核、高频成簇后再议映射）。
   UNKNOWN 是正常状态，不为消灭 NEEDS_REVIEW 而扩字典。
2. **元数据成功不等于正文成功**：`detail_status`（官方尝试）与 `body_status`
   （FULL_TEXT / MIRROR_MATCHED / METADATA_ONLY / FAILED）分离；METADATA_ONLY/PARTIAL 不进自动风险判断。
3. **schema 通过不等于事实正确**：`SCHEMA_VALIDATED` 仅表示格式与字典检查通过；
   事实正确性以人工评审与证据约束字段为准。
4. **P0 必须通过否定语境和证据校验**：关键词+标题绝不能直接产出可建模 P0。
   P0 且 `model_eligible=1` 的必要条件：`assertion_status='AFFIRMED'` +
   body ∈ {FULL_TEXT, MIRROR_MATCHED} + 实体/类型映射成功 + 字段证据约束全过
   （"不存在/未涉及/已解除/否认/自查未发现/历史上曾"一律拦截或降级）。
5. **镜像正文必须保留双来源证据链**：`body_source='CNINFO_MIRROR'` + `body_hash` +
   `mirror_match_method` + `mirror_match_confidence`；匹配键 = 证券代码 + 公告日期(±1) +
   规范化标题相似度(≥0.6)（+PDF 哈希可得时）。

## 字段信任级别

| 字段 | 级别 | 规则 |
|---|---|---|
| published_at / available_at | 高 | 采集链确定性产物；可见性一律以 available_at 为准，与 effective_at 不得混用 |
| event_type_raw / entities / action | 中 | LLM 抽取，实体交叉核对（entity_in_text）+ 人工抽检 |
| canonical_event_type / materiality_tier | 中 | 关键词映射，P0 另过安全门 |
| effective_date | **证据约束** | 四层：raw/normalized/status/evidence；EXPLICIT 可用，INFERRED/CONFLICT 禁入模型，MISSING 正常；禁止 LLM 补年份 |
| amount | **证据约束** | 四层：raw_text/value/unit/type/status/evidence；仅 EXPLICIT 可用；CALCULATED（高精度合成数）直接禁入；P1 容器文档默认不参与 |
| assertion_status | 门控 | AFFIRMED / NEGATED / UNCERTAIN / HISTORICAL_REFERENCE，P0 仅 AFFIRMED |

## Qwen 提示词禁令（llm_extract.py 已生效）

- 原文没有明确完整年月日时 `effective_date` 必须返回 null；禁止按当前日期/发布时间/常识补年份；
- `amount` 只准复制原文直接出现的数字，禁止乘除、汇总、换算。

## 已知局限（接受，记录在案）

- 否定门对"主事件肯定+局部否定"的文本（如"新增诉讼…不存在未披露诉讼"）过度触发，
  方向安全（进人工评审），recall 损失待 P1 正文二次判定回收；
- 无单位数字（股数 vs 金额）语义需人工看 evidence 区分；
- 图片型 PDF 无文本层（文件可找回，抽取走 NEEDS_REVIEW）。

## 状态

- 50 条针对性复验：2026-09-13 **通过**（见 data/reports/qc_reverify_50.md）；
- 新闻影响模型：**不开训**，继续前瞻积累事件与生命周期数据；
- 外围线：F-v2.2 已完成，见 data/reports/global_baseline_c0c1c2.md。
