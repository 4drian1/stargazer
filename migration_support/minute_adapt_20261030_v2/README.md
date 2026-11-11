# 分钟适配 v3：4项接线修正 + 明确P/E重放（路线 18.176）

承接 [v2审阅](D:/stargazer/migration_support/minute_adapt_v2_review_20261030_v1/REVIEW.md)。
**保留 v2 已修进展**（21项守卫全部保留并通过），只补 4 项。
固定9样本未变；未扩样、未重训、未打包、0 HTTP/付费/fit/实推。

重放（v3 快照）：**9/9 建数组、9/9 接口一致、0 错误、9/9 源版本校验通过**（`EVIDENCE_v3.json`）。
真实9样本联合表验收（v4 冻结）：**9/9 建数组、9/9 接口一致、0 错误、0 源版本变化**，`overall_pass=True`，并产出 `JOINT_TABLE.parquet`（7312 行 × 145 列，按 `code+cutoff_index` 连接，缺行原位 NaN）。
守卫 **52/52 PASS**（原21 + 4项15 + 验收收口12 + 联合表收口4）。
证据：`EVIDENCE_v3.json`/`RECEIPT_v3.json`（18.176 早期快照，9/9）、`EVIDENCE_v4.json`/`RECEIPT_v4.json`/`JOINT_TABLE.parquet`（当前源版本冻结 + 联合表）；旧 `EVIDENCE.json`/`RECEIPT.json`（18.175）保留不动。

> **总验收收口（用户复现的漏洞）**：原本 SHA 失配样本从"分母"扣掉，导致 9/9 全部源变化、构建0、接口0 仍 `overall_pass=True`。现已改为**硬失败**：固定样本全部完成、接口全部通过、**源版本变化必须为 0**，整轮才 PASS；失配样本绝不进分母。未变化子集成绩由 `acceptance.unchanged_subset` 另列。回归测试覆盖"全部计算前失配 / 单个计算前失配 / 计算后失配 / 干净全过 / 含 STALE 但无变化"。

> **联合表三处收口（真实入口复现）**：
> 1. **失败即整轮失败**：原先连接前就写成功证据、parquet 无覆盖保护；现改为**先完成联合表的构建/校验/落盘，再写成功回执**，任一步失败都记 `RUN_FAILED_*` 并 `return 1`，成功证据与回执不落盘；且 `JOINT_TABLE.parquet` 纳入输出预检，已存在即拒绝覆盖。
> 2. **连接键与元数据校验**：要求 `meta` 与股票集合**完全一致**、`calendar_id` **非空**、每股 `cutoff_index` **唯一且非空**，并使用 `merge(validate='one_to_one')`；不同源/起点/频率或重复键 → `CALENDAR_MISMATCH`/`META_STOCK_SET_MISMATCH`/`CALENDAR_ID_EMPTY`/`JOIN_KEY_NOT_UNIQUE`/`JOIN_KEY_NULL`（不再产生 2×2=4 行的错误展开）。
> 3. **越界索引拒绝**：`decode_cutoff` 拒绝负数、非整数、越界、`5m`/`15m` 之外频率，**不再 clip 到末日**；3 天 15 分钟日历（索引 0–47）下 `decode_cutoff(48)` 抛 `CUTOFF_OUT_OF_BOUNDS`。

> **源版本与快照的关系**：当前磁盘 SHA 与 `EVIDENCE_v3` 旧快照不同，**不代表新一轮必然失败**。入口会在本轮重新采样并冻结当前文件 SHA；真正要阻断的是**本轮采样→计算之间**发生变化（这正是 `SOURCE_VERSION_CHANGED` 捕获的）。`EVIDENCE_v3` 旧快照继续保留即可。v4 即按此对当前源重新冻结。

---

## 一、4项修正

### #1 P/E 参考日不再混淆（P1）
引入显式三元组，语义不再靠字段名默认：

| 字段 | 含义 | 本轮取值 |
|---|---|---|
| `data_day` P | 数据截止日 | **20261029** |
| `monitor_day` E | 被准备的监控日 | **20261030**（声明值，非日历推导） |
| `reference_day` | 实际出指标的参照日 | **= P = 20261029** |

- `NEXT_DAY_PREP`：`reference = P`，并强制校验 `prev(E) == P`，否则 `P_E_NOT_ADJACENT`。
- `REPLAY`（盘中回放，显式给 `observation_day=O`）：`reference = prev(O)`。**10/29 盘中回放用 10/28 是对的，但那是另一种模式**，不会再由"P日"的默认值走到。
- 未声明 → `NO_P_E_DECLARED`，不输出任何昨日资格。

旧代码把 daily_parts 末观测日当 P 再取前一日，得到 10/28，属"退一天"。现已修正，9/9 样本参照日均为 **20261029**。

**日历口径限定**：规范日历止于 20261029，这是 daily_parts 的**数据覆盖截止**，**不是**"20261030 非交易日"的结论。E=20261030 为声明值（当日挂钟日期），不冒充日历推导。日历末日、各源覆盖末日、声明 P/E 已分列记录。

### #2 冲突重复记录隔离（P1）
- `build_minute_arrays` **先按 ts 分组**再写入：同一 ts 的多条记录若**任一值不一致** → 判冲突，该 5 分钟槽置 NaN，其 15 分钟组随之不完整（不再保留首条）。完全一致的重复仅去重、不算冲突。
- `rejected.conflicting_ts` 单独计数；`quality.conflicting_timestamps` 含冲突。
- 新增 `allowed_days`：builder **自身**拒绝未通过 `classify_file` 的日子，不再依赖调用者先过滤；runner 同时按 `eligible_days` 过滤并记录 `excluded_ineligible_days`。
- 反例（同 ts 两种 close）：`conflicting_ts=1`、槽为 NaN、15 分钟组为 NaN、`close=10.5` 不再保留。

### #3 缺槽对拍预期统一掩码（P2）
`interface_consistency_test` 先算 `group_valid`（每组 **3 根 × 全部必需字段** 均有限），所有期望数组在**同一掩码**下构造：
- `agg_open/agg_close/agg_high/agg_low/agg_volume` 全部 `where(group_valid, expected, NaN)`；
- `incomplete_group_absent` 改为与完整 `group_valid` 对拍，不再只看首根 close；
- `day_return` 匹配真实内核：**只有全日 16 根均有效**才形成次日参考价。
- `NOT_COMPARABLE` 单列为"不可比"，既不记通过也不记失败；**全部不可比则判失败**。比较器未放宽回有限值交集。
- 覆盖：缺首槽 / 缺中槽 / 缺尾槽 / 单字段坏值 / 整日缺失，接口测试均一致。删 09:40 后正确生成的 NaN 不再被判不一致。

### #4 采样→计算 SHA 硬校验（P1）
- 新增 `verify_source_version(path, frozen_sha)`：重算哈希与冻结值比对。
- runner 在**计算前**强制比对；不一致 → 该样本 `SOURCE_VERSION_CHANGED`，**停止计算**，记录新旧 SHA，绝不用旧 SHA 描述新字节。计算**后**再校验一次。
- 单次读取前后哈希一致只证明那次读是原子的，不证明两次读取之间没变——已按此补硬校验。
- **输出预检前置**：开工前即检查 `EVIDENCE_v3/RECEIPT_v3` 是否已存在，冲突立刻中止，不再跑完全程才报 `FileExistsError`。
- 本轮 9/9 前后均匹配（含 002051）。

**本轮重新冻结**：源文件仍在写入，002051 由 136,704 → **159,744** 字节（sha `b8e2c7ee…`），指数两文件哈希也变了。故本轮**另发 v3 证据绑定新版本**，不沿用旧 SHA。名单仍固定，未用其它票替换。

## 二、重放结果（P=20261029，E=20261030，参照=20261029）

| 样本 | 板块 | 结果 | 覆盖(日历457) | 末日 |
|---|---|---|---|---|
| 000001.SH | 沪指 | **OK** | 457 | 20261029 |
| 399006.SZ | 创业板指 | **OK** | 457 | 20261029 |
| 600000.SH | 主板 | **OK** | 88 | 20261029 |
| 603601.SH | 主板 | **OK** | 88 | 20261029 |
| 002051.SZ | 主板 | **OK**（本轮已补齐） | 103 | 20261029 |
| 688533.SH | 科创板 | **OK** | 88 | 20261029 |
| 688798.SH | 科创板 | **OK** | 88 | 20261029 |
| 300215.SZ | 创业板 | **STALE** | 74 | 20261009 |
| 301195.SZ | 创业板 | **STALE** | 74 | 20261009 |

7 个样本在参照日 20261029 输出 `rsi6/rsi15/rsi14/dif/dea`，末根与低点根**逐点** `eligible=True`；2 个 STALE 保留在分母但不输出可用于次日的资格。
真实数据未触发冲突/离格/坏值（计数全 0），相应分支由合成反例覆盖。

镜像对照：TDX 段 453 天逐位一致（maxdiff 0.0）；公开源 5 天 `[20261026..20261030]` 按回执排除。已绑定 sidecar 来源契约、镜像 SHA、sidecar SHA；**该分段是来源证据推断，不是供应商认证**。

## 二（补）、正式联合表（route 18.176）

`joint_table.py`：按 `(code, cutoff_index)` 外连接各股特征帧。已通过**真实 9 样本联合表验收**（v4 冻结）：`JOINT_TABLE.parquet` = 7312 行 × 145 列，`cutoff_index` 唯一非空、`calendar_id=daily_parts@…epoch20261029…` 非空、`start=20241209`、`freq=15m`。`n_missing_cells=692258` 由**缺行 + 指标预热（RSI/EMA warmup）+ 其它字段未知**共同构成，不能全部称为"缺行数量"——此措辞仅影响描述，不影响对齐结论（见下注）。

`cutoff_index` 是内核丢 NaN-close 行后保留的**全局价组位置** = 共享 15 分钟数组中的 `(全局日序×16 + 槽位)`，因所有样本都建在同一共享日历 `cal` 上，可反解为具体 `(交易日, 槽位)`，故它是跨股对齐键，而非"第几行"。

- **连接前必须校验**（全部硬报错，不静默）：`meta` 与股票集合**完全一致**；`calendar_id` **非空**；所有样本共享**同一日历版本 + 起点 + 频率**（`CALENDAR_MISMATCH`）；每股 `cutoff_index` **唯一且非空**（`JOIN_KEY_NOT_UNIQUE`/`JOIN_KEY_NULL`）；合并用 `validate='one_to_one'`（重复键不会展开成 4 行）。
- **缺行原位保留 NaN**：某股某根 bar 在另一股存在、本基金无观测 → 该行该股列整体 NaN，不按位置补 0 或回填。
- 每列按 code 加前缀（`stock_rsi14_15m_raw_000001.SH` 等），避免合并后缀歧义。
- `decode_cutoff(cutoff_index, cal, freq)` 反解 `(day, slot)`；拒绝负数、非整数、越界、非 `5m/15m` 频率，不 clip 修正。
- `run_selfcheck` 在**整轮 PASS 且源版本全校验通过**时，先完成联合表构建/校验/落盘，再写成功证据与回执；任一步失败即记 `RUN_FAILED_*` 并 `return 1`，成功证据不落盘；`JOINT_TABLE.parquet` 纳入输出预检，已存在即拒绝覆盖。

## 三、v4 冻结登记与存储计账（route 18.176 正式验收）

v4 为本路线**当前源版本冻结 + 9 样本联合表对齐**的不可变证据，独立核验通过（用户）：
- 52/52 守卫通过；
- 联合表确为 7312 行 × 145 列，索引唯一、完整覆盖共享日历、无越界；
- 9 份源文件计算前后 SHA 均匹配；
- 逐股重算后，全部 144 列与已保存联合表逐值一致，缺失位置一致；
- 9 个样本参照日 5 分钟 / 15 分钟指标资格均为 OK。

**产物（只读、不得覆盖）**：`EVIDENCE_v4.json`、`RECEIPT_v4.json`、`JOINT_TABLE.parquet`（2,648,587 字节）。
**下游入口**：`JOINT_TABLE.parquet` 作为 `risk_r0_v1/joint_shadow_context_v1.build_features(snapshot, candidates, feature_cache=…)` 的 `feature_cache.stock(code, days, cutoff)` 输入源之一，按 `(code, cutoff_index)` 对齐；**仅作 9 样本接口验证，不充当全市场风险输入**。既有的股票覆盖、板块成员、特征顺序（`fields232/266` 来自冻结模型 `lightgbm_preprocessing.json`，漂移即 `FROZEN_FIELD_LIST_DRIFT`）、缺失处理（`native_missing_view` 保留 NaN、不删除）一律保留，未改。

**存储计账**：联合表 2,648,587 字节已计入 `risk_r0_v1` 所属 `REGISTERED_RESEARCH_BUDGET_V2`（4 GiB 上限，见 `stargazer_ml/v5_hybrid/risk_r0_v1/contracts/research_storage_4gib_20261015.json` 的 `registered_artifacts`），并在本目录 `STORAGE_LEDGER.json` 留痕（路径 / 字节 / sha256 / 冻结时间 / 路线 / 消费方）。

## 四、仍未完成（正式接线前）

- 生产应把**合法交易日历 / 各数据源覆盖末日 / 当前预期 P/E** 分列（本轮已分列，但官方日历仍需独立来源）。
- 陈旧两票（300215/301195）待补齐或显式降级；`WAIT_HISTORY` 分支无真实触发样本。
- 日线 qfq/gbbq 与股本/ST 三项限定仍并行推进，`pending_gbbq` 保留。

## 五、复跑与对接

```
python test_minute_adapt.py            # 52 守卫（分钟适配 + 联合表收口）
python run_selfcheck.py                # 写 EVIDENCE_v4/RECEIPT_v4/JOINT_TABLE.parquet（已存在则拒绝覆盖）
python test_risk_feature_integration.py  # 9 样本：联合表→risk_r0_v1 接口契约验证（覆盖/板块/特征序/缺失）
python e2e_fixed_cutoff_acceptance.py   # 固定截止端到端：截断/篡改截止后数据不改变当刻结果
```

**可认定**：P/E 与参照日语义已分离并校验；冲突槽隔离；缺槽对拍在统一掩码下正确；采样到计算有 SHA 硬校验；验收收口（源变化即整轮失败）；联合表连接键/元数据严格校验（meta=股票集、calendar_id 非空、键唯一非空、one_to_one）；越界索引拒绝不 clip；联合表失败即整轮失败、parquet 预检拒绝覆盖；真实 9 样本联合表已落盘且 `cutoff_index` 解码全在界内、缺行 NaN 原位保留；本轮 v4 证据绑定的是当前文件版本（重采样冻结 SHA）；联合表已登记为 `risk_r0_v1.build_features` 的 `feature_cache` 输入源且经 9 样本接口验证（覆盖/板块/特征序 `fields232/266`/缺失处理均保留，未改冻结模型）。
**不能认定**：当前磁盘源已变化，重跑会 FAIL（见上注，这是期望行为）；联合表为 9 样本接口验证，未充当全市场风险输入；完整风险迁移与昨日监控可用性最终验收尚未完成。
