# T5 影子双名单监控设计 v1.1（讨论定稿，待开工指令）

> 状态：DESIGN_FROZEN_PENDING_IMPLEMENTATION。**收到用户开工指令前不改任何代码。**
> 来源：2026-09-23 用户与 Agent 讨论定稿；相关裁决见 开发路线.txt 2026-09-23 各变更记录。
> v1.1 增补（2026-09-23）：交易计划界面（本机专属）+ 删尾盘扫描 + 日终汇总 + 监控机迁移方案（§5-§8，用户已批准方向，开工前须先验证三依赖）。

## 0. T5 的身份（最高优先级，不得漂移）

```text
T5 = GREEN-v2.2 Shadow 的"健康状态观察器 / baseline scorer"
≠ Selection 模型
≠ 交易决策器
≠ v2.2 有效性的证明工具
```

- 冻结版本：model_version=2026-09-09 四头打分器，**永久钉死**。
- **禁止**用 T5 过滤差候选来证明 v2.2 有效（ALPHA RESCUE = FORBIDDEN 的 T5 版）。
- 前瞻输出**不得**反哺调 T5 或规则层参数；出现"模型说买结果涨了"类轶事只进假设日志。
- T5 与用户的**分歧**是最有价值的数据（已知前科：DIP_RECLAIM 5 例人工下杀买入 5/5 全 REJECT——期待值按"分歧记录器"校准，不按"准不准"校准）。

## 1. 每日双名单制（核心结构，永不合并）

| | 规则层名单 | 模型名单 |
|---|---|---|
| session_tag | `green_watch_daily`（现有） | `model_shadow_daily`（新建，同候选结构） |
| 候选来源 | coverage 池（build_pool 冻结代码）tier/score top-N | T5 冻结版对**全 eligible universe** 打分取 top-K |
| source 标签 | `PCLOSE_RULE` | `MODEL_SHADOW`（头部强制标注"未验证模型输出"） |
| 定位 | 正式交易计划候选：你讨论→冻结→监控机监控 | **纯观察**：不进冻结计划、不接监控推送 |
| 落库 | stargazer_plan 现有链路 | 新 session，同 plan_candidate/trigger_rule 结构 |

呈现形态：两张**并排**清单，重叠票高亮（双认 vs 各认各的本身就是信息）；规则名单带 T5 分数列。

### 模型名单三条 caveat（写入名单头）
1. **口径偏移**：T5 原生为盘中决策点打分器（AM/PM 双头），盘后打次日名单 = AM 头特征近似，标注 "out-of-native-regime approximation"，评估"训练是否符合预期"时计入折扣。
2. **无 edge 声明**：与规则名单并排 ≠ 被验证；STEP1 COVERAGE_ONLY 裁决对一切"排序器名单"适用，模型名单只是另一个待观察排序器。
3. **人工从模型名单挑票买入 = 自由**（人工轨迹），落库 source 标人工；但 T5 分数不构成任何买卖理由代码。

## 2. 三个功能（定稿形态）

### 功能 1：盘中触发 → T5 打分 → 建议（v1 = 盘后批量版）
- 监控机照旧只跑规则层 DSL 触发 + 微信推送（现有链路不动）。
- 当日 events jsonl 拷回本机后，T5 对本日所有触发票批量打分 → 次日晨会给你：每只触发票的 T5 四头分数/象限 + 买/卖节点建议。
- **实时版（T5 跑监控机、触发后分钟级出分）明确推迟**：监控机迁移包无 T5 模型文件与 pullback_train 脚本（15GB 打包纪律），等于第二条迁移通道；待 shadow 数据证明 T5 分歧有信息后再立项。

### 功能 2：持仓 + 观察名单持续影子判断（买卖节点提示）
- 基建复用：T5 `--score-live` 脚本、shadow_decisions 表、exe model-minute-monitor 都在（旧调用随 3% 规则摘除，接口未删）。
- 每日盘后批量：你的持仓 + 观察名单逐只过 T5 → 输出买/卖节点建议（同样次日晨会材料，v1 非实时）。
- **卖出优先级硬契约**：`HARD_RISK_FLOOR（冻结灾难底线） > T5 卖出节点 > 人工裁量`。真实卖出落库照旧选 risk_type 九枚举；T5 节点只能进 evidence_ref 当证据。

### 功能 3：盘后双名单（见 §1）
- 名单产生逻辑规则层一行不动；T5 名单独立生成、独立呈现。

## 3. 工程实现清单（开工时按此执行）

1. `model_shadow_daily` draft 生成器：读 coverage/全 universe 候选 → T5 批量打分 → draft_gen 同款落库（session_tag 不同）。
2. T5 盘后批量打分脚本（本机 .venv，torch 可用；批量版 `--score-batch`，输入=候选列表+日期，输出四头分数+象限）。
3. 每日输出格式：双名单并排 markdown（含 T5 分数列、重叠高亮、三 caveat 头）。
4. 持仓/观察名单打分任务：从 PG 持仓/trades + watchlist 读入，同 2 复用。
5. 全部 T5 输出走 Snapshot 契约（`opportunity_v2/contracts/snapshot_contracts.py`），frozen_artifact_ref=`T5FOURHEAD:v20260909:<hash>`。
6. 监控机零改动（v1 批量版前提下）。

## 4. 验收（开工后）
- 连续 5 个交易日出双名单+持仓打分，无人工干预可全链重跑。
- 分歧样本（规则买/T5 REJECT、T5 FOCUS/你未买）自动计数入假设日志。
- 冻结检查：T5 权重文件 hash 每日校验，变动即报警。

---

## 5. 交易计划界面（本机专属，2026-09-23 用户裁定）

**只在本地 exe 运行**；监控机保持无界面纯监控。计划产物照旧人工拷贝（frozen JSON → 监控机 dist\data\plan_frozen.json）。

流程：
1. 侧栏/顶栏入口「交易计划」页 → 点**盘后选股**按钮；
2. 后端跑全链：数据增量检查 → build_pool（规则名单）→ T5 批量打分（模型名单）——全部本机 subprocess；
3. 页面两列并排展示：规则层名单（tier/score/触发价公式）∥ 模型名单（T5 四头分数/象限/建议节点），**重叠票高亮**，两名单各带模型名单三条 caveat 头（§1）；
4. 你勾选/改价/否决（否决填理由）→ 点**生成计划** → 组装 draft.json（含 dialogues 证据）→ create_draft 落 PG；
5. 你确认后冻结 → export_frozen → 显示文件路径，你拷去监控机（沿用现有拷贝流）。

技术形态照 desktop 惯例：Flask 路由 + 线程 subprocess（plan_bridge 模式）+ 原生 JS 页面；监控机 config 隐藏该页（或该页在监控机无意义，因盘后全链不在监控机跑——开工时按"界面只在有全链的机器显示"处理）。

## 6. 删除 14:30 尾盘扫描（用户裁定"可以删掉了"）

- 删：UI 按钮 + `/api/sector_pullback/scan` 路由 + run_afternoon_scan 调用链 + 盘前助手中依赖该 watchlist 的推送引用。
- sector_pullback.py 中 scan 相关函数：死代码清除还是保留模块（其 realtime 板块设置 UI 仍在用 save/load_user_sectors），开工时清理并在此登记处置。
- **唯一保留**：实时监控板块的板块设置（user_watch_sectors 增删）与 watchlist.json 读取兼容。

## 7. 盘中影子监控日终汇总（用户裁定：无结果也要报告）

- 监控线程 15:05 最后一轮后强制发一条**日终汇总**：有触发="今日计划监控：候选 X 只，触发 N：<代码+时刻+类型> 清单"；无触发="今日计划监控：今日无触发，X 只候选已全程记录"。
- 通道：微信一条 + `data/plan_daily_summary_YYYYMMDD.json` 落盘（events jsonl 之外的日终级记录，拷回本机归档用）。
- 现有每轮日志不变（审计留痕）。

## 8. 监控机迁移方案（用户已批准方向："重新打包重新替换"；执行前三依赖先验证）

**打包原则：选择性，非 15GB 全量**——当年排除的是整个 pullback_train 目录（训练数据/分钟缓存占大头）；T5 投产只需 `intraday_t5_features.py` + 冻结模型文件（MB 级）+ 现成 .runtime 依赖。

**三依赖（开工/打包前必须先探明，缺一不可）**：
1. **监控机 stockdb 分钟数据新鲜度（命门）**：T5 特征需分钟数据且 ≥20 个前交易日历史拼接；监控机现有的是日线刷新链路（v5_9_portable_refresh），分钟数据从哪来、覆盖多全，直接决定监控机上 T5 分数可不可信。探明结果登记于此。
2. **版本一致性**：两机同跑冻结模型，权重文件 hash 每日校验（已在 §4 验收），迁移包内带 hash 清单。
3. **职责边界不动**：监控机升级后 = 监控 + 打分 + 推送 + 日终汇总；盘后全链（池/双名单/draft/冻结/导出）仍本机产出、人工拷贝。禁止把研究链路搬上监控机。

**迁移步骤**：①用户把监控机当前数据发本机 merge（corvus/events 等，走 _slave 惯例）→ ②本机探三依赖（主要是分钟数据，可远程指导用户在本机 stockdb 验证同等口径）→ ③出打包清单（exe + T5 推理包 + 部署说明增量版）→ ④用户全量替换。

## 9. 两机职责终态（本设计全部落地后）

| | 本机 | 监控机 |
|---|---|---|
| 盘后全链（池/双名单/draft/冻结/导出） | ✓ 唯一 | ✗ |
| 交易计划界面 | ✓ 唯一 | ✗ |
| 盘中规则触发监控 + 微信推送 | 可跑（备用） | ✓ 主 |
| T5 打分 | ✓ 盘后批量 + 次日晨会 | ✓ 实时（迁移后） |
| 持仓/观察名单影子判断 | ✓ 盘后批量 | ✓（迁移后，可选） |
| 日终汇总 | — | ✓ |
| PG/研究/训练 | ✓ 唯一 | ✗ |

---

## 10. v1.1 实现登记（2026-09-23，双代理交付，待 exe 构建完成后本机投产）

- 交付：`pullback_train/t5_batch_score.py`（批量打分，10 worker，dp=1500 双头，schema 冻结）、`green_v2/model_shadow_draft.py`（全 universe→top20→model_shadow_daily session，trigger 全空）、`green_v2/holdings_watch_score.py`（持仓/观察名单 advisory）、`t5_freeze_manifest.json`（sha256_12=71513c0df8b9，篡改实测拒绝）。
- desktop：「交易计划」页（盘后选股→双名单→勾选改价→create_draft→freeze+export 直写 data\plan_frozen.json）、尾盘扫描已删（板块设置保留）、15:05 日终汇总（微信+json+SSE，幂等）。
- **实测（20260918）**：全量 6345/6650（8.2min；305 失败全为北交所，stockdb 无其分钟数据）；model_shadow session=44 落库干净；双名单 md 已产出（当日规则∥模型重叠=0，真实结果）。
- 实现偏差登记（不改设计意图）：①universe 加股票前缀过滤（60/68/00/30/8/4/92）——T5 是个股打分器，ETF 混入无意义（5282 只）；②排序键=PM 头象限优先 FOCUS>WAIT>WATCH>REJECT（设计未钉死，要改只动 rank_model_list）；③STOCKDB_ROOT 默认路径本机不存在，批量脚本 import 前自动补候选（冻结模块内部未动）。
- 坑入档：create_draft 幂等锚=(version_id,sort_order,code)，**同日内容变更重跑会追加污染 session**（session 42 已删重建为 44；今后改内容须换日期或先删 session）；Windows spawn worker 不继承全局变量，date/dp 经 initializer 传。
- 监控机部署：config 设 `plan.enabled: false` 隐藏交易计划页；监控机迁移（§8 三依赖）未动，待令。
