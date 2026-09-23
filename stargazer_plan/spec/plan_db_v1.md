# 第 4 步：最小交易计划数据库规格 v1.0（2026-09-15 冻结）

目标：把"人与 Agent 讨论 → 结构化计划 → 人工冻结 → 次日确定性监控 → 触发/失效 →
人工决策与成交 → 盘后结果与未执行反事实"完整落进同一个 PG 实例的独立 schema
`stargazer_plan`。这不是训练数据的生产线启动令，只是让未来的每一滴数据都流进可审计的库。

成功标准：**计划、提醒、人工决策、成交和结果能完整关联，且任何一步都可回放、可审计、不重复。**

## 0. 红线（当前阶段不能做）

- 不让 LLM 自行确认计划（冻结动作只有人工能做）；
- 盘中变化只能追加新版本，盘前冻结版本不可改（immutable）；
- 不自动下单；
- 不把"你更喜欢"当"收益更好"；
- 不从一两天对话直接训练；不接 DPO/RL；
- 收盘后禁止补写成"盘前已经知道"（所有时间戳由系统生成，不接受人工回填 past time）。

## 1. 表结构（9 张，全部幂等 DDL）

| 表 | 幂等锚 | 要点 |
|---|---|---|
| plan_session | UNIQUE(trading_date, session_tag) | 一次"明日计划"讨论=一个 session；dialogue_ref 指向对话来源 |
| plan_version | UNIQUE(session_id, version_no) | DRAFT→FROZEN→SUPERSEDED；frozen_at 由系统时钟写入；content_hash 防篡改审计 |
| plan_candidate | UNIQUE(version_id, sort_order, code) | 每条含 trigger/invalid/observe 三条件（JSON 结构化 + 确定性 DSL 字符串） |
| trigger_rule | UNIQUE(candidate_id, rule_type, rule_dsl) | rule_type ∈ TRIGGER/INVALIDATE/OBSERVE；condition_json 给人看，rule_dsl 给状态机执行 |
| model_recommendation | UNIQUE(candidate_id, as_of_time, model_name) | 影子模型输出，只记录不推荐 |
| human_decision | UNIQUE(candidate_id, decided_at, source) | APPROVE/REJECT/MODIFY/WATCH + 理由；source ∈ CHAT/PUSH/DESK |
| execution_log | UNIQUE(source, source_ref) | 人工成交记录；action 允许 NONE（决定不买也是记录） |
| outcome_attribution | UNIQUE(candidate_id, label_version) | 含 counterfactual_return（未执行反事实）与 counterfactual_basis；CENSORED_TAIL 不填 0 |
| dialogue_evidence | UNIQUE(session_id, message_ref) | 原始对话与结构化字段互相追溯的桥 |

金额/价格一律 nullable，禁止 0 占位。所有表 created_at 默认 now()。

## 2. 确定性状态机（次日监控）

输入：FROZEN 版本 + 触发/失效规则 DSL + 分钟线（新浪源缓存，与 Corvus 第 3 步同源）。
规则 DSL v1（只读价量，确定性、无随机、无网络）：

```text
price >= X            价格上穿
price <= X            价格下穿
pct_chg <= -Y%        当日跌幅超过
volume_ratio >= Z     分钟量/今日均量比
time >= HH:MM         时间条件（用于失效）
and / or / not        组合
```

输出：trigger_event（candidate_id, fired_at, rule_type, inputs_hash）。
**回放纪律**：同一份分钟输入必须产生完全相同的 fired_at 序列（状态机为纯函数，
inputs_hash = sha256(分钟数据区间+规则) 落库，回放时校验一致）。

## 3. 五种结局严格区分（验收硬门槛）

```text
EXECUTED        触发 → 人工批准 → 有成交记录
NOT_TRIGGERED   全天未触发（状态机输出为空）
DECLINED        触发 → 人工否决（human_decision=REJECT）
NO_DECISION     触发了但人工未响应（超时，区别于否决）
DATA_MISSING    分钟数据缺失无法判定（绝不猜）
```

## 4. 反事实（未执行计划）

- NOT_TRIGGERED / DECLINED / NO_DECISION 的计划，盘后按 Corvus 第 3 步同款口径
  计算 counterfactual_return（1/3/5 日）与 mfe/mae，basis 记
  `would_fill_at=trigger_price_if_fired`；
- 未触发的计划反事实基于"若触发价出现"的假设价，必须在 counterfactual_basis
  中显式记录假设，不得伪装成真实成交；
- 重叠事件同样记录（与 Corvus 的 overlap 口径一致：只标"事件后反应"不声称因果）。

## 5. 验收门槛（逐条可测）

1. FROZEN 版本 UPDATE 任何业务字段都被拒绝（触发器或权限层，二选一实现并记录）；
2. 盘中变更只能以新 version_no 追加；
3. 每条 candidate 三条件齐全（CHECK：trigger/invalid/observe 均非空）；
4. dialogue_evidence ↔ plan_version ↔ plan_candidate 可双向追溯；
5. 计划→提醒→人工决策→成交→结果五环外键关联完整；
6. 重复写入（同幂等锚）不产生重复行；
7. 分钟重放复现触发时间（inputs_hash 校验一致）；
8. 五种结局分类正确区分；
9. 无人工回填时间戳通道（所有 *_at 由系统生成）。

## 6. 明确不做（第 4 步）

自动下单、LLM 自动冻结、盘中改冻结版、DPO/RL、与 Pyxis/Astrolabe 联动。
模型推荐只进 model_recommendation 影子表，不影响状态机。

## 7. 规则层 v1 迁移（2026-09-17）

**risk_type 冻结枚举**（`record_sell` 专用；`human_decision.risk_type`、
`human_decision.evidence_ref`、`execution_log.risk_type` 三个可空列）：

```text
E0_HARD_STOP   硬止损
TRAIL_STOP     移动止盈/止损
REBOUND_FAIL   反弹失败
TIME_EXIT      时间退出
RISK_FLOOR     风险底线
SECTOR_WEAK    板块走弱
MARKET_RISK    系统性风险
NEWS_EVENT     突发消息
MANUAL_OTHER   其他人工理由（此时 --evidence-ref 必填说明）
```

枚举在 cli.py 的 argparse choices 冻结；库层不加 CHECK（保持可空、向后兼容）。

**DDL 变更**（`ddl.ensure_columns(conn)` 幂等，`ADD COLUMN IF NOT EXISTS`，
`cli.py main()` 启动时调用一次；新库由 CREATE TABLE 直接含列）：

- `human_decision` + `risk_type TEXT`、`+ evidence_ref TEXT`（可空）
- `execution_log` + `risk_type TEXT`（可空）

**新增模块**（规则层最小闭环）：

- `draft_gen.py`：green_watch 池（`pool_latest.json`）→ draft.json →
  `cli create_draft` 落库（trading_date=next_trading_date，
  session_tag=green_watch_daily）。BOTTOM_SETUP 取 tier∈{WATCH,
  HIGH_PRIORITY_WATCH} 按 score 前 5；INTRADAY_DIP_SETUP 前 3
  （v1 不出 BUY trigger，仅观察）。每日只跑一次。
- `cli.py` 新命令：`settle`（盘后五结局打标 + 未执行反事实，
  包装 plan_db/daily.py 与 outcome.py 既有函数）、`record_buy`、
  `record_sell`（risk_type 必填且在冻结枚举内）；`load_minute` 增加
  keyword-only 参数 `force=False`（force=True 忽略 parquet 缓存直接重拉，
  盘中监控用，缓存写回逻辑不变）。
- `monitor_loop.py`：盘中监控循环（`--date/--interval/--once/--push-observe`），
  force 重拉分钟线跑状态机，新 trigger_event 经 PushPlus 推送
  （token 读 `stargazer_desktop/config.yaml` 的 `push.pushplus_token`，
  为空只写日志；游标存 `data/monitor_state.json`）。

### 7.1 监控机独立模式（2026-09-17 追加）

监控机（无 PG/psycopg）每晚拿本机 `cli export_frozen --date YYYYMMDD --out
frozen.json` 导出的冻结计划（全部 FROZEN 版本候选 + 三条件 DSL + pre_close，
缺失标 `pre_close_missing`），次日 `monitor_loop.py --plan-file frozen.json
[--events-out events.jsonl]` 独立跑盘：顶层与运行时不 import cli/plan_db/
psycopg（dsl.py 按文件路径加载），失效抑制与 inputs_hash 口径与 machine.py
逐位一致；事件写本地 jsonl（去重键 session_tag+code+rule_type+fired_at）。
盘后本机 `cli merge_events --file events.jsonl` 按
UNIQUE(candidate_id, fired_at, rule_type, inputs_hash) 幂等并入 trigger_event
（定位链：day+session_tag→session→最新 FROZEN 版本→sort_order+code 候选，
fallback 只按 code；unknown>0 退出码 1）。分钟线/昨收读取独立成 `mquote.py`
（无 PG 依赖，日线目录可用环境变量 STARGAZER_DAILY_PARTS 覆盖），
cli.py 的 load_minute/load_pre_close 改为同签名包装。
