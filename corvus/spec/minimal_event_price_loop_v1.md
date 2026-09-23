# Corvus 最小信息闭环规格 v1.0（2026-09-13 冻结）

成功标准一句话：**信息事件能够被可靠地关联到"当时真正可见之后"的价格路径**。
不是"新闻模型能预测涨跌"（那是第 10 步的事）。

## 1. 什么事件允许进入闭环

| 来源状态 | 处置 |
|---|---|
| P0 + model_eligible=1（AFFIRMED + 正文完整 + 证据约束全过） | **进入正式标签闭环** |
| P1 容器文档 | 只有提取出具体底层事件并重新走事实层后才进入；容器本身不进 |
| P2 例行中性 | 保留为**负对照样本**（可生成标签、标记 negative_control=true 语义，训练时用作"无事件"基准） |
| UNMAPPED_CONTEXT | 继续观察，**不进入** |
| NEGATED / 证据不足 / 正文缺失 | 不可建模，但**不可删除**，留审计 |

进入闭环的判据全部来自事实层 v1.2 门控（spec/f_v1.2_fact_layer.md），本层不复查语义、只查门控结果与证据字段存在。

## 2. 三时间戳（唯一口径）

| 字段 | 含义 | 用途 |
|---|---|---|
| published_at | 来源标称发布时间 | 审计/回溯，**不作标签起点** |
| available_at | 本系统按可见性规则判定"市场可知道"的时间 | **标签起点的唯一依据** |
| effective_at | 事件实际生效时间 | 语义参考，**不得替代** available_at |

## 3. 实体映射优先级（规则优先，LLM 只提名）

1. `RULE_CODE_OFFICIAL`：原文/列表页出现的证券代码（6 位，交易所字段佐证）——最高优先；
2. `RULE_NAME_MATCH`：公司全称/简称对（全称优先于简称）；
3. `BOARD_COMPONENT`：板块成分归属（题材/行业类事件映射板块时）；
4. `LLM_CANDIDATE`：Qwen 实体字段提名，**无证据不得自动绑定**， confidence ≤0.5 且须人工或规则复核后升级；
5. `MANUAL`：人工指定。

entity_type 六态：DIRECT_STOCK / SECTOR / MARKET / GLOBAL / MULTI_ENTITY / UNKNOWN_SCOPE。
宏观新闻不强行贴个股；无法确认 → UNKNOWN_SCOPE，标签挂在 MARKET 或干脆不生成个股标签。

## 4. 价格反应起算时点（由 available_at 决定）

| 信息可见时间 | 标签起点 label_start | basis |
|---|---|---|
| 交易日前可见（≤09:30） | 当日开盘（09:30 的第一笔） | PREMARKET_OPEN |
| 盘中可见 | 可见后的下一分钟 | INTRADAY_NEXT_MINUTE |
| 收盘后可见（>15:00） | 下一交易日开盘 | POST_CLOSE_NEXT_DAY |
| 周末/节假日 | 下一交易日开盘 | WEEKEND_NEXT_DAY（与上一可见日间隔 >1 个自然日即视为跨周末/节假日，周五盘后同此口径） |
| 停牌期间 | 复牌后第一个可交易时点 | RESUME_FIRST_TRADE |

实现约束：日线标签用下一交易日开盘价（与 tarot 标签口径一致）；30 分钟标签用分钟线从 label_start 起算。

## 5. 标签口径

- 收益：ret_30m / ret_to_close / ret_1d / ret_3d / ret_5d
- 相对收益：abnormal = 个股 − 基准；基准 = 大盘指数（沪深300，按股票所在市场）与所属板块指数；**个股公告同时算两个基准**，宏观/板块事件不强行贴个股
- 路径：mfe/mae（30m/1d/3d/5d）、gap_return、volatility_expansion（事件后5日波动/前20日波动）、volume_shock（事件日量/20日均量）、reversal_30m/1d
- 反身性（只存现象）：initial_direction、initial_shock_atr、liquidity_sweep_depth、reclaim_vwap、reclaim_pre_event_price、reversal_strength、reversal_speed、post_reversal_ret_3d/5d
- 状态：limit_locked / suspended / corp_action_during_horizon

**尾部截断**：标签窗口（1/3/5 日）未走完 → label_status='CENSORED_TAIL'，字段留 NULL，**绝不填 0**。
盘中起点当日的 ret_to_close 从 label_start 价格算到收盘。

## 6. 重复事件与修订

- 同一 raw 公告重复采集：information_event 以 UNIQUE(raw_id, event_version) 幂等；
- 修订/撤回公告：新行 event_version+1 且 supersedes_event_id 指向旧事件；旧事件标签保留但训练时按版本链取最新；
- 同一事件多源报道（东财转巨潮）：按 body_hash + canonical_event_type + 标题规范化去重，只正式化一个 canonical，其余记 evidence。

## 7. 事件重叠（防因果错误）

同一实体、标签窗口（事件日起 ±5 交易日）内存在 ≥1 个其他事件 → 写 event_overlap：
attribution_quality：CLEAN（0 个重叠）/ MODERATE（1-2 个或仅宏观）/ HEAVY（≥3 个或同类型叠加）。
**标签一律命名为"事件后反应"（post-event reaction），不声称因果**；HEAVY 样本第一代训练降权或排除。

## 8. 可训练 vs 只审计

| 字段 | 级别 |
|---|---|
| available_at、label_start、ret_*、abnormal_*、mfe/mae、overlap/attribution | **可训练**（仍需按重叠降权） |
| effective_at、official_scope、assertion_status、evidence_span、amount_evidence | **只审计**（不进特征） |
| initial_direction 等反身性字段 | 可训练但第一代只用现象不做结论 |

## 9. 幂等与审计

- 每批标签生成先写 event_label_run（参数/版本/计数），code_version 用代码文件哈希；
- 所有写入用 (event_id, entity_id, label_version) 幂等锚，重复运行不产生重复行；
- 标签可从原始行情（parquet/分钟线缓存）整体重算，重算产生新 label_version，旧版本保留。

## 10. 明确不做（第 3 步）

不训练收益模型；Qwen 不判利好利空强度；不把"事件后上涨"当"事件造成上涨"；
不接绿盘买入/退出；不动 Pyxis；不用前瞻结果调阈值。
