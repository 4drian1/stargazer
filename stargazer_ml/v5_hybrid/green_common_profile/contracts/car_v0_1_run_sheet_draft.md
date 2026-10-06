# GREEN-CAR-v0.1 Run Sheet / 参数表（工程范围冻结 — 用户已逐项裁定）

日期：2026-09-29。
状态：**ENGINEERING_SCOPE_FROZEN（用户已逐项裁定，工程范围冻结）**。本文件是《conditional_anchor_reaction_v0_1.md》（GREEN-CAR-v0.1，下称"合同"）§8 交付顺序的第一项：run sheet/参数表。
授权边界（合同抬头 + 用户裁定）：**本轮只做工程范围冻结 + 写 CAR 新标签 + 跑 13 类工程验收及事件字典；不训练 CAR-G/CAR-LR、不抓取（除非工程验收需要数据补抓另签）、不阈值搜索、不接入线上。** 本文件所有数值均为**结果生成前的冻结值**；改则重登记哈希。
上位文件：合同；D:/stargazer/新标准特征工程.txt 第八节；D:/stargazer/开发路线.txt 记录#11。
登记材料（已生成，仅本地只读计算，0 HTTP）：
- `contracts/car_v0_1_candidate_list.txt` — 33 候选股票日；字节SHA256 = `0ef1c647d5cbd81cd40d6c62dd4b97bb4213efad0a6c70cb1d00e61164b65f48`；规范化(UTF-8/LF/末尾换行)SHA256 = `0ef1c647d5cbd81cd40d6c62dd4b97bb4213efad0a6c70cb1d00e61164b65f48`（本文件本身即 LF+末尾换行，两者一致；**与旧 865 清单的字节/规范化哈希不同属正常，不得混用比对**）。
- `contracts/car_v0_1_sampling_summary.json` — 抽样统计、缓存核算、全部冻结输入哈希（gcp_v0_1_spec.py / gcp_v0_1_pools.py / gcp_v0_1_round_a.py / gcp_v0_1_round_b_spec.py / 引擎 / 两池 parquet / 合同 / 抽样脚本）。
- 生成脚本：`gcp_car_v0_1_sampling.py`（RNG 调用顺序固定，可复现）。

---

## 1. 冻结输入与数据范围

| 项 | 值 | 状态 |
|---|---|---|
| 共性成员来源 | `round_a_v0_1_impl_fix/pool_a_sdp_members.parquet`（10151 行）+ `pool_b_lvb_members.parquet`（7533 行）；date 列 = P 资格日 | 冻结（哈希见 summary） |
| 共性阈值 | `gcp_v0_1_spec.py`（ATR14=TR 14日算术均值；ATR_ref=截至P 5日ATR14中位；BOLL=MA20±2·std20 ddof=1；RSI14=Wilder ewm α=1/14 adjust=False 自上市全历史预热） | 冻结，不修改以增加事件 |
| 成员构建代码 | `gcp_v0_1_pools.py` / `gcp_v0_1_round_a.py` | 冻结（哈希见 summary） |
| 分钟纪元 | [20250102, 20260917]；共同结束日提案 **20260917**（P2） | 提案 |
| 网格 | 241 槽（<20260624）/ 242 槽（≥20260624，13:00 槽源格式差异）；沿用引擎 `expected_grid` | 沿用既有离线探针结论 |
| 日线 | `data_daily_parts` qfq 镜像；PC/TR 前收来源与复权尺度验收沿用 Round A 对拍 | 已验收路径 |
| 复权链 | `gcp_round_b_small_sample_v1/adj_cache.json`（16 股）；新抽中股票需补请求 | 提案 P-DATA |
| 已有分钟缓存 | 16 股 `minute/` + 7 股 `minute_ext/`（旧 v0.2 输入，保留不覆盖） | 工程事实；**不作为抽样过滤条件**（本次 20 股全部为新股，0 缓存命中） |
| LATR | 无可核验合同路径/公式/单位/版本 → **PENDING**；相关字段本轮 NULL，不用 ATR 或密集区距离替代 | 合同 §3 |

## 2. 抽样登记（合同 §2）

**程序 P-SAMP（用户裁定接受作为工程抽样）**：抽样单位=(symbol,E)。
1. 合格宇宙：{(code,P,setup)} ∈ 冻结池，E = P 下一交易所交易日，E ∈ [20250102, 20260917]，P 为日历交易日；北交所（4/8/92 开头）无分钟，**显式排除并分别计数（逐池记录 330 股票日 / 138 去重股票）**；主板/创业板/科创板分别标记（宇宙分布 MAIN=5257 / CHINEXT=2357 / STAR=654）。
2. 股票：每 setup 内 code 升序，`default_rng(20260929)` 无放回抽 10 股 → 20 股。
3. 日期：每股合格 E 日升序 → episode 划分（相邻 E 间隔 >2 交易日开新 episode，**P-EP 间隔 >2 日仅用于 episode 分组，不代表事件已独立**）→ 无放回抽 ≤5 个 episode、取 episode 首日 → 每股 ≤5 日。
4. 不按反弹结果/盈利记录/历史成功数筛股票；不为凑 n 重抽；一次性抽样。
5. `candidate_episode_id=(code|setup|epNNN)` 已入清单；`event_group_id` 由事件提取层赋值（同 symbol+E 多锚点/多周期同组，不重复计独立样本）。

**抽样性质（用户明确裁定）**：这是"先抽股票，再抽 episode 首日"的**工程抽样，不是候选股票日等概率抽样**。用途是机制/因果/数据分流验收；**后续统计研究阶段应另定抽样与聚类规则**。"相邻日期不独立"只用于 episode 分组，不代表事件已独立。
**实际结果**：20 股（A=10 / B=10），**33 候选股票日（A=15 / B=18）**。配额未满（33 << 100）主因：**抽中 20 股的资格日本身稀疏（合计仅 44 个资格日，最长 episode 仅 2 日）**，归并后变为 33 日。不是长连续 episode 归并导致。
**用户裁定：本轮保留这 33 日，不为增加数量重抽。**

**预算核算（去重 (code,date) 集合后扣缓存；合同 §7 上限 ≤2000 股日；仅登记未批准请求）**：
| 类别 | 请求 (code,date) 或 code 数 | 说明 |
|---|---|---|
| 类1+2 特征+标签分钟 | 66 股日 | E 日 + 未来窗可能跨到的下一交易日；**持续性标签可能延长至 S+70 交易分钟（60m 窗+收回后 10m），下午信号可能跨到 E+1 甚至 E+2，用交易分钟日历核验** |
| 类3 R 组历史基线 | 1761 股日 | ≤60 交易日/股，去重后缺失；**本轮默认不作（见 P-BUDGET）** |
| 类4 复权链 | 18 股 | 每股一次；18/20 抽中股未在旧 adj_cache 中；**失败保持 UNKNOWN，不当空链** |
| **合计（不含 R 组，本轮默认）** | **84** | 66 + 18；**仅预算登记，补抓需另签** |
| 合计（含 R 组，仅供参考） | 1845 | < 2000 上限，**不超预算** |

**P-BUDGET（用户裁定）**：本轮不做历史同刻量比消融，不缩短历史，不提高上限。先修正预算核算（去重后）。vwap_distance_atr 只需 E 日截至 S 的量额前缀，不需要历史分钟，可保留为预注册候选；历史同刻量比/额比后置。
**授权边界（用户明确）**：84 项只是预算登记，**不代表已批准请求**。真实回放前再冻结必要的去重请求清单与完整 SHA256。现在只实现函数 + 跑合成测试。
**真实回放窗口范围（用户裁定）**：本轮真实回放只做 **30m / 60m / 收回后保持 10m**；**5d/10d 保留合成测试，真实数据诊断后置**。不要因缺少预算中的未来数据，把 5d/10d 窗口默认为完整。

## 3. 锚点、尺度与事件（合同 §3 — 冻结确认项）

- 锚点首版：**MA20 / BOLL_LOWER / PREVIOUS_DAY_LOW**，全用 P 完成日线；MA20 与 BOLL 中轨同一锚点，别名不增加票数。
- 区域构造：0.35·ATR_ref 相邻链式合并（沿用引擎 `build_zones`）；簇跨度可超 0.35ATR，报告分布；多锚点保留全部 anchor_sources 与逐锚点距离，不事后挑收益最大的线。
- 目标区域：武装区域（zone_upper ≤ close_P）中 zone_upper 最高者（沿用 `target_zone`）。
- 接近带 0.20·ATR_ref；监控窗 [09:35,14:55]；**本轮不搜索这些参数**（未证明最优）。
- 事件状态机：UNARMED→SIGNAL→CONSUMED；每股票 E 至多一次物理事件（引擎"每日一信号"即此语义）。event_branch ∈ {APPROACH, BREAK, ALREADY_BELOW}（=引擎 signal_type）。
- **S = 该物理事件首个完整信号bar结束**。源时间戳为 bar 起始槽（如 093000 覆盖 09:30:00–59），故 S 的未来窗 = 信号槽位之后第 1..N 个交易分钟槽位 —— 与引擎 `_compute_minute_window` 槽位约定一致。事件开始冻结锚点与 ATR_ref，后续不随价格移动。
- 5/15 分钟仅用于历史响应视图与连续特征，只用 S 前已完成槽，不重复计新事件。

## 4. 标签定义（合同 §4 — 冻结确认项）

**主行为标签 `future_reclaim_60m`**：
- 适用：S 时 close < zone_upper 的 **BREAK / ALREADY_BELOW** 事件。
- =1：(S, S+60 交易分钟] 内完成 bar close 首次 ≥ zone_upper；=0：完整 60 分钟未收回。
- 分支（不计入主标签分母，单独报告）：信号 bar 自身已收回（low ≤ zone_lower 且 close ≥ zone_upper，含原 v0.2 的"同 bar AMBIGUOUS"情形——CAR 在 S 时点用 close 判定，**确定性分流，无顺序歧义**）→ `RECLAIMED_AT_S`；APPROACH 从未跌破 → `ALREADY_ABOVE`。
- AMBIGUOUS / 源缺 / 尾部截断 分列，不硬塞 0/1；**未收回 time=NULL + 观察终点/截断标志**，不用窗长+1 求均值。
- 起点不可混成 fill_ts。

**辅助标签**：future_reclaim_30m；future_penetration_60m = max(0,(zone_lower − min_low_(S,S+60m))/ATR_ref)；time_to_first_future_reclaim（未收回=NULL）。
**5d/10d（用户裁定）**：本轮真实回放**不做** 5d/10d；保留合成测试，真实数据诊断后置。**不要因缺少预算中的未来数据，把这些窗口默认为完整**——真实数据上 5d/10d window_complete 一律 False（未抓取），不挑通过窗替换 60m。
**持续性辅助（P-HOLD，用户裁定接受）**：首次未来收回后 10 交易分钟 close 均不低于 zone_upper；无收回→按完整 60m 标 0；有收回但后续不足 10m→CENSORED 不算失败；**各标签分别记录 label_end_ts**（持续性标签的 label_end_ts = 首次收回时刻 + 10 交易分钟，可能到 S+70 交易分钟，跨日时用交易分钟日历核验）；**不能为了预算截短定义**；不回灌至 S。
**H3/H5**：复用引擎 `label_event`（下一合法可执行分钟成交、交易所日历退出、顺延、成本、复权链、参与率）；**在正式执行引擎验收前一律标 RESEARCH_APPROXIMATION，不解锁 H-E**。
**分立**：收回率 / 风险概率 / 收益分别建模报告；无互斥穷尽契约不得令其和为 1。

## 5. 字段白名单提案（合同 §5 + 特征工程 §8.3 — 实际入模子集在标签分析前写入 run manifest）

| 组 | 首轮提案字段 | 来源/备注 |
|---|---|---|
| 背景 C | setup_type, session, event_branch | 事件层，直接可得 |
| 背景 C | rsi14_P, boll_width_atr | 池 parquet 已含（P 时点）；B 股日/非池日需日线补算 |
| 背景 C | ret20_P, ret60_P, drawdown60_P | 池 B 已含；A 股日从日线镜像补算（本地，无请求） |
| 背景 C | ma20_slope_atr | **P-SLOPE（用户裁定）**：5 交易日变化量 (MA20_P − MA20_{P−5td})/ATR_ref；**注明不是回归斜率** |
| 背景 C | sector/market_context_asof | **本轮 NULL**（无版本化共享源），不复用已失败硬门 |
| 锚点 A | dist_ma20_atr, dist_boll_lower_atr, dist_prev_low_atr | (close_S − anchor_P)/ATR_ref；三距离保留，中轨去别名 |
| 锚点 A（审计表） | zone_lower/upper/mid/sources/confluence_count | **区域绝对价格只保留在审计表，不进 LR**；LR 用 ATR 标准化距离/区域宽度/来源数/合流数建模 |
| 锚点 A | anchor_gap_atr / anchor_age / previous_touch_count | **PENDING**（可见窗口未定义），首轮不入 |
| 冲击 V（少量） | penetration_atr_to_S | max(0,(zone_lower − min_low_seen_to_S))/ATR_ref；S 前缀口径 |
| 冲击 V | signed_distance_atr, opening_gap_atr | 有符号区域距离；隔夜缺口 |
| 冲击 V | avg_penetration_rate_atr_per_min | **用户裁定改名**：穿透深度÷已观察交易分钟 = **平均穿透速率**（不是真正的分段下杀速度）；speed_change/range15 首轮不入（PENDING/切片易缺） |
| 承接 R | vwap_distance_atr | **用户裁定保留为预注册候选**：只需 E 日截至 S 的量额前缀，不需要历史分钟；vwap_reclaim_to_S 同源 |
| 承接 R（后置，**本轮不作**） | volume_ratio_same_time, amount_ratio_same_time | 历史同刻量比/额比需 ≤60 交易日历史分钟；P-BUDGET 本轮不作 |
| 质量 Q | feature_asof, anchor_asof, source_snapshot_ref, window_valid, prefix_complete, scale_status | 质量列；不自动当作风险预测输入 |

禁止输入：symbol、绝对买入价、人工盈利名单。LATR 系字段全部 PENDING。

## 6. 模型配置（合同 §5）

- **CAR-G**（主比较基线）：训练段组先验，父组顺序冻结为 **(setup, event_branch, session) → (setup, event_branch) → setup → 全训练段**；无训练支持=UNKNOWN，不从验证/诊断段回填。收缩 w=n/(n+λ_G)，**P-LAMBDA（用户裁定）λ_G=20 可登记固定默认值，但本轮不拟合、不调参**。
- **CAR-LR**（共享条件主模型）：CAR-G 信息 + §5 白名单连续/类别字段；正则化逻辑回归，**P-L2（用户裁定）固定 L2 C=1.0 可登记默认值，但本轮不拟合、不调参**；标准化/缺失处理/类别编码只用训练段拟合，不用全表标准化。
- **CAR-TREE**：仅有共享信号后另批轻量树，本轮关闭。**CAR-P**：本轮关闭（个股项决策权重=0）。
- **本轮定位（用户明确裁定）**：**暂不训练 CAR-G/CAR-LR**。当前拟合段仅 9 候选日、开发段 8 日、诊断段 15 日，有效主标签还会更少。**只验证计算、因果性和数据分流，不能验证模型效果**。统计门的事件数/日期覆盖门在后续 run sheet 签署，不依据看到的收益降门槛。

## 7. 时间切分、隔离与统计（合同 §6 — 提案待确认）

- 切分提案：**拟合 E ∈ [20250102, 20250630]（9 候选日）；开发比较 E ∈ [20250701, 20251231]（8 日）；压力诊断 E ∈ [20260101, 20260831]（15 日）**；20260901–0917 未纳入段，不参与本轮判定。实际范围缺某段=INSUFFICIENT，不按最好月份重切。
- 同 event_group 所有锚点/周期/标签同折；**按每行真实 label_end purge（用户裁定：真正边界必须由逐行 label_end 决定）**；P-EMB embargo=15 交易所日**可作附加隔离**（覆盖 H5+最大顺延 10 日）。
- 误差差值 = baseline − new，正值才改善；P-BLK/P-GATE **留到统计阶段签署，33 日不计算晋级结论**。
- 行为主闸门：60m Brier 增量 95%CI 下界 >0；PR-AUC/AUC 与实际正例率、校准图同报；A/B 不仅池化通过；跨期无明确反转。辅助窗无论如何不救主窗。
- H-E / H-P 另门，本轮不启动。

## 8. 旧引擎复用审计（R0，合同记录#11.3"先审计旧引擎复用项"）

| 组件 | 裁定 | 说明 |
|---|---|---|
| `build_atr_ref` / `build_zones` / `target_zone` | **直接复用** | 口径=合同 §3（MA20/BOLL_L/PREV_LOW、0.35ATR 链式合并、ATR_ref 5日中位） |
| `detect_signal` | **复用** | APPROACH/BREAK/ALREADY_BELOW=合同 event_branch；监控窗/接近带一致；每日一信号=每 E 至多一次物理事件 |
| `expected_grid` / `validate_minute_day` | **复用** | 241/242 槽源格式差异已核实 |
| `label_event` 执行引擎部分（H3 退出/复权/参与率/顺延） | **复用，标 RESEARCH_APPROXIMATION** | 正式执行引擎验收另做；不解锁 H-E |
| `_compute_minute_window` / `_compute_daily_window` 框架（槽位窗、完整性/截断独立、ALREADY_ABOVE 分流） | **框架复用，标签函数新写** | CAR 主标签条件与窗起点不同（S 时 close 条件 + (S,S+60]），新函数不改旧函数，避免旧 v0.2 报告口径被追溯改写 |
| `reclaim30_invalid` | 不复用 | 旧兼容字段；CAR 用 window_complete_* |
| `estimate()`（v0.2 P0/P1 单元格） | **不复用** | CAR-G 为父组收缩新实现；v0.2 画像未过 60m 闸门，决策权重=0 |
| v0.2 稳定性/个性画像逻辑 | **不迁移** | 已否决（v02_feasibility_v3.json established=false）；留档 |
| 案例脚本 EWM ATR / ddof=0 BOLL | **禁止迁入** | 与冻结合同不同口径（记录#11.1） |
| LATR | 无实现 → PENDING | 合同 §3 |
| RSI14 | 池 parquet 已含（P 时点） | 冻结口径核对一致 |

已有 46 项验收测试**不**当作新标签/新样本的自动通过（记录#11.3）；§9 映射的新必测项须在新数据上重跑。

## 9. 工程验收必测项映射（合同 §7"必测"→ 具体测试，工程验收阶段执行）

1. 共性资格 P→E 映射：抽中候选逐条对拍池 parquet + 交易日历（prev_td 语义）。
2. PIT/因果截断与未来扰动：已闭环事件（label_end < 扰动点）字段逐位不变（沿用 H6 范式，新标签重写）。
3. 多周期/多锚点不重复计数：同 symbol+E 多锚点 → 同 event_group，profile/统计按组去重。
4. 同 bar 模糊：BREAK/ALREADY_BELOW 且 close_S ≥ zone_upper → RECLAIMED_AT_S 分流（不再有 v0.2 式 AMBIGUOUS 归属）。
5. 信号时已收回不算未来成功：RECLAIMED_AT_S 不进主标签分母。
6. 未收回 NULL：time_to_first_future_reclaim=NULL + 观察终点/截断标志。
7. 各窗截断独立：30m/60m/5d/10d 完整性互不覆盖（沿用 H5c）。
8. 内部缺分钟 → 特征/标签 INVALID：窗口覆盖缺失槽即 INVALID，不插值。
9. 确认零成交 vs 源洞区分：零量占位 ≠ SOURCE_MISSING ≠ 停牌（沿用逐位比对范式）。
10. 除权/单位与日历：复权链逐日因子作用于退出价与持有期 low；volume 量纲=股对拍。
11. NO_FILL 不删除有效行为：NO_FILL_ZERO_QTY 仍贡献行为标签（沿用 H7）。
12. 全缺失/全被过滤不能退化 PASS：非空转守卫。
13. 未来窗完整性：E 为窗末日时 60m 尾部截断显式标记，不改判"未收回"。

## 10. 预算与资源（合同 §7 / 记录#11.4）

- 请求：84（不含 R 组：66 分钟股日 + 18 复权链）或按 P-BUDGET 裁决（含 R 组 1845 < 2000 上限）；**命中预算即停**；失败有限重试/显式记录，不静默加股/拉长历史。
- 磁盘：临时 ≤30GB、最终 ≤5GB；本轮新增快照预计 <1GB（33 股日分钟 + 日线派生）。
- 执行：只串行低并发，不与 Corvus/其他全量任务争抢。

## 11. 提案清单（**用户已逐项裁定；状态记录如下**）

| 编号 | 提案 | 用户裁定 |
|---|---|---|
| P-SAMP | 抽样程序（种子 20260929；20 股/33 日；episode 首日规则） | **接受作为工程抽样**（非候选股票日等概率抽样；后续统计另定） |
| P-EP | episode 划分：相邻 E 间隔 >2 交易日开新 episode | **接受**（间隔 >2 日仅用于 episode 分组，不代表事件已独立） |
| P-BUDGET | R 组消融取舍 | **本轮不作**（不缩短历史、不提高上限；先修正预算核算去重后） |
| P-END | 候选截止 20260917；**标签可用数据截止另列，窗口未走完就截断** | **接受**（补充：标签截止另列） |
| P-SLOPE | ma20_slope_atr = 5 交易日变化量 | **接受**（注明不是回归斜率） |
| P-HOLD | 持续性辅助标签（收回后保持 10 交易分钟，CENSORED 规则） | **接受**（仅辅助标签，不能救主标签） |
| P-LAMBDA | CAR-G λ_G=20 | **可登记固定默认值，但本轮不拟合、不调参** |
| P-L2 | CAR-LR 固定 L2 C=1.0 | **可登记固定默认值，但本轮不拟合、不调参** |
| P-EMB | embargo=15 交易所日 | **可作附加隔离**；真正边界必须由逐行 label_end 决定 |
| P-BLK | bootstrap 日期块长 5 交易日、B=1000 | **留到统计阶段签署** |
| P-GATE | 月度门（≥60% 月严格改善、每月 ≥10 独立事件） | **留到统计阶段签署**；33 日不计算晋级结论 |
| P-DATA | 新股复权链请求（18 股，并入请求预算） | **接受**；失败保持 UNKNOWN 不当空链 |
| 字段修正 | 承接组 vwap_distance_atr 保留为预注册候选 | **接受**（只需 E 日截至 S 前缀，不需历史分钟） |
| 字段修正 | 区域绝对价格 zone_lower/upper/mid 只保留审计表 | **接受**（LR 用 ATR 标准化距离/宽度/来源/合流数） |
| 字段修正 | drop_speed → avg_penetration_rate（平均穿透速率） | **接受**（不是真正的分段下杀速度） |
| 计数修正 | 北交所排除拆成股票日数 + 去重股票数 | **接受**（330 股票日 / 138 去重股） |

## 12. 状态与下一步（用户裁定顺序）

**修正 run sheet 与去重请求清单 → 冻结工程范围 → 写 CAR 新标签 → 跑 13 类工程验收及事件字典。**

1. ✅ 本 run sheet 已按用户四类修正修正完毕（预算去重、抽样性质、字段白名单、计数）。
2. **冻结工程范围**（本文件即为工程范围冻结材料；哈希见 summary）。
3. **写 CAR 新标签**：future_reclaim_60m / future_penetration_60m / time_to_first_future_reclaim / 持续性辅助 / RECLAIMED_AT_S 分流 —— 新函数不改旧 v0.2 函数。
4. **跑 13 类工程验收及事件字典**（§9 必测项）；本轮**不训练 CAR-G/CAR-LR**。
5. 本轮真正的成功标准（用户原话）：**每个事件"在什么时候、接近哪条锚点、当时是否已收回、之后是否收回"都算对、可重放，而不是必须跑出盈利**。
6. 本文件不修改旧 v0.2 否决结论；旧报告/清单全部留档。
