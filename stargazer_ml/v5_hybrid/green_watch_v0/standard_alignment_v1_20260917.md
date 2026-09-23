# 选股标准（GREEN-WATCH-v1）↔ build_pool.py（v0）对齐记录

- 记录时间：2026-09-17
- 标准文件：`C:\Users\22107\Desktop\交易\选股标准.txt`（自称 GREEN-WATCH-v1）
- 代码文件：`stargazer_ml/v5_hybrid/green_watch_v0/build_pool.py`（rule_version=GREEN-WATCH-v0-backfill）
- 用途：v1 最小闭环开工前留痕。标准 §38 要求任何调整记录 rule_version/原因/口径；
  本文件即该记录，未改代码、未改标准原文。
- **优先级决议（2026-09-17，用户批准）：开发中规则层出现冲突时，以本选股标准
  （GREEN-WATCH-v1）为准。** 即：新写代码与标准条文冲突 → 改代码对齐标准；
  此前冻结的实现口径（如触发价公式）与标准冲突 → 按标准执行并在此留痕，
  不再沿用旧口径。本决议不改变"已验收工程件不回头改"的红线
  （评分逻辑 v0 已 POOL_NEGATIVE 终裁，保持原样跑影子攒数据；冲突处置适用于
  新开发部分：draft 生成器、盘中规则、LATR 特征组）。

## 一、已一致的部分（核对通过）

- §4.1 ATR_ref = median(ATR14[T-4..T])，T 日冻结 ✓（build_pool.py:103）
- §4.2 MA20Slope5 ✓；§4.3 DistLow20/60 ✓；§4.4 RangePos20/60 分别保存 ✓
- §五 市值 25–100 亿 ✓（2.5e9–1.0e10）；§12 基础条件（MA20S5>0、C≥MA20−0.15ATR、
  MACD 零轴上/近零轴、低点抬高容差 0.15ATR）✓
- §9 底部证据 9 条全部实现为 E1–E9 ✓
- §10/§13 两个 100 分模块结构与满分一致（15+20+25+20+10+10 / 20+20+15+15+15+10+5）✓
- §2 双分数保留 + primary_setup 取大者 ✓（平票归 BOTTOM，标准未规定，记为口径 A）
- §16/§19 大部分量价与流动性子项逐条对应 ✓

## 二、已记录在代码 DEVIATIONS（build_pool.py:39-52）的偏离（不重复展开）

MACD/RSI/BOLL 常规参数；swing 用 High60/Low60 近似；AVWAP/POC/VAL 缺失用
MA20/Low20/BOLL/Low60 近似；板块/龙头用全池等权指数代理；PULLBACK_BOTTOM
不独立成硬门槛。

## 三、新发现的未记录偏离（本记录新增，开工时按此执行）

| # | 标准条文 | 代码现状 | 处置 |
|---|---|---|---|
| N1 | §21 分层 4 档（<55 DROP） | 5 档：多 NEAR_MISS 带（45–55），且 score≥40 即入库 | 刻意偏离：依据标准开头"保留 NEAR_MISS/CONTROL"与 §33 训练数据采集。保持不动 |
| N2 | §12.3 放量破位"不进入高优先级" | BREAK_DOWN 是 DIP_GATE 硬排除（量>2×前5日均量 ∧ 收破20日前低 ∧ 收阴） | 比标准更严，属保守化。保持，待 LATR 消融时再评估是否降格为扣分 |
| N3 | §10.2 回调充分"不预设区间" | 线性映射 (PB_ATR−0.5)/2.5 clip0-1，>8ATR 减半 | 自由裁量，保持 |
| N4 | §10.1 非脉冲/无顶部派发 | 代理：60 日涨停<3 次 / 当日额<3×20日中位额 | 自由裁量代理，保持 |
| N5 | §17 RSI>65"不额外奖励" | rsi_s 给 1 分 | 轻微偏差，保持 |
| N6 | §12.1 MACD 近零轴 | 门槛用 |DIF/C|≤0.005，评分用 ≤0.01 | 两处不一致，保持现状并留痕（差异 2 分权重，影响小） |
| N7 | §18 字段名 sector_return/leader_return/stock_vs_sector 等 9 项 | 只有 sector_proxy_ret1/3（等权池指数），无 leader 字段 | 代理口径已记 DEVIATIONS；字段缺口留痕，等真实板块映射接入时补 |
| N8 | §5 基础资格抽象要求（复权尺度一致、非长期停牌等） | ST/VOL>0/价域/Low60-notna 代理 | 留痕。分钟/引擎层已另有尺度校验 |
| N9 | §33 CONTROL 样本组 | 未显式构造 CONTROL（NEAR_MISS 带部分承担） | v1 不补；训练 Timing 模型前需补，记入 LATR 阶段待办 |

## 四、与 v1 最小闭环决策的张力（留痕，不动已冻结决议）

| # | 事项 | 说明 |
|---|---|---|
| T1 | §26 BOTTOM 次日三路径 vs 冻结触发价 | 标准允许 HOLD_TRIGGER（OpenDipATR<0.05 守位+修复也 ARMED）；v1 闭环只有单 trigger `price<=昨收−0.25×ATR_ref`，实际只覆盖 STANDARD_LATR 路径。属 v1 已知限制：BOTTOM 票次日不回踩 0.25ATR 就无提醒。**优先级决议（见文首）后：后续规则层开发（LATR 特征组、盘中状态机扩展）按 §26 三路径实现，冻结的单 trigger 仅作为 v1 过渡口径，不升级、不扩散到新代码。** |
| T2 | §23 盘后计划字段（zone_id/zone_upper/lower/mid、arm/trigger/invalidation_conditions、expected_scenario、after_close_rank） | LATR_ZONE 未开工，池输出无这些字段。v1 由 draft 生成器用冻结公式生成 trigger/invalidation（昨收−0.25×ATR_ref；翻红 or time≥14:30 失效），zone 字段空缺记入 model_version 说明 |
| T3 | §25/§34 DIP 的 OpenDipATR 0.10–0.40 ARMED 与 0.20–0.25 核心桶 | v1 DIP 票不出 BUY trigger（冻结决议），只 observe。OpenDipATR 分桶全采集属 LATR 盘中规格，v1 监控循环不采集，待 LATR 接入 |
| T4 | 版本命名 | 标准自称 v1，代码 v0。纪律：代码保持 GREEN-WATCH-v0-* 命名（增量模式用 `GREEN-WATCH-v0-daily`），直到消融/前瞻复验通过后才升 v1 命名，不提前宣称 |

## 五、开工影响清单（落实到已批准计划）

- 计划步骤 1（build_pool 增量模式）：评分逻辑/gate/tier 全部原样（含上表 N1–N8 现状），
  仅加 `--date`、pre_close/close 输出列、rule_version 改 `GREEN-WATCH-v0-daily`。
- 计划步骤 2（draft 生成器）：触发价/失效条件按 T1/T2 的 v1 口径生成；
  selection_reasons/risk_reasons 用池现有字段拼人读文本（N7 的 leader 字段缺失
  在 risk_reasons 中以"板块代理口径"注明）。
- N9、T1 放开、T3 采集 → 全部进 LATR 阶段，不在本闭环内。
