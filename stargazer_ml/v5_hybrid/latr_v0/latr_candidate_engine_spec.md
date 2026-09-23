# LATR Candidate Engine 规格 v0（用户定稿 2026-09-16，盘后规则待补）

> 定位（用户原文）：**规则不负责判断"这只一定能买"，规则只负责判断"这只已经进入值得让模型看的低吸场景"。**
> 然后模型对候选排序、判风险、判断承接质量。与"监控程序规则选股 → 模型打分"架构契合，
> 符合路线中 LATR 最终重新进入绿盘体系、不独立外挂的设计。
> 参数一律称 **v0 research threshold**，不是永久交易规则。

## 一、三段状态机（每层可单独训练）

```text
昨日盘后候选 → LATR-WATCH（值得监控）
  → 价格进入 ATR + 流动性区域 → LATR-ARMED（出现低吸场景）
  → 观察承接/修复/量价 → MODEL_SCORE_REQUEST
  → BUY_CANDIDATE / REJECT
```
直接对接既有 WATCH → ARMED → BUY_CANDIDATE 状态机。

## 二、ATR 三拆（必须分开）

ATR_ref = median(ATR14_{T-4..T})（沿用冻结定义）。

```text
GapATR      = (Open - PrevClose) / ATR_ref        隔夜跳空
OpenDipATR  = (Open - Low_15m) / ATR_ref          开盘后被砸多少（研究核心）
TotalDipATR = (PrevClose - Low_15m) / ATR_ref     相对昨收总共跌多少
```
反例：昨收10元、ATR=0.8、低开9.80最低9.79（TotalDip=0.26ATR 但 OpenDip=0.0125ATR），
与"平开杀到9.80被接起"完全不是一种行情。**0.20–0.25 低吸假设真正应研究的是 OpenDipATR。**

## 三、分桶全采集（20–25% 不作硬门槛）

| OpenDipATR | 标记 |
|---|---|
| <0.10 | NO_DIP |
| 0.10–0.15 | LIGHT_DIP |
| 0.15–0.20 | NORMAL_DIP |
| 0.20–0.25 | HYPOTHESIS_CORE（latr_core_2025=1） |
| 0.25–0.30 | DEEP_DIP |
| 0.30–0.40 | VERY_DEEP_DIP |
| >0.40 | EXTREME_DIP |

监控采集 0.10–0.40 全带宽；只把 0.20–0.25 打核心标记——否则以后没有 15–20%、25–30%
的对照组，无法回答"0.20–0.25 是否真是最佳区间"。

## 四、流动性区域（第二核心）

前低/确认拐点低点/MA20/BOLL下轨/Anchored VWAP/POC/VAL/整数位；多位置在 **0.35 ATR** 内合并为一个区域。
盘后每日生成：`zone_id / zone_upper / zone_lower / zone_mid / zone_strength / zone_sources`。
盘中计算 `ZoneDistanceATR = (Price-ZoneMid)/ATR_ref` 与路线已有的 `PenetrationATR = (ZoneLower-Price)/ATR_ref`。
真正的 LATR = "跌了 0.22 ATR **并且**正好杀到 3 支撑重叠区"——两者信息量完全不同。

## 五、LATR-v0 触发规则（研究阈值）

```text
latr_watch = premarket_candidate and data_valid and tradable   ← 昨入 INTRADAY_DIP_SETUP 即 WATCH
latr_armed = 0.10 <= open_dip_atr <= 0.40 and zone_distance_atr <= 0.20
```
**故意不写 0.20–0.25**——现阶段最重要是收集研究样本，不是把主观经验写死。

## 六、承接特征（抓"跌到合理位置有人接"，不是"跌"）

```text
ClosePosition_15m = (Close_15m - Low_15m) / (High_15m - Low_15m)   0=收最低 1=明显拉回
RecoveryATR       = (CurrentPrice - LowSinceOpen) / ATR_ref
reclaim_zone_lower / reclaim_zone_mid / reclaim_vwap
lower_wick_ratio / down_volume_ratio / rebound_volume_ratio
new_low_flag / second_low_higher_flag / minutes_below_zone
```
（对应路线已写的：5/15/30分速度、开盘15分ATR占比、创新低、二次回踩、阴线缩量、量对比、VWAP收复、收针、区域下方停留时间——**不是重新发明 LATR，是把文档概念变成机器可采集字段**。）

## 七、每次盘中扫描落全字段（样例行）

```text
symbol, timestamp, setup_type=INTRADAY_DIP_SETUP,
atr_ref, atr_pct, gap_atr, open_dip_atr, total_dip_atr, latr_bucket,
zone_mid/lower/upper, zone_strength, zone_sources, zone_distance_atr, penetration_atr,
close_position_15m, recovery_atr, reclaim_vwap, reclaim_zone,
sector_rel_strength, leader_strength, stock_sector_lag,
rule_state, rule_score, model_score
```
后补真实结果：`entry_if_triggered, mfe_30m, mae_30m, mfe_1d, mae_1d, net_return_1d, net_return_3d, hard_stop, failed_rebound`。
→ 几个月后得到可训练 **盘中 Timing Head** 的数据。

## 八、保存 PASS / NEAR_MISS / CONTROL / INVALIDATED（关键改动）

不只存 rule_pass=1。接近样本（如 OpenDipATR=0.18/0.28 且 zone 条件满足）同样是重要对照——
只存 A 类会让模型只在人为窄域里学习（复制规则而非发现"0.27ATR+强承接可能优于0.22ATR+弱承接"）。
第一版监控故意保留探索空间。

## 九、rule_score 可保留但不当标签

LATR_RULE_SCORE 百分制：ATR下杀位置20 + 流动性区域25 + 承接/收复25 + 量价15 + 板块/龙头15。
作用：①压缩实时送模型数量；②日后比较 纯规则 vs 模型 vs 规则+模型。
**禁止 `rule_score>70` 当机器学习正标签——标签必须来自未来真实经济结果。**

## 十、总结构

```text
盘后 Tarot → BOTTOM_SETUP / INTRADAY_DIP_SETUP → 次日 LATR Watch
  → ATR下杀+Zone接近 → LATR ARMED → 保存全部候选/近似候选
  → 模型实时打分 → 承接/修复状态持续更新
  → BUY_CANDIDATE / REJECT → 真实路径+退出结果 → 重训 Timing/S0/R0
```
LATR 同时是：**现在的规则选股器 + 未来的训练数据生成器。**
模型拆三头：S0（昨晚值不值得观察）/ LATR Timing Head（这个下杀是不是好介入时机）/ R0（左尾风险）。
与路线一致：两类观察池不混排，INTRADAY_DIP_SETUP 必须进入预定区域且出现承接/修复后才输出 BUY_CANDIDATE。

## 下一步（用户指示）

盘后总结规则由用户随后发出，与本规格串联成总方案后冻结、再改代码。
关联证据：DIP_RECLAIM 假设注册表 5 例（hypothesis_dip_reclaim_case1.md，模型 5/5 全漏、
P_opp 压 0.43–0.48 门槛下沿、OOD 高热闸否决大赢家）——本引擎的承接/收复字段正是教模型认识该形态的数据基础。
