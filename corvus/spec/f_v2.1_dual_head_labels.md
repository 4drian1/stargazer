# F-v2.1 双头标签规格（冻结版 v1.0）

- 冻结日期：2026-09-12
- 状态：**只冻结定义，未生成结果、未训练、未调阈值**
- 前置：F-v2.2 复杂度实验结论为依据（见 `data/reports/global_baseline_c0c1c2.md` 第 12 节）
- 铁律不变：一日一行（trade_date × strategy_id）；候选池版本 `tarot.hermit.a2@v1` 先行；
  成本口径沿用 `costs_v2_frozen`；标签结束日隔离 + 5 交易日 embargo；CENSORED_TAIL 排除；
  阈值只由训练期/校准期冻结。

## 0. 动机（为什么拆分）

F-v2.0/v2.2 已证明：3% 触碰率高 ≠ 不值得买。外围下跌提高的是**短期波动与下探概率**，
但池内股票之后经常修复，"先下杀、后反弹"正是绿盘低吸/Wheel of Fortune 想要的机会。
若用单一 `loss3_touch` 头直接决定"暂停买入"，会把好机会错误过滤掉。

因此把目标拆成两个头：**波动头只警告，持续亏损头才门控**。

## 1. 波动头（Volatility Head）

```text
P_TOUCH3_3D
```

- 定义：候选池在 t+1 开盘入场后，未来 3 个交易日内触及 entry×(1-3%) 的**池级触碰率**。
- 与现有标签的关系：等价于现有 `stop_rate_3d`（`loss3_touch_rate`），**不新建标签列**。
- 用途（白名单）：
  - 波动警告（前端提示"今日波动风险高"）；
  - 提醒等待盘中下杀、调整入场时点（配合 Tarot Wheel of Fortune）；
  - 降低仓位/分批建仓的参考；
  - **不得**直接触发"暂停买入"。

## 2. 持续亏损头（Persistence Head）——互斥路径定义

对池内每只股票，t+1 开盘入场后按优先顺序判定唯一路径：

```text
NO_TOUCH             未来 3 日内从未触及 entry×(1-3%)
TOUCH_RECOVERED      曾触及 -3%，但第 5 日扣成本净收益 > 0（回到成本以上）
TOUCH_UNRECOVERED    曾触及 -3%，且第 5 日扣成本净收益 ≤ 0（仍未修复）
```

池级标签（一行一个 trade_date × strategy_id）：

```text
touch_rate_3d              = TOUCH 路径占比（= 1 - NO_TOUCH 占比）
unrecovered_given_touch    = TOUCH_UNRECOVERED / (TOUCH_RECOVERED + TOUCH_UNRECOVERED)
persistence_score          = touch_rate_3d × unrecovered_given_touch
```

- `persistence_score` 即 **P_persistent_loss**：下探且 5 日未修复的池级比例，是"暂停买入"
  的唯一候选信号。
- 边界规则：
  - 第 5 日净收益恰为 0 → 计入 TOUCH_UNRECOVERED（保守方向，宁可错杀）；
  - 停牌/数据缺失导致第 5 日净收益不可得 → 该股票剔除出本标签统计（不得填 0）；
  - 窗口与成本口径与 `tarot_day_labels@v1` 完全一致（t+1 开盘入场、costs_v2_frozen、
    3% 触碰、5 日修复判定）。

## 3. 动作映射（接入层契约，非模型输出）

| 波动头 P_TOUCH3_3D | 持续亏损头 P_persistent | 建议动作 |
|---|---|---|
| 低 | 低 | 正常观察 |
| 高 | 低 | 可能有下杀机会，等待盘中确认（交给 Wheel of Fortune） |
| 低 | 高 | 谨慎，可能缓慢走弱 |
| 高 | 高 | 暂停买入 / 高风险 |
| 数据异常（STALE/PARTIAL/FAILED/OOD） | 任意 | **UNKNOWN，不得视为安全** |

"高/低"阈值各自独立冻结：波动头阈值沿用 F-v2.2 冻结产物口径（校准期 80 分位），
持续亏损头阈值待标签生成后只在训练/校准期数据上冻结。

## 4. 训练顺序（待批准后执行）

1. 生成互斥路径标签（扩展 `tarot_labels` 或新表 `tarot_day_persistence`），
   与 `tarot_day_labels` 同粒度同闸门；
2. 波动头：复用现有 C2-S/C2-F 预测（标签未变）；
3. 持续亏损头：用 F-v2.2 胜出的简单模型（或精简版）训练 `persistence_score`；
4. 两头只在信息风险融合层（Astrolabe 前）合成，模型内部不互相调用；
5. 验收照搬放行池口径：只有 `persistence` 头参与的动作才考核"放行池收益不恶化 +
   避免亏损 > 错失盈利"；波动头只考核排序指标。

## 5. 明确不做

- 不重建 `loss3_touch_rate`；
- 不在本轮加入 HMM/ruptures/PatchTST/MASTER；
- 不用 2024–2026 数据调整任何阈值；
- 不让波动头单独触发暂停买入。
