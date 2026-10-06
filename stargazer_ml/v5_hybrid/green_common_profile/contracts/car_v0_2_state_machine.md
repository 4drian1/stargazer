# CAR v0.2 状态机定义

> 版本: v0.2; 状态: 待签署; 日期: 2026-09-29
>
> v0.1 行为: APPROACH 首次接近即触发事件 (旧版); 保留 v0.1 结果不覆盖.

---

## 1. 核心状态转移

```
[开盘]
  |
  +-- 首监控bar open ≤ zone_lower → ALREADY_BELOW
  |
  +-- bar.low ≤ zone_lower → 跌破候选
  |     |
  |     +-- bar.close ≥ zone_upper → RECLAIMED_AT_S (不触发事件)
  |     +-- bar.close < zone_upper → 触发事件 (BREAK / ALREADY_BELOW)
  |
  +-- zone_lower < bar.low ≤ zone_upper + band → APPROACH (仅观察)
  |
  +-- 其他 → 继续观察
```

- **接近 zone**: low 进入 (zone_lower, zone_upper + band], 仅观察, 不触发.
- **跌破 zone_lower**: low ≤ zone_lower → 候选触发.
- **同 bar 收回**: low ≤ zone_lower 且 close ≥ zone_upper → RECLAIMED_AT_S.
- **触发条件**: low ≤ zone_lower 且 close < zone_upper → 事件 S.

---

## 2. 信号类型分流 (v0.2)

| 信号类型 | 判定条件 | 主标签分母 |
|----------|----------|-----------|
| APPROACH | zone_lower < low ≤ zone_upper + band, 无跌破 | **否** (仅观察) |
| BREAK | low ≤ zone_lower, close < zone_upper, 非首bar跳空 | 是 |
| ALREADY_BELOW | 首监控bar open ≤ zone_lower, close < zone_upper | 是 |
| RECLAIMED_AT_S | low ≤ zone_lower, close ≥ zone_upper | 否 |

### 2.1 APPROACH 的 v0.2 定位

- APPROACH **不触发事件**, 仅记录进入观察状态.
- 后续 bar 若满足跌破条件 → 从该 bar 触发事件.
- 全天仅 APPROACH 无跌破 → 当日无事件.

### 2.2 跳空已在下方 (ALREADY_BELOW)

- 首监控 bar open ≤ zone_lower 且 close < zone_upper → ALREADY_BELOW.
- 首监控 bar open ≤ zone_lower 且 close ≥ zone_upper → RECLAIMED_AT_S (跳空即收回).

### 2.3 同 bar 跌破又收回

- 判定: **low ≤ zone_lower 且 close ≥ zone_upper** (独立于 close < zone_lower 分支).
- 结果: RECLAIMED_AT_S, 不触发事件.
- 不进入 `close < zone_lower` 分支 (该分支 close 必然 < zone_lower < zone_upper).

---

## 3. 多锚点合并

- 引擎 `build_zones` 已按 0.35*ATR 链式合并锚点为区域; `target_zone` 选取最近武装支撑.
- 单事件关联一个 zone; `zone_sources` 记录合并的锚点列表 (MA20/BOLL_L/PREV_LOW).
- **不临时合成新边界**: 沿用引擎冻结的 zone_upper / zone_lower, 不取最小/最大值.

---

## 4. 当日去重规则

- 每 (code, E) 至多一个物理事件.
- 取监控窗 [09:35, 14:55] 内**首个**满足触发条件的 bar.
- 先 APPROACH 后跌破 → 只计跌破事件, APPROACH 不消耗事件机会.
- ALREADY_BELOW (首bar) 后不再扫描 → 当日事件已确定.

---

## 5. 状态转移伪代码

```
# 输入: E日分钟数据 (date/open/close/high/low/volume/amount), zones, atr_ref, ref_price
# 输出: 事件 dict 或 None

tz = target_zone(zones, ref_price)  # 冻结区域, 不临时合成
if tz is None: return None

zone_upper = tz["zone_upper"]
zone_lower = tz["zone_lower"]
band = APPROACH_ATR * atr_ref

for bar in E_day_monitor_window:  # 09:35 ≤ hhmm ≤ 14:55
    # 分支1: 首bar跳空已在下方
    if first_bar and bar.open <= zone_lower:
        if bar.close >= zone_upper:
            return RECLAIMED_AT_S(bar)
        else:
            return ALREADY_BELOW(bar)

    # 分支2: 跌破 (含同bar收回)
    if bar.low <= zone_lower:
        if bar.close >= zone_upper:
            return RECLAIMED_AT_S(bar)  # 同bar收回, 不触发
        else:
            return BREAK(bar)  # 跌破且未收回

    # 分支3: 接近 (仅观察, 不触发)
    if zone_lower < bar.low <= zone_upper + band:
        continue  # APPROACH → 观察

    # 分支4: 在 zone 上方, 继续观察
    continue

# 全天未触发
return None
```

---

## 6. 与 v0.1 的差异

| 场景 | v0.1 | v0.2 |
|------|------|------|
| APPROACH | 触发事件 | 仅观察, 不触发 |
| 接近后未跌破 | 事件 (RECLAIMED_AT_S) | 无事件 |
| 先接近后跌破 | 事件 (首个 APPROACH) | 观察后触发, 取跌破 bar |
| 同 bar 跌破收回 | RECLAIMED_AT_S | RECLAIMED_AT_S (一致) |
| ALREADY_BELOW | 触发事件 | 触发事件 (一致) |
| BREAK | 触发事件 | 触发事件 (一致) |

---

## 7. 标签计算

v0.2 触发的事件使用与 v0.1 相同的 CAR 标签函数 (`compute_future_reclaim_window` / `compute_future_hold`):
- 60m reclaim (主标签)
- 30m reclaim (辅助)
- 10m hold (辅助)
- 穿透深度 (截至 S)
- 窗口完整性/截断标志

标签计算逻辑不变, 仅事件触发规则改变.

---

## 8. 修正记录

- 2026-09-29: 修正伪代码不可达分支 (`close < zone_lower` 内判断 `close ≥ zone_upper` → 独立分流).
- 2026-09-29: 修正多锚点处理 (沿用引擎冻结区域, 不临时合成边界).
- 2026-09-29: 修正盘点读取旧报告问题 (v0.2 提取器直接扫描缓存重新生成).
