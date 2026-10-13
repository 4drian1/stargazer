# R0-V1 外部证据核销报告（v2：机器可核验证据；多项 UNVERIFIED）

- **状态**：构建器 `build_r0_labels_v1.py` v5 已收口（39 项合成测试全 PASS）。本报告为**证据核验产物，不是核销通过报告**；停止、不训练的判断不变。
- **方法**：`risk_r0_v1/verify_external_evidence.py`（只读、零网络、可复跑）实算；输出 `verify_external_evidence_output.json`。所有数字来自本地真实文件，非估算。
- **放行前提**：下列任一项标 `UNVERIFIED(未核销)` 的，须补齐证据后方可裁决标签工程构建；目前**不宜签署放行或训练**。

---

## 0. 三处表述纠正（用户 2026-10-10）

1. **canonical 日历 ≠ 已认证交易所日历**。`trading_calendar.load_calendar()` 仅汇总日线中出现的日期；保证各模块一致，但**不证明没有漏掉交易日**。还须与独立交易所日历逐日比对（差集/补集）。
2. **`data_epoch_fingerprint()` 不是内容哈希**。它只用路径、大小、mtime、日期终点；可作快照标识，不能替代文件 SHA256。报告中旧 `34 文件 / 20260917` 指纹**不能直接绑定当前输入**——已用实算证明漂移（见 §1）。
3. **`get_daily()` 没有保证 RAW**。源码为"parquet 优先、HTTP 兜底"，无价格尺度转换或验证。"原样返回记录" ≠ "返回未复权价格"。原报告"设计上单复权"降为**待验证假设**（见 §4），不是已核销结论。

---

## 1. 日历证据表

| 项 | 实测值 | 结论 |
|---|---|---|
| 来源 | `trading_calendar.load_calendar()` 镜像，来自 `data_daily_parts` 全部 `date` 列 | 模块一致源 |
| 实际文件数 | **39**（当前） | 见指纹漂移 |
| 交易日数 | **2125** | 由日线日期推导 |
| 区间 | **20180102 .. 20261009** | 由日线推导 |
| 周末污染 | **0**（无周六/日） | 自洽，但非完整证明 |
| 长缺口 (>4 日) | **118** | 须独立日历判定是否漏交易日 |
| 当前 `data_epoch_fingerprint` | `daily_parts@39files_s641835930_m1791546571_epoch20261009_h0947991259779687` | 元数据快照，**非内容哈希** |
| 冻结时 `data_epoch_fingerprint` | `daily_parts@34files_s638278847_m1790509458_epoch20260924_h51f0af4422d8ab5c` | 34→39 文件、epoch 20260924→20261009、哈希不同 |
| **与独立交易所日历差集** | **UNVERIFIED** | 仓库内无独立交易所日历源；需 SSE/SZSE 官方休市日历逐日比对（缺日 / 多余非交易日） |

**结论**：日历是"模块一致源"，**不是认证交易所日历**；与独立日历的差集核验 **UNVERIFIED**。

---

## 2. 复权证据表（逐码，非全局布尔）

- 来源：`feb2026_adjustment_cache_v1/adj_cache.json` + `union_codes.json` + 失败日志（本地，零网络）。

| 状态 | 计数 | 说明 |
|---|---|---|
| OK | **5407** | 有事件已建链 |
| EMPTY | **70** | 确认无公司行为 |
| MISSING（相对并集 5477） | **0** | 并集码全部已缓存 |
| FAIL（去重） | **3** | `301552` / `603095` / `688686`（未落盘，须重试） |

- **覆盖证据**：OK 码事件日期跨度 `min_event .. max_event`；整体覆盖终点 **`max_event_date = 20260928`**（仅为最晚事件日，非"完整覆盖终点"声明）。
- **逐候选 `[P, W末日]` 覆盖交叉核验**：**UNVERIFIED**。原因有二：① 1,013 候选实际存储表待定位（见 §3）；② 即便定位，若某候选 `W_end > 20260928` 则覆盖不足，须逐候选判定。
- **验收结论**：缓存状态分离（OK/EMPTY/MISSING/FAIL）已落地可用；但"覆盖 `[P,W]` 完整"与"逐码验收" **UNVERIFIED**。全局 `chain_coverage_verified` 仍须改为逐码覆盖证据（不得单布尔代替）。

---

## 3. 候选资格证据表（1,013）

- 来源识别：`DUAL_PATH = 1,013`（`risk_candidate_source_reconciliation_v1.md` 已识别）；但**实际存储表待定位**（该文档 §4.1 仍列为 TODO）。
- registry 汇总（`dual_path_e_day_registry.json`，2026-10-01）：`box_members_2025plus=1072`、`unique_codeE=1050`、`hit_valid=94`、`hit_invalid=4`、`planned_requests_need=952`、`outside_cutoff_codeP=22`、`calendar_max_date=20260928`。

| 逐行字段 | 状态 | 原因 |
|---|---|---|
| 原始资格引用 | **UNVERIFIED** | 实际存储表待定位（reconciliation §4.1） |
| P 日 | **UNVERIFIED** | registry 无逐事件 P |
| `candidate_rule_version` | **UNVERIFIED** | 规则 SHA256 已记录于 reconciliation 文档，未绑定到逐行 |
| `decision_available_ts` 状态 | **UNVERIFIED** | 无逐行到达时刻 |

- **原则纠正**：资格 = **截至 P 日决策时刻已知**，不能用后续行动（成交/后续价格）反推；`DUAL_PATH` 1,013 已识别但未物化逐行，不得用 124 准入子集替代完整人口。
- **验收结论**：人口线已识别、聚合计数可得；**逐行绑定全部 UNVERIFIED**。

---

## 4. 价格尺度证据表

- 日线 schema（实测算首文件）：`date, code, name, open, high, low, close, pre_close, volume, amount, turnover, pct_chg, amplitude, is_st, vol_ratio, total_share, float_share, total_mv, float_mv, pe_ttm, pb`。
- **内容哈希（实算 39 文件 vs 库存清单）**：`match=39, mismatch=0, unrecorded=0` → 日线数据文件与 `risk_population_inventory_v2.json` 逐字节一致（**内容哈希，非指纹**，正确回应第 2 点）。
- **库存清单缺陷**：其 `input_manifest` 混入 **9 个非数据文件**（`.py`/`.md`/`.json` 源码与文档），且 parquet 条目数与实际文件数可能不一致（陈旧）。该清单仅作参考，以本脚本重算的 39 文件为准。
- **RAW / 已调整口径**：**UNVERIFIED_HYPOTHESIS**。`get_daily` 无尺度转换/验证；schema 含 `pre_close` / `pct_chg`（可能已除权）。"原样返回记录" ≠ "返回未复权价格"。公司行为跨日归一须实测：`SymbolChain` 原始价接口不能未经核验套在已复权分片上（`R0_V1_DEVELOPMENT.md` §2）。
- **已有口径诊断**（库存清单备注，非结论）：前收缺失 1,980,503 行；与可用日线 close 比较差异 6,912,784 行——仅为口径诊断，不直接证明损坏或公司行为。
- **验收结论**：文件完整性已验证（SHA256 全一致）；**RAW/已调整口径 UNVERIFIED**；公司行为跨日案例 UNVERIFIED（须选含拆分/送股的具体码，比对 raw_low 与 `factor_at` 归一结果）。

---

## 5. 放行判定

- **已达可确认**：日线数据文件内容哈希一致（39/39）；复权缓存状态分离（OK/EMPTY/MISSING/FAIL，MISSING=0）；日历模块一致（周末污染 0）。
- **仍 UNVERIFIED（缺证据，未核销）**：
  1. 独立交易所日历差集（§1）；
  2. 逐候选 `[P, W末日]` 复权覆盖（§2）；
  3. 1,013 逐行资格绑定（§3）；
  4. RAW/已调整口径与公司行为跨日验证（§4）。
- 完成上述后方可裁决标签工程构建；**目前仍不宜签署放行或训练**。

---

## 6. 机器可复跑

- 脚本：`risk_r0_v1/verify_external_evidence.py`（只读、零网络；`python verify_external_evidence.py`）。
- 输出：`risk_r0_v1/verify_external_evidence_output.json`（含全部实测值与 `UNVERIFIED` 标记）。
- 本轮未发生：全量复权、真实人口标签生成、训练、数据采集、部署、旧输入修改。
