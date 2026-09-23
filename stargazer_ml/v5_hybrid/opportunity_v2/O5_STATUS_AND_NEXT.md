# O5 STATUS & NEXT — Opportunity-Discovery-V2 交接文档

> 用途：新对话/agent 读本文件 + `D:\stargazer\开发路线.txt`（2026-09-23 各变更记录）即可继续 O5。
> 状态（2026-09-23 收尾）：**O5-0 / O5-1A / O5-1B / O5 主实验 全部完成；裁决 = `O5_MAIN_VERDICT = FAIL` → HYP-05 ARCHIVED_FAIL(no_edge)，O5 线关闭。**
> 证据：`o5_sector_lag/{O5_1A_REPORT.md, O5_1B_PARITY_REPORT.md, sector_lag_v0.md, sector_lag_v0_report.json}`。
> 路线已切换（见 §8）：**暂停新增 P_CLOSE 静态假设 → Gate1C 分钟级 same-time 基建优先 → INTRADAY_SECTOR_DIFFUSION_V1**。
> 环境：`.venv\Scripts\python.exe -X utf8`（pandas/pyarrow/torch 在此）；_dev_venv 用于 akshare/psycopg。

## 0. 记分板（截至 2026-09-23）

```text
O1  Strength Continuation    → ARCHIVED_FAIL（修复 LOO 后 4/4 年反向，IC −0.031 t=−5.6）
O2  Compression→Expansion   → ARCHIVED_FAIL（raw 口径倒挂，/ATR 是分母伪影）
O3  Breakout Retest         → ARCHIVED_FAIL（A vs B CI 含 0，A 组负 alpha）
Route Review                → FRAMEWORK_HEALTHY（12 项审计一致；universe 周~月尺度均值回复主导）
STEP1 GREEN V21 vs V22      → V22_COVERAGE_ONLY（recall +54% 但 enrich≈1.01，无 selection edge）
GREEN-v1.1                  → RESEARCH_NEGATIVE / SHADOW（每日闭环照常跑）
O5-0 申万 PIT membership ETL → PASS_FOR_RESEARCH（本文件 §1）
O5-1A SectorProxy            → DONE（32 键 = 31 index_code_l1 + 1 名称回退；valid 86-90% dev）
O5-1B GATE_LOO_PARITY        → PASS（1048 条 meta/math/store 全 0 不一致 + 10 项合成结构用例全过）
O5 主实验 (A/B/C/D)          → FAIL → HYP-05 ARCHIVED_FAIL(no_edge)（本文件 §4.1）
NEXT                        → 不开第五个 P_CLOSE 假设；交还路线级讨论（§8）
```

## 1. O5-0 产物（已完成，直接复用勿重跑）

目录 `stargazer_ml/v5_hybrid/opportunity_v2/o5_sector_lag/`：
- `pit_membership.parquet`：12920 行 × 5929 票；字段含 symbol/industry_code/l1/l2/l3 名称/effective_from/effective_to/name_source。**membership(P) 仅由 effective_from≤P<effective_to 决定（半开区间）**。
- `ETL_QA_REPORT.md`：QA 五项全 PASS；**GATE_1B = PASS_FOR_RESEARCH**，availability assumption 明写：官方 SW 分类自 effective_from 起可用；residual risk：源数据无可靠公告/可知时间戳（update_time 是维护戳，95.5% 晚于生效日，已证不可当知情时刻）。**日后若找到正式公告日期可补强，不得悄悄改历史结果。**
- `_upstream/`：原始 xls（sha256 `8940ac98…`）+ 各映射来源件，可复现。

已知边界：北交所（4/8 开头）不在申万表内（O5 universe 本就排除）；l3 与 stockdb 55% "不一致"是码空间错位（分类码 vs 85xxxx 指数码），**O5 用一级行业，不受影响**。

## 2. O5-1A：SectorProxy 构造（先做，不看收益）

对每个 (stock i, P)：membership 取 `pit_membership`；聚合**必须 exclude target**：

```text
sector_return_ex_i = Σ(j≠i, j∈sector(P)) w_j·r_j / Σ(j≠i) w_j
w_j = P_CLOSE 已知 float_mv；r_j = P_CLOSE 已知收益
输出: sector_l1_code_asof_P / sector_member_count / sector_return_5d_ex_i / sector_return_10d_ex_i / sector_strength_5d_ex_i / sector_strength_10d_ex_i
```
- sector_strength = sector_return − market_proxy_return（同窗口；MarketProxy 复用 O1 修复版向量化实现，含 LOO）。
- **成员过少或退化情形 `sector_proxy_valid=0` 置空，不硬算**：行业只剩 1 只、除 target 后有效成员 <3、目标票权重过大（如 >50%）、float_mv 缺失、停牌/无收益、行业切换 effective 当日。

## 3. O5-1B：LOO parity 硬 Gate（O1 教训，不过 gate 不许看 performance）

- 抽样 ≥300 stock-date：覆盖不同年份/行业/行业规模/市值/行业变更附近日期。
- 向量化 `sector_ex_stock` vs reference `exclude_symbol` 实现逐个对拍：**membership/member_count/denominator/最终 return 四项全查**，max_abs_error ≤ 数值容差。
- 专项用例：单票行业、双票行业、target 大权重、float_mv 缺失、停牌、effective date 当日。
- 未 PASS → 修到 PASS 为止，不进入 O5 主实验。

## 4. O5 主实验（parity 过了才开工）

**机制一句话**：已出现明确 sector impulse 的行业中，尚未同步反应的个股，未来是否存在独立 catch-up，而不是普通弱股均值回复？

**连续变量先行（防 threshold fishing）**：sector_strength / stock_relative_strength / lag_gap（= sector_excess − stock_excess，禁单独使用）——先看预注册分箱（quintile/decile）的 future H3/H5 alpha 梯度；**禁止事后改阈值找最好区间**。

**四组职责**：
- **A** = 强板块 + 滞后股（treatment；sector_strength≥冻结线 且 stock_response<sector_response 且 lag_gap≥冻结线——冻结线在看到分箱梯度前从经济意义预注册，不看结果挑）
- **B** = 同强板块 + 已同步股（**核心对照，Primary = A−B**）
- **C** = 同日 matched eligible（MATCH_V1：日期+float_mv 桶+成交额桶）
- **D** = generic reversal control（非强行业中 past-return/weakness 幅度相近票）——**A>B 且 A>D 才支持信息扩散；A≈D>B = 只是弱股反转，O5 故事不成立**

**Concentration gate（O5 特有，强制）**：by_sector 结果 + leave-one-sector-out（删最佳行业/删最大样本行业后主效应是否还在）+ sector event concentration + top-10 symbol 贡献。

**流水线**：事件经济学（count/coverage/H3/H5/A−B/A−D/MFE/MAE/loss/年份/fold/分箱梯度带方向标注）→ 判 PASS（A−B>0 且 CI 明确偏正且多年份多 fold 同向 + lag_gap 梯度非单桶驱动 + A 对 D 有增量 + 幅度有经济意义非 1-2bp + 集中度合格）→ 过才 ML。**A vs B 不成立 = ARCHIVED_FAIL，不上 ML 救。**

### 4.1 结果登记（2026-09-23，`o5_sector_lag/sector_lag_v0.md`）：FAIL

- **实现口径**：预注册阈值硬编码于 `sector_lag_v0.py` 顶部（STRONG_PP=+2.0pp / LAG_PP=+2.0pp / MIN_ALPHA=0.30pp）后单次执行；C 用 MATCH_V1（同日 qcut5 float_mv × qcut5 amount），D 用同日 `stock_ex_5d` 五分位桶、池 = `~grp_strong`（与 A 集合不重叠）。
- **母池**：dev(2021-2024) 有 PIT 行业归属 **4,486,489 行 / 969 日**（conf 2025+ 仅列示）。组规模：A=364,483(8.12%) / B=312,760(6.97%) / A2=249,670 / B2=149,568 / strong=826,811(18.43%)。
- **预注册判据 8 条**：P1 A−B>0 且 CI 下界>0 **YES**；P1b pooled A−B>0 **YES**；P2 年份≥3/4 为正 **YES**；P2b fold≥3/4 为正 **YES**；**P3 lag_gap 剂量梯度 spearman≥0.6 且去最强桶仍>0 → NO（实得 0.20）**；**P4 A−D>0 且 CI 下界>0 → NO**；**P5 |A−B|≥+0.30pp → NO（实得 0.261pp）**；P6 集中度 **YES**。→ 任一不满足即 FAIL，**不调阈值重跑**。
- **主数字**：paired A−B = **+0.261pp** [+0.145,+0.364] p<0.001；pooled A−B = +0.125pp；**A−C (MATCH_V1) = +0.041pp [−0.030,+0.116] p=0.132**；**A−D (generic reversal control) = +0.015pp [−0.072,+0.101] p=0.370**。
- **年份 / fold**：A−B 2021 +0.037 / 2022 +0.390 / 2023 −0.076 / 2024 +0.082（dev 3/4 正）；fold f1 +0.037 / f2 +0.390 / f3 −0.076 / f4 +0.082。conf: 2025 +0.354 / 2026 −0.128（仅列示，不参与裁决）。
- **梯度（关键机制证据，dev 10 分位）**：强行业内 lag_gap 均值 —— bin0（已跑赢行业，lag_gap≤−0.067）= **−0.139%**，bin1-8 几乎恒定 **+0.26~+0.32%**，bin9 +0.443%；单调 spearman **0.20** → 是**单步台阶（"是否已跑赢行业"）而非"越落后越补涨"的剂量关系**。对照：全母池 stock_ex_5d 梯度 spearman −0.04（hi−lo −0.055pp，弱反转）；全母池 sec_str 梯度 spearman 0.76（hi−lo +0.144pp）。
- **集中度**：base paired 0.212%；去最佳行业 0.211%；去最大行业 0.251%；top-10 票贡献 10.4% → 非集中驱动（P6 达标）。
- **三点判读（均指向"无跨实体信息增量"）**：
  1. A−B 的 0.26pp **全部来自 B 组（强行业内过去 5 日已跑赢行业的票）未来更差**（−0.14% vs +0.28%）= **短期 winner 反转**，不是 laggard 补涨；与 O1 的 `EXTREME_STRENGTH_SHORT_TERM_REVERSAL`（HYPOTHESIS_LOG_ONLY）及 Route Review Q2（k≥5 日自相关全负）一致。
  2. **A≈D 且 A≈C**（两者 CI 均含 0）→ A 的收益可被"同档弱势、非强行业票"或"同日 matched eligible"完全解释 = **无 sector 信息增量**。
  3. lag_gap **无剂量梯度** → 处理组不是"行业已动、个股未吸收"的连续强度，只是"未跑赢行业"的二元分界。
- **结论**：O5 故事（板块冲击 → 个股有暴露未同步 → 独立 catch-up）**不成立**；HYP-05 = ARCHIVED_FAIL(no_edge)，不进 ML/Timing/Risk。**禁止**改阈值/换窗口/换匹配口径救援；不设 HYP-05b。

## 5. 复用基建清单（全部已验证，禁止重造）

| 用途 | 文件 |
|---|---|
| 数据装载/eligibility/标签日历/split/MATCH_V1 | `opportunity_v2/hyp02_compression/expansion_v0.py`（最新，含 warm-up 修复） |
| MarketProxy 向量化复刻（LOO 已对拍） | `hyp01_strength_continuation/evidence_v1.py` 内实现 |
| 事件/对照/bootstrap 范式 | `hyp03_breakout_retest/retest_v0.py` |
| 报告纪律 | 方向标注合同 / DENOMINATOR_ARTIFACT_GATE / LONG_LOOKBACK_WARMUP_CONTRACT（路线 2026-09-23） |
| dev/confirmation 隔离 | dev=2021-2024 主裁决；2025+ 只列示不裁决 |

## 6. 纪律红线（违反即作废）
不看结果挑阈值/挑桶；raw 为主口径、归一化只作 diagnostic；长回看特征全历史预热；LOO 类实现必须 parity gate；LLM 不进决策路径；未过事件层不上 ML；**O5 若 FAIL：不开第五个 P_CLOSE 假设填坑，交还路线级讨论（届时议题：研发预算是否从 P_CLOSE 静态状态部分转向信息事件/盘中形成的机会）。**

## 7. 并行线（不与 O5 互阻塞）
GREEN-v2.2 Shadow 每日闭环照常；T5 双名单设计的开工依据 = `green_v2/T5_SHADOW_DUAL_LIST_DESIGN_v1.md`（DESIGN_FROZEN 待令）；R0/Pyxis 数据积累继续。

## 8. O5 收尾 + 路线级交还（2026-09-23）

- **O5 线正式关闭**：HYP-05 ARCHIVED_FAIL。按 §6 纪律：**不开第五个 P_CLOSE 静态状态假设填坑**，不做阈值微调，不上 ML/Timing/Risk 救。
- **已裁定（2026-09-23 用户三条裁决；详见 `开发路线.txt` 路线变更记录 #6 与 registry `track_pivot_2026_09_23`）**：
  1. **转向盘中 / 信息事件：YES**。新假设另立编号 `HYP-07 = INTRADAY_SHOCK_OVERREACTION`（**不复用已归档的 O3**，避免证据链混乱）。
  2. **先补 Gate1C：YES，最高优先级**：它一次解锁 `INTRADAY_SECTOR_DIFFUSION` / `O6 LIQUIDITY_IMBALANCE` / `INTRADAY_SHOCK·WASHOUT` / GREEN-v2.2 Source B / same-time volume·turnover controls。Gate1C 后第一个正式盘中实验 = **`INTRADAY_SECTOR_DIFFUSION_V1`**（先于 Shock，因其测试的是新的信息维度；明确为**全新 intraday hypothesis**，不是"O5 换时间窗重跑"）。
  3. **`O5b-RELATIVE-STRENGTH-DIVERGENCE` 不立项** → `REGISTERED / DORMANT / DO_NOT_START`（本质仍是"单票自身相对强弱形态→未来收益"，处均值回复逆风；除非出现新的独立机制证据）。
- **新开发顺序**：`O1/O2/O3/O5 FAIL → 暂停新增 P_CLOSE hypothesis → Gate1C → INTRADAY_SECTOR_DIFFUSION_V1 →（若 FAIL）HYP-07 INTRADAY_SHOCK_OVERREACTION → O6 LIQUIDITY_IMBALANCE`。
- **Gate1C 契约**：`opportunity_v2/gate1c/GATE1C_CONTRACT.md`（冻结输出字段、数据源口径与硬约束）。
- **O5 留下的可复用资产（与本次结论无关，保留）**：
  - `o5_sector_lag/pit_membership.parquet`：PIT 申万 membership（12920 行 × 5929 票），GATE_1B = PASS_FOR_RESEARCH；`etl_pit_membership.py` 可复现（原始 xls sha256 `8940ac98…`）。
  - `o5_sector_lag/o5_1a_sector_proxy.parquet`：exclude-target 的 float_mv 加权 sector/market 5d/10d proxy（7,749,477 行 × 1386 日，2021-01-04~2026-09-18），已过 LOO parity；键 = `index_code_l1`。
  - `o5_sector_lag/o5_1b_parity_gate.py`：**两层对拍范式**（float64 数学层 + float32 存储层 + 合成退化用例），可作后续任何 LOO/剔除类实现的 Gate 模板。
  - **不入档为可复用**：`sector_lag_v0.py`（O5 专用事件经济学，阈值硬编码，禁止换参重跑）。
