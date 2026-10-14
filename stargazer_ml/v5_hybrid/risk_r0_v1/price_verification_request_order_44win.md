# 44 窗价格核验批次（有界执行件 — 零网络准备完成，取数待单独授权）

状态：ZERO_NETWORK_PREP_DONE（代码/冻结清单/自测已就绪）；**取数未授权、未执行**。本文件是 `bounded_independent_verification_request_order.md` 的 44 窗实例化，且已落到可执行代码 `price_verification_batch_44win.py`。

## 0. 定位（诚实定性）
- **外部交叉核验源**：复用**已验收的腾讯受控客户端** `independent_fetch_client.TencentDailyClient`（直连 `web.ifzq.gtimg.cn`/`proxy.finance.qq.com` 原始 `day` 序列，拒收 qfq/hfq 复权键）。
- **不重新宣称强独立性**：腾讯源与 stockdb 属不同口径/不同通道，但**非强独立源**；定位为"交叉核验"以发现系统性口径偏差，不能作为独立来源证据替代。强独立性须另定来源（见框架单 §1，仍 TBD）。
- 本批**只做价格一致性比对**，不认证公司行为完整性（覆盖核销另走 R1–R4，分别出结论）。

## 1. 比较点（冻结清单已生成）
- 由 `five_day_window_map_v3.json` 枚举 44 窗 → **264 引用 = 1 P 收盘 + 5 逐日 low**；按 `(code,date,field)` 去重 → **225 唯一点、30 股票**。
- 冻结清单：`independent_fetch/batch_d_mainpop_30/price_comparison_points_44win.json`（含 map 输入哈希 + 三脚本代码版本）。
- **既有响应覆盖 = 0**：扫描 `run_20261013T...` 仅含 Sina 公司行为页（分红送转），非腾讯价格响应，**不可复用为价格点** → 本批需新取。

## 2. 取数计划（按股票合并、去重）
- 每股票一次区间请求（start=min(所需日)−7d，end=max(所需日)+2d）→ **30 请求 + 2 冒烟 = 32**；**请求硬顶 = 36**（估算 +4 余量）。
- 冒烟 2 次零重试（本批股票各一）→ 失败即整批终止（REV6 治理，沿用 `independent_fetch_run.py`）。
- 北交所 2 例（`920729`/`920808`）登记覆盖缺口，不请求（与本批 30 股票无交集，已确认空）。

## 3. 比对字段与容差
- P 日参考收盘 `R_P` + 后五日（P+1..P+5）**各自 low**（逐日保留比对，不聚合为单值）。
- **|diff| ≤ 0.01 元** 同口径逐点判定 MATCH；否则 DIFF；任一侧缺失 → MISS_LOCAL/MISS_REMOTE。
- 离线比对器：`compare_point` / `mae_5d` / `boundary_sensitivity`（已 `--selftest` 6/6 PASS）。

## 4. 边界敏感性（MAE≈−8%，非 ±8%）
- 标签边界 `MAE_5d = min(low_d/R_P − 1)` 的判定线 ≈ **−0.08**。
- 对基线接近 −8% 的窗口，以 ±0.01 元扰动绑定 low 与 `R_P` 重算；若翻转标签方向则标 `BOUNDARY_SENSITIVE`，**单列、不静默判类**。

## 5. 结论范围（不一键放行）
- 取数后**分别**输出：价格匹配 / 缺失 / 差异 / 公司行为覆盖状态。
- 价格匹配可为**已查价格点**提供尺度证据，**不代表覆盖完整性或历史可知时间也通过**；覆盖与历史时刻门各自独立核销。
- 结果**仅覆盖已查 44 窗**，不外推全库、不自动授权标签或训练。

## 6. 治理与记账（复用已验收客户端）
- `FetchCounter` 全批共享硬顶 36（含冒烟/重试/辅助）；超限即停。
- `ByteCounter` 响应 5 MiB + 落盘 5 MiB（含 256KB 报告预留）；禁重定向；原始响应 `.bin`+`.json`+`index.jsonl` 留证（异常同样留证）。
- 失败即停、**不自动换源**；不覆盖旧数据；不扩 1,013；不生成标签；不训练。

## 7. 历史研究用途（另签范围，不在本批）
- 历史到达日志无法补造，不因此无限等待。可另签"历史价格重建研究"范围，明确可用性假设与局限；**不能称为真实当时可用的回放**。
- 标签验收后先盘点有效事件、尾部阳性和日期簇，再决定是否扩大人口、冻结特征与拟合合同。44 窗只是工程批次，**不能直接视为 LightGBM 训练人口**。

## 8. 执行入口
- 零网络：`python price_verification_batch_44win.py --dry-run`（计划/覆盖）/ `--selftest`（比对器自测）。
- 取数（**需单独授权**）：`python price_verification_batch_44win.py --fetch <archive_dir>`，落盘 `price_match_table_44win.json` + 账本/报告。
