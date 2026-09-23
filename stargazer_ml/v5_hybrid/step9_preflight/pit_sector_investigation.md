# PIT Sector 历史来源追查 (Gate 1B 解锁前置)

状态: **未解决 -> Step 9B (A5-A8) 硬阻塞**。当前 `bk`(申万 category=1) 仅返回**当前成分**，无历史生效日期，回填历史会产生 membership look-ahead（未来分类泄露）。本文件跟踪按顺序的追查路径；第 ⑤ 步若仍无解，Sector/Leader 从 Step 9A 正式关闭，禁止 current mapping backfill。

## 优先级顺序 (先易后难，内部优先)

### ① stockdb 内部是否还有历史快照表 / 旧 bk 表 / 备份 / 日期分区 / 日志
- [x] 检查 stockdb 数据目录下是否存在按日期分区的板块快照（如 `bk_20250101` / `bk_history` / `板块历史` 之类表名）——**否定**：D:\a股数据\stockdb\ 下 data/(981 .ldb)/data1/(log 记载 9/16 打开失败)/mydb/ 全是 LevelDB，无日期命名文件；free-stockdb-windows zip 是安装包非数据
- [x] 检查交接文档 `STARGAZER_STOCKDB_交接文档.md` 是否提到历史成分接口或备份位置——**否定**：全文只覆盖日K/分钟K/复权/代码表
- [x] 直接 probe 候选表名（HTTP `get&t=<候选表>:<code>:<date>`）确认是否返回带 `effective_date/start_date/end_date/as_of_date` 的记录——**否定**：41 个候选表名全空；决定性证据：`板块*` 前缀扫描返回完整 1337 条（scan 上限不截断），字段并集仅 category/code/group/name/source/symbols/type，**无任何日期字段**；服务端 server.cpp:341-346 只做 scan_prefix("板块:")，README 自述板块设计即"当前映射毫秒查询"
- 判定: 若命中任一含生效日的历史表 -> 直接进入 Gate 1B 解锁，写回 `feature_registry` 对应特征 `gate1_phase` 改回 `CORE` 并补 `as_of` 字段。

### ② 现有上游数据源是否实际上提供 as_of，但当前接口没暴露
- [x] 比对 `free_stockdb_loader` / `stock_sdk` 是否对任意函数传入 `as_of` 参数但被 wrapper 忽略——**否定**：stock_sdk.get_data(start/end) 只作用于 K 线表
- [x] 检查 `v4_2_stockdb.py` / `loader_stockdb.py` 的 `boards` 实现是否有可选日期维度——**否定**：v4_2_stockdb.py:99-105 纯透传；loader_stockdb 不接触板块；zhibiao.py:66 BoardIndex.get(x, category, fields) 签名无日期参数，__init__ 一次性加载板块建内存索引
- 判定: 若 SDK 支持 `bk.get(x, category, date=<as_of>)` 形式 -> 用 as_of 拉取历史成分快照，构建本地 PIT 成分表。

### ③ 项目旧数据目录 / parquet / csv / dump 中是否保存过历史成分
- [x] 在 `d:/stargazer` 下搜索历史板块/申万成分文件——**否定**：全盘按关键词搜索，无任何带生效日期/区间的成分文件；现存全是当前快照（pullback_train/_fetch_cache/map_industry.json、cons_*.json 自标"幸存者偏差"；experiments/*/current_sw1_mapping_proxy.json 文件名即注明 current proxy）
- [x] 检查 `targetCorvusdata/`、`models/`、`migration_support/`、`_slave/` 等是否有历史快照 dump——**否定**（corvus.sqlite 10 表无行业/成分表）
- **附带发现（需另立标注）**：`v4_2_horizontal_doc_retest.py:28-49` 的 `build_sector_trend` 用**当前** SW1 映射回填 2018 年以来的 `sw1_sector_trend_proxy.parquet`——已有 sector 历史趋势产物本身是 look-ahead 构造，建议标记或待 PIT 表就绪后重建
- 判定: 若有带日期的历史成分文件 -> 解析为 PIT 成分表（确认其生效日字段口径），导入本地缓存。

### ④ 内部真的没有，再找具有历史生效日期的外部数据源
- [x] 评估可获取的外部申万/中信历史成分源——**已找到，且无需 token**：akshare `stock_industry_clf_hist_sw()`（1.18.94，本机 _dev_venv 已装），数据源=申万宏源研究官网官方 xls《行业分类变动历史》。实测 2026-09-17：12913 条变更记录，symbol/start_date(生效日)/industry_code(6位分类码)/update_time，覆盖 1990-01-01→2026-09-11，含 SW2014→SW2021 大改版全部变更。**Tushare 路径降级为备选**（当前账号全接口 40203 无权限；index_member_all 需 2000 积分；仅高校师生认证可免费领积分时再启用，token 已存 corvus/config/tokens.yaml）
- [ ] 抓 industry_code(6位分类码)→一/二/三级名称映射表（申万官网同目录，抓一次固化进本地）
- [ ] ETL 落成内部 PIT 成分表（规则：P 日 membership = start_date≤P 的最新记录），每次拉取**快照固版**（存档原始 xls + 版本号进 feature_registry）
- [ ] 低成本补充：从现在开始每日快照 stockdb `板块*`（增量小），与官方历史互为校验
- 判定: 若引入外部源 -> 统一 ETL 为内部 PIT 成分表，记录来源与版本，纳入 `feature_registry` 的 `source_table: sector` 注释。

### ⑤ 都没有：Sector / Leader 从 Step 9A 正式关闭
- [ ] 保持 `gate1_phase: CONTEXT_BLOCKED` 不变，A5-A8 状态置 `WAITING_FOR_PIT_SECTOR`
- [ ] 在 `label_contract_v1.yaml` 的 `blocked_until_pit_sector` 维持 SANITY_CHECK/EXPLORATORY_ONLY 限制
- [ ] **绝不允许** 用当前板块成分回填历史 F9 sector 特征或 sector alpha 标签
- 结论: 第 ⑤ 不是失败，只是 Step 9A 验证 Entry Core、Step 9B 等待数据基础设施。

## 当前已确认事实
- stockdb `bk`(申万 category=1) = 当前成分快照，无 as_of 维度（probe 实测 `sample_rows=1`）。
- `market_index_minute` FAIL（指数日线/分钟缺失），已由 `MARKET_PROXY-V1` 降级（不依赖 Sector）。
- 真实 VP/POC/VAL 仍 `OPTIONAL_DISABLED`（Gate 1C），不阻塞 Core。

## 跟踪记录
| 步骤 | 日期 | 结果 | 负责人 |
|------|------|------|--------|
| ①    | 2026-09-17 | 否定：LevelDB 无日期分区、交接文档无记载、41 候选表名全空、`板块*` 全量 1337 条无日期字段（server.cpp 佐证仅当前映射设计） | 探活代理 |
| ②    | 2026-09-17 | 否定：BoardIndex.get 签名无日期参数、get_data(start/end) 只作用 K 线、项目侧纯透传 | 探活代理 |
| ③    | 2026-09-17 | 否定：全盘无带生效日的成分 dump，现存皆当前快照；附带发现 sw1_sector_trend_proxy.parquet 系当前映射回填（look-ahead，另立标注） | 探活代理 |
| ④    | 2026-09-17 | 路径确认：Tushare index_member_all（in_date/out_date）为首选；待用户注册 token 实测覆盖；每日板块*快照作为增量补充 | 待用户决策 |
| ⑤    |      | 未触发 |        |
