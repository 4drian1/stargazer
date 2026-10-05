# DEVIATIONS.md — 本包对 corvus 规格的偏离登记

口径来源：`corvus/spec/minimal_event_price_loop_v1.md`（冻结 v1.0）、
`corvus/corvus/price_labels.py`（event_loop_v1）、`corvus/corvus/event_taxonomy.py`
（2026-09-12 冻结 v1.0）、`开发路线.txt` Corvus-C2/C3 段。
本包自包含实现，凡与上述口径不一致处在此登记。

## D1. available_at 粒度退化到日（标签起点）

- corvus：available_at 为分钟级时间戳，按 PREMARKET_OPEN / INTRADAY_NEXT_MINUTE /
  POST_CLOSE_NEXT_DAY / WEEKEND_NEXT_DAY 四态决定 label_start。
- 本包：akshare 巨潮个股接口的公告时间字段实测只有日期（`公告时间`=YYYY-MM-DD），无盘中时点。
  默认假设**盘后可见**（巨潮公告绝大多数 15:00 后挂网，且个股日度口径下盘前假设会引入前视风险），
  label_start = 下一交易日开盘，basis ∈ {POST_CLOSE_NEXT_DAY, WEEKEND_NEXT_DAY}。
  少数盘前发布的公告会被错记一天；提供 `--assume-premarket` 全局开关切换口径后整库重算
  （`--phase label` 幂等重算）。
- 影响：ret_to_close / ret_30m / 盘中起点不可用；日内反应信息全部丢失。
- 生产冻结契约（materiality_head_v1_candidate）：冻结数据集采用默认口径（无 `--assume-premarket`），
  label_start_basis ∈ {POST_CLOSE_NEXT_DAY, WEEKEND_NEXT_DAY, RESUME_FIRST_TRADE}，
  即“次日开盘可见”保守可见性；premarket 全局开关**不用于生产冻结**，避免将当日盘中行情归因成反应
  （见 canonicalize_dataset.py / walk_forward_final_results.json）。

## D2. 收益口径双轨

- corvus：ret_h 以 label_start 开盘价入场，ret_h = close(idx)/entry × chain(idx+1..idx+h) − 1。
- 任务口径（本包主口径）：**相对 label_start 日前收**，ret_h_pc = close(idx+h−1)/pre_close(idx) − 1，
  用 pct_chg 链跨除权（照抄 corvus 的 daily_factor 除权守卫）。
- 两口径并列落盘（`ret_{h}d` 与 `ret_{h}d_pc`），materiality/direction 用 `_pc` 口径。

## D3. materiality 初标签为本包新增（corvus 无此标签）

- corvus 只有反应字段，不定义"显著异常"二值标签。
- 本包按任务规格：mat_h = |ret_h_pc| > 2 × σ250（σ250=前 250 交易日 pct_chg std，≥120 个有效值，
  不足则 NULL 不编造）；materiality = mat_1d ∨ mat_3d ∨ mat_5d。
- 这是**初标签**（烟基线用），非 corvus 规格口径。

## D4. 基准：仅大盘，无板块基准

- corvus：个股公告同时算大盘（沪深300）与所属板块两个基准。
- 本包：stockdb 日k 无板块/行业字段，板块基准不可得；只算沪深300
  （akshare 新浪 `stock_zh_index_daily`，缓存 parquet，失败 abnormal 列留 NULL，不编造）。
  板块特征用代码前缀板块代理（MAIN_SH/MAIN_SZ/CYB/KCB/BSE，见 `_taxonomy.board_of`）。

## D5. 无 30m / 分钟级 / 反身性标签

- corvus：ret_30m、initial_direction、reversal_30m、reclaim_* 等从分钟线算。
- 本包：stockdb 分钟表实测对本机历史日期无覆盖（`分钟k:600519:20260916` 返回空），
  监控机不保证分钟源；日级标签外一律不生成（字段缺失即缺失，不填 0）。

## D6. 分类无 LLM 抽取，P1 容器不升级

- corvus：P1 容器文档（董事会决议等）经 Qwen 读正文后可升级 P0。
- 本包：无 LLM；分类=标题关键词规则（照抄 taxonomy 冻结词表），P1 保留为 P1，
  训练域只取 P0。DISTRIBUTION 按 corvus 语义不进风险模型（训练域剔除）。

## D7. 存活者偏差（数据源限制）

- akshare 巨潮个股接口对已退市代码报 KeyError（secid 映射缺失）或返回 0 行
  （实测 400072 KeyError、000003 空）。退市股历史公告回填不到。
- universe 来自 stockdb（2000 年起含退市股），但公告拉取会跳过这部分：
  `source_stats.json.fetch_windows_permanent_error` 记录规模。
  2019-2025 区间内退市股占比有限，但正式结论须带此 caveat。

## D8. 存储与幂等锚

- corvus：PG 表，幂等锚 (event_id, entity_id, label_version)，UNIQUE(raw_id, event_version)。
- 本包：parquet + jsonl + progress.json。去重键 = code|announce_date|title_hash
  （对应 raw 级）；簇级锚 = event_cluster_id + label_version。修订/撤回公告（corvus 第 6 节
  event_version 链）**未实现**——同一标题修订版会作为独立公告存在（title_hash 不同），
  同题转载被 title_hash 吸收。

## D9. 交易日历

- corvus：日线 parquet 日期 ∪ 沪深300 指数日期。
- 本包：stockdb 6 只跨板块大票（000001/600000/300750/688981/601398/830799）日k 日期并集，
  缓存 `calendar.json`。与交易所日历差异限于这 6 票共同停牌日（概率可忽略）。

## D10. 事件重叠 / HEAVY 处置

- 照抄 corvus：同票 ±5 交易日重叠计数，CLEAN/MODERATE/HEAVY。
  HEAVY 样本训练域排除（corvus：第一代训练降权或排除，取更保守的排除）。
