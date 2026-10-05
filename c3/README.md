# C3 Pack：C2 公告回填 + C3 烟基线训练包（监控机可移植）

自包含：不依赖 corvus 代码、不依赖 PG、不依赖 Qwen/LLM、不吃本机开发数据。
外部依赖只有两样：**stockdb HTTP 服务**（行情 + universe）和**巨潮资讯**（akshare 个股公告接口，公网）。

## 监控机操作四步

```bat
REM 1. 装环境（用已有的 .runtime venv，或新建 .c3venv）
prepare_env.bat

REM 2. 启 stockdb（127.0.0.1:7899，日k 表可用即可）
REM    （按监控机自己的 stockdb 启动方式；本包只连 HTTP）

REM 3. 跑回填（2019-2025 全量；断点续跑，中断后重复同一命令即可）
python -X utf8 backfill_pipeline.py --phase all --start 20190101 --end 20251231 --out out

REM 4. 跑烟基线训练（自动闸门自检：不足 500 事件/150 交易日只出描述统计）
python -X utf8 c3_smoke_train.py --events out/event_cluster.parquet --out out
```
（`python` 用 prepare_env.bat 选定并验证过的解释器；开发机上即 `.venv\Scripts\python.exe`。）

产物（`out/`）：
- `event_cluster.parquet` — **canonical** 主表：1 行/event_cluster_id（cluster 级标签 + 代表性文档字段）
- `events_SUPERSEDED_V2_labeled.parquet` — SUPERSEDED/AUDIT_ONLY：修复后标签落盘中间 dump（含 3.28x 公告冗余），仅审计用
- `event_document_link.parquet` — 公告级文档链接（每条公告一行 + event_cluster_id）
- `raw_announcements.jsonl` — 原始拉取（幂等，去重键 code|date|title_hash）
- `progress.json` — 断点续跑进度（code|window 粒度）
- `source_stats.json` / `completeness_checklist.md` — 源完整性统计 + 人工核对清单
- `c3_smoke_report.md` / `train_manifest.json` / `models/` — 训练报告与模型文件
- `daily_cache/` — 逐票日k缓存（标签与特征复用，重跑不再拉行情）

## event_cluster.parquet 字段（canonical）

| 字段 | 含义 |
|---|---|
| code / name | 6 位代码 / 简称（巨潮返回） |
| announce_time / announce_date | 公告时间（巨潮个股接口仅日期粒度，见 DEVIATIONS D1） |
| title / title_hash / raw_url | 标题 / 标题哈希（去重键成分）/ 巨潮详情链接 |
| category / tier / matched_keyword | 标准事件类别（照抄 corvus 三层路由）/ P0-P2 / 命中关键词 |
| event_cluster_id | 事件簇（同票同日多公告合并，sha1(code\|date)） |
| label_start / label_start_basis | 反应起算交易日 / PREMARKET_OPEN、POST_CLOSE_NEXT_DAY、WEEKEND_NEXT_DAY、RESUME_FIRST_TRADE |
| ret_1d_pc / ret_3d_pc / ret_5d_pc | 相对 label_start 日前收的收益链（任务口径） |
| ret_1d / ret_3d / ret_5d | corvus entry 口径（label_start 开盘入场） |
| abnormal_1d/3d/5d | 个股 − 沪深300 同期收益（基准拉取失败则 NULL） |
| mfe_1d/3d/5d / mae_1d/3d/5d | 路径标签（窗口含除权日则 NULL） |
| gap_return / volume_shock / volatility_expansion | 跳空 / 量能冲击 / 波动扩张（corvus 第 5 节） |
| sigma250 / mat_1d / mat_3d / mat_5d / materiality | 前 250 日波动基线 / 单视野显著异常 / materiality=任一视野异常 |
| direction_3d | ret_3d_pc 符号（±0.1% 带宽内 FLAT，Secondary） |
| overlap_count / attribution_quality | 同票 ±5 交易日重叠数 / CLEAN(0) MODERATE(1-2) HEAVY(≥3) |
| label_status | OK / CENSORED_TAIL / NO_FUTURE_TRADE_DAY / NO_PRICE_DATA / FAILED:* |
| label_version | c3_pack_v1 |

## 参数与环境变量

| 参数/环境变量 | 默认 | 说明 |
|---|---|---|
| `STOCKDB_BASE` | `http://127.0.0.1:7899` | stockdb HTTP 地址（`--base` 可覆盖） |
| `STOCKDB_ROOT` | 无 | stockdb 数据根目录；仅 HTTP keys 失败时作 universe 降级路径（读其下 `data_daily_parts/part_*.parquet`），监控机路径不同必须能覆盖 |
| `--start` / `--end` | 20190101 / 20251231 | 回填窗口 |
| `--window-years` | 3 | 单请求年窗口（实测 7 年单窗口也可行；窗口越小断点粒度越细） |
| `--request-interval` | 0.7 | 请求间隔秒数（限速；实测 burst 7.2 req/s 不封，保守限速） |
| `--limit` / `--codes` | 全量 | 开发验证用：只跑前 N 票 / 指定代码 |
| `--assume-premarket` | 关 | 公告按盘前可见（默认盘后 → label_start 为下一交易日，见 DEVIATIONS D1） |
| `--refresh-universe` | 关 | 强制重建 universe 缓存（默认 `out/universe.txt` 复用） |
| `--spotcheck N` | 0 | 抽 N 例标签独立重算对拍后退出 |
| `--profile-mem` | 关 | tracemalloc 报告内存峰值 |
| `--seeds` / `--embargo-days` / `--train-frac` | 3 / 5 / 0.7 | 训练参数 |

## 速率与耗时实测（2026-09-27 本机）

- 巨潮个股接口：单票单年 ≈0.6–1.0s（含分页）；burst 7.2 req/s；7 年单窗口 600519=472 行/2.6s。
- 全量预估：universe ≈7,500 票（stockdb 2000 年起含退市）；`window-years=3` → 3 请求/票
  ≈ 22,600 请求，按 0.7s 间隔 + 服务时间 ≈ **8–12 小时**；`window-years=7` → 1 请求/票 ≈ **4–6 小时**。
- 行情回填（标签阶段）逐票 `日k:CODE:*` 全历史，本机 HTTP 约 1–3s/票 → ≈3–6 小时（缓存后重跑免费）。
- 内存：逐票流式处理，tracemalloc 峰值远低于 4GB（`--profile-mem` 实测打印）。

## 常见问题

- **akshare 报 KeyError 'xxxxxx'**：该股 secid 映射缺失（多为已退市代码）。回填如实记为
  permanent error 并跳过——存在存活者偏差，见 DEVIATIONS D7；不影响在交易股票。
- **stockdb 连不上**：两步都依赖 stockdb。先 `curl http://127.0.0.1:7899/?cmd=get&t=日k:600519:20260916` 验证。
- **中断续跑**：直接重跑同一命令；`progress.json` 跳已完成窗口，raw 载入时按去重键幂等去重。
- **events_SUPERSEDED_V2_labeled.parquet 想重算标签**：改口径后 `--phase label` 单独跑（用已缓存 daily_cache，秒级）。
- **闸门前别下结论**：source_stats.json 里 `trade_days_covered` / materiality 分布不够 500/150 时，
  c3 报告自动只出描述统计（BELOW_GATE）。
- **分钟级/30m 标签**：本包只有日级标签（stockdb 分钟表覆盖不足，见 DEVIATIONS D5）。
