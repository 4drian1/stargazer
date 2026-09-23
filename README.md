# Stargazer

A 股绿盘低吸研究与实盘影子监控系统。盘后生成候选与交易计划 → 人工冻结 → 盘中规则监控推送 → 人工买卖 → 三轨迹归因落库；并行运行机会发现研究线（Opportunity-Discovery-V2）。

## 仓库内容

| 目录 | 内容 |
|---|---|
| `stargazer_desktop/` | 桌面程序（pywebview+Flask）：交易计划页（双名单）、持仓监控、推送、盘后总结 |
| `stargazer_plan/` | 交易计划库：CLI（draft/freeze/settle/record/export/merge）、PG 状态机、不可变版本 |
| `corvus/` | 新闻/公告采集（监控机 CorvusCollector 源码） |
| `stargazer_ml/v5_hybrid/` | 精选研究脚本：green_watch_v0（池）、green_v2（T5 影子）、opportunity_v2（O5 等）、step9_preflight、green_exit_v1 |
| `开发路线.txt` | **唯一主路线**（新对话必读） |
| `交接摘要.md` | 状态速查与每日操作协议 |

## 密钥政策（红线）

- 真实密钥**永不入库**：`config.yaml`、`corvus/config/pg.yaml`、`corvus/config/tokens.yaml` 均被 gitignore，仓库只提供 `*.example.yaml` 占位模板。
- 历史即未来：即使仓库某天转 Public，任何一次提交里的密钥都会永久可见。所以密钥只走本地文件，不走 git。
- `.gitignore` 为白名单制（默认全忽略再放行），并全局兜底排除 `*.parquet/*.sqlite/*.zip/*.exe/**/tokens.yaml` 等。

## 环境

- 双 Python：`.venv`（torch/ML，pandas 3.0.5）、`_dev_venv`（psycopg/akshare/PG 工具）
- 数据：stockdb 引擎（`D:/a股数据`，日线/分钟/板块）、PG（localhost:5432，计划库+Corvus）
- 命令统一 `python -X utf8`，中文输出需 `PYTHONIOENCODING=utf-8`

## 本机日常流程

1. 盘后：`数据更新.exe` → 重启引擎 → 打开 Stargazer.exe → 「交易计划」页 → **盘后选股**（全量 T5 打分约 10 分钟）→ 看双名单（规则层 ∥ 模型影子，重叠高亮）→ 勾选/改价/填理由 → **生成计划** → 确认后 **冻结并导出**（产物 `data/plan_frozen.json`）
2. 拷 `plan_frozen.json` 到监控机 `stargazer_desktop/dist/data/` 覆盖
3. 盘中：监控机自动按冻结计划跑 DSL 规则，触发/失效合并一条微信推送；15:05 日终汇总（无触发也报告）
4. 收盘后：监控机 `plan_events_*.jsonl` + `plan_daily_summary_*.json` 拷回本机 → `python -m cli merge_events` 归库；有买卖报我落 `record_buy/record_sell`（卖出必选 risk_type 九枚举）→ `settle` 五结局归因

## 目标机（监控机）部署与更新

**首次**：本机 release 的 `Stargazer.exe` 覆盖监控机 `dist\`；config 用监控机**现有本地 config**（勿覆盖，token 已在里面）；`plan.enabled: false` 隐藏交易计划页。

**代码更新（本仓库启用后）**：
```bat
git pull
```
**程序更新**：到 GitHub 仓库 → Releases → 下载最新 `Stargazer.exe` 覆盖 `dist\`。

## 开发流程

- 路线变更先记 `开发路线.txt` 再动代码（路线纪律 §15）
- 研究纪律：先注册假设再实现；raw 口径为主、归一化只作 diagnostic；LOO 类实现必须 parity gate；未过事件层不上 ML
- 发版：改代码 → push → PyInstaller 重打 exe → 挂 Release

## 状态（2026-09-23）

GREEN-WATCH v1.1 = RESEARCH_NEGATIVE（SHADOW 照常）；OD-V2：O1/O2/O3 归档、Route Review HEALTHY、O5 进行中；T5 双名单 + 交易计划 UI 已交付本机。
