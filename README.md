# Stargazer

本地运行的 A 股观察与交易计划辅助程序：盘后本地选股 → 人工确认计划 → 盘中规则监控推送提醒 → 人工登记成交 → 复盘归库。程序不自动下单。

使用说明见 [docs/使用说明.md](docs/使用说明.md)，部署说明见 [docs/部署说明.md](docs/部署说明.md)。

## 目录说明

| 路径 | 作用 |
|---|---|
| `stargazer_desktop/` | 桌面程序本体（pywebview + Flask 服务） |
| `ui_next/` | 新界面前端（React + TypeScript） |
| `stargazer_ml/` | 选股与风险模型、行情与板块数据相关代码 |
| `corvus/` | 公告信息采集与信息层 |
| `c3/` | Corvus 公告重要性模型（I33/D2 推理） |
| `stargazer_plan/` | 交易计划库（草案/冻结/成交登记 CLI 与状态机） |
| `migration_support/` | 迁移与恢复支持资料 |
| `release_audit_v1/` | 发布指纹清单与打包校验工具 |
| `docs/` | 使用说明、部署说明 |
| `开发路线.txt` | 开发主路线记录 |
| `交接摘要.md` | 状态速查 |
| `README.md` | 本文件 |
| `.gitignore` | 白名单制忽略规则（密钥/数据/模型/构建产物不入库） |

密钥只存本机 `config.yaml`（不入库），仓库仅提供 `config.example.yaml` 占位模板。
