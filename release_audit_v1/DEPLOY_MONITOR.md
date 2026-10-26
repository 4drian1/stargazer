# Stargazer 监控机部署说明（路线 #18.46，2026-10-26）

目标机：`C:\Users\10757\trading\stargazer`（下称 <项目根>）。EXE 与代码分离发布：
代码/模型/资产随 git tag 走，EXE（>100MiB）走 GitHub Release。

## 1. 装哪个版本

- 代码：clone 后 `git checkout v1.0.0-rc1`（tag 由 #18.46 推送）。
- EXE：GitHub Release `v1.0.0-rc1` 资产 `Stargazer.exe`，放到 `<项目根>\stargazer_desktop\dist\Stargazer.exe`。
- 资产包：Release 资产 `stargazer_assets_v1.0.0-rc1.zip`（内容与 git 内资产一致，附 sha256manifest.txt），
  可选下载校验用；git 已包含全部必需小资产。

## 2. 环境要求（硬性）

- **Python 3.12+**（开发/验收机为 3.13.14）。监控机现有 3.11.9 venv **不可用**：
  NumPy 2.5.2 / SciPy 1.18.1 要求 ≥3.12。请新建 venv：
  ```
  cd C:\Users\10757\trading\stargazer
  py -3.13 -m venv rule_venv
  rule_venv\Scripts\python -m pip install -r release_audit_v1\requirements-monitor.txt
  ```
- free-stockdb 引擎运行中（127.0.0.1:7899）+ 数据根（STOCKDB_DATA_ROOT）。
- data_daily_parts 镜像：放 `<项目根>\data_daily_parts`，或设环境变量 `STARGAZER_DAILY_PARTS`。

## 3. 放哪些文件/目录

git clone 后项目根应包含（均已随 tag 入库）：stargazer_desktop、stargazer_ml、stargazer_plan、
c3、corvus、release_audit_v1，以及模型与冻结资产（Tarot fold 模型、tarot_live 选股头、
Pyxis 232/266 四件套、板块成员表、分钟种子账本、分钟网格、概念板块表、Tarot 冻结 parquet、
隔离名单、相对键指纹清单三份）。
本机数据（不入库）：`stargazer_desktop\data\`（config.yaml、watchlist、学习记录）、
data_daily_parts、stockdb 数据。DPAPI 密钥（DeepSeek/推送）**监控机需重新填写**（设置页），
旧机密 blob 不能跨机解密。

## 4. 关键路径配置（stargazer_desktop\data\config.yaml）

程序已改为相对项目根定位 + 环境变量映射，监控机只需：

- `model.python` / `plan.rule_python`：填 `C:\Users\10757\trading\stargazer\rule_venv\Scripts\python.exe`
  （或统一设环境变量 `STARGAZER_PYTHON`）。
- `plan.ml_root` / `plan.plan_dir` / `model.batch_script` / `model.refresh_script`：留空即自动
  相对项目根解析；留空不满足时再填绝对路径。
- 指数分钟源：监控机无本机 TDX 目录，走公开源 `public_index_feed_v1`（程序自动）；
  如另有 TDX 导出可设 `STARGAZER_TDX_ROOT`。
- 每日 stockdb→日线分片同步：环境变量 `STARGAZER_DAILY_PARTS` 或默认 `<项目根>\data_daily_parts`。

## 5. 每日更新跑什么（真实入口）

- **不存在 `c3\_sync_daily_parts.py`**（开发机也没有）。每日盘后更新入口：
  `python -X utf8 stargazer_ml\v5_hybrid\green_v2\refresh_pool.py`
  （探测 stockdb 最新交易日 → 增量写 data_daily_parts → 重跑 build_pool 观察池；
  仅同步分片用 `refresh_pool.py --daily-only`；解释器缺省取仓库根 .venv，可用 `STARGAZER_ML_PYTHON` 覆盖）。
- 风险影子 worker 由桌面程序按需拉起（joint_shadow_worker_v1.py），无需手工跑。

## 6. 验收步骤（按顺序）

1. `rule_venv\Scripts\python release_audit_v1\verify_monitor_env.py` → 应输出 PASS（依赖版本 + 三份指纹清单 SHA 全过）。
2. 启动 stockdb（127.0.0.1:7899），`refresh_pool.py` 跑通到 `[done] pool_date=...`。
3. 放 EXE 到 `stargazer_desktop\dist\`，启动；设置页填 DPAPI 密钥。
4. 桌面「星币侍从演示/选股」能出名单；「影子建议」接口返回非 ERROR（指纹校验通过）。
5. 升级纪律：拉 tag → 覆盖代码 → 重跑第 1 步校验；config.yaml/data/密钥不覆盖。
