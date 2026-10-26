# 发布工程核查（路线 #18.46，2026-10-26）

## A2. 绝对路径清单（运行源码，逐条处置）

| 文件 | 位置 | 现状 | 处置 |
|---|---|---|---|
| stargazer_desktop/info_panel.py:26,31 | `_sibling()` 兜底 `D:/stargazer` | 可改相对 | 改为向上扫描含目标子目录的项目根；取消 D 盘兜底 |
| stargazer_desktop/Stargazer_demo_safe.spec:36 | `runtime_tmpdir='D:/stargazer/.../runtime_tmp'` | 可改相对 | 删除参数，用系统临时目录（EXE 不再固定 D 盘解包） |
| stargazer_desktop/config.example.yaml:31-34,57,60-61 | `D:\stargazer\...` 五项 | 应改为数据目录映射/占位 | 示例改为 `<项目根>` 占位 + 注释；监控机按部署文档填 |
| stargazer_desktop/assistant_runtime.py:19 | `project/_dev_venv/Scripts/python.exe` | 已相对，venv 名固定 | 加 `STARGAZER_PYTHON` 环境变量覆盖 |
| stargazer_desktop/joint_risk_bridge.py:32 | 同上 | 同上 | 同上 |
| stargazer_desktop/services.py:208 | 同上（单股按需推理 worker） | 同上 | 同上 |
| stargazer_desktop/plan_bridge.py:41 | `rule_python` 缺省 `_dev_venv` | 可由 config 覆盖 | 文档化；不动代码 |
| stargazer_desktop/pentacles_demo.py:40-43,81 | data_daily_parts 探测（root→D:/a股数据）+ `_dev_venv` | 已是数据目录映射 | 文档化 `STARGAZER_DAILY_PARTS`；python 走 config/env |
| stargazer_desktop/index_minutes_v1.py:16 | `TDX_ROOT` 本机微信目录（D 盘） | 应改为数据目录映射 | 加 `STARGAZER_TDX_ROOT` 覆盖；监控机走公开分钟源 |
| stargazer_desktop/ml_bridge.py:34 | `STOCKDB_DATA_ROOT` 缺省 D:/a股数据 | 已是 env 映射 | 不动 |
| stargazer_ml/v5_hybrid/green_v2/refresh_pool.py:~160 | 缺省 `D:\stargazer\.venv\Scripts\python.exe` | 可改相对 | 缺省改仓库根相对 `.venv`，保留 `STARGAZER_ML_PYTHON` 覆盖 |
| stargazer_ml/.../risk_r0_v1/joint_shadow_service_v1.py | verify_fingerprints 以清单键为绝对路径直读 | 应改为数据目录映射 | 清单键改项目相对路径；键相对时按 `STARGAZER_PROJECT_ROOT`/文件定位解析 |
| stargazer_ml/.../green_common_profile/tarot_live_v2.py:25,30 | MANIFEST frozen_inputs 绝对路径 | 同上 | 新清单用相对键；verified_manifest 按项目根解析 |
| stargazer_desktop/on_demand_model_worker.py:18 | 引用绝对键清单 v2 | 同上 | 指向新相对清单 |

指纹清单（绝对路径，需重算 SHA 并换相对键）：
- `live_monitor_fingerprints_20261023_v2.json`（151 项，live 监控清单）
- `joint_shadow_fingerprints_dev_rec4.json`（worker 期望指纹）
- `tarot_live_manifest_v17_20261026_flow.json.gz`（154 项 frozen_inputs，Tarot 冻结输入）

说明：全仓另约 150 个研究脚本含 `D:\stargazer`（pullback_train、gcp_247 系列、批次核验等），
不在运行调用链上，本次只标注不解耦（见 A1）。

## A3. Git 缺口核实

- 模型跟踪状态：`git ls-files | grep -iE "model|pkl|lightgbm"` 只命中代码/文档，
  **三个模型目录（fold 模型、tarot model.pkl、Pyxis 232/266）均未入库**。
  此前"fold 模型已在库"的说法不成立，本次按需精确补提。
- `stargazer_desktop/tools/market.py`（整个 tools 包）被全局 `*` 默认忽略，
  且 spec hiddenimports 依赖 tools.* 与 strategy.* → 需白名单。
- 指纹清单：`risk_r0_v1/*.json` 已白名单（v2 可提）；`tarot_live_manifest_v17...json.gz`
  无白名单（gz 未放行）→ 需补。
- 未提交的运行源码（git status 摘要）：server.py/app.py 等 30 个修改文件 +
  decision_layer.py、shadow_ui.py、oracles_v1.py、jev_adapter_v1.py、
  postclose_risk_runtime.py、index_minutes_v1.py、public_index_feed_v1.py、
  joint_risk_bridge.py、on_demand_model_worker.py、tarot_live_v2.py、
  joint_shadow_service_v1.py、rebuild_features_v*.py、golden_ladder_v3.py 等未跟踪文件。
- 全局兜底 `**/*.parquet / **/*.pkl` 与"小资产入库"冲突 → 按文件精确放行（见 B2 清单）。
