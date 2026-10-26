# A1. 运行必需资产清单（沿当前调用链反推，路线 #18.46）

调用链：server.py / pentacles / decision_layer → Tarot 选股（tarot_live_v2）
→ stargazer_ml/v5_hybrid/green_v2/refresh_pool.py（盘后日线+观察池）
→ Pyxis 推理（rebuild_features 链 + sidecar_rebuild_v4，joint_shadow_worker_v1）
→ shadow_ui（jev_adapter_v1 + oracles_v1）。

判定依据：三份冻结指纹清单（换相对键后全量保留 SHA 校验）+ 源码 import/文件引用。
"required"= 监控机上缺失则对应链路拒绝运行或降级为缺口；"研究专用"= 仅研究/验收脚本使用。

## 1. 模型（required，本次发布对象）

| 资产 | 路径 | 用途 | 处置 |
|---|---|---|---|
| Tarot fold F3_B2 | stargazer_ml/v5_hybrid/green_common_profile/round_a_v0_1_impl_fix/fold_model_F3_B2.pkl | fold 选股模型（决策指定发布） | 入库+资产包 |
| Tarot fold FINAL_B2 | 同目录 fold_model_FINAL_B2.pkl | joint_shadow_service PKL_PATH 选股评分 | 入库+资产包 |
| Tarot live 选股头 | .../tarot_a_second_leg_fit_20261022_v1/initial_node_cash_fit_20261023_v1/model.pkl | tarot_live_v2.MODEL（TAROT_A_SECOND_CASH_V2，实际启用） | 入库+资产包 |
| Pyxis 232主版 | stargazer_ml/v5_hybrid/risk_r0_v1/auxiliary_120m_fixed_fit_20261018_v1/（lightgbm_model.txt.gz + lightgbm_preprocessing.json + manifest.json + plan.json） | 风险推理主模型 | 入库+资产包 |
| Pyxis 266对照版 | stargazer_ml/v5_hybrid/risk_r0_v1/common_candidate_fixed_fit_20261019_v1/（同上 4 文件） | 风险推理对照 | 入库+资产包 |
| Tarot 冻结域代码 | .../tarot_a_second_leg_fit_20261022_v1/run.py | tarot_live_v2.frozen() importlib 加载 | 入库 |

说明：预测/评估类伴生文件（predictions.parquet、report.json、ridge_*、stop_record.json）
不在任何运行 import 链上 → 研究专用，不进资产包、不入库（留在开发机）。

## 2. 必需小资产（required，数据类，入库 + 资产包）

| 资产 | 路径 | 用途 |
|---|---|---|
| 板块成员表 | stargazer_ml/v5_hybrid/opportunity_v2/o5_sector_lag/_upstream/stockdb_boards_snapshot.parquet (208KB) | 冻结指纹输入；板块上下文 |
| 板块成员表(PIT) | stargazer_ml/v5_hybrid/opportunity_v2/o5_sector_lag/pit_membership.parquet (208KB) | info_panel PIT 引用 |
| 分钟种子账本 | stargazer_ml/v5_hybrid/risk_r0_v1/relative_stall_5m_20261018_v1/raw_input_ledger.json (1.4MB) | 冻结指纹输入；相对停滞特征 |
| 分钟网格 | stargazer_ml/v5_hybrid/green_common_profile/gcp_round_b_small_sample_v1/minute_grid.json | 冻结指纹输入；触发重放 |
| 概念板块表 | stargazer_ml/v5_hybrid/risk_r0_v1/ths_full_concept_scope_20261017_v1.json (290KB) | 冻结指纹输入 |
| Tarot 冻结输入 | round_a_v0_1_impl_fix/{full_master_table_v2,full_intraday_replay_v2,rolling_lr_v2_predictions}.parquet (共3.1MB) + full_master_table_v2.py | tarot_live_v2 verified_manifest frozen_inputs |
| 隔离名单 | stargazer_ml/pullback_train/quarantine_2026.csv (527KB) | 冻结指纹输入 |
| 指纹清单（新，相对键） | risk_r0_v1/live_monitor_fingerprints_20261026_v27_rel.json；risk_r0_v1/joint_shadow_fingerprints_rel_v1.json；green_common_profile/tarot_live_manifest_v18_20261026_rel.json.gz | SHA 校验基准（SHA 校验全量保留） |
| 日历 | stargazer_ml/v5_hybrid/step9_preflight/trading_calendar.py | 冻结指纹输入（代码） |

## 3. 本地数据层（监控机自备，不入库不入包）

- free-stockdb 引擎（127.0.0.1:7899）及其数据根（STOCKDB_DATA_ROOT）
- data_daily_parts 镜像（STARGAZER_DAILY_PARTS / 项目根 data_daily_parts）
- stargazer_desktop/data（config.yaml、watchlist、学习记录）——DPAPI 密钥监控机重填
- index_minutes_v1 镜像目录（运行期生成）；TDX 分钟源仅开发机用（监控机走 public_index_feed_v1）

## 4. 研究专用（只标注，本次不解耦）

- 全部 gcp_247_* / 批次核验 / freeze_comparison / price_verification_* 等研究脚本
  （risk_r0_v1 下约 70 个 .py 在冻结指纹清单内：监控机上由 git 提供，缺失即校验拒跑——
  因此这些代码随仓库发布，属"代码资产"而非数据资产）
- pullback_train 训练/消融全家桶；模型伴生预测 parquet；accept_*.py 验收脚本
- tarot fit 目录下 diagnose/adapter 脚本（在 v17 清单内 → 随仓库发布）
- minute_grid/raw_input_ledger 的衍生研究目录（sidecar_rebuild_v1..v3/v5 等）

## 5. 风险标注

- 只有 stockdb + data_daily_parts 不足以跑通风险推理：还缺第 1、2 节资产与仓库代码
  （缺一不可，SHA 校验会硬拒绝）。
- 监控机每日更新入口不是 c3\_sync_daily_parts.py（开发机不存在该文件）：
  实际为 `stargazer_ml/v5_hybrid/green_v2/refresh_pool.py`（见部署文档）。
