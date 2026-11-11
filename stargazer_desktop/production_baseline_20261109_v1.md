# 生产基线对应关系（2026-11-09，本轮稳定性修复起点）

## 核对起点
- 生产 EXE：`stargazer_desktop/dist/Stargazer.exe`
  - SHA256：`54825f573009163a64c6569dd9fcf79ecdbeb89d0151c09298b3667ee01b4d36`（= 54825F57）
  - 来源：2026-11-09 14:38 恢复旧界面资源后的版本（备份见 `server.py.bak_20261109_143836_pre_recovery` / `app.py.bak_20261109_143836_pre_recovery`）。
- `release_audit_v1/verify_live_release.py` 结果：`ok=true`，零失配（已核验）。

## 已发布清单（不得原地修改；本轮修复定稿后生成后继清单）
- 运行时指纹清单：`stargazer_ml/v5_hybrid/risk_r0_v1/live_monitor_fingerprints_20261109_v178_flow.json.gz`（335 项，frozen_inputs: 相对路径→SHA256）。
- Tarot 清单：`stargazer_ml/v5_hybrid/green_common_profile/tarot_live_manifest_v166_20261109_rel.json.gz`（212 项）。
- 7 个运行时引用文件均已收敛指向 v178（无分歧）：
  - `stargazer_desktop/golden_stock_analysis.py`
  - `stargazer_desktop/on_demand_model_worker.py`
  - `stargazer_desktop/sector_rule_scan.py`
  - `stargazer_ml/v5_hybrid/green_common_profile/market_price_evidence_v1.py`
  - `stargazer_ml/v5_hybrid/risk_r0_v1/tarot_intraday_risk_v2.py`
  - `stargazer_ml/v5_hybrid/risk_r0_v1/live_postclose_job_v1.py`
  - `stargazer_ml/v5_hybrid/risk_r0_v1/accept_live_postclose_v1.py`

## 构建与打包（用于本轮收尾）
- 构建入口：`stargazer_desktop/build_windows.bat` → `PyInstaller Stargazer_demo_safe.spec`（产物名 `Stargazer.exe`，onefile，console=False）。
- 打包前预检：`release_audit_v1/verify_live_release.py`（校验清单 + 全部 frozen_inputs SHA + 7 引用收敛 + 8 个 REQUIRED_WORKERS 在 spec datas 白名单）。
- 候选验证：`release_audit_v1/verify_frozen_package_v1.py dist_candidate/Stargazer.exe`。
- 后继清单生成模板：`release_audit_v1/publish_pentacles_batch_fix_v1.py`（OLD_FLAT/NEW_FLAT + 7 处引用字符串替换 + 重算 frozen_inputs SHA）。

## 生产修复 vs 新 UI 改动（隔离）
- 当前 `web/` = 旧界面资源（54825F57 已恢复）。本轮生产修复只改监控/推送/去重/Pyxis 相关共享源码，**不触碰新 UI（ui_next / 17891）文件**。
- 新 UI 改动单独保存，不得借本次修复覆盖；不得修改已发布清单（v178/v166）来掩盖版本差异。改动收尾时由 publish 脚本生成 v179（及对应 tarot 后继）作为新清单。

## 本轮已改动 / 将改动文件（收尾时纳入 v179）
- 新增：`stargazer_desktop/instance_guard_v1.py`（单实例锁）
- 已改：`stargazer_desktop/server.py`（start 先取单实例锁）、`stargazer_desktop/notifier.py`（去重/回执持久化）
- 后续步骤将改：`pentacles_monitor.py` / `touch_alert_lane_v1.py`（触价两段消息、指标补充）、`golden_monitor.py` / `risk_outlook_v1.py` / `push_compact_v1.py`（巡检指标展示）、`stargazer_ml/.../intraday_public_context_v1.py`（盘中 Pyxis 前缀恢复）。
