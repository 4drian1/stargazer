# Stargazer 新 UI 接入发布报告 (v2)

- 日期：2026-11-09
- 状态：**四处接入缺陷已修复并通过专用验收入口 15/15；候选 EXE 已按定稿源码重建；生产 EXE 替换按红线#9 暂缓**
- 本轮未安装任何内容；生产 `dist/Stargazer.exe` 与保护文件保持不变。

## 0. 勘误（撤回）
- 撤回 v1 报告中「`config.yaml` 与 `de488f9c…` 不同，说明曾被其他操作改动」的推断。
  `de488f9c…` 与 `6595acce…` 分别来自**开发配置**与**生产配置**，此前已分别核实，不能跨路径比较。
  正确表述：本轮未触碰生产 `config.yaml`（本轮前后 SHA 一致：`6595acce…`）。

## 1. 本轮修复的四处接入缺陷
| 缺陷 | 修复 |
|---|---|
| 日历合同回退 | `/api/calendar/next_trading_day` 改为接受 **`?from=`**（前端 `client.ts` 实际参数），兼容 `?date=`，缺省取今天；响应含 `from`/`date`/`next_trading_day`/`source`，非法日期 400 |
| 路径推导错误 | `ui_backend_config.py` 改用 `paths.APP_DIR`（dev 与冻结 EXE 均正确；不再指向 `_MEI` 临时解包目录）；快照默认路径改到安装目录 `data/isolated/` |
| 成交状态读错契约 | `_enrich_execution_status` 改为 `cfg['agent']['trades_db']`（dict 访问），并按 `actual_trades()` **汇总 dict** 契约处理：`MATCHED_GROSS_ONLY`/`MATCH_UNKNOWN` 且 `matched_sales>0` → `REGISTERED`，`LEDGER_UNAVAILABLE` → `LEDGER_UNAVAILABLE`，否则 `UNKNOWN`（不伪造已执行） |
| 确认状态未闭环 | `assistant_workflow.state()` 现按持久化 `confirmed_<id>.json` 返回 `plan.confirmed`/`confirmed_status`（刷新/重启后仍可识别）；前端 `Plans.tsx` 消费 `confirmed`（已确认=绿色徽标、按钮禁用）并已重新构建 |

另：`server.py` 曾在会话中被环境还原（`app.py` 同样），已重新应用并经编译 + 验收复核（见 §4 观察记录）。

## 2. 前端重建
- `ui_next/frontend`（React/TS）修改 `Plans.tsx` + `types.ts` 后 `npm run build` 成功。
- 新资产已复制到 `web/`：`web/index.html` + `web/assets/index-PJ_Huq9D.css` (26.5KB) + **`web/assets/index-CIOVEJrB.js` (242KB)**；旧 `index-Z0TqdGIw.js` 已删除。

## 3. 显式 EXE 验收入口（新增 `acceptance_ui_integration_v1.py`，入口 `app.py --acceptance` / `--acceptance-restart`）
- 隔离在 `import server` 之前完成：可写数据（assistant 草案/确认、`data/trade_plan.json`、`dist/data/trade_plan.json`、`data/pentacles_demo/assistant_plan.json`、`data/isolated`）全部重定向到 **验收沙箱**；只读资源（`calendar.json`/`pentacles_demo/snapshot.json`/journal）保持安装目录原路径。
- 不调用 `server.start()`；用 Flask test_client；注册正式业务路由及 I34 只读路由；不启动监控/行情同步/采集/风险 worker。
- 边界拦截：`market.quotes`（网络）、`deepseek_api.chat`（付费）、`minute_history`（后台）。
- 普通启动模式不受影响；未修改任何生产配置。

## 4. 验收结果（dev 隔离环境，`_acceptance_result.json`，15/15 PASS，ok=true）
- 新 UI/API 加载：`/`、`/api/calendar/next_trading_day?from=20261109`（→20261110）、缺省今天、非法日期 400、`/api/plan/versions`、`/api/market/global_overview`（`decision_use=false`）、`/api/trade_plan`、`/api/assistant/state` 全部通过。
- 确认状态闭环：真实 `GOLDEN_ANALYSIS` 草案（合法 `entry_rule`/`rule_inputs` 绑定真实文件 SHA）→ **真实路由** `/api/assistant/plan/confirm` → `conversation_plans.confirm` 真实落盘 → `state()` 读回 `plan.confirmed=true`。
- 过期拒绝：`plan_expired({'monitor_date':20200101}) → True`。
- 复盘保存成功 → 读回（`facts_hash` 一致）；错误观察日 `review_start` 在后台线程被拒（`JOB['error']` 置位，未触碰 collect/DeepSeek）。
- 真实成交隔离样例：沙箱 `trades.db` 含真实 buy/sell（目标日成交）→ `manual_trade_registered='REGISTERED'`。
- **新进程重启保留**：独立子进程 `--acceptance-restart` 读回确认状态与复盘草案均保留。

**冻结 EXE 验收未完成（待授权）**：候选 EXE（console=False GUI 程序）启动需审批，本轮两次尝试均未获批，未再重试。
已准备好 staging 目录 `stargazer_desktop/acc_install/`（候选 EXE + 生产 `dist/data` 只读副本，生产未动）。授权后执行：
`Start-Process -FilePath acc_install\Stargazer.exe -ArgumentList '--acceptance','<sandbox>','<result>' -WorkingDirectory acc_install -Wait -PassThru`
（说明：`dist_candidate` 不含运行时 `data/`，直接在 `dist_candidate` 下跑会因缺日历/快照文件出现 500——必须用带运行时数据的安装目录。）

## 5. 会话内文件还原观察（重要）
- 本会话中 `app.py` 与 `server.py` 曾被还原为不含本轮改动的旧版本（`app.py` 的 `--acceptance` 入口块、`server.py` 的 3 个新路由 + `_enrich_execution_status` 消失）。
  已重新应用并经编译 + 15/15 验收复核确认恢复。**后续构建前请先复核这三个文件的改动仍在。**
- 复核命令：`Select-String -Path server.py -Pattern 'api_calendar_next|_enrich_execution_status'`、`Select-String -Path app.py -Pattern 'acceptance-restart'`。

## 6. 候选 EXE（已按定稿源码重建）
- 路径：`dist_candidate/Stargazer.exe`
- sha256：`d4883a77bb5479ef37e31495299e078b8dd0c9fd7f786a6248f66ed948280729`
- 大小：143,632,187 B (~143.6 MB)
- 旧候选 `d6c1f045…` 已被本次构建覆盖（其源码不含本轮修复，视为过期）。

## 7. 后继双清单（v178 / v166，已含定稿源码）
- `live_monitor_fingerprints_20261109_v178_flow.json.gz`（335 项）/ `tarot_live_manifest_v166_20261109_rel.json.gz`（212 项）。
- 本轮将定稿的 `app.py` / `assistant_workflow.py` / `server.py` 三个文件哈希更新进 v178/v166（同名清单，未新增版本、未改 8 处引用）。
- `verify_live_release.py`：两份清单 `mismatch=[]`，`errors=[]`（**门绿**）。旧 v176/v164 及更早清单原样保留。

## 8. 生产保护文件（本轮未修改）
- 生产 EXE `dist/Stargazer.exe`：`b40fb3c0…`（未变）。
- `config.yaml`：`6595acce…`；`watchlist`：`9c374e17…`；`trade_plan`：`9539b1b4…`（本轮前后一致）。

## 9. 剩余步骤（按序，待授权）
1. 在 `acc_install/` 下运行冻结 EXE 验收入口（见 §4 命令），确认 15/15 在冻结环境同样通过。
2. 验收通过后统一安装：备份 `dist/Stargazer.exe` → `exe_backups/Stargazer.pre_ui_integration_20261109_<sha>.exe`，再以候选 `d4883a77…` 覆盖；安装后复算保护文件 SHA 应不变。
3. 若本轮无法完成冻结验收，须按约定**恢复匹配的旧源码/资源/引用与旧 EXE**，不能只保留旧 EXE 而让共享源码停在新版本。
- 已知坏档不得用于回滚：`dist/Stargazer_bad_incomplete_20261105_2045.exe`、`dist/Stargazer.exe.bad_no_worker_20261106`。

## 10. 红线遵守
- 未改任何模型/权重/特征/选择/金标/交易阈值；报价调度 / batch_size=30 / 2000 额度闸 / 超时 / 校验 / 去重 / 发送前重核未改；未重跑 4200 只 / 清缓存 / 删旧证据。
- 生产 config/positions/fills/confirmed plans/risk evidence 未改；无真实行情采集 / 付费 / 推送用于验收（全部在边界拦截或沙箱）。
- 未整份照搬 `host_app.py` 作为生产 `server.py`（只复用其合同：`?from=`、缺省今天、响应字段）。
- 未绕过指纹校验、未覆盖旧清单（v178/v166 为本轮定稿更新，旧清单保留）。
