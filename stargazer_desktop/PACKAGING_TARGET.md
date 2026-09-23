# 迁移打包方案（2026-09-09 定稿；用户确认：stockdb 与 stargazer 在目标机同级）

## 目标机目录结构（两者同级，都在 C:\Users\10757\trading 下）

```
C:\Users\10757\trading\
├── free-stockdb-windows-v0.3.2-more-power\   ← stockdb 服务（用户自备发行版，分钟数据源）
└── stargazer\                                 ← 我们的迁移包（解压到这里）
    ├── stargazer_desktop\                     ← 桌面 exe + data\（影子库）+ config.yaml
    ├── stargazer_ml\                          ← 影子脚本 + 特征表 parquet + shadow_models\
    ├── data_daily_parts\                      ← 加工日线分区（602M，从 D:\a股数据 拷贝）
    └── (stockdb 启动/刷新辅助脚本，如有)
```

## 打包清单（四类 + 数据）

1. `stargazer_desktop/`（不含 __pycache__；data\ 整体带上，model_state.sqlite 是影子留痕）
2. `stargazer_ml/pullback_train/` 的影子脚本 + 特征表 parquet（约 1.2G）+ `shadow_models/`
3. `D:\a股数据\data_daily_parts\` → 包内 `stargazer\data_daily_parts\`（602M）
4. Python 环境说明/requirements（目标机装 .venv）
   桌面 exe 由用户自己打包；stockdb 服务用 trading 下的发行版，**不属于本包**。

## 路径替换（打包时预改，或首启前一次性替换）

| 本机 | 目标机 |
|---|---|
| `D:\stargazer` | `C:\Users\10757\trading\stargazer` |
| `D:\a股数据`（STOCKDB_DATA_ROOT，代码找 `data_daily_parts` 子目录） | `C:\Users\10757\trading\stargazer`（分区数据放在包内根，少一处外部依赖） |
| stockdb 服务目录 | `C:\Users\10757\trading\free-stockdb-windows-v0.3.2-more-power` |

新增登记（冻结计划监控，2026-09-17 规则替换后）：config.yaml `plan:` 段内的
`plan_dir`（stargazer_plan）、`rule_python`（`_dev_venv`）含本机根路径，
打包时与上表同规则机械替换（`D:\stargazer` → 目标机根）。`plan_file`
（`data/plan_frozen.json`）为相对路径无需替换；监控机/portable 环境无需
`plan_dir`/`rule_python`（「盘后总结」按钮点后会提示连接失败，属预期，不崩）。
盘中规则监控在桌面进程 intraday_monitor 线程内完成（规则源 plan_file），
**不启动任何外部监控进程**；旧的 rule_monitor/rule_venv/start_monitor.bat
外置方案已随 20260917 规则包废弃。

注意：pullback_train 所有 .py 的 `OUT_DIR` 硬编码 `D:/stargazer/...`，打包脚本需对
包内 .py 与 config.yaml 做机械字符串替换（两个根路径），替换后抽查 shadow_batch.py
能 `--validate2026` 通过再放行。

## 首启验证顺序（避免搞错）

1. 先启动 stockdb 服务（发行版目录），确认行情接口可用；
2. 确认 `stargazer\data_daily_parts\part_*.parquet` 可达；
3. 跑 `shadow_batch.py --validate2026`——PASS 说明模型+特征+路径全通；
4. 启动桌面 exe，点"生成建议"，检查 last_shadow.log 无报错、
   `/t5` 能看到当日决策记录。

## 数据回流（监控机 → 本机分析/调优）

本机（D:\stargazer）为 master。监控机的 model_state.sqlite 拷回后按主键
（信号日×代码×决策点）幂等合并，不直接覆盖；minute_snapshots 可不带回。
合并+回测/调优在用户喊开始时执行。
