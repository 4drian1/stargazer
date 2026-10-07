# C3 信息模型 · 监控机运行时包（2026-10-06）

目的：在监控机上运行信息模型每日采集，观察公告信息质量（路线 #12/#13）。

## 包内容

- C3 源码（daily_ingest.py / formal_score.py / pg_store.py / _taxonomy.py 等）
- 冻结模型 `corvus/materiality_v1/artifacts/`（v1.0.0 + manifest + parity 凭证）
- `out/calendar.json`、`out/universe.txt`
- 不含：历史回填数据（258/304MB parquet，监控机日常运行不需要——daily_ingest 只保留目标日，幂等重写；如需历史另行回填）、daily_cache、venv

## 部署（一次性）

1. 解压到监控机仓库根，如 `C:\Users\10757\trading\stargazer\c3`（与 stargazer_desktop / stargazer_plan 同级——桌面端按"exe 同级/上级"自动发现）
2. 环境：`prepare_env.bat`（用已有 .runtime venv 或新建）
3. stockdb 启动（口诀：数据更新.exe 跑完必须重启引擎）

## 每日操作（看信息质量）

```
python -X utf8 daily_ingest.py --date <YYYYMMDD> --pool
```

- 盘后跑：拉池内票当日公告 → 冻结 taxonomy 路由 → 证据行写 `out/daily_events/events_<date>.jsonl` + `evidence_<date>.json`
- 桌面 app「信息面板」页自动展示：公告事实（类目/tier/关键词方向/链接）+ 板块/外盘/快讯
- **p_materiality 在监控机显示"待 T+1 收盘后评分"**——正式打分写 PG（corvus_c4 双表），PG 只在本机；两机事件 ID 确定生成（sha(code|date)），本机 formal_score 的结果与监控机采集天然对齐，后续如需在监控机看分数，把本机 `out/daily_events/events_<date>.jsonl` 随每日文件拷贝放过去即可（反向也行）
- 北交所（4/8/92 开头）无分钟/部分日线能力属源限制，相关票在名单里会 unscored，正常

## 红线

- 不改模型/特征/阈值；daily_ingest 只出事实不出概率（路线 A）
- 正式评分链（formal_score/attach_realization）仍在**本机**跑（需要 PG）
- 采集失败/源缺如实看 `out/daily_events/errors_<date>.json`，不静默
