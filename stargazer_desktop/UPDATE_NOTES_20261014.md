# Stargazer 更新说明（2026-10-14）

本版核心：**AI 助手落地 + LangGraph 影子图接入三模型接口 + 星币侍从学习记录闭环**。
仅预留接口与编排，不融合、不互询、无交易权限；风险标签/训练/667 码扩采仍未批准。

## 一、新增功能

### 1. AI 助手（DeepSeek）
- 对话、次日计划讨论、新会话、恢复最近对话（独立会话 JSONL，重启可恢复）。
- 上下文重构：候选改列名+逐行数组，修复 206 候选嵌套快照撞本地 60000 字符门的问题；
  超大工具结果完整归档、返回证据 ID，不发送损坏截断 JSON。
- 密钥用 **Windows 当前用户 DPAPI** 保存（`.secrets/deepseek.dpapi`），不入源码/报告/Git/exe。
- 护栏：每日 40 次 / 120000 预留 Token，失败请求仍记额度，不自动重试；
  本地助手记录 10MiB、学习文档 2MiB 不清旧证据。

### 2. LangGraph 影子图（三模型接口）
- `assistant_runtime.py` 以独立 Python 子进程运行 `assistant_graph_worker.py`：
  图依次登记**不可变草案引用 → 三模型状态 → 人工确认**，SQLite checkpoint 支持中断重启恢复。
- `model_slots.py` 复用 `SnapshotV2` 预留三模型槽位：选股 = DISPLAY_ONLY/DEGRADED 规则观察；
  **信息与风险 = UNKNOWN 空预测**（真实接口待后续接入，读取时刻不伪造历史到达时间）。
- 图节点不调用 API、不下单、不发通知；环境缺失/超时只暂停助手确认，报价与风控线程继续，
  绝不自动跳过人工门。报价 exe 不导入 LangGraph（依赖见 `requirements-shadow.txt`）。

### 3. 星币侍从学习记录
- 「写入学习记录」选已完成观察日，预览后确认追加，界面显示实际文档路径。
- 七日复盘带**证据门**：无认证模拟入场证据则可执行 PnL 保留 NULL；
  实际盈亏独立读真实成交账本（有股数且合法匹配才报毛盈亏，无费用不报净盈亏，不归因模型）。

## 二、稳定外壳调整

- 盘前手填与旧交易计划、绿盘/横盘/T5 选股入口撤下；旧计划新增/冻结/结算 HTTP 入口返回 **410**；
  已冻结计划的独立风控不关闭。
- 保留：持仓结构止损、观察目标价提醒、手工真实成交、推送设置。
- 新 `Stargazer_demo_safe.spec`（安全打包用）。

## 三、安全与脱敏

- 已扫描确认：工作区与 git 历史中**无任何硬编码密钥**（DeepSeek/PushPlus/ServerChan/Tavily/tushare）。
- `.secrets/` 与 `data/assistant/` 已入 `.gitignore`；`config.example.yaml` 仅含占位符。
- DPAPI 密钥绑定当前 Windows 用户：迁移账号/电脑须重新填写；曾在聊天暴露的密钥建议轮换。

## 四、验证

- 零网络回归：`test_stable_shell + test_assistant_workflow + test_pentacles_demo +
  test_pentacles_monitor + test_pentacles_journal` = **74/74 PASS**。
- DeepSeek 固定 JSON 连通成功（deepseek-flash，44 Token，未上传交易/持仓/行情）。
- LangGraph 1.2.15 + checkpoint-sqlite：中断恢复、过期确认拒绝、与轻量判定一致性 PASS（独立原型验证）。

## 五、使用与边界

- 使用入口：`dist/Stargazer.exe`（需在监控机另行安装/验收独立图运行环境，仅复制 exe 不宣称图就绪）。
- 运行顺序：更新 stockdb → 盘后总结生成名单 → AI 对话制定计划 → 确认后启用到价提醒 →
  观察日完成后写入学习记录。
- 边界：不接入信息/风险真实模型、不提前搭 JEV、不取风险外部数据、不训练；
  checkpoint 不是交易依赖验证，七日净收益仍需真实执行/尺度/成本证据。
