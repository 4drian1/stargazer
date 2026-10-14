# 桌面助手更新验收（2026-10-14）

## 交付

- 活动入口：`D:\stargazer\stargazer_desktop\dist\Stargazer.exe`
- 大小：130,757,386 B；修改时间：2026-10-14 20:07:33。
- SHA256：`56ba67399e8ecc4f33cd79a473dd9202c66a87a939c98e2eb175a21552292f5d`
- 同名旧副本5份及升级前原exe1份回退包：`D:\stargazer\_archive\retired_executables_20261014\*.exe.disabled`。恢复时先关闭程序，按说明复制回主入口；不覆盖data/config。
- 密钥用Windows当前用户DPAPI保存，在聊天里暴露过，建议轮换；不入源码、报告、Git、exe。迁移账号/电脑须重新填写。

## 验收证据

- `python -B -X utf8 -m unittest test_assistant_workflow test_pentacles_demo test_pentacles_monitor test_pentacles_journal`：52/52 PASS，包含19项初始助手测试、4项追加测试及29项原提醒/日记回归。
- JS语法、Python AST、diff空白检查通过。
- DeepSeek固定JSON连通：成功，返回模型deepseek-flash，44总Token；没有上传交易/持仓/行情，未做失败自动重试。
- LangGraph 1.2.15 / checkpoint-sqlite 3.1.2：独立原型SQLite中断后重启恢复、过期确认拒绝、与轻量判定一致性PASS。生产exe无LangGraph/LangChain/LangSmith模块；没有图编排生产部署。
- 独立UI预览截图：`.work/assistant-learning-ui.png` / `.work/assistant-plan-ui.png`；后台使用假响应，无监控线程/微信/付费API。服务已停止。
- 包内7个核心模块与源码字节码一致；HTML/JS/CSS四资源逐字节一致；无私有config.yaml与.secrets资源。
- 升级前后dist非exe63文件哈希相同；未改持仓/观察名单/成交库/日记/原推送配置。额外凭据在此比较基线前按用户授权建立。

## 真实输入与限制

- 当前本机同步分片止于20261013。10月13日归档130候选的同日日K全部在场；HTTP失效时可以只读这些分片，不以昨日价格补今日。
- 10月14日必要日K未同步；该日复盘拒绝生成，先启动/更新stockdb并同步日线。
- 现有130事件没有认证模拟入场证据，七日可执行PnL保留NULL；只实现固定退出计算及证据门，不把报价触区当作买入、不自证旧tracker标志。
- 日线触区及相对名单收盘的变动为描述；价格尺度未核销不能推成策略收益。
- 实际盈亏独立读取真实成交账本：有股数且合法匹配可报毛盈亏；没有费用就不报净盈亏；不归因模型。
- 新功能人工确认/真实数据下的后续桌面体验与交易时段微信送达由用户实机验收。本轮未启动正式exe，包内核对与假数据预览不是完整生产端到端验收。
- 风险标签、训练、JEV与监控机部署门均未解除。

## 用户运行顺序

1. 打开唯一活动exe，必要时在设置更换DeepSeek密钥。
2. 更新stockdb后点击「盘后总结 · 分析选股」，确保生成未来观察日名单。
3. 「制定次日计划」在原AI对话中讨论，确认最新草案才启用到价提醒。
4. 「写入学习记录」选已完成观察日，生成预览后确认追加。界面显示实际文档路径。
