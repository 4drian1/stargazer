# Stargazer 盯盘 Agent（桌面版）

一个跑在你自己电脑上的 A 股交易助手：点开就能对话，盘中自动盯盘，风控信号推到微信。

## 功能

- **对话式分析**：直接说人话（"体检一下600207""扫一下存储芯片板块"），背后是 Kimi（kimi-k2.6）+ 完整交易模型手册，会自动调数据工具。
- **隔夜美股**：道指/纳指/标普/费城半导体/金龙中国/英伟达/台积电，问"昨晚美股怎么样"自动调数据+映射分析。
- **深度搜索（Tavily）**：投行/券商研报观点、公司公告求证（照妖镜）、海外深度消息。问"XX题材是真的吗""有没有研报覆盖XX"自动调用。
- **晨报/新闻解读**：直接说"给我一份晨报"，自动组合美股+快讯+持仓风控给分析。
- **确定性命令**（不耗 token）：`/check` 风控检查、`/flow` 板块资金流、`/news` 快讯、`/quote 代码` 个股体检、`/scan 板块` 四层漏斗、`/score 代码1,代码2` 垃圾度打分、`/watch` `/pos` `/del` 名单管理。
- **持仓/观察名单**：左侧栏登记，买入逻辑、止损线都在本地 `data/watchlist.json`。
- **自动盯盘**：交易时段每 30 分钟巡检一次（时段 9:25-11:35 / 12:55-15:10），15:05 收盘总结；触发止损/止盈/超卖信号才推送，同一信号 4 小时内不重复。
- **微信推送**：支持 Server酱 Turbo 和 PushPlus，软件里「推送设置」填 key 即可，可发测试推送验证。

## 快速开始（源码运行）

```bash
pip install -r requirements.txt
python app.py
```

- 装了 pywebview → 弹出独立桌面窗口；没装 → 自动用浏览器打开。
- API key 已内置在 `config.yaml`（kimi-k2.6）。

## 打包成 Windows 软件（exe）

在 Windows 上（需先装 Python 3.10+，安装时勾选 Add to PATH）：

1. 把整个 `stargazer_desktop` 文件夹拷到 Windows。
2. 双击 `build_windows.bat`。
3. 完成后得到 `dist\Stargazer盯盘Agent.exe`，双击即用；可以单独把 exe 拷到任何地方。

打包后 `config.yaml` 和 `data\` 会自动生成在 **exe 同目录**，升级 exe 不丢数据。

## 微信推送配置（二选一）

| 通道 | 获取方式 | 填到哪 |
|---|---|---|
| Server酱 Turbo | sct.ftqq.com 微信扫码登录 → 复制 SendKey | 推送设置 → Server酱 SendKey |
| PushPlus | pushplus.plus 微信扫码 → 复制 token | 推送设置 → 切到 PushPlus → 填 token |

配置后点「发测试推送」，微信收到即成功。之后电脑开着软件、处于交易时段，就会自动巡检推送。

## 文件说明

```
app.py            入口：启动后端+桌面窗口
server.py         Flask API（聊天/SSE推送/状态/配置）
notifier.py       定时巡检 + Server酱/PushPlus 推送 + 信号去重
web/              前端界面（聊天+侧栏+设置）
agent/ tools/ strategy/ services.py   策略内核（与CLI版一致）
STRATEGY.md       交易模型手册（AI 的行为准则）
config.yaml       API key + 推送配置
data/             watchlist.json（持仓/观察）、push.log（推送日志）
build_windows.bat Windows 一键打包脚本
```

## 注意

- `config.yaml` 里有你的 API key，**别整个文件夹传给别人或传到公开仓库**。
- 巡检依赖网络数据源（腾讯/东财公开行情），盘中偶发延迟属正常；推送日志在 `data/push.log`。
- 法定节假日软件不知道，休市日巡检会照常跑但数据不变，等于空转，不影响。
