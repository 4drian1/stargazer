# Corvus（乌鸦座）—— 信息风险模型 第一批

Corvus 是股票量化交易系统的**信息风险模型**代号。命名体系：

| 代号 | 中文 | 定位 | 本期状态 |
|---|---|---|---|
| **Corvus** | 乌鸦座 | 信息风险模型 | 本期开工（第一批） |
| Pyxis | 罗盘座 | 风险模型（市场环境/盘中/组合） | 不开工 |
| Astrolabe | 星盘 | 最终综合风险服务 | 不开工 |

Corvus 内部三个模块保持功能名：

- **News Event Engine**（新闻事件引擎）：新闻/公告/政策采集 + Qwen 事实抽取
- **Global Context Model**（外围行情传导）：美港日韩/汇率/利率/商品 20 序列及其时间可见性
- **Information Risk Fusion**（信息风险融合）：后融合，**本期只留接口占位**

## 完整命名体系（冻结）

```text
Corvus：信息模型（本项目）
Pyxis：风险模型（未开工）
Astrolabe：综合风险信息服务（未开工）
Tarot：选股服务
  ├─ The Hermit：调整到位/阶段谷底
  ├─ Wheel of Fortune：盘中下杀反转
  ├─ The Hanged Man：横盘埋伏
  └─ The Chariot：打板追龙
Stargazer：总Agent与人机交互入口
```

**规则：名字只代表模块，数据库和接口仍使用稳定英文 ID**（如 instrument_id=SPX、source_id=CNINFO，不出现星座/塔罗名）。

依据规格：《信息模型开发方案 v1.1》（外部文档，契约冻结版）。

## 目录结构

```text
corvus/
├── config/
│   ├── sources.yaml        # 新闻来源注册表（首批官方源 + 聚合源）
│   └── instruments.yaml    # 外围工具注册表（20 序列）
├── corvus/
│   ├── db.py               # SQLite 持久层（字段按 PostgreSQL 设计，WAL，含旧库列迁移）
│   ├── timeutil.py         # 统一时间解析 parse_source_time（tz_inference/time_parse_status）
│   ├── visibility.py       # 时间可见性规则（防前视执行层）+ 自检
│   ├── llm_extract.py      # Qwen3-4B-GGUF 事实抽取封装（事实层 only）
│   ├── market_collect.py   # 外围行情采集器（FRED / yfinance / akshare）
│   ├── official.py         # 官方源采集器（CNINFO 巨潮 / SSE 上交所 / SZSE 深交所，均已实测固化）
│   └── news_collect.py     # 新闻采集器（原文落盘 raw/news/{源}/{日期}/{raw_id}.json + 运行账本 + verify）
├── data/                   # corvus.sqlite 与 raw/ 原文落盘（git 忽略）
└── requirements.txt
```

## 如何运行

Python：`d:\stargazer\_dev_venv\Scripts\python.exe`（Git Bash 下用正斜杠路径）。

```bash
PY=d:/stargazer/_dev_venv/Scripts/python.exe

# a. 建库
PYTHONIOENCODING=utf-8 $PY -c "from corvus.db import init_db; print(init_db())"

# b. 可见性自检
PYTHONIOENCODING=utf-8 $PY -m corvus.visibility

# c. LLM 事实抽取冒烟
PYTHONIOENCODING=utf-8 $PY -m corvus.llm_extract

# d. FRED 拉数（key 只从环境变量读，禁止写死）
FRED_API_KEY=xxx PYTHONIOENCODING=utf-8 $PY -m corvus.market_collect US10Y 2026-09-01 2026-09-10

# e. yfinance 拉数
PYTHONIOENCODING=utf-8 $PY -m corvus.market_collect SPX 2026-09-01 2026-09-10

# f. akshare 冒烟（上证指数日线）
PYTHONIOENCODING=utf-8 $PY -c "from corvus.market_collect import smoke_akshare; print(smoke_akshare())"

# 新闻采集
PYTHONIOENCODING=utf-8 $PY -m corvus.news_collect
```

## 铁律

1. 一切记录带 `published_at` / `collected_at` / `available_at` 三个时间戳，分开存。
2. 缺失数据存 NULL，绝不填 0。
3. 原始数据只追加、不覆盖、不修改。
4. 本期不训练任何模型，不做收益预测。
5. LLM 只做事实抽取：不输出风险概率、不参与裁决、事实层与推断层不混出。

## 明确不做（本期，照抄规格第 10 节）

- 不做个股级信息否决（只到板块级）
- LLM 不输出概率、不参与裁决、不混合事实与推断
- 新闻向量不直接塞进 R1/R2 或 A2
- 不上 PatchTST/MASTER/FinGPT（后置，先证明 LightGBM 基线有效）
- 不共用训练模型/标签/概率给新闻与外围两个模块
- 缺数据不允许当成安全输出
- 不动现有盘中模型、日线模型、盯盘程序代码（盘前优先级修正只进规格）
