# T5 影子双名单 20260918（交易日 20260921）

- T5 冻结版：model_version=2026-09-09，model_hash=`71513c0df8b9`（sha256 前 12 位，已与 t5_freeze_manifest.json 校验一致）
- 打分口径：盘后批量 dp=1500；**PM 头 = dp1500 原生，AM 头 = out-of-native-regime approximation**
- universe：复用打分 universe_scores_20260918.jsonl（universe 股票 5282 只，打分文件 6345 只）
- 生成时间：2026-09-23 15:42:19

## 模型名单三条 caveat（必读）

1. **口径偏移**：T5 原生为盘中决策点打分器（AM/PM 双头），本名单为盘后 dp=1500 批量口径——PM 头为原生，AM 头为 out-of-native-regime approximation；评估“训练是否符合预期”时计入折扣。
2. **无 edge 声明**：与规则名单并排 ≠ 被验证；模型名单只是另一个待观察排序器（STEP1 COVERAGE_ONLY 对一切排序器名单适用）。
3. **人工从模型名单挑票买入 = 自由**（人工轨迹），落库 source 标人工；T5 分数不构成任何买卖理由代码。

## 双名单重叠：0 只（无）

## 规则层名单（green_watch_daily，正式候选）

| # | 代码 | 名称 | setup | tier | 规则分 | T5 (risk/opp 象限 AM) | 重叠 |
|---|------|------|-------|------|--------|------------------------|------|
| 1 | 300877 | 金春股份 | BOTTOM_SETUP | HIGH_PRIORITY_WATCH | 76.0 | 0.026 / 0.422 **WATCH**/WATCH |  |
| 2 | 301587 | 中瑞股份 | BOTTOM_SETUP | WATCH | 75.0 | 0.029 / 0.439 **WATCH**/WATCH |  |
| 3 | 688186 | 广大特材 | BOTTOM_SETUP | WATCH | 75.0 | 0.030 / 0.421 **WATCH**/WATCH |  |
| 4 | 300840 | 酷特智能 | BOTTOM_SETUP | WATCH | 73.0 | 0.037 / 0.433 **WATCH**/WATCH |  |
| 5 | 000518 | 四环生物 | BOTTOM_SETUP | WATCH | 73.0 | 0.074 / 0.421 **REJECT**/REJECT |  |
| 6 | 688680 | 海优新材 | BOTTOM_SETUP | WATCH | 73.0 | 0.032 / 0.472 **WATCH**/WATCH |  |
| 7 | 600125 | 铁龙物流 | BOTTOM_SETUP | WATCH | 73.0 | 0.049 / 0.421 **WATCH**/WATCH |  |
| 8 | 002110 | 三钢闽光 | BOTTOM_SETUP | WATCH | 72.0 | 0.058 / 0.421 **WATCH**/REJECT |  |
| 9 | 001358 | 兴欣新材 | BOTTOM_SETUP | WATCH | 72.0 | 0.081 / 0.429 **REJECT**/REJECT |  |
| 10 | 000848 | 承德露露 | BOTTOM_SETUP | WATCH | 72.0 | 0.033 / 0.389 **WATCH**/WATCH |  |
| 1 | 300585 | 奥联电子 | INTRADAY_DIP_SETUP | HIGH_PRIORITY_WATCH | 77.6 | 0.074 / 0.413 **REJECT**/REJECT |  |
| 2 | 600789 | 鲁抗医药 | INTRADAY_DIP_SETUP | HIGH_PRIORITY_WATCH | 76.6 | 0.062 / 0.429 **WATCH**/WATCH |  |
| 3 | 300291 | 百纳千成 | INTRADAY_DIP_SETUP | HIGH_PRIORITY_WATCH | 76.6 | 0.062 / 0.421 **WATCH**/WATCH |  |
| 4 | 002392 | 北京利尔 | INTRADAY_DIP_SETUP | HIGH_PRIORITY_WATCH | 76.6 | 0.019 / 0.478 **WATCH**/WATCH |  |
| 5 | 000701 | 厦门信达 | INTRADAY_DIP_SETUP | HIGH_PRIORITY_WATCH | 76.6 | 0.058 / 0.422 **WATCH**/REJECT |  |
| 6 | 300368 | 汇金股份 | INTRADAY_DIP_SETUP | HIGH_PRIORITY_WATCH | 76.6 | 0.025 / 0.433 **WATCH**/WATCH |  |
| 7 | 600503 | 华丽家族 | INTRADAY_DIP_SETUP | HIGH_PRIORITY_WATCH | 76.6 | 0.098 / 0.439 **REJECT**/REJECT |  |
| 8 | 603628 | 清源股份 | INTRADAY_DIP_SETUP | WATCH | 75.0 | 0.043 / 0.439 **WATCH**/WATCH |  |
| 9 | 300523 | 辰安科技 | INTRADAY_DIP_SETUP | WATCH | 75.0 | 0.096 / 0.421 **REJECT**/WATCH |  |
| 10 | 003030 | 祖名股份 | INTRADAY_DIP_SETUP | WATCH | 75.0 | 0.029 / 0.456 **WATCH**/WATCH |  |

## 模型名单（model_shadow_daily，**纯观察**，未验证模型输出）

| # | 代码 | PM risk/opp | PM 象限 | AM 象限 | OOD |
|---|------|-------------|---------|---------|-----|
| 1 | **600630** | 0.060 / 0.613 | **FOCUS** | WAIT |  |
| 2 | **603316** | 0.060 / 0.501 | **FOCUS** | REJECT |  |
| 3 | **603757** | 0.047 / 0.500 | **FOCUS** | WATCH |  |
| 4 | **002285** | 0.105 / 0.544 | **WAIT** | REJECT |  |
| 5 | **688292** | 0.096 / 0.532 | **WAIT** | REJECT |  |
| 6 | **000504** | 0.098 / 0.532 | **WAIT** | REJECT |  |
| 7 | **001233** | 0.124 / 0.532 | **WAIT** | REJECT |  |
| 8 | **003041** | 0.181 / 0.532 | **WAIT** | REJECT |  |
| 9 | **002965** | 0.106 / 0.511 | **WAIT** | REJECT |  |
| 10 | **000509** | 0.158 / 0.511 | **WAIT** | REJECT |  |
| 11 | **001238** | 0.158 / 0.507 | **WAIT** | REJECT |  |
| 12 | **603248** | 0.125 / 0.501 | **WAIT** | REJECT |  |
| 13 | **002623** | 0.143 / 0.501 | **WAIT** | REJECT |  |
| 14 | **002836** | 0.181 / 0.501 | **WAIT** | REJECT |  |
| 15 | **600847** | 0.110 / 0.500 | **WAIT** | REJECT |  |
| 16 | **002402** | 0.020 / 0.486 | **WATCH** | WATCH |  |
| 17 | **688591** | 0.026 / 0.486 | **WATCH** | WATCH |  |
| 18 | **603402** | 0.033 / 0.486 | **WATCH** | REJECT |  |
| 19 | **600577** | 0.036 / 0.486 | **WATCH** | REJECT |  |
| 20 | **001212** | 0.050 / 0.486 | **WATCH** | REJECT |  |

> 模型名单候选一律 dsl=null 纯观察（trigger 不进监控）；人工挑票买入落库 source 标人工，T5 分数不构成买卖理由代码。
