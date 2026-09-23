# 持仓/观察名单 T5 影子打分 20260918

- T5 冻结版 model_version=2026-09-09，model_hash=`71513c0df8b9`（已与 t5_freeze_manifest.json 校验）
- 口径：盘后批量 dp=1500；PM 头原生 / AM 头 out-of-native-regime approximation；advisory 非交易指令。
- 卖出优先级硬契约：**HARD_RISK_FLOOR（冻结灾难底线）> T5 卖出节点 > 人工裁量**。本表为 advisory（建议性质），T5 节点只能作为 evidence_ref 证据，不构成自动卖出指令。

## 持仓（positions）

| 代码 | 名称 | 建仓 | 逻辑 | PM risk/opp | PM 象限 | AM 象限 | 节点建议 |
|------|------|------|------|-------------|---------|---------|----------|
| 600207 | 安彩高科 | 2026-08-05 @ 4.37 | ????+???? | 0.074/0.389 | REJECT | WATCH | **SELL** PM=REJECT/AM=WATCH |
| 600725 | 云维股份 | 2026-08-15 @ 3.3 | 候选胚子 | 0.030/0.389 | WATCH | WATCH | **HOLD** PM=WATCH/AM=WATCH |

## 观察名单（watch）

| 代码 | 理由 | PM risk/opp | PM 象限 | AM 象限 | 节点建议 |
|------|------|-------------|---------|---------|----------|
| 300155 | ???? | 0.055/0.352 | WATCH | REJECT | **SELL** PM=WATCH/AM=REJECT |
| 688602 | 康鹏科技，候选胚子，参考止损5.95 | 0.042/0.453 | WATCH | WATCH | **HOLD** PM=WATCH/AM=WATCH |
| 600774 | 汉商集团，候选胚子，参考止损5.96 | 0.047/0.421 | WATCH | WATCH | **HOLD** PM=WATCH/AM=WATCH |

> 统计：买节点候选 0，卖节点候选 2。打分 5/5 只，耗时 8s。
