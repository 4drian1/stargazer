# 盘后风险研究候选：只读评分交付

状态：RESEARCH_ONLY；工程评分入口可用，不是生产部署或独立验证通过。

## 当前固定版本

候选 `R0_POSTCLOSE_D2_FIXED_RESEARCH_20261017`，沿用第二轮D2的54棵树、131字段、原生缺失指示和固定校准偏移。没有按D1/D3最好指标替换，也没有重新拟合或启用60成熟日滚动校准。

黄金位置、大盘、板块、当时领涨与捕获跟踪、个股仍是完整联合架构；不新增ATR门、不改成右侧等待。这个入口只交付已完成的**盘后研究模型**，不宣称盘中5/15分钟预警模型已经训练完成。

`q90_next_session_adverse_excursion` 是既有条件次日下探幅度的90%分位数估计，越大表示估计的较坏下探越大。例如0.04表示幅度估计4%，不是“下跌概率4%”，更不是可保证的最大亏损。目标定义沿用原拟合合同，未换成五日8%分类或实际交易收益。当前不同月份的分位数校准不稳，不能把Q90当可靠风控承诺。

## 如何运行

PowerShell：

```powershell
Set-Location D:\stargazer\stargazer_ml\v5_hybrid\risk_r0_v1
& D:\stargazer\_dev_venv\Scripts\python.exe -B -X utf8 postclose_research_scorer_v1.py --input postclose_scoring_handoff_20261017_v1/example_input.json
```

示例只有两条已经评价过的历史样本，**不是今日建议或新增验证数据**。命令仅读输入和固定原权重，将JSON写到标准输出，不调用stockdb/公网、不写学习记录、不改监控或EXE。默认manifest位于 `postclose_scoring_handoff_20261017_v1/manifest.json`；它引用原模型，不复制大文件，所以该交付不是脱离原目录的便携包。

输入要求：

- `candidate_id` 与 `feature_schema_sha256` 必须匹配manifest；`usage_mode=HISTORICAL_RECONSTRUCTION`。
- 每行含code、day、feature_asof_ts、decision_available_ts、features，禁止附带标签列。131个字段键必须齐全，可选缺值显式null，不能缺键或用0替代未知。
- `feature_asof_ts=P日15:00:00`仅是历史重建的已完成日线假定截止，不是实际到达时刻。当前历史实际到达未绑定，`decision_available_ts`必须null，非空会被拒绝。没有开放实时调用模式。
- 现阶段仅43个既有证据码。该代码范围不是全市场适用性证明；其他代码返回不可评分，不自动扩域。
- 单次最多512行、JSON最多4MiB，重复(code,day)、NaN/inf、模式/版本不符、文件SHA漂移均拒绝。原文件不要手动修改来绕过检查。

## 缺失与状态

与D2队列一致：个股日RSI、沪指和创业板指日RSI、板块日收益、至少一种当前/捕获领涨日收益、黄金位置必须有值。可选缺值保留为原生NaN及缺失指示，不新增全字段非空门。

必要组缺失或单行格式/范围不符：`status=UNSCORABLE`、Q90=null并附reasons；不是低风险。正常行为`RESEARCH_SCORE`，但整批仍明确RESEARCH_ONLY、provenance_verified=false、deployment_authorized=false。消费者必须逐行检查status，不能仅凭进程退出0就使用所有行。整批模式/字段版本/哈希等拒绝时退出2。

输入内容哈希只证明“这次输入是什么”，不认证来源、PIT、特征计算正确或到达时间真实。评分器不从原始K线重建字段；上游仍须沿用冻结构造及验收，不能把未来信息塞进字段后以“接口接受”认证无泄漏。manifest自身也不是身份签名或安全认证。

## 本轮验收

- 新接口20项离线单元测试及7项原修复拟合/批次失败回归，27/27通过。
- 原D2 test全部2162行、108日期逐条评分复现，最大绝对差 **0.0**；原权重、预处理、校准、原预测哈希均未变。
- 测试涵盖必要组缺失、可选NULL、市场双指数/领涨任一的原资格逻辑、顺序不敏感、重复键、非有限值、非法日期、版本不符、加载后篡改、CLI成功及拒绝；CLI临时输入/目录前后完全一致。
- 新包20196字节，小于2MiB；不改变四根累计4GiB合同。验收为工程复现，不是新增样本外效果或金融Gate。

机器记录：`postclose_scoring_handoff_20261017_v1/acceptance.json`；冻结引用：同目录manifest.json；终局：stop_record.json。构建脚本拒绝覆盖该目录，不重跑生成新权重。

## 下一步

固定这一版本，准备**未参与开发时段**的输入与评价安排：先登记数据暴露、来源/到达时间、覆盖/缺失规则、对照与晋级标准，再运行一次固定评价。现有重建访问记录末日2026-10-16，不能把其中换个日期切片重新称为全新盲测；未暴露输入目前UNBOUND。历史研发可以继续，但须明确名称与范围，不靠反复调整当前test晋级。

这轮不启动采集/T0、不自动接监控、不做JEV或三模型融合，不产生买卖、仓位或自动否决指令。选股模型单线运行的应用外壳保持原状；信息/风险模型后续接入仍是独立任务。
