# 完整框架120分钟连续风险固定研究交付（#18.105）

状态：**ENGINEERING_ENTRY_AVAILABLE / FIT_AUTHORIZATION_PENDING**。

研究配置、真实输入准备与训练入口已验收，尚未拟合真实样本预处理或模型。此处的技术就绪不等于风险能力有效、拟合授权或部署授权。不要修改配置中的授权开关来执行训练。

## 1. 此次检验的问题

检验“完整黄金位置/时间＋大盘＋板块全分布与上涨组滞涨＋当前/曾领涨跟踪＋个股”的联合输入，能否对未来120交易分钟最大下探幅度提供超过常数及线性基线的预测能力。

不是继续单独调整滞涨阈值/时间窗来找好结果，也不是保证树模型一定选中某条字段。完整输入不等于强制逐条规则否决，更不等于自动买卖；最终人工决策、后续三模型/JEV与监控集成范围均不变。

## 2. 冻结配置

机器配置：`auxiliary_120m_fit_preparation_20261018_v1/config.json`。

- 配置SHA256：`8a3ffb0cb2ea6f04f4f6d9cb32030d6fd7281f96ed92c1fd94c9aadcc7d0baaa`。
- 入口验收SHA256：`a7d5e3c7c09840ff107efb7e1ab557b0885a61f17644902f8a29dcd02b3b4319`。
- 原完整232逻辑字段，显式继承已冻结C1字段顺序及源映射；每字段保留缺失指示，共464维。既有36个非缺失常量列、19组训练精确重复只披露不事后删除，不称232条独立信息。
- 同86319共同人口、28只并集股票，不因短窗口增加旧研究以外样本；真实逐键/目标/截止终点与20026008个数值单元独立核对通过。这些单元不是2000万次独立事故。
- 目标为已验收`max_adverse_drop_120m`，严格决策后120交易分钟、连续非负比例、无二分类阈值。旧1200分钟/8%五日任务、全部旧模型与产物不改。

| 段 | 行数 | 独立决策日期 | 本任务用途 |
| --- | ---: | ---: | --- |
| train | 60712 | 177 | 预处理、常数、Ridge、LightGBM拟合 |
| select | 18071 | 55 | LightGBM按L2早停，固定50轮耐心 |
| calibration | 4548 | 48 | 仅报告跨期表现，不拟合校准器、不并入train |
| test | 2988 | 56 | 研发复用历史段，只报告、不选参数/阈值 |

原窗口终点仍须严格在决策后、且在原分段边界内，归属不改；旧长窗口准入带来的保守选样与代表性限制继续披露。数据原始到达时刻未知，不冒充真实历史实时回放；受限普通公司行为/快照成员/重叠目标边界均不因技术通过而消失。

## 3. 固定模型与基线

1. train标签均值：主要MSE常数基线；train中位数：次要MAE常数基线。
2. Ridge：完整232字段，`alpha=1, fit_intercept=True, solver=cholesky`。仅train中位数填补、均值/标准差标准化，附全部缺失指示；全空训练列直接阻断，不自动删列。
3. LightGBM回归：同232字段，原生NaN＋相同缺失指示；L2目标，学习率0.03、最多800树、15叶、深度5、叶样本至少300、列采样0.8、L2正则5、seed42、CPU2线程、确定性列模式；select的L2早停50。没有搜参或挑战模型。

实际后续仅拟合上述两个模型各一次。此连续目标不应套Logistic分类或sigmoid概率校准。模型估计的是下探幅度，不是“下跌概率”。

统一预先规定物理幅度输出`clip(raw,0,1)`，同时保留float64原预测，报告raw误差和小于0/大于1数量。不能看test后在raw/clip间择优，不把物理裁剪说成概率校准。源辅助标签的`training_eligible=False`保持原样；未来单独研究授权允许建立拟合视图，并非修改原始标签表或自动正式训练资格。

## 4. 固定评价与停止边界

- 主要：MSE、LightGBM相对Ridge及train均值的配对MSE差；次要MAE、train中位数常数、分月均值偏差。
- 同一时刻连续风险排序一致度，以及相同cutoff至少5股、剔除预测风险最高20%后的实际平均最大下探，与同剔除比例随机期望比较。它们不是分类AUC/胜率或收益率。
- 不确定性沿用10观察交易日块、1000次、seed42；少日期明确NOT_TESTED，重叠目标和稀疏支持不当独立充分证据。
- MSE单位是比例平方，MAE可表达为下探幅度百分点，不能与旧分类Brier直接比较为进步。
- 没有临时警报阈值、金融晋级线、PnL Gate、自动最优模型选择或部署。真实后续独立验证只能在实际权重冻结之后另行授权开始，不把旧test重新命名为盲测。

## 5. 资源与证据

准备产物67976B<256KiB；后续拟合提案上限3MiB，含128KiB终局留因。LightGBM文本无损gzip、两个float64原始预测列、预处理/模型/报告与manifest独占落盘；不复制宽特征矩阵，常数基线和裁剪视图按已冻结公式重建。

四个已登记研究根累计仍限4GiB，训练前须复验剩余额度。当前可保留这3MiB提案，但不是未来文件大小保证；触线则终结留因，不自动重跑、删旧证据或换目录逃预算。原失败记录均保留。

## 6. 已执行检查与安全运行顺序

13项单元测试使用**小型合成数据**拟合测试模型，覆盖持久化重放、cal/test标签不参与拟合、全空列、白名单/目标泄漏、幅度裁剪、配对误差及授权拒绝；没有对真实样本拟合。真实准备与入口独立验收均通过。

只读工程检查：

```powershell
cd D:\stargazer\stargazer_ml\v5_hybrid\risk_r0_v1
& D:\stargazer\_dev_venv\Scripts\python.exe -B -X utf8 auxiliary_regression_fit_v1.py --check
```

当前实测返回`engineering_ready=true`、`training_authorized=false`、`fit_performed=false`，退出0。

实测未签FIT命令即使给`--authorize-fixed-research`，仍返回`SEPARATE_AUXILIARY_FIT_AUTHORIZATION_REQUIRED`、退出2；实际拟合目录不存在。这是正确拒绝，不是模型或数据故障。

后续执行必须先收到用户对**本次固定方案**的单独批准，再生成`contracts/AUXILIARY_120M_FIXED_FIT_CHAT_AUTHORIZATION_20261018.json`，绑定上述配置和入口验收SHA；旧分类或其他批次授权不能复用。当前该授权文件不存在，不能只把开关改为true。

仅在该独立授权核验通过后才可执行一次：

```powershell
& D:\stargazer\_dev_venv\Scripts\python.exe -B -X utf8 auxiliary_regression_fit_v1.py --train --authorize-fixed-research
```

完成后执行`audit_auxiliary_fixed_fit_v1.py --result`，用保存模型/预处理重放全部原始预测并直接核算误差。若该次失败，保留失败终局、不得自动重试；若成功，也仍不能自动接入监控/JEV。

## 7. 当前交付

- `auxiliary_120m_fit_preparation_20261018_v1/`：配置、train字段剖面/重复模式、人口/目标盘点、依赖与独占终局。
- `auxiliary_fixed_preparation_acceptance_20261018_v1.json`、`auxiliary_fixed_fit_entry_acceptance_20261018_v1.json`：独立真实输入和入口验收。
- `prepare_auxiliary_fixed_fit_v1.py`、`auxiliary_regression_fit_v1.py`、`auxiliary_regression_eval_v1.py`、对应测试和审计脚本。
- `contracts/AUXILIARY_120M_FIXED_RESEARCH_SPEC_20261018.md`：研究规范冻结，FIT仍待授权。

无需再找行情、重建板块或重复本次输入盘点。下一项是明确批准后执行这一回固定研究，而不是继续调单条规则或改目标。
