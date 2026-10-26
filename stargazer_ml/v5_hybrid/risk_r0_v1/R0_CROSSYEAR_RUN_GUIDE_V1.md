# R0跨年固定验证：执行交付 V1

> 现行更新（#18.31）：用户单独批准后本批已completed并独立审计PASS，**不要再次运行train**。F1/F2各一次、F3只复用；结果见 `R0_CROSSYEAR_RESULT_V1.md`。以下“待授权/目录不存在”保留为之前交付时的历史状态，不能据此重启已终结批次。跨年一致性未过，不部署。

2026-10-15。ENGINEERING_ENTRY_AVAILABLE / SEPARATE_FIT_AUTHORIZATION_REQUIRED。

## 当前已完成什么

跨年执行器 `risk_crossyear_v1.py` 与新评价器 `risk_crossyear_eval_v1.py` 已实现。16项新增离线测试加31项既有纯函数测试共47项通过；新测试不调用真实或合成模型fit（模型调用均mock），但真实执行了评价、写盘和异常终局路径。

245项输入/代码/合同已冻结，逐折每段数量、排序主键与完整行SHA256入manifest。三折原样本门通过；原母表、首轮模型和预测不覆盖。实际check及使用草案授权的train均原生退出2，data_ready=true / fit_ready=false，原因SEPARATE_CROSSYEAR_FIT_AUTHORIZATION_REQUIRED；真实训练目录不存在，不是环境故障。

## 一次执行的明确范围（待单独批准）

- F1：2019–2021训练，2022H1只早停，2022H2独立校准，2023测试。
- F2：2019–2022训练，2023H1只早停，2023H2独立校准，2024测试。
- F3：只复用原首轮2025冻结预测，不重新fit模型/预处理/校准器。
- 原43股受限母人口、16字段、五日−8%尾部标签、LR/GB配置不变。零取数、无搜参、无部署、无板块龙头扩展、无2026拟合。
- 新运行产物≤64MiB，四个登记目录累计≤4GiB，使用原两CPU线程；失败即整批终结，不自动重试/换折/续跑。原授权不适用于本批。
- 每年按相同候选和每日保留数比较GB−ATR/随机/直接GB−LR，原2000次10日期块区间；深尾部每次重算5%MAE均值。MAE差为负意味着更严重，不能反向判好。完整定义见 `contracts/R0_CROSSYEAR_EXECUTION_CONTRACT_V1.md`。

## 已实证的只读评价验收

`accept_risk_crossyear_entry_v1.py` 用原2025预测重现所有原概率、分组、过滤与配对区间指标，逐项完全一致；未载入模型或fit。新增比较：

| GB减基准 | 点估计（百分点） | 95%区间（百分点） | 含义 |
|---|---:|---:|---|
| 尾部率减LR | +0.014 | [−0.284, +0.242] | 未证明GB优于LR |
| 最差5%MAE减ATR | +0.131 | [−0.800, +0.757] | 未证明深尾部改善 |
| 最差5%MAE减LR | −0.210 | [−0.450, +0.064] | 未证明非劣或显著恶化 |

这些是已看历史上的补充评价，非新盲测；区间含0不能读成两模型等效。研究对象是价格不利波动，不是可交易收益。

## 用户执行顺序

工作目录：`D:\stargazer\stargazer_ml\v5_hybrid\risk_r0_v1`。

先只读检查（目前预期退出2）：

```powershell
D:\stargazer\_dev_venv\Scripts\python.exe -B -X utf8 risk_crossyear_v1.py check
```

只有用户明确批准上述一次F1/F2拟合、F3只复用后，才登记**新的**授权记录 `contracts/crossyear_risk_authorization_user_v1.json`（当前不存在）；不修改草案、不代签、不把fit_authorized改true绕门。随后：

```powershell
D:\stargazer\_dev_venv\Scripts\python.exe -B -X utf8 risk_crossyear_v1.py check --authorization contracts/crossyear_risk_authorization_user_v1.json
D:\stargazer\_dev_venv\Scripts\python.exe -B -X utf8 risk_crossyear_v1.py train --authorization contracts/crossyear_risk_authorization_user_v1.json
```

第二条仅在第一条原生退出0且fit_ready=true时执行。不要连续盲目运行这两条；已存在运行目录即不得重启。运行后须独立读回核验预测/区间/输入哈希，再解释逐年结果，不预判跨年通过，更不接入交易否决。

## 冻结指纹与交付

- manifest SHA256：`1ff309da5d300bcbeb471f128e5e36e94fb678573434a5f1b2f7beeeda7b857a`
- config SHA256：`7d4fec957d4a193a59bbf3c18042cef20e19ea034eeeb56638548c94e59941b8`
- contract SHA256：`e5290500e5ab85d49e30cefc8adb2fe11805cf20bead7af0c6ddb0dcf4ded08e`
- 独立验收：`risk_crossyear_entry_acceptance_v1.json`；技术manifest及非授权模板：`contracts/crossyear_risk_manifest_v1.json`、`contracts/crossyear_risk_authorization_draft_v1.json`。

当前停止在“技术准备完成、等待单独FIT批准”。未触网、未训练、未改桌面程序/EXE/DeepSeek预算。
