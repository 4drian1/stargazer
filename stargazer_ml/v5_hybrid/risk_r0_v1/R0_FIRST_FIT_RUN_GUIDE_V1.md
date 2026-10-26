# R0首轮风险模型：训练入口交付 V1

> 当前进展（2026-10-15）：用户已认可并授权，首次真实FIT已完成、终局completed，不能照下文再次train。当前结果见 `R0_FIRST_FIT_REPORT_V1.md`；下文为授权前工程交付的操作记录，保留供审计。

原交付日期2026-10-15；当时状态 **ENGINEERING_READY / FIT_AUTHORIZATION_PENDING**。

数据准备与训练入口均已完成。目前没有数据/样本数/依赖阻碍；唯一未放行项是本次真实FIT授权。合同技术草案已哈希冻结，不代签、不冒称已签。不要改配置布尔值或复制GREEN授权文件来运行。

## 1. 本轮会训练什么

固定43股受限历史研究母表，只使用已准入的59,923股票日（41股）。train35,373、select4,270、calibration4,584、test9,107、2026描述6,589。其他20,917行保留但不训练；不扩大为5198码。

- 标签仍为未来5交易日MAE≤−8%的价格风险，非交易盈亏。
- 固定16字段：11个个股+5个真实沪深300价格指数上下文；code/P只用于主键、时间切分和并列裁决，不作为输入。
- train常数概率 + ATR风险排序规则 + LR(C=1.0/L2) + LightGBM(15叶/深5/min_child300/800树上限等)。LightGBM为固定主模型，不从测试结果另挑赢家。
- 仅train拟合LR与预处理；LightGBM仅select早停50轮；两模型只在calibration分别拟合sigmoid，不能拿test调参。
- 主剔除20%，同日同保留数量；10/30%仅描述。风险差方向为模型−对照（尾部率负值较好）；10日期移动块/2,000次配对bootstrap。
- 仍保留来源选择、历史时间可用性假设、潜在漏记及证券覆盖限制。coverage_verified=False不改成认证完整。2025不是新盲测，此轮不是多年Walk-forward；无自动上线权限。

完整技术条款见 `contracts/RESTRICTED_RISK_FIT_CONTRACT_V1.md` 和 `contracts/restricted_risk_fit_config_v1.json`。

## 2. 已验证到哪一步

21项新增测试 + 78项既有回归 = **99项通过**。新增覆盖：授权范围/哈希绑定、缺失和非有限值拒绝、跨段purge、同日同保留数、差值方向、随机期望、移动块配对、写前预算、失败终结及再次运行拒绝。

使用真实安装的sklearn/LightGBM对**合成数组**测试了训练/早停/独立校准；合成端到端输出还使用了明确的非拟合假模型。它们均不读取真实母表执行fit，不将临时模型用于监控。

对真实母表只执行了preflight/授权拒绝测试，结果：

| 命令（草案授权） | 原生退出码 | data_ready | fit_ready | 结果 |
|---|---:|---|---|---|
| check | 2 | true | false | FIT_AUTHORIZATION_REQUIRED |
| train | 2 | true | false | FIT_AUTHORIZATION_REQUIRED，真实fit未调用 |

真实输出目录 `historical_risk_fit_v1_run` 不存在。不是训练失败，也不是环境故障；训练尚未授权。

## 3. 当前可以执行的只读检查

```powershell
& 'D:\stargazer\_dev_venv\Scripts\python.exe' -B -X utf8 'D:\stargazer\stargazer_ml\v5_hybrid\risk_r0_v1\risk_fit_v1.py' check --authorization 'D:\stargazer\stargazer_ml\v5_hybrid\risk_r0_v1\contracts\restricted_risk_fit_authorization_draft_v1.json'
```

预期退出2，数据通过、FIT拒绝。哈希漂移/样本门/依赖/预算等异常退出1；取得有效授权后的check才退出0。不会自动修复哈希、更新数据或联网。

## 4. 真正开训只差什么

用户确认本技术合同并单独批准“一次首轮受限历史FIT：常数/ATR/LR/LightGBM，按冻结配置及固定评价，新增64MiB/四目录累计4GiB，不上线”。收到真实授权后另存授权记录（例如 `contracts/restricted_risk_fit_authorization_user_v1.json`），保留DRAFT，绑定下面三个实测哈希，并登记实际签署人/授权时刻/聊天依据。

不通过手填status=AUTHORIZED、虚构时间或仅改fit_authorized=True取得授权。工程代码的授权文件校验是运行门禁，不是数字签名或身份认证系统；实际权限来自用户明确批准。

| 冻结件 | SHA256 |
|---|---|
| manifest（227项输入） | d9a8a1ada719b9e30e70ae032dc26ef0c2dca5e3b240341477fe34e98079699c |
| config | 5de1f90d6efa47ff8aa4f32670b271c67bdf7193afed5c40e1e55248d9584a8a |
| contract | 166667e686068d7564c5d13579e0a57b5e5ce8018a65f6f444c8b20f7f2dd2da |

技术配置中的fit_authorized=false是冻结时快照，不需也不能改。授权只在独立记录生效，任何冻结文件变更都须先停止并走新版本，不运行freeze覆盖旧manifest。

## 5. 取得实际授权后，用户运行顺序

下面授权文件目前**不存在**，这些不是现在就可直接运行的命令：

```powershell
& 'D:\stargazer\_dev_venv\Scripts\python.exe' -B -X utf8 'D:\stargazer\stargazer_ml\v5_hybrid\risk_r0_v1\risk_fit_v1.py' check --authorization 'D:\stargazer\stargazer_ml\v5_hybrid\risk_r0_v1\contracts\restricted_risk_fit_authorization_user_v1.json'
```

只在上一条退出0时运行：

```powershell
& 'D:\stargazer\_dev_venv\Scripts\python.exe' -B -X utf8 'D:\stargazer\stargazer_ml\v5_hybrid\risk_r0_v1\risk_fit_v1.py' train --authorization 'D:\stargazer\stargazer_ml\v5_hybrid\risk_r0_v1\contracts\restricted_risk_fit_authorization_user_v1.json'
```

固定输出：`D:/stargazer/stargazer_ml/v5_hybrid/risk_r0_v1/historical_risk_fit_v1_run/`。独占创建，失败或完成后均不可覆盖/直接重跑；保留停止记录，先诊断再决定另批。无需重启stockdb，本入口不发HTTP请求。

## 6. 跑完看什么

- `stop_record.json`：先确认completed与文件哈希；failed时不拿残留预测作结论。
- `models.joblib`：LR/预处理、LightGBM、两套校准器、训练常数和最佳树数。仅加载可信的本地产物，不能加载外来同名pickle。
- `predictions.parquet`：逐样本标签/MAE、raw与校准概率、常数概率、ATR排序值、主比例保留标志。
- `report.json`：各段概率质量、校准分箱、尾部过滤配对CI、MAE左尾和板块暴露、逐日充分统计；校准段是校准器样本内，2026仅描述。

优先解释2025历史留出的LightGBM−ATR/随机尾部率差与Brier−常数差，而不是只看训练段或平均MAE。研究证据标志即使全通过，production_gate_passed仍为false；没有账户PnL、没有自动否决交易、没有进入监控。
