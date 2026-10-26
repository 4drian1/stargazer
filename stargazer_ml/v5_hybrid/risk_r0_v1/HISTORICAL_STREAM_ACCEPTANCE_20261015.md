# 历史训练准备执行件复核与修复（2026-10-15）

当前仍training_ready=false，未拟合。用户继续历史数据首版方向；旧42分片人口与冻结清单保留。

## 本轮修复

1. market_index_source_v1.py：从仅检查路径改为校验同名`.binding.json`中的SH000300/CSI300_PRICE_INDEX/NONE、来源名及内容SHA256。日期须真实YYYYMMDD且非周末；价格须有限正数；冲突日期拒绝，同值重复可合并；非法行不再静默丢弃。绑定信息提供身份与内容关联，不会仅凭自填元数据证明供应商数据真实。
2. historical_stream_v1.py + materialize_historical_risk_v1.py：补真实来源→标签/16字段→Parquet分块路径，删除NotImplementedError与“构建母表需要FIT授权”的错误限制。仍不运行训练。旧报告三布尔快照或--force不能替代逐码证据；source-binding需绑定价格、因子、日历、证券范围披露、尺度与公司行为审计文件及依据引用的内容哈希。
3. 流式保护：固定Arrow字段类型，全NULL首块不会导致后续数值块schema不一致；真实压缩字节在写前预检；单次2MiB/四目录累计128MiB及16KiB终局预留；失败终局、旧目录拒绝重启、部分产物不得当完整母表。即使非准入记录也拒绝NaN/inf。逐码价格/因子必须含code，防止引用另一股票文件。
4. disclose脚本：新写入采用独占创建并核验累计预算。原披露文件保留；旧措辞“未改动盘点目录”仅可理解为原JSON未改，实际新增了披露文件，本报告予以更正，不重写旧证据。

## 验收

34项测试通过：历史核心17、指数源8、流式来源/写盘9。流式端到端使用临时合成来源：80行、15行满足完整特征/标签/时间条件；合成证据只用于测试，不是新增真实股票标签。覆盖身份/哈希错误、非法价格、冲突重复、证据未决/区间不足、跨股票误绑、预算写前停止、空流、非有限值、中途失败及目录拒绝重启。

旧check_historical_preparation_v1.py仍复验PASS且退出2；它只核验原冻结准备包，不包含本轮新增模块，不能引用其PASS声称本轮新模块已进入旧manifest。CLI只给--force继续拒绝，未新建真实母表目录。旧盘点JSON、已终结取数批次和拟合授权未改。

## 真实来源与资源现状

本地进程/监听核查未发现stockdb进程或7899监听。已读取安装SDK源码：HTTP底层日K和SDK qfq转换为不同路径；当前SDK对OHLC/pre_close均可执行动态复权，不能据其参数反推过去42个分片实际走了哪条生成路径。尚不能生成全历史统一RAW认证输入；不启动/终止数据库进程，也不以旧局部225点匹配外推整库。

沪深300真实来源未绑定；证券可见范围披露不证明完整退市覆盖。物化体积212–354MiB是按行数和3–5倍压缩假设粗估，未计全部审计列/分块元数据，不是硬上界。分块只控制内存/单块落盘；完整保存仍需实际压缩测量与足额累计存储预算，不能换目录逃额度。

下一步实质输入：运行中的本机stockdb服务或明确未复权的历史导出、对应有效公司行为数据及区间说明、带来源与身份绑定的沪深300价格日线。取得后先核验固定工程案例和源哈希，再建立真实source-binding；现有CLI已经具备计算/分块路径，无需再搭空壳。训练前三阶段尚未全部通过。

## 调用接口

```powershell
& D:\stargazer\_dev_venv\Scripts\python.exe -B -X utf8 -m unittest test_historical_stream_v1 test_market_index_source_v1 test_historical_risk_v1 -v
& D:\stargazer\_dev_venv\Scripts\python.exe -B -X utf8 materialize_historical_risk_v1.py --source-binding <真实来源绑定JSON> --index <沪深300价格CSV或Parquet> --output <risk_r0_v1内新批次目录>
```

来源绑定schema=HISTORICAL_SOURCE_BINDING_V1，calendar/security_scope_disclosure各为path+sha256；sources每项含code/prices/factors/evidence，后三者各为path+sha256。prices表含code/date/open/high/low/close；factors含code/date/factor。evidence逐码包含scale=RAW、price_sha256、factor_sha256、calendar_sha256、from/through、scale_verdict与corporate_action_coverage_verdict=ACCEPTED及非空basis_refs(path+sha256)。这些判定须来自实际核销，不得为使程序运行自行填ACCEPTED。

指数同名binding需symbol=SH000300、semantics=CSI300_PRICE_INDEX、adjustment=NONE、source、sha256。具体格式由反例测试提供合成示例；不提供假造真实通过记录。
