# 固定首30码事件页试批：工程交付

> 现行更新（#18.35）：用户单独批准后本批已completed，30请求/1,594,542B响应体/1,664,184B实际落盘，独立核账通过。**不要再执行fetch**，目录存在即拒重启。结果见 `R0_FIRST30_FETCH_RESULT_V1.md`；以下待授权/目录不存在是此前工程交付时的历史状态，原文保留，不据此续用本批额度。未授权本机快照/标签/FIT或剩余366码。

2026-10-15。ENGINEERING_ENTRY_ACCEPTED / SEPARATE_INDEPENDENT_FETCH_AUTHORIZATION_REQUIRED。**本轮0外部/本机HTTP、0标签、0训练、0监控变更。**

## 现行版本与已验证事实

入口 `expansion_first30_v1.py`，绑定 **manifest_v2 / 授权草案_v2 / 验收_v2**；不是v1技术冻结。研究人口及30码清单未改，旧草案/源指纹保留。

- 新25项+既有69项=94项无拟合/无网络测试PASS。覆盖GET/HTML/identity头、禁重定向、30次共用计数与2次冒烟响应复用、1秒间隔、10秒socket timeout、累计响应触顶不多读探针、HTTP/身份/表头/分页/编码拒绝、连接失败和部分读取原文归档、精确磁盘及四根累计预检、首失败终局、同对象/重启/不完整目录拒绝，以及未授权不构造网络Batch。
- 43份既有真实归档页全部通过新解析机制检查；全30码使用合成响应跑通完整写盘/哈希/终局读回，明确不是新30码真实证据或远端可用性验证。
- 验收脚本禁止socket.create_connection，0真实网络；check和带DRAFT授权的fetch在真实输入上原生退出2，生产目录不存在。空批准/旧scope/错哈希/无时区/未来时刻被拒。
- v1技术冻结最外层异常曾误报network_started=false，已撤回，源码快照与旧冻结件保留；现行未知错误为null并提示查claim/attempt。不是“批次取数失败”，没有烧毁请求。见 `FIRST30_TECHNICAL_FREEZE_V1_WITHDRAWAL.md`。

## 请求批准的唯一取数范围

固定 `contracts/r0_expansion_first30_request_draft_v1.json` 的30只股票，来自400码机械名单中缺外部页者的原哈希顺序前30，不替补。前两码300923/000761冒烟，响应直接进入本批档案，不重复发送。

| 项目 | 硬边界 |
|---|---|
| 来源 | 固定Sina分红配股HTML页，GET，不切源/不翻页/不跟重定向 |
| 请求 | 全批最多30次尝试，含2次冒烟与全部辅助；实现没有辅助请求，0重试 |
| 响应 | 累计≤8MiB，单次≤512KiB并受剩余预算限制；恰到读取限额即拒绝 |
| 落盘 | ≤12MiB，含原文/sidecar/attempt/索引/报告/终局；四研究根累计≤4GiB |
| 时间 | 开始间隔≥1秒，socket timeout=10秒；不承诺整批30秒完成 |
| 失败 | 首失败全批终结，不换代码/批次名续跑，不退还计数 |
| 交付 | 原始证据与机制状态；不置覆盖，不生成标签/训练/部署 |

取数权限不包含本机RAW快照、事件对拍、标签/特征物化或FIT。取到页面不等于公司行为完整；即使30页全部通过，也不能说400股已准备好训练。

## 授权与运行顺序

工作目录 `D:\stargazer\stargazer_ml\v5_hybrid\risk_r0_v1`。

当前只读检查（无批准应退出2）：

```powershell
D:\stargazer\_dev_venv\Scripts\python.exe -B -X utf8 expansion_first30_v1.py check
```

只有用户明确批准上述一次有界INDEPENDENT_FETCH后，才登记新 `contracts/r0_first30_fetch_authorization_user_v2.json`，绑定下列三哈希；不修改原DRAFT，不代填用户签名，不改fetch_authorized开关。随后：

```powershell
D:\stargazer\_dev_venv\Scripts\python.exe -B -X utf8 expansion_first30_v1.py check --authorization contracts/r0_first30_fetch_authorization_user_v2.json
D:\stargazer\_dev_venv\Scripts\python.exe -B -X utf8 expansion_first30_v1.py fetch --authorization contracts/r0_first30_fetch_authorization_user_v2.json
```

第二条仅在第一条原生0且fetch_ready=true时执行；不是让用户现在连续运行这两条。生产目录固定 `independent_fetch/expansion_first30_v1`，任何已存在目录均拒绝运行；首失败停止，报告后等待新裁决。未知外层错误不能当零请求，按目录/attempt/原文核账。

## 现行指纹与交付

- request：`edf6ced3e667abf0175c18c46224c1275236aed76265c657c540ac9d970cb7dc`
- manifest_v2：`95a1b7a41576aa88ba62177e151a11a26f771ae09782661e78fcce07f76dd317`
- contract：`7791068ba269d76b61a3ca92aeb6a7cc7f76b4d4b6f374e77f42a0e1d2194b2c`
- 技术冻结310项：`contracts/r0_first30_fetch_manifest_v2.json`。
- 独立零网络验收：`expansion_first30_entry_acceptance_v2.json`。
- 执行合同：`contracts/R0_FIRST30_FETCH_EXECUTION_V1.md`。

当前用户批准文件不存在；下一步只是是否授权这一次30请求试批，而不是继续训练或全量396码采集。
