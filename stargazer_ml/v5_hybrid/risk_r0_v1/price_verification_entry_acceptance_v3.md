# REV3响应超集适配：零网络验收

日期：2026-10-14；状态：REV3_ZERO_NETWORK_PREFLIGHT_PASS / FETCH_AUTHORIZATION_REQUIRED。
本轮只修复与验收，不取数、不重启旧批次、不生成正式标签、不训练。

## 实施与版本

只新增REV3入口、守卫、冻结脚本、请求单、测试和授权DRAFT模板。REV2代码不改，继续复用其RAW/日期/OHLC/重复冲突校验、Decimal差额、授权、预算和原始留证。

完整响应全部归档并校验；“请求范围”与“返回范围”分别登记，记录早于start/晚于end的数量和起止，以及实际参与比较的日期数。只有冻结必要价格进入比较。超集不再导致范围误拒；必要点缺失、冲突、无效或复权响应仍立即失败，普通案例同样检查，分母不缩小。

新manifest的输入格式仍为V2，但runtime_revision=3及精确response_scope_policy明确绑定新语义。60输入全SHA256复验，包含未变的55父输入、父manifest、REV3三份代码及请求单。225点表引用原件，不复制日线或重建原结果。

- 新manifest：`independent_fetch/batch_d_mainpop_30/price_verification_manifest_v3.json`，39,980 B。
- SHA256：`9391f1ce07f1f6915262175764dfb7157b478bf3ef45be97c2da9c66a6ffdb23`。
- 父manifest：`cd1a23de1e6a1f7e766bffc44da012466cb66fe0270264221478d873f94aba7f`。

## 验收

100项零网络检查通过：新版本34项（含继承的20项授权/失败/预算/重启回归及14项超集/版本/回放反例）＋原入口34项＋客户端18项＋盘点8项＋比对器6项。

覆盖超集完整、超集缺点、普通案例缺点、冲突重复、额外行非法OHLC、错符号、复权键、必要点本身在请求区间之外、旧manifest拒绝及新CLI无授权拒绝。真实落盘计数和终局回归保持。

已归档600313响应的离线回放：640行中628行早于start；冻结必要点8个、必要日期7个，全齐且8/8 MATCH。还运行了完整入口临时测试：第一只股读取该旧原文，另外两只是假数据mock；验证第一只不再误拒且能进入后续阶段。这不是新取数批次，也不是44窗真实供应商验收，不能把mock通过当作新增行情覆盖。

- 新dry-run：44窗/264引用/225点/30股不变，60输入哈希PASS。
- 无授权新CLI：退出2、零请求、零目录。
- 旧批准记录尝试仅授权校验新manifest：AUTHORIZATION_BINDING_MISMATCH，未调用取数。
- 15个旧代码及原失败批次文件哈希全保持；旧终局仍failed:ValueError/1请求/168,806B，目录实占与原账本一致。
- 监控exe仍130,778,571B、2026-10-14 21:00:04，本轮未修改或打包。

## 待批准的下一步

请求单`price_verification_request_order_44win_REV3.md`与`price_verification_authorization_REV3_DRAFT.json`已交付，模板仍DRAFT，不沿用旧签署记录。

提案不扩大范围：腾讯RAW day外部交叉核验44窗/30股；正常30请求（两次冒烟复用）、硬顶36、零重试、1s间隔/10s超时；响应5MiB、落盘5MiB含256KiB报告预留。若单独获批，使用新目录和新批准记录，首个失败终结，不补请求、不换源。

原批次新增请求为0，真实价格证据仍仅限此前600313八点；其余217点未核验。所有label_eligible/training_eligible仍false，公司行为覆盖、历史资格/到达时刻、源口径、日历来源分别待核销，价格匹配不自动放行标签或FIT。
