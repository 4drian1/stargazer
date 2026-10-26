# 44窗价格交叉核验请求单REV3（未取数、待独立批准）

REV2于1请求后失败终结，旧代码/manifest/原文/点表/授权均保留。本REV3仅修响应区间超集适配，不追认REV2成功，不沿用其剩余额度。

## 唯一变化：响应范围与比较范围分离

仍按冻结30股的目标日期区间请求腾讯RAW day；接口返回较宽历史序列时，完整响应原文先归档，沿用REV2全部符号/复权键/日期/OHLC/有限正数/冲突重复验证，不丢掉“额外行无效”的反例。

每次响应登记raw_rows、验证后唯一日期数、返回起止日、请求起止日、早于start/晚于end的行数及起止、冻结必要点/日期数、已提取日期数、排除比较的日期数和缺点清单。只取225冻结必要点，不让额外历史改变44窗人口或进入特征。

所有冻结必要日期必须落在所签请求区间，全部必要close/low须存在；超集缺必要点仍立即失败，保留MISS_REMOTE与完整分母，不自动替代日期、翻页、重试或补查。普通案例也执行完整性门。冒烟仍要求必要价格与本地冻结值匹配。

## 继承且不放宽的纪律

- 单一腾讯RAW day通道、独立性未认证；硬顶36请求，正常30（沪/深两次全区间冒烟复用），零重试、1s间隔、10s超时；禁止重定向、换源、辅助扩围。
- 响应上限5MiB；全部落盘上限5MiB含256KiB终局报告预留，真实序列化增量先查后写，实占与预留分列。原文无法存预算时登记事实，不声称完整。
- 新目录独占，任何已有/终结/崩溃批次拒绝重启；失败立即终结。无SIGNED真实批准记录或scope/有效期/manifest/目录/预算绑定不符则零请求拒绝。
- Decimal原差绝对值≤0.01元才MATCH；P收盘与逐日low同时扰动，仅输出diagnostic_mae，不造正式MAE/Y_tail。价格一致性≠公司行为覆盖≠历史可知≠标签准入；label_eligible/training_eligible保持false。

## 版本与冻结

新入口`price_verification_batch_44win_v3.py`，新守卫`price_verification_guarded_run_v3.py`。复用未修改的REV2价格纯函数/客户端/授权门及输入格式，不改旧执行代码。

`price_verification_manifest_v3.json`保留输入schema V2，显式增加runtime_revision=3与精确response_scope_policy、父manifest全SHA256以及REV3入口/守卫/冻结脚本/本请求单绑定。225点表仍引用旧冻结v2原件，不复制大数据。旧manifest不能用作新规则的运行许可；新批准记录须绑定新manifest。

只读回放600313旧原文不是新取数批次，也不补写旧结果。独立取数须另获用户批准，本轮仅零网络构建与反例验收。

零网络：`python -B -X utf8 -m unittest test_price_verification_superset_v3 -v`；冻结后`python -B -X utf8 price_verification_batch_44win_v3.py --dry-run`。

获批后：`python -B -X utf8 price_verification_batch_44win_v3.py --fetch <新批准目录> --authorization <新真实批准记录.json> --authorization-sha256 <全SHA256>`。
