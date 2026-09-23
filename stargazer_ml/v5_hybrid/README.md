# Stargazer v5.0 Phase 0

本目录是人机结合决策支持系统的冻结规格与日志基础设施，不包含自动下单功能。

## 目录

- `V5_0_HYBRID_SPEC.md`：三层架构、状态流转、归因与前瞻验证边界。
- `config/schemas_v1.json`：四类每日文件Schema。
- `config/override_reasons_v1.json`：结构化理由库v1。
- `hybrid_core.py`：初始化、关联校验、理由校验、时间校验、封存和篡改验证。
- `replay_export.py`：从已观察的2026产物生成无未来字段的格式回放。
- `test_hybrid_core.py`：Phase 0自动测试。
- `replay/20260811_v3/`：历史格式回放样例，不构成新样本外证据。
- `V5_1_DECISION_MODEL_PROTOCOL.md`：大盘、板块、个股联合选股和点位建议实验协议。
- `config/recommendation_schema_v1.json`：独立的决策支持推荐输出Schema。
- `recommendation_core.py`：推荐输出的概率、时点、未来字段和价格关系校验。

## 每日工作流

初始化空目录：

```powershell
python v5_hybrid\hybrid_core.py init --day-dir <目录> --date YYYYMMDD
```

信号生成后，人工或上层界面填写决策日志，执行适配器写入成交日志。封存前验证：

```powershell
python v5_hybrid\hybrid_core.py validate --day-dir <目录> --date YYYYMMDD
```

验证通过后生成SHA-256哈希链：

```powershell
python v5_hybrid\hybrid_core.py seal --day-dir <目录> --date YYYYMMDD
python v5_hybrid\hybrid_core.py verify --day-dir <目录> --date YYYYMMDD
```

封存后的修正不得覆盖旧文件。应创建新修订文件或新日包，并通过ID引用旧记录。

## 回放样例

```powershell
python v5_hybrid\replay_export.py --date 20260811 --output-dir v5_hybrid\replay\20260811_new
```

回放只验证Schema、字段时点、完整分母和哈希链。2026-08-25以前的数据已经被观察，不能因此重新获得锁箱资格。

## 测试

```powershell
python -m unittest v5_hybrid.test_hybrid_core -v
```

## 尚未启用

- 实时或模拟订单适配器。
- 人工决策界面。
- 券商成交导入。
- 完整的机械反事实重放和五段Human Alpha归因。
- 2026-09-01后的前瞻影子运行。

这些属于Phase 1/2，不得在Phase 0样例上伪造。
