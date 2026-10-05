# 快照 v2 接入预留

`snapshot_contracts_v2.py` 为未来模型适配器和影子运行器提供输入封套。
五类 `kind` 为 `opportunity / entry_state / risk / context / position_state`。
旧 `snapshot_contracts.py` v1 保持独立；v1 数据升级到 v2 时必须由适配器补齐
真实可见时间、质量和有效期，不能通过改版本字符串冒充迁移完成。

## 生产者输出

- 时间采用带时区的 ISO 字符串，例如 `2026-09-26T10:30:00+08:00`。
- `asof_ts` 表示信息对应时间，`available_at` 表示真实可获得时间，
  `created_ts` 表示输出完成时间。有效期由生产者冻结口径确定。
- 预测输出应附 `PredictionMetadataV2`，明确目标、期限与标签/退出/成本/执行版本。
  状态、事实快照可以不提供预测元数据。
- 模型尚未训练或本次推理失败时，调用
  `unavailable_snapshot(kind, header, reason="MODEL_NOT_READY")`；故障可使用
  `SOURCE_TIMEOUT`、`ADAPTER_ERROR` 等原因。其 payload 为空、质量为 UNKNOWN，
  不生成零风险或虚构概率。此时 header 的产物引用应指向真实适配器版本，不能编造模型 hash。
- `REPLAY` 输出保留真实创建时间；当前检查器只允许 LIVE 影子输入。
  历史回放资格由后续 C6/回放运行器单独审计。

## 消费者检查

```python
from stargazer_ml.v5_hybrid.opportunity_v2.contracts.snapshot_contracts_v2 import (
    SnapshotRequirementV2, assess_for_shadow,
)

# 调用方先冻结需求，不能从收到的 snapshot 抄一份期望值来通过检查。
required = SnapshotRequirementV2(
    kind="context", symbol="600000", data_scope="A_SHARE_RESEARCH_V1",
)
usable, reason = assess_for_shadow(
    snapshot, "2026-09-26T10:30:00+08:00", requirement=required,
)
```

各模块按自己的输入语义定义 requirement；大盘上下文与个股快照不必使用同一个 symbol。
`benchmark_scope=None` 要求输入不带基准域；有基准的模型必须显式填写。
预测消费者应在 requirement 中提供完整的 `prediction` 元数据以核对口径。
未提供 requirement 的调用只做原有单张时间/质量检查，不证明跨模块匹配。

`READY` 只表示这张快照满足本次输入要求。不同期限的概率是否可组合、如何形成
买卖建议、是否需要人工确认，仍由后续版本化 Policy 定义。
UNKNOWN/DEGRADED/INVALID 可以记录和展示，但当前检查器不把它们判为 READY。

## 验证与边界

在项目根目录运行：

```powershell
& 'D:\stargazer\_dev_venv\Scripts\python.exe' -m unittest discover `
  -s stargazer_ml/v5_hybrid/opportunity_v2/contracts -p test_snapshot_contracts_v2.py
```

封套支持 JSON 往返；这不是 SQL 持久化、不可变日志或原始证据正确性的证明。
冻结 dataclass 也不提供 payload 内嵌容器的深层不可变性；未来落库边界应保存
当次完整序列化内容，不能把内存对象当审计账本。
框架选择、提醒去重、重启恢复、监控主程序接入留到 SHADOW_RUNTIME_V1 验收。
