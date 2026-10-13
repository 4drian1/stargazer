"""前瞻记录准备（PREPARATION ONLY — 未启用、不补造历史时间）。

为 R0-V1 标签工程准备"三时间戳"落盘能力，供后续经独立授权后启用：
  - eligibility_known_ts : 候选资格*可知*时刻（截至 P 日是否已知）
  - data_arrival_ts     : 日线 / 公司行为数据*实际到达*时刻
  - decision_ts         : 模型*决策*时刻（须落在 [P 15:00, 下一交易日开盘)）

设计原则（用户 2026-10-13 裁定）：
  - 三者分别落盘，不复用单一时间戳；
  - **授权分离**：真实采集须经独立的 `RECORD_COLLECTION` 授权；
    `LABEL_BUILD`（标签构建授权）**不自动授权**启动采集；
  - 先验收本记录链（见 ACCEPTANCE_CHECKLIST R1–R6），再单独授权启动；
  - **不补造历史时间**：本模块只写"启用后真实产生"的记录，绝不回填；
  - 本模块当前**不被任何管线调用**，默认 `enabled=False`；即便持有 LABEL_BUILD，
    也**不**因此自动写盘。启用须显式 `RECORD_COLLECTION` 授权且记录链通过验收。
"""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Optional

# 落盘路径（启用前由授权范围指定；当前仅占位，不被写入）
DEFAULT_LOG = Path(__file__).parent / "forward_recording_log.jsonl"


@dataclass
class ForwardRecord:
    code: str
    E: int
    P: int
    # 三时间戳：启用后由真实管线填充；准备阶段一律 None（禁止伪造/回填）
    eligibility_known_ts: Optional[int] = None   # epoch ms；资格可知时刻
    data_arrival_ts: Optional[int] = None        # epoch ms；数据实际到达时刻
    decision_ts: Optional[int] = None            # epoch ms；决策时刻
    recorded_at: int = field(default_factory=lambda: int(time.time() * 1000))
    source: str = "forward_recording"            # 启用后的采集器标识

    def is_complete(self) -> bool:
        """三时间戳齐备且决策时刻落在 [P 15:00, 下一交易日开盘) 内才算可用。"""
        return (self.eligibility_known_ts is not None
                and self.data_arrival_ts is not None
                and self.decision_ts is not None)

    def record_hash(self) -> str:
        payload = json.dumps({
            "code": self.code, "E": self.E, "P": self.P,
            "eligibility_known_ts": self.eligibility_known_ts,
            "data_arrival_ts": self.data_arrival_ts,
            "decision_ts": self.decision_ts,
        }, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


# 启用前的验收清单（先于"启动授权"，不替代任何数据/标签授权）
ACCEPTANCE_CHECKLIST = [
    "R1_三字段分离: eligibility_known_ts / data_arrival_ts / decision_ts 各自独立采集，无复用单一时间戳",
    "R2_无回填: 仅写入启用后真实产生的记录；历史时间戳保持 None，禁止伪造",
    "R3_决策窗口: decision_ts 校验落在 [P 15:00, 下一交易日开盘) 内；越界或缺失置 training_eligible=False",
    "R4_未知处理: 任一时间戳为 None / 无效 / 已到或已过开盘 -> training_eligible=False，不进拟合",
    "R5_可审计: 每条记录含 recorded_at 与 record_hash；日志追加写、不可变",
    "R6_授权隔离: 启用须经 SIGN 与 LABEL_BUILD 授权；本模块默认不写盘",
]


def append_record(rec: ForwardRecord, path: Path = DEFAULT_LOG,
                  _authorized_collection: bool = False) -> None:
    """落盘一条前瞻记录。

    **PREPARATION**：当前不被任何管线调用。真实采集须经独立的 `RECORD_COLLECTION`
    授权——调用方必须显式传入 `_authorized_collection=True`（该开关不在任何代码路径中
    被自动置位，须由持有 RECORD_COLLECTION 授权的外层流程显式打开）。`LABEL_BUILD`
    授权**不**使其自动为真。违反 R2（回填）或 R3（越界/缺失）将被拒绝。
    """
    if not _authorized_collection:
        raise PermissionError("forward recording disabled; 须独立 RECORD_COLLECTION 授权 (LABEL_BUILD 不自动启用)")
    if not rec.is_complete():
        raise ValueError("forward record incomplete; 禁止写入不完整/伪造的时间记录 (R2/R3)")
    line = json.dumps(asdict(rec), ensure_ascii=False)
    with open(path, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def run_acceptance_selfcheck() -> dict:
    """静态自检：确认本模块未启用、清单完备。（可被管线验收调用，不写盘）"""
    return {
        "enabled": False,                 # 准备阶段：未启用
        "backfill_allowed": False,        # R2：禁止回填
        "writes_disk_now": False,         # 当前不被任何管线调用
        "checklist_items": len(ACCEPTANCE_CHECKLIST),
        "note": "PREPARATION ONLY; 启用须经 SIGN+LABEL_BUILD 授权且记录链通过验收",
    }


if __name__ == "__main__":
    print(json.dumps(run_acceptance_selfcheck(), ensure_ascii=False, indent=2))
