"""data_regimes — 数据集「可用性制度变化」登记表 (用户 2026-09-18 裁定 #3)。

为什么单独登记:
  某些字段的可得性会在某一天发生阶跃 (availability regime change), 使同一特征在
  regime 前后的分布 / 覆盖不可比。任何跨 regime 的池化统计若不切片, 会把
  「数据可得性变化」误读成「市场结构变化」, 从而污染 A0-A4 的结论。

登记项:
  FLOAT_MV_DATA_REGIME_CHANGE = 20260701
    现象: canonical 日线 float_mv 有效率在 2026-07-01 从 ~72.9% 阶跃到 ~99.96%。
    根因: regime 前从未有 float_mv 的 ~2,035 只标的 (name 主体为 ETF/基金) 落在
          get_universe() 名单内; float_share 同步缺失 -> 无法用 PIT float_share*close 重建
          (见 probe_float_mv_reconstruct.py: reconstructable_share = 0.0)。
    影响: GREEN-WATCH 候选母池 (total_mv 25-100亿) 经数据证明无台阶;
          Market Proxy universe 自 20260701 起纳入 ~2,020 只 ETF (~4.54% P 日 float_mv 权重)。
    约束 (用户裁定): 不缩窄 data_epoch (否则仅剩两个月); A0-A4 必须强制 pre/post 切片。
"""
from __future__ import annotations

from typing import Dict

# 制度边界: 该日常量值以下的日期为 PRE 段
FLOAT_MV_DATA_REGIME_CHANGE = 20260701

# 所有已登记的可用性制度边界 (升序)。A0-A4 必须据此切片。
DATA_AVAILABILITY_REGIME_BOUNDARIES: Dict[str, int] = {
    "FLOAT_MV_DATA_REGIME_CHANGE": FLOAT_MV_DATA_REGIME_CHANGE,
}


def regime_segment(date: int) -> str:
    """返回 date 所属的制度段标签 (用于 A0-A4 强制切片)。"""
    return ("PRE_FLOAT_MV_REGIME" if int(date) < FLOAT_MV_DATA_REGIME_CHANGE
            else "POST_FLOAT_MV_REGIME")


def forced_slices() -> dict:
    """A0-A4 必须同时报告的切片定义 (缺任一即视为 A0-A4 未完成)。"""
    return {
        "boundary_token": "FLOAT_MV_DATA_REGIME_CHANGE",
        "boundary_date": FLOAT_MV_DATA_REGIME_CHANGE,
        "slices": [
            f"PRE_FLOAT_MV_REGIME (< {FLOAT_MV_DATA_REGIME_CHANGE})",
            f"POST_FLOAT_MV_REGIME (>= {FLOAT_MV_DATA_REGIME_CHANGE})",
        ],
        "policy": ("任何跨越 20260701 的池化统计必须同时给出 pre/post 两段结果; "
                   "禁止只报全期合并值。"),
        "why": ("float_mv 可得性在 20260701 阶跃 (72.9%->99.96%), 影响 Market Proxy "
                "universe 构成; 候选母池经数据证明无台阶但为审计完整仍强制切片。"),
    }
