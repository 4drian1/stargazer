"""GREEN-COMMON-PROFILE-v0.1 参数冻结表 (第一轮A 前预注册).

来源: 用户 2026-09-28 提供的《新标准特征工程》+ 开发路线记录 #10.
纪律: 所有 FIRST_ROUND_PROPOSAL 参数在看任何收益结果之前写入本表; 看到结果后不得回调,
     若证伪则整轮作废并留档, 不做阈值搜索.

三层假设: 共性形态负责"值得观察哪些股票"; 个股历史负责"哪些价格区域值得等待";
盘中数据负责"现在有没有承接、能不能入场". 三者分别验证, 本轮只做第一层(共性日线选池).
"""
import hashlib
import json

# provenance 枚举:
#   USER_SPECIFIED      - 用户在规格文本中明示 (至少2天/3天, RSI<35, 低量且稳定方向)
#   PREVIOUSLY_FROZEN   - 沿用既有冻结口径 (build_pool / 成本合同), 不另造同名指标
#   FIRST_ROUND_PROPOSAL- 本轮预注册提案, 看结果前冻结, 不因收益回调

SPEC = {
    "version": "GREEN-COMMON-PROFILE-v0.1",
    "status": "FIRST_ROUND_A_PRE_REGISTERED",
    "scope": "Layer1 共性日线选池 (第一轮A); 不含个股画像(P1)与流动性区域(P2), 不改 Opportunity 输入",

    "common": {
        "atr_ref": {"def": "TR=max(H-L,|H-PC|,|L-PC|); ATR14=mean(TR,14); ATR_ref=median(ATR14,5)",
                    "provenance": "PREVIOUSLY_FROZEN (build_pool 选股标准 §4.1, 均值口径)"},
        "rsi": {"def": "RSI14 Wilder: ewm(alpha=1/14, adjust=False) 自上市起点(与 build_pool 全历史预热一致)",
                "provenance": "PREVIOUSLY_FROZEN (build_pool DEVIATIONS)"},
        "boll": {"def": "MA20 ± 2*std(close,20) (pandas rolling std, ddof=1)",
                 "provenance": "PREVIOUSLY_FROZEN (build_pool DEVIATIONS: BOLL20/2σ)"},
        "missing": "缺失保留 NULL 及原因, 不用 0 冒充 (PREVIOUSLY_FROZEN 契约)",
        "causality": "日线特征仅用截至 D 收盘数据; 无未来引用 (构建后做截断重算对拍)",
    },

    "outer_universe": {
        "def": "两池与旧池共用的外层合格宇宙 (入场/退出/成本/宇宙一致的前提)",
        "rules": {
            "warmup_rows_ge": {"value": 60, "provenance": "FIRST_ROUND_PROPOSAL (60日特征预热)"},
            "not_st_at_D": {"provenance": "USER_SPECIFIED (质量边界沿用)"},
            "float_mv_gt_0_at_D": {"provenance": "PREVIOUSLY_FROZEN (Gate 惯例)"},
            "amount_ge": {"value": 5e7, "provenance": "PREVIOUSLY_FROZEN (build_pool LiquidityPass AMT 下限)"},
            "volume_gt_0_at_D": {"provenance": "PREVIOUSLY_FROZEN"},
        },
    },

    "pool_a_strong_first_divergence": {
        "name": "A_SDP 强势首次分歧(对齐 DIP_SETUP 方向)",
        "state_machine": {
            "strength_at_div_prev_day": {
                "ret10_ge": {"value": 0.10, "provenance": "FIRST_ROUND_PROPOSAL (对齐 build_pool §6 IMPULSE>=10% 预注册研究阈值)"},
                "close_above_ma20": {"provenance": "FIRST_ROUND_PROPOSAL"},
            },
            "first_divergence_day_t": {
                "div_volume_ratio_ge": {"value": 2.0, "provenance": "FIRST_ROUND_PROPOSAL (量/前20日中位)"},
                "shape": "div_high_close_atr>=0.5 (冲高回落) OR close<前收 (方向任一)",
                "shape_high_close_atr_ge": {"value": 0.5, "provenance": "FIRST_ROUND_PROPOSAL"},
                "close_ge_boll_mid_at_t": {"provenance": "FIRST_ROUND_PROPOSAL (分歧时仍处中轨上方)"},
                "first_reset_window_trading_days_gt": {"value": 20,
                    "provenance": "FIRST_ROUND_PROPOSAL ('首次'状态机: 距上次登记分歧>20交易日才可再登记, 防事后指定)"},
            },
            "pullback_t1_to_D": {
                "length_trading_days": {"min": 2, "max": 10,
                    "provenance": "USER_SPECIFIED (至少两天) + FIRST_ROUND_PROPOSAL (上限10)"},
                "down_def": "close<前收 (红绿实体单独记录, 不混用) [USER_SPECIFIED]",
                "all_days_down": "down_streak(D) >= L (t+1..D 全部下跌)",
                "pullback_volume_ratio_le": {"value": 0.8,
                    "provenance": "FIRST_ROUND_PROPOSAL (调整段量中位/分歧日量, 缩量)"},
                "near_boll_mid_band_atr": {"value": 1.0,
                    "provenance": "FIRST_ROUND_PROPOSAL (|close_D-BOLL_mid|<=1.0*ATR_ref)"},
            },
            "dedup": "同一(code,D)取最近一次登记分歧 (最小L)",
        },
    },

    "pool_b_lowvol_bottom": {
        "name": "B_LVB 调整/底部候选(对齐 BOTTOM_SETUP 方向)",
        "gates": {
            "ret60_le": {"value": -0.10, "provenance": "FIRST_ROUND_PROPOSAL (此前持续下跌)"},
            "ret20_le": {"value": 0.0, "provenance": "FIRST_ROUND_PROPOSAL"},
            "down_flat_streak_ge": {"value": 3,
                "provenance": "USER_SPECIFIED (至少三天下跌或横盘)"},
            "flat_tolerance_abs_pct": {"value": 0.005,
                "provenance": "FIRST_ROUND_PROPOSAL (横盘=|close/前收-1|<=0.5%)"},
            "volume_level_ratio_le": {"value": 0.70,
                "provenance": "USER_SPECIFIED(低位) + FIRST_ROUND_PROPOSAL(0.70阈值; 近3日量中位/此前60日量中位, 窗口不重叠)"},
            "volume_stability_le": {"value": 0.30,
                "provenance": "USER_SPECIFIED(稳定) + FIRST_ROUND_PROPOSAL(0.30阈值; MAD(近3日)/中位(近3日))"},
            "volume_trend_abs_le": {"value": 0.10,
                "provenance": "USER_SPECIFIED(趋稳) + FIRST_ROUND_PROPOSAL(0.10阈值; log量3日逐日斜率)"},
            "rsi_lt": {"value": 35, "provenance": "USER_SPECIFIED"},
            "boll_lower_dist_atr_le": {"value": 1.0,
                "provenance": "USER_SPECIFIED(接近下轨) + FIRST_ROUND_PROPOSAL(<=1.0*ATR_ref, 可为负=跌破)"},
        },
    },

    "old_pool_baseline": {
        "def": "pool_daily_v11.primary_tier in {WATCH, HIGH_PRIORITY_WATCH}",
        "note": "旧规则仅作对照; 外层宇宙/标签/成本与新池完全一致",
    },

    "label_research_v0": {
        "status": "RESEARCH_APPROXIMATION (日线级, 不含分钟成交/限价逐笔核验)",
        "entry": "E=D的下一交易日(统一交易日历), 成交价=E开盘(raw); E无bar=UNFILLED_SUSPENDED; E一字板(O=H=L)=EXCLUDED_LIMIT_BOARD",
        "exit": "H3=E后第3交易日收盘; H5=E后第5交易日收盘; 退出日无bar=NULL(EXIT_DATA_MISSING); T+1 自动满足",
        "mae_mfe": "持有窗口(E..退出日)内 low/high 相对入场开盘; 停牌缺失bar不计(近似, 已注记)",
        "cost": "INTRADAY_SD_COST_ASSUMPTIONS_V1: 买1.0013(佣金0.0003+滑点0.001) / 卖0.9982(佣金0.0003+印花0.0005+滑点0.001)",
        "net": "sell*0.9982/(buy*1.0013)-1",
        "primary": "H3 成本后净收益为主, H5 复核 [USER_SPECIFIED]",
    },
}


def spec_sha256() -> str:
    return hashlib.sha256(json.dumps(SPEC, sort_keys=True, ensure_ascii=False)
                          .encode("utf-8")).hexdigest()
