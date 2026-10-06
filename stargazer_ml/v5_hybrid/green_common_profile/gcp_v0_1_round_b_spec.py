"""GREEN-COMMON-PROFILE-v0.1 第一轮B 预注册规格 REV3 (小样本工程验收版, 未启动正式 P0/P1).

REV2→REV3 (用户 2026-09-28 第三轮裁决):
  裁决1: 锚点集选 (a) MA20 / BOLL下轨 / 前日低点(=low(P)); Low20/Low60 留后续版本.
  裁决2: 门槛保留 TAU_MU=0.0 但净收益条件用严格 mu_hat > 0; TAU_P=0.50;
         仅为固定研究假设 (FIXED_RESEARCH_HYPOTHESIS), 撤销 '最小任意性' 措辞.
  硬性1: 条件画像不得退化为股票平均收益 —— P0/P1 均按 (setup, zone_sources, AM/PM)
         单元格匹配; 样本不足即回退, 禁止取消条件凑样本.
  硬性2: 历史标签独立于放行决策 —— 全部入组事件 (含后来被估计量拒绝者) 都生成统一
         假设执行标签; 未成交/源缺失分别记录, 不填零.
  粒度修正: 源 '分钟k' 实为 1 分钟 bar (探针 2026-09-28: 600000@20260210, 240 根/日,
         ts=YYYYMMDDHHMMSS, 09:30:00-15:00:00, 无重复); REV2 的 5min 约定全部改为
         1 分钟, 执行直接接 EXECUTION_CONTRACT_V1 (下一合法可执行分钟).
  参与率: 执行分钟自身 1 分钟量 × 1% (源即 1 分钟, 撤销 5min/5 代理);
         1 分钟量仍属研究近似, 不证明开盘瞬间实际成交.
  范围边界: 本轮仅检验 '个性历史是否改善同区域下的入场/放弃判断';
         '个性穿透分位优化具体买入价格' 保留为下一阶段, 两个假设不混合.
"""
import hashlib
import json

SPEC = {
    "version": "GREEN-COMMON-PROFILE-v0.1-ROUNDB-REV3",
    "status": "SYNCED_USER_DECISIONS_20260928; 小样本工程验收 A-F 已授权;"
              " 正式 P0/P1 运行、大范围抓取、全量训练未批准",
    "scope": "小规模诊断: 同锚点区域下, 个股历史条件估计是否改善入场/放弃判断;"
             " 不含穿透分位买价优化(下一阶段), 不含成交密集区代理, 不含深度训练,"
             " 不改共性阈值, 不启动大范围数据抓取",
    "hypothesis_boundary": "本轮检验: 个性历史 vs 组统计 在相同区域/相同决策函数下的"
                           " 入场/放弃判断增量. 不检验: 个性穿透分位能否优化具体买入价格"
                           " (保留为下一阶段独立假设, 禁止混入本轮任何结论) [USER_SPECIFIED REV3]",

    # ------------------------------------------------------------------ #
    # 0. 唯一差异原则 + 估计量→决策连接
    # ------------------------------------------------------------------ #
    "single_difference": {
        "identical_across_P0_P1": [
            "候选池 (Round A 修复版 v0.1.1 冻结成员)", "区域几何 (§2)", "事件状态机 (§3)",
            "观察时点与承接条件", "决策函数形式与门槛 (§0)", "执行/退出/标签/成本 (§4/§5)",
            "评价口径与停止条件 (§6/§7)",
        ],
        "ONLY_difference": "决策函数输入估计量 (mu_hat, p_fail_hat) 的数据来源: P0=组统计,"
                           " P1=个股收缩统计; 且两者必须使用相同的条件单元格 (§0.cells)",
    },

    "decision_link": {
        "design_choice": "方案A (统计仅通过决策函数改变 入场/放弃); 方案B (穿透分位调区域几何)"
                         " 不采用且属下一阶段假设, 禁止混用 [USER_SPECIFIED REV3]",
        "cells": {
            "key": "(setup, zone_sources, session)",
            "setup": "历史事件发生时该股的 setup 归类 (A_SDP/B_LVB), 未来池名单变动不回填",
            "zone_sources": "规范化锚点组合 (合并后锚点名有序元组); 不同锚点组合不得混算",
            "session": "AM (信号 bar HHMM <= 11:30) / PM (> 11:30)",
            "rule": "P0 组统计与 P1 个股统计都必须在**同一单元格**内计算; 单元格样本不足 →"
                    " 按回退链处理, 禁止取消任何条件 (zone_sources/session/setup) 来凑样本"
                    " [USER_SPECIFIED REV3 硬性1]",
            "anti_degradation": "禁止把条件画像退化成股票/组的无条件平均收益",
        },
        "primary_estimators": {
            "mu_hat": "单元格内事件 y_k 均值 (y_k 定义见 §5, 统一假设执行标签)",
            "p_fail_hat": "单元格内事件 f_k 频率 (线性收缩, 不做 logit, 记为设计选择)",
        },
        "P0_group": {
            "mu_g": "mean{ y_k : 同 setup 同 zone_sources 同 session, 事件日 < E,"
                    " label_end_ts < E, 事件日 ∈ 回看窗 }",
            "p_g": "mean{ f_k : 同上 }", "n_g": "一并输出",
        },
        "P1_stock_shrunk": {
            "mu_1": "w*mu_stock + (1-w)*mu_g, w=n/(n+lambda), lambda=20 [PROPOSAL_NOT_PROVEN]",
            "p_1": "同式收缩", "mu_stock": "同股同单元格事件 y_k 均值, n = 该股该格事件数",
            "fallback_chain": "n < n_min=10 → 取 P0 (记 UNKNOWN_GROUP_FALLBACK);"
                              " n_g < n_min → NO_ESTIMATE 放弃 [PROPOSAL_NOT_PROVEN]",
            "uncertainty": "每次输出附 n, n_g, bootstrap 95%CI; 不输出单点必涨",
        },
        "decision_function": {
            "form": "enter = 1[ signal_fired AND mu_hat > TAU_MU AND p_fail_hat <= TAU_P ]"
                    "  (净收益条件为严格 >, 不是 >=) [USER_SPECIFIED REV3 裁决2]",
            "TAU_MU": {"value": 0.0, "provenance": "FIXED_RESEARCH_HYPOTHESIS (固定研究假设,"
                       " 非已验证安全门槛, 非最小任意性取值)"},
            "TAU_P": {"value": 0.50, "provenance": "FIXED_RESEARCH_HYPOTHESIS (同上)"},
            "identical_for_P0_P1": True,
            "on_reject": "SKIPPED_LOW_MU / SKIPPED_HIGH_PFAIL, 记录估计值; 当日不重估不重触发",
            "on_no_profile": "NO_ESTIMATE → 放弃",
            "no_threshold_search": "门槛敏感性仅描述性附录; 看过结果后换主门槛 = 违规",
            "zero_trade_guard": ">=80% 月份零成交 → INFORMATIVE_ZERO, 不得调门槛救交易数",
        },
        "track1_bypass": "轨道1 直接比较 mu_g/mu_1 (及 p) 与同一标签的关联, 不经决策函数",
    },

    # ------------------------------------------------------------------ #
    # 1. 候选池
    # ------------------------------------------------------------------ #
    "candidate_pools": {
        "source": "round_a_v0_1_impl_fix/ (MAD 修复版 v0.1.1)",
        "pool_a_sdp_sha256": "654f41fbbc66682f2cae39a0dd57881d6ee94280709901ddca67c8631b588e99",
        "pool_b_lvb_sha256": "096cc8a7d6246d40c922e7743c4151546b51be5a495ab1fffd8c9b63204f5722",
        "evaluation_window": [20210104, 20260917],
        "reporting": "A/B 分别报告, 不合并; 不重新调共性阈值",
    },

    # ------------------------------------------------------------------ #
    # 2. 日历与区域构造
    # ------------------------------------------------------------------ #
    "calendar": {
        "E": "实际监控交易日 = Round A 候选文件中 D(盘后选股日) 的下一交易日",
        "P": "E 的上一交易日 (= Round A 的 D); 区域全部由 P 日完成值构造",
        "historical_parallel": "历史事件日 H 的区域/特征由 H 的上一交易日完成值, 与实时同构",
        "no_double_lag": "全文统一 E/P 记号",
    },
    "zones": {
        "anchor_set": {
            "decision": "(a) [USER_SPECIFIED REV3 裁决1]",
            "anchors": ["MA20 = rolling(20).mean(close) @P",
                        "BOLL_L = MA20 - 2*std(close,20,ddof=1) @P (PREVIOUSLY_FROZEN)",
                        "PREV_LOW = low(P) (前日低点, 即 P 日最低价)"],
            "deferred": "Low20/Low60 留后续版本, 本轮不引入 (不同时扩范围)",
        },
        "merge_rule": {
            "algorithm": "相邻链式 single-linkage: 锚点价升序, 相邻价差 <= 0.35*ATR_ref 即合并;"
                         "簇跨度可超 0.35*ATR_ref, 跨度分布列入验收 D",
            "zone_fields": "zone_id/zone_upper(簇内最高)/zone_lower(簇内最低)/zone_mid/"
                           "zone_strength(锚点数)/zone_sources(锚点名有序元组, 逐事件保留)",
        },
        "atr_ref": "median(ATR14[E-5..E-1]) @P 完成; ATR14 = TR rolling(14).mean",
        "no_search": "锚点集/合并半径/ATR 口径不搜索",
        "no_open_dip_quantile": "禁止开盘下跌分位作区域来源",
    },

    # ------------------------------------------------------------------ #
    # 3. 历史事件 (1 分钟 bar; 确定公式状态机; 严格因果; 三类结果全保留)
    # ------------------------------------------------------------------ #
    "history_events": {
        "bar_source": "STOCKDB_HTTP '分钟k' = 1 分钟 bar (探针 2026-09-28 核实: 240 根/日,"
                      " ts=YYYYMMDDHHMMSS, 09:30:00-15:00:00); 小样本仅限登记清单, 原始快照留存;"
                      " 大范围抓取须另行批准",
        "bar_completeness": {
            "grid": "预期槽位 = 09:30:00-11:30:00 与 13:01:00-15:00:00 的 1 分钟网格"
                    " (验收 A 对照探针枚举后冻结); 缺失>2 槽或时间戳错位 → 该日 invalid 留原因",
            "feature_invalidity": "缺失槽不插值; 计算窗 (VWAP/收复时长/路径低点) 覆盖缺失槽"
                                  " → 涉及特征标 INVALID(原因), 不宣布完整 [REV2]",
            "detection_risk": "信号检测只用已到达完成 bar; 验收 A 含缺失槽注入测试与漏检率报告",
        },
        "lookback_trading_days": {"value": 250, "provenance": "PROPOSAL_NOT_PROVEN, 本轮不搜索;"
                                   " 小样本真实抓取每股 <=60 交易日封顶 [USER_SPECIFIED]"},
        "event_state_machine": {
            "bar_unit": "已完成 1 分钟 bar",
            "signed_distance": "d_atr(bar) = (bar.close - zone_upper)/ATR_ref (有符号)",
            "penetration": "pen_atr(bar) = max(0, (zone_lower - bar.low)/ATR_ref); 事件级 = 观察窗 max",
            "states": "UNARMED → SIGNAL → CONSUMED, 每 (股票, E) 至多一信号",
            "approach": "监控窗内首个完成 bar: bar.low <= zone_upper + 0.20*ATR_ref"
                        " 且 bar.low > zone_lower → APPROACH [PROPOSAL_NOT_PROVEN]",
            "break": "首个 bar.low <= zone_lower → BREAK",
            "already_below": "监控窗首 bar open <= zone_lower (跳空穿越) → ALREADY_BELOW,"
                             " 语义同 BREAK, 单独标记",
            "one_shot_per_day": "触发后无论入场/放弃, 当日 CONSUMED; 跨日无冷却 (简化, 记录)",
        },
        "event_outcomes_all_three": "成功反弹/失败续跌/持续阴跌全保留",
        "observation_window": "触发 bar 起至当日收盘 [PROPOSAL_NOT_PROVEN]",
        "event_features": {
            "pen_atr/d_atr": "连续深度/位置变量 (全量记录, 不设入组深度门槛)",
            "reclaim30": "触发后 30 交易分钟内收复 0.5*ATR_ref [PROPOSAL_NOT_PROVEN]; 窗含缺失槽→INVALID",
            "vwap_reclaim": "观察窗内收复当日累计 VWAP(截至该 bar); 含缺失槽→INVALID",
            "y_k/f_k": "见 §5 统一假设执行标签",
        },
        "causality": {
            "closed_only": "画像仅用 label_end_ts(退出日) < E 的事件",
            "pit_rule": "估计量 asof=E; 验收 B: 截至任意 E 与全历史截断重算逐位一致;"
                        " E 后 bar 扰动 (+7%价/×2.5量) 不改变截至 E 估计",
            "zones_pit": "历史事件区域只含事件日前一交易日及更早完成值",
        },
        "group_statistics": "按 (setup, zone_sources, session) 单元格聚合 (§0.cells);"
                            " 未来池名单变动不回填 [REV3 硬性1]",
        "shrinkage": {"lambda": 20, "n_min": 10, "provenance": "PROPOSAL_NOT_PROVEN, 本轮不搜索"},
    },

    # ------------------------------------------------------------------ #
    # 4. 信号与执行
    # ------------------------------------------------------------------ #
    "signal_and_execution": {
        "monitor_window": {"start": "09:35", "end": "14:55", "provenance": "PROPOSAL_NOT_PROVEN"},
        "signal_rule": "§3 状态机 (P0/P1 相同); 信号 bar 后立即用当期估计量跑决策函数",
        "execution": {
            "contract": "EXECUTION_CONTRACT_V1: signal_ts < fill_ts 严格, 合法成交分钟"
                        " [09:31,11:30]∪[13:01,15:00], 复用 execution_engine.find_next_executable_minute",
            "fill_price": "下一合法可执行 1 分钟 bar 的 open (raw)",
            "participation_cap": "执行分钟自身 1 分钟成交量 × 1% 与 50,000 元名义取小;"
                                 " 1 分钟量仍属研究近似, 不证明开盘瞬间实际成交 [USER_SPECIFIED REV3]",
            "no_same_bar_fill": "禁止信号 bar 自身成交",
            "no_close_gt_low_proof": "承接字段仅作画像特征, 不充当成交依据",
            "tradability": "涨跌停/停牌近似判定 (engine 现状 ±10% 近似) → RESEARCH_APPROXIMATION,"
                           " 不计入正式经济晋级门; 可信限价/ST 元数据 adapter 仍 OPEN",
        },
        "no_fill": "未触发/决策拒绝 → NO_FILL/SKIPPED_*, 资金留现金",
        "t_plus_1": "入场日不可卖",
    },

    # ------------------------------------------------------------------ #
    # 5. 统一假设执行标签 (独立于放行决策) [USER_SPECIFIED REV3 硬性2]
    # ------------------------------------------------------------------ #
    "labels": {
        "independence": "全部入组事件 (含后来被估计量拒绝的事件, 含 P0 会买而 P1 不买者及反之)"
                        " 都尝试生成统一假设执行标签; 画像更新不得只用 '历史上 P0/P1 愿意买' 的事件",
        "y_k": "信号 bar 下一合法可执行分钟 open 买入(×1.0013), 实际入场日后第 3 交易所交易日"
               " 收盘卖出(×0.9982), y_k = 卖出净额/买入净额 - 1; 与实时策略同一执行语义",
        "f_k": "1[ 入场后至退出日间 min(low) < 买入净价 - 0.5*ATR_ref ] [PROPOSAL_NOT_PROVEN]",
        "no_zero_fill": "未成交 (无下一可执行分钟/停牌) → status=NO_FILL_UNLABELED;"
                        " 源缺失 → status=SOURCE_MISSING; 两者从 mu/p 中排除并分别计数,"
                        " **禁止填零收益** [USER_SPECIFIED REV3]",
        "delay_rules": "退出日停牌→延至下一有 bar 日; 跌停无法卖出→延迟并记录 (研究近似)",
        "corporate_actions": "复权沿 AdjustmentCache 链; 入场-退出窗内公司行为按链调整并标记;"
                             " 跨月持有允许",
        "cost": "INTRADAY_SD_COST_ASSUMPTIONS_V1 (买 1.0013 / 卖 0.9982)",
    },

    # ------------------------------------------------------------------ #
    # 6. 评价
    # ------------------------------------------------------------------ #
    "evaluation": {
        "track_1_information_diagnosis": {
            "design": "同候选 × 预注册决策点网格 (10:00/11:00/13:30/14:30) × 同一标签;"
                      " 比较 mu_g/mu_1 (及 p_g/p_1) 与标签的关联",
            "test": "corr(mu_1,label)-corr(mu_g,label) (秩相关) 及分位组单调性差, 按日期聚类"
                    " bootstrap 95%CI (5,000 次, 种子 20260928); 主指标 = 关联量之差 CI 下界",
        },
        "track_2_strategy_evaluation": {
            "design": "完整冻结候选序列各跑一套 (同决策函数, P0 vs P1 估计); NO_FILL/SKIPPED 留现金",
            "rules": "每候选 50,000 元; 同日等权, 20 槽上限",
            "metrics": "净值/回撤、成交率、空仓率、利用率、SKIPPED_*/NO_ESTIMATE/FALLBACK 占比、"
                       "左尾、跨月分解",
        },
        "common_filled_subset": "仅辅助报告",
        "monthly_consistency": {"min_filled_events": 10,
                                "rule": "月成交<10 / 该月 P1 全回退 / 零成交 → 不进 >=60% 计数,"
                                        " 单独列示; 相等不充当支持证据"},
        "attribution_guard": "A/B 分别报告; 不以候选范围变化冒充排序提升",
    },

    # ------------------------------------------------------------------ #
    # 7. 停止条件
    # ------------------------------------------------------------------ #
    "stop_conditions": {
        "pass": "轨道1 关联增量 CI 下界>0 且轨道2 账户口径 P1 不劣于 P0 且月度一致性达标 →"
                " 方可谈后续; 仍需前瞻影子验证; 研究近似结果不得直接晋级",
        "fail": "CI 含 0 或仅相对少亏 → 裁定 '未证明增量, 不晋级' (非 '已证明个性无效');"
                " 留档负结论; 不追加阈值搜索、不换主指标",
        "no_rescue": "禁止调 lambda/n_min/TAU_*/回看期/监控窗救结果; 改参数 → 新版本号 +"
                     " 标注 '看过 Round B 结果后修改'",
    },

    # ------------------------------------------------------------------ #
    # 8. 前置验收 (已授权执行)
    # ------------------------------------------------------------------ #
    "pre_run_acceptance": {
        "sample": {
            "size": "<=20 股票 × <=15 监控日 E; 显式登记清单冻结于 run sheet 后才允许请求",
            "coverage": "上午/下午成交、跳空缺口、跨月标签、无画像回退、失败事件、"
                        "接近未跌破、严重穿透(辅助层)、invalid 日 (9 类)",
            "budget_includes_history": "预算按 股票×(E 日+真实回看日) 计; 真实回看每股 <=60 交易日"
                                       " 显式封顶, 禁止为凑 n>=10 静默扩到 250 日",
            "synthetic_for_branch": "合成历史 (确定性种子) 覆盖: 单元格 n>=10 个性分支、n=9 回退边界、"
                                    " 标签独立性 (被拒绝事件仍入画像)、条件单元格匹配、缺失槽 INVALID;"
                                    " 仅工程验收, 不进任何绩效统计",
            "raw_snapshot_retention": "登记清单内原始 1 分钟 bar 全量留存, 支持离线复验, 计入 <=5GB",
        },
        "checks": {
            "A_bar_grid": "枚举并冻结预期 1 分钟槽位; 缺失/重复槽检出正确; 缺失槽注入 →"
                          " 信号漏检率 + 特征 INVALID 标记正确",
            "B_profile_replay": "截至 E 估计 vs 全历史截断重算逐位一致; E 后扰动不变;"
                                " 合成分支 (n>=10 / n=9 回退 / 单元格隔离) 全覆盖",
            "C_closed_boundary": "全部画像事件 label_end_ts < E, 违例=0",
            "D_zone_sanity": "zone_upper/lower 相对前收分布、倒挂/负值比例、ATR 离群、"
                             "0.35ATR 链式合并率与簇跨度分布",
            "E_execution_sim": "信号→成交分钟差分布; 1 分钟参与率上限触发率;"
                               " RESEARCH_APPROXIMATION (涨跌停近似) 占比; NO_FILL/SOURCE_MISSING 分计",
            "F_budget": "峰值 <=30GB、留存 <=5GB (含快照); 抓取串行可续跑",
        },
        "gate": "A-F 全过 → 允许起草 P0/P1 运行申请 (另行批准); 大范围抓取另批",
    },

    # ------------------------------------------------------------------ #
    # 9. 变更日志
    # ------------------------------------------------------------------ #
    "change_log": [
        {"rev": "REV3", "item": "锚点集裁决 (a): MA20/BOLL下轨/前日低点(low(P)); Low20/Low60 延后",
         "type": "USER_DECISION", "status": "RESOLVED_20260928"},
        {"rev": "REV3", "item": "TAU_MU=0.0 但条件为严格 mu_hat > 0; 门槛定性为固定研究假设",
         "type": "USER_DECISION", "status": "RESOLVED_20260928"},
        {"rev": "REV3", "item": "条件单元格 (setup, zone_sources, AM/PM) 匹配, 不得退化无条件平均",
         "type": "HARD_REQUIREMENT", "status": "DONE"},
        {"rev": "REV3", "item": "标签独立于放行决策; NO_FILL/SOURCE_MISSING 分计不填零",
         "type": "HARD_REQUIREMENT", "status": "DONE"},
        {"rev": "REV3", "item": "bar 粒度 5min→1min (源 '分钟k' 探针核实 240 根/日);"
                                " 执行接 EXECUTION_CONTRACT_V1; 参与率用执行分钟 1 分钟量",
         "type": "SOURCE_TRUTH_CORRECTION", "status": "DONE"},
        {"rev": "REV3", "item": "范围边界: 穿透分位买价优化为下一阶段, 不与本轮混合",
         "type": "SCOPE_BOUNDARY", "status": "DONE"},
        {"rev": "REV2", "item": "decision_link / E-P 日历 / 链式合并 / E-P 滞后消除 等", "status": "DONE"},
        {"rev": "REV1", "item": "六点修正 (锚点统一/网格校验/连续深度入组/下一bar成交/双轨道/小样本)",
         "status": "DONE"},
    ],

    "out_of_scope_this_round": [
        "正式 P0/P1 运行 (须另行批准)", "大范围数据抓取 (须另行批准)", "全量深度训练",
        "穿透分位买价优化 (下一阶段假设)", "成交密集区/POC/VAL/AVWAP 代理 (P2/P3)",
        "共性阈值调整", "正式买入建议接入",
        "围绕 lambda/n_min/250日/TAU_*/监控窗的参数搜索",
    ],
}


def spec_sha256() -> str:
    return hashlib.sha256(json.dumps(SPEC, sort_keys=True, ensure_ascii=False)
                          .encode("utf-8")).hexdigest()
