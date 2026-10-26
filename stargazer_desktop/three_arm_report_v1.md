# 三臂对照 V1（同证据包/同硬门/同选项语义；可接受集合事前登记）

- C1_pullback_ok（优先参考=—）: {'A1_code_rules': '3/3', 'A2_JEV': '3/3', 'A3_DeepSeek': '2/2'}
- C2_leader_both_weak（优先参考=DEFER）: {'A1_code_rules': '3/3', 'A2_JEV': '3/3', 'A3_DeepSeek': '3/3'}
- C3_partial_weak（优先参考=CONTINUE_OBSERVE）: {'A1_code_rules': '3/3', 'A2_JEV': '3/3', 'A3_DeepSeek': '3/3'}
- C4_hold_no_exit（优先参考=—）: {'A1_code_rules': '3/3', 'A2_JEV': '0/3', 'A3_DeepSeek': '2/3'}
- C5_stale（优先参考=—）: {'A1_code_rules': '3/3', 'A2_JEV': '3/3', 'A3_DeepSeek': '3/3'}
- A2_JEV 延迟中位 0.547s（n=15，真实调用）
- A3_DeepSeek 延迟中位 19.491s（n=14，真实调用）
- A1 为本地规则臂，无供应商调用；C5 为三臂共同本地硬门拦截。

> 适用范围注记（2026-10-20 收口）：以上结论只适用于**当前 JEV 问题设计+冻结映射+这 5 个案例**；JEV 在 C4 与代码规则结果不同，非纯规则投影；"JEV 无规则外价值"不作普遍表述。C3 的 A1 机械臂（任一走弱→DEFER）仅作对照，不进入生产规则。DeepSeek 的 C4 出集与超时进入后续真实事件观察。
