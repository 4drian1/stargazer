# JEV 映射规则冻结 V1（2026-10-20；用户要求：排除"代码表现记到模型头上"）

## 规则（冻结，任何修改须登记新版本）

输入：JEV 对 3-4 个 noul 问题的 0..1 评分（原始分逐次保存）。
UNPOSITIONED：
1. `insufficient>0.5` → EVIDENCE_INSUFFICIENT（conf=ins）
2. 否则 `buy_ready>0.5 且 defer≤0.5` → READY_BUY_CONFIRM_HUMAN（conf=buy）
3. 否则 `defer>0.5` → DEFER（conf=defer）
4. 否则 → CONTINUE_OBSERVE（conf=max(buy,defer,1-ins)）
POSITIONED：
1. `insufficient>0.5` → EVIDENCE_INSUFFICIENT
2. `reduce>0.5` → REDUCE_EXIT_CONFIRM_HUMAN（conf=reduce）
3. `heighten>0.5` → HEIGHTEN_RISK_WATCH（conf=heighten）
4. 否则 → HOLD_OBSERVE

## 已核查的贡献点（用户关切）

- **代码不强制"走弱→暂缓/减仓"**：规则只在 JEV 对"defer/reduce"问题的**评分**>0.5
  时才选该选项；走弱标记本身不经代码进入规则。冲突案例 DEFER=0.84 是 JEV 对
  "是否已出现反对原计划的证据"的评分，非代码硬编码。
- **代码不新增交易偏好**：规则只做选项映射；触发/时间/证据硬门全在 judge()，
  与 LLM  arms 共用同一套。
- **问题措辞的引导性**（残余风险，登记）：问题"是否已经出现了反对原计划的证据、
  **应当暂缓**推进？"含结论性措辞，可能抬高 defer 分。本轮不改（避免按已看结果调
  提示），下轮以**中性措辞**平行问一组对照问题（如"反对买入的证据强度？"），
  比较措辞敏感性——登记为待办，不回溯改分。
- **JEV 原生 Choice 接口**：未知（文档未提供；不猜类型值乱试）。若 vendor 文档
  确认支持直接返回冻结选项，优先切换原生 Choice，代码只保留硬门。
- **三臂对照要求**：arm1=本规则+代码布尔输入（无供应商）——用于度量规则自身
  贡献；arm2=JEV 评分+本规则；arm3=DeepSeek 自由判断。同硬门同选项语义。
