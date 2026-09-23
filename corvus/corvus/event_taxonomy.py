"""事件分类法（event_taxonomy.py）：三层路由 + 标准类别映射，2026-09-12 冻结 v1.0。

设计原则（用户规格）：
- 字典不无限扩张：保存两层 event_type_raw（Qwen 原始描述）与 canonical_event_type
  （固定标准类别），自由文本经关键词规则映射到标准类别，映射不上归 OTHER_* 并留原始值；
- 三级路由 materiality_tier：
  P0 重大风险与强影响事件 → 完整抽取，未来进入 Corvus 事件模型 v1 的八个事件族；
  P1 可能重要（容器文档）→ 保留并抽取，is_container_document=true，需读正文决定是否升级 P0；
  P2 例行中性 → 保存原始记录 + 轻量规则分类，只随机抽 5%–10% 作中性对照，不逐条调 LLM；
- 权益分派（DISTRIBUTION）不进风险模型，先进除权除息标记/复权清洗/股息率特征；
- 业绩说明会（INVESTOR_COMMUNICATION）只提取其中具体新信息，"召开说明会"本身无方向。
"""
import re

# ================= 标准类别（canonical_event_type，稳定英文 ID） =================
# tier P0：Corvus 事件模型 v1 的八个高价值事件族
TIER_P0 = {
    "REGULATORY_PENALTY": "监管处罚与立案（立案调查、行政处罚、警示函）",
    "DELISTING_RISK": "ST、退市、重大违法强制退市",
    "EARNINGS_PREVIEW": "业绩预告（预增/预亏/扭亏，含快报）",
    "EARNINGS_REVISION": "业绩预告修正/大幅修正",
    "SHAREHOLDER_REDUCTION": "大股东/董监高减持",
    "SHAREHOLDER_INCREASE": "大股东/董监高增持",
    "BUYBACK": "回购（大额回购）",
    "SHARE_UNLOCK": "集中解禁",
    "RESTRUCTURING": "重大资产重组、并购、控制权变更",
    "MAJOR_CONTRACT": "重大合同、订单、经营事故、停产、召回",
    "LITIGATION_DEFAULT": "重大诉讼、债务违约",
    "INDUSTRY_POLICY": "行业监管与扶持政策",
    "SANCTION_TRADE": "制裁、出口管制、关税",
    "MACRO_POLICY": "宏观政策（货币/财政/产业）",
    "REGULATION_RELEASE": "监管法规发布（办法/条例/规章公布）",
}
# tier P1：容器文档/可能重要，先抽取不自动定风险
TIER_P1 = {
    "BOARD_RESOLUTION": "董事会决议（容器：可能含重大收购，只换董事也可能）",
    "SHAREHOLDER_MEETING": "股东大会决议/召开（容器）",
    "INVESTOR_COMMUNICATION": "业绩说明会、投资者调研（容器，提取具体新信息）",
    "DISTRIBUTION": "权益分派、现金分红（除权除息标记/复权清洗/股息率特征）",
    "FINANCING_GUARANTEE": "融资、担保",
    "RELATED_TRANSACTION": "日常关联交易",
    "PRICE_ADJUSTMENT": "产品价格调整",
    "EQUITY_CHANGE": "权益变动（举牌等，未达 P0 程度）",
    "RISK_NOTICE": "风险提示类公告（非退市级）",
}
# tier P2：例行中性，保存原始记录 + 轻量分类，抽样作中性对照
TIER_P2 = {
    "MEETING_NOTICE": "会议召开通知",
    "RULE_REVISION": "公司制度修订（章程/议事规则/管理办法）",
    "PERSONNEL": "普通人员名单、聘任、辞职",
    "ROUTINE_RECORD": "无实质事项的会议记录",
    "LEGAL_OPINION": "法律意见书（无重大事件附随）",
    "MARKET_BROADCAST": "市场日常播报、行情转载",
    "REPUBLISH": "重复转载",
    "EQUITY_INCENTIVE_SELF_CHECK": "股权激励内幕信息自查报告（中性）",
    "ROUTINE_DISCLOSURE": "持续督导、募集说明书、H 股翌日披露等例行披露",
    "COMMODITY_MARKET_MOVE": "商品期货/收益率/外盘即时行情播报",
    "OTHER_NEUTRAL": "其他中性事项",
}

# 容器文档：标题只是容器，须读正文决定是否升级 P0
CONTAINER_TYPES = {"BOARD_RESOLUTION", "SHAREHOLDER_MEETING", "INVESTOR_COMMUNICATION"}

# ================= 自由文本 → 标准类别映射（关键词规则，顺序敏感：先 P0 后 P1 后 P2） =================
# (canonical, [关键词...])：event_type_raw 或标题命中任一关键词即映射
_RULES = [
    # ---- P0 ----
    ("DELISTING_RISK", ["退市", "终止上市", "ST", "重大违法", "面值退市", "摘牌"]),
    ("REGULATORY_PENALTY", ["立案", "处罚", "警示函", "违规", "监管函", "纪律处分", "行政处罚"]),
    ("EARNINGS_REVISION", ["修正", "下修", "业绩预告修正", "更正公告"]),
    ("EARNINGS_PREVIEW", ["业绩预告", "业绩快报", "预亏", "预增", "扭亏", "预降", "净利润"]),
    ("SHAREHOLDER_REDUCTION", ["减持"]),
    ("SHAREHOLDER_INCREASE", ["增持"]),
    ("BUYBACK", ["回购"]),
    ("SHARE_UNLOCK", ["解禁", "限售股上市"]),
    ("RESTRUCTURING", ["重组", "收购", "并购", "控制权", "易主", "借壳", "资产注入"]),
    ("MAJOR_CONTRACT", ["重大合同", "中标", "订单", "停产", "事故", "召回", "火灾", "爆炸"]),
    ("LITIGATION_DEFAULT", ["诉讼", "仲裁", "违约", "逾期", "债务"]),
    ("SANCTION_TRADE", ["制裁", "出口管制", "关税", "实体清单", "管制清单"]),
    ("INDUSTRY_POLICY", ["扶持政策", "产业政策", "行业政策", "规划", "补贴", "十四五", "双碳"]),
    ("MACRO_POLICY", ["降准", "降息", "货币政策", "财政政策", "LPR", "逆回购", "MLF"]),
    ("REGULATION_RELEASE", ["监督管理办法", "法规发布", "条例", "规章", "审议通过"]),
    # ---- P1 ----
    ("RISK_NOTICE", ["风险提示"]),
    ("DISTRIBUTION", ["权益分派", "利润分配", "现金分红", "分红", "派息", "除权", "除息"]),
    ("BOARD_RESOLUTION", ["董事会决议", "董事会公告"]),
    ("SHAREHOLDER_MEETING", ["股东大会", "股东会", "临时股东大会"]),
    ("INVESTOR_COMMUNICATION", ["业绩说明会", "投资者调研", "投资者关系", "路演"]),
    ("FINANCING_GUARANTEE", ["融资", "担保", "借款", "授信", "发行债券", "定增"]),
    ("RELATED_TRANSACTION", ["关联交易"]),
    ("PRICE_ADJUSTMENT", ["调价", "价格调整", "涨价", "降价"]),
    ("EQUITY_CHANGE", ["权益变动", "举牌"]),
    # ---- P2 ----
    ("EQUITY_INCENTIVE_SELF_CHECK", ["激励计划", "自查报告", "内幕信息知情人"]),
    ("ROUTINE_DISCLOSURE", ["持续督导", "募集说明书", "翌日披露", "跟踪报告"]),
    ("COMMODITY_MARKET_MOVE", ["期货", "夜盘", "收益率", "涨幅", "跌幅", "期市",
                               "涨超", "跌超", "价格变动", "触及"]),
    ("LEGAL_OPINION", ["法律意见书"]),
    ("MEETING_NOTICE", ["召开通知", "会议通知", "召集"]),
    ("RULE_REVISION", ["章程", "议事规则", "管理制度", "实施细则", "办法（修订", "制度修订"]),
    ("PERSONNEL", ["监事", "独立董事候选人", "聘任", "辞职", "换届"]),
    ("ROUTINE_RECORD", ["会议记录", "监事会决议", "确认意见"]),
    ("MARKET_BROADCAST", ["收盘", "开盘", "快讯", "播报", "涨幅扩大", "跌幅扩大"]),
    ("REPUBLISH", ["转载", "重复"]),
]
_RULES = [(c, kws) for c, kws in _RULES]

# P2 标题预路由：命中即不逐条调 LLM（抽取前粗筛，省算力）
P2_TITLE_PATTERNS = [
    r"召开.{0,6}(股东会|股东大会|董事会|监事会).{0,4}(通知|公告)",
    r"(董事会|监事会|股东大会).{0,4}决议(公告)?$",
    r"法律意见书",
    r"(章程|议事规则|管理制度|实施细则)",
    r"(监事|独立董事候选人|聘任|辞职|换届)",
    r"(收盘|开盘|涨幅扩大|跌幅扩大|市场播报)",
    r"内幕信息知情人买卖公司股票情况的自查报告",
    r"(持续督导|募集说明书|翌日披露表格|跟踪报告)",
    r"转载",
]
P2_TITLE_RE = [re.compile(p) for p in P2_TITLE_PATTERNS]

# 旧 v1 字典 → 标准类别（兼容历史数据）
LEGACY_EVENT_TYPE_MAP = {
    "立案处罚": "REGULATORY_PENALTY", "ST退市": "DELISTING_RISK",
    "减持": "SHAREHOLDER_REDUCTION", "增持回购": "SHAREHOLDER_INCREASE",
    "解禁": "SHARE_UNLOCK", "业绩预减": "EARNINGS_PREVIEW", "业绩预增": "EARNINGS_PREVIEW",
    "预告修正": "EARNINGS_REVISION", "问询函": "REGULATORY_PENALTY",
    "停复牌": "RISK_NOTICE", "龙虎榜异动": "MARKET_BROADCAST",
    "产业扶持政策": "INDUSTRY_POLICY", "监管政策": "INDUSTRY_POLICY",
    "宏观政策": "MACRO_POLICY", "海外政策制裁": "SANCTION_TRADE",
    "海外产业链事件": "MAJOR_CONTRACT", "大宗商品扰动": "PRICE_ADJUSTMENT",
    "地缘冲突": "SANCTION_TRADE",
}


def tier_of(canonical):
    if canonical in TIER_P0:
        return "P0"
    if canonical in TIER_P1:
        return "P1"
    return "P2"


def map_canonical(event_type_raw, title=""):
    """自由文本/标题 → (canonical_event_type 或 None, matched_keyword 或 None)。
    先查旧字典直译，再按 P0→P1→P2 关键词顺序匹配；映射不上返回 (None, None)。"""
    s = str(event_type_raw or "").strip()
    if s in LEGACY_EVENT_TYPE_MAP:
        return LEGACY_EVENT_TYPE_MAP[s], f"legacy字典直译: {s}"
    if s in TIER_P0 or s in TIER_P1 or s in TIER_P2:
        return s, None
    haystacks = [s, str(title or "")]
    for canon, kws in _RULES:
        for kw in kws:
            for h in haystacks:
                if kw in h:
                    return canon, f"关键词'{kw}'→{canon}"
    return None, None


def is_p2_by_title(title):
    """P2 标题预路由：True = 例行中性，可不调 LLM（抽样除外）。"""
    t = str(title or "")
    return any(r.search(t) for r in P2_TITLE_RE)


def underlying_hint(canonical):
    """容器文档的提示性底层事件族（仅提示，实际以正文抽取为准）。"""
    if canonical == "BOARD_RESOLUTION":
        return ["RESTRUCTURING", "MAJOR_CONTRACT", "FINANCING_GUARANTEE",
                "LITIGATION_DEFAULT", "DISTRIBUTION"]
    if canonical == "SHAREHOLDER_MEETING":
        return ["DISTRIBUTION", "RULE_REVISION", "EQUITY_CHANGE", "RESTRUCTURING"]
    if canonical == "INVESTOR_COMMUNICATION":
        return ["EARNINGS_PREVIEW", "EARNINGS_REVISION", "MAJOR_CONTRACT"]
    return []
