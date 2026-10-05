"""事件分类法（自包含，规则照抄 corvus/corvus/event_taxonomy.py 2026-09-12 冻结 v1.0。

偏离 corvus 处（见 DEVIATIONS.md）：
- corvus 用 LLM 抽正文后对 P1 容器文档升级 P0；本包无 LLM，分类只依据标题关键词，
  P1 容器不升级，保留 tier 供训练过滤。
- DISTRIBUTION（权益分派）corvus 归公司行为特征；本包保留类别但在 material 集合中剔除
  （除权噪声，不当作信息事件训练）。
"""
from __future__ import annotations

import re

TIER_P0 = {
    "REGULATORY_PENALTY": "监管处罚与立案",
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
    "REGULATION_RELEASE": "监管法规发布",
}
TIER_P1 = {
    "BOARD_RESOLUTION": "董事会决议（容器）",
    "SHAREHOLDER_MEETING": "股东大会决议/召开（容器）",
    "INVESTOR_COMMUNICATION": "业绩说明会、投资者调研（容器）",
    "DISTRIBUTION": "权益分派、现金分红",
    "FINANCING_GUARANTEE": "融资、担保",
    "RELATED_TRANSACTION": "日常关联交易",
    "PRICE_ADJUSTMENT": "产品价格调整",
    "EQUITY_CHANGE": "权益变动（举牌等）",
    "RISK_NOTICE": "风险提示类公告（非退市级）",
}
TIER_P2 = {
    "MEETING_NOTICE": "会议召开通知",
    "RULE_REVISION": "公司制度修订",
    "PERSONNEL": "普通人员名单、聘任、辞职",
    "ROUTINE_RECORD": "无实质事项的会议记录",
    "LEGAL_OPINION": "法律意见书（无重大事件附随）",
    "MARKET_BROADCAST": "市场日常播报、行情转载",
    "REPUBLISH": "重复转载",
    "EQUITY_INCENTIVE_SELF_CHECK": "股权激励内幕信息自查报告（中性）",
    "ROUTINE_DISCLOSURE": "持续督导、募集说明书、H股翌日披露等例行披露",
    "COMMODITY_MARKET_MOVE": "商品期货/收益率/外盘即时行情播报",
    "OTHER_NEUTRAL": "其他中性事项",
}

_RULES = [
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
    ("RISK_NOTICE", ["风险提示"]),
    ("DISTRIBUTION", ["权益分派", "利润分配", "现金分红", "分红", "派息", "除权", "除息"]),
    ("BOARD_RESOLUTION", ["董事会决议", "董事会公告"]),
    ("SHAREHOLDER_MEETING", ["股东大会", "股东会", "临时股东大会"]),
    ("INVESTOR_COMMUNICATION", ["业绩说明会", "投资者调研", "投资者关系", "路演"]),
    ("FINANCING_GUARANTEE", ["融资", "担保", "借款", "授信", "发行债券", "定增"]),
    ("RELATED_TRANSACTION", ["关联交易"]),
    ("PRICE_ADJUSTMENT", ["调价", "价格调整", "涨价", "降价"]),
    ("EQUITY_CHANGE", ["权益变动", "举牌"]),
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

# 关键词 -> 方向感（规则基线用；人工粗表，仅作基线不作结论）
KEYWORD_DIRECTION = {
    "预增": +1, "扭亏": +1, "中标": +1, "回购": +1, "增持": +1,
    "收购": +1, "资产注入": +1, "补贴": +1, "涨价": +1,
    "预亏": -1, "预降": -1, "减持": -1, "退市": -1, "立案": -1,
    "处罚": -1, "诉讼": -1, "违约": -1, "逾期": -1, "停产": -1,
    "下修": -1, "警示函": -1, "监管函": -1, "解禁": -1, "降价": -1,
    "退市风险": -1, "终止上市": -1, "火灾": -1, "爆炸": -1, "事故": -1,
}
# 规则基线权重（materiality 先验强度粗表）
KEYWORD_MATERIAL_WEIGHT = {
    "退市": 3.0, "终止上市": 3.0, "立案": 2.5, "重组": 2.0, "收购": 1.5,
    "并购": 1.5, "业绩预告": 1.5, "业绩快报": 1.5, "预增": 1.5, "预亏": 1.5,
    "扭亏": 1.5, "中标": 1.2, "减持": 1.2, "增持": 1.0, "回购": 1.0,
    "解禁": 1.0, "诉讼": 1.2, "违约": 1.5, "逾期": 1.2, "停产": 1.5,
    "处罚": 2.0, "警示函": 1.5, "监管函": 1.5, "火灾": 1.5, "爆炸": 1.5,
    "事故": 1.2, "修正": 1.2, "下修": 1.5, "控制权": 1.5, "借壳": 2.0,
}


def tier_of(canonical: str) -> str:
    if canonical in TIER_P0:
        return "P0"
    if canonical in TIER_P1:
        return "P1"
    return "P2"


def map_canonical(title: str):
    """标题 -> (canonical, matched_keyword)。映射不上 -> ('UNMAPPED', None)。"""
    t = str(title or "")
    for canon, kws in _RULES:
        for kw in kws:
            if kw in t:
                return canon, kw
    return "UNMAPPED", None


def board_of(code: str) -> str:
    """板块代理特征（stockdb 无行业字段，见 DEVIATIONS.md）。"""
    c = str(code)
    if c.startswith(("300", "301", "302")):
        return "CYB"
    if c.startswith("688"):
        return "KCB"
    if c.startswith(("4", "8", "92")):
        return "BSE"
    if c.startswith("6"):
        return "MAIN_SH"
    return "MAIN_SZ"


def title_keyword_features(title: str):
    """标题关键词命中向量（规则基线 + LR/LGBM 共享特征源）。"""
    t = str(title or "")
    return {kw: 1.0 for kw in KEYWORD_MATERIAL_WEIGHT if kw in t}
