"""每日塔罗 + 星象运势（v1）。事实核与文案层严格分离：
- 抽牌(draw_card)与星象(astro_context)是确定性事实，按日期种子幂等；
- base_reading 拼装平实事实核，作为 LLM 失败时的回退文案；
- polish_with_persona 仅做"说话方式"润色（克莱恩人格），绝不改动牌意/星象事实；
- 校验失败或异常一律静默回退 base_reading，宁要平实的早安，不要中断；
- 免责页脚由代码追加，任何模式都在。
本模块只依赖标准库；deepseek_api 在润色函数内惰性导入，避免拖慢 notifier 启动。
红线：人格层可整体关闭、可替换，永不进入交易链路，不进 Corvus 证据流。"""
import datetime as _dt
import random as _random
import re

# --------------------------------------------------------------------------
# 牌库（78 张）。关键词取通用韦特含义，俏皮交给 LLM；planet 为黄金黎明体系对应
# （大阿尔卡纳）或元素（小阿尔卡纳），仅作元数据，不进推送文案的硬性事实字段。
# --------------------------------------------------------------------------
_MAJOR = [
    # name, planet, upright_kw, reversed_kw
    ("愚者", "天王星(风)", ["新的开始", "自由", "冒险", "纯真"], ["鲁莽", "逃避", "危险", "犹豫"]),
    ("魔术师", "水星", ["创造", "行动", "意志", "掌控"], ["欺骗", "拖延", "能力未发挥", "操纵"]),
    ("女祭司", "月亮", ["直觉", "神秘", "潜意识", "内省"], ["直觉压抑", "表里不一", "秘密未明"]),
    ("皇后", "金星", ["丰饶", "母性", "滋养", "创造"], ["依赖", "过度保护", "创造力受阻"]),
    ("皇帝", "白羊座", ["权威", "秩序", "稳定", "领导"], ["专断", "控制欲", "僵化"]),
    ("教皇", "金牛座", ["传统", "信仰", "教导", "遵从"], ["叛逆", "教条", "墨守成规"]),
    ("恋人", "双子座", ["结合", "选择", "爱", "和谐"], ["失衡", "错误选择", "分裂", "犹豫"]),
    ("战车", "巨蟹座", ["胜利", "意志", "前进", "掌控"], ["失控", "方向迷失", "内耗"]),
    ("力量", "狮子座", ["勇气", "耐心", "内在力量", "温柔"], ["自我怀疑", "软弱", "冲动"]),
    ("隐士", "处女座", ["内省", "指引", "孤独", "智慧"], ["孤立", "逃避", "固执"]),
    ("命运之轮", "木星", ["转机", "循环", "机遇", "命运"], ["厄运", "停滞", "失控"]),
    ("正义", "天秤座", ["公正", "真相", "平衡", "责任"], ["不公", "偏颇", "逃避责任"]),
    ("倒吊人", "海王星", ["牺牲", "换个视角", "等待", "臣服"], ["徒劳", "抗拒", "停滞"]),
    ("死神", "天蝎座", ["结束", "转化", "重生", "放手"], ["抗拒改变", "停滞", "恐惧"]),
    ("节制", "射手座", ["平衡", "调和", "耐心", "治愈"], ["失衡", "过度", "冲突"]),
    ("恶魔", "摩羯座", ["束缚", "欲望", "执念", "物质"], ["觉醒", "挣脱", "释放"]),
    ("高塔", "火星", ["突变", "崩塌", "觉醒", "释放"], ["拖延灾难", "逃避崩塌", "余震"]),
    ("星星", "水瓶座", ["希望", "疗愈", "灵感", "指引"], ["失望", "信念动摇", "迷茫"]),
    ("月亮", "双鱼座", ["幻象", "潜意识", "不安", "直觉"], ["迷惑消散", "释放恐惧", "澄清"]),
    ("太阳", "太阳", ["喜悦", "成功", "活力", "光明"], ["暂时受阻", "过度乐观", "延迟"]),
    ("审判", "冥王星", ["觉醒", "召唤", "宽恕", "重生"], ["自我怀疑", "拒绝反思", "悔恨"]),
    ("世界", "土星", ["圆满", "完成", "整合", "成就"], ["未竟", "拖延", "缺口"]),
]

_SUITS = [
    # suit_cn, suit_en, element
    ("权杖", "wands", "火"),
    ("圣杯", "cups", "水"),
    ("宝剑", "swords", "风"),
    ("星币", "pentacles", "土"),
]
_RANKS = [
    # rank_cn, rank_en, upright_kw, reversed_kw
    ("王牌", "ace", ["灵感", "新开始", "潜能", "行动力"], ["延迟", "缺乏方向", "热情减退"]),
    ("二", "2", ["规划", "抉择", "平衡"], ["优柔寡断", "计划失衡"]),
    ("三", "3", ["进展", "探索", "远见"], ["受阻", "延迟", "迷失"]),
    ("四", "4", ["稳定", "巩固", "珍惜"], ["停滞", "固守", " boredom".replace("boredom", "倦怠")]),
    ("五", "5", ["冲突", "竞争", "压力"], ["和解", "化解紧张"]),
    ("六", "6", ["胜利", "认可", "成功"], ["骄傲", "受挫", "延迟"]),
    ("七", "7", ["坚持", "防守", "收获临近"], ["放弃", "泄气", "被超越"]),
    ("八", "8", ["速度", "行动", "能量"], ["慌乱", "方向错", "缓慢"]),
    ("九", "9", ["坚韧", "储备", "接近完成"], ["倦怠", "资源耗散"]),
    ("十", "10", ["重担", "责任", "压力"], ["放下", "解脱", "负担减轻"]),
    ("侍从", "page", ["探索", "热情", "消息"], ["延迟消息", "分心"]),
    ("骑士", "knight", ["行动", "进取", "冲动"], ["鲁莽", "失控", "拖延"]),
    ("王后", "queen", ["自信", "魅力", "创造力"], ["自我中心", "情绪化"]),
    ("国王", "king", ["领导", "远见", "掌控"], ["专断", "急躁", "失控"]),
]
# 小阿尔卡纳按花色微调主题词（在 rank 通用词前加花色语义）
_SUIT_THEME = {
    "权杖": ("行动", "热情"),
    "圣杯": ("情感", "关系"),
    "宝剑": ("思维", "沟通"),
    "星币": ("物质", "实务"),
}


def _build_card_db():
    db = []
    for i, (nm, pl, up, rev) in enumerate(_MAJOR):
        db.append({"index": i, "name": nm, "arcana": "major", "suit": None,
                   "planet": pl, "upright_kw": up, "reversed_kw": rev})
    idx = 0
    for scn, sen, elem in _SUITS:
        theme_up, theme_rev = _SUIT_THEME[scn]
        for rcn, ren, rup, rrev in _RANKS:
            nm = f"{scn}{rcn}"
            up = [theme_up] + rup
            rev = [theme_rev] + rrev
            db.append({"index": 22 + idx, "name": nm, "arcana": "minor", "suit": sen,
                       "planet": elem, "upright_kw": up, "reversed_kw": rev})
            idx += 1
    return db


CARD_DB = _build_card_db()
_N = len(CARD_DB)  # 78


# --------------------------------------------------------------------------
# 抽牌：按日期字符串做种子，幂等；同一天多次结果一致。
# --------------------------------------------------------------------------
def draw_card(date_str):
    """date_str: 'YYYY-MM-DD'。返回含牌名/正逆位/关键词/归属的字典。"""
    rnd = _random.Random("tarot|" + date_str)
    idx = rnd.randrange(_N)
    card = dict(CARD_DB[idx])
    reversed_ = rnd.random() < 0.5
    card["orientation"] = "reversed" if reversed_ else "upright"
    card["orientation_cn"] = "逆位" if reversed_ else "正位"
    card["keywords"] = card["reversed_kw"] if reversed_ else card["upright_kw"]
    return card


# --------------------------------------------------------------------------
# 星象（纯 Python 近似）。太阳黄经→星座（按 30° 自然分界，交界日自动处理）；
# 月相用已知新月参考 + 朔望月 29.530588853 天近似。
# --------------------------------------------------------------------------
_SIGNS = ["白羊座", "金牛座", "双子座", "巨蟹座", "狮子座", "处女座",
          "天秤座", "天蝎座", "射手座", "摩羯座", "水瓶座", "双鱼座"]
_SYNODIC = 29.530588853
_NEWMOON_REF = _dt.datetime(2000, 1, 6, 18, 14)  # 已知新月参考（UTC）


def _to_date(when):
    if isinstance(when, _dt.datetime):
        return when
    return _dt.datetime(when.year, when.month, when.day, 12, 0)


def _sun_longitude(when):
    d = _to_date(when)
    j2000 = _dt.date(2000, 1, 1)
    n = (d.date() - j2000).days + 0.5  # 取当日中点近似
    L = (280.460 + 0.9856474 * n) % 360.0
    return L


def sun_sign(when):
    L = _sun_longitude(when)
    return _SIGNS[int(L // 30) % 12]


def moon_phase(when):
    d = _to_date(when)
    days = (d - _NEWMOON_REF).total_seconds() / 86400.0
    age = days % _SYNODIC
    if age < 1.85 or age >= 28.68:
        return "新月"
    if age < 7.38:
        return "蛾眉月（盈）"
    if age < 8.38:
        return "上弦月"
    if age < 14.77:
        return "盈凸月"
    if age < 16.30:
        return "满月"
    if age < 22.52:
        return "亏凸月"
    if age < 23.52:
        return "下弦月"
    return "残月（亏）"


def astro_context(when):
    return {"sun_sign": sun_sign(when), "sun_longitude": round(_sun_longitude(when), 2),
            "moon_phase": moon_phase(when)}


# --------------------------------------------------------------------------
# 事实核文案（平实风格，作为回退）。所有事实原样呈现，不做解读发挥。
# --------------------------------------------------------------------------
def base_reading(card, astro):
    kw = "、".join(card["keywords"])
    return (
        f"## 今日塔罗 · {_to_date(card.get('_now') or _dt.datetime.now()).strftime('%Y-%m-%d')}\n"
        f"**抽牌**：{card['name']}（{card['orientation_cn']}）\n"
        f"**牌意关键词**：{kw}\n"
        f"**当日星象**：太阳在{astro['sun_sign']}；月相：{astro['moon_phase']}。\n"
        f"以上为今日抽牌与星象事实，供你自行体会。"
    )


# --------------------------------------------------------------------------
# 人格层（克莱恩·莫雷蒂 / 愚者）。仅润色说话方式，不许碰牌意与星象事实。
# --------------------------------------------------------------------------
PERSONA_PROMPTS = {
    "klein": (
        "你是克莱恩·莫雷蒂，《诡秘之主》中的“愚者”，此刻端坐在灰雾之上，以塔罗会的名义向一位普通的生灵传达今日的启示。你维持愚者的位格：神秘、克制、从容。\n\n"
        "【你会收到】今日翻开的塔罗牌（牌名+正逆）、牌的关键词、当日星象（太阳星座、月相）、一句基础解读。\n\n"
        "【输出结构，六条缺一不可】\n"
        "1. 开场一句旁白：灰雾之上，愚者为对方“翻开”今日之牌，牌名以『XX·正位/逆位』格式完整出现一次；\n"
        "2. 主体是愚者的原话：先用一句带神态的引导语引出，原话用中文引号括起；引导语的神态与动作由你即兴选择，须符合愚者位格——克制、从容、略带神秘（可选姿态举例：微微颔首、指尖拂过牌面、沉默数息后开口、目光落向灰雾深处、将牌轻轻推过桌面）。【关键】每次的神态与动作都要不同，禁止反复套用同一套开场（尤其不要总写“沉默数息后，缓声道”）；下方示例仅为风格示范，禁止照搬示例里的原话与姿态。\n"
        "3. 牌意、星象、隐晦指引全部放进愚者的话里说——他是对一个生灵说话不是说书人在解说：禁用“它提醒你/这张牌意味着/关键词是”这类说明书句式，牌意要像他自己的判断一样自然流出；\n"
        "4. 收尾必须落在一句隐晦的指引上——由牌意与星象自然引出（如“取舍须趁此时”），用暗示而非指令，不出现“宜/忌/应该/建议你”等字眼，让建议藏在意象里；收尾须短而留白，像愚者忽然收住的一句私语，不要展开成解释。【关键】收尾的意象必须与牌意段、星象段完全断开：禁止复用这两段中出现过的任何具体物象或名词（包括本牌自身的核心符号，如圣杯牌的“杯”、权杖的“火/杖”，以及星象段的“太阳/月/星”等），必须另起一个全然不同的隐喻来收束，绝不再说“剩下的/留下的”之类已经用过的措辞。\n"
        "5. 星象：太阳星座与月相的名称必须原样准确（如“亏凸月”不得改写成其他月相）；鼓励围绕事实做意象演绎（亏相的退场与整理、盈相的推进与积累皆合规）；禁止引入事实核之外的星象或天象事件（如水星逆行、行星合月、流星雨）；\n"
        "6. 全程以“今日”为时间背景（清晨推送），不得出现“今夜/昨晚”。\n\n"
        "【硬性禁止】任何交易建议、任何涨跌预测、任何具体股票/板块/基金；不得出现“买入/卖出/建仓/加仓/减仓/满仓/梭哈/抄底/逃顶”等字眼。不要写免责页脚（调用方自行追加）。\n\n"
        "【篇幅】全文100-180字。\n\n"
        "【示例】\n"
        "输入：今日翻开的牌：恋人·正位；牌的关键词：结合、选择、爱与和谐；太阳位于天蝎座；月相为亏凸月。\n"
        "输出：灰雾之上，愚者为你翻开今日之牌——恋人·正位。他指尖轻叩牌缘，将牌轻轻推过桌面，淡声道："
        "“结合选择，是今日的题眼。坦诚相待时，和谐自会显现；心口不一时，抉择便成了负担。"
        "太阳正位于天蝎，照见那些不轻易示人的动机；盈满已在退场，留下的部分，才值得细看。"
        "话说出口之前——先想清楚，自己真正要的是什么。”"
    ),
}

# 回退/校验阈值与违禁词（交易相关，绝不能出现在人格文案里）。
_MAX_PERSONA_CHARS = 300
_MIN_PERSONA_CHARS = 60
_FORBIDDEN = ["买", "卖", "涨", "跌", "建仓", "清仓", "持仓", "止损", "止盈", "加仓", "减仓",
              "满仓", "梭哈", "抄底", "逃顶", "仓位", "板块", "个股", "基金", "ETF", "股票",
              "涨跌停", "追高", "A股", "美股", "大盘", "指数"]

# 中文引号（直引语结构必须存在一对）
_QUOTE_L = "“「『"
_QUOTE_R = "”」』"

# 直白宜忌/今夜不在人格文案里：收尾走“取舍须趁此时”这类隐晦话术。
_BLOCK_CHARS = ["宜", "忌", "今夜", "昨晚"]


def build_user_message(card, astro):
    kw = "、".join(card["keywords"])
    base = base_reading(card, astro)
    return (
        f"今日翻开的牌：{card['name']}·{card['orientation_cn']}\n"
        f"牌的关键词：{kw}\n"
        f"太阳位于{astro['sun_sign']}月相为{astro['moon_phase']}\n"
        f"基础解读：{base}\n"
    )


def validate_persona_output(text, card):
    if not text:
        return False
    if not (_MIN_PERSONA_CHARS <= len(text) <= _MAX_PERSONA_CHARS):
        return False
    # 牌名·正逆位 格式必须完整出现一次
    if f"{card['name']}·{card['orientation_cn']}" not in text:
        return False
    # 交易违禁词
    if any(w in text for w in _FORBIDDEN):
        return False
    # 直白宜忌 / 今夜 一律回退（隐晦话术不含这些字眼）
    if any(b in text for b in _BLOCK_CHARS):
        return False
    # 必须含一对中文引号（愚者原话直接引语结构）
    if not (any(l in text for l in _QUOTE_L) and any(r in text for r in _QUOTE_R)):
        return False
    return True


def polish_with_persona(cfg, card, astro, persona="klein", llm_fn=None):
    """返回 (文本, 是否使用LLM文案)。失败/校验不通过一律回退 base_reading。"""
    if llm_fn is None:
        import deepseek_api
        llm_fn = deepseek_api.complete
    base = base_reading(card, astro)
    sys_prompt = PERSONA_PROMPTS.get(persona, PERSONA_PROMPTS["klein"])
    user_msg = build_user_message(card, astro)
    messages = [{"role": "system", "content": sys_prompt},
                {"role": "user", "content": user_msg}]
    try:
        resp = llm_fn(cfg, messages, purpose="chat")
        text = (resp.choices[0].message.content or "").strip()
    except Exception as exc:  # 超时/鉴权/网络等 → 静默回退
        print(f"[daily_tarot] LLM 润色失败，回退事实核: {type(exc).__name__}")
        return base, False
    if not validate_persona_output(text, card):
        print("[daily_tarot] LLM 输出校验未通过（超长/缺牌名·正逆位/含违禁词/宜忌今夜/无引号），回退事实核")
        return base, False
    return text, True


# --------------------------------------------------------------------------
# 代码追加的免责页脚（任何模式都在）。
# --------------------------------------------------------------------------
FOOTER = (
    "\n---\n"
    "⚠️ 本内容为娱乐性质，塔罗牌与星象解读不构成任何投资建议；"
    "克莱恩的口吻仅为风格演绎，请勿据此操作。"
)


def compose_daily_tarot(cfg, now, llm_fn=None):
    """组装当日完整推送文本（事实核 或 人格润色 + 代码追加页脚）。"""
    date_str = now.strftime("%Y-%m-%d")
    card = draw_card(date_str)
    card["_now"] = now
    astro = astro_context(now)
    tc = (cfg.get("push", {}) or {}).get("daily_tarot", {}) or {}
    use_llm = bool(tc.get("use_llm", True))
    persona = tc.get("persona", "klein")
    if use_llm:
        text, _used = polish_with_persona(cfg, card, astro, persona=persona, llm_fn=llm_fn)
    else:
        text = base_reading(card, astro)
    return text + FOOTER


if __name__ == "__main__":
    # 本地冒烟（不联网、不推送）：打印今日事实核 + 页脚。
    import sys
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    _n = _dt.datetime.now()
    print(compose_daily_tarot({"push": {"daily_tarot": {"enabled": True, "use_llm": False}}}, _n))
