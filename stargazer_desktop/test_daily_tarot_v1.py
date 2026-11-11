# -*- coding: utf-8 -*-
"""daily_tarot_v1 测试清单：
- 抽牌确定性（同日幂等）与 365 天分布均匀性；
- 星象：太阳星座（交界日自然分界）+ 月相有效；
- 事实核含全部事实字段；
- 提示词组装：事实全部注入、密钥/银标不入；
- 五种 mock LLM 路径：正常/超长/缺牌名/含违禁词/抛异常，仅正常用 LLM 文案；
- 工作日门控 + 当日仅触发一次（不依赖网络，_deliver 被替换记录）。
"""
import datetime as dt
import json

import daily_tarot_v1 as T
import notifier


def _fake_resp(text):
    msg = type("M", (), {"content": text})()
    ch = type("C", (), {"message": msg})()
    return type("R", (), {"choices": [ch]})()


class _MockLLM:
    def __init__(self, mode, card):
        self.mode = mode
        self.card = card
        self.calls = 0

    def _dot(self):
        return f"{self.card['name']}·{self.card['orientation_cn']}"

    def __call__(self, cfg, messages, purpose="chat"):
        self.calls += 1
        # 模拟 DeepSeek 返回（六段式：开场+愚者原话中文引号、牌名·正逆位、今日、无今夜/宜忌）
        if self.mode == "normal":
            return _fake_resp(
                f"灰雾之上，愚者为你翻开今日之牌——{self._dot()}。"
                f"他微微颔首，目光落向灰雾深处，淡声道：“结合选择，是今日的题眼。"
                f"太阳位于天蝎，月相为亏凸月，照见那些不轻易示人的动机；"
                f"盈满已在退场，留下的部分，才值得细看。"
                f"话说出口之前——先想清楚，自己真正要的是什么。”")
        if self.mode == "no_quotes":  # 缺中文引号直引语结构（长度够，仅此一项不达标）
            return _fake_resp(
                f"灰雾之上，愚者为你翻开今日之牌——{self._dot()}。"
                f"结合选择是今日的题眼，太阳位于天蝎座，月相为亏凸月；"
                f"盈满已在退场，留下的部分才值得细看，取舍须趁此时。")
        if self.mode == "long":  # 超 300 字
            return _fake_resp(self._dot() + "。" + ("今日之牌静静陈列于灰雾之上。" * 25))
        if self.mode == "missing_name":  # 不含牌名·正逆位
            return _fake_resp("灰雾之上的愚者说，今日之牌静静陈列，太阳位于天蝎座，月相为亏凸月。")
        if self.mode == "forbidden":  # 含交易违禁词（满仓/抄底/板块）
            return _fake_resp(
                f"灰雾之上，愚者为你翻开今日之牌——{self._dot()}，建议满仓抄底相关板块。")
        if self.mode == "raise":
            raise RuntimeError("simulated network error")
        raise AssertionError("unknown mode")


results = []


def chk(name, cond):
    results.append((name, bool(cond)))
    print(("PASS " if cond else "FAIL ") + name)


# 1) 抽牌确定性 + 分布
c1 = T.draw_card("2026-01-01")
c2 = T.draw_card("2026-01-01")
chk("draw_card_deterministic", c1 == c2 and c1["name"] == c2["name"]
    and c1["orientation"] == c2["orientation"])

# 365 天分布（从 2026-01-01 起）
counts = {}
rev = 0
start = dt.date(2026, 1, 1)
for i in range(365):
    d = (start + dt.timedelta(days=i)).strftime("%Y-%m-%d")
    c = T.draw_card(d)
    counts[c["name"]] = counts.get(c["name"], 0) + 1
    if c["orientation"] == "reversed":
        rev += 1
missing = [nm for nm in {x["name"] for x in T.CARD_DB} if counts.get(nm, 0) == 0]
maxc = max(counts.values())
chk("draw_distribution_no_dominant", maxc <= 12)
chk("draw_distribution_coverage", len(missing) <= 8)
chk("draw_reversed_ratio_reasonable", 0.35 <= rev / 365 <= 0.65)
chk("card_db_has_78", len(T.CARD_DB) == 78)

# 2) 星象
n = dt.datetime(2026, 8, 1)   # 狮子座区间
chk("sun_sign_leo", T.sun_sign(n) == "狮子座")
c = dt.datetime(2026, 1, 15)  # 摩羯座区间
chk("sun_sign_capricorn", T.sun_sign(c) == "摩羯座")
ph = T.moon_phase(dt.datetime(2026, 10, 30))
chk("moon_phase_valid", ph in {"新月", "蛾眉月（盈）", "上弦月", "盈凸月",
                               "满月", "亏凸月", "下弦月", "残月（亏）"})
ac = T.astro_context(n)
chk("astro_context_keys", set(ac) == {"sun_sign", "sun_longitude", "moon_phase"})

# 3) 事实核含事实
card = T.draw_card("2026-01-01")
ac2 = T.astro_context(n)
base = T.base_reading(card, ac2)
chk("base_reading_has_name", card["name"] in base)
chk("base_reading_has_orientation", card["orientation_cn"] in base)
chk("base_reading_has_sun", ac2["sun_sign"] in base)
chk("base_reading_has_moon", ac2["moon_phase"] in base)

# 4) 提示词组装
um = T.build_user_message(card, ac2)
chk("prompt_open_card", "今日翻开的牌" in um)
chk("prompt_dot_format_in_input", f"{card['name']}·{card['orientation_cn']}" in um)
chk("prompt_plain_astro", "太阳位于" in um and "月相为" in um)
chk("prompt_has_name", card["name"] in um)
chk("prompt_has_orientation", card["orientation_cn"] in um)
chk("prompt_has_sun", ac2["sun_sign"] in um)
chk("prompt_has_moon", ac2["moon_phase"] in um)
chk("prompt_no_secret", "sk-" not in um and "dingtalk" not in um and "SECRET" not in um)
chk("persona_no_secret", "sk-" not in T.PERSONA_PROMPTS["klein"]
    and "dingtalk" not in T.PERSONA_PROMPTS["klein"])
# 提示词内含五段结构规则：牌名·正逆位格式、今夜禁令、星象平实、收尾隐晦
chk("prompt_dot_format_rule", "·正位" in T.PERSONA_PROMPTS["klein"])
chk("prompt_no_jinye_rule", "今夜" in T.PERSONA_PROMPTS["klein"])
chk("prompt_astro_plain_rule", "太阳星座与月相的名称必须原样准确" in T.PERSONA_PROMPTS["klein"])
chk("prompt_ending_implicit", "取舍须趁此时" in T.PERSONA_PROMPTS["klein"])

# 5) 六种 mock LLM 路径（正常/无引号/超长/缺牌名/含违禁词/抛异常）
cfg = {"push": {"daily_tarot": {"enabled": True, "time": "09:00", "use_llm": True}}}
for mode, expect_used in [("normal", True), ("no_quotes", False), ("long", False),
                          ("missing_name", False), ("forbidden", False), ("raise", False)]:
    mock = _MockLLM(mode, card)
    out, used = T.polish_with_persona(cfg, card, ac2, persona="klein", llm_fn=mock)
    chk(f"llm_path_{mode}_used_llm={expect_used}", used is expect_used)
    chk(f"llm_path_{mode}_output_nonempty", bool(out) and card["name"] in out)
    if not expect_used:
        # 回退路径必须精确返回事实核文案（含牌名），且不依赖 LLM
        chk(f"llm_path_{mode}_falls_back_to_base", out == T.base_reading(card, ac2))

# 5b) 六段结构 / 风格断言（针对正常 LLM 文案）
norm_mock = _MockLLM("normal", card)
out_norm, used_norm = T.polish_with_persona(cfg, card, ac2, persona="klein", llm_fn=norm_mock)
chk("style_dot_format_present", f"{card['name']}·{card['orientation_cn']}" in out_norm)
chk("style_opening", "今日之牌" in out_norm)                  # 开场翻开句
chk("style_has_quotes", "“" in out_norm and "”" in out_norm)    # 愚者原话中文引号直引语
chk("style_closing_guidance", "要的是什么" in out_norm)        # 收尾隐晦指引
chk("style_uses_jinri", "今日" in out_norm)
chk("style_no_jinye", "今夜" not in out_norm and "昨晚" not in out_norm)
chk("style_no_textbook", "它提醒" not in out_norm)             # 禁用说明书句式
chk("style_moon_intact", "亏凸月" in out_norm)                 # 星象名称原样未被改写
# 宜/忌/今夜已纳入回退拦截：良好输出应自然避开这些字眼
chk("style_no_yiji", "宜" not in out_norm and "忌" not in out_norm)
# 含交易违禁词（满仓/抄底/板块）必须回退
fb_mock = _MockLLM("forbidden", card)
out_fb, used_fb = T.polish_with_persona(cfg, card, ac2, persona="klein", llm_fn=fb_mock)
chk("forbidden_trade_words_fallback", used_fb is False and out_fb == T.base_reading(card, ac2))

# 6) 工作日门控 + 当日仅触发一次
# 找一个周六与一个周一
sat = None
mon = None
d0 = dt.date(2026, 11, 1)
for i in range(10):
    cur = d0 + dt.timedelta(days=i)
    if cur.weekday() == 5 and sat is None:
        sat = cur
    if cur.weekday() == 0 and mon is None:
        mon = cur
rec = []


def fake_deliver(self, cfg, title, text, kind):
    rec.append((title, kind, text))


notifier.Notifier._deliver = fake_deliver
ntf = notifier.Notifier(lambda: cfg)
# 周末不触发
ntf._maybe_daily_tarot(cfg, dt.datetime(sat.year, sat.month, sat.day, 9, 0))
chk("weekend_no_fire", len(rec) == 0)
# 工作日触发且仅一次（09:00 到点）
ntf._maybe_daily_tarot(cfg, dt.datetime(mon.year, mon.month, mon.day, 9, 0))
ntf._maybe_daily_tarot(cfg, dt.datetime(mon.year, mon.month, mon.day, 9, 0))  # 同日二次
chk("weekday_fires_once", len(rec) == 1 and rec[0][1] == "daily_tarot")
chk("delivered_has_footer", rec and "不构成任何投资建议" in rec[0][2])

n_fail = sum(1 for _, ok in results if not ok)
print("\nDAILY_TAROT TEST: %d passed, %d failed" % (len(results) - n_fail, n_fail))
raise SystemExit(1 if n_fail else 0)
