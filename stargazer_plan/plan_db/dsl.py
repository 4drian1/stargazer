"""确定性规则 DSL v1（plan_db/dsl.py，规格第 2 节）。

只读价量，确定性、无随机、无网络：

    price >= X            价格上穿（分钟收盘价）
    price <= X            价格下穿
    pct_chg <= -Y%        当日跌幅超过（相对昨收，单位 %）
    volume_ratio >= Z     分钟量/当日分钟均量（含当分钟的扩展均值，因果且确定性）
    time >= HH:MM         时间条件（常用于失效）
    and / or / not        组合，not 优先级最高，and 次之，or 最低

纯函数：同一份分钟输入 + 同一条规则必然产生同一个布尔序列，
边缘触发（False→True 跳变）时刻即 fired_at。
"""
import re

import pandas as pd

_TOKEN_RE = re.compile(r"\d{1,2}:\d{2}|-?\d+(?:\.\d+)?%?|>=|<=|[A-Za-z_]+|[^\s()]")

_FIELDS = ("price", "pct_chg", "volume_ratio", "time")


class DslError(ValueError):
    """规则 DSL 语法或语义错误。"""


def tokenize(text):
    toks, i = [], 0
    for m in _TOKEN_RE.finditer(text):
        if text[i:m.start()].strip():
            raise DslError("无法识别的字符: {!r}".format(text[i:m.start()]))
        toks.append(m.group(0))
        i = m.end()
    if text[i:].strip():
        raise DslError("无法识别的字符: {!r}".format(text[i:]))
    return toks


def parse(text):
    """解析 DSL 字符串为 AST 元组；语法非法抛 DslError。"""
    toks = tokenize(text)
    if not toks:
        raise DslError("空规则")
    pos = [0]

    def peek():
        return toks[pos[0]] if pos[0] < len(toks) else None

    def take():
        t = peek()
        pos[0] += 1
        return t

    def parse_or():
        node = parse_and()
        while peek() == "or":
            take()
            node = ("or", node, parse_and())
        return node

    def parse_and():
        node = parse_not()
        while peek() == "and":
            take()
            node = ("and", node, parse_not())
        return node

    def parse_not():
        if peek() == "not":
            take()
            return ("not", parse_not())
        return parse_atom()

    def parse_atom():
        if peek() == "(":
            take()
            node = parse_or()
            if take() != ")":
                raise DslError("缺少右括号")
            return node
        field = take()
        if field not in _FIELDS:
            raise DslError("非法字段: {!r}（可用 {}）".format(field, "/".join(_FIELDS)))
        op = take()
        if field in ("price", "volume_ratio") and op not in (">=", "<="):
            raise DslError("{} 只支持 >= / <=".format(field))
        if field == "pct_chg" and op != "<=":
            raise DslError("pct_chg 只支持 <=")
        if field == "time" and op != ">=":
            raise DslError("time 只支持 >=")
        val = take()
        if val is None:
            raise DslError("缺少阈值")
        if field == "time":
            h, m = (val.split(":") + ["0"])[:2] if ":" in val else (None, None)
            if h is None or not (0 <= int(h) <= 23 and 0 <= int(m) <= 59):
                raise DslError("非法时间: {!r}".format(val))
        else:
            try:
                float(str(val).rstrip("%"))
            except ValueError:
                raise DslError("非法数值: {!r}".format(val))
        return ("cond", field, op, val)

    node = parse_or()
    if pos[0] != len(toks):
        raise DslError("规则末尾有多余内容: {!r}".format(" ".join(toks[pos[0]:])))
    return node


def canonical(ast):
    """AST 规范化打印：同语义规则得到同一字符串，用于 inputs_hash。"""
    if ast[0] in ("and", "or"):
        return "{} {} {}".format(canonical(ast[1]), ast[0], canonical(ast[2]))
    if ast[0] == "not":
        inner = canonical(ast[1])
        return "not ({})".format(inner)
    return "{} {} {}".format(ast[1], ast[2], ast[3])


def evaluate(ast, minute_df, pre_close=None):
    """在分钟 DataFrame（dt/open/close/high/low/volume）上求布尔序列。

    pre_close 仅 pct_chg 规则需要（昨收）；缺失时报 DslError，绝不猜测。
    """
    kind = ast[0]
    if kind == "and":
        return evaluate(ast[1], minute_df, pre_close) & evaluate(ast[2], minute_df, pre_close)
    if kind == "or":
        return evaluate(ast[1], minute_df, pre_close) | evaluate(ast[2], minute_df, pre_close)
    if kind == "not":
        return ~evaluate(ast[1], minute_df, pre_close)
    _, field, op, val = ast
    if field == "price":
        s, x = minute_df["close"], float(val)
    elif field == "pct_chg":
        if not pre_close:
            raise DslError("pct_chg 规则需要 pre_close（昨收）")
        s = (minute_df["close"] / float(pre_close) - 1.0) * 100.0
        x = float(str(val).rstrip("%"))
    elif field == "volume_ratio":
        # 分钟量 / 当日分钟均量（含当分钟的扩展均值：只用当时及以前的信息，因果且确定性）
        s = minute_df["volume"] / minute_df["volume"].expanding().mean()
        x = float(val)
    else:  # time
        s = minute_df["dt"].dt.strftime("%H:%M")
        x = val
    return (s >= x) if op == ">=" else (s <= x)


def edge_fired_times(cond, minute_df):
    """边缘触发：条件由 False（或序列首）跳变为 True 的分钟序列。

    返回 list[pd.Timestamp]，即该规则的 fired_at 序列（纯函数，可重放）。
    """
    s = cond.fillna(False).astype(bool)
    edges = s & ~s.shift(1, fill_value=False)
    return [pd.Timestamp(t) for t in minute_df.loc[edges, "dt"]]
