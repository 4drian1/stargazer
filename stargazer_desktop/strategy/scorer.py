"""垃圾度打分(v2.1/v2.3): 十项指标池内百分位加权, 0-100分, 越低越烂越是胚子。
阈值: <40烂到根优选 / 40-60结合标签形态 / >60不够烂放弃。
风险旗标: 负债率>=90近资不抵债 / 营收YoY<=-50营收腰斩 / 净利率<=-50巨亏 / 流动比率<0.8短债压顶。"""
import numpy as np
import pandas as pd

WEIGHTS = {"ROE": 20, "资产负债率": 15, "净利率": 10, "经营现金流/股": 10, "营收YoY": 10,
           "归母净利YoY": 10, "PE": 10, "毛利率": 5, "EPS": 5, "流动比率": 5}
REVERSE = {"资产负债率"}  # 越高越差


def _pe_score(pe):
    if pe is None or (isinstance(pe, float) and np.isnan(pe)):
        return 50
    if pe < 0: return 0
    if pe > 80: return 20
    if pe > 40: return 45
    if pe > 20: return 70
    return 100


def score_pool(pool: pd.DataFrame):
    """pool: DataFrame, 索引=代码, 列=九项指标+PE。返回加了'总分'和'旗标'的副本, 按总分升序。"""
    df = pool.copy()
    comp = pd.DataFrame(index=df.index)
    for col in WEIGHTS:
        if col == "PE":
            comp["PE"] = [_pe_score(p) for p in df["PE"]]
            continue
        s = pd.to_numeric(df[col], errors="coerce")
        r = s.rank(pct=True, na_option="keep") * 100
        r = r.fillna(50)
        comp[col] = (100 - r) if col in REVERSE else r
    df["总分"] = round(sum(comp[k] * w for k, w in WEIGHTS.items()) / 100, 1)

    def flags(row):
        f = []
        if pd.notna(row.get("资产负债率")) and row["资产负债率"] >= 90: f.append("近资不抵债")
        if pd.notna(row.get("营收YoY")) and row["营收YoY"] <= -50: f.append("营收腰斩")
        if pd.notna(row.get("净利率")) and row["净利率"] <= -50: f.append("巨亏")
        if pd.notna(row.get("流动比率")) and row["流动比率"] < 0.8: f.append("短债压顶")
        return "/".join(f)
    df["旗标"] = df.apply(flags, axis=1)
    return df.sort_values("总分")
