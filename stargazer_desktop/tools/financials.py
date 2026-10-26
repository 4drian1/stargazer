"""财务数据层：akshare（新浪源财务指标，与iFinD实测差异≤0.1）。
十项打分指标映射见 INDICATOR_MAP。已知坑：个别票毛利率/净利润增长率缺失 -> 打分按中位50处理。"""
import warnings

import pandas as pd

warnings.filterwarnings("ignore")

_ak = None


def _load_ak():
    global _ak
    if _ak is None:
        import akshare as ak
        _ak = ak
    return _ak


# 打分因子十项 -> akshare字段
INDICATOR_MAP = {
    "ROE": "加权净资产收益率(%)",
    "净利率": "销售净利率(%)",
    "毛利率": "销售毛利率(%)",
    "EPS": "摊薄每股收益(元)",
    "经营现金流/股": "每股经营性现金流(元)",
    "营收YoY": "主营业务收入增长率(%)",
    "归母净利YoY": "净利润增长率(%)",
    "资产负债率": "资产负债率(%)",
    "流动比率": "流动比率",
}


def annual_indicators(code: str, year: int):
    """取某年度年报九项指标。返回 dict 或 None。year=报告年度，如 2025。"""
    ak = _load_ak()
    try:
        df = ak.stock_financial_analysis_indicator(symbol=code, start_year=str(year - 1))
    except Exception:
        return None
    if df is None or len(df) == 0:
        return None
    import datetime as dt
    target = dt.date(year, 12, 31)
    row = df[df["日期"] == target]
    if len(row) == 0:  # 年报未出则退取最新一期
        row = df.tail(1)
    row = row.iloc[0]
    out = {}
    for k, col in INDICATOR_MAP.items():
        hit = [c for c in df.columns if col.split("(")[0] in c]
        val = row[hit[0]] if hit else None
        out[k] = None if val is None or (isinstance(val, float) and pd.isna(val)) else float(val)
    out["报告期"] = str(row["日期"])
    return out
