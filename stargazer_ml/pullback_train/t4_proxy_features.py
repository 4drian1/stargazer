# -*- coding: utf-8 -*-
"""T4：板块分钟代理软特征（评审冻结口径，§18.17）。

输入：proxy_baskets.json（T-1 月度冻结，t4_baskets.py 产出）。
成分分钟数据：minute_cache_clean（proxy_fetch.py 走新管线补拉+验收）。

每 (code, dp_date, dp) 输出（dp ∈ AM 1000/1030 + PM 1330/1400/1430/1450）：
  t4_proxy_available      0/1；无板块映射或篮子当日无数据=0（**不等于板块弱**）
  t4_basket_cumret_dp     篮子(流动性成分)累计收益中位数（开盘=1，至DP）
  t4_stock_excess_dp      个股累计收益 − 篮子累计收益（至DP）
  t4_basket_up_ratio      DP时篮子成分累计收益>0的比例
  t4_basket_h2c           篮子自盘中累计收益高位的回落深度（中位，正=低于高位）
  t4_basket_above_vwap    DP时篮子成分 close>自身sessionVWAP 的比例（VWAP状态）
  t4_leader_cumret_dp     龙头自开盘累计收益（至DP）
  t4_center_cumret_dp     中军自开盘累计收益（至DP）
  t4_leader_h2c           龙头自盘中累计收益高位的回落深度（至DP）
  t4_stock_excess_leader  个股累计收益 − 龙头累计收益（至DP）

口径固定：
- 成分篮子按月冻结（proxy_baskets.json），禁止事后回填；
- 只作软特征，不设板块硬门；proxy_available=0 的行其余特征保持 NaN；
- 所有取值只用 ≤DP 已完成的 bar，cumret 以当日首根 bar 开盘为 1。
幂等；--samples/--out 跑 2025 与 2026。
"""
import json
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from minute_label_engine import load_code  # noqa: E402

OUT_DIR = "D:/stargazer/stargazer_ml/pullback_train"
SAMPLES = os.path.join(OUT_DIR, "minute_samples_2025.parquet")
OUT = os.path.join(OUT_DIR, "minute_t4_proxy_2025.parquet")
BASKETS = os.path.join(OUT_DIR, "proxy_baskets.json")
CONS = os.path.join(OUT_DIR, "board_cons.json")
DPS = (1000, 1030, 1330, 1400, 1430, 1450)
FEATS = ("t4_proxy_available", "t4_basket_cumret_dp", "t4_stock_excess_dp",
         "t4_basket_up_ratio", "t4_basket_h2c", "t4_basket_above_vwap",
         "t4_leader_cumret_dp", "t4_center_cumret_dp", "t4_leader_h2c",
         "t4_stock_excess_leader")


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


class Member:
    """单只成分股索引：day -> (t[], cr[], hwm[], vwap[]) 或 None。

    cr    = close / 当日首bar开盘（开盘基准累计收益）
    hwm   = cr 的 running max（盘中累计收益高位）
    vwap  = 自09:30起累计 amount/volume（session VWAP）
    """

    def __init__(self, code):
        self.code = code
        self.cm = load_code(code)
        self.cache = {}

    def day(self, day):
        if day in self.cache:
            return self.cache[day]
        out = None
        if self.cm is not None:
            hit = np.where(self.cm["days"] == day)[0]
            if len(hit):
                di = int(hit[0])
                idx = np.where(self.cm["day_idx"] == di)[0]
                t = self.cm["t"][idx].astype(int)
                o0 = float(self.cm["o"][idx][0])
                if o0 > 0:
                    cr = self.cm["c"][idx] / o0
                    hwm = np.maximum.accumulate(cr)
                    a = self.cm["a"][idx].astype(float)
                    v = self.cm["v"][idx].astype(float)
                    cv = np.cumsum(v)
                    with np.errstate(all="ignore"):
                        vwap = np.cumsum(a) / np.where(cv > 0, cv, np.nan)
                    out = (t, cr.astype(float), hwm.astype(float), vwap)
        self.cache[day] = out
        if len(self.cache) > 40:  # 防内存膨胀
            for k in list(self.cache)[:20]:
                if k != day:
                    self.cache.pop(k, None)
        return out


def val_at(t_arr, v_arr, tq):
    """时间<=tq 的最后一根 bar 取值；无则 NaN。"""
    i = int(np.searchsorted(t_arr, tq, side="right")) - 1
    return float(v_arr[i]) if i >= 0 else np.nan


def above_vwap_frac(rec, tq):
    """当日截至 tq 的 bar 中 close>sessionVWAP 的比例。"""
    if rec is None:
        return np.nan
    t, cr, hwm, vwap = rec
    i = int(np.searchsorted(t, tq, side="right")) - 1
    if i < 0:
        return np.nan
    m = np.isfinite(vwap[:i + 1])
    if not m.any():
        return np.nan
    return float(np.mean(cr[:i + 1][m] > vwap[:i + 1][m]))


def drawdown_at(rec, tq):
    """盘中累计收益高位 − 当前累计收益（至DP，正=低于高位）。"""
    if rec is None:
        return np.nan
    t, cr, hwm, vwap = rec
    i = int(np.searchsorted(t, tq, side="right")) - 1
    return float(hwm[i] - cr[i]) if i >= 0 else np.nan


def main():
    global SAMPLES, OUT
    args = sys.argv[1:]
    if "--samples" in args:
        SAMPLES = args[args.index("--samples") + 1]
    if "--out" in args:
        OUT = args[args.index("--out") + 1]
    t0 = time.time()
    cons = json.load(open(CONS, encoding="utf-8"))
    code2board = {}
    for b, codes in cons.items():
        for c in codes:
            code2board.setdefault(str(c), b)
    baskets = json.load(open(BASKETS, encoding="utf-8"))
    all_members = sorted({str(v2[k]) for mv in baskets.values()
                          for v2 in mv.values() for k in ("leader", "center")}
                         | {str(c) for mv in baskets.values()
                            for v2 in mv.values() for c in v2["liquid"]})
    members = {c: Member(c) for c in all_members}
    have = sum(1 for m in members.values() if m.cm is not None)
    log(f"成分 {len(members)} 只，分钟缓存可用 {have} 只")

    df = pd.read_parquet(SAMPLES)
    df = df[df["dp"].isin(DPS)].copy()
    g = (df.groupby(["code", "dp_date", "dp"]).size().reset_index()
           .rename(columns={0: "n"}))
    log(f"唯一 (code,dp_date,dp): {len(g):,}")
    rows = []
    cur, cm = None, None
    for _, r in g.iterrows():
        code, day, dp = str(r["code"]), int(r["dp_date"]), int(r["dp"])
        tq = dp * 100
        f = {k: np.nan for k in FEATS}
        f["t4_proxy_available"] = 0.0
        board = code2board.get(code)
        bk = baskets.get(board, {}).get(str(day)[:6]) if board else None
        if bk:
            liq = [(c, members[c].day(day)) for c in
                   (str(x) for x in bk["liquid"])]
            liq = [(c, x) for c, x in liq if x is not None]
            ld = members[str(bk["leader"])].day(day)
            ct = members[str(bk["center"])].day(day)
            if liq or ld or ct:
                f["t4_proxy_available"] = 1.0
            if liq:
                crs = [val_at(x[0], x[1], tq) for _, x in liq]
                f["t4_basket_cumret_dp"] = float(np.nanmedian(crs))
                f["t4_basket_up_ratio"] = float(np.nanmean(
                    [x > 1.0 for x in crs if np.isfinite(x)]))
                f["t4_basket_h2c"] = float(np.nanmedian(
                    [drawdown_at(x, tq) for _, x in liq]))
                fr = [above_vwap_frac(x, tq) for _, x in liq]
                fr = [x for x in fr if np.isfinite(x)]
                if fr:
                    f["t4_basket_above_vwap"] = float(np.mean(fr))
            if ld:
                f["t4_leader_cumret_dp"] = val_at(ld[0], ld[1], tq)
                f["t4_leader_h2c"] = drawdown_at(ld, tq)
            if ct:
                f["t4_center_cumret_dp"] = val_at(ct[0], ct[1], tq)
            # 个股自身累计收益（当日首bar开盘=1）
            if code != cur:
                cur = code
                cm = load_code(code)
            stock_cr = np.nan
            if cm is not None:
                hit = np.where(cm["days"] == day)[0]
                if len(hit):
                    idx = np.where(cm["day_idx"] == int(hit[0]))[0]
                    o0 = float(cm["o"][idx][0])
                    if o0 > 0:
                        stock_cr = val_at(cm["t"][idx].astype(int),
                                          (cm["c"][idx] / o0).astype(float),
                                          tq)
            if np.isfinite(stock_cr):
                if np.isfinite(f["t4_basket_cumret_dp"]):
                    f["t4_stock_excess_dp"] = stock_cr - f[
                        "t4_basket_cumret_dp"]
                if np.isfinite(f["t4_leader_cumret_dp"]):
                    f["t4_stock_excess_leader"] = stock_cr - f[
                        "t4_leader_cumret_dp"]
        rows.append({"code": code, "dp_date": day, "dp": dp, **f})
    out = pd.DataFrame(rows)
    out.to_parquet(OUT, index=False)
    log(f"写入 {OUT}: {out.shape}；用时 {time.time()-t0:.0f}s")
    log(f"proxy_available 占比: {out['t4_proxy_available'].mean():.1%}")
    nan = out.loc[out["t4_proxy_available"] == 1, FEATS[1:]].isna().mean()
    log(f"available=1 行特征NaN率: {nan.round(4).to_dict()}")


if __name__ == "__main__":
    main()
