"""强势中继低吸 + 独立逻辑低吸（绿盘低吸 v2）。

入口：
- load_watchlist(cfg) / save_watchlist(cfg, data) -> 读写观察名单（读取兼容保留，
  scan_intraday_new 仍在用）
- evaluate_intraday(cfg, now) -> 盘中按冻结计划 DSL 规则求值（plan_frozen.json，
  语义同 stargazer_plan.plan_db；老的下杀3%观察名单规则已 deprecated）
- build_premarket_state(cfg) / save_premarket(cfg, data) -> 九点盘前助手

2026-09-23（T5 设计 §6）14:30 尾盘扫描已删除：run_afternoon_scan 及下游
（_SCAN_STATE/scan_state/scan_async）此前仅被 /api/sector_pullback/scan 路由
链调用，现为死代码保留（本文件不动其内部），无任何路由/线程可达。
save/load_user_sectors（实时监控板块设置）不受影响。

本模块只跑在日频/实时数据上，不依赖 ML 模型训练。
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timedelta
from typing import Any

import numpy as np
import pandas as pd

from tools import market

try:
    import paths
    BASE = paths.APP_DIR
except ImportError:
    BASE = os.path.dirname(os.path.abspath(__file__))


# ---------------- 配置常量 ----------------
MIN_CAP_YI = 10.0           # 流通市值下限 10 亿
MAX_CAP_YI = 100.0          # 流通市值上限 100 亿
MAX_TOTAL_CAP_YI = 100.0    # 总市值上限 100 亿（user 指定，总市值口径）
MIN_ADV_WAN = 5000          # 20 日平均成交额下限 5000 万
MAX_BOARD_CONS = 200        # 每个板块最多看 200 只成分股
MIN_BOARD_AMOUNT_YI = 10.0  # 板块 20 日成交额下限 10 亿
TARGET_CANDIDATES = 5       # 每日目标候选数，按板块强弱排名依次补足

# 固定观察板块（用户指定的 12 个，名称为同花顺口径）→ 东财板块名候选（按优先级）
# deprecated（2026-09-17）：盘中规则已换成冻结计划 DSL（见下方「冻结计划规则」）。
# 2026-09-23：尾盘扫描（run_afternoon_scan 唯一调用方）已删除，本表仅被
# 死代码引用保留，不再驱动任何扫描/推送。
WATCH_SECTORS = [
    ("储能", ["储能概念"]),
    ("半导体", ["半导体"]),
    ("共封装光学(CPO)", ["CPO概念", "共封装光学"]),
    ("数据中心(AIDC)", ["数据中心"]),
    ("农业种植", ["农业种植"]),
    ("创新药", ["创新药"]),
    ("油气开采及服务", ["油气设服", "油气开采"]),
    ("军工", ["军工"]),
    ("房地产", ["房地产"]),
    ("通信服务", ["通信服务"]),
    ("通用设备", ["通用设备"]),
    ("存储芯片", ["存储芯片"]),
]


def _resolve_watch_boards(boards_sorted: list) -> tuple[list, list]:
    """把固定观察板块映射到东财板块。返回 (已解析[(用户名,东财名,bk,pct)], 未找到[用户名])。"""
    resolved, missing = [], []
    for user_name, keywords in WATCH_SECTORS:
        hit = None
        for kw in keywords:
            # 先精确匹配，再包含匹配
            exact = [x for x in boards_sorted if x[0] == kw]
            if exact:
                hit = exact[0]
                break
            partial = [x for x in boards_sorted if kw in x[0]]
            if partial:
                hit = partial[0]
                break
        if hit:
            resolved.append((user_name, hit[0], hit[1], hit[2]))
        else:
            missing.append(user_name)
    # 按当日涨跌幅降序 = 强弱排名
    resolved.sort(key=lambda x: -x[3])
    return resolved, missing


def _data_path(cfg, name):
    p = cfg.get("model", {}).get("cache_dir", "data/model")
    out = os.path.join(BASE, p)
    os.makedirs(out, exist_ok=True)
    return os.path.join(out, name)


def _wl_path(cfg):
    p = cfg.get("agent", {}).get("sector_pullback_watchlist", "data/sector_pullback_watchlist.json")
    if os.path.isabs(p):
        return p
    return os.path.join(BASE, p)


# ---------------- 数据工具 ----------------
def _kline_one(code: str, start: str, end: str):
    """拉单只股票 K 线并计算特征（供线程池调用）。"""
    df = market.kline(code, start=start, end=end, qfq=True)
    if df is None or len(df) < 40:
        return code, None
    df["code"] = code
    df = df.sort_values("date").reset_index(drop=True)
    df["prev_close"] = df["close"].shift(1)
    df["ret1"] = df["close"] / df["prev_close"] - 1
    df["ret5"] = df["close"].pct_change(5, fill_method=None)
    df["ret20"] = df["close"].pct_change(20, fill_method=None)
    df["high5"] = df["high"].rolling(5, min_periods=5).max()
    df["high20"] = df["high"].rolling(20, min_periods=20).max()
    df["low5"] = df["low"].rolling(5, min_periods=5).min()
    df["low20"] = df["low"].rolling(20, min_periods=20).min()
    df["low60"] = df["low"].rolling(60, min_periods=40).min()
    df["high60"] = df["high"].rolling(60, min_periods=40).max()
    # pos5：5日位置（用low5/high5）；pos20：真正的20日位置（用low20/high20）
    df["pos5"] = (df["close"] - df["low5"]) / (df["high5"] - df["low5"]).replace(0, np.nan)
    df["pos20"] = (df["close"] - df["low20"]) / (df["high20"] - df["low20"]).replace(0, np.nan)
    df["pos60"] = (df["close"] - df["low60"]) / (df["high60"] - df["low60"]).replace(0, np.nan)
    # 腾讯 K 线 vol 单位是"手"（1手=100股），成交额需 ×100 才是元；
    # adv20 与 amount 必须同口径，否则 amount_ratio20 失真
    df["amount"] = df["close"] * df["vol"] * 100
    df["adv20"] = df["amount"].shift(1).rolling(20, min_periods=15).mean()
    df["amount_ratio20"] = df["amount"] / df["adv20"].replace(0, np.nan)
    # ATR14（真实波幅均值，用于止损位计算）
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - df["prev_close"]).abs(),
        (df["low"] - df["prev_close"]).abs(),
    ], axis=1).max(axis=1)
    df["atr14"] = tr.rolling(14, min_periods=10).mean()
    df["vol_mean5"] = df["vol"].shift(1).rolling(5, min_periods=5).mean()
    df["vol_contract_5d"] = df["vol"] < df["vol_mean5"] * 0.8
    # 下跌缩量：近3日成交量逐日递减
    df["vol_declining_3d"] = (df["vol"] < df["vol"].shift(1)) & (df["vol"].shift(1) < df["vol"].shift(2))
    # 简单 RSI14
    delta = df["close"].diff()
    gain = delta.clip(lower=0).rolling(14, min_periods=14).mean()
    loss = (-delta.clip(upper=0)).rolling(14, min_periods=14).mean()
    rs = gain / loss.replace(0, np.nan)
    df["rsi14"] = 100 - 100 / (1 + rs)
    # RSI 动能扩展：3日变化 + 近5日最低
    df["rsi_delta3"] = df["rsi14"] - df["rsi14"].shift(3)
    df["rsi_lowest5"] = df["rsi14"].rolling(5, min_periods=3).min()
    # BBI 趋势完整性 = (MA3+MA6+MA12+MA24)/4
    bbi = (df["close"].rolling(3).mean() + df["close"].rolling(6).mean()
           + df["close"].rolling(12).mean() + df["close"].rolling(24).mean()) / 4
    df["bbi"] = bbi
    df["bbi_distance"] = df["close"] / bbi - 1
    df["bbi_slope5"] = bbi / bbi.shift(5) - 1
    # BOLL(20,2) 波动位置
    boll_mid = df["close"].rolling(20, min_periods=20).mean()
    boll_std = df["close"].rolling(20, min_periods=20).std()
    df["boll_upper"] = boll_mid + 2 * boll_std
    df["boll_lower"] = boll_mid - 2 * boll_std
    df["boll_pct_b"] = ((df["close"] - df["boll_lower"])
                        / (df["boll_upper"] - df["boll_lower"]).replace(0, np.nan))
    df["boll_width"] = (df["boll_upper"] - df["boll_lower"]) / boll_mid.replace(0, np.nan)
    # 量能结构：近5日下跌日均量/上涨日均量；近3日量斜率（相对变化率/日）
    down_vol = float(df["vol"].where(df["ret1"] < 0).tail(5).mean())
    up_vol = float(df["vol"].where(df["ret1"] > 0).tail(5).mean())
    df["down_volume_ratio5"] = down_vol / up_vol if up_vol > 0 else np.nan
    df["volume_slope3"] = (df["vol"] - df["vol"].shift(2)) / df["vol"].shift(2).replace(0, np.nan) / 2
    # 5 日回撤
    df["pullback_from_high5"] = df["close"] / df["high5"] - 1
    # 实体/下影线
    df["body"] = (df["close"] - df["open"]).abs()
    df["body_prev"] = df["body"].shift(1)
    df["lower_shadow"] = (df[["open", "close"]].min(axis=1) - df["low"]) / df["prev_close"].replace(0, np.nan)
    # 振幅
    df["amplitude"] = (df["high"] - df["low"]) / df["prev_close"].replace(0, np.nan)
    df["amp_mean5"] = df["amplitude"].shift(1).rolling(5, min_periods=5).mean()
    df["amp_contract"] = df["amplitude"] < df["amp_mean5"] * 0.8
    return code, df


def _kline_features(codes: list[str], max_workers: int = 6) -> dict[str, pd.DataFrame]:
    """批量拉取 K 线并计算常见特征（多线程加速）。"""
    end = date.today().isoformat()
    start = (date.today() - timedelta(days=180)).isoformat()
    out = {}
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        for code, df in pool.map(lambda c: _kline_one(c, start, end), codes):
            if df is not None:
                out[code] = df
    return out


def _compute_quality_flag(df: pd.DataFrame) -> bool:
    """流动性+市值硬过滤（A/B 共用）：流通市值 10~100亿 且 20日成交额>=5000万。"""
    if df is None or len(df) < 20:
        return False
    last = df.iloc[-1]
    if not pd.notna(last.get("float_mv")):
        return False
    if last["float_mv"] > MAX_CAP_YI or last["float_mv"] < MIN_CAP_YI:
        return False
    adv20 = df["amount"].shift(1).rolling(20, min_periods=15).mean().iloc[-1]
    if pd.isna(adv20) or adv20 < MIN_ADV_WAN * 1e4:
        return False
    return True


def _compute_quality_score(df: pd.DataFrame) -> float:
    """妖股基因连续评分 0~1（排序加分用，不作硬门槛）：
    近20日涨幅最多 0.4 + 近60日涨停次数最多 0.4 + 市值达标 0.2。"""
    if df is None or len(df) < 60:
        return 0.0
    last = df.iloc[-1]
    score = 0.0
    ret20 = last.get("ret20")
    if pd.notna(ret20) and ret20 > 0:
        score += min(ret20, 0.5) / 0.5 * 0.4
    limit_up = int((df["ret1"].tail(60) >= 0.095).sum())
    score += min(limit_up, 5) / 5 * 0.4
    fmc = last.get("float_mv")
    if pd.notna(fmc) and MIN_CAP_YI <= fmc <= MAX_CAP_YI:
        score += 0.2
    return round(float(score), 3)


# ---------------- 利空公告过滤 ----------------
BAD_NEWS_PATTERNS = [
    "强制赎回", "提前赎回", "赎回登记", "赎回提示",
    "减持", "被立案", "立案告知", "警示函", "监管函", "行政处罚",
    "退市风险", "终止上市", "风险警示", "业绩预亏", "预亏",
    "商誉减值", "股权冻结", "冻结", "爆仓", "重大诉讼", "败诉", "裁决",
]


def _has_bad_news(code: str, days: int = 20) -> str | None:
    """拉取个股近期公告标题，命中利空关键词则返回命中词，否则 None。
    接口失败时不阻塞（返回 None 放行），只记录。"""
    url = ("https://np-anotice-stock.eastmoney.com/api/security/ann?sr=-1"
           f"&page_size=30&page_index=1&ann_type=A&client_source=web&stock_list={code}")
    try:
        import urllib.request
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        raw = urllib.request.urlopen(req, timeout=8).read().decode("utf-8", errors="replace")
        data = json.loads(raw)
        items = (data.get("data") or {}).get("list") or []
        from datetime import timedelta as _td
        cutoff = (date.today() - _td(days=days)).isoformat()
        for it in items:
            notice_date = (it.get("notice_date") or "")[:10]
            if notice_date and notice_date < cutoff:
                continue
            title = it.get("title") or ""
            for kw in BAD_NEWS_PATTERNS:
                if kw in title:
                    return f"{kw}:{title[:30]}"
    except Exception:
        return None
    return None


def _compute_adjustment_ready(df: pd.DataFrame) -> bool:
    """调整到位：放宽版，至少满足 4/5 条。"""
    if df is None or len(df) < 10:
        return False
    last = df.iloc[-1]
    prev = df.iloc[-2] if len(df) >= 2 else last
    checks = []
    # 1. 从 5 日高点回落 3%~18%
    pb = last.get("pullback_from_high5")
    checks.append(pd.notna(pb) and -0.18 <= pb <= -0.03)
    # 2. 当日缩量
    ar = last.get("amount_ratio20")
    checks.append(pd.notna(ar) and ar < 1.0)
    # 3. pos20 > 30%（真正的20日位置，不是高位崩盘）
    pos20 = last.get("pos20")
    checks.append(pd.notna(pos20) and pos20 > 0.30)
    # 4. 波动收敛
    checks.append(bool(last.get("amp_contract")))
    # 5. 下影线 / 十字星 / 小阳线
    body = last.get("body")
    body_prev = prev.get("body")
    lower_shadow = last.get("lower_shadow")
    has_tail = pd.notna(lower_shadow) and lower_shadow > 0.005
    small_body = pd.notna(body) and pd.notna(body_prev) and body_prev > 0 and body < body_prev * 0.7
    checks.append(has_tail or small_body)
    return sum(checks) >= 4


def _compute_first_divergence(df: pd.DataFrame) -> bool:
    """强势股第一次分歧：近5日大涨，今天冲高回落/下跌，放量。"""
    if df is None or len(df) < 10:
        return False
    last = df.iloc[-1]
    ret5 = last.get("ret5")
    if pd.isna(ret5) or ret5 < 0.15:
        return False
    # 此前 5 日无单日回落 > 3%
    recent = df.tail(6).head(5)
    if (recent["ret1"] < -0.03).any():
        return False
    # 今日从高点回落 >= 3%
    high_pct = (last["high"] - last["prev_close"]) / last["prev_close"] if last["prev_close"] else np.nan
    close_pct = last["ret1"]
    if pd.isna(high_pct) or pd.isna(close_pct):
        return False
    if high_pct - close_pct < 0.03:
        return False
    # 放量
    ar = last.get("amount_ratio20")
    if pd.isna(ar) or ar < 1.0:
        return False
    return True


# 交易成本与滑点（往返），计算净盈亏比时扣除
ROUND_TRIP_COST = 0.002
# 止损最大允许距离（1.5×ATR 超过该比例说明波动过大，放弃）
MAX_STOP_PCT = 0.07


def _trend_gate(df: pd.DataFrame) -> bool:
    """趋势关（A模式硬门槛，替代"20日线向下一票否决"的放宽版）：
    - MA20 向上 且 收盘在 MA20 上方 → 通过（强趋势）
    - MA20 向下/走平 → 仅当收盘距20日最低价 ≤6%（跌透低位、下方结构空间小）才通过
    - 其余（半山腰下降票）→ 淘汰
    """
    if df is None or len(df) < 26:
        return False
    close, low = df["close"], df["low"]
    ma20 = close.rolling(20).mean()
    last_ma, last_ma_prev = ma20.iloc[-1], ma20.iloc[-6] if len(ma20) >= 6 else np.nan
    last_close = float(close.iloc[-1])
    if pd.notna(last_ma) and pd.notna(last_ma_prev) \
            and last_ma > last_ma_prev and last_close > float(last_ma):
        return True
    llv20 = low.rolling(20, min_periods=20).min().iloc[-1]
    if pd.isna(llv20) or llv20 <= 0:
        return False
    return (last_close - float(llv20)) / float(llv20) <= 0.06


def _compute_rr(df: pd.DataFrame, entry_price: float) -> dict | None:
    """统一盈亏比（ATR口径 v2）：目标价、止损价、RR 由同一函数计算，筛选与展示口径一致。

    止损 = 买入价 − 1.5×ATR14（唯一标准，不再混用结构低点/固定比例）；
    止损距离超过 7% 说明波动太大，直接放弃（返回 None）。
    目标 = 上方最近阻力位（5日/20日高点中较近者）；无阻力参考时用 2R 扩展兜底。
    RR 分档：>=3.0 强候选；1.5~3.0 观察级；<1.5 排除。
    """
    if df is None or len(df) < 5 or entry_price <= 0:
        return None
    last = df.iloc[-1]
    atr = last.get("atr14")
    if pd.isna(atr) or atr <= 0:
        return None
    stop = entry_price - 1.5 * float(atr)
    risk_dist = entry_price - stop
    if risk_dist <= 0 or risk_dist / entry_price > MAX_STOP_PCT:
        return None  # 波动太大，放弃
    # 目标：上方最近阻力位（5日/20日高点中较近者）；无阻力参考时用 2R 扩展兜底
    resistances = [float(t) for t in (last.get("high5"), last.get("high20"))
                   if pd.notna(t) and t > entry_price]
    target = min(resistances) if resistances else entry_price + 2.0 * risk_dist
    if target <= entry_price:
        return None
    gain = (target - entry_price) / entry_price - ROUND_TRIP_COST
    risk = risk_dist / entry_price + ROUND_TRIP_COST
    if risk <= 0 or gain <= 0:
        return None
    rr = gain / risk
    tier = "STRONG" if rr >= 3.0 else ("OBSERVE" if rr >= 1.5 else "WEAK")
    return {"rr": float(rr), "target": target, "stop": stop, "tier": tier,
            "gain_pct": float(gain), "risk_pct": float(risk)}


def _model_advice_index(cfg) -> dict:
    """读取本地模型批次输出(g0/横盘建议)，按代码索引胜率代理与预期收益。
    没有模型输出时返回 {}（候选只显示 RR，不影响筛选）。"""
    out = {}
    for fn, key in (("g0_advice.json", "g0"), ("horizontal_advice.json", "horizontal")):
        p = _data_path(cfg, fn)
        try:
            with open(p, encoding="utf-8") as f:
                rows = json.load(f)
        except (OSError, ValueError):
            continue
        if not isinstance(rows, list):
            continue
        for r in rows:
            code = str(r.get("code") or "")
            if not code or code in out:
                continue
            if key == "g0":
                p_win = 1.0 - float(r.get("risk_probability_t2") or 0.5)
                exp = float(r.get("expected_return_t2") or 0)
            else:
                p_win = 1.0 - float(r.get("deep_risk_probability") or 0.5)
                exp = float(r.get("deep_opportunity_score") or 0)
            out[code] = {"win_prob": round(p_win, 4), "exp_return": round(exp, 4)}
    return out


def _enrich_expectancy(candidates: list, model_idx: dict) -> list:
    """给候选附加模型胜率/预期收益/期望值，并按期望值排序（无模型数据则按 RR）。"""
    for c in candidates:
        m = model_idx.get(c["code"])
        if not m:
            continue
        c["win_prob"] = m["win_prob"]
        c["exp_return"] = m["exp_return"]
        # 期望值 = 胜率×净盈利 − (1−胜率)×净亏损（净盈/净亏来自 ATR 口径 RR 计算）
        gain_pct = c.get("gain_pct")
        risk_pct = c.get("risk_pct")
        if gain_pct is None or risk_pct is None:
            # 兼容旧数据：用 target/stop 反推
            entry = c.get("price") or 0
            if entry and c.get("target_price") and c.get("stop_loss"):
                gain_pct = (c["target_price"] - entry) / entry - ROUND_TRIP_COST
                risk_pct = (entry - c["stop_loss"]) / entry + ROUND_TRIP_COST
            else:
                continue
        c["expectancy"] = round(m["win_prob"] * gain_pct
                                - (1 - m["win_prob"]) * risk_pct, 4)
    have_exp = [c for c in candidates if "expectancy" in c]
    if have_exp:
        candidates.sort(key=lambda x: x.get("expectancy", -999), reverse=True)
    return candidates


# ---------------- 板块扫描 ----------------
def _sector_metrics(boards: dict[str, str], max_boards: int = 60,
                    progress=None, min_amount_yi: float | None = None) -> list[dict]:
    """计算每个板块的强度指标。boards 为预筛选后的 {名称: 代码}。
    min_amount_yi 为成分平均成交额下限（亿），None 用全局 MIN_BOARD_AMOUNT_YI。"""
    min_amt = MIN_BOARD_AMOUNT_YI if min_amount_yi is None else min_amount_yi
    rows = []
    total = min(len(boards), max_boards)
    for idx, (name, bk) in enumerate(list(boards.items())[:max_boards]):
        cons = market.board_constituents(bk)
        if not cons or len(cons) < 5:
            continue
        # 只保留成交额最高的前 MAX_BOARD_CONS 只，避免大板块拉取过慢
        cons = dict(sorted(cons.items(), key=lambda kv: -kv[1].get("amount_yi", 0))
                    [:MAX_BOARD_CONS])
        pcts = np.array([v["pct"] for v in cons.values()], dtype=float)
        amts = np.array([v.get("amount_yi") or 0 for v in cons.values()], dtype=float)
        limit_up = int((pcts >= 9.8).sum())
        limit_dn = int((pcts <= -9.8).sum())
        avg_pct = float(np.nanmean(pcts))
        avg_amount = float(np.nanmean(amts))
        if avg_amount < min_amt:
            continue
        rows.append({
            "name": name,
            "bk": bk,
            "cons_n": len(cons),
            "avg_pct": round(avg_pct, 2),
            "limit_up": limit_up,
            "limit_dn": limit_dn,
            "avg_amount": round(avg_amount, 2),
            "cons": cons,
        })
        if progress and total:
            progress(f"板块 {idx + 1}/{total}：{name}")
        time.sleep(0.05)
    return rows


def _select_sectors(metrics: list[dict], market_pct: float) -> dict[str, list[dict]]:
    """分出 A 模式（强势/企稳）和 B 模式（恐慌）板块各 2 个。"""
    # 强势：涨停多 或 涨幅高
    hot = sorted(metrics, key=lambda x: (x["limit_up"], x["avg_pct"]), reverse=True)[:MODE_A_SECTORS]
    # 企稳：涨幅可能不高，但相对大盘强，且上涨家数>下跌家数
    stable = sorted(
        [m for m in metrics if m["avg_pct"] > market_pct and m["avg_pct"] > -1.0],
        key=lambda x: (x["avg_pct"] - market_pct, x["limit_up"]), reverse=True
    )[:MODE_A_SECTORS]
    # 恐慌：5日跌幅最大——但这里只有当日数据；用当日跌幅最大近似
    panic = sorted(
        [m for m in metrics if m["avg_pct"] < -1.5],
        key=lambda x: x["avg_pct"]
    )[:MODE_B_SECTORS]
    return {"hot": hot, "stable": stable, "panic": panic}


# ---------------- 候选生成：A 模式（强势/企稳板块中的中继低吸） ----------------
def _candidate_metrics(last, fmc: float) -> dict:
    """汇总输出连续指标（不转布尔硬阈值，供排序和后续训练用）。"""
    fmc_yuan = fmc * 1e8 if fmc else np.nan
    turnover = float(last["amount"] / fmc_yuan) if fmc_yuan and fmc_yuan > 0 else np.nan
    if pd.notna(turnover) and turnover > 1.0:  # 日换手>100% 视为流通市值数据异常
        turnover = np.nan
    return {
        "turnover": round(turnover * 100, 2) if pd.notna(turnover) else None,
        "rsi14": round(float(last.get("rsi14") or 0), 2),
        "rsi_delta3": round(float(last.get("rsi_delta3") or 0), 2),
        "bbi_distance": round(float(last.get("bbi_distance") or 0) * 100, 2),
        "bbi_slope5": round(float(last.get("bbi_slope5") or 0) * 100, 2),
        "boll_pct_b": round(float(last.get("boll_pct_b") or 0), 3),
        "boll_width": round(float(last.get("boll_width") or 0) * 100, 2),
        "down_volume_ratio5": (round(float(last.get("down_volume_ratio5") or 0), 2)
                               if pd.notna(last.get("down_volume_ratio5")) else None),
        "volume_slope3": round(float(last.get("volume_slope3") or 0), 3),
        "vol_declining": bool(last.get("vol_declining_3d")),
    }


def _screen_candidates(board_meta: dict, mode: str, klines: dict[str, pd.DataFrame]) -> list[dict]:
    """A 模式：强板块中的强股首次健康分歧。
    硬过滤只有市值+流动性；妖股基因改为 quality_score 排序加分；
    形态关保留（中继低吸核心）；rr 分档：>=3.0 强候选，1.5~3.0 观察级；利空公告剔除。"""
    out = []
    cons = board_meta.get("cons", {})
    for code, info in cons.items():
        df = klines.get(code)
        if df is None or len(df) < 5:
            continue
        # 质量关需要 float_mv：从板块成分信息合并进来（K 线本身没有该字段）
        df = df.copy()
        fmc = float(info.get("fmc") or 0)
        df["float_mv"] = fmc
        last = df.iloc[-1]
        # 硬过滤：流通市值 10~100 亿 + 20日成交额>=5000万
        if not _compute_quality_flag(df):
            continue
        # 总市值上限（info 带 tmc 时检查；尾盘扫描在选完后再统一用实时行情核）
        tmc = float(info.get("tmc") or 0)
        if tmc and tmc > MAX_TOTAL_CAP_YI:
            continue
        # 涨停日的票不是低吸买点（K线特征基于前收，实时价却是涨停价，口径错位）
        if _is_limit_up(code, float(info.get("pct") or 0)):
            continue
        # 趋势关：20日线向上且在上方 → 过；20日线向下 → 须跌透（距20日低点≤6%）
        if not _trend_gate(df):
            continue
        # 调整到位 / 第一次分歧
        adj_ready = _compute_adjustment_ready(df)
        first_div = _compute_first_divergence(df)
        if not (adj_ready or first_div):
            continue
        entry = info.get("price") or last["close"]
        rr_info = _compute_rr(df, entry)
        if rr_info is None or rr_info["tier"] == "WEAK":
            continue
        # 利空公告剔除（300112 债转股强制赎回这类）
        bad = _has_bad_news(code)
        if bad:
            continue
        q_score = _compute_quality_score(df)
        metrics = _candidate_metrics(last, fmc)
        out.append({
            "code": code,
            "name": info.get("name", ""),
            "sector": board_meta["name"],
            "bk": board_meta["bk"],
            "mode": mode,
            "submode": "A1",
            "price": round(entry, 2),
            "pct": round(info.get("pct", 0), 2),
            "ret5": round(float(last.get("ret5") or 0) * 100, 2),
            "ret20": round(float(last.get("ret20") or 0) * 100, 2),
            "pos20": round(float(last.get("pos20") or 0) * 100, 2),
            "pos60": round(float(last.get("pos60") or 0) * 100, 2),
            "pullback_from_high5": round(float(last.get("pullback_from_high5") or 0) * 100, 2),
            "quality_score": q_score,
            "amount_ratio20": round(float(last.get("amount_ratio20") or 1), 2),
            "target_price": round(rr_info["target"], 2),
            "stop_loss": round(rr_info["stop"], 2),
            "rr_ratio": round(rr_info["rr"], 2),
            "rr_tier": rr_info["tier"],
            "gain_pct": round(rr_info.get("gain_pct", 0), 4),
            "risk_pct": round(rr_info.get("risk_pct", 0), 4),
            "adj_ready": bool(adj_ready),
            "first_divergence": bool(first_div),
            "float_mv_yi": round(fmc, 2),
            **metrics,
            "reason": _build_reason(last, adj_ready, first_div, rr_info["rr"]),
        })
    # 排序：STRONG 优先，其次观察级；档内按 rr + 0.5*质量分
    out.sort(key=lambda x: (-(x["rr_tier"] == "STRONG"),
                            -(x["rr_ratio"] + 0.5 * x["quality_score"])))
    return out[:5]


# ---------------- 候选生成：B 模式（恐慌板块中的独立止跌股） ----------------
def _screen_candidates_b(board_meta: dict, klines: dict[str, pd.DataFrame]) -> list[dict]:
    """B 模式：弱板块中率先止跌、与板块脱钩的个股。
    不用妖股基因硬门槛，看：抗跌（相对板块收益）、止跌确认、缩量、盈亏比>=1.5。"""
    out = []
    board_pct = float(board_meta.get("avg_pct") or 0)
    cons = board_meta.get("cons", {})
    for code, info in cons.items():
        df = klines.get(code)
        if df is None or len(df) < 20:
            continue
        fmc = float(info.get("fmc") or 0)
        if not (MIN_CAP_YI <= fmc <= MAX_CAP_YI):
            continue
        tmc = float(info.get("tmc") or 0)
        if tmc and tmc > MAX_TOTAL_CAP_YI:
            continue
        last = df.iloc[-1]
        # 流动性：20日平均成交额 >= 5000万
        adv20 = df["amount"].shift(1).rolling(20, min_periods=15).mean().iloc[-1]
        if pd.isna(adv20) or adv20 < MIN_ADV_WAN * 1e4:
            continue
        # 抗跌：个股当日跌幅明显小于板块（downside_capture < 0.5）
        stock_pct = float(info.get("pct", 0))
        if board_pct < 0:
            capture = abs(stock_pct) / abs(board_pct) if abs(board_pct) > 0.01 else np.inf
        else:
            capture = 0.0
        relative = stock_pct - board_pct
        # 止跌确认：未创新低（收盘不低于近5日最低）且 收盘位置/下影线 二选一
        no_new_low = bool(last["close"] >= df["low"].tail(5).min() * 0.995)
        lower_shadow = last.get("lower_shadow")
        has_tail = pd.notna(lower_shadow) and lower_shadow > 0.005
        pos5 = last.get("pos5")
        close_pos_ok = pd.notna(pos5) and pos5 > 0.35
        stabilizing = no_new_low and (has_tail or close_pos_ok)
        # 下跌缩量
        ar = last.get("amount_ratio20")
        vol_shrink = pd.notna(ar) and ar < 1.0
        if not (stabilizing and (capture < 0.5 or relative > 1.0) and vol_shrink):
            continue
        entry = info.get("price") or last["close"]
        rr_info = _compute_rr(df, entry)
        if rr_info is None or rr_info["tier"] == "WEAK":
            continue
        # 利空公告剔除
        if _has_bad_news(code):
            continue
        q_df = df.copy()
        q_df["float_mv"] = fmc
        q_score = _compute_quality_score(q_df)
        fmc = float(info.get("fmc") or 0)
        metrics = _candidate_metrics(last, fmc)
        out.append({
            "code": code,
            "name": info.get("name", ""),
            "sector": board_meta["name"],
            "bk": board_meta["bk"],
            "mode": "panic",
            "submode": "B1",
            "price": round(entry, 2),
            "pct": round(stock_pct, 2),
            "ret5": round(float(last.get("ret5") or 0) * 100, 2),
            "ret20": round(float(last.get("ret20") or 0) * 100, 2),
            "pos20": round(float(last.get("pos20") or 0) * 100, 2),
            "pos60": round(float(last.get("pos60") or 0) * 100, 2),
            "relative_pct": round(relative, 2),
            "downside_capture": round(capture, 2) if np.isfinite(capture) else None,
            "quality_score": q_score,
            "amount_ratio20": round(float(ar or 1), 2),
            "pullback_from_high5": round(float(last.get("pullback_from_high5") or 0) * 100, 2),
            "target_price": round(rr_info["target"], 2),
            "stop_loss": round(rr_info["stop"], 2),
            "rr_ratio": round(rr_info["rr"], 2),
            "rr_tier": rr_info["tier"],
            "gain_pct": round(rr_info.get("gain_pct", 0), 4),
            "risk_pct": round(rr_info.get("risk_pct", 0), 4),
            "float_mv_yi": round(fmc, 2),
            **metrics,
            "reason": (f"抗跌(相对板块{relative:+.1f}%,capture={capture:.2f})、"
                       f"止跌确认、缩量(ar={float(ar):.2f})、盈亏比{rr_info['rr']:.2f}"),
        })
    out.sort(key=lambda x: (-(x["rr_tier"] == "STRONG"),
                            -(x["rr_ratio"] + 0.5 * x.get("quality_score", 0))))
    return out[:5]


def _is_limit_up(code: str, pct: float) -> bool:
    """按板块判断当日是否涨停（涨停价不是低吸买点）。"""
    if code.startswith(("300", "301", "688")):
        th = 19.5
    elif code.startswith(("8", "4", "92")):
        th = 29.5
    else:
        th = 9.5
    return pct >= th


def _risk_warnings(board_meta: dict, klines: dict[str, pd.DataFrame],
                   min_rr: float = 3.5) -> list[dict]:
    """风险示警：过了质量/市值关但形态关不过（回调未到位、无分歧信号），
    却顶着很高盈亏比（RR>min_rr）的票。不计入5只候选，不做分析，
    只提示'高盈亏比但风险高'，防止追高半山腰。
    注：ATR口径下RR普遍压缩（旧口径常见>5，新口径少见），阈值定为3.5。"""
    out = []
    cons = board_meta.get("cons", {})
    for code, info in cons.items():
        df = klines.get(code)
        if df is None or len(df) < 5:
            continue
        if _is_limit_up(code, float(info.get("pct") or 0)):
            continue
        df = df.copy()
        fmc = float(info.get("fmc") or 0)
        df["float_mv"] = fmc
        if not _compute_quality_flag(df):
            continue
        tmc = float(info.get("tmc") or 0)
        if tmc and tmc > MAX_TOTAL_CAP_YI:
            continue
        # 涨停日的票不是低吸买点（形态/K线口径与实时价错位，且追高风险大）
        if _is_limit_up(code, float(info.get("pct") or 0)):
            continue
        # 形态关过了的属于正常候选，不在此列
        if _compute_adjustment_ready(df) or _compute_first_divergence(df):
            continue
        entry = info.get("price") or df.iloc[-1]["close"]
        rr_info = _compute_rr(df, entry)
        if rr_info is None or rr_info["rr"] <= min_rr:
            continue
        out.append({
            "code": code,
            "name": info.get("name", ""),
            "sector": board_meta["name"],
            "price": round(entry, 2),
            "rr_ratio": round(rr_info["rr"], 2),
            "target_price": round(rr_info["target"], 2),
            "stop_loss": round(rr_info["stop"], 2),
            "note": "高盈亏比但风险高",
        })
    out.sort(key=lambda x: -x["rr_ratio"])
    return out[:5]


def _build_reason(last, adj_ready, first_div, rr):
    parts = []
    if adj_ready:
        parts.append("调整到位")
    if first_div:
        parts.append("强势股首次分歧")
    parts.append(f"盈亏比 {rr:.2f}")
    parts.append(f"回撤 {float(last.get('pullback_from_high5') or 0)*100:.1f}%")
    return "、".join(parts)


# ---------------- 公开接口 ----------------
def run_afternoon_scan(cfg, progress=None) -> dict:
    """尾盘扫描：固定 12 个观察板块按当日强弱排名，从第 1 名开始逐板块筛选，
    每板块最多取（目标数-已有数）只，候选凑满 TARGET_CANDIDATES 为止。
    每个板块的挑选情况写入 sector_notes。

    progress: 可选回调，接收字符串进度信息（供异步扫描轮询显示）。
    """
    def _p(msg):
        if progress:
            progress(msg)

    idx = market.index_quotes()
    market_pct = 0.0
    for i in idx:
        if i["code"] == "000001":
            market_pct = i.get("pct", 0)
            break
    _p("拉取板块行情…")
    boards_sorted = market.board_list_sorted()
    if not boards_sorted:
        boards_sorted = [(n, b, 0.0) for n, b in market.board_list().items()]
    resolved, missing = _resolve_watch_boards(boards_sorted)
    if not resolved:
        raise RuntimeError("固定观察板块一个都没有映射到东财板块，无法扫描")
    notes = [f"板块池：{len(resolved)} 个已映射" + (f"，未找到：{'、'.join(missing)}" if missing else "")]
    _p(f"板块排名：" + " > ".join(f"{n}({p:+.1f}%)" for n, _, _, p in resolved[:5]) + " …")

    # 先对每个板块拉成分股并计算强度
    board_metas = []
    for rank, (user_name, em_name, bk, pct) in enumerate(resolved, 1):
        _p(f"板块 {rank}/{len(resolved)}：{user_name}")
        # 固定观察板块池是用户指定的，不应用全市场排序用的高成交额门槛，
        # 只剔除成分极少（<5）或几乎无成交（均值<0.3亿，疑停牌/数据缺失）的板块
        metrics = _sector_metrics({em_name: bk}, max_boards=1, min_amount_yi=0.3)
        if not metrics:
            notes.append(f"#{rank} {user_name}({pct:+.2f}%)：板块数据无效（成分不足或成交额过低），跳过")
            continue
        m = metrics[0]
        m["user_name"] = user_name
        m["pct_board"] = pct
        m["rank"] = rank
        board_metas.append(m)

    # 一次性拉所有成分股 K 线
    all_codes = set()
    for m in board_metas:
        all_codes.update(m.get("cons", {}).keys())
    _p(f"拉取 {len(all_codes)} 只成分股 K 线…")
    klines = _kline_features(list(all_codes))
    _p("按板块排名筛选候选…")
    # K线时效检查：东财盘后更新有延迟/抖动，部分票可能缺最新交易日。
    # 缺数据的票形态判断基于前收，扫描结果里明确提示，避免口径错位（如用涨停实时价低吸）。
    _kl_dates = {str(df["date"].max()) for df in klines.values() if len(df)}
    if _kl_dates:
        _latest = max(_kl_dates)
        _stale = sum(1 for df in klines.values()
                     if len(df) and str(df["date"].max()) < _latest)
        if _stale:
            notes.append(f"⚠ K线时效：{_stale} 只成分股未更新到最新交易日（{_latest}），"
                         f"这些票的形态判断基于前收数据，结果供参考")

    result = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "market_pct": market_pct,
        "date": date.today().isoformat(),
        "sectors": {"ranked": []},
        "candidates": [],
        "risk_warnings": [],
        "sector_notes": notes,
    }
    seen = set()
    for m in board_metas:
        # 所有板块都标注模式（strong=强势中继候选，panic=恐慌板块独立低吸），便于完整排名展示
        m["mode"] = "panic" if m["pct_board"] < -1.5 else "strong"
    for m in board_metas:
        if len(result["candidates"]) >= TARGET_CANDIDATES:
            break
        mode = m["mode"]
        cands = (_screen_candidates_b(m, klines) if mode == "panic"
                 else _screen_candidates(m, "strong", klines))
        # 去重（一只股票可能属于多个板块）
        cands = [c for c in cands if c["code"] not in seen]
        need = TARGET_CANDIDATES - len(result["candidates"])
        take = cands[:need]
        # 总市值上限核验（成分数据只有流通市值；用实时行情的总市值字段）
        if take:
            try:
                tqs = market.quotes([c["code"] for c in take]) or {}
            except Exception:
                tqs = {}
            kept = [c for c in take
                    if not tqs.get(c["code"], {}).get("total_mv_yi")
                    or tqs[c["code"]]["total_mv_yi"] <= MAX_TOTAL_CAP_YI]
            if len(kept) < len(take):
                notes.append(f"#{m['rank']} {m['user_name']}："
                             f"{len(take) - len(kept)} 只总市值>{MAX_TOTAL_CAP_YI:.0f}亿被剔除")
            take = kept
        for c in take:
            seen.add(c["code"])
            result["candidates"].append(c)
        m["picked"] = len(take)
        if len(take) == 0:
            notes.append(f"#{m['rank']} {m['user_name']}({m['pct_board']:+.2f}%)：无符合个股，跳到下一板块")
        elif len(take) < need:
            notes.append(f"#{m['rank']} {m['user_name']}({m['pct_board']:+.2f}%)：仅 {len(take)} 只符合，已全取"
                         f"（共{len(result['candidates'])}/{TARGET_CANDIDATES}），继续下一板块")
        else:
            notes.append(f"#{m['rank']} {m['user_name']}({m['pct_board']:+.2f}%)：选出 {len(take)} 只，已凑满"
                         f" {TARGET_CANDIDATES} 只")
    # 完整排名：全部映射板块都展示（含未参与挑选的），picked 记录实际选出数量
    for m in board_metas:
        result["sectors"]["ranked"].append({
            "rank": m["rank"], "name": m["user_name"], "em_name": m["name"],
            "pct": m["pct_board"], "avg_pct": m["avg_pct"],
            "limit_up": m["limit_up"], "limit_dn": m["limit_dn"],
            "cons_n": m["cons_n"], "mode": m.get("mode"),
            "picked": m.get("picked", 0),
        })
    if len(result["candidates"]) < TARGET_CANDIDATES:
        notes.append(f"已扫完所有映射板块，候选 {len(result['candidates'])}/{TARGET_CANDIDATES} 只，不足目标数")
    # 风险示警：高盈亏比(>5)但形态关不过的票，单独展示，不计入候选、不做分析
    warnings, wseen = [], set()
    for m in board_metas:
        for w in _risk_warnings(m, klines):
            if w["code"] in wseen:
                continue
            wseen.add(w["code"])
            warnings.append(w)
    if warnings:
        try:
            wqs = market.quotes([w["code"] for w in warnings]) or {}
        except Exception:
            wqs = {}
        warnings = [w for w in warnings
                    if not wqs.get(w["code"], {}).get("total_mv_yi")
                    or wqs[w["code"]]["total_mv_yi"] <= MAX_TOTAL_CAP_YI]
    result["risk_warnings"] = warnings[:10]
    # 模型胜率/期望值增强（本地模型批次输出存在时；没有则保持只显示 RR）
    _enrich_expectancy(result["candidates"], _model_advice_index(cfg))
    # 保存观察名单
    save_watchlist(cfg, result)
    _p(f"扫描完成，选出 {len(result['candidates'])} 只候选")
    return result


def save_watchlist(cfg, data):
    p = _wl_path(cfg)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_watchlist(cfg) -> dict:
    p = _wl_path(cfg)
    if not os.path.exists(p):
        return {"candidates": []}
    with open(p, encoding="utf-8") as f:
        return json.load(f) or {"candidates": []}


# ---------------- 盘中监控 ----------------
_INTRADAY_CHECKPOINTS = [
    ("09:20", False), ("09:30", False), ("09:45", False), ("10:00", False),
    ("13:10", False), ("13:30", False), ("13:45", False), ("14:00", True),
]


def _time_passed(target: str, now: datetime) -> bool:
    h, m = map(int, target.split(":"))
    return now.hour * 60 + now.minute >= h * 60 + m


# ---------------- 冻结计划规则（plan DSL） ----------------
# 盘中触发规则源：冻结计划 JSON（stargazer_plan `python -m cli export_frozen` 的输出，
# 默认 data/plan_frozen.json，本机/监控机同一路径）。求值语义与
# stargazer_plan/plan_db/dsl.py + machine.py 逐位一致：not>and>or 优先级、
# 边缘触发（False→True）、首个 INVALIDATE 时刻 cutoff 抑制其后的 TRIGGER、
# OBSERVE 原样保留；inputs_hash = sha256(分钟区间指纹 + 规则规范化串)。
# 老的 sector_pullback_watchlist 下杀3%规则（A 模式 pct<=-3%）已被本 DSL 替代。
_TOKEN_RE = re.compile(r"\d{1,2}:\d{2}|-?\d+(?:\.\d+)?%?|>=|<=|[A-Za-z_]+|[^\s()]")
_PLAN_FIELDS = ("price", "pct_chg", "volume_ratio", "time")


class PlanDslError(ValueError):
    """计划规则 DSL 语法或语义错误。"""


def _plan_tokenize(text):
    toks, i = [], 0
    for m in _TOKEN_RE.finditer(text):
        if text[i:m.start()].strip():
            raise PlanDslError("无法识别的字符: {!r}".format(text[i:m.start()]))
        toks.append(m.group(0))
        i = m.end()
    if text[i:].strip():
        raise PlanDslError("无法识别的字符: {!r}".format(text[i:]))
    return toks


def plan_parse(text):
    """解析 DSL 字符串为 AST 元组；语法非法抛 PlanDslError（同 plan_db/dsl.py）。"""
    toks = _plan_tokenize(text)
    if not toks:
        raise PlanDslError("空规则")
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
                raise PlanDslError("缺少右括号")
            return node
        field = take()
        if field not in _PLAN_FIELDS:
            raise PlanDslError("非法字段: {!r}（可用 {}）".format(field, "/".join(_PLAN_FIELDS)))
        op = take()
        if field in ("price", "volume_ratio") and op not in (">=", "<="):
            raise PlanDslError("{} 只支持 >= / <=".format(field))
        if field == "pct_chg" and op != "<=":
            raise PlanDslError("pct_chg 只支持 <=")
        if field == "time" and op != ">=":
            raise PlanDslError("time 只支持 >=")
        val = take()
        if val is None:
            raise PlanDslError("缺少阈值")
        if field == "time":
            h, m = (val.split(":") + ["0"])[:2] if ":" in val else (None, None)
            if h is None or not (0 <= int(h) <= 23 and 0 <= int(m) <= 59):
                raise PlanDslError("非法时间: {!r}".format(val))
        else:
            try:
                float(str(val).rstrip("%"))
            except ValueError:
                raise PlanDslError("非法数值: {!r}".format(val))
        return ("cond", field, op, val)

    node = parse_or()
    if pos[0] != len(toks):
        raise PlanDslError("规则末尾有多余内容: {!r}".format(" ".join(toks[pos[0]:])))
    return node


def plan_canonical(ast):
    """AST 规范化打印：同语义规则得到同一字符串，用于 inputs_hash。"""
    if ast[0] in ("and", "or"):
        return "{} {} {}".format(plan_canonical(ast[1]), ast[0], plan_canonical(ast[2]))
    if ast[0] == "not":
        return "not ({})".format(plan_canonical(ast[1]))
    return "{} {} {}".format(ast[1], ast[2], ast[3])


def _plan_cond_series(ast, bars, pre_close):
    """在分钟 bar 列表（dict: dt/open/close/high/low/volume，dt 为 'YYYY-MM-DD HH:MM:SS'）
    上求布尔序列。pre_close 仅 pct_chg 需要；缺失抛 PlanDslError，绝不猜测。"""
    kind = ast[0]
    if kind == "and":
        a = _plan_cond_series(ast[1], bars, pre_close)
        b = _plan_cond_series(ast[2], bars, pre_close)
        return [x and y for x, y in zip(a, b)]
    if kind == "or":
        a = _plan_cond_series(ast[1], bars, pre_close)
        b = _plan_cond_series(ast[2], bars, pre_close)
        return [x or y for x, y in zip(a, b)]
    if kind == "not":
        return [not x for x in _plan_cond_series(ast[1], bars, pre_close)]
    _, field, op, val = ast
    if field == "price":
        s, x = [b["close"] for b in bars], float(val)
    elif field == "pct_chg":
        if not pre_close:
            raise PlanDslError("pct_chg 规则需要 pre_close（昨收）")
        s = [(b["close"] / float(pre_close) - 1.0) * 100.0 for b in bars]
        x = float(str(val).rstrip("%"))
    elif field == "volume_ratio":
        # 分钟量 / 当日分钟均量（含当分钟的扩展均值：只用当时及以前的信息，因果且确定性）
        s, run_sum, run_n = [], 0.0, 0
        for b in bars:
            run_sum += b["volume"]
            run_n += 1
            s.append(b["volume"] / (run_sum / run_n))
        x = float(val)
    else:  # time
        s = [b["dt"][11:16] for b in bars]  # 'HH:MM'，字符串比较与 pandas strftime 口径一致
        x = val
    return [(v >= x) if op == ">=" else (v <= x) for v in s]


def plan_edge_fired_times(cond, bars):
    """边缘触发：条件由 False（或序列首）跳变为 True 的分钟序列。
    返回 fired_at 字符串列表（'YYYY-MM-DD HH:MM:SS'，纯函数，可重放）。"""
    out, prev = [], False
    for flag, b in zip(cond, bars):
        if flag and not prev:
            out.append(b["dt"])
        prev = bool(flag)
    return out


def plan_run_rules(bars, rules, pre_close=None):
    """纯函数核心（复刻 machine.run_rules）：bars 按 dt 升序；
    rules: [(rule_type, dsl_text), ...]；
    返回 [(fired_at, rule_type), ...]，已按时间排序并做失效抑制。"""
    events = []
    for rule_type, text in rules:
        ast = plan_parse(text)
        cond = _plan_cond_series(ast, bars, pre_close)
        for t in plan_edge_fired_times(cond, bars):
            events.append((t, rule_type))
    inv_times = [t for t, rt in events if rt == "INVALIDATE"]
    if inv_times:
        cutoff = min(inv_times)
        events = [e for e in events if not (e[1] == "TRIGGER" and e[0] >= cutoff)]
    events.sort(key=lambda e: (e[0], e[1]))
    return events


def plan_inputs_hash(code, trade_day, bars, pre_close, rules):
    """分钟区间指纹 + 规则 DSL（规范化后排序拼接）的 sha256（复刻 machine.compute_inputs_hash）。"""
    fp = {
        "code": str(code),
        "trade_day": str(trade_day),
        "n_bars": int(len(bars)),
        "dt_first": str(bars[0]["dt"]),
        "dt_last": str(bars[-1]["dt"]),
        "pre_close": None if pre_close is None else float(pre_close),
        "rules": sorted(
            {"{}|{}".format(rt, plan_canonical(plan_parse(text))) for rt, text in rules}
        ),
    }
    raw = json.dumps(fp, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


# ---------- 分钟线（tools.market 只有日频/实时报价，无分钟能力；直连新浪 JSONP，
# 与 stargazer_plan/mquote.py 同接口口径：akshare stock_zh_a_minute 同款） ----------
def _sina_symbol(code):
    return ("sh" if str(code).startswith(("6", "9")) else "sz") + str(code)


def fetch_minute_bars(code, day=None, datalen=320, timeout=10):
    """拉 1 分钟 K 线并截取当日 09:00 起到现在。
    返回 [ {dt, open, close, high, low, volume}, ... ]（按 dt 升序）；失败/无数据返回 []。
    day: 'YYYYMMDD'，默认今天。"""
    day = str(day or datetime.now().strftime("%Y%m%d"))
    url = ("https://quotes.sina.cn/cn/api/jsonp_v2.php/var%20_x=/CN_MarketDataService."
           "getKLineData?symbol={}&scale=1&ma=no&datalen={}".format(_sina_symbol(code), datalen))
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        txt = urllib.request.urlopen(req, timeout=timeout).read().decode("utf-8", errors="replace")
    except Exception:
        return []
    i, j = txt.find("["), txt.rfind("]")
    if i < 0 or j <= i:
        return []
    try:
        rows = json.loads(txt[i:j + 1])
    except ValueError:
        return []
    prefix = "{}-{}-{} ".format(day[:4], day[4:6], day[6:])
    bars = []
    for r in rows or []:
        d = str(r.get("day") or "")
        if not d.startswith(prefix) or d < prefix + "09:00":
            continue
        try:
            bars.append({
                "dt": d[:19],  # 'YYYY-MM-DD HH:MM:SS'
                "open": float(r["open"]), "close": float(r["close"]),
                "high": float(r["high"]), "low": float(r["low"]),
                "volume": float(r["volume"]),
            })
        except (KeyError, TypeError, ValueError):
            continue
    return bars


# ---------- 冻结计划加载（mtime 热加载） ----------
_PLAN_CACHE = {"path": None, "mtime": None, "plan": None, "_last_missing_log": None}


def load_plan_file(path):
    """读取 export_frozen 的 JSON；缺文件/坏 JSON 抛 OSError/ValueError。"""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    sessions = []
    for s in data.get("sessions") or []:
        cands = []
        for c in s.get("candidates") or []:
            rules = []
            for rt in ("TRIGGER", "INVALIDATE", "OBSERVE"):
                r = (c.get("rules") or {}).get(rt)
                if r and r.get("dsl"):
                    rules.append((rt, r["dsl"], r.get("desc")))
            cands.append({
                "sort_order": c.get("sort_order", 0), "code": str(c.get("code") or ""),
                "name": c.get("name") or "", "pre_close": c.get("pre_close"),
                "rules": rules,
            })
        sessions.append({"session_tag": s.get("session_tag") or "",
                         "version_no": s.get("version_no"), "candidates": cands})
    return {"trading_date": str(data.get("trading_date") or ""), "sessions": sessions}


def _load_plan_cached(path):
    """按 mtime 热加载冻结计划；缺失/坏文件返回 None（每次状态变化只 log 一行）。"""
    try:
        mtime = os.path.getmtime(path)
    except OSError:
        mtime = None
    if _PLAN_CACHE["path"] == path and _PLAN_CACHE["mtime"] == mtime:
        return _PLAN_CACHE["plan"]
    _PLAN_CACHE.update(path=path, mtime=mtime, plan=None)
    import notifier
    if mtime is None:
        if _PLAN_CACHE["_last_missing_log"] != ("missing", path):
            _PLAN_CACHE["_last_missing_log"] = ("missing", path)
            notifier.log("冻结计划文件缺失 {}，盘中规则检查空转待命".format(path))
        return None
    try:
        _PLAN_CACHE["plan"] = load_plan_file(path)
        _PLAN_CACHE["_last_missing_log"] = None
        plan = _PLAN_CACHE["plan"]
        n = sum(len(s["candidates"]) for s in plan["sessions"])
        notifier.log("已加载冻结计划 {}（{} 个 session / {} 个候选，交易日 {}）".format(
            path, len(plan["sessions"]), n, plan["trading_date"] or "?"))
        return plan
    except (OSError, ValueError) as exc:
        _PLAN_CACHE["_last_missing_log"] = ("bad", path)
        notifier.log("冻结计划解析失败 {}: {}".format(path, exc))
        return None


def _plan_fired_seed(events_path):
    """从当日 jsonl 恢复已落盘事件键，重启后不重复推送/落盘。"""
    seed = set()
    try:
        with open(events_path, encoding="utf-8") as f:
            for line in f:
                try:
                    ev = json.loads(line)
                    seed.add((str(ev.get("code")), str(ev.get("rule_type")),
                              str(ev.get("fired_at"))))
                except ValueError:
                    continue
    except OSError:
        pass
    return seed


def evaluate_intraday(cfg, now: datetime | None = None) -> dict:
    """盘中按冻结计划 DSL 规则求值（替代老的下杀3%观察名单规则，见文件头 deprecated 注）。

    规则源 cfg plan.plan_file（默认 data/plan_frozen.json，相对本程序目录）；
    文件缺失/解析失败/交易日不匹配 → log 一行并返回空（不炸）。
    新事件全量追加 data/plan_events_YYYYMMDD.jsonl（格式同 cli merge_events：
    {day,session_tag,code,sort_order,rule_type,fired_at,dsl,inputs_hash,minute_rows,
    label_version}），供本机 `python -m cli merge_events` 幂等归库。
    返回本轮新事件：triggered/invalidated（各含 dsl/desc/fired_at/session_tag/
    version_no，供批量推送）、observed（只落盘）、errors（逐候选求值失败留痕）。
    """
    now = now or datetime.now()
    today = now.strftime("%Y%m%d")
    pcfg = cfg.get("plan") or {}
    rel = pcfg.get("plan_file") or "data/plan_frozen.json"
    plan_path = rel if os.path.isabs(rel) else os.path.join(BASE, rel)
    plan = _load_plan_cached(plan_path)
    empty = {"triggered": [], "invalidated": [], "observed": [], "errors": [],
             "status": "no_plan", "checked_at": now.strftime("%H:%M")}
    if plan is None:
        return empty
    day = plan["trading_date"] or today
    if day != today:
        import notifier
        notifier.log("冻结计划交易日 {} 与今日 {} 不一致，跳过盘中规则检查"
                     "（请用当日 export_frozen 输出覆盖 {}）".format(day, today, plan_path))
        return {**empty, "status": "day_mismatch"}
    events_path = os.path.join(BASE, "data", "plan_events_{}.jsonl".format(day))
    fired = _plan_fired_seed(events_path)
    triggered, invalidated, observed, errors = [], [], [], []
    for session in plan["sessions"]:
        for cand in session["candidates"]:
            code = cand["code"]
            rules = [(rt, text) for rt, text, _ in cand["rules"]]
            if not rules:
                continue
            try:
                bars = fetch_minute_bars(code, day)
                if not bars:
                    raise RuntimeError("分钟数据为空（新浪源不可得，不猜测）")
                events = plan_run_rules(bars, rules, pre_close=cand["pre_close"])
            except Exception as exc:
                errors.append("{} {}: {}".format(session["session_tag"], code, exc))
                continue
            if not events:
                continue
            inputs_hash = plan_inputs_hash(code, day, bars, cand["pre_close"], rules)
            dsl_of = {rt: text for rt, text, _ in cand["rules"]}
            desc_of = {rt: desc for rt, _, desc in cand["rules"]}
            fresh = []
            for fired_at, rule_type in events:
                key = (code, rule_type, fired_at)
                if key in fired:
                    continue
                fired.add(key)
                ev = {"day": day, "session_tag": session["session_tag"], "code": code,
                      "sort_order": cand["sort_order"], "rule_type": rule_type,
                      "fired_at": fired_at, "dsl": dsl_of[rule_type],
                      "inputs_hash": inputs_hash, "minute_rows": len(bars),
                      "label_version": "prod_v1"}
                with open(events_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(ev, ensure_ascii=False) + "\n")
                fresh.append(ev)
            for ev in fresh:
                item = {"code": code, "name": cand["name"],
                        "sort_order": cand["sort_order"], "rule_type": ev["rule_type"],
                        "fired_at": ev["fired_at"], "dsl": ev["dsl"],
                        "desc": desc_of.get(ev["rule_type"]),
                        "session_tag": session["session_tag"],
                        "version_no": session["version_no"]}
                if ev["rule_type"] == "TRIGGER":
                    triggered.append(item)
                elif ev["rule_type"] == "INVALIDATE":
                    invalidated.append(item)
                else:
                    observed.append(item)
    return {"triggered": triggered, "invalidated": invalidated, "observed": observed,
            "errors": errors, "status": "ok", "checked_at": now.strftime("%H:%M")}


# ---------------- 盘中板块内新票扫描 ----------------
_CONS_CACHE = {"date": None, "data": {}}
_USER_BOARD_CACHE = {"date": None, "data": {}}


def _user_sectors_path(cfg):
    p = os.path.join(BASE, "data", "user_watch_sectors.json")
    return p


# 上限：实时监控板块个数（前端/后端/文案三处同步）
MAX_USER_SECTORS = 20


def load_user_sectors(cfg) -> list[str]:
    try:
        with open(_user_sectors_path(cfg), encoding="utf-8") as f:
            data = json.load(f)
        return [str(n).strip() for n in (data.get("names") or []) if str(n).strip()][:MAX_USER_SECTORS]
    except (OSError, ValueError):
        return []


def _boards_cache_path(cfg):
    return os.path.join(BASE, "data", "board_list.json")


def _boards_with_cache(cfg) -> dict:
    """东财全板块列表：优先实时拉取并落盘缓存；拉取失败回退本地缓存
    （监控机可能访问不了东财接口，缓存保证映射仍可用；板块名录变化慢，
    缓存过期也用，最多映射名对不上时体现在 missing 里）。"""
    boards = None
    try:
        boards = market.board_list() or {}
    except Exception:
        boards = None
    if boards:
        try:
            with open(_boards_cache_path(cfg), "w", encoding="utf-8") as f:
                json.dump({"date": date.today().isoformat(), "boards": boards},
                          f, ensure_ascii=False)
        except OSError:
            pass
        return boards
    try:
        with open(_boards_cache_path(cfg), encoding="utf-8") as f:
            return json.load(f).get("boards") or {}
    except (OSError, ValueError):
        return {}


def resolve_user_boards(cfg) -> tuple[list, list]:
    """把用户填写的板块名（同花顺口径）映射到东财板块。
    返回 (mapped[{name,em_name,bk}], missing[name])，映射结果按日缓存。"""
    names = load_user_sectors(cfg)
    if not names:
        return [], []
    today = date.today().isoformat()
    if _USER_BOARD_CACHE.get("date") == today and _USER_BOARD_CACHE.get("key") == tuple(names):
        return _USER_BOARD_CACHE["data"], _USER_BOARD_CACHE.get("missing") or []
    boards = _boards_with_cache(cfg)
    mapped, missing = [], []
    for n in names:
        # 候选关键词：全名 → 括号内（如"共封装光学(CPO)"→"CPO"）→ 括号前前缀 → 全名
        cands = [n]
        if "(" in n:
            cands.insert(0, n.split("(", 1)[1].rstrip(")"))
            cands.append(n.split("(", 1)[0])
        em_name = None
        for kw in cands:
            if kw in boards:
                em_name = kw
                break
            hits = [b for b in boards if kw in b or b in kw]
            if hits:
                em_name = hits[0]
                break
        if em_name:
            mapped.append({"name": n, "em_name": em_name, "bk": boards[em_name]})
        else:
            missing.append(n)
    _USER_BOARD_CACHE.update(date=today, key=tuple(names), data=mapped, missing=missing)
    return mapped, missing


def save_user_sectors(cfg, names: list[str]) -> dict:
    names = [str(n).strip() for n in names if str(n).strip()][:MAX_USER_SECTORS]
    os.makedirs(os.path.dirname(_user_sectors_path(cfg)), exist_ok=True)
    with open(_user_sectors_path(cfg), "w", encoding="utf-8") as f:
        json.dump({"names": names}, f, ensure_ascii=False)
    _USER_BOARD_CACHE.update(date=None)  # 强制重解析
    mapped, missing = resolve_user_boards(cfg)
    return {"names": names, "mapped": mapped, "missing": missing}


def _board_cons_cached(bk: str) -> dict:
    """板块成分日内缓存，避免每个检查点重复请求。"""
    today = date.today().isoformat()
    if _CONS_CACHE.get("date") != today:
        _CONS_CACHE.update(date=today, data={})
    if bk not in _CONS_CACHE["data"]:
        try:
            _CONS_CACHE["data"][bk] = market.board_constituents(bk) or {}
        except Exception:
            _CONS_CACHE["data"][bk] = {}
    return _CONS_CACHE["data"][bk]


# 盘中扫描只关注流动性最好的前 N 只成分（低成交额的小票本就不符合 adv20 门槛）
_MAX_QUOTE_CONS = 120


def _minute_snapshot_hook(candidates: list, sector_name: str, trigger: str,
                          near_miss: list | None = None,
                          control_pool: list | None = None) -> None:
    """分钟快照采集（绿盘低吸 ML 训练数据，四组样本：trigger/near_miss/control/rejected）。

    只采集不阻塞：任何失败静默吞掉，不影响扫描/推送主流程。"""
    try:
        from tools import minute_snapshot as msnap
    except Exception:
        return
    try:
        triggered = set()
        for c in candidates:
            grp, extra = "trigger", {"live_pct": c.get("live_pct"),
                                     "rr_ratio": c.get("rr_ratio")}
            ev = c.get("expectancy")
            if ev is not None and ev <= 0:
                grp = "rejected"
                extra["expectancy"] = ev
                extra["reason"] = "EV<=0"
            triggered.add(c["code"])
            msnap.snapshot_async(c["code"], c.get("sector") or sector_name,
                                 trigger, grp, extra=extra)
        for item in (near_miss or [])[:5]:
            code, sec = item[0], (item[1] if len(item) > 1 else sector_name)
            if code not in triggered:
                msnap.snapshot_async(code, sec, trigger, "near_miss")
        if control_pool:
            for code in msnap.pick_controls(control_pool, triggered, 2, trigger):
                msnap.snapshot_async(code, sector_name, trigger, "control")
    except Exception:
        pass


def _quotes_fast(codes: list, workers: int = 10) -> dict:
    """多线程批量行情。market.quotes 每请求只放6只（串行太慢），这里按6只分批并发。"""
    codes = list(dict.fromkeys(codes))
    if not codes:
        return {}
    batches = [codes[i:i + 6] for i in range(0, len(codes), 6)]
    out = {}
    from concurrent.futures import as_completed
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(market.quotes, b): b for b in batches}
        for fut in as_completed(futs):
            try:
                out.update(fut.result() or {})
            except Exception:
                continue
    return out


def scan_intraday_new(cfg, max_new: int = 5, max_klines: int = 24) -> dict:
    """盘中在固定板块池内主动寻找新符合要求的个股（不限于昨日观察名单）。

    轻量流程：解析12板块→成分(日内缓存)→批量实时行情(并发)→绿盘/抗跌预筛→
    少量个股拉日K→复用完整规则(_screen_candidates/_screen_candidates_b)复验。
    通过的加入观察名单并在后续检查点参与触发判断。"""
    boards_sorted = market.board_list_sorted()
    if not boards_sorted:
        boards_sorted = [(n, b, 0.0) for n, b in market.board_list().items()]
    resolved, missing = _resolve_watch_boards(boards_sorted)
    if not resolved:
        return {"new_candidates": [], "status": "no_sectors", "missing": missing}
    wl = load_watchlist(cfg)
    known = {c["code"] for c in wl.get("candidates", [])}

    # 组装板块实时成分并批量取行情（每板块只取成交额前 MAX_QUOTE_CONS 的成分）
    sector_live = []
    for user_name, em_name, bk, pct0 in resolved:
        cons = _board_cons_cached(bk)
        if not cons:
            continue
        cons = dict(sorted(cons.items(), key=lambda kv: -(kv[1].get("amount_yi") or 0))
                    [:_MAX_QUOTE_CONS])
        mode = "panic" if pct0 < -1.5 else "strong"
        sector_live.append((user_name, em_name, bk, pct0, mode, cons))
    all_codes = sorted({c for _, _, _, _, _, cons in sector_live for c in cons})
    try:
        qs = _quotes_fast(all_codes)
    except Exception:
        qs = {}

    # 预筛：市值/名称/ST + 绿盘幅度(strong) 或 抗跌未补跌(panic)
    pre = []
    for user_name, em_name, bk, pct0, mode, cons in sector_live:
        for code, info in cons.items():
            if code in known:
                continue
            q = qs.get(code)
            if not q:
                continue
            name = str(info.get("name") or q.get("name") or "")
            if "ST" in name.upper():
                continue
            fmc = float(info.get("fmc") or 0)
            if not (MIN_CAP_YI <= fmc <= MAX_CAP_YI):
                continue
            pct = float(q.get("pct") or 0)
            if mode == "strong":
                if not (-10.0 <= pct <= -3.0):
                    continue
            else:  # panic：板块恐慌中个股红盘或微跌（未补跌）
                if not (-1.0 < pct <= 5.0):
                    continue
            pre.append((code, user_name, em_name, bk, mode, info, q))
    pre = pre[:max_klines]

    # 拉日K（线程池，量小）
    end = date.today().isoformat()
    start = (date.today() - timedelta(days=140)).isoformat()
    klines = {}
    if pre:
        from concurrent.futures import as_completed
        with ThreadPoolExecutor(max_workers=6) as ex:
            futs = {ex.submit(_kline_one, row[0], start, end): row[0] for row in pre}
            for fut in as_completed(futs):
                code, df = fut.result()
                if df is not None:
                    klines[code] = df

    # 按板块组装 live board_meta，复用完整筛选逻辑复验
    from collections import defaultdict
    groups = defaultdict(list)
    for row in pre:
        groups[(row[1], row[2], row[3], row[4])].append(row)
    now_str = datetime.now().strftime("%H:%M")
    new_cands = []
    for (user_name, em_name, bk, mode), rows in groups.items():
        live_cons, pcts = {}, []
        for code, _u, _e, _b, _m, info, q in rows:
            live_cons[code] = {
                "name": info.get("name", ""),
                "pct": float(q.get("pct") or 0),
                "price": float(q.get("price") or 0),
                "fmc": float(info.get("fmc") or 0),
                "tmc": float(q.get("total_mv_yi") or 0),
            }
            pcts.append(float(q.get("pct") or 0))
        meta = {"name": user_name, "bk": bk,
                "avg_pct": float(np.nanmean(pcts)) if pcts else 0.0,
                "cons": live_cons}
        found = (_screen_candidates_b(meta, klines) if mode == "panic"
                 else _screen_candidates(meta, "strong", klines))
        for c in found:
            c["live_price"] = live_cons[c["code"]]["price"]
            c["live_pct"] = live_cons[c["code"]]["pct"]
            c["trigger_time"] = now_str
            c["note"] = ("盘中新发现：板块内新出现符合条件的"
                         + ("绿盘回调股" if mode == "strong" else "抗跌未补跌股"))
        new_cands.extend(found)

    # 去重排序限量
    seen, dedup = set(), []
    for c in sorted(new_cands, key=lambda x: (x["rr_tier"] != "STRONG", -x["rr_ratio"])):
        if c["code"] in seen:
            continue
        seen.add(c["code"])
        dedup.append(c)
    new_candidates = dedup[:max_new]

    # 加入观察名单，后续检查点继续跟踪（附模型胜率/期望值）
    if new_candidates:
        _enrich_expectancy(new_candidates, _model_advice_index(cfg))
        wl.setdefault("candidates", []).extend(new_candidates)
        save_watchlist(cfg, wl)
    _minute_snapshot_hook(
        new_candidates, "板块池扫描", "intraday_new",
        near_miss=[(row[0], row[2]) for row in pre
                   if row[0] not in {c["code"] for c in new_candidates}],
        control_pool=all_codes)

    return {"new_candidates": new_candidates, "status": "ok",
            "prefilter_n": len(pre), "klines_n": len(klines),
            "sectors_n": len(sector_live), "missing_sectors": missing}


# ---------------- 盘前助手 ----------------
def build_premarket_state(cfg) -> dict:
    p = os.path.join(BASE, "data", "premarket_state.json")
    if os.path.exists(p):
        with open(p, encoding="utf-8") as f:
            return json.load(f)
    return {"date": date.today().isoformat(), "fields": {}, "decision": "待输入"}


def save_premarket(cfg, data: dict):
    p = os.path.join(BASE, "data", "premarket_state.json")
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def apply_premarket_decision(cfg, fields: dict) -> dict:
    """根据用户固定字段输出今日建议。"""
    decision = "继续观察"
    reasons = []
    us = float(fields.get("us_pct", 0))
    jp = float(fields.get("jp_pct", 0))
    kr = float(fields.get("kr_pct", 0))
    risk = str(fields.get("risk_news", "")).strip()
    impact = str(fields.get("sector_impact", "")).strip()
    # 规则
    if us < -2 or jp < -2 or kr < -2:
        decision = "暂停买入"
        reasons.append(f"外围大跌（美股{us:.1f}%/日股{jp:.1f}%/韩股{kr:.1f}%）")
    if risk:
        decision = "暂停买入"
        reasons.append(f"有风险新闻：{risk}")
    if impact:
        # 格式：板块名:影响程度(-1~+1)
        bad_sectors = []
        for seg in impact.replace("；", ";").split(";"):
            if ":" not in seg:
                continue
            sec, sc = seg.split(":", 1)
            try:
                if float(sc.strip()) < -0.5:
                    bad_sectors.append(sec.strip())
            except ValueError:
                continue
        if bad_sectors:
            decision = "剔除指定板块"
            reasons.append(f"以下板块受冲击：{', '.join(bad_sectors)}")
    if not reasons:
        reasons.append("外围无明显利空，按冻结计划名单正常监控")
    result = {
        "date": date.today().isoformat(),
        "fields": fields,
        "decision": decision,
        "reason": "；".join(reasons),
    }
    save_premarket(cfg, result)
    return result
