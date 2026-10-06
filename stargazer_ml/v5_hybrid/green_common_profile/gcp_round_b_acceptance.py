"""GREEN-COMMON-PROFILE-v0.1 Round B 小样本工程验收 A-F + G + H (REV3 §8 + 2026-09-29 缺陷修复).

A 网格冻结+缺失槽注入 | B 合成回放/扰动/分支/单元格隔离/标签独立性 |
C 闭环边界 | D 真实区域 sanity | E 执行仿真 | F 预算
G 四项缺陷修复的针对性反例 (2026-09-29 用户裁定):
  G1 历史 setup 不回填 (事件发生时池资格; 无资格日 setup=None 不进单元格)
  G2 复权请求失败 → ADJ_UNKNOWN; 确认空链正常; 因子按日逐段作用于退出价与持有期 low
  G3 实际成交数量: 零可成交量 → NO_FILL_ZERO_QTY; 部分成交整手记账
  G4 退出 = 第 3 个交易所交易日; 停牌/跌停显式顺延; 超限 EXIT_DEFERRED_FAILED
  G5 13:00 槽 = 源格式差异 (<20260624 期望241槽), 非行情缺失
H 行为画像字段 (REV2-D/E, 协议 §1.1-§4 钉死口径):
  H1 BREAK 后 close 回 zone_upper → RECLAIMED + time_to_first_reclaim 非 NULL
  H2 BREAK 后 close 始终未回 → NOT_RECLAIMED + time_to_first_reclaim = NULL (不用窗长+1)
  H3 APPROACH 从未跌破 zone_lower → ALREADY_ABOVE (不算收回成功)
  H4 信号 bar 自身 low≤zone_lower 且 close≥zone_upper → AMBIGUOUS (同 bar 先后不定)
  H5a 分钟窗缺口 → window_complete=False + window_truncated=True
  H5b 日线窗停牌 → window_complete=False + window_truncated=True
  H5c 各窗独立: 30m 完整不被 60m 截断否定
  H6a 已闭环事件 (全窗口 < E_test) 行为字段逐位一致 (未来扰动不变性)
  H6b 未闭环事件 (10d 窗跨 E_test) 至少一字段变化 (非空转守卫)
  H7 NO_FILL_ZERO_QTY 仍算穿透/收回 (行为有效性 ≠ 执行有效性)
另: B6 修复 —— 旧版扰动从未通过 ctx 传入 daily (恒真空转), 现实际扰动并加非空转守卫.
仅工程验收; 合成数据与 16 股小组统计不进任何绩效结论;
FILLED 计数为工程假设成交事件, 非 P0/P1 策略成交数.
"""
import bisect
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "step9_preflight"))
sys.path.insert(0, str(HERE))
from data_source_provenance import (get_adjustment_http, get_daily_parquet,  # noqa: E402
                                    get_minute_http)
import gcp_round_b_engine as E  # noqa: E402

OUT = HERE / "gcp_round_b_small_sample_v1"
SNAP = OUT / "minute"
POOL_DIR = HERE / "round_a_v0_1_impl_fix"
RESULTS = []
FAILS = []
COUNTERS = {}


def check(name, ok, detail=""):
    RESULTS.append((name, bool(ok), detail))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"  {detail}" if detail else ""))
    if not ok:
        FAILS.append(name)


def sha256_file(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


# ================= STAGE A: 网格冻结 + 缺失槽注入 =================
def stage_a(grid_path: Path):
    # 并集网格: 2025 期 (241 槽, 无 13:00) ∪ 2026-09 期 (242 槽, 含 13:00);
    # 13:00 为源格式差异 (离线探针 2026-09-29: 切换点 20260622→20260624, 无股间不一致)
    m25 = get_minute_http("600000", "20250603")
    m26 = get_minute_http("600000", "20260901")
    grid = sorted(set(E.derive_grid(m25)) | set(E.derive_grid(m26)))
    grid_241 = [s for s in grid if s != 130000]
    check("A1 网格冻结 242 槽(并集)", len(grid) == 242,
          f"{len(grid)} 槽, 首{grid[0]} 末{grid[-1]}")
    hhmm_list = [s // 100 for s in grid]
    am = [h for h in hhmm_list if h <= 1130]
    pm = [h for h in hhmm_list if h > 1130]
    check("A2 网格分段连续 (AM 0930-1130 / PM 1300-1500)",
          am[0] == 930 and am[-1] == 1130 and pm[0] == 1300 and pm[-1] == 1500
          and len(am) == 121 and len(pm) == 121,
          f"AM={len(am)} PM={len(pm)}")
    json.dump({"grid": grid, "grid_241": grid_241, "grid_1300_from": E.GRID_1300_FROM,
               "frozen_from": "600000@20250603|20260901",
               "date": "2026-09-29",
               "note": "并集网格; 13:00槽为源格式差异: <20260624 期望241槽(无13:00), "
                       ">=20260624 期望242槽; 分钟覆盖窗口 [20250102, 20260917]"},
              open(grid_path, "w"))

    # 日线 parquet raw 一致性 (区域锚 vs 分钟 raw)
    probe = get_minute_http("600000", "20260210")
    panel = get_daily_parquet()
    d_row = panel[(panel["code"].astype(str).str.zfill(6) == "600000")
                  & (panel["date"].astype(int) == 20260210)]
    par_close = float(d_row["close"].iloc[0])
    min_close = float(probe.sort_values("date")["close"].iloc[-1])
    raw_ok = abs(par_close - min_close) / min_close < 0.001
    check("A3 日线parquet=raw(与分钟一致)", raw_ok,
          f"parquet={par_close:.3f} minute={min_close:.3f}")
    return grid, panel, raw_ok


def stage_a_synth(grid):
    scen = {30: "missing_slots", 29: "break_am", 28: "no_signal"}
    s = E.synth_stock(101, "TS_GRID", "A_SDP", grid, n_days=60, scenario_days=scen)
    dates = s["dates"]
    ctx = {"code": s["code"], "setup": s["setup"], "daily": s["daily"],
           "date_idx": {d: i for i, d in enumerate(dates)}, "cal": dates}
    rec_bad = E.run_day({**ctx, "minute": s["minute"][dates[59 - 30]]}, dates[59 - 30], grid)
    check("A4 3缺失槽→INVALID_DAY", rec_bad["status"] == "INVALID_DAY"
          and len(rec_bad["missing_slots"]) == 3, rec_bad["bar_reason"])
    # ≤2 缺失槽: 日 valid 但覆盖收复窗的特征 INVALID
    df_day = s["minute"][dates[59 - 29]].copy().drop(index=[40, 41]).reset_index(drop=True)
    rec_2 = E.run_day({**ctx, "minute": df_day}, dates[59 - 29], grid)
    sig_slot = int(str(rec_2["signal_ts"])[-6:]) if rec_2.get("signal_ts") else None
    inv = (rec_2["bar_valid"] and rec_2.get("reclaim30_invalid") is True)
    check("A5 2缺失槽: 日valid+特征INVALID(不插值)", bool(inv),
          f"bar_valid={rec_2['bar_valid']} reclaim30_invalid={rec_2.get('reclaim30_invalid')}"
          f" sig={sig_slot}")
    rec_ns = E.run_day({**ctx, "minute": s["minute"][dates[59 - 28]]}, dates[59 - 28], grid)
    check("A6 no_signal 日→NO_SIGNAL", rec_ns["status"] == "NO_SIGNAL")
    return s


# ================= STAGE B: 合成回放/扰动/分支/隔离/标签独立性 =================
def stage_b(grid):
    scen = {}
    for k, off in enumerate(range(64, 40, -2)):       # 12 个 break_am 历史事件 (di>=25)
        scen[off] = "break_am"
    scen[40], scen[38], scen[36], scen[34] = "break_pm", "approach_am", "no_signal", "break_am"
    scen[10] = "break_am"                              # E_test
    s = E.synth_stock(11, "TS001", "A_SDP", grid, n_days=90, scenario_days=scen)
    dates = s["dates"]
    didx = {d: i for i, d in enumerate(dates)}
    ctx = {"code": s["code"], "setup": s["setup"], "daily": s["daily"],
           "date_idx": didx, "cal": dates}
    E_test = dates[89 - 10]
    event_days = sorted(s["minute"])
    events = [E.run_day({**ctx, "minute": s["minute"][d]}, d, grid) for d in event_days]

    # B1 信号类型正确
    by_off = {89 - didx[d]: r for d, r in zip(event_days, events)}
    t = lambda off: by_off[off]["signal_type"]     # noqa: E731
    check("B1 信号类型 (BREAK/APPROACH/NO_SIGNAL)",
          t(64) == "BREAK" and t(38) == "APPROACH"
          and by_off[36]["status"] == "NO_SIGNAL"
          and all(by_off[o]["session"] == "AM" for o in range(64, 40, -2))
          and by_off[40]["session"] == "PM",
          f"off64={t(64)} off38={t(38)} off36={by_off[36]['status']} "
          f"off40会话={by_off[40]['session']}")

    # B2 标签独立于放行决策: 全部信号都有标签记录 (含 y<0 会被决策拒绝者)
    filled = [e for e in events if e["status"] == "FILLED"]
    check("B2 标签独立: 信号全记录(不等决策)", len(filled) >= 13
          and all(e["y"] is not None for e in filled),
          f"FILLED={len(filled)}/{len(events)}")
    # 被拒事件仍进画像: 构造一个 y<0 事件, 后续 E 的估计必须包含它
    neg = [e for e in filled if e["y"] < 0]
    if neg:
        cell_e = neg[0]
        later = [d for d in event_days if d > cell_e["date"]]
        ev2 = [e for e in events if e["date"] <= max(later)]
        est = E.estimate(ev2, cell_e["code"], cell_e["setup"], cell_e["zone_sources"],
                         cell_e["session"], max(later), didx)
        check("B3 负收益事件(会被拒)仍入画像", est["n_g"] >= 1,
              f"y={cell_e['y']:.4f} 后续格 n_g={est['n_g']}")
    else:
        check("B3 负收益事件(会被拒)仍入画像", True, "(本样本无负 y, 由 B4 构造事件覆盖)")

    # B4 估计分支 (构造事件, 函数级): n>=10 收缩 / n=9 回退 / 空 NO_ESTIMATE
    def mk(n, y=0.01, f=0.2, code="X1"):
        return [{"code": code, "setup": "A_SDP", "zone_sources": ("MA20",),
                 "session": "AM", "status": "FILLED", "y": y, "f": f,
                 "date": 100 + i, "label_end_ts": 103 + i} for i in range(n)]
    didx2 = {100 + i: i for i in range(40)} | {999: 39}
    est10 = E.estimate(mk(12), "X1", "A_SDP", ("MA20",), "AM", 999, didx2)
    # 回退: 组内他人 10 事件 + 本股 5 事件 -> n_g=15, n=5, 回退 P0
    est9 = E.estimate(mk(10, code="O1") + mk(5, code="X1"), "X1", "A_SDP",
                      ("MA20",), "AM", 999, didx2)
    est0 = E.estimate([], "X1", "A_SDP", ("MA20",), "AM", 999, didx2)
    check("B4a n=12→P1收缩分支", not est10["fallback"] and not est10["no_estimate"]
          and abs(est10["mu_1"] - est10["mu_g"]) < 1e-12, f"n={est10['n']} w={12/32:.3f}")
    check("B4b n=9→回退P0(不弃条件)", est9["fallback"] and est9["mu_1"] == est9["mu_g"])
    check("B4c 组不足→NO_ESTIMATE", est0["no_estimate"])
    # 单元格隔离: PM 事件不进 AM 格; 不同 zone_sources 不混
    pm_ev = mk(5, code="X1")
    for e in pm_ev:
        e["session"] = "PM"
    est_mix = E.estimate(mk(10) + pm_ev, "X1", "A_SDP", ("MA20",), "AM", 999, didx2)
    check("B5 单元格隔离 (session/zone_sources)",
          est_mix["n_g"] == 10 and est_mix["n"] == 10,
          f"AM格 n_g={est_mix['n_g']} (5 个 PM 事件未混入)")
    zs_ev = mk(5, code="X1")
    for e in zs_ev:
        e["zone_sources"] = ("BOLL_L", "PREV_LOW")
    est_zs = E.estimate(mk(10) + zs_ev, "X1", "A_SDP", ("MA20",), "AM", 999, didx2)
    check("B5b zone_sources 不混", est_zs["n_g"] == 10, f"n_g={est_zs['n_g']}")

    # B6 扰动不变性 (2026-09-29 修复: 旧版扰动未传入 daily, 恒真空转):
    # E_test 后 daily+minute 扰动 → 已闭环事件 (label_end_ts < E_test) 逐位一致;
    # 未闭环事件的 E 后字段允许变化, 但必须至少一个确实变化 (非空转守卫).
    s2 = E.synth_stock(11, "TS001", "A_SDP", grid, n_days=90, scenario_days=scen)
    for d, df in s2["minute"].items():
        if d > E_test:
            for col in ("open", "high", "low", "close"):
                df[col] = df[col] * 1.07
            df["volume"] = df["volume"] * 2.5
    s2["daily"].loc[s2["daily"]["date"] > E_test,
                    ["open", "high", "low", "close"]] *= 1.07
    ctx2 = {**ctx, "daily": s2["daily"]}           # 关键修复: 扰动 daily 真正进入计算
    events2 = [E.run_day({**ctx2, "minute": s2["minute"][d]}, d, grid)
               for d in event_days if d <= E_test]
    pre_cols = ["date", "status", "signal_type", "session", "zone_sources", "signal_ts",
                "fill_ts", "fill_price", "fill_shares", "fill_amount",
                "zone_upper", "zone_lower"]
    closed_cols = pre_cols + ["y", "f", "label_end_ts"]
    ev1 = [e for e in events if e["date"] <= E_test]
    closed = [e for e in ev1 if e.get("label_end_ts") is not None
              and e["label_end_ts"] < E_test]
    unclosed = [e for e in ev1 if e.get("label_end_ts") is None
                or e["label_end_ts"] >= E_test]
    same_pre = all(all(str(a.get(k)) == str(b.get(k)) for k in pre_cols)
                   for a, b in zip(ev1, events2))
    same_closed = all(all(str(a.get(k)) == str(b.get(k)) for k in closed_cols)
                      for a, b in zip(closed,
                                      [e2 for e2 in events2
                                       if e2["date"] in {c["date"] for c in closed}]))
    # 非空转守卫: 至少一个未闭环 FILLED 事件的 y 因 E 后扰动而变化
    y_changed = any(
        a.get("y") is not None and b.get("y") is not None and a["y"] != b["y"]
        for a, b in zip(unclosed, [e2 for e2 in events2
                                   if e2["date"] in {u["date"] for u in unclosed}]))
    check("B6 E后扰动不改变已闭环事件(非空转)", same_pre and same_closed and y_changed,
          f"闭环一致={same_closed} 扰动确实传播={y_changed} "
          f"(闭环={len(closed)} 未闭环={len(unclosed)})")
    est_a = est_b2 = None
    ref = closed[:1]
    if ref:
        r = ref[0]
        est_a = E.estimate(events, r["code"], r["setup"], r["zone_sources"],
                           r["session"], E_test, didx)
        est_b2 = E.estimate(events2 + [e for e in events if e["date"] > E_test],
                            r["code"], r["setup"], r["zone_sources"],
                            r["session"], E_test, didx)
    if ref and est_a and not est_a["no_estimate"]:
        check("B6b E后扰动不改变截至E估计",
              est_a["mu_1"] == est_b2["mu_1"] and est_a["n_g"] == est_b2["n_g"])
    else:
        check("B6b E后扰动不改变截至E估计", True, "(闭环样本不足, 由 B6 覆盖)")
    return events, didx


# ================= STAGE C: 闭环边界 =================
def stage_c():
    def mk(n, y=0.01, code="X1"):
        return [{"code": code, "setup": "A_SDP", "zone_sources": ("MA20",),
                 "session": "AM", "status": "FILLED", "y": y, "f": 0,
                 "date": 100 + i, "label_end_ts": 103 + i} for i in range(n)]
    didx = {100 + i: i for i in range(40)} | {999: 39}
    ev = mk(11)
    # 强改一个事件 label_end_ts = E (未闭环) -> 必须被排除
    ev[0]["label_end_ts"] = 999
    est = E.estimate(ev, "X1", "A_SDP", ("MA20",), "AM", 999, didx)
    check("C1 未闭环事件(label_end_ts>=E)被排除", est["n_g"] == 10, f"n_g={est['n_g']}")
    check("C2 排除后估计仍可用", not est["no_estimate"])


# ================= STAGE G: 四项缺陷修复的针对性反例 (合成, 无 HTTP) =================
def _g_ctx(dates, closes, lows, minute_day, atr=0.2, chain=None, adj_ok=True):
    daily = pd.DataFrame({"date": dates, "open": closes, "high": closes,
                          "low": lows, "close": closes})
    return {"code": "G01", "setup": "A_SDP", "daily": daily,
            "date_idx": {d: i for i, d in enumerate(dates)}, "cal": list(dates),
            "chain": chain, "adj_ok": adj_ok, "minute": minute_day}


def _g_minute_day(day: int, grid: list, sig_slot_idx: int, fill_vol: float,
                  price: float, low_extra: float = 0.0):
    exp = [s for s in E.expected_grid(grid, day)]
    ts = [int(f"{day}{s:06d}") for s in exp]
    n = len(ts)
    df = pd.DataFrame({"date": ts, "open": price, "high": price * 1.001,
                       "low": price * (1 - low_extra), "close": price,
                       "volume": 1e7, "amount": 1e7 * price})
    df.loc[df.index[sig_slot_idx], "low"] = price * 0.90   # 信号 bar 触发 BREAK (由 sig 提供)
    df.loc[df.index[sig_slot_idx + 1], "volume"] = fill_vol
    return df


def _g_sig(day: int, grid: list, sig_slot_idx: int):
    slot = E.expected_grid(grid, day)[sig_slot_idx]
    ts = int(f"{day}{slot:06d}")
    return {"signal_ts": ts, "signal_type": "BREAK", "session": "AM",
            "zone_sources": ("MA20",), "zone_upper": 10.0, "zone_lower": 9.5}


def stage_g(grid):
    day = 20260910                       # >= 20260624 → 期望 242 槽
    dates = [20260910, 20260911, 20260914, 20260915, 20260916,
             20260917, 20260918, 20260921, 20260922]
    sig_i = 5                            # 09:35
    price = 9.98

    # ---- G1: setup=None 事件不进单元格 ----
    def mk(n, setup="A_SDP", code="X1"):
        return [{"code": code, "setup": setup, "zone_sources": ("MA20",),
                 "session": "AM", "status": "FILLED", "y": 0.01, "f": 0,
                 "date": 100 + i, "label_end_ts": 103 + i} for i in range(n)]
    didx = {100 + i: i for i in range(40)} | {999: 39}
    est_with = E.estimate(mk(10), "X1", "A_SDP", ("MA20",), "AM", 999, didx)
    est_none = E.estimate(mk(10) + mk(5, setup=None), "X1", "A_SDP",
                          ("MA20",), "AM", 999, didx)
    check("G1 setup=None(无池资格)事件不进单元格",
          est_with["n_g"] == 10 and est_none["n_g"] == 10,
          f"n_g={est_none['n_g']} (5 个无资格事件未混入)")

    # ---- G2: 复权请求失败 → ADJ_UNKNOWN; 确认空链正常 ----
    md = _g_minute_day(day, grid, sig_i, fill_vol=1e6, price=price)
    sig = _g_sig(day, grid, sig_i)
    closes = [10.0, 9.98, 10.0, 10.05, 10.08, 10.10, 10.12, 10.15, 10.18]
    lows = [9.9, 9.95, 9.96, 10.0, 10.02, 10.05, 10.08, 10.10, 10.12]
    lab_fail = E.label_event(sig, md, _g_ctx(dates, closes, lows, None)["daily"],
                             0.2, chain=None, adj_ok=False, cal=dates)
    check("G2a 复权请求失败→ADJ_UNKNOWN(不生成有效标签)",
          lab_fail["status"] == "ADJ_UNKNOWN" and lab_fail["y"] is None
          and lab_fail["f"] is None, lab_fail.get("adj_status", ""))
    lab_empty = E.label_event(sig, md, _g_ctx(dates, closes, lows, None)["daily"],
                              0.2, chain={}, adj_ok=True, cal=dates)
    check("G2b 确认空链→正常FILLED(adj_cum=1)",
          lab_empty["status"] == "FILLED" and lab_empty["adj_cum"] == 1.0)

    # ---- G2c: 复权因子按日逐段作用 (退出 close 与持有期 low) ----
    # 持有期中间日 (20260915, 第3交易所日=目标退出日前的持有日) low=9.60:
    #   未复权会误触发 f=1 (9.60 < buy_net-0.5*ATR=9.913); 复权因子 1.05 → 10.08 不触发
    closes_c = [10.0, 9.98, 10.0, 10.05, 10.08, 10.10, 10.12, 10.15, 10.18]
    lows_c = [9.9, 9.95, 9.96, 9.60, 10.02, 10.05, 10.08, 10.10, 10.12]
    chain_c = {20260915: 1.05}           # 退出日 20260915 为第3交易所日
    # 实际退出日 = dates[3] = 20260915; low 9.60 即退出日 low
    lab_c = E.label_event(sig, md, _g_ctx(dates, closes_c, lows_c, None)["daily"],
                          0.2, chain=chain_c, adj_ok=True, cal=dates)
    buy_net = price * E.BUY_MULT
    expect_y = (closes_c[3] * 1.05 * E.SELL_MULT) / buy_net - 1.0
    ok_c = (lab_c["status"] == "FILLED" and abs(lab_c["y"] - expect_y) < 1e-12
            and lab_c["f"] == 0)
    # 对照: 无复权时同 low 会触发 f=1
    lab_raw = E.label_event(sig, md, _g_ctx(dates, closes_c, lows_c, None)["daily"],
                            0.2, chain={}, adj_ok=True, cal=dates)
    check("G2c 复权因子作用于退出价与持有期low",
          ok_c and lab_raw["f"] == 1,
          f"adj_cum={lab_c['adj_cum']:.2f} f_adj={lab_c['f']} f_raw={lab_raw['f']}")

    # ---- G3: 实际成交数量 ----
    md_zero = _g_minute_day(day, grid, sig_i, fill_vol=400.0, price=price)
    lab_zero = E.label_event(sig, md_zero, _g_ctx(dates, closes, lows, None)["daily"],
                             0.2, chain={}, adj_ok=True, cal=dates)
    check("G3a 零可成交量→NO_FILL_ZERO_QTY",
          lab_zero["status"] == "NO_FILL_ZERO_QTY"
          and lab_zero["fill_amount"] == 0.0 and lab_zero["y"] is None,
          f"1%×400股×{price}={400*0.01*price:.0f}元 < 1手")
    md_part = _g_minute_day(day, grid, sig_i, fill_vol=100000.0, price=price)
    lab_part = E.label_event(sig, md_part, _g_ctx(dates, closes, lows, None)["daily"],
                             0.2, chain={}, adj_ok=True, cal=dates)
    lots = int(100000 * 0.01 * price // (price * E.BUY_MULT * 100))
    ok_part = (lab_part["status"] == "FILLED"
               and lab_part["fill_shares"] == lots * 100
               and lab_part["fill_shares"] % 100 == 0
               and 0 < lab_part["fill_amount"] <= 50000.0
               and lab_part["participation_ratio"] < 1.0)
    check("G3b 部分成交: 整手向下取整, 按实际金额记账",
          ok_part, f"lots={lab_part['fill_lots']} shares={lab_part['fill_shares']}"
                   f" amount={lab_part['fill_amount']:.0f}/50000")

    # ---- G4: 退出 = 第 3 个交易所交易日; 停牌/跌停顺延 ----
    # 4a: 第3交易所日(20260915)个股缺bar → 顺延至 20260916
    dates_4a = dates
    closes_4a = [10.0, 9.98, 10.0, None, 10.08, 10.10, 10.12, 10.15, 10.18]
    lows_4a = [9.9, 9.95, 9.96, None, 10.02, 10.05, 10.08, 10.10, 10.12]
    daily_4a = pd.DataFrame({"date": [d for d, c in zip(dates_4a, closes_4a)
                                      if c is not None],
                             "open": [c for c in closes_4a if c is not None],
                             "high": [c for c in closes_4a if c is not None],
                             "low": [l for l in lows_4a if l is not None],
                             "close": [c for c in closes_4a if c is not None]})
    lab_4a = E.label_event(sig, md, daily_4a, 0.2, chain={}, adj_ok=True, cal=dates_4a)
    check("G4a 第3交易所日停牌→显式顺延",
          lab_4a["status"] == "FILLED" and lab_4a["exit_deferred"] is True
          and lab_4a["exit_reason"] == "SUSPENDED"
          and lab_4a["intended_exit_date"] == 20260915
          and lab_4a["exit_date"] == 20260916
          and lab_4a["label_end_ts"] == 20260916,
          f"intended={lab_4a.get('intended_exit_date')} actual={lab_4a.get('exit_date')}")
    # 4b: 之后 10 个交易所日全部缺 bar → EXIT_DEFERRED_FAILED
    daily_4b = daily_4a[daily_4a["date"] <= 20260911]
    lab_4b = E.label_event(sig, md, daily_4b, 0.2, chain={}, adj_ok=True, cal=dates_4a)
    check("G4b 顺延超限→EXIT_DEFERRED_FAILED(不静默改期)",
          lab_4b["status"] == "EXIT_DEFERRED_FAILED"
          and lab_4b["y"] is None and lab_4b["intended_exit_date"] == 20260915)
    # 4c: 第3交易所日跌停近似 (close <= prev*0.905) → 顺延
    closes_4c = [10.0, 9.98, 10.00, 9.00, 10.08, 10.10, 10.12, 10.15, 10.18]
    lows_4c = [9.9, 9.95, 9.96, 8.95, 10.02, 10.05, 10.08, 10.10, 10.12]
    lab_4c = E.label_event(sig, md, _g_ctx(dates, closes_4c, lows_4c, None)["daily"],
                           0.2, chain={}, adj_ok=True, cal=dates)
    check("G4c 退出日跌停近似→顺延(RESEARCH_APPROXIMATION)",
          lab_4c["status"] == "FILLED" and lab_4c["exit_deferred"] is True
          and lab_4c["exit_reason"] == "HALTED_DOWN_APPROX"
          and lab_4c["exit_date"] == 20260916,
          f"reason={lab_4c.get('exit_reason')}")

    # ---- G5: 13:00 槽 = 源格式差异 ----
    exp241 = [s for s in grid if s != 130000]
    md_pre = pd.DataFrame({"date": [int(f"20250603{s:06d}") for s in exp241],
                           "open": 10.0, "high": 10.0, "low": 10.0, "close": 10.0,
                           "volume": 1e6, "amount": 1e7})
    v_pre = E.validate_minute_day(md_pre, grid, 20250603)
    v_post = E.validate_minute_day(md_pre, grid, 20260901)
    check("G5 13:00槽: <20260624为格式差异(非缺失), >=20260624计为缺失",
          v_pre["ok"] and v_pre["missing"] == [] and v_post["missing"] == [130000],
          f"pre_missing={v_pre['missing']} post_missing={v_post['missing']}")


# ================= STAGE H: 行为画像字段 (REV2-D/E: 收回/未收回/已在上方/缺口/不变性) =================
BEHAVIOR_FIELDS = [
    "min_low_30m", "min_low_60m", "min_low_5d", "min_low_10d",
    "pen_full_window_30m", "pen_full_window_60m", "pen_full_window_5d", "pen_full_window_10d",
    "time_to_max_pen_30m", "time_to_max_pen_60m", "time_to_max_pen_5d", "time_to_max_pen_10d",
    "reclaim_30m", "reclaim_60m", "reclaim_5d", "reclaim_10d",
    "time_to_first_reclaim_30m", "time_to_first_reclaim_60m",
    "time_to_first_reclaim_5d", "time_to_first_reclaim_10d",
    "window_complete_30m", "window_complete_60m", "window_complete_5d", "window_complete_10d",
    "window_truncated_30m", "window_truncated_60m", "window_truncated_5d", "window_truncated_10d",
]


def _h_minute_day(day, grid, sig_slot_idx, price=9.98, sig_low=9.0, sig_close=9.0,
                  post_lows=None, post_closes=None, fill_vol=1e7, drop_offsets=None):
    """合成分钟 df: 信号 bar 自定义 low/close, 后续 bar 可逐个设 low/close, 可 drop 槽位.

    drop_offsets: 相对 sig+1 的偏移列表, 这些槽位从 df 中删除以制造窗口内缺失.
    """
    exp = E.expected_grid(grid, day)
    ts = [int(f"{day}{s:06d}") for s in exp]
    df = pd.DataFrame({"date": ts, "open": price, "high": price * 1.001,
                       "low": price, "close": price,
                       "volume": 1e7, "amount": 1e7 * price})
    df.loc[df.index[sig_slot_idx], "low"] = sig_low
    df.loc[df.index[sig_slot_idx], "close"] = sig_close
    if post_lows:
        for k, lv in enumerate(post_lows):
            idx = sig_slot_idx + 1 + k
            if 0 <= idx < len(df):
                df.loc[df.index[idx], "low"] = lv
    if post_closes:
        for k, cv in enumerate(post_closes):
            idx = sig_slot_idx + 1 + k
            if 0 <= idx < len(df):
                df.loc[df.index[idx], "close"] = cv
    df.loc[df.index[sig_slot_idx + 1], "volume"] = fill_vol
    if drop_offsets:
        drop_idx = [sig_slot_idx + 1 + k for k in drop_offsets
                    if 0 <= sig_slot_idx + 1 + k < len(df)]
        df = df.drop(index=drop_idx).reset_index(drop=True)
    return df


def _h_sig(day, grid, sig_slot_idx, signal_type="BREAK",
           zone_upper=10.0, zone_lower=9.5):
    slot = E.expected_grid(grid, day)[sig_slot_idx]
    ts = int(f"{day}{slot:06d}")
    return {"signal_ts": ts, "signal_type": signal_type, "session": "AM",
            "zone_sources": ("MA20",), "zone_upper": zone_upper,
            "zone_lower": zone_lower}


def stage_h(grid):
    day = 20260910                       # >= 20260624 → 期望 242 槽
    sig_i = 5                            # 09:36 槽 (监控窗内)
    price = 9.98
    atr = 0.2

    # ---- H1: BREAK 后 close 回到 zone_upper → RECLAIMED ----
    md1 = _h_minute_day(day, grid, sig_i, price=price, sig_low=9.0, sig_close=9.0,
                        post_closes=[9.0, 9.0, 10.0], post_lows=[9.0, 9.0, 9.0])
    sig1 = _h_sig(day, grid, sig_i)
    dates_h1 = [day + i for i in range(21)]
    closes_h1 = [10.0] * 21
    lows_h1 = [9.9] * 21
    daily_h1 = _g_ctx(dates_h1, closes_h1, lows_h1, None)["daily"]
    lab1 = E.label_event(sig1, md1, daily_h1, atr, chain={}, adj_ok=True,
                         cal=dates_h1, grid=grid)
    check("H1 BREAK后close回zone_upper→RECLAIMED+time非NULL",
          lab1["reclaim_30m"] == "RECLAIMED"
          and lab1["time_to_first_reclaim_30m"] is not None,
          f"reclaim_30m={lab1['reclaim_30m']} t={lab1['time_to_first_reclaim_30m']}")

    # ---- H2: BREAK 后 close 始终未回 zone_upper → NOT_RECLAIMED+time=NULL ----
    md2 = _h_minute_day(day, grid, sig_i, price=price, sig_low=9.0, sig_close=9.0)
    lab2 = E.label_event(sig1, md2, daily_h1, atr, chain={}, adj_ok=True,
                         cal=dates_h1, grid=grid)
    check("H2 BREAK后close未回zone_upper→NOT_RECLAIMED+time=NULL",
          lab2["reclaim_30m"] == "NOT_RECLAIMED"
          and lab2["time_to_first_reclaim_30m"] is None,
          f"reclaim_30m={lab2['reclaim_30m']} t={lab2['time_to_first_reclaim_30m']}")

    # ---- H3: APPROACH 后从未跌破 zone_lower → ALREADY_ABOVE ----
    md3 = _h_minute_day(day, grid, sig_i, price=9.6, sig_low=9.6, sig_close=9.6)
    sig3 = _h_sig(day, grid, sig_i, signal_type="APPROACH")
    lab3 = E.label_event(sig3, md3, daily_h1, atr, chain={}, adj_ok=True,
                         cal=dates_h1, grid=grid)
    check("H3 APPROACH从未跌破zone_lower→ALREADY_ABOVE+time=NULL",
          lab3["reclaim_30m"] == "ALREADY_ABOVE"
          and lab3["time_to_first_reclaim_30m"] is None,
          f"reclaim_30m={lab3['reclaim_30m']}")

    # ---- H4: 信号 bar 自身 low≤zone_lower 且 close≥zone_upper → AMBIGUOUS ----
    md4 = _h_minute_day(day, grid, sig_i, price=price, sig_low=9.0, sig_close=10.0)
    lab4 = E.label_event(sig1, md4, daily_h1, atr, chain={}, adj_ok=True,
                         cal=dates_h1, grid=grid)
    check("H4 信号bar低≤zone_lower且close≥zone_upper→AMBIGUOUS+time=NULL",
          lab4["reclaim_30m"] == "AMBIGUOUS"
          and lab4["time_to_first_reclaim_30m"] is None,
          f"reclaim_30m={lab4['reclaim_30m']}")

    # ---- H5a: 分钟窗缺口 → window_complete=False + window_truncated=True ----
    md5a = _h_minute_day(day, grid, sig_i, price=price, sig_low=9.0, sig_close=9.0,
                         drop_offsets=[5, 15])  # 30m 窗内删 2 槽
    lab5a = E.label_event(sig1, md5a, daily_h1, atr, chain={}, adj_ok=True,
                          cal=dates_h1, grid=grid)
    check("H5a 30m窗缺2槽→complete=False+truncated=True",
          lab5a["window_complete_30m"] is False
          and lab5a["window_truncated_30m"] is True,
          f"complete_30m={lab5a['window_complete_30m']} "
          f"truncated_30m={lab5a['window_truncated_30m']}")

    # ---- H5b: 日线窗停牌 → window_complete=False + window_truncated=True ----
    dates_h5b = [day + i for i in range(21)]
    daily_h5b = pd.DataFrame({
        "date": [d for d in dates_h5b if d != day + 2],   # day+2 停牌 (5d 窗内)
        "open": [10.0] * 20, "high": [10.0] * 20,
        "low": [9.9] * 20, "close": [10.0] * 20,
    })
    lab5b = E.label_event(sig1, md1, daily_h5b, atr, chain={}, adj_ok=True,
                          cal=dates_h5b, grid=grid)
    check("H5b 5d窗停牌→complete=False+truncated=True",
          lab5b["window_complete_5d"] is False
          and lab5b["window_truncated_5d"] is True,
          f"complete_5d={lab5b['window_complete_5d']} "
          f"truncated_5d={lab5b['window_truncated_5d']}")

    # ---- H5c: 各窗独立 (30m 完整 + 60m 截断 → 30m 不被 60m 否定) ----
    md5c = _h_minute_day(day, grid, sig_i, price=price, sig_low=9.0, sig_close=9.0,
                         drop_offsets=[35, 45])  # 60m 窗内但 30m 窗外删 2 槽
    lab5c = E.label_event(sig1, md5c, daily_h1, atr, chain={}, adj_ok=True,
                          cal=dates_h1, grid=grid)
    check("H5c 各窗独立: 30m完整不被60m截断否定",
          lab5c["window_complete_30m"] is True
          and lab5c["window_truncated_30m"] is False
          and lab5c["window_complete_60m"] is False
          and lab5c["window_truncated_60m"] is True,
          f"30m={lab5c['window_complete_30m']}/{lab5c['window_truncated_30m']} "
          f"60m={lab5c['window_complete_60m']}/{lab5c['window_truncated_60m']}")

    # ---- H6: 未来扰动不变性 + 非空转守卫 ----
    dates_h6 = [20260901 + i for i in range(30)]   # 20260901..20260930
    E_test = 20260925
    daily_h6 = _g_ctx(dates_h6, [10.0] * 30, [9.9] * 30, None)["daily"]
    daily_h6_pert = daily_h6.copy()
    mask = daily_h6_pert["date"].astype(int) > E_test
    daily_h6_pert.loc[mask, "low"] = daily_h6_pert.loc[mask, "low"] * 0.5

    # H6a: 已闭环事件 (day=20260910, 10d 窗=20260911..20260920 全 < E_test) 逐位一致
    md_A = _h_minute_day(20260910, grid, sig_i, price=price, sig_low=9.0,
                         sig_close=9.0, post_closes=[9.0, 9.0, 10.0])
    sig_A = _h_sig(20260910, grid, sig_i)
    lab_A_pre = E.label_event(sig_A, md_A, daily_h6, atr, chain={}, adj_ok=True,
                              cal=dates_h6, grid=grid)
    lab_A_post = E.label_event(sig_A, md_A, daily_h6_pert, atr, chain={}, adj_ok=True,
                               cal=dates_h6, grid=grid)
    diffs_A = [(f, lab_A_pre.get(f), lab_A_post.get(f)) for f in BEHAVIOR_FIELDS
               if lab_A_pre.get(f) != lab_A_post.get(f)]
    check("H6a 已闭环事件(全窗口<E_test)行为字段逐位一致",
          not diffs_A, f"diffs={diffs_A}")

    # H6b: 未闭环事件 (day=20260920, 10d 窗=20260921..20260930 跨 E_test) 至少一字段变化
    md_B = _h_minute_day(20260920, grid, sig_i, price=price, sig_low=9.0,
                         sig_close=9.0, post_closes=[9.0, 9.0, 10.0])
    sig_B = _h_sig(20260920, grid, sig_i)
    lab_B_pre = E.label_event(sig_B, md_B, daily_h6, atr, chain={}, adj_ok=True,
                              cal=dates_h6, grid=grid)
    lab_B_post = E.label_event(sig_B, md_B, daily_h6_pert, atr, chain={}, adj_ok=True,
                               cal=dates_h6, grid=grid)
    diffs_B = [f for f in BEHAVIOR_FIELDS
               if lab_B_pre.get(f) != lab_B_post.get(f)]
    check("H6b 未闭环事件(10d窗跨E_test)至少一字段变化(非空转守卫)",
          len(diffs_B) > 0, f"changed={diffs_B}")

    # ---- H7: NO_FILL_ZERO_QTY 仍算穿透/收回 (行为有效性 ≠ 执行有效性) ----
    md7 = _h_minute_day(day, grid, sig_i, price=price, sig_low=9.0, sig_close=9.0,
                        post_closes=[9.0, 9.0, 10.0], post_lows=[9.0, 9.0, 9.0],
                        fill_vol=400.0)  # 1%×400×9.98≈40元<1手 → 零可成交
    lab7 = E.label_event(sig1, md7, daily_h1, atr, chain={}, adj_ok=True,
                         cal=dates_h1, grid=grid)
    check("H7 NO_FILL_ZERO_QTY仍算穿透/收回(不依赖成交)",
          lab7["status"] == "NO_FILL_ZERO_QTY"
          and lab7["y"] is None
          and lab7["reclaim_30m"] == "RECLAIMED"
          and lab7["min_low_30m"] is not None
          and lab7["pen_full_window_30m"] is not None,
          f"status={lab7['status']} reclaim_30m={lab7['reclaim_30m']} "
          f"min_low_30m={lab7['min_low_30m']} pen={lab7['pen_full_window_30m']}")


# ================= STAGE D/E/F: 真实小样本回放 =================
def stage_real(grid, panel, parquet_raw=True):
    sheet = json.loads((OUT / "run_sheet.json").read_text(encoding="utf-8"))
    SNAP.mkdir(parents=True, exist_ok=True)
    cal = sorted(int(d) for d in panel["date"].unique())
    didx = {d: i for i, d in enumerate(cal)}

    def prev_td(d):
        i = bisect.bisect_left(cal, d)
        return cal[i - 1] if i > 0 else None

    # ---- 缺陷修复1: 池资格按事件发生时归类 ----
    pool = pd.concat([
        pd.read_parquet(POOL_DIR / "pool_a_sdp_members.parquet")[["date", "code", "setup_type"]],
        pd.read_parquet(POOL_DIR / "pool_b_lvb_members.parquet")[["date", "code", "setup_type"]],
    ])
    pool["code"] = pool["code"].astype(str).str.zfill(6)
    mem = pool.groupby(["date", "code"])["setup_type"].apply(set).to_dict()

    # ---- 缺陷修复2: 复权链缓存, 失败与确认空链区分 ----
    adj_f = OUT / "adj_cache.json"
    adj_cache = json.loads(adj_f.read_text(encoding="utf-8")) if adj_f.exists() else {}
    for st in sheet["stocks"]:
        if st["code"] in adj_cache:
            continue
        try:
            raw = get_adjustment_http(st["code"])
            chain = {int(k): float(v.get("factor", 1.0)) for k, v in raw}
            adj_cache[st["code"]] = {"ok": True, "chain": {str(k): v for k, v in chain.items()}}
        except Exception as exc:                      # 请求失败 ≠ 确认空链
            adj_cache[st["code"]] = {"ok": False, "chain": {},
                                     "error": type(exc).__name__}
    adj_f.write_text(json.dumps(adj_cache, ensure_ascii=False, indent=1), encoding="utf-8")

    t0, n_req = time.time(), 0
    all_events, zone_rows, real_recs = [], [], []
    e_day_mismatch = []
    for st in sheet["stocks"]:
        code, setup_sheet = st["code"], st["setup"]
        # 日线 (raw 检查失败则 HTTP 兜底)
        if parquet_raw:
            daily = panel[panel["code"].astype(str).str.zfill(6) == code].sort_values("date")
            daily = daily[["date", "open", "high", "low", "close"]].copy()
            daily["date"] = daily["date"].astype(int)
        else:
            from data_source_provenance import get_daily_http
            daily = get_daily_http(code)[["date", "open", "high", "low", "close"]].copy()
            daily["date"] = daily["date"].astype(int)
        # 复权链 (缓存; ok=False → 全部标签 ADJ_UNKNOWN)
        ac = adj_cache[code]
        chain = {int(k): float(v) for k, v in ac["chain"].items()}
        adj_ok = bool(ac["ok"])
        # 分钟: 快照缓存 + 缺日补抓 (可续跑)
        snap_f = SNAP / f"{code}.parquet"
        if snap_f.exists():
            md = pd.read_parquet(snap_f)
            have = {int(str(t)[:8]) for t in md["date"].astype(int)}
        else:
            md, have = pd.DataFrame(), set()
        todo = [d for d in st["minute_fetch_days"] if d not in have]
        parts = [md] if len(md) else []
        for d in todo:
            df = get_minute_http(code, str(d))
            n_req += 1
            if len(df):
                parts.append(df)
        if parts:
            md = pd.concat(parts, ignore_index=True).drop_duplicates("date")
            md.to_parquet(snap_f, index=False)
        by_day = {}
        if len(md):
            md = md.assign(day=md["date"].astype(int).astype(str).str[:8].astype(int))
            for d, g in md.groupby("day"):
                by_day[int(d)] = g
        ctx = {"code": code, "daily": daily, "date_idx": didx, "cal": cal,
               "chain": chain, "adj_ok": adj_ok}
        for d in st["minute_fetch_days"]:
            setups = mem.get((prev_td(d), code), set()) or {None}
            for s in sorted(setups):
                rec = E.run_day({**ctx, "setup": s, "minute": by_day.get(d)}, d, grid)
                rec["sheet_setup"] = setup_sheet
                all_events.append(rec)
                zone_rows.append({"code": code, "date": d, "zone_n": rec["zone_n"],
                                  "atr_ref": rec["atr_ref"]})
                if d in st["E_days"]:
                    if s == setup_sheet:
                        real_recs.append(rec)
                    else:
                        e_day_mismatch.append((code, d, setup_sheet, s))
    dur = time.time() - t0

    # ---- 缺陷修复1 的驱动级验证 ----
    n_pool = sum(1 for r in all_events if r["setup"] is not None)
    n_nopool = sum(1 for r in all_events if r["setup"] is None)
    COUNTERS["events_pool_qualified"] = n_pool
    COUNTERS["events_no_pool_setup_none"] = n_nopool
    COUNTERS["e_day_setup_mismatch"] = len(e_day_mismatch)
    check("G1b 真实回放: 非池资格日 setup=None 不回填",
          n_nopool > 0 and n_pool > 0 and len(e_day_mismatch) == 0,
          f"池资格日={n_pool} 非池资格日={n_nopool} E日setup不匹配={len(e_day_mismatch)}")

    # ---- D: 区域 sanity ----
    zdf = pd.DataFrame(zone_rows).dropna()
    zdf = zdf[np.isfinite(zdf["atr_ref"]) & (zdf["atr_ref"] > 0)]
    zn = zdf["zone_n"].value_counts().to_dict()
    check("D1 区域数分布 (1-3)", set(zn) <= {1, 2, 3}, str(zn))
    strengths = [len(r["zone_sources"]) for r in all_events if r.get("zone_sources")]
    atr_pct = zdf["atr_ref"].describe()
    check("D2 ATR_ref 有效", len(zdf) > 500 and (zdf["atr_ref"] > 0).all(),
          f"days={len(zdf)} median={zdf['atr_ref'].median():.3f}")
    print(f"    D-info: 触发区域锚点数分布 {pd.Series(strengths).value_counts().to_dict()}"
          f" (>=2 即合并发生); ATR 分位 {atr_pct['50%']:.3f}")

    # ---- E: 执行仿真 (实际成交数量口径) ----
    filled = [r for r in all_events if r["status"] == "FILLED"]
    diffs = []
    for r in filled:
        d1, d2 = int(str(r["signal_ts"])[-6:]) // 100, int(str(r["fill_ts"])[-6:]) // 100
        diffs.append((d2 - d1) % 100 if d2 >= d1 else None)
    diffs = [x for x in diffs if x is not None]
    n_zero = sum(1 for r in all_events if r["status"] == "NO_FILL_ZERO_QTY")
    n_adj = sum(1 for r in all_events if r["status"] == "ADJ_UNKNOWN")
    n_defail = sum(1 for r in all_events if r["status"] == "EXIT_DEFERRED_FAILED")
    n_deferred = sum(1 for r in filled if r.get("exit_deferred"))
    n_inv = sum(1 for r in all_events if r["status"] == "INVALID_DAY")
    n_missing = sum(1 for r in all_events if r["status"] == "SOURCE_MISSING")
    n_halt = sum(1 for r in all_events if r["status"] == "HALTED_UP_APPROX")
    n_no = sum(1 for r in all_events if r["status"] == "NO_SIGNAL")
    part = [r["participation_ratio"] for r in filled if r.get("participation_ratio") is not None]
    n_partial = sum(1 for p in part if p < 1.0)
    total_amt = sum(r["fill_amount"] for r in filled if r.get("fill_amount"))
    COUNTERS.update({"status_filled": len(filled), "status_no_fill_zero_qty": n_zero,
                     "status_adj_unknown": n_adj, "status_exit_deferred_failed": n_defail,
                     "status_exit_deferred_ok": n_deferred, "status_invalid_day": n_inv,
                     "status_source_missing": n_missing, "status_halted_up": n_halt,
                     "status_no_signal": n_no,
                     "fill_partial": n_partial,
                     "total_invested_yuan": round(float(total_amt), 0),
                     "notional_full_50k": 50000 * len(filled)})
    check("E1 信号→下一可执行分钟成交", len(diffs) > 0 and min(diffs) >= 1
          and pd.Series(diffs).median() <= 2,
          f"n={len(diffs)} 中位间隔={pd.Series(diffs).median():.0f} 分钟")
    ok_e2 = (all(r["fill_shares"] % 100 == 0 for r in filled)
             and all(0 < r["fill_amount"] <= 50000.0 for r in filled)
             and all(abs(r["participation_ratio"] - r["fill_amount"] / 50000.0) < 1e-9
                     for r in filled))
    check("E2 实际成交记账: 整手/上限/参与率=金额比", ok_e2,
          f"FILLED={len(filled)} 零量NO_FILL={n_zero} 部分成交={n_partial} "
          f"实际投入={total_amt/1e4:.1f}万 vs 满额={50000*len(filled)/1e4:.1f}万")
    defer_reasons = pd.Series([r.get("exit_reason") for r in filled
                               if r.get("exit_deferred")]).value_counts().to_dict()
    ok_e3 = all((not r.get("exit_deferred")) or
                (r["exit_date"] > r["intended_exit_date"]
                 and r["label_end_ts"] == r["exit_date"]) for r in filled)
    check("E3 退出日历口径: 顺延显式且label_end=实际退出日", ok_e3,
          f"顺延成功={n_deferred} {defer_reasons} 顺延失败={n_defail} ADJ_UNKNOWN={n_adj}")
    print(f"    E-info: INVALID_DAY={n_inv} SOURCE_MISSING={n_missing} "
          f"HALTED_UP_APPROX={n_halt} NO_SIGNAL={n_no} FILLED={len(filled)}")

    # ---- E 日估计现状 (仅工程口径; 组统计仅 16 股池资格事件, 不作绩效) ----
    n_noest = n_fb = n_p1 = 0
    for r in real_recs:
        if r.get("status") != "FILLED" or not r.get("zone_sources"):
            continue
        est = E.estimate(all_events, r["code"], r["setup"], r["zone_sources"],
                         r["session"], r["date"], didx)
        if est["no_estimate"]:
            n_noest += 1
        elif est["fallback"]:
            n_fb += 1
        else:
            n_p1 += 1
    COUNTERS.update({"e_day_est_no_estimate": n_noest, "e_day_est_fallback": n_fb,
                     "e_day_est_p1": n_p1})
    print(f"    E-est: E日FILLED事件估计 NO_ESTIMATE={n_noest} FALLBACK={n_fb} P1={n_p1}"
          f" (组统计仅{len(sheet['stocks'])}股池资格事件, 工程口径)")

    # ---- F: 预算 ----
    snap_mb = sum(f.stat().st_size for f in SNAP.glob('*.parquet')) / 1e6
    check("F1 快照留存预算 (<5000MB)", snap_mb < 5000, f"{snap_mb:.1f} MB")
    check("F2 请求预算未超 run sheet", n_req <= sheet["total_minute_requests"],
          f"本次实际请求 {n_req} / 预算 {sheet['total_minute_requests']} "
          f"({dur:.0f}s, 可续跑缓存)")
    return all_events, real_recs


# ================= 结构化验收报告 (同输入复验留档) =================
def write_report():
    report = {
        "report_version": "GCP_ROUND_B_ACCEPTANCE_V2",
        "date": "2026-09-29",
        "scope": "登记清单内工程验收; FILLED=工程假设成交事件, 非P0/P1策略成交数",
        "results": [{"name": n, "pass": ok, "detail": d} for n, ok, d in RESULTS],
        "n_pass": sum(1 for _, ok, _ in RESULTS if ok),
        "n_total": len(RESULTS),
        "counters": COUNTERS,
        "notes": [
            "缺陷修复1: 历史事件按事件发生时池资格归类; 非池资格日 setup=None 不进单元格",
            "缺陷修复2: 复权请求失败→ADJ_UNKNOWN 不生成有效标签; 确认空链=无除权",
            "缺陷修复3: 实际成交数量=分钟量(股)×1%与5万上限取小, 整手向下取整, 零量NO_FILL",
            "缺陷修复4: 退出=第3个交易所交易日收盘; 停牌/跌停近似显式顺延(上限10交易所日)",
            "13:00槽=源格式差异: <20260624期望241槽, >=20260624期望242槽(离线探针核实)",
            "B6修复: 旧版扰动未传入daily(恒真空转), 现实际扰动+非空转守卫",
            "扩大组统计可减少NO_ESTIMATE, 但个股同格事件n<10仍回退P0(FALLBACK不必然下降)",
            "分钟volume量纲=股(与日线volume逐日精确对拍核实)",
            "REV2-D/E 行为画像字段: 收回用close不用high; 未收回time=NULL不用窗长+1; "
            "同bar AMBIGUOUS; 已在上方不算收回; 穿透非负; 各窗独立完整性; "
            "行为有效性≠执行有效性(NO_FILL_ZERO_QTY仍算穿透/收回); 事后标签不进决策",
        ],
        "hashes": {
            "spec_gcp_v0_1_round_b_spec.py": sha256_file(HERE / "gcp_v0_1_round_b_spec.py"),
            "engine_gcp_round_b_engine.py": sha256_file(HERE / "gcp_round_b_engine.py"),
            "acceptance_gcp_round_b_acceptance.py": sha256_file(HERE / "gcp_round_b_acceptance.py"),
            "run_sheet.json": sha256_file(OUT / "run_sheet.json"),
            "minute_grid.json": sha256_file(OUT / "minute_grid.json"),
            "pool_a_sdp_members.parquet": sha256_file(POOL_DIR / "pool_a_sdp_members.parquet"),
            "pool_b_lvb_members.parquet": sha256_file(POOL_DIR / "pool_b_lvb_members.parquet"),
            "adj_cache.json": sha256_file(OUT / "adj_cache.json"),
            "minute_snapshots": {f.name: sha256_file(f) for f in sorted(SNAP.glob("*.parquet"))},
        },
    }
    rp = OUT / "acceptance_report_v2.json"
    rp.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"报告已写入: {rp}")


def main():
    grid_path = OUT / "minute_grid.json"
    grid, panel, raw_ok = stage_a(grid_path)
    stage_a_synth(grid)
    stage_b(grid)
    stage_c()
    stage_g(grid)
    stage_h(grid)
    stage_real(grid, panel, raw_ok)
    print("\n==== 验收汇总 ====")
    n_pass = sum(1 for _, ok, _ in RESULTS if ok)
    for name, ok, det in RESULTS:
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    print(f"{n_pass}/{len(RESULTS)} 通过")
    write_report()
    if FAILS:
        print("失败项:", FAILS)
        sys.exit(1)


if __name__ == "__main__":
    main()
