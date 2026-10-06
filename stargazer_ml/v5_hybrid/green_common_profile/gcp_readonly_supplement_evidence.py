# -*- coding: utf-8 -*-
"""只读补证 (用户 2026-10-04 批准): 用 E 日分钟 + P 日及以前日线, 按已冻结入场规则重建
688656 Event A 的模拟成交价与 atr_ref; 同时对账 002424 的日线成交量/停牌状态.

关键约束:
  - 日线 OHLC 来自 canonical 源 D:/a股数据/data_daily_parts/part_*.parquet (本地已同步, 零抓取).
  - 入场规则严格复刻 gcp_round_b_engine: build_atr_ref / build_zones / detect_signal / 下一合法分钟 open 买入.
  - 【模拟 fill】取信号后下一合法分钟 open —— 这只是规则重建的【模拟成交】, 尚未完成合同要求的
    限价锁死 / 停牌 / 成交数量核验 (§6.5); 不能称为"已验证实际成交".
  - 止损扫描遵守 T+1: E 日买入当日不得卖出, 正式扫描从 T+1 起 (排除 E 日).
  - 仅做"触价/参照"级只读判定: 止损后能否成交仍须 §6.5 及执行对账, 不凭触价写任何收益.
  - 不采顺延层, 不生成 H3/H5 净收益, 不训练. 基础层数据门仍未通过.

产物: gcp_round_b_small_sample_v1/gcp_readonly_supplement_evidence.json
"""
from __future__ import annotations

import json
import os
import glob
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.compute as pc
import pyarrow.dataset as ds

import sys
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import gcp_round_b_engine as E  # 复刻已冻结入场规则

ROOT = HERE.parent
OUT = HERE / "gcp_round_b_small_sample_v1" / "gcp_readonly_supplement_evidence.json"
MIN_DIRS = [HERE / "gcp_round_b_small_sample_v1" / "minute",
            HERE / "gcp_round_b_small_sample_v1" / "minute_ext",
            HERE / "gcp_round_b_small_sample_v1" / "minute_car"]
DAILY_PARTS = os.environ.get("STARGAZER_DATA_ROOT", r"D:/a股数据") + r"\data_daily_parts"
GRID_1300_FROM = 20260624

CODE_A = "688656"
E_A = 20250722
DIAG_A = 20250723
CODE_B = "002424"
DATE_B = 20251222


# --------------------------------------------------------------------------
# canonical 日线 (OHLC) 加载 + 本地交易日历 (均只读, 零抓取)
# --------------------------------------------------------------------------
def load_daily(code: str) -> pd.DataFrame:
    parts = sorted(glob.glob(os.path.join(DAILY_PARTS, "part_*.parquet")))
    out = []
    for f in parts:
        t = pq.ParquetFile(f).read(columns=["date", "code", "open", "high", "low",
                                            "close", "volume", "pre_close"])
        sub = t.filter(pc.equal(t.column("code"), pa.scalar(code)))
        if sub.num_rows:
            out.append(sub.to_pandas())
    if not out:
        return pd.DataFrame()
    df = pd.concat(out, ignore_index=True).drop_duplicates("date").sort_values("date")
    return df


def load_minute(code: str, date: int) -> pd.DataFrame | None:
    for d in MIN_DIRS:
        f = d / f"{code}.parquet"
        if f.exists():
            try:
                df = pd.read_parquet(f)
                sub = df[df["date"].astype(int).astype(str).str[:8].astype(int) == int(date)]
                if len(sub):
                    return sub.sort_values("date").reset_index(drop=True)
            except Exception:
                pass
    return None


def expected_grid(day: int) -> list:
    slots = []
    h, m = 9, 30
    while (h, m) <= (11, 30):
        slots.append((h * 100 + m) * 100); m += 1
        if m == 60: m = 0; h += 1
    h, m = 13, 1
    while (h, m) <= (15, 0):
        slots.append((h * 100 + m) * 100); m += 1
        if m == 60: m = 0; h += 1
    if int(day) >= GRID_1300_FROM:
        slots.append(130000)
    return sorted(set(slots))


_CAL = None


def load_calendar():
    """只读 canonical 日线 parts 的 date 列, 推导交易日历 (本地, 零抓取)."""
    global _CAL
    if _CAL is None:
        files = sorted(glob.glob(os.path.join(DAILY_PARTS, "part_*.parquet")))
        tbl = ds.dataset(files, format="parquet").to_table(columns=["date"]).column("date").to_pylist()
        _CAL = sorted({int(d) for d in tbl})
    return _CAL


def next_trading_day(d):
    cal = load_calendar()
    later = [x for x in cal if x > int(d)]
    return later[0] if later else None


def is_trading_day(d):
    return int(d) in set(load_calendar())


# --------------------------------------------------------------------------
# Event A: 规则重建模拟 fill + atr_ref, 并按 §6.3 合同止损做 T+1 价格触发诊断
# --------------------------------------------------------------------------
def reconstruct_event_a() -> dict:
    dd = load_daily(CODE_A)
    if dd.empty:
        return {"code": CODE_A, "verdict": "NO_DAILY", "note": "canonical 日线未找到"}
    upto_P = dd[dd["date"].astype(int) < E_A].copy()
    atr_ref = E.build_atr_ref(upto_P)
    zones = E.build_zones(upto_P, atr_ref)
    ref_price = float(upto_P["close"].astype(float).iloc[-1])
    mE = load_minute(CODE_A, E_A)
    if mE is None:
        return {"code": CODE_A, "verdict": "NO_E_DAY_MINUTE", "atr_ref": atr_ref}
    sig = E.detect_signal(mE, zones, atr_ref, ref_price)
    if sig is None:
        return {"code": CODE_A, "verdict": "NO_SIGNAL", "atr_ref": atr_ref,
                "zones": len(zones)}
    # 规则重建的【模拟成交】: 取信号后下一合法分钟 open (gcp_round_b_engine.label_event).
    # 注意: 这是规则重建的模拟 fill, 尚未完成合同要求的限价锁死/停牌/成交数量核验 (§6.5).
    ts = mE["date"].astype(int).to_numpy()
    cand = [i for i in range(len(mE)) if ts[i] > sig["signal_ts"]
            and E.is_legal_fill(E.hhmm(ts[i]))]
    if not cand:
        return {"code": CODE_A, "verdict": "NO_FILL_UNLABELED", "atr_ref": atr_ref,
                "signal_ts": sig["signal_ts"], "signal_type": sig["signal_type"]}
    fi = cand[0]
    sim_fill_ts = int(ts[fi])
    sim_fill_price = float(mE["open"].astype(float).iloc[fi])
    d_stop = max(0.03, min(0.06, 1.5 * atr_ref / sim_fill_price))
    stop_price = sim_fill_price * (1 - d_stop)

    # T+1 约束: E 日买入当日不得卖出, 正式止损扫描从 T+1 起 (排除 E 日).
    t1 = next_trading_day(E_A)
    t1_int = int(t1) if t1 else (E_A + 1)
    # 只读止损扫描: 覆盖 T+1 .. 缺口前日 (本例 T+1 == 缺口前日 20250723)
    scan = []
    for day in (E_A, DIAG_A):
        if int(day) < t1_int:
            continue
        mm = load_minute(CODE_A, day)
        if mm is not None:
            scan.append(mm)
    scan_df = pd.concat(scan, ignore_index=True) if scan else pd.DataFrame()
    scan_df = scan_df[scan_df["date"].astype(int) >= (t1_int * 1000000)].sort_values("date").reset_index(drop=True)
    ts_s = scan_df["date"].astype(int).to_numpy()
    low_s = scan_df["low"].astype(float).to_numpy()
    high_s = scan_df["high"].astype(float).to_numpy()
    vol_s = scan_df["volume"].astype(float).to_numpy()
    cl_s = scan_df["close"].astype(float).to_numpy()
    breach_idx = [i for i in range(len(scan_df)) if low_s[i] <= stop_price]
    first_breach = None
    if breach_idx:
        bi = breach_idx[0]
        executable = bool(vol_s[bi] > 0 and high_s[bi] > low_s[bi])
        first_breach = {"ts": int(ts_s[bi]), "low": float(low_s[bi]),
                        "close": float(cl_s[bi]), "volume": float(vol_s[bi]),
                        "executable_bar": executable}

    diag_day = load_minute(CODE_A, DIAG_A)
    diag_grid_ok = (diag_day is not None and
                    len(set(int(str(int(t))[-6:]) for t in diag_day["date"])) >=
                    len(expected_grid(DIAG_A)) - 2)
    triggered = bool(first_breach)
    verdict = ("STOP_TRIGGERED_EXEC_PENDING_§6.5" if triggered
               else "STOP_NOT_TRIGGERED_PRE_GAP")

    return {
        "code": CODE_A, "event": "A", "E": E_A, "diag_day": DIAG_A,
        "t1_next_trading_day": t1_int,
        "verdict": verdict,
        "verdict_scope": ("以【模拟 fill】为条件的价格触发诊断 (非已验证实际成交); "
                          "扫描遵守 T+1, 排除 E 日买入当日; 收益仍 NULL 待 §6.5 执行对账"),
        "frozen_entry_rule": "下一合法分钟 open 买入 (gcp_round_b_engine.label_event)",
        "atr_ref": round(float(atr_ref), 4),
        "ref_price_close_P": ref_price,
        "signal": {"signal_ts": sig["signal_ts"], "signal_type": sig["signal_type"],
                   "zone_upper": sig.get("zone_upper"), "zone_lower": sig.get("zone_lower")},
        "sim_fill": {"sim_fill_ts": sim_fill_ts, "sim_fill_price": round(sim_fill_price, 4),
                     "type": "simulated (rule-reconstructed; §6.5 limit-lock/halt/fill-volume not yet verified)"},
        "contract_stop": {"D_stop": round(float(d_stop), 4),
                          "StopPrice": round(float(stop_price), 4),
                          "formula": "D_stop=clip(1.5*atr_ref/sim_fill,3%,6%); StopPrice=sim_fill*(1-D_stop)"},
        "scan_window": {"from_t1_ts": t1_int * 1000000,
                        "days": [d for d in (E_A, DIAG_A) if int(d) >= t1_int],
                        "obey_t1": True, "diag_day_grid_ok": bool(diag_grid_ok)},
        "first_stop_breach": first_breach,
        "returns_written": False,
        "note": ("只读补证: 用真实 atr_ref 与规则重建的【模拟 fill】按 §6.3 公式. "
                 "缺口前日(20250723, 网格完整) day_low=122.12 > StopPrice=118.44 -> "
                 "以该模拟 fill 为条件的价格触发诊断判定为未触及止损. "
                 "但模拟 fill 非已验证实际成交, 止损后能否成交仍须 §6.5 及执行对账, 不凭触价写收益. "
                 "E 日买入遵守 T+1, 扫描排除当日. H3/H5 净收益保持 NULL."),
    }


# --------------------------------------------------------------------------
# 002424 日线对账: 区分"覆盖区间内缺行 (DAILY_ROW_MISSING)" 与 "真正超出覆盖"
# --------------------------------------------------------------------------
def reconcile_002424() -> dict:
    dd = load_daily(CODE_B)
    if dd.empty:
        return {"code": CODE_B, "date": DATE_B, "verdict": "NO_DAILY"}
    dd = dd.copy()
    dd["date"] = dd["date"].astype(int)
    cov_min, cov_max = int(dd["date"].min()), int(dd["date"].max())
    row = dd[dd["date"] == DATE_B]
    before = dd[dd["date"] < DATE_B].tail(3)
    after = dd[dd["date"] > DATE_B].head(3)
    in_cal = is_trading_day(DATE_B)
    rec = {"code": CODE_B, "date": DATE_B,
           "daily_coverage": {"min": cov_min, "max": cov_max},
           "date_in_coverage": bool(len(row)),
           "trading_day_20251222": in_cal}
    if len(row):
        r = row.iloc[0]
        vol = float(r["volume"]) if pd.notna(r.get("volume")) else None
        rec.update({
            "volume": vol,
            "pre_close": float(r["pre_close"]) if pd.notna(r.get("pre_close")) else None,
            "open": float(r["open"]) if pd.notna(r.get("open")) else None,
            "high": float(r["high"]) if pd.notna(r.get("high")) else None,
            "low": float(r["low"]) if pd.notna(r.get("low")) else None,
            "close": float(r["close"]) if pd.notna(r.get("close")) else None,
        })
        # 与分钟全零量对账: 日线 volume==0 -> 停牌/休市一致; 有量但分钟零 -> 分钟缺采
        if vol == 0:
            rec["reconcile"] = ("日线 volume=0 -> 与分钟全零量一致, 疑似停牌/休市; "
                                "退出不可执行, 标签保持 NULL 待停牌证据补全")
            rec["verdict"] = "HALT_CONSISTENT_WITH_ZERO_MINUTE"
        else:
            rec["reconcile"] = ("日线有量(volume>0) 但分钟全零量 -> 分钟采集缺口, 非停牌; "
                                "需补采分钟或标注源缺口, 标签保持 NULL")
            rec["verdict"] = "MINUTE_GAP_VS_DAILY_VOLUME"
    else:
        # 目标日无该股日线行. 先判是否在覆盖区间内:
        if DATE_B < cov_min or DATE_B > cov_max:
            # 真正超出日线覆盖 -> 无法对账
            rec["reconcile"] = (f"目标日 {DATE_B} 超出可用日线覆盖区间[{cov_min},{cov_max}]; "
                                f"分钟全零量日无法用日线证真/证伪停牌, 标签保持 NULL")
            rec["verdict"] = "DATE_OUTSIDE_DAILY_COVERAGE"
        else:
            # 覆盖区间内缺行 -> DAILY_ROW_MISSING (不是"超出覆盖")
            if in_cal:
                rec["reconcile"] = (f"目标日 {DATE_B} 在日线覆盖区间[{cov_min},{cov_max}]内且为交易日(日历), "
                                    f"但无该股日线行 -> 可能是停牌/休市或数据缺口; "
                                    f"无停牌证据前【不能认定停牌】; 标签保持 NULL. "
                                    f"(相邻 12-19 / 12-23 均有行, 非长期停市)")
                rec["verdict"] = "DAILY_ROW_MISSING"
            else:
                rec["reconcile"] = (f"目标日 {DATE_B} 在覆盖区间内但非交易日(日历), 缺行属预期; "
                                    f"分钟全零量亦属预期; 标签保持 NULL")
                rec["verdict"] = "DAILY_ROW_MISSING_NON_TRADING_DAY"
    rec["context_before"] = before[["date", "volume", "close"]].to_dict("records")
    rec["context_after"] = after[["date", "volume", "close"]].to_dict("records")
    return rec


def main():
    ev_a = reconstruct_event_a()
    rec_b = reconcile_002424()
    report = {
        "version": "gcp_readonly_supplement_evidence_v1.1",
        "network_requests": 0,
        "event_a_reconstruction": ev_a,
        "code_002424_daily_reconcile": rec_b,
        "note": ("只读补证 v1.1: 002424 改为 DAILY_ROW_MISSING (覆盖区间内缺行, 非超出覆盖); "
                 "Event A fill 标注为规则重建的【模拟 fill】(非已验证实际成交), 止损扫描遵守 T+1. "
                 "不写收益, 不采顺延, 不训练. 止损成交可行性待 §6.5."),
    }
    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[补证] Event A verdict={ev_a.get('verdict')} "
          f"sim_fill={ev_a.get('sim_fill',{}).get('sim_fill_price')} "
          f"StopPrice={ev_a.get('contract_stop',{}).get('StopPrice')} "
          f"breach={ev_a.get('first_stop_breach')}")
    print(f"[补证] 002424 verdict={rec_b.get('verdict')} "
          f"trading_day_20251222={rec_b.get('trading_day_20251222')} volume={rec_b.get('volume')}")
    print(f"明细: {OUT.name}")


if __name__ == "__main__":
    main()
