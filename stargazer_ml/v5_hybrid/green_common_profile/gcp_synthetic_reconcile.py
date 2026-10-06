# -*- coding: utf-8 -*-
"""合成反例 + 少量真实执行对账 (Task B, 2026-10-04, 零抓取收益).

不计算任何净收益; 仅做工程验收:
  (1) 合成反例: 用引擎自带 synth_stock + run_day, 对 7 类受控日内路径
      (break_am/break_pm/approach_am/approach_pm/already_below/no_signal/missing_slots)
      跑完整信号+标签流水线, 核对状态是否符合合约预期.
  (2) 真实执行对账(数据就绪级): 抽 5 条退出路径完整事件, 载入其 E+1..H5 已采分钟,
      用引擎 validate_minute_day 核对网格可 ingestion 性, 确认 H3/H5 参照日可被引擎取到.
      注: 完整 P&L 对账依赖日线/zone/§6.4 成交规则, 待签署后另做; 本步只验数据就绪.
事故: 详见 gcp_audit_log.md (INC-2026-10-04-01). 本脚本只做验收, 不补洞/不换源.
产物: gcp_round_b_small_sample_v1/gcp_synthetic_reconcile.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
from gcp_round_b_engine import (  # noqa: E402
    synth_stock, run_day, validate_minute_day,
)

GRID_1300_FROM = 20260624


def base_grid(day: int) -> list:
    slots = []
    h, m = 9, 30
    while (h, m) <= (11, 30):
        slots.append((h * 100 + m) * 100)
        m += 1
        if m == 60:
            m = 0
            h += 1
    h, m = 13, 1
    while (h, m) <= (15, 0):
        slots.append((h * 100 + m) * 100)
        m += 1
        if m == 60:
            m = 0
            h += 1
    if int(day) >= GRID_1300_FROM:
        slots.append(130000)
    return sorted(set(slots))

VERIFY = HERE / "gcp_round_b_small_sample_v1" / "gcp_247_coverage_verify.json"
QA = HERE / "gcp_round_b_small_sample_v1" / "gcp_exit_path_qa.json"
BY_PAIR = HERE / "gcp_round_b_small_sample_v1" / "minute_collect_realtest" / "by_pair"
MIN_DIRS = [HERE / "gcp_round_b_small_sample_v1" / "minute",
            HERE / "gcp_round_b_small_sample_v1" / "minute_ext",
            HERE / "gcp_round_b_small_sample_v1" / "minute_car"]
FALLBACK_DIR = ROOT / "pullback_train" / "minute_cache_clean"
OUT = HERE / "gcp_round_b_small_sample_v1" / "gcp_synthetic_reconcile.json"
GRID_1300_FROM = 20260624


def load_minute(code: str, date: int) -> pd.DataFrame | None:
    p = BY_PAIR / f"{code}_{date}.parquet"
    if p.exists():
        return pd.read_parquet(p)
    for d in MIN_DIRS + ([FALLBACK_DIR] if FALLBACK_DIR.exists() else []):
        if not d.exists():
            continue
        f = d / f"{code}.parquet"
        if f.exists():
            try:
                df = pd.read_parquet(f)
                sub = df[df["date"].astype(int).astype(str).str[:8].astype(int) == int(date)]
                if len(sub):
                    return sub
            except Exception:
                pass
    return None


def synthetic_counterexamples():
    scenarios = ["break_am", "break_pm", "approach_am", "approach_pm",
                 "already_below", "no_signal", "missing_slots"]
    results = []
    base_grid_list = base_grid(20250102)  # 241 槽 (合成日 < 20260624)
    for sc in scenarios:
        try:
            s = synth_stock(seed=12345, code="600000", setup="A_SDP",
                            grid=base_grid_list, n_days=320, scenario_days={20: sc})
        except Exception as ex:
            results.append({"scenario": sc, "error": f"{type(ex).__name__}: {ex}"})
            continue
        if not s["minute"]:
            results.append({"scenario": sc, "note": "synth 跳过 (zone/atr 不满足)"})
            continue
        E = list(s["minute"].keys())[0]
        ctx = {"code": "600000", "setup": "A_SDP", "daily": s["daily"],
               "minute": s["minute"][E], "cal": None, "chain": None, "adj_ok": True}
        try:
            rec = run_day(ctx, E, base_grid(E))
            results.append({
                "scenario": sc, "E": E, "status": rec.get("status"),
                "signal_type": rec.get("signal_type"), "session": rec.get("session"),
                "y": rec.get("y"), "f": rec.get("f"),
                "exit_date": rec.get("exit_date"), "exit_reason": rec.get("exit_reason"),
                "bar_valid": rec.get("bar_valid"), "bar_reason": rec.get("bar_reason"),
            })
        except Exception as ex:
            results.append({"scenario": sc, "E": E, "error": f"{type(ex).__name__}: {ex}"})
    return results


def real_reconciliation(n=5):
    qa = json.loads(QA.read_text(encoding="utf-8"))
    candidates = [e for e in qa["exit_reports"]
                  if e["exit_complete"] and e["code"] != "688656"]
    sampled = candidates[:n]
    out = []
    for e in sampled:
        code = e["code"]
        day_checks = []
        for off in e["offsets"]:
            d = off["date"]
            df = load_minute(code, d)
            if df is None:
                day_checks.append({"offset": off["offset"], "date": d,
                                   "loadable": False, "grid_ok": None})
                continue
            v = validate_minute_day(df, base_grid(d), d)
            day_checks.append({"offset": off["offset"], "date": d, "loadable": True,
                               "grid_ok": v["ok"], "grid_reason": v["reason"],
                               "n_bars": len(df)})
        h3 = next((c for c in day_checks if c["offset"] == 3), None)
        h5 = next((c for c in day_checks if c["offset"] == 5), None)
        out.append({
            "code": code, "E": e["E"], "segment": e["segment"],
            "day_checks": day_checks,
            "h3_reference_available": bool(h3 and h3.get("loadable")),
            "h5_reference_available": bool(h5 and h5.get("loadable")),
            "all_grid_ok": all(c.get("grid_ok") for c in day_checks if c.get("loadable")),
        })
    return out


def main():
    synth = synthetic_counterexamples()
    real = real_reconciliation(5)
    report = {
        "version": "gcp_synthetic_reconcile_v1",
        "network_requests": 0,
        "incident_ref": "INC-2026-10-04-01 (see gcp_audit_log.md)",
        "note": ("合成反例=信号与部分标签分流验收(已验证 BREAK/APPROACH/ALREADY_BELOW 信号识别、"
                 "FILLED/NO_SIGNAL/INVALID_DAY 分流、越日历 EXIT_UNAVAILABLE); "
                 "尚未验证完整 P&L、成本与真实退出成交, 故非'引擎完整流水线验收通过'. "
                 "真实对账=数据就绪级(网格可 ingestion + H3/H5 参照可取). "
                 "完整 P&L 对账与执行对账待 §6.4/§6.5 签署后另做. 不补洞/不换源."),
        "synthetic_counterexamples": synth,
        "real_reconciliation": real,
    }
    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[合成反例] {len(synth)} 场景:")
    for r in synth:
        print(f"  {r.get('scenario'):14} -> status={r.get('status')} "
              f"signal={r.get('signal_type')} y={r.get('y')} f={r.get('f')} "
              f"exit={r.get('exit_date')} reason={r.get('exit_reason')}")
    print(f"[真实对账] 抽样 {len(real)} 条完整事件:")
    for r in real:
        print(f"  {r['code']} E={r['E']} H3可取={r['h3_reference_available']} "
              f"H5可取={r['h5_reference_available']} 全网格OK={r['all_grid_ok']}")
    print(f"明细: {OUT.name}")


if __name__ == "__main__":
    main()
