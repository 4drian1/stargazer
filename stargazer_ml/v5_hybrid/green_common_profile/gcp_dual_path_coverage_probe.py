"""双路径合同 §9 只读覆盖盘点与行动漏斗 (2026-10-01, v2: Swing 主口径).

零新请求: 只在日线 parquet、A/B 池 parquet 与本地分钟缓存上重算.
输出 dual_path_coverage_probe.json, 回答:
  "以横代调" 是否已被 A/B 共性池覆盖; 主口径 Swing≤2.5 漏斗各层数量.
预注册对照: P85/P15 稳健边、Swing≤3.0 — 只披露, 不自动切换, 不追加为硬门.
"""
from __future__ import annotations

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
from data_source_provenance import get_daily_parquet  # noqa: E402
import gcp_round_b_engine as E  # noqa: E402
from gcp_car_v0_2_extractor import detect_signal_v02  # noqa: E402

CONTRACTS = HERE / "contracts"
OUT = HERE / "gcp_round_b_small_sample_v1"
POOL_DIR = HERE / "round_a_v0_1_impl_fix"
MIN_DIRS = (OUT / "minute", OUT / "minute_ext", OUT / "minute_car")
FALLBACK_DIR = ROOT / "pullback_train" / "minute_cache_clean"
REPORT_JSON = OUT / "dual_path_coverage_probe.json"

MINUTE_ERA_START = 20250102

# 冻结时间切分边界 (task order §4; 与既有研究口径一致, 不重切)
SEG_TRAIN_END = 20260305
SEG_VAL_START = 20260309
SEG_VAL_END = 20260609
SEG_TEST_START = 20260610


def segment_of(e: int) -> str:
    """按 E 日映射到冻结的 train/val/test 段 (不移动切分点)."""
    if e <= SEG_TRAIN_END:
        return "train"
    if SEG_VAL_START <= e <= SEG_VAL_END:
        return "val"
    if e >= SEG_TEST_START:
        return "test"
    return "gap"

# ---- 合同 §3 冻结参数 ----
BOX_LOOKBACK_DAYS = 20
SWING_K = 2
BOX_WIDTH_MAX = 2.5
BOX_INSIDE_MIN = 0.55
SWING3_WIDTH_MAX = 3.0
PCTL_H, PCTL_L = 0.85, 0.15
PCT_WIDTH_MAX = 2.5
PCT_INSIDE_MIN = 0.6
PCT_SLOPE_MAX = 0.20
PCT_EDGE_TOL = 0.15

LOWER_TOL = 0.10
WATCH_FRAC = 1 / 3
BREAKOUT_VOL_RATIO_MIN = 1.5
ATR_CONTRACTION_MAX = 0.9
VOL_STABILITY_CV_MAX = 0.5


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _swing_pivots(arr: np.ndarray, k: int) -> list:
    """严格 fractal: i 处值严格优于左右各 k 根; 并列极值不确认. 返回索引列表."""
    out = []
    for i in range(k, len(arr) - k):
        left, right = arr[i - k:i], arr[i + 1:i + k + 1]
        if arr[i] > max(left) and arr[i] > max(right):
            out.append(i)
    return out


def _swing_pivots_low(arr: np.ndarray, k: int) -> list:
    out = []
    for i in range(k, len(arr) - k):
        left, right = arr[i - k:i], arr[i + 1:i + k + 1]
        if arr[i] < min(left) and arr[i] < min(right):
            out.append(i)
    return out


def evaluate_calibers(d_sub: pd.DataFrame, atr_ref: float) -> dict:
    """返回三口径箱体评估 {main, swing3, pct}; 每个含 valid/edges/reason/metrics."""
    res = {}
    if len(d_sub) < BOX_LOOKBACK_DAYS or not np.isfinite(atr_ref) or atr_ref <= 0:
        return {"main": {"valid": False, "reason": "INSUFFICIENT_DAILY_OR_ATR"}}
    win = d_sub.iloc[-BOX_LOOKBACK_DAYS:].reset_index(drop=True)
    highs = win["high"].astype(float).to_numpy()
    lows = win["low"].astype(float).to_numpy()
    closes = win["close"].astype(float).to_numpy()
    slope = float(np.polyfit(np.arange(len(closes), dtype=float), closes, 1)[0] / atr_ref)

    # ---- Swing 边 (main 与 swing3 共用边) ----
    piv_h = _swing_pivots(highs, SWING_K)
    piv_l = _swing_pivots_low(lows, SWING_K)
    if len(piv_h) >= 2 and len(piv_l) >= 2:
        up = float(np.mean(highs[piv_h[-2:]]))
        lo = float(np.mean(lows[piv_l[-2:]]))
        width = (up - lo) / atr_ref
        inside = float(np.mean((closes >= lo) & (closes <= up)))
        for name, wmax in (("main", BOX_WIDTH_MAX), ("swing3", SWING3_WIDTH_MAX)):
            ok = width <= wmax and inside >= BOX_INSIDE_MIN and up > lo
            res[name] = {"valid": ok, "upper": up, "lower": lo,
                         "width_atr": round(width, 3), "inside": round(inside, 3),
                         "reason": "OK" if ok else
                         ("WIDTH" if width > wmax else "INSIDE")}
        # box_age (main): 尾部滑动 20 根子窗连续有效
        age = BOX_LOOKBACK_DAYS
        for k in range(BOX_LOOKBACK_DAYS + 1, min(len(d_sub), 40) + 1):
            w2 = d_sub.iloc[-k:].reset_index(drop=True)
            h2, l2 = w2["high"].astype(float).to_numpy(), w2["low"].astype(float).to_numpy()
            c2 = w2["close"].astype(float).to_numpy()
            ph = _swing_pivots(h2, SWING_K)
            pl = _swing_pivots_low(l2, SWING_K)
            if len(ph) >= 2 and len(pl) >= 2:
                u2, l2e = float(np.mean(h2[ph[-2:]])), float(np.mean(l2[pl[-2:]]))
                ww = (u2 - l2e) / atr_ref
                ii = float(np.mean((c2 >= l2e) & (c2 <= u2)))
                if ww <= BOX_WIDTH_MAX and ii >= BOX_INSIDE_MIN and u2 > l2e:
                    age = k
                    continue
            break
        res["main"]["box_age"] = age
    else:
        reason = f"NO_BOX n_highs={len(piv_h)} n_lows={len(piv_l)}"
        res["main"] = {"valid": False, "reason": reason}
        res["swing3"] = {"valid": False, "reason": reason}

    # ---- P85/P15 稳健边 ----
    up_p = float(np.percentile(highs, PCTL_H * 100))
    lo_p = float(np.percentile(lows, PCTL_L * 100))
    width_p = (up_p - lo_p) / atr_ref
    inside_p = float(np.mean((closes >= lo_p) & (closes <= up_p)))
    tol = PCT_EDGE_TOL * atr_ref
    n_up = int((highs >= up_p - tol).sum())
    n_lo = int((lows <= lo_p + tol).sum())
    ok_p = (width_p <= PCT_WIDTH_MAX and inside_p >= PCT_INSIDE_MIN
            and n_up >= 2 and n_lo >= 2 and abs(slope) <= PCT_SLOPE_MAX)
    res["pct"] = {"valid": ok_p, "upper": up_p, "lower": lo_p,
                  "width_atr": round(width_p, 3), "inside": round(inside_p, 3),
                  "slope_atr_day": round(slope, 3),
                  "reason": "OK" if ok_p else "FAIL_robust_checks"}

    # 交集 (展示, 非硬门)
    res["intersection_main_and_pct"] = {
        "valid": res["main"].get("valid", False) and res["pct"]["valid"]}
    for name in ("main", "swing3", "pct"):
        res[name]["slope_atr_day"] = round(slope, 3)
    return res


def load_used_keys():
    used = set()
    sampling = json.loads((CONTRACTS / "car_v0_1_sampling_summary.json").read_text(encoding="utf-8"))
    for c in sampling["selected"]:
        used.add((str(c["code"]).zfill(6), int(c["E"])))
    for fname in ("car_v0_2_expansion_candidate_list.txt",
                  "car_v0_2_train_top_up_candidate_list.txt"):
        for line in (CONTRACTS / fname).read_text(encoding="utf-8").strip().split("\n"):
            code, _setup, _p, e = line.split(",")
            used.add((code, int(e)))
    return used


def load_minute_index():
    idx: dict = {}
    targets = []
    for d in MIN_DIRS:
        if d.exists():
            targets.append((d, "small_sample"))
    if FALLBACK_DIR.exists():
        targets.append((FALLBACK_DIR, "fallback"))
    for directory, tag in targets:
        for pf in directory.glob("*.parquet"):
            code = pf.stem
            try:
                m = pd.read_parquet(pf, columns=["date"])
                if "date" in m and not m.empty:
                    for d in pd.unique(m["date"].astype(str).str[:8].astype(int)):
                        idx.setdefault((code, int(d)), []).append((pf, tag))
            except Exception:
                pass
    return idx


def scan_views(e_df: pd.DataFrame, upper: float, lower: float, atr_ref: float,
               contraction_ok: bool, volstable_ok: bool) -> dict:
    """按合同 §2 在 E 日监控窗扫描视图 (不含收益)."""
    df = e_df.copy()
    df["hhmm"] = df["date"].apply(E.hhmm)
    df = df[(df["hhmm"] >= E.MONITOR_START) & (df["hhmm"] <= E.MONITOR_END)].sort_values("date")
    width = upper - lower
    watch_line = lower + WATCH_FRAC * width
    tol_l = LOWER_TOL * atr_ref
    vols = df["volume"].astype(float).to_numpy()
    lows = df["low"].astype(float).to_numpy()
    closes = df["close"].astype(float).to_numpy()

    watch = False
    s1_ts = None
    s2_ts = None
    lower_fail_ts = None
    breakout_fail = False
    n_above = 0
    n_above_volok = 0
    prev_below = False
    for i in range(len(df)):
        ts = int(df["date"].iloc[i])
        if closes[i] <= watch_line and contraction_ok and volstable_ok:
            watch = True
        if s1_ts is None and lows[i] <= lower + tol_l and closes[i] >= lower:
            o = float(df["open"].iloc[i])
            if not (o == float(df["high"].iloc[i]) == lows[i] == closes[i]):
                s1_ts = ts
        if closes[i] > upper:
            n_above += 1
            if s2_ts is None:
                base = vols[max(0, i - 20):i]
                vr = vols[i] / np.mean(base) if len(base) and np.mean(base) > 0 else np.inf
                if vr >= BREAKOUT_VOL_RATIO_MIN:
                    s2_ts = ts
                    n_above_volok += 1
                elif i + 1 < len(df) and closes[i + 1] < upper:
                    breakout_fail = True
        if closes[i] < lower:
            if prev_below and lower_fail_ts is None:
                lower_fail_ts = ts
            prev_below = True
        else:
            prev_below = False
    action_ts, action = None, None
    cands = sorted((t, k) for t, k in ((s1_ts, "S1"), (s2_ts, "S2")) if t is not None)
    if cands:
        action_ts, action = cands[0]
    s2_reason = None
    if s2_ts is None:
        if not watch:
            s2_reason = "NO_WATCH"
        elif n_above == 0:
            s2_reason = "NO_BREAKOUT"
        elif n_above_volok == 0:
            s2_reason = "BREAKOUT_VOL_LOW"
        else:
            s2_reason = "OTHER"
    return {"watch": watch, "s1_ts": s1_ts, "s2_ts": s2_ts, "action": action,
            "action_ts": action_ts, "lower_fail_ts": lower_fail_ts,
            "breakout_failed": breakout_fail, "s2_reason": s2_reason,
            "n_above": n_above, "n_above_volok": n_above_volok}


def fill_feasibility(e_df_full: pd.DataFrame, action_ts: int) -> dict:
    """S 后下一合法分钟的候选成交状态 (非真实成交)."""
    ts_all = e_df_full["date"].astype(int).to_numpy()
    pos = int(np.searchsorted(ts_all, action_ts, side="right"))
    if pos >= len(ts_all):
        return {"status": "NO_NEXT_BAR"}
    nxt_ts = int(ts_all[pos])
    slot = E.hhmm(nxt_ts)
    if not E.is_legal_fill(slot):
        return {"status": "ILLEGAL_SLOT", "next_ts": nxt_ts}
    row = e_df_full.iloc[pos]
    o, h, l, c = (float(row["open"]), float(row["high"]),
                  float(row["low"]), float(row["close"]))
    if o == h == l == c:
        return {"status": "LIMIT_LOCK", "next_ts": nxt_ts}
    return {"status": "FILLABLE", "next_ts": nxt_ts}


def main():
    t0 = time.time()
    print("=" * 60)
    print("双路径只读覆盖盘点与行动漏斗 (零新请求, Swing 主口径)")
    print("=" * 60)

    panel = get_daily_parquet()
    panel["code"] = panel["code"].astype(str).str.zfill(6)
    panel["date"] = panel["date"].astype(int)
    cal = sorted(int(d) for d in panel["date"].unique())
    cal_idx = {d: i for i, d in enumerate(cal)}
    daily_by_code = {c: g.sort_values("date").reset_index(drop=True)
                     for c, g in panel.groupby("code")}
    print(f"日线: {len(cal)} 交易日, {len(daily_by_code)} 股")

    pool_a = pd.read_parquet(POOL_DIR / "pool_a_sdp_members.parquet")
    pool_b = pd.read_parquet(POOL_DIR / "pool_b_lvb_members.parquet")
    pool_a["code"] = pool_a["code"].astype(str).str.zfill(6)
    pool_b["code"] = pool_b["code"].astype(str).str.zfill(6)
    members = sorted({(c, int(p), "A_SDP") for c, p in pool_a[["code", "date"]].itertuples(index=False)}
                     | {(c, int(p), "B_LVB") for c, p in pool_b[["code", "date"]].itertuples(index=False)})
    print(f"池成员 (去重): {len(members)}")

    used_keys = load_used_keys()
    minute_idx = load_minute_index()
    print(f"分钟缓存索引: {len(minute_idx)} (code,date)")
    grid = json.loads((OUT / "minute_grid.json").read_text(encoding="utf-8"))["grid"]

    # ---- 逐成员三口径箱体 ----
    member_evals = []
    for code, p, setup in members:
        dc = daily_by_code.get(code)
        if dc is None:
            continue
        loc = int(dc["date"].searchsorted(p, side="right"))
        if loc < 20:
            continue
        d_sub = dc.iloc[:loc]
        atr_ref = E.build_atr_ref(d_sub)
        calibs = evaluate_calibers(d_sub, atr_ref)
        atr14 = pd.Series(np.maximum(
            d_sub["high"].astype(float).to_numpy() - d_sub["low"].astype(float).to_numpy(),
            np.abs(d_sub["high"].astype(float).to_numpy()
                   - np.concatenate([[d_sub["close"].astype(float).to_numpy()[0]],
                                     d_sub["close"].astype(float).to_numpy()[:-1]]))
        )).rolling(14).mean().to_numpy()
        amed60 = float(np.nanmedian(atr14[-60:])) if len(atr14) >= 14 else np.nan
        contraction = (atr_ref / amed60 if np.isfinite(amed60) and amed60 > 0 else np.nan)
        vol_arr = d_sub["volume"].astype(float).to_numpy()
        last10 = vol_arr[-10:]
        cv = float(np.std(last10, ddof=1) / np.mean(last10)) if np.mean(last10) > 0 else np.nan
        member_evals.append({
            "code": code, "P": p, "setup": setup, "atr_ref": atr_ref,
            "calibs": calibs,
            "contraction": float(contraction) if np.isfinite(contraction) else None,
            "vol_cv10": cv if np.isfinite(cv) else None,
            "minute_era": p >= MINUTE_ERA_START,
        })
    n_main = sum(m["calibs"].get("main", {}).get("valid") for m in member_evals)
    n_s3 = sum(m["calibs"].get("swing3", {}).get("valid") for m in member_evals)
    n_pct = sum(m["calibs"]["pct"]["valid"] for m in member_evals)
    print(f"[全历史箱体] main={n_main}, swing3={n_s3}, pct={n_pct}, "
          f"intersection={sum(m['calibs']['intersection_main_and_pct']['valid'] for m in member_evals)}")

    # ---- 2025+ 漏斗明细 ----
    era = [m for m in member_evals if m["minute_era"]]
    detail = []
    for m in era:
        code, p, setup = m["code"], m["P"], m["setup"]
        i_p = cal_idx.get(p)
        if i_p is None or i_p + 1 >= len(cal):
            continue
        e = cal[i_p + 1]
        mkeys = minute_idx.get((code, e))
        entry = {"code": code, "P": p, "E": e, "setup": setup,
                 "used_before": (code, e) in used_keys,
                 "minute_cached": bool(mkeys),
                 "caliber_layers": {}}

        # 对每个有效口径建边 → 扫描 (有缓存且合法时)
        e_df = None
        minute_valid = False
        if mkeys:
            try:
                e_df_raw = pd.read_parquet(mkeys[0][0])
                day_int = e_df_raw["date"].astype(str).str[:8].astype(int)
                e_df = e_df_raw[day_int == e].sort_values("date").reset_index(drop=True)
                vd = E.validate_minute_day(e_df, grid, e)
                minute_valid = vd["ok"]
                entry["minute_valid"] = vd["ok"]
                entry["minute_reason"] = vd["reason"]
                entry["source"] = mkeys[0][1]
            except Exception as ex:
                entry["minute_valid"] = False
                entry["minute_reason"] = f"READ_ERROR:{type(ex).__name__}"
                e_df = None

        for cal_name in ("main", "swing3", "pct"):
            cb = m["calibs"].get(cal_name, {})
            layer = {"box_valid": bool(cb.get("valid"))}
            if cb.get("valid") and e_df is not None and minute_valid:
                c_ok = (m["contraction"] is not None
                        and m["contraction"] <= ATR_CONTRACTION_MAX)
                v_ok = (m["vol_cv10"] is not None
                        and m["vol_cv10"] <= VOL_STABILITY_CV_MAX)
                views = scan_views(e_df, cb["upper"], cb["lower"],
                                   m["atr_ref"], c_ok, v_ok)
                fill = None
                if views["action_ts"] is not None:
                    fill = fill_feasibility(e_df, views["action_ts"])
                layer.update({"views": views, "fill": fill})
            entry["caliber_layers"][cal_name] = layer
        detail.append(entry)

    # ---- 漏斗聚合: setup × year × caliber ----
    def empty_layer_counts():
        return {"box": 0, "box_P_dates": set(),
                "minute_cached": 0, "minute_valid": 0,
                "watch": 0, "s1": 0, "s2": 0,
                "action": 0, "fillable": 0, "fill_blocked": 0,
                "s2_reasons": {},
                "E_dates": {k: set() for k in
                            ("minute_valid", "watch", "s1", "s2", "action", "fillable")}}

    funnel: dict = {}
    funnel_seg: dict = {}
    s2_reasons_main: dict = {}
    s2_reasons_by_seg: dict = {}
    seen_e = set()
    for r in detail:
        key_e = (r["code"], r["E"])
        if key_e in seen_e:
            continue
        seen_e.add(key_e)
        year = int(str(r["P"])[:4])
        seg = segment_of(r["E"])
        for cal_name in ("main", "swing3", "pct"):
            layer = r["caliber_layers"][cal_name]
            if not layer["box_valid"]:
                continue
            d = funnel.setdefault((r["setup"], year, cal_name), empty_layer_counts())
            ds = funnel_seg.setdefault((seg, r["setup"], cal_name), empty_layer_counts())
            d["box"] += 1
            ds["box"] += 1
            d["box_P_dates"].add(r["P"])
            ds["box_P_dates"].add(r["P"])
            if r["minute_cached"]:
                d["minute_cached"] += 1
                ds["minute_cached"] += 1
            if r.get("minute_valid"):
                d["minute_valid"] += 1
                d["E_dates"]["minute_valid"].add(r["E"])
                ds["minute_valid"] += 1
                ds["E_dates"]["minute_valid"].add(r["E"])
            v = layer.get("views")
            if v is not None:
                if v["watch"]:
                    d["watch"] += 1
                    d["E_dates"]["watch"].add(r["E"])
                    ds["watch"] += 1
                    ds["E_dates"]["watch"].add(r["E"])
                if v["s1_ts"] is not None:
                    d["s1"] += 1
                    d["E_dates"]["s1"].add(r["E"])
                    ds["s1"] += 1
                    ds["E_dates"]["s1"].add(r["E"])
                if v["s2_ts"] is not None:
                    d["s2"] += 1
                    d["E_dates"]["s2"].add(r["E"])
                    ds["s2"] += 1
                    ds["E_dates"]["s2"].add(r["E"])
                if v["action"] is not None:
                    d["action"] += 1
                    d["E_dates"]["action"].add(r["E"])
                    ds["action"] += 1
                    ds["E_dates"]["action"].add(r["E"])
                if cal_name == "main" and v["s2_ts"] is None and v["watch"]:
                    rs = v.get("s2_reason", "OTHER")
                    d["s2_reasons"][rs] = d["s2_reasons"].get(rs, 0) + 1
                    ds["s2_reasons"][rs] = ds["s2_reasons"].get(rs, 0) + 1
                    s2_reasons_main[rs] = s2_reasons_main.get(rs, 0) + 1
                    s2_reasons_by_seg.setdefault(seg, {})
                    s2_reasons_by_seg[seg][rs] = s2_reasons_by_seg[seg].get(rs, 0) + 1
                f = layer.get("fill")
                if f is not None:
                    if f["status"] == "FILLABLE":
                        d["fillable"] += 1
                        d["E_dates"]["fillable"].add(r["E"])
                        ds["fillable"] += 1
                        ds["E_dates"]["fillable"].add(r["E"])
                    else:
                        d["fill_blocked"] += 1
                        ds["fill_blocked"] += 1

    funnel_out = {}
    for (setup, year, cal_name), d in sorted(funnel.items()):
        funnel_out[f"{setup}|{year}|{cal_name}"] = {
            "box_members": d["box"], "box_unique_P": len(d["box_P_dates"]),
            "minute_cached": d["minute_cached"], "minute_valid": d["minute_valid"],
            "watch": d["watch"], "s1": d["s1"], "s2": d["s2"],
            "action_unique": d["action"],
            "fillable": d["fillable"], "fill_blocked": d["fill_blocked"],
            "unique_E_dates": {k: len(v) for k, v in d["E_dates"].items()},
        }

    funnel_out_seg = {}
    for (seg, setup, cal_name), d in sorted(funnel_seg.items()):
        funnel_out_seg[f"{seg}|{setup}|{cal_name}"] = {
            "box_members": d["box"], "box_unique_P": len(d["box_P_dates"]),
            "minute_cached": d["minute_cached"], "minute_valid": d["minute_valid"],
            "watch": d["watch"], "s1": d["s1"], "s2": d["s2"],
            "action_unique": d["action"],
            "fillable": d["fillable"], "fill_blocked": d["fill_blocked"],
            "unique_E_dates": {k: len(v) for k, v in d["E_dates"].items()},
            "s2_non_trigger_reasons": d["s2_reasons"],
        }

    # ---- 全历史箱体检定按 setup ----
    box_all = {}
    for setup in ("A_SDP", "B_LVB"):
        sub = [m for m in member_evals if m["setup"] == setup]
        box_all[setup] = {
            "total": len(sub),
            "main": sum(m["calibs"].get("main", {}).get("valid", False) for m in sub),
            "swing3": sum(m["calibs"].get("swing3", {}).get("valid", False) for m in sub),
            "pct": sum(m["calibs"]["pct"]["valid"] for m in sub),
            "intersection": sum(m["calibs"]["intersection_main_and_pct"]["valid"] for m in sub),
        }

    # ---- used/unused 标签 (main 口径) ----
    tag = {"used": {"box": 0, "fillable": 0}, "unused": {"box": 0, "fillable": 0}}
    tag_seg = {"used": {}, "unused": {}}
    for r in detail:
        layer = r["caliber_layers"]["main"]
        if not layer["box_valid"]:
            continue
        k = "used" if r["used_before"] else "unused"
        seg = segment_of(r["E"])
        tag[k]["box"] += 1
        tag_seg[k].setdefault(seg, {"box": 0, "fillable": 0})
        tag_seg[k][seg]["box"] += 1
        f = layer.get("fill")
        if f is not None and f["status"] == "FILLABLE":
            tag[k]["fillable"] += 1
            tag_seg[k][seg]["fillable"] = tag_seg[k][seg].get("fillable", 0) + 1

    # ---- 冻结切分下的采集质量与网格异常 (只读, 来自采集报告) ----
    seg_quality = {}
    grid_reasons_overall = {}
    grid_by_seg = {}
    crep_path = OUT / "dual_path_e_day_collect_report.json"
    if crep_path.exists():
        crep = json.loads(crep_path.read_text(encoding="utf-8"))
        for seg in ("train", "val", "test"):
            bs = crep.get("by_segment", {}).get(seg)
            if bs:
                req = bs.get("requested", 0)
                seg_quality[seg] = {
                    "requested": req,
                    "fetched_success": bs.get("fetched_success", 0),
                    "source_missing": bs.get("source_missing", 0),
                    "grid_anomaly": bs.get("grid_anomaly", 0),
                    "anomaly_rate": round(bs.get("grid_anomaly", 0) / req, 4) if req else None,
                }
        for g in crep.get("grid_anomaly_detail", []):
            reason = g.get("reason", "UNKNOWN")
            grid_reasons_overall[reason] = grid_reasons_overall.get(reason, 0) + 1
            sg = segment_of(int(g["E"]))
            grid_by_seg.setdefault(sg, {})
            grid_by_seg[sg][reason] = grid_by_seg[sg].get(reason, 0) + 1

    report = {
        "version": "dual_path_coverage_probe_v2",
        "date": "2026-10-01",
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "contract_ref": "dual_path_event_execution_contract.md (Swing≤2.5 main; pending signature)",
        "network_requests": 0,
        "main_parameters": {"BOX_LOOKBACK_DAYS": BOX_LOOKBACK_DAYS,
                            "SWING_K": SWING_K,
                            "BOX_WIDTH_MAX_ATR": BOX_WIDTH_MAX,
                            "BOX_INSIDE_MIN": BOX_INSIDE_MIN,
                            "LOWER_TOL": LOWER_TOL,
                            "BREAKOUT_VOL_RATIO_MIN": BREAKOUT_VOL_RATIO_MIN,
                            "ATR_CONTRACTION_MAX": ATR_CONTRACTION_MAX,
                            "VOL_STABILITY_CV_MAX": VOL_STABILITY_CV_MAX},
        "registered_controls": {"pct_P85_P15": "coverage/robustness only",
                                "swing3_width_3.0": "coverage/robustness only; "
                                                    "needs new version to promote",
                                "intersection": "display only, not a gate"},
        "box_valid_all_history": box_all,
        "funnel": funnel_out,
        "used_vs_unused_main": tag,
        "used_vs_unused_by_segment": tag_seg,
        "frozen_segment_boundaries": {
            "train": "E<=20260305", "val": "20260309..20260609",
            "test": "E>=20260610"},
        "funnel_by_segment": funnel_out_seg,
        "s2_non_trigger_reasons_main": {
            "overall": s2_reasons_main,
            "by_segment": s2_reasons_by_seg},
        "segment_quality": seg_quality,
        "grid_anomaly_reasons_overall": grid_reasons_overall,
        "grid_anomaly_by_segment": grid_by_seg,
        "detail_n": len(detail),
        "data_hashes": {
            "pool_a_sdp_members.parquet": sha256_file(POOL_DIR / "pool_a_sdp_members.parquet"),
            "pool_b_lvb_members.parquet": sha256_file(POOL_DIR / "pool_b_lvb_members.parquet"),
            "minute_grid.json": sha256_file(OUT / "minute_grid.json"),
        },
        "elapsed_s": round(time.time() - t0, 1),
    }
    REPORT_JSON.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n[全历史箱体检定]")
    for setup, d in box_all.items():
        print(f"  {setup}: main={d['main']}/{d['total']} ({d['main']/d['total']:.3f}), "
              f"swing3={d['swing3']}, pct={d['pct']}, inter={d['intersection']}")
    print("\n[漏斗] (setup|year|caliber)")
    for k, d in funnel_out.items():
        print(f"  {k}: 箱={d['box_members']}(P{d['box_unique_P']}) "
              f"缓存={d['minute_cached']} 合法={d['minute_valid']} "
              f"看={d['watch']} S1={d['s1']} S2={d['s2']} "
              f"行动={d['action_unique']} 可成交={d['fillable']}/阻={d['fill_blocked']}")
    print("\n[used/unused main]", tag)
    print("\n[冻结切分漏斗] (segment|setup|caliber)  -- main/pct/swing3 有重叠, 勿相加")
    for k, d in funnel_out_seg.items():
        print(f"  {k}: 箱={d['box_members']} 合法={d['minute_valid']} "
              f"看={d['watch']} S1={d['s1']} S2={d['s2']} 可成交={d['fillable']}")
    print("\n[段采集质量] (来自采集报告; 异常率=网格异常/请求)")
    for seg, q in seg_quality.items():
        print(f"  {seg}: 请求={q['requested']} 成功={q['fetched_success']} "
              f"源缺={q['source_missing']} 网格异常={q['grid_anomaly']} "
              f"异常率={q['anomaly_rate']}")
    print("  >> 验证段(val)网格异常率明显偏高, 单列")
    print("\n[网格异常原因] 总体:")
    for rsn, c in sorted(grid_reasons_overall.items(), key=lambda x: -x[1]):
        print(f"  {rsn}: {c}")
    print("  按段:")
    for seg, d in grid_by_seg.items():
        print(f"  {seg}: " + ", ".join(f"{k}={v}" for k, v in d.items()))
    print("\n[S2 未触发原因 main 口径] (研究线索, 非'漏斗泄漏', 不断 S2 过严)")
    print("  overall:", s2_reasons_main)
    print("  by_segment:", s2_reasons_by_seg)
    print(f"\n耗时 {report['elapsed_s']}s; 报告: {REPORT_JSON.name}")


if __name__ == "__main__":
    main()
