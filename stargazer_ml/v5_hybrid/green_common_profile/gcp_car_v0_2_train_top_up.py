"""CAR v0.2 训练段追加 150 条执行 (2026-10-01, 方案已签署).

执行边界 (car_v0_2_train_top_up_plan.md):
  - 只从 E<=20260305、未使用的 A/B 合格候选抽样; seed=43, A/B 各半, 月度分层
  - 抽样与数量先冻结 (清单+双轨SHA256), 看标签前完成
  - 采集: 单项请求 3 次仍失败 -> SOURCE_MISSING/ADJ_UNKNOWN 并继续, 不换股
  - 仅服务持续不可用/清单或预算完整性无法保证才暂停整批
  - 质量决策口径=A 严格全门; B 仅披露
  - 不训练; 追加后全量重核 label_end_ts 跨界; 验证/测试不动
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
from data_source_provenance import (get_adjustment_http,  # noqa: E402
                                    get_daily_parquet, get_minute_http)
import gcp_round_b_engine as E  # noqa: E402
import gcp_car_v0_1_labels as L  # noqa: E402
from gcp_car_v0_1_engineering_gates import check_gap_propagation, check_corp_action  # noqa: E402
from gcp_car_v0_2_extractor import detect_signal_v02  # noqa: E402

CONTRACTS = HERE / "contracts"
SNAP_CAR = HERE / "gcp_round_b_small_sample_v1" / "minute_car"
SNAP_OLD = HERE / "gcp_round_b_small_sample_v1" / "minute"
SNAP_EXT = HERE / "gcp_round_b_small_sample_v1" / "minute_ext"
ADJ_CACHE = HERE / "gcp_round_b_small_sample_v1" / "adj_cache.json"
OUT = HERE / "gcp_round_b_small_sample_v1"
POOL_DIR = HERE / "round_a_v0_1_impl_fix"

TOPUP_LIST = CONTRACTS / "car_v0_2_train_top_up_candidate_list.txt"
TOPUP_REGISTRY = CONTRACTS / "car_v0_2_train_top_up_registry.json"
TOPUP_CKPT = OUT / "car_v0_2_train_top_up_checkpoint.jsonl"
OLD_REPORT = OUT / "car_v0_2_full_batch_report.json"
NEW_REPORT = OUT / "car_v0_2_train_top_up_report.json"

MAX_RETRY = 3
DISK_CAP_BYTES = 50 * 1024 * 1024
TRAIN_E_MAX = 20260305
VAL_FIRST_E = 20260309
EPISODE_MIN_GAP = 5
SEED = 43
TARGET = 150
NET_HALT_CONSECUTIVE = 8  # 连续网络异常达到则暂停整批


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _np_default(obj):
    if hasattr(obj, "item"):
        return obj.item()
    raise TypeError(f"not serializable: {type(obj)}")


def build_topup_pool(panel, cal, cal_idx):
    pool_a = pd.read_parquet(POOL_DIR / "pool_a_sdp_members.parquet")
    pool_b = pd.read_parquet(POOL_DIR / "pool_b_lvb_members.parquet")
    pool_a = pool_a[pool_a["date"] >= 20250102].copy()
    pool_b = pool_b[pool_b["date"] >= 20250102].copy()
    pool_a["code"] = pool_a["code"].astype(str).str.zfill(6)
    pool_b["code"] = pool_b["code"].astype(str).str.zfill(6)
    pool_a["setup"] = "A_SDP"
    pool_b["setup"] = "B_LVB"
    pool = pd.concat([
        pool_a[["date", "code", "setup"]].rename(columns={"date": "P"}),
        pool_b[["date", "code", "setup"]].rename(columns={"date": "P"}),
    ], ignore_index=True)
    pool = pool.drop_duplicates(subset=["code", "P", "setup"]).sort_values(
        ["P", "code"]).reset_index(drop=True)

    pool["E"] = pool["P"].map(
        lambda p: cal[cal_idx[p] + 1] if p in cal_idx and cal_idx[p] + 1 < len(cal) else None)
    pool = pool.dropna(subset=["E"]).copy()
    pool["E"] = pool["E"].astype(int)
    pool = pool[pool["E"] <= TRAIN_E_MAX].copy()

    sampling = json.loads((CONTRACTS / "car_v0_1_sampling_summary.json").read_text(encoding="utf-8"))
    existing33 = set((str(c["code"]).zfill(6), int(c["E"])) for c in sampling["selected"])
    list400 = (CONTRACTS / "car_v0_2_expansion_candidate_list.txt").read_text(
        encoding="utf-8").strip().splitlines()
    used400 = set()
    used_p_by_code: dict = {}
    for c0, e0 in existing33:
        i_e = cal_idx.get(e0)
        if i_e is not None:
            used_p_by_code.setdefault(c0, []).append(cal[i_e - 1])
    for line in list400:
        c0, s0, p0, e0 = line.split(",")
        used400.add((c0, int(e0)))
        used_p_by_code.setdefault(c0, []).append(int(p0))

    n_before = len(pool)
    pool = pool[~pool.apply(lambda r: (r["code"], r["E"]) in existing33, axis=1)].copy()
    n_excl_33 = n_before - len(pool)
    pool = pool[~pool.apply(lambda r: (r["code"], r["E"]) in used400, axis=1)].copy()
    n_excl_400 = n_before - n_excl_33 - len(pool)

    # episode: 新候选间 + 对已用集合
    pool = pool.sort_values(["code", "P"]).reset_index(drop=True)
    last_used = {c: max(v) for c, v in used_p_by_code.items()}
    keep = pd.Series(True, index=pool.index)
    for idx, row in pool.iterrows():
        code, p = row["code"], int(row["P"])
        ref = last_used.get(code)
        if ref is not None and cal_idx.get(p, 0) - cal_idx.get(ref, 0) < EPISODE_MIN_GAP:
            keep.iloc[idx] = False
            continue
        last_used[code] = p
    pool = pool[keep].reset_index(drop=True)
    counts = {"pool_before_exclusions_in_train_window": n_before,
              "excluded_33": n_excl_33, "excluded_400": n_excl_400,
              "after_episode_filter": len(pool)}
    return pool, counts


def sample_frozen(pool):
    rng = np.random.RandomState(SEED)
    pool = pool.copy()
    pool["month"] = pool["P"].astype(str).str[:6]
    picked = []
    for setup in ("A_SDP", "B_LVB"):
        sub = pool[pool["setup"] == setup]
        per = round(TARGET / 2)
        months = sorted(sub.groupby("month").size().to_dict().items())
        quotas = {m: round(n / len(sub) * per) for m, n in months}
        diff = per - sum(quotas.values())
        if diff > 0:
            quotas[months[0][0]] += diff
        elif diff < 0:
            quotas[months[-1][0]] += diff
        for m, q in quotas.items():
            ms = sub[sub["month"] == m]
            n_pick = min(q, len(ms))
            if n_pick > 0:
                idx = rng.choice(ms.index, size=n_pick, replace=False)
                picked.extend(ms.loc[idx].to_dict("records"))
    seen, unique = set(), []
    for r in picked:
        key = (r["code"], int(r["E"]))
        if key not in seen:
            seen.add(key)
            unique.append({"code": r["code"], "setup": r["setup"],
                           "P": int(r["P"]), "E": int(r["E"])})
    unique.sort(key=lambda x: (x["E"], x["code"]))
    return unique


def freeze_or_load(pool, counts):
    if TOPUP_LIST.exists() and TOPUP_REGISTRY.exists():
        reg = json.loads(TOPUP_REGISTRY.read_text(encoding="utf-8"))
        lines = TOPUP_LIST.read_text(encoding="utf-8").strip().splitlines()
        if sha256_file(TOPUP_LIST) == reg["list_file_sha256"]:
            candidates = []
            for line in lines:
                c0, s0, p0, e0 = line.split(",")
                candidates.append({"code": c0, "setup": s0, "P": int(p0), "E": int(e0)})
            print(f"[冻结] 复用已有清单: {len(candidates)} 条 (哈希核验通过)")
            return candidates, reg
        raise RuntimeError("冻结清单哈希与登记不一致, 需人工核对")
    candidates = sample_frozen(pool)
    list_text = "\n".join(f"{r['code']},{r['setup']},{r['P']},{r['E']}" for r in candidates)
    TOPUP_LIST.write_text(list_text + "\n", encoding="utf-8")
    reg = {
        "version": "car_v0_2_train_top_up_registry",
        "signed_date": "2026-10-01",
        "seed": SEED, "target": TARGET,
        "sampling_rule": "A/B each half; monthly stratified by P; episode gap>=5 incl. used set",
        "pool_counts": counts,
        "n_candidates": len(candidates),
        "n_stocks": len(set(r["code"] for r in candidates)),
        "list_content_sha256": sha256_text(list_text),
        "list_file_sha256": sha256_file(TOPUP_LIST),
        "hash_note": "content hash = normalized text UTF-8/LF/no trailing newline; "
                     "file hash = actual written bytes",
    }
    TOPUP_REGISTRY.write_text(json.dumps(reg, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[冻结] 新生成清单: {len(candidates)} 条, 已登记双轨SHA256")
    return candidates, reg


def main():
    SNAP_CAR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    print("=" * 60)
    print("CAR v0.2 训练段追加 150 条 (口径 A; 不训练)")
    print("=" * 60)

    panel = get_daily_parquet()
    panel["code"] = panel["code"].astype(str).str.zfill(6)
    panel["date"] = panel["date"].astype(int)
    cal = sorted(int(d) for d in panel["date"].unique())
    cal_idx = {d: i for i, d in enumerate(cal)}
    daily_by_code = {c: g.reset_index(drop=True) for c, g in panel.groupby("code")}

    def daily_lookback(code, lookback_dates):
        dc = daily_by_code.get(code)
        if dc is None:
            return pd.DataFrame()
        return dc[dc["date"].isin(lookback_dates)].sort_values("date").reset_index(drop=True)

    grid = json.loads((OUT / "minute_grid.json").read_text(encoding="utf-8"))["grid"]

    # ---- 建池 + 冻结 (先于一切采集) ----
    pool, counts = build_topup_pool(panel, cal, cal_idx)
    candidates, registry = freeze_or_load(pool, counts)

    minute_needs = set()
    adj_needs = set()
    for c in candidates:
        code, e_date = c["code"], c["E"]
        minute_needs.add((code, e_date))
        i_e = cal_idx[e_date]
        if i_e + 1 < len(cal):
            minute_needs.add((code, cal[i_e + 1]))
        adj_needs.add(code)
    needs_by_code: dict = {}
    for c2, d2 in minute_needs:
        needs_by_code.setdefault(c2, set()).add(d2)

    # ---- 缓存加载 (仅所需日期) ----
    print("\n[1/4] 加载已有缓存...")
    t_cache = time.time()
    minute_cache = {}
    for snap_dir in (SNAP_OLD, SNAP_EXT, SNAP_CAR):
        if not snap_dir.exists():
            continue
        for pf in snap_dir.glob("*.parquet"):
            code = pf.stem
            want = needs_by_code.get(code)
            if want is None:
                continue
            try:
                m = pd.read_parquet(pf)
                if "date" in m and not m.empty:
                    day_int = m["date"].astype(str).str[:8].astype(int)
                    for d in pd.unique(day_int):
                        if int(d) not in want:
                            continue
                        minute_cache.setdefault((code, int(d)), []).append(
                            m[day_int == int(d)].copy())
            except Exception:
                pass
    t_cache_done = time.time()
    print(f"  缓存加载 {t_cache_done - t_cache:.1f}s, 命中 (code,date): {len(minute_cache)}")

    adj_data = json.loads(ADJ_CACHE.read_text(encoding="utf-8")) if ADJ_CACHE.exists() else {}

    halt_reason = None
    consec_net_errors = 0

    # ---- 复权 ----
    print(f"\n[2/4] 复权链 ({len(adj_needs)} 股)...")
    t_adj = time.time()
    adj_hit = adj_new = adj_fail = 0
    for code in sorted(adj_needs):
        if code in adj_data and adj_data[code].get("ok"):
            adj_hit += 1
            continue
        result = None
        for attempt in range(MAX_RETRY):
            try:
                raw = get_adjustment_http(code)
                consec_net_errors = 0
                if raw:
                    adj_data[code] = {"ok": True, "chain": raw}
                    result = "SUCCESS"
                else:
                    result = "EMPTY"
                break
            except Exception:
                consec_net_errors += 1
                result = "NET_ERROR"
                if attempt < MAX_RETRY - 1:
                    time.sleep(0.5)
        if result == "SUCCESS":
            adj_new += 1
        else:
            adj_data[code] = {"ok": False, "reason": "ADJ_UNKNOWN"}
            adj_fail += 1
        if consec_net_errors >= NET_HALT_CONSECUTIVE:
            halt_reason = "SERVICE_PERSISTENT_UNAVAILABLE"
            break
    ADJ_CACHE.write_text(json.dumps(adj_data, ensure_ascii=False), encoding="utf-8")
    t_adj_done = time.time()
    print(f"  复权 {t_adj_done - t_adj:.1f}s (命中={adj_hit}, 新增={adj_new}, ADJ_UNKNOWN={adj_fail})")

    # ---- 分钟 ----
    print(f"\n[3/4] 分钟数据 ({len(minute_needs)} 条)...")
    t_min = time.time()
    min_hit = min_new = min_fail = min_missing = 0
    new_bytes = 0
    footprint_now = sum(f.stat().st_size for f in OUT.rglob("*") if f.is_file())
    for code, date in sorted(minute_needs):
        key = (code, date)
        if key in minute_cache:
            min_hit += 1
            continue
        if footprint_now + new_bytes >= DISK_CAP_BYTES:
            halt_reason = halt_reason or "DISK_BUDGET"
            break
        result = None
        df = pd.DataFrame()
        for attempt in range(MAX_RETRY):
            try:
                df = get_minute_http(code, str(date))
                consec_net_errors = 0
                result = "SUCCESS" if not df.empty else "SOURCE_MISSING"
                break
            except Exception:
                consec_net_errors += 1
                result = "NET_ERROR"
                if attempt < MAX_RETRY - 1:
                    time.sleep(0.3)
        if result == "SUCCESS":
            out_f = SNAP_CAR / f"{code}.parquet"
            if out_f.exists():
                old = pd.read_parquet(out_f)
                df = pd.concat([old, df]).drop_duplicates(subset=["date"]).reset_index(drop=True)
            df.to_parquet(out_f, index=False)
            new_bytes += out_f.stat().st_size
            min_new += 1
            minute_cache[key] = [df[df["date"].astype(str).str[:8].astype(int) == date].copy()]
            if min_new % 20 == 0:
                footprint_now = sum(f.stat().st_size for f in OUT.rglob("*") if f.is_file())
                print(f"  已新增 {min_new}, footprint={footprint_now / 1024 / 1024:.1f}MB", flush=True)
        elif result == "SOURCE_MISSING":
            min_missing += 1
        else:
            min_fail += 1
        if consec_net_errors >= NET_HALT_CONSECUTIVE:
            halt_reason = "SERVICE_PERSISTENT_UNAVAILABLE"
            break
    t_min_done = time.time()
    print(f"  分钟 {t_min_done - t_min:.1f}s (命中={min_hit}, 新增={min_new}, "
          f"SOURCE_MISSING={min_missing}, NET_FAIL={min_fail})")

    # ---- 标签 (断点续算) ----
    print(f"\n[4/4] 标签生成 ({len(candidates)} 候选)...")
    t_label = time.time()
    done_map: dict = {}
    if TOPUP_CKPT.exists():
        for line in TOPUP_CKPT.read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            done_map[(r["code"], r["E"])] = r
    ckpt_handle = open(TOPUP_CKPT, "a", encoding="utf-8")
    results = []

    def append_result(r):
        results.append(r)
        ckpt_handle.write(json.dumps(r, ensure_ascii=False, default=_np_default) + "\n")
        ckpt_handle.flush()

    for c in candidates:
        code, e_date, setup = c["code"], c["E"], c["setup"]
        if (code, e_date) in done_map:
            results.append(done_map[(code, e_date)])
            continue
        result = {"code": code, "E": e_date, "setup": setup,
                  "P": c["P"], "signal_found": False,
                  "adj_ok": adj_data.get(code, {}).get("ok", False)}
        e_key = (code, e_date)
        if e_key not in minute_cache or not minute_cache[e_key]:
            result["replay_error"] = "E_DAY_MISSING"
            append_result(result)
            continue
        e_df = minute_cache[e_key][0].sort_values("date").reset_index(drop=True)
        lookback = cal[max(0, cal_idx[e_date] - 250):cal_idx[e_date]]
        d_sub = daily_lookback(code, lookback)
        if len(d_sub) < 25:
            result["replay_error"] = f"DAILY_INSUFFICIENT n={len(d_sub)}"
            append_result(result)
            continue
        atr_ref = float(E.build_atr_ref(d_sub))
        zones = E.build_zones(d_sub, atr_ref)
        ref_price = float(d_sub["close"].iloc[-1])
        sig = detect_signal_v02(e_df, zones, atr_ref, ref_price)
        if sig is None:
            result["replay_error"] = "NO_SIGNAL"
            append_result(result)
            continue
        result["signal_found"] = True
        result["signal_ts"] = sig["signal_ts"]
        result["sig_type"] = sig["signal_type"]
        if sig["signal_type"] == "RECLAIMED_AT_S":
            result["reclaim_status"] = "RECLAIMED_AT_S"
            append_result(result)
            continue
        fields = L.init_car_label_fields()
        sig_idx_list = e_df[e_df["date"].astype(int) == sig["signal_ts"]].index
        if len(sig_idx_list) == 0:
            result["replay_error"] = "SIGNAL_TS_NOT_IN_DF"
            append_result(result)
            continue
        sig_idx = sig_idx_list[0]
        ts_arr = e_df["date"].astype(int).to_numpy()
        lo_arr = e_df["low"].astype(float).to_numpy()
        cl_arr = e_df["close"].astype(float).to_numpy()
        L.compute_future_reclaim_window(sig, ts_arr, lo_arr, cl_arr, sig_idx,
                                        atr_ref, grid, e_date, 30, "30m", fields, cal)
        L.compute_future_reclaim_window(sig, ts_arr, lo_arr, cl_arr, sig_idx,
                                        atr_ref, grid, e_date, 60, "60m", fields, cal)
        reclaim_ts_60m = None
        if fields.get("future_reclaim_60m") == 1 and fields.get("time_to_first_future_reclaim_60m"):
            ttr = fields["time_to_first_future_reclaim_60m"]
            future_slots = L._build_future_window_slots(
                int(str(sig["signal_ts"])[-6:]), e_date, 60, grid, cal)
            if 0 < ttr <= len(future_slots):
                reclaim_ts_60m = int(f"{future_slots[ttr - 1][0]}{future_slots[ttr - 1][1]:06d}")
        L.compute_future_hold(sig, ts_arr, lo_arr, cl_arr, sig_idx, atr_ref,
                              grid, e_date, reclaim_ts_60m, fields, cal)
        result.update({
            "reclaim_30m": fields.get("future_reclaim_30m"),
            "reclaim_60m": fields.get("future_reclaim_60m"),
            "hold_10m": fields.get("future_hold_10m"),
            "penetration_30m": fields.get("future_penetration_30m"),
            "penetration_60m": fields.get("future_penetration_60m"),
            "time_to_first_reclaim_30m": fields.get("time_to_first_future_reclaim_30m"),
            "time_to_first_reclaim_60m": fields.get("time_to_first_future_reclaim_60m"),
            "window_complete_30m": fields.get("future_window_complete_30m"),
            "window_complete_60m": fields.get("future_window_complete_60m"),
            "window_truncated_30m": fields.get("future_window_truncated_30m"),
            "window_truncated_60m": fields.get("future_window_truncated_60m"),
            "label_end_ts": fields.get("future_label_end_ts"),
            "hold_label_end_ts": fields.get("future_hold_label_end_ts"),
            "reclaim_status": fields.get("future_reclaim_status"),
            "ineligible_reason": fields.get("future_ineligible_reason"),
            "status_30m": fields.get("future_status_30m"),
            "status_60m": fields.get("future_status_60m"),
            "status_hold": fields.get("future_status_hold"),
            "reason_30m": fields.get("future_reason_30m"),
            "reason_60m": fields.get("future_reason_60m"),
            "reason_hold": fields.get("future_reason_hold"),
        })
        append_result(result)
    ckpt_handle.close()
    t_label_done = time.time()
    print(f"  标签耗时 {t_label_done - t_label:.1f}s")

    # ---- 工程门 (仅新 150) ----
    print("\n[工程门] 新候选逐条检查...")
    t_gate = time.time()
    pool_a = pd.read_parquet(POOL_DIR / "pool_a_sdp_members.parquet")
    pool_b = pd.read_parquet(POOL_DIR / "pool_b_lvb_members.parquet")
    pool_a_set = set(zip(pool_a["code"].astype(str).str.zfill(6), pool_a["date"].astype(int)))
    pool_b_set = set(zip(pool_b["code"].astype(str).str.zfill(6), pool_b["date"].astype(int)))

    gate1_results = []
    for c in candidates:
        code, p_date, e_date, setup = c["code"], c["P"], c["E"], c["setup"]
        in_pool = ((code, p_date) in pool_a_set) if setup == "A_SDP" else ((code, p_date) in pool_b_set)
        expected_e = cal[cal_idx[p_date] + 1] if cal_idx[p_date] + 1 < len(cal) else None
        ok = in_pool and expected_e == e_date
        gate1_results.append({"code": code, "E": e_date, "in_pool": in_pool, "pass": ok})
    n_g1 = sum(1 for r in gate1_results if r["pass"])
    ce = [(c["code"], c["E"]) for c in candidates]
    gate2_ok = len(ce) == len(set(ce))

    scale_results = []
    sub_pass = {"price": 0, "volume": 0, "grid": 0, "adj": 0}
    for c in candidates:
        code, e_date, p_date = c["code"], c["E"], c["P"]
        item = {"code": code, "E": e_date, "sub_checks": {}, "pass": True}
        dc = daily_by_code.get(code)
        daily_e = dc[dc["date"] == e_date] if dc is not None else pd.DataFrame()
        if (code, e_date) not in minute_cache or not minute_cache[(code, e_date)] or daily_e.empty:
            item["sub_checks"]["data_missing"] = True
            item["pass"] = False
            scale_results.append(item)
            continue
        e_df = minute_cache[(code, e_date)][0].sort_values("date").reset_index(drop=True)
        d_close = float(daily_e["close"].iloc[0])
        d_high = float(daily_e["high"].iloc[0])
        d_low = float(daily_e["low"].iloc[0])
        d_open = float(daily_e["open"].iloc[0])
        d_amt = float(daily_e["amount"].iloc[0])

        def _rdiff(a, b):
            return abs(a - b) / (max(abs(a), abs(b)) + 1e-12)
        ohlc = {"open": _rdiff(d_open, float(e_df["open"].iloc[0])),
                "high": _rdiff(d_high, float(e_df["high"].max())),
                "low": _rdiff(d_low, float(e_df["low"].min())),
                "close": _rdiff(d_close, float(e_df["close"].iloc[-1]))}
        price_ok = all(v < 0.01 for v in ohlc.values())
        try:
            lb = cal[max(0, cal_idx[e_date] - 250):cal_idx[e_date]]
            atr_g = float(E.build_atr_ref(daily_lookback(code, lb)))
            atr_ratio = atr_g / d_close
            atr_ok = 0.001 < atr_ratio < 0.30
        except Exception:
            atr_g, atr_ratio, atr_ok = None, None, False
        price_pass = price_ok and atr_ok
        item["sub_checks"]["price_scale"] = {"ohlc_rdiffs": ohlc, "atr_ratio": atr_ratio,
                                              "pass": price_pass}
        if price_pass:
            sub_pass["price"] += 1
        else:
            item["pass"] = False

        ratio_arr = e_df["amount"].astype(float) / (
            e_df["volume"].astype(float) * e_df["close"].astype(float) + 1e-12)
        valid = ratio_arr[e_df["volume"] > 0]
        med = float(np.median(valid)) if len(valid) > 0 else None
        amt_ratio = float(e_df["amount"].sum()) / (d_amt + 1e-12)
        if not (0.8 < amt_ratio < 1.2):
            vol_status, vol_pass = "AMOUNT_MISMATCH", False
        elif med is not None and 0.5 < med < 1.5:
            vol_status, vol_pass = "CONSISTENT", True
        else:
            vol_status, vol_pass = "VOLUME_SCALE_UNKNOWN", False
        item["sub_checks"]["volume_scale"] = {"status": vol_status, "amount_ratio": amt_ratio,
                                               "pass": vol_pass}
        if vol_pass:
            sub_pass["volume"] += 1
        else:
            item["pass"] = False

        grid_check = E.validate_minute_day(e_df, grid, e_date)
        grid_pass = grid_check["ok"]
        replay_match = next((r for r in results if r["code"] == code and r["E"] == e_date
                             and r.get("signal_found")), None)
        gap = None
        if replay_match and replay_match.get("signal_ts"):
            nd_idx = cal_idx.get(e_date, -1) + 1
            next_df_v = None
            if nd_idx < len(cal):
                nd_key = (code, cal[nd_idx])
                if nd_key in minute_cache and minute_cache[nd_key]:
                    next_df_v = minute_cache[nd_key][0]
            gap = check_gap_propagation(e_df, next_df_v, int(replay_match["signal_ts"]),
                                        replay_match.get("time_to_first_reclaim_60m"),
                                        grid, cal, e_date)
            if any(gap[k] for k in ("30m", "60m", "hold_10m")):
                grid_pass = False
            nd = gap.get("next_day")
            if nd and any(nd[k] for k in ("30m", "60m", "hold_10m")):
                grid_pass = False
        item["sub_checks"]["window_grid"] = {"grid_ok": grid_check["ok"], "gap": gap,
                                              "pass": grid_pass}
        if grid_pass:
            sub_pass["grid"] += 1
        else:
            item["pass"] = False

        adj_res = check_corp_action(adj_data.get(code, {}), p_date,
                                    replay_match.get("label_end_ts") if replay_match else None)
        item["sub_checks"]["corp_action"] = {"status": adj_res["status"], "pass": adj_res["pass"]}
        if adj_res["pass"]:
            sub_pass["adj"] += 1
        else:
            item["pass"] = False
        scale_results.append(item)
    n_scale = sum(1 for s in scale_results if s["pass"])
    print(f"  门1: {n_g1}/150; 门2: {'PASS' if gate2_ok else 'FAIL'}; "
          f"门3全过: {n_scale}/150; 明细 价格={sub_pass['price']} "
          f"量额={sub_pass['volume']} 网格={sub_pass['grid']} 公司行为={sub_pass['adj']}")
    t_gate_done = time.time()
    print(f"  工程门耗时 {t_gate_done - t_gate:.1f}s")

    # ---- 合并原批次: 全量训练段重算 + 边界重核 ----
    old = json.loads(OLD_REPORT.read_text(encoding="utf-8"))
    old_gate = {(it["code"], it["E"]): it for it in old["engineering_gates"]["gate3_scale"]["results"]}
    new_gate = {(it["code"], it["E"]): it for it in scale_results}

    def include(row, caliber):
        it = new_gate.get((row["code"], row["E"])) or old_gate.get((row["code"], row["E"]))
        if it is None:
            return False
        sc = it["sub_checks"]
        if sc.get("data_missing"):
            return False
        if caliber == "A":
            return bool(it["pass"])
        if it["pass"]:
            return True
        return (not sc["volume_scale"]["pass"]
                and sc["price_scale"]["pass"]
                and sc["window_grid"]["pass"]
                and sc["corp_action"]["pass"])

    # 仅训练窗口 (E <= TRAIN_E_MAX); 不得纳入原报告验证/测试段行
    train_rows_old = [r for r in old["per_candidate"]
                      if r.get("E", 0) <= TRAIN_E_MAX
                      and r.get("reclaim_status") in ("RECLAIMED", "NOT_RECLAIMED")]
    train_rows_new = [r for r in results
                      if r.get("E", 0) <= TRAIN_E_MAX
                      and r.get("reclaim_status") in ("RECLAIMED", "NOT_RECLAIMED")]

    def tally(rows, caliber):
        dates, pos, neg = set(), 0, 0
        crossings = []
        for r in rows:
            if not include(r, caliber):
                continue
            dates.add(r["E"])
            if r.get("reclaim_60m") == 1:
                pos += 1
            elif r.get("reclaim_60m") == 0:
                neg += 1
            end = r.get("label_end_ts")
            if end:
                end_date = int(str(int(end))[:8])
                if end_date >= VAL_FIRST_E:
                    crossings.append({"code": r["code"], "E": r["E"], "label_end_ts": int(end)})
        return dates, pos, neg, crossings

    old_tally_a = tally(train_rows_old, "A")
    consistency = {"old_only_A": {"dates": len(old_tally_a[0]), "pos": old_tally_a[1],
                                  "neg": old_tally_a[2], "crossings": len(old_tally_a[3])}}
    combined = {}
    for cal_name in ("A", "B"):
        dates, pos, neg, crossings = tally(train_rows_old + train_rows_new, cal_name)
        combined[cal_name] = {"effective_dates": len(dates), "positive": pos, "negative": neg,
                              "boundary_crossings": crossings,
                              "crossings_count": len(crossings)}
    thresholds = {"min_dates": 40, "min_positive": 30, "min_negative": 30}
    checks_a = {k: combined["A"][v] >= thresholds[k2]
                for k, k2, v in (("dates", "min_dates", "effective_dates"),
                                 ("pos", "min_positive", "positive"),
                                 ("neg", "min_negative", "negative"))}
    training_ready = all(checks_a.values()) and len(combined["A"]["boundary_crossings"]) == 0
    print("\n[合并后训练段]")
    for cal_name in ("A", "B"):
        c = combined[cal_name]
        print(f"  口径{cal_name}: 有效日={c['effective_dates']} 正={c['positive']} "
              f"负={c['negative']} 跨界={c['crossings_count']}")
    print("  旧批次口径A复算:", consistency["old_only_A"],
          "(应为 dates=54 pos=48 neg=24 crossings=0)")
    print("  门槛核查:", checks_a, "→ READY" if training_ready else "→ INSUFFICIENT")

    # ---- 磁盘最终实测 ----
    TOPUP_CKPT.unlink(missing_ok=True)

    def measure():
        return sum(f.stat().st_size for f in OUT.rglob("*") if f.is_file())

    disk_report = {"footprint_bytes": measure(), "cap_bytes": DISK_CAP_BYTES,
                   "new_bytes_this_run": new_bytes}
    disk_report["footprint_mb"] = round(disk_report["footprint_bytes"] / 1024 / 1024, 2)
    disk_report["within_cap"] = disk_report["footprint_bytes"] <= DISK_CAP_BYTES

    t_total = time.time() - t0
    report = {
        "version": "car_v0_2_train_top_up_report",
        "batch_date": "2026-10-01",
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "training_ready": training_ready,
        "training_status": "READY_FOR_BASELINES" if training_ready else "INSUFFICIENT_AFTER_TOP_UP",
        "halt_reason": halt_reason,
        "integrity_note": ("单项失败记 SOURCE_MISSING/ADJ_UNKNOWN 并继续; "
                           "仅服务持续不可用/清单预算完整性问题才暂停整批; 未换股"),
        "timing": {
            "total": round(t_total, 1),
            "cache_load": round(t_cache_done - t_cache, 1),
            "adj": round(t_adj_done - t_adj, 1),
            "minute": round(t_min_done - t_min, 1),
            "labels": round(t_label_done - t_label, 1),
            "gates": round(t_gate_done - t_gate, 1),
        },
        "requests": {
            "adj_hit": adj_hit, "adj_new": adj_new, "adj_unknown": adj_fail,
            "minute_hit": min_hit, "minute_new": min_new,
            "minute_source_missing": min_missing, "minute_net_fail": min_fail,
        },
        "disk": disk_report,
        "thresholds": thresholds,
        "threshold_checks_A": checks_a,
        "combined_training_segment": combined,
        "old_only_consistency": consistency,
        "new_batch_diversion": {
            "n": len(candidates),
            "signal": sum(1 for r in results if r.get("signal_found")),
            "reclaimed_at_s": sum(1 for r in results if r.get("reclaim_status") == "RECLAIMED_AT_S"),
            "eligible": len(train_rows_new),
            "gate1_pass": n_g1, "gate2_pass": gate2_ok, "gate3_all_pass": n_scale,
            "gate3_sub_pass": sub_pass,
        },
        "blind_test_disclaimer": ("原验证/测试段汇总结果在本批前已查看, 不是全新盲测; "
                                  "其候选/边界/结果本轮未改动"),
        "he_locked": True, "buy_advice_locked": True,
        "new_gate_results": scale_results,
        "new_per_candidate": results,
    }

    def serialize():
        return json.dumps(report, ensure_ascii=False, indent=2, default=_np_default)

    NEW_REPORT.write_text(serialize(), encoding="utf-8")
    fp = measure()
    if fp != disk_report["footprint_bytes"]:
        disk_report["footprint_bytes"] = fp
        disk_report["footprint_mb"] = round(fp / 1024 / 1024, 2)
        NEW_REPORT.write_text(serialize(), encoding="utf-8")
        fp2 = measure()
        disk_report["footprint_bytes"] = fp2
        disk_report["footprint_mb"] = round(fp2 / 1024 / 1024, 2)
        NEW_REPORT.write_text(serialize(), encoding="utf-8")

    print(f"\n报告: {NEW_REPORT.name}")
    print(f"最终 footprint: {disk_report['footprint_mb']} MB (≤50MB: {disk_report['within_cap']})")
    if training_ready:
        print("训练段达标. 下一步需单独授权: 常数基线 → CAR-G → CAR-LR.")
    else:
        print("训练段仍不足. 按方案归档, 不做第二次补抽, 转向其他研究线.")


if __name__ == "__main__":
    main()
