# -*- coding: utf-8 -*-
"""第三步前置 S2 量比门审计 + 冻结执行器逐轨道执行标签 (零网络/零采集/零拟合).

用户裁决 (2026-10-08): 允许第三步推进, 但 S2 检查为同一任务的前置; 不展开全库审计.

Phase A — S2 量比门审计:
  事实链:
  - 已签事件合同 §2.4 第 2 条: S2 = 有效箱体存在 且 close > box_upper_P (无任何量能条件).
  - 已签合同全目录检索: 无 S2 量比的分母/窗口/早盘处置冻结定义 ("同刻量比"仅出现于 CAR 线
    PENDING 特征, 与 S2 无关).
  - probe scan_views 实现 (L235-238): vr = vol[i]/mean(前<=20根监控窗bar); 空基线或零基线 -> inf 放行;
    vr >= 1.5 才记 s2_ts.
  双向偏离:
  (a) 09:35 首根监控窗 bar 基线必空 -> vr=inf 自动放行 ("缺少比较依据被当成最强确认");
  (b) vr<1.5 压掉合同文本成立的 S2 (实现比合同更严).
  处置 (不自行换算法):
  - branch=BASE_BREAKOUT (S2 定分支) -> PENDING_S2_UNRESOLVED;
  - 可行动候选中 first close>upper 早于当前决策触发 (或同 bar 且当前为 DIP) -> 分支依赖 S2 定义
    -> PENDING_S2_EXPOSURE;
  - NO_ACTION 认证中存在 close>upper -> 认证依赖 S2 定义 -> PENDING_S2_EXPOSURE;
  - 全部保留观察记录, 不改分支, 不删事件.

Phase B — 冻结执行器 (dual_path_execution_v0_1_1, manifest SHA256 绑定):
  对 cert=CERTIFIED 且 branch in {DIP_REPAIR, BASE_HOLD} 且非 S2 PENDING 的事件,
  以 branch_at_decision 的决策触发 ts 为冻结信号逐事件跑 H3/H5 双轨道.
  执行器冻结 MIN_DIRS 覆盖全部 489 候选的 E 日分钟 (预检实测), 不扩展读路径.
  逐轨道核验: 完全退出/残仓/NULL/实际依赖终点(label_end_ts)/公司行为(pre_close 跳变)/跨段隔离.

Phase C — 报告 + 重新推导缺口需求 (旧 65 对按实际依赖终点重分类; 新缺口单列).

预算: 第三步新增产物 <= 1 MiB; 研究目录累计 <= 55 MiB; 不复制分钟缓存; 不删旧证据.
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "step9_preflight"))
sys.path.insert(0, str(HERE))

import gcp_dual_path_coverage_probe as PROBE  # noqa: E402
import gcp_round_b_engine as E  # noqa: E402
import dual_path_execution_v0_1_1 as D  # noqa: E402

RUN = HERE / "gcp_round_b_small_sample_v1"
CAP_BASE_BYTES = 55 * 1024 * 1024
CAP_NEW_BYTES = 1 * 1024 * 1024
MON_LO, MON_HI = 93500, 145500

OUT_S2 = RUN / "r1_s2_volume_gate_audit_v0_1_2.json"
OUT_EXEC = RUN / "r1_exec_labels_v0_1_2.json"
OUT_REPORT = RUN / "r1_exec_labels_report_v0_1_2.md"


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def dir_size(p: Path) -> int:
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file())


def decidable(signal_ts: int) -> int:
    dt = datetime.strptime(f"{int(signal_ts):014d}", "%Y%m%d%H%M%S")
    return int((dt + timedelta(minutes=1)).strftime("%Y%m%d%H%M%S"))


def seg_of(date: int) -> str:
    if date <= PROBE.SEG_TRAIN_END:
        return "train"
    if PROBE.SEG_VAL_START <= date <= PROBE.SEG_VAL_END:
        return "val"
    if date >= PROBE.SEG_TEST_START:
        return "test"
    return "gap"


def board_of(code: str) -> str:
    c = str(code).zfill(6)
    if c.startswith(("688", "689")):
        return "STAR"
    if c.startswith("60"):
        return "SH_MAIN"
    if c.startswith("00"):
        return "SZ_MAIN"
    if c.startswith("30"):
        return "GEM"
    return "OUT_OF_SCOPE"


KIND_OF = {"BASE_HOLD": "S1", "BASE_BREAKOUT": "S2", "DIP_REPAIR": "DIP"}


def main():
    t0 = time.time()
    print("=" * 70)
    print("Step3 前置 S2 审计 + 冻结执行器逐轨道标签 (零网络/零采集/零拟合)")
    print("=" * 70)

    # ---- 输入 ----
    B = json.loads((RUN / "r1_branch_at_decision_v0_1_2.json").read_text(encoding="utf-8"))
    A = json.loads((RUN / "r1_migration_table_v0_1_2.json").read_text(encoding="utf-8"))
    r1 = json.loads((RUN / "r1_dip_scan_and_gap_freeze.json").read_text(encoding="utf-8"))
    pe = {(e["code"], int(e["E"])): e for e in r1["per_event"]}
    bevents = B["events"]
    assert len(bevents) == 1013

    panel = PROBE.get_daily_parquet()
    panel["code"] = panel["code"].astype(str).str.zfill(6)
    panel["date"] = panel["date"].astype(int)
    daily_by_code = {c: g.sort_values("date").drop_duplicates("date").reset_index(drop=True)
                     for c, g in panel.groupby("code")}
    print(f"日线 panel: {len(daily_by_code)} 股 ({time.time()-t0:.1f}s)")

    # 分钟索引 (与 v0.1.2 修复件同源: 5 目录)
    dirs = [RUN / "minute", RUN / "minute_ext", RUN / "minute_car",
            RUN / "minute_collect_realtest",
            ROOT / "pullback_train" / "minute_cache_clean"]
    minute_idx: dict = {}
    for d in dirs:
        if not d.exists():
            continue
        for pf in d.glob("*.parquet"):
            try:
                m = pd.read_parquet(pf, columns=["date"])
                if "date" in m and not m.empty:
                    for day in pd.unique(m["date"].astype(str).str[:8].astype(int)):
                        minute_idx.setdefault((pf.stem, int(day)), pf)
            except Exception:
                pass

    def load_day(code, day):
        pf = minute_idx.get((code, day))
        if pf is None:
            return None
        try:
            df = pd.read_parquet(pf)
            df = df[df["date"].astype(int) // 1000000 == day].sort_values("date").reset_index(drop=True)
            return df if len(df) else None
        except Exception:
            return None

    # ================= Phase A: S2 审计 =================
    print("\n--- Phase A: S2 量比门审计 ---")
    pending_27 = []          # BASE_BREAKOUT 决策事件明细
    expo_exec = []           # 可行动候选中的 S2 暴露 -> PENDING
    expo_noaction = []       # NO_ACTION 认证中的 S2 暴露 -> PENDING
    expo_unclass = []        # UNCLASSIFIABLE 观察级暴露 (本就 PENDING, 仅登记)
    s2_flag = {}             # (code,E) -> pending/exposure 标记

    cal = D.load_calendar()
    for b in bevents:
        code, Ed = b["code"], int(b["E"])
        key = (code, Ed)
        e_df = load_day(code, Ed)
        if e_df is None:
            continue
        src = pe.get(key)
        Pd = int(src["P"]) if src else None
        dc = daily_by_code.get(code)
        if dc is None or Pd is None:
            continue
        loc = int(dc["date"].searchsorted(Pd, side="right"))
        if loc < 20:
            continue
        d_sub = dc.iloc[:loc]
        atr_ref = E.build_atr_ref(d_sub)
        cb = PROBE.evaluate_calibers(d_sub, atr_ref).get("main", {})
        if not cb.get("valid"):
            continue
        upper = float(cb["upper"])
        mon = e_df.copy()
        hh = mon["date"].astype(int).apply(E.hhmm)
        mon = mon[(hh >= E.MONITOR_START) & (hh <= E.MONITOR_END)].sort_values("date")
        ts = mon["date"].astype(int).to_numpy()
        cl = mon["close"].astype(float).to_numpy()
        vl = mon["volume"].astype(float).to_numpy()
        first_above_ts = None
        fa_i = None
        for i in range(len(mon)):
            if cl[i] > upper:
                first_above_ts = int(ts[i])
                fa_i = i
                break
        obs_branch = b["observed_first_branch"]
        obs_ts = b["observed_ts"]
        cert = b["cert"]
        seg = b["segment"]

        if b["branch_at_decision"] == "BASE_BREAKOUT":
            # S2 定分支: 逐条登记触发细节 (不删不改)
            s2_ts = b["s2_ts"]
            slot = int(str(int(s2_ts))[-6:])
            s2_i = next((i for i in range(len(mon)) if int(ts[i]) == int(s2_ts)), None)
            base_n = s2_i if s2_i is not None else None
            vr = None
            if s2_i is not None and s2_i > 0:
                bm = float(np.mean(vl[:s2_i]))
                vr = round(float(vl[s2_i]) / bm, 4) if bm > 0 else None
            pending_27.append({
                "code": code, "E": Ed, "segment": seg, "s2_ts": s2_ts,
                "slot": slot, "at_0935": slot == 93500, "base_n": base_n,
                "vr_at_trigger": vr, "s1_s2_source": b["s1_s2_source"],
                "first_above_ts": first_above_ts,
                "contract_text_s2_earlier_than_recorded": bool(
                    first_above_ts is not None and first_above_ts < int(s2_ts)),
            })
            s2_flag[key] = "PENDING_S2_UNRESOLVED"
            continue

        if first_above_ts is None:
            continue
        fa_dec = decidable(first_above_ts)
        if obs_branch == "NO_ACTION":
            expo = {"code": code, "E": Ed, "first_above_ts": first_above_ts,
                    "current": "NO_ACTION" if cert == "CERTIFIED" else "NO_ACTION(uncertified)"}
            if cert == "CERTIFIED":
                expo_noaction.append(expo)
                s2_flag[key] = "PENDING_S2_EXPOSURE_NOACTION"
            else:
                expo_unclass.append(expo)
            continue
        obs_dec = decidable(obs_ts)
        kind = KIND_OF.get(obs_branch, "S1")
        s2_wins = (fa_dec < obs_dec) or (fa_dec == obs_dec and kind == "DIP")
        if not s2_wins:
            continue
        if cert == "CERTIFIED" and b["branch_at_decision"] in ("DIP_REPAIR", "BASE_HOLD"):
            expo_exec.append({"code": code, "E": Ed, "segment": seg,
                              "first_above_ts": first_above_ts,
                              "current_branch": b["branch_at_decision"],
                              "decided_by_ts": obs_ts, "decided_kind": kind})
            s2_flag[key] = "PENDING_S2_EXPOSURE"
        else:
            expo_unclass.append({"code": code, "E": Ed, "first_above_ts": first_above_ts,
                                 "current": f"{b['branch_at_decision']}(uncertified)"})

    n_0935 = sum(1 for p in pending_27 if p["at_0935"])
    base0 = sum(1 for p in pending_27 if p["base_n"] == 0)
    base_partial = sum(1 for p in pending_27 if p["base_n"] is not None and 0 < p["base_n"] < 20)
    base20 = sum(1 for p in pending_27 if p["base_n"] is not None and p["base_n"] >= 20)
    print(f"BASE_BREAKOUT 决策事件: {len(pending_27)} (09:35 触发 {n_0935}; "
          f"空基线 {base0} / 部分 1-19 {base_partial} / 满基线 {base20})")
    print(f"可行动候选 S2 暴露 -> PENDING: {len(expo_exec)}")
    print(f"NO_ACTION 认证 S2 暴露 -> PENDING: {len(expo_noaction)}")
    print(f"UNCLASSIFIABLE 观察级暴露 (登记): {len(expo_unclass)}")

    # ================= Phase B: 冻结执行器 =================
    print("\n--- Phase B: 冻结执行器 H3/H5 ---")
    exec_events = [b for b in bevents
                   if b["cert"] == "CERTIFIED"
                   and b["branch_at_decision"] in ("DIP_REPAIR", "BASE_HOLD")
                   and (b["code"], int(b["E"])) not in s2_flag]
    print(f"执行入口: {len(exec_events)} "
          f"({dict(Counter(b['branch_at_decision'] for b in exec_events))})")

    by_code = defaultdict(list)
    for b in exec_events:
        by_code[b["code"]].append(b)

    records = []
    code_prov = {}
    exec_t0 = time.time()
    done = 0
    for code, evs in by_code.items():
        try:
            mby, mprov = D.load_minute_code(code)
        except Exception as ex:
            mby, mprov = {}, [{"error": repr(ex)[:200]}]
        code_prov[code] = [{"path": p.get("path"), "sha256": p.get("sha256"),
                            "naming": p.get("naming")} for p in mprov if isinstance(p, dict)]
        ddf = daily_by_code.get(code)
        close_map = dict(zip(ddf["date"].astype(int), ddf["close"].astype(float))) if ddf is not None else {}
        pclose_map = dict(zip(ddf["date"].astype(int),
                              ddf["pre_close"].astype(float))) if ddf is not None else {}
        for b in evs:
            Ed = int(b["E"])
            src = pe.get((code, Ed), {})
            setup = src.get("setup")
            board = board_of(code)
            signal_ts = int(b["observed_ts"])
            try:
                r = D.execute_event(code, Ed, board, None, setup, signal_ts=signal_ts,
                                    minute_by_day=mby, daily_df=ddf, cal=cal)
            except Exception as ex:
                r = {"code": code, "E": Ed, "status": "EXECUTE_ERROR", "error": repr(ex)[:300]}
            done += 1
            if done % 50 == 0:
                print(f"  ... {done}/{len(exec_events)} ({time.time()-exec_t0:.0f}s)")

            ent = r.get("entry") or {}
            fill_ts = ent.get("sim_fill_ts")
            e_seg = b["segment"]

            def corp_dates(le_ts):
                if not le_ts:
                    return []
                le_day = int(le_ts) // 1000000
                out = []
                for d in cal:
                    if d <= Ed or d > le_day:
                        continue
                    prev = close_map.get(d - 1)
                    if prev is None:
                        prev = close_map.get(max([x for x in close_map if x < d], default=None)) \
                            if close_map else None
                    pc = pclose_map.get(d)
                    if prev is not None and pc is not None and abs(prev - pc) > 1e-6:
                        out.append(d)
                return out

            def slim_track(t):
                if not t:
                    return None
                le = t.get("label_end_ts")
                xseg = bool(le is not None and seg_of(int(le) // 1000000) != e_seg)
                md = sorted({int(m.get("date")) for m in (t.get("missing_dependencies") or [])
                             if m.get("date") is not None})
                return {"st": t.get("status"), "rsn": (t.get("reason") or "")[:60],
                        "fe": bool(t.get("fully_exited")), "net": t.get("net"),
                        "rem": t.get("remaining_shares"), "le": le,
                        "oe": t.get("observed_endpoint_ts"),
                        "xseg": xseg, "md": md, "ca": corp_dates(le),
                        "valid": bool(le is not None and not xseg)}

            records.append({
                "code": code, "E": Ed, "segment": e_seg, "setup": setup,
                "branch": b["branch_at_decision"], "signal_ts": signal_ts,
                "signal_source": ("S1_" + (b["s1_s2_source"] or "?")) if b["branch_at_decision"] == "BASE_HOLD"
                                 else ("DIP_" + ("RESCAN" if b["s1_s2_source"] == "RESCAN_V0_1_2" else "R1")),
                "board": board,
                "entry_st": r.get("status"), "fill_ts": fill_ts,
                "fill_px": ent.get("sim_fill_price"), "shares": ent.get("shares"),
                "err": r.get("error"),
                "H3": slim_track(r.get("H3")) if r.get("H3") else
                      ({"st": r.get("status"), "rsn": "entry_failed", "fe": False, "net": None,
                        "rem": None, "le": None, "oe": None, "xseg": False, "md": [], "ca": [],
                        "valid": False} if r.get("status") not in (None, "EXECUTED") else None),
                "H5": slim_track(r.get("H5")) if r.get("H5") else
                      ({"st": r.get("status"), "rsn": "entry_failed", "fe": False, "net": None,
                        "rem": None, "le": None, "oe": None, "xseg": False, "md": [], "ca": [],
                        "valid": False} if r.get("status") not in (None, "EXECUTED") else None),
            })

    # ---- 汇总统计 ----
    def track_stats(tr):
        if not tr:
            return None
        return tr

    agg = {
        "n_input": len(records),
        "by_branch": {},
        "by_segment": {},
        "entry_status": dict(Counter(r["entry_st"] for r in records)),
    }
    for br in ("DIP_REPAIR", "BASE_HOLD"):
        rs = [r for r in records if r["branch"] == br]
        row = {"n": len(rs), "filled": sum(1 for r in rs if r["fill_ts"])}
        for tk in ("H3", "H5"):
            ts_ = [r[tk] for r in rs if r[tk]]
            valid = [t for t in ts_ if t["valid"]]
            row[tk] = {
                "n": len(ts_),
                "st": dict(Counter(t["st"] for t in ts_)),
                "valid": len(valid),
                "pos": sum(1 for t in valid if (t["net"] or 0) > 0),
                "neg": sum(1 for t in valid if (t["net"] or 0) <= 0),
                "residual": sum(1 for t in ts_ if t["fe"] is False and t["rem"] is not None and t["rem"] > 0),
                "xseg": sum(1 for t in ts_ if t["xseg"]),
                "corp_action": sum(1 for t in ts_ if t["ca"]),
                "valid_dates": len({int(t["le"]) // 1000000 for t in valid}),
            }
        agg["by_branch"][br] = row
    for seg in ("train", "val", "test"):
        rs = [r for r in records if r["segment"] == seg]
        agg["by_segment"][seg] = {
            "n": len(rs),
            "H3_valid": sum(1 for r in rs if r["H3"] and r["H3"]["valid"]),
            "H5_valid": sum(1 for r in rs if r["H5"] and r["H5"]["valid"]),
        }

    # ---- 缺口重推: 执行器实际依赖终点 ----
    gap_needs = defaultdict(lambda: {"events": [], "tracks": set(), "types": set()})
    for r in records:
        for tk in ("H3", "H5"):
            t = r[tk]
            if not t:
                continue
            for d in t["md"]:
                g = gap_needs[(r["code"], int(d))]
                g["events"].append({"E": r["E"], "track": tk})
                g["tracks"].add(tk)
                g["types"].add("minute_day_missing_or_gap")
    gap_list = [{"code": k[0], "date": k[1], "n_events": len(v["events"]),
                 "tracks": sorted(v["tracks"]), "events": v["events"][:6]}
                for k, v in sorted(gap_needs.items())]

    # 旧 65 对重分类 (仅对已进执行入口的事件)
    old65 = r1["gap_freeze"]["gaps"]
    rec_by_key = {(r["code"], r["E"]): r for r in records}
    old65_reclass = []
    for g in old65:
        key = (g["code"], int(g["E"]))
        ad = int(g["actual_date"]) if g.get("actual_date") else None
        r = rec_by_key.get(key)
        if r is None:
            cls = "PENDING_EVENT_NOT_IN_EXEC_SET"
        else:
            h3md = set(r["H3"]["md"]) if r["H3"] else set()
            h5md = set(r["H5"]["md"]) if r["H5"] else set()
            if ad in h3md or ad in h5md:
                cls = "CONFIRMED_BLOCKS_EXIT"
            elif (r["H3"] and r["H3"]["valid"]) and (r["H5"] and r["H5"]["valid"]):
                cls = "NOT_NEEDED_BOTH_VALID"
            elif (r["H3"] and r["H3"]["valid"]) and ad and ad >= (r["H3"]["le"] or 0) // 1000000:
                cls = "NOT_NEEDED_H3_CLOSED_BEFORE"
            else:
                cls = "NOT_REACHED_OR_PENDING"
        old65_reclass.append({"code": g["code"], "E": int(g["E"]), "slot": g["slot"],
                              "actual_date": ad, "old_class": "PENDING_EXECUTOR",
                              "new_class": cls})
    old65_dist = dict(Counter(x["new_class"] for x in old65_reclass))
    new_only = [g for g in gap_list
                if not any(x["code"] == g["code"] and x["actual_date"] == g["date"]
                           for x in old65_reclass)]
    print(f"\n缺口重推: 执行器实际阻断 (code,date) {len(gap_list)} 个; 旧 65 对重分类 {old65_dist}; "
          f"新增(不在旧 65) {len(new_only)} 个")

    # ================= 产物组装与预算自检 =================
    gen_at = time.strftime("%Y-%m-%d %H:%M:%S")
    common = {
        "generated_at": gen_at,
        "network_requests": 0, "training_performed": False, "data_collection_performed": False,
        "user_ruling": "2026-10-08: 允许第三步推进, S2 检查为前置; S2 证据未解决事件 PENDING; "
                       "输出后停止, 不采集不拟合; 新增 <=1MiB, 累计 <=55MiB",
        "contract_refs": {
            "event_v0_1_sec24": "contracts/dual_path_event_execution_contract.md §2.4 "
                                "(S2=有效箱体 且 close>box_upper_P; 无量能条件)",
            "event_v0_1_2": "contracts/dual_path_event_execution_contract_v0_1_2.md (决策时分支)",
            "execution_v0_1_1_final": "contracts/dual_path_execution_contract_v0_1_1_final.md (执行唯一依据)",
        },
        "frozen_input_hashes": {
            "r1_branch_at_decision_v0_1_2.json": sha256_file(RUN / "r1_branch_at_decision_v0_1_2.json"),
            "r1_migration_table_v0_1_2.json": sha256_file(RUN / "r1_migration_table_v0_1_2.json"),
            "r1_dip_scan_and_gap_freeze.json": sha256_file(RUN / "r1_dip_scan_and_gap_freeze.json"),
            "dual_path_execution_v0_1_1.py": sha256_file(HERE / "dual_path_execution_v0_1_1.py"),
        },
    }

    S2 = dict(common)
    S2.update({
        "phase": "R1_S2_VOLUME_GATE_AUDIT_V0_1_2",
        "findings": {
            "contract_text": "§2.4 第 2 条: 有效箱体存在且 close > box_upper_P -> BASE_BREAKOUT(S2); "
                             "已签合同目录全检索无 S2 量比的分母/窗口/早盘处置冻结定义; "
                             "'同刻量比'仅见于 CAR 线 PENDING 特征(volume_ratio_same_time), 与 S2 无关",
            "implementation": "probe scan_views L235-238: vr = vol[i]/mean(vols[max(0,i-20):i]) "
                              "(监控窗内此前<=20根bar均量); 空基线或零基线 -> np.inf 放行; "
                              "vr >= BREAKOUT_VOL_RATIO_MIN(1.5) 才记 s2_ts",
            "deviation_bidirectional": [
                "09:35 首根监控窗 bar 基线必空 -> vr=inf 自动放行 (缺少比较依据被当成最强确认)",
                "vr<1.5 压掉合同文本成立的 S2 (实现比合同文本更严); 首个 close>upper bar 可早于被记录的 s2_ts",
            ],
            "adjudication_pending": [
                "S2 是否应含量能确认 (合同 §2.4 文本无; 若加, 属合同修订须用户签)",
                "若含量能确认: 分母定义 (同刻历史/此前N根/其他), N 值, 最少基线根数, 早盘不足窗口处置",
                "空基线或零基线处置: 不得 inf 放行 (用户已裁决); 应记 NULL/待定还是视为量能未知",
            ],
        },
        "pending_base_breakout": {
            "n": len(pending_27), "at_0935": n_0935,
            "base_n_dist": {"empty(0)": base0, "partial(1-19)": base_partial, "full(>=20)": base20},
            "contract_text_s2_earlier_than_recorded": sum(
                1 for p in pending_27 if p["contract_text_s2_earlier_than_recorded"]),
            "events": pending_27,
        },
        "pending_exposure_exec_candidates": {"n": len(expo_exec), "events": expo_exec},
        "pending_exposure_noaction": {"n": len(expo_noaction), "events": expo_noaction},
        "unclassifiable_observed_exposure": {"n": len(expo_unclass), "events": expo_unclass},
        "note": "全部保留观察触发记录; 不改分支不删事件; PENDING 状态待 S2 定义裁决后重判",
    })

    EXEC = dict(common)
    EXEC.update({
        "phase": "R1_EXEC_LABELS_V0_1_2",
        "input_definition": "cert=CERTIFIED 且 branch_at_decision in {DIP_REPAIR, BASE_HOLD} "
                            "且非 S2 PENDING (BASE_BREAKOUT 27 条 + S2 暴露条已排除)",
        "executor": "dual_path_execution_v0_1_1.execute_event (manifest 绑定; 冻结 MIN_DIRS 未扩展, "
                    "预检实测全部候选 E 日分钟均在冻结目录内; daily 经 panel 快路径传入, "
                    "等价列集 date/pre_close/is_st/OHLC/volume; 分钟依赖 per-code 哈希见 code_provenance)",
        "signal_rule": "signal_ts = branch_at_decision 的决策触发 bar 时间戳 (v0.1.2 无前视分支)",
        "label_valid_rule": "完全退出(fully_exited) 且 label_end_ts 非空 且 依赖无缺失 且 不跨段; "
                            "公司行为(pre_close 跳变)仅记录不判无效, 训练准入留拟合合同裁决; "
                            "H5 缺失不连带删除有效 H3; 343 条认证无行动不是亏损负样本",
        "aggregates": agg,
        "gap_rederivation": {
            "note": "执行器实际依赖终点重推; 旧 65 对为退出路径需求且仅原 187 人口; "
                    "本表不构成补采批准, 同源重取一致仅证明缺口可复现不据此恢复样本",
            "new_blocked_pairs_n": len(gap_list),
            "new_blocked_pairs": gap_list,
            "old65_reclassified": {"distribution": old65_dist, "pairs": old65_reclass},
            "not_in_old65_n": len(new_only),
        },
        "code_provenance": code_prov,
        "records": records,
    })

    # ---- 报告 md ----
    R = EXEC["aggregates"]
    lines = [
        "# 第三步: S2 前置审计 + 冻结执行器逐轨道标签 (v0.1.2 决策人口)", "",
        f"生成: {gen_at} | 零网络 / 零采集 / 零拟合; 执行器 v0.1.1 (manifest 绑定)", "",
        "## 1. S2 量比门审计 (前置检查)", "",
        "- 已签合同 §2.4 的 S2 只有 `close > box_upper_P`; **任何已签合同无 S2 量比定义**;",
        "  probe 实现自加 `vr>=1.5` 门 (此前<=20根监控窗bar均量, 空基线/零基线 -> inf 放行).",
        "- 双向偏离: 09:35 空基线自动放行; `vr<1.5` 压掉合同文本成立的 S2.",
        f"- **BASE_BREAKOUT 决策事件 {len(pending_27)} 条 -> PENDING_S2_UNRESOLVED** "
        f"(其中 09:35 触发 {n_0935} 条; 空基线 {base0} / 部分 1-19 {base_partial} / 满基线 {base20}).",
        f"- 可行动候选中 S2 暴露 (合同文本 S2 会更早/同bar压过当前决策) **{len(expo_exec)} 条 -> PENDING**.",
        f"- NO_ACTION 认证中存在 close>upper **{len(expo_noaction)} 条 -> 认证 S2 依赖, PENDING**.",
        f"- UNCLASSIFIABLE 观察级暴露 {len(expo_unclass)} 条 (本就 PENDING, 仅登记).",
        "- 裁决问题见 audit JSON `findings.adjudication_pending`; 未裁决前不得自行换算法.", "",
        "## 2. 执行入口", "",
        f"- 认证可行动 516 = BASE_HOLD 159 + BASE_BREAKOUT 27 + DIP_REPAIR 330;",
        f"  排除 S2 PENDING 后执行入口 **{R['n_input']} 条**"
        f" (DIP {R['by_branch']['DIP_REPAIR']['n']} / S1 {R['by_branch']['BASE_HOLD']['n']}).",
        f"- 入场状态: {R['entry_status']}.",
        "- 516 是待核验入口, 不是训练样本数; 343 条认证无行动不是亏损负样本.", "",
        "## 3. 逐轨道标签 (H3 主 / H5 辅, 独立核验)", "",
        "| 分支 | n | 成交 | H3 有效 | H3 正/负 | H3 残仓 | H3 跨段 | H3 公司行为 | H5 有效 | H5 正/负 | H5 残仓 |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for br in ("DIP_REPAIR", "BASE_HOLD"):
        x = R["by_branch"][br]
        h3, h5 = x["H3"], x["H5"]
        lines.append(f"| {br} | {x['n']} | {x['filled']} | {h3['valid']} | "
                     f"{h3['pos']}/{h3['neg']} | {h3['residual']} | {h3['xseg']} | {h3['corp_action']} | "
                     f"{h5['valid']} | {h5['pos']}/{h5['neg']} | {h5['residual']} |")
    lines += [
        "", "| 段 | n | H3 有效 | H5 有效 |", "|---|---|---|---|",
    ]
    for seg in ("train", "val", "test"):
        s = R["by_segment"][seg]
        lines.append(f"| {seg} | {s['n']} | {s['H3_valid']} | {s['H5_valid']} |")
    lines += [
        "", "有效标签 = 完全退出 + label_end_ts 非空 + 依赖无缺失 + 不跨段; "
        "公司行为 (pre_close 跳变) 仅记录, 训练准入留拟合合同裁决; "
        "H5 缺失不连带删除有效 H3。所有模拟成交为 RESEARCH_APPROXIMATION。", "",
        "## 4. 重新推导的缺口需求", "",
        f"- 执行器实际阻断 (code, date): **{len(gap_list)}** 个物理日 "
        f"(新增不在旧 65 对内: {len(new_only)} 个).",
        f"- 旧 65 对重分类: {old65_dist}.",
        "- 本清单不构成补采批准; 同源重取一致仅证明缺口可复现, 不据此恢复样本; "
        "旧 65 对仅覆盖原 187 人口, 不能当 DIP 后完整需求。", "",
        "## 5. 边界", "",
        "- 输出后停止: 不采集、不拟合、不跑训练; S2 定义与补采待用户裁决。",
        "- 未删除旧证据; 未复制分钟缓存。",
        f"- 预算: 本步新增 (以下实测行填写) 限 1 MiB; 研究目录累计限 55 MiB。",
    ]
    report_md = "\n".join(lines)
    # ---- 预算自检 + 写盘 ----
    base_before = dir_size(RUN)
    docs = {"s2": (OUT_S2, S2), "exec": (OUT_EXEC, EXEC)}
    texts = {k: json.dumps(v, ensure_ascii=False, indent=1) for k, (_, v) in docs.items()}
    total_new = sum(len(t.encode("utf-8")) for t in texts.values()) + len(report_md.encode("utf-8"))
    if total_new > CAP_NEW_BYTES:
        texts = {k: json.dumps(v, ensure_ascii=False, separators=(",", ":"))
                 for k, (_, v) in docs.items()}
        total_new = sum(len(t.encode("utf-8")) for t in texts.values()) + len(report_md.encode("utf-8"))
    if total_new > CAP_NEW_BYTES:
        raise RuntimeError(f"新增 {total_new} B 超 1 MiB, 停机不写")
    if base_before + total_new > CAP_BASE_BYTES:
        raise RuntimeError(f"累计 {base_before}+{total_new} 超 55 MiB, 停机不写")
    # newline="" 关闭换行翻译 (CRLF 会放大磁盘占用且与自检口径不一致)
    OUT_S2.write_text(texts["s2"], encoding="utf-8", newline="")
    OUT_EXEC.write_text(texts["exec"], encoding="utf-8", newline="")
    OUT_REPORT.write_text(report_md, encoding="utf-8", newline="")
    base_after = dir_size(RUN)

    print(f"\n写入: {OUT_S2.name} / {OUT_EXEC.name} / {OUT_REPORT.name}")
    print(f"预算: 前 {base_before} B + 新增 {total_new} B = {base_after} B "
          f"({base_after/1048576:.3f} MiB / 55 MiB; 本步新增 {total_new/1024:.0f} KB / 1 MiB)")
    print(f"\n=== 执行汇总 ===")
    print(json.dumps(agg["entry_status"], ensure_ascii=False))
    for br in ("DIP_REPAIR", "BASE_HOLD"):
        x = agg["by_branch"][br]
        print(f"{br}: n={x['n']} filled={x['filled']} | H3 valid={x['H3']['valid']} "
              f"pos/neg={x['H3']['pos']}/{x['H3']['neg']} 残仓={x['H3']['residual']} "
              f"xseg={x['H3']['xseg']} ca={x['H3']['corp_action']} | "
              f"H5 valid={x['H5']['valid']} pos/neg={x['H5']['pos']}/{x['H5']['neg']} 残仓={x['H5']['residual']}")
    print(f"耗时 {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
