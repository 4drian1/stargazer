# -*- coding: utf-8 -*-
"""R1 正确性修复 v0.1.2: 决策时分支 + 迁移表 A + 分支表 B + 缺口审计 C.

零网络 / 零采集 / 零拟合. 写前预算自检: 研究目录累计 <= 55 MiB, 本步新增 <= 1 MiB.

输入 (全部本地只读):
- r1_dip_scan_and_gap_freeze.json (1,013 per_event, 含 v3 冻结 s1/s2 + r1 DIP 扫描)
- gcp_247_full_population_inventory_v3.json (口径参照)
- minute_grid.json (版本化网格: <20260624 241 槽, >=20260624 242 槽)
- minute/ minute_ext/ minute_car/ minute_collect_realtest/ + pullback_train/minute_cache_clean
- 日线 panel (data_daily_parts, 只读)
- execution_reconcile_v0_1_1/run_20261005_125148/limited_reconciliation.json (16 条复盘)

输出:
- r1_migration_table_v0_1_2.json        (A: 1,013 迁移表)
- r1_branch_at_decision_v0_1_2.json     (B: 无前视分支表 + 反例报告)
- r1_gap_dependency_audit_v0_1_2.json   (C: C1 退出依赖 / C2 决策依赖 / C3 物理去重)

核心规则 (合同 v0.1.2 §1):
1. 可决策时刻 = 信号 bar 时间戳 + 1 分钟 (真实 datetime 运算);
2. 仅比较可行动分支 S1/S2/DIP(BREAK|ALREADY_BELOW); RECLAIMED_AT_S 不行动不抢占;
3. 最早可决策取胜; 同 bar 并列按 S1>S2>DIP; 后续触发不回填;
4. 监控窗内首决策前缺槽 -> UNCLASSIFIABLE(BLOCKED_BY_PRE_DEP_GAP);
   无信号但监控窗缺槽 -> 不得认证 NO_ACTION(NO_SIGNAL_UNVERIFIED);
   窗外/决策后缺槽 -> 不影响分支认证, 登记为执行依赖;
5. 对 v3 从未扫描 S1/S2 的 295 条缺槽缓存补扫 scan_views (结果标 RESCAN, 不覆盖 v3).
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

RUN = HERE / "gcp_round_b_small_sample_v1"
CAP_BASE_BYTES = 55 * 1024 * 1024          # 用户 2026-10-08 批准: 累计上限暂调 55 MiB
CAP_NEW_BYTES = 1 * 1024 * 1024            # 本步新增产物合计 <= 1 MiB
MON_LO, MON_HI = 93500, 145500             # 监控窗 [09:35, 14:55] (槽位 HHMMSS)
PRIO = {"S1": 0, "S2": 1, "DIP": 2}
BRANCH_OF = {"S1": "BASE_HOLD", "S2": "BASE_BREAKOUT", "DIP": "DIP_REPAIR"}

OUT_A = RUN / "r1_migration_table_v0_1_2.json"
OUT_B = RUN / "r1_branch_at_decision_v0_1_2.json"
OUT_C = RUN / "r1_gap_dependency_audit_v0_1_2.json"


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def dir_size(p: Path) -> int:
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file())


def decidable(signal_ts: int) -> int:
    """信号 bar (开盘分钟戳 YYYYMMDDHHMM00) 完成时刻 = +1 分钟, 真实 datetime 进位."""
    dt = datetime.strptime(f"{int(signal_ts):014d}", "%Y%m%d%H%M%S")
    return int((dt + timedelta(minutes=1)).strftime("%Y%m%d%H%M%S"))


def slot_of(ts: int) -> int:
    return int(str(int(ts))[-6:])


def first_actionable(triggers):
    """triggers: [(decidable, prio, kind, raw_ts)] -> 最早可决策; 同刻按优先级; 不回填."""
    if not triggers:
        return None
    return sorted(triggers, key=lambda c: (c[0], c[1]))[0]


def main():
    t0 = time.time()
    print("=" * 70)
    print("R1 正确性修复 v0.1.2 (决策时分支; 零网络/零采集/零拟合)")
    print("=" * 70)

    # ---- 输入 ----
    r1 = json.loads((RUN / "r1_dip_scan_and_gap_freeze.json").read_text(encoding="utf-8"))
    per = r1["per_event"]
    assert len(per) == 1013, f"per_event != 1013: {len(per)}"
    grid_full = json.loads((RUN / "minute_grid.json").read_text(encoding="utf-8"))["grid"]

    panel = PROBE.get_daily_parquet()
    panel["code"] = panel["code"].astype(str).str.zfill(6)
    panel["date"] = panel["date"].astype(int)
    daily_by_code = {c: g.sort_values("date").reset_index(drop=True)
                     for c, g in panel.groupby("code")}
    print(f"日线: {len(daily_by_code)} 股")

    # 分钟索引 (与 v3 同源: 5 个目录)
    dirs = [RUN / "minute", RUN / "minute_ext", RUN / "minute_car",
            RUN / "minute_collect_realtest",
            ROOT / "pullback_train" / "minute_cache_clean"]
    minute_idx: dict = {}
    n_files = 0
    for d in dirs:
        if not d.exists():
            continue
        for pf in d.glob("*.parquet"):
            n_files += 1
            try:
                m = pd.read_parquet(pf, columns=["date"])
                if "date" in m and not m.empty:
                    code = pf.stem
                    for day in pd.unique(m["date"].astype(str).str[:8].astype(int)):
                        minute_idx.setdefault((code, int(day)), pf)
            except Exception:
                pass
    print(f"分钟索引: {len(minute_idx)} (code,day); 文件 {n_files}")

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

    # 复盘 (16 条) — C1 分类用
    rec_path = RUN / "execution_reconcile_v0_1_1" / "run_20261005_125148" / "limited_reconciliation.json"
    reconcile = {}
    if rec_path.exists():
        for e in json.loads(rec_path.read_text(encoding="utf-8")):
            reconcile[(e["code"], int(e["E"]))] = e

    # ---- 逐事件 ----
    A_events, B_events = [], []
    C2_events = []
    ver = {"fwd": [], "same_bar": [], "later": [], "reclaimed_first": []}
    rescan_found_log = []
    cnt = Counter()

    for ev in per:
        code, Ed, Pd = ev["code"], int(ev["E"]), int(ev["P"])
        seg, mutual = ev["segment"], ev["mutual"]
        dps = ev["resolved_branch"]                    # day_pattern_summary (旧全天优先级, 仅描述)
        cov_label = ev["e_day_coverage"]
        s1_v3, s2_v3 = ev.get("s1_ts"), ev.get("s2_ts")
        dip_view, dip_ts = ev.get("dip_view"), ev.get("dip_signal_ts")
        dip_act = dip_view in ("BREAK", "ALREADY_BELOW") and dip_ts is not None

        # -- E 日缓存核查 (逐槽, 不信 COMPLETE 标签) --
        e_df = load_day(code, Ed)
        missing, dups, unexpected, bad_price, zero_all = [], [], [], False, False
        if e_df is None:
            cov_actual = "NO_FILE"
        else:
            expected = E.expected_grid(grid_full, Ed)
            slots = [slot_of(t) for t in e_df["date"].astype(int)]
            sset = set(slots)
            missing = sorted(set(expected) - sset)
            dups = sorted({s for s in slots if slots.count(s) > 1})
            unexpected = sorted(sset - set(expected))
            bad_price = bool((e_df[["open", "high", "low", "close"]].astype(float) <= 0).any().any())
            zero_all = bool((e_df["volume"].astype(float) > 0).sum() == 0)
            cov_actual = "COMPLETE" if not (missing or dups or unexpected or bad_price or zero_all) else "GAP"
        label_mismatch = (cov_label != cov_actual)

        # -- 补扫: v3 对非 COMPLETE 从未跑 scan_views --
        s1, s2 = s1_v3, s2_v3
        s1s2_source = "V3_FROZEN" if cov_label == "COMPLETE" else None
        rescan_found = False
        if cov_label != "COMPLETE" and e_df is not None:
            dc = daily_by_code.get(code)
            loc = int(dc["date"].searchsorted(Pd, side="right")) if dc is not None else 0
            box_ok = False
            if dc is not None and loc >= 20:
                d_sub = dc.iloc[:loc]
                atr_ref = E.build_atr_ref(d_sub)
                calibs = PROBE.evaluate_calibers(d_sub, atr_ref)
                cb = calibs.get("main", {})
                box_ok = bool(cb.get("valid"))
                if box_ok:
                    a14 = pd.Series(np.maximum(
                        d_sub["high"].astype(float).to_numpy() - d_sub["low"].astype(float).to_numpy(),
                        np.abs(d_sub["high"].astype(float).to_numpy()
                               - np.concatenate([[d_sub["close"].astype(float).to_numpy()[0]],
                                                 d_sub["close"].astype(float).to_numpy()[:-1]]))
                    )).rolling(14).mean().to_numpy()
                    amed60 = float(np.nanmedian(a14[-60:])) if len(a14) >= 14 else np.nan
                    contraction = (atr_ref / amed60 if np.isfinite(amed60) and amed60 > 0 else np.nan)
                    vol_arr = d_sub["volume"].astype(float).to_numpy()
                    last10 = vol_arr[-10:]
                    cv = float(np.std(last10, ddof=1) / np.mean(last10)) if np.mean(last10) > 0 else np.nan
                    c_ok = (np.isfinite(contraction) and contraction <= PROBE.ATR_CONTRACTION_MAX)
                    v_ok = (np.isfinite(cv) and cv <= PROBE.VOL_STABILITY_CV_MAX)
                    views = PROBE.scan_views(e_df, cb["upper"], cb["lower"], atr_ref, c_ok, v_ok)
                    s1, s2 = views["s1_ts"], views["s2_ts"]
                    s1s2_source = "RESCAN_V0_1_2"
                    if s1 is not None or s2 is not None:
                        rescan_found = True
                        rescan_found_log.append({"code": code, "E": Ed, "mutual": mutual,
                                                 "s1_ts": s1, "s2_ts": s2,
                                                 "dip_ts": dip_ts, "dip_view": dip_view,
                                                 "missing_n": len(missing)})
            if not box_ok:
                s1s2_source = "RESCAN_V0_1_2_BOX_INVALID"

        # -- 可行动触发与可决策时刻 --
        trig = []
        if s1 is not None:
            trig.append((decidable(s1), PRIO["S1"], "S1", s1))
        if s2 is not None:
            trig.append((decidable(s2), PRIO["S2"], "S2", s2))
        if dip_act:
            trig.append((decidable(dip_ts), PRIO["DIP"], "DIP", dip_ts))
        first = first_actionable(trig)
        if first:
            obs_branch = BRANCH_OF[first[2]]
            obs_ts, obs_dec, obs_slot = first[3], first[0], slot_of(first[3])
        else:
            obs_branch, obs_ts, obs_dec, obs_slot = "NO_ACTION", None, None, None

        # -- 认证 (缺槽不强行认证) --
        anomaly = bool(dups or unexpected or bad_price or zero_all)
        if cov_actual == "NO_FILE":
            cert, unclass_reason = "NO_CACHE", "NO_CACHE"
            gap_status = "NO_FILE"
        elif anomaly:
            cert, unclass_reason = "BLOCKED_BY_DATA_ANOMALY", "DATA_ANOMALY"
            gap_status = "GAP_BLOCKS_BRANCH"
        elif obs_slot is not None:
            pre = [s for s in missing if MON_LO <= s < obs_slot]
            if pre:
                cert, unclass_reason = "BLOCKED_BY_PRE_DEP_GAP", "PRE_DEP_GAP"
                gap_status = "GAP_BLOCKS_BRANCH"
            else:
                cert, unclass_reason = "CERTIFIED", None
                gap_status = "E_DAY_COMPLETE" if not missing else "GAP_EXEC_ONLY"
        else:
            mon = [s for s in missing if MON_LO <= s <= MON_HI]
            if mon:
                cert, unclass_reason = "NO_SIGNAL_UNVERIFIED", "MONITOR_GAP"
                gap_status = "GAP_BLOCKS_BRANCH"
            else:
                cert, unclass_reason = "CERTIFIED", None
                gap_status = "E_DAY_COMPLETE" if not missing else "GAP_OUTSIDE_MONITOR"

        branch = "UNCLASSIFIABLE" if cert != "CERTIFIED" else obs_branch

        # -- 旧规则前视验证 (仅用 v3 冻结触发, 不混入补扫新发现) --
        trig_orig = []
        if s1_v3 is not None:
            trig_orig.append((decidable(s1_v3), PRIO["S1"], "S1", s1_v3))
        if s2_v3 is not None:
            trig_orig.append((decidable(s2_v3), PRIO["S2"], "S2", s2_v3))
        if dip_act:
            trig_orig.append((decidable(dip_ts), PRIO["DIP"], "DIP", dip_ts))
        fo = first_actionable(trig_orig)
        obs_orig = BRANCH_OF[fo[2]] if fo else "NO_ACTION"
        fwd = (obs_orig == "DIP_REPAIR" and dps in ("BASE_HOLD", "BASE_BREAKOUT"))
        same_bar = bool(dip_act and ((s1_v3 is not None and dip_ts == s1_v3)
                                     or (s2_v3 is not None and dip_ts == s2_v3)))
        dip_later = bool(dip_act and obs_orig in ("BASE_HOLD", "BASE_BREAKOUT")
                         and fo is not None and decidable(dip_ts) > fo[0])
        min_s1s2_v3 = min([t for t in (s1_v3, s2_v3) if t is not None], default=None)
        reclaimed_first = bool(dip_view == "RECLAIMED_AT_S" and dip_ts is not None
                               and min_s1s2_v3 is not None and dip_ts < min_s1s2_v3)

        branch_changed = (branch != dps) if branch != "UNCLASSIFIABLE" else None
        fwd_kind = None
        if fwd:
            fwd_kind = "S1_OVER_DIP" if s1_v3 is not None else "S2_OVER_DIP"

        # -- 身份迁移 (仅 4 类, 互斥; 缺槽/回填放独立字段) --
        if mutual == "1_有行动":
            identity = "ACTION_IDENTITY_STABLE"
        elif branch in ("DIP_REPAIR", "BASE_HOLD", "BASE_BREAKOUT"):
            identity = "NEW_ACTION_CANDIDATE"
        elif branch == "NO_ACTION":
            identity = "STILL_NO_ACTION"
        else:
            identity = "UNJUDGEABLE"

        reasons = []
        if identity == "ACTION_IDENTITY_STABLE":
            reasons.append("IDENTITY_STABLE")
        if fwd:
            reasons.append("PRIORITY_BACKFILL_FIXED")
        if cert == "BLOCKED_BY_PRE_DEP_GAP":
            reasons.append("GAP_BLOCKS_FIRST_BRANCH")
        if cert == "NO_SIGNAL_UNVERIFIED":
            reasons.append("NO_SIGNAL_UNVERIFIED")
        if cert == "NO_CACHE":
            reasons.append("NO_CACHE")
        if cert == "BLOCKED_BY_DATA_ANOMALY":
            reasons.append("DATA_ANOMALY")
        if rescan_found:
            reasons.append("S1_S2_FOUND_ON_RESCAN")
        if identity == "NEW_ACTION_CANDIDATE" and mutual != "1_有行动":
            reasons.append("NEW_DIP_CANDIDATE" if branch == "DIP_REPAIR" else "NEW_S1_S2_CANDIDATE")
        if label_mismatch:
            reasons.append("COVERAGE_LABEL_MISMATCH")

        if cert == "CERTIFIED" and branch in ("DIP_REPAIR", "BASE_HOLD", "BASE_BREAKOUT"):
            recommendation = "EXEC_READY"
        elif cert == "CERTIFIED":
            recommendation = "NO_ACTION_CERTIFIED"
        else:
            recommendation = "PENDING"

        # -- 计数与登记 --
        cnt["identity_" + identity] += 1
        cnt["branch_" + branch] += 1
        cnt["cert_" + cert] += 1
        cnt["gap_" + gap_status] += 1
        if fwd:
            ver["fwd"].append({"code": code, "E": Ed, "dip_ts": dip_ts,
                               "s1_v3": s1_v3, "s2_v3": s2_v3, "old": dps, "fixed": "DIP_REPAIR"})
        if same_bar:
            ver["same_bar"].append({"code": code, "E": Ed, "dip_ts": dip_ts,
                                    "s1_v3": s1_v3, "s2_v3": s2_v3, "old": dps})
        if dip_later:
            ver["later"].append({"code": code, "E": Ed, "dip_ts": dip_ts,
                                 "s1_v3": s1_v3, "s2_v3": s2_v3, "old": dps})
        if reclaimed_first:
            ver["reclaimed_first"].append({"code": code, "E": Ed, "reclaim_ts": dip_ts,
                                           "s1_v3": s1_v3, "s2_v3": s2_v3, "old": dps,
                                           "naive_wrong": "DIP_REPAIR",
                                           "correct": "BASE_HOLD" if s1_v3 is not None else "BASE_BREAKOUT"})

        # C2: 有缺槽/NO_FILE/异常的事件
        if cov_actual != "COMPLETE":
            if branch not in ("NO_ACTION",) or gap_status != "GAP_OUTSIDE_MONITOR":
                dclass = {"NO_FILE": "NO_FILE",
                          "BLOCKED_BY_PRE_DEP_GAP": "BLOCKS_FIRST_BRANCH",
                          "NO_SIGNAL_UNVERIFIED": "NO_SIGNAL_UNVERIFIED",
                          "BLOCKED_BY_DATA_ANOMALY": "DATA_ANOMALY",
                          }.get(cert, "EXEC_ONLY_GAPS" if branch in ("DIP_REPAIR", "BASE_HOLD", "BASE_BREAKOUT")
                                else "IRRELEVANT_NO_TRADE")
                C2_events.append({"code": code, "E": Ed, "segment": seg, "mutual": mutual,
                                  "branch_at_decision": branch, "cert": cert,
                                  "decision_class": dclass,
                                  "missing_slots": missing, "missing_n": len(missing),
                                  "dups": dups, "unexpected": unexpected,
                                  "bad_price": bad_price, "zero_all": zero_all,
                                  "s1_s2_rescan_found": rescan_found,
                                  "coverage_label": cov_label, "coverage_actual": cov_actual})

        A_events.append({
            "code": code, "E": Ed,
            "identity": identity, "branch_at_decision": branch,
            "branch_changed": branch_changed,
            "gap_status": gap_status, "missing_n": len(missing),
            "label_mismatch": label_mismatch, "s1_s2_source": s1s2_source,
            "reasons": reasons, "recommendation": recommendation,
        })
        B_events.append({
            "code": code, "E": Ed, "segment": seg,
            "s1_ts": s1, "s2_ts": s2, "dip_ts": dip_ts, "dip_view": dip_view,
            "s1_dec": decidable(s1) if s1 is not None else None,
            "s2_dec": decidable(s2) if s2 is not None else None,
            "dip_dec": decidable(dip_ts) if dip_ts is not None else None,
            "observed_first_branch": obs_branch, "observed_ts": obs_ts,
            "branch_at_decision": branch, "cert": cert,
            "unclass_reason": unclass_reason,
            "day_pattern_summary": dps, "branch_changed": branch_changed,
            "fwd_old_rule": fwd, "fwd_kind": fwd_kind,
            "same_bar": same_bar, "dip_later": dip_later,
            "missing_n": len(missing), "s1_s2_source": s1s2_source,
        })

    # ---- 验证 (用户口径: 46 / 6 / 11 / 15) ----
    verification = {
        "fwd_strictly_earlier_actionable": {"expected": 46, "found": len(ver["fwd"]),
                                            "match": len(ver["fwd"]) == 46},
        "same_bar_actionable": {"expected": 6, "found": len(ver["same_bar"]),
                                "match": len(ver["same_bar"]) == 6},
        "dip_later_actionable": {"expected": 11, "found": len(ver["later"]),
                                 "match": len(ver["later"]) == 11},
        "reclaimed_first_then_s1s2": {"expected": 15, "found": len(ver["reclaimed_first"]),
                                      "match": len(ver["reclaimed_first"]) == 15},
        "note": "验证仅用 v3 冻结触发集; 补扫新发现单列, 不混入.",
    }

    # ---- C1: 原 65 对退出依赖 ----
    C1_gaps = []
    c1_dist = Counter()
    for g in r1["gap_freeze"]["gaps"]:
        key = (g["code"], int(g["E"]))
        rc = reconcile.get(key)
        if rc is None:
            cls, basis = "PENDING_EXECUTOR", "无复盘结果, 实际依赖终点未知"
        else:
            def tcls(track, relevant):
                if not relevant:
                    return "NA"
                st = (track or {}).get("status")
                le = (track or {}).get("label_end_ts")
                if st == "SIM_COMPLETE":
                    return "NOT_BLOCKED_VERIFIED"
                if le is None:
                    return "PENDING_EXECUTOR"
                return "BLOCKS" if int(g["actual_date"]) <= int(str(int(le))[:8]) else "NO_IMPACT"
            c3 = tcls(rc.get("H3"), int(g["offset_days"]) <= 3)
            c5 = tcls(rc.get("H5"), True)
            parts = [p for p in (c3, c5) if p != "NA"]
            if any(p == "BLOCKS" for p in parts):
                cls = "BLOCKS_EXIT"
            elif all(p == "NOT_BLOCKED_VERIFIED" for p in parts):
                cls = "NOT_BLOCKED_VERIFIED"
            elif all(p in ("NO_IMPACT", "NOT_BLOCKED_VERIFIED") for p in parts):
                cls = "NO_IMPACT"
            else:
                cls = "PENDING_EXECUTOR"
            basis = f"reconcile: H3={c3}, H5={c5}"
        note_extra = ""
        if g.get("known_source_hole"):
            note_extra = "同源重取一致仅证明缺口可复现, 不证明数据有效, 不据此恢复样本"
        C1_gaps.append({"code": g["code"], "E": int(g["E"]), "slot": g["slot"],
                        "actual_date": g["actual_date"], "use": g["use"],
                        "cache_status": g["cache_status"],
                        "known_source_hole": g.get("known_source_hole", False),
                        "exit_class": cls, "basis": basis, "note": note_extra})
        c1_dist[cls] += 1

    # ---- C3: 物理 (code, date) 去重 ----
    c3_map = defaultdict(lambda: {"in_C1": False, "in_C2": False, "c1_slots": [], "c2_events": []})
    for g in C1_gaps:
        if g["actual_date"] is None:
            continue
        k = (g["code"], int(g["actual_date"]))
        c3_map[k]["in_C1"] = True
        c3_map[k]["c1_slots"].append({"E": g["E"], "slot": g["slot"], "use": g["use"],
                                      "exit_class": g["exit_class"]})
    for e in C2_events:
        k = (e["code"], int(e["E"]))
        c3_map[k]["in_C2"] = True
        c3_map[k]["c2_events"].append({"E": e["E"], "decision_class": e["decision_class"],
                                       "missing_n": e["missing_n"]})
    C3_entries = [{"code": k[0], "date": k[1], **v} for k, v in sorted(c3_map.items())]

    # ---- 汇总 ----
    gen_at = time.strftime("%Y-%m-%d %H:%M:%S")
    common = {
        "generated_at": gen_at,
        "network_requests": 0, "training_performed": False, "data_collection_performed": False,
        "contract_refs": {
            "v0_1_2": "contracts/dual_path_event_execution_contract_v0_1_2.md (SIGNED 2026-10-08, 仅 §2.4)",
            "v0_1_1_final_execution": "contracts/dual_path_execution_contract_v0_1_1_final.md (成交/退出/限价唯一依据)",
            "v0_1_event": "contracts/dual_path_event_execution_contract.md (SIGNED; §2.4 全天优先级降级为 day_pattern_summary)",
        },
        "frozen_input_hashes": {
            "r1_dip_scan_and_gap_freeze.json": sha256_file(RUN / "r1_dip_scan_and_gap_freeze.json"),
            "gcp_247_full_population_inventory_v3.json": sha256_file(RUN / "gcp_247_full_population_inventory_v3.json"),
            "minute_grid.json": sha256_file(RUN / "minute_grid.json"),
            "dual_path_event_execution_contract_v0_1_2.md": sha256_file(HERE / "contracts" / "dual_path_event_execution_contract_v0_1_2.md"),
        },
        "budget_rule": {"cap_base_mib": 55, "cap_new_bytes": CAP_NEW_BYTES,
                        "approved_by_user": "2026-10-08; 仅合同/修正表/反例报告; 不删旧证据"},
    }

    A = dict(common)
    A.update({
        "phase": "R1_MIGRATION_TABLE_V0_1_2", "population_n": len(A_events),
        "identity_definitions": {
            "ACTION_IDENTITY_STABLE": "旧 mutual=1 (identity 即旧身份, mutual 可由此推导); 行动身份不变 (分支可能被前视回填, 见 branch_changed)",
            "NEW_ACTION_CANDIDATE": "旧 mutual!=1 且 branch_at_decision 为可行动分支 (含补扫新发现)",
            "STILL_NO_ACTION": "旧 mutual!=1 且 branch_at_decision=NO_ACTION 已认证",
            "UNJUDGEABLE": "旧 mutual!=1 且 branch_at_decision=UNCLASSIFIABLE (NO_FILE/缺槽/异常)",
        },
        "fields_note": "segment/mutual/observed_first_branch 见 B 表 (r1_branch_at_decision_v0_1_2.json), 避免重复占预算.",
        "identity_distribution": {k.replace("identity_", ""): v for k, v in cnt.items() if k.startswith("identity_")},
        "branch_at_decision_distribution": {k.replace("branch_", ""): v for k, v in cnt.items() if k.startswith("branch_")},
        "certification_distribution": {k.replace("cert_", ""): v for k, v in cnt.items() if k.startswith("cert_")},
        "gap_status_distribution": {k.replace("gap_", ""): v for k, v in cnt.items() if k.startswith("gap_")},
        "crosstab_identity_x_gap": None,  # filled below
        "s1_s2_rescan": {"note": "v3 对 E 日非 COMPLETE 的 295 条从未跑 scan_views; 本轮补扫, 不覆盖 v3 冻结记录",
                         "n_rescanned": sum(1 for e in A_events if e["s1_s2_source"] == "RESCAN_V0_1_2"),
                         "n_found_s1s2": len(rescan_found_log),
                         "found": rescan_found_log},
        "note_summing": "各分布含重叠维度时不得直接相加; 61 旧口径已纠正为 46 可行动前视 (15 条 RECLAIMED_AT_S 不行动).",
        "events": A_events,
    })
    xtab = defaultdict(Counter)
    for e in A_events:
        xtab[e["identity"]][e["gap_status"]] += 1
    A["crosstab_identity_x_gap"] = {k: dict(v) for k, v in xtab.items()}

    B = dict(common)
    B.update({
        "phase": "R1_BRANCH_AT_DECISION_V0_1_2", "population_n": len(B_events),
        "rule_summary": {
            "decidable_moment": "信号 bar 时间戳 + 1 分钟 (真实 datetime 运算; DIP 依赖 close, 须 bar 完成)",
            "actionable_only": "S1/S2/DIP(BREAK|ALREADY_BELOW); RECLAIMED_AT_S/APPROACH/WATCH 不行动不抢占",
            "arbitration": "最早可决策取胜; 同 bar 并列按 S1>S2>DIP; 后续触发不回填; 分支不等于成交",
            "certification": "监控窗内首决策前缺槽->UNCLASSIFIABLE; 无信号+监控窗缺槽->不得认证 NO_ACTION; "
                             "窗外/决策后缺槽->执行依赖, 由执行器按实际依赖终点判断",
            "prefix_gate": "合同 §7.5 前缀完整性在执行器阶段强制, 不在分支认证阶段替代执行",
        },
        "verification": verification,
        "distribution": {
            "branch_at_decision": {k.replace("branch_", ""): v for k, v in cnt.items() if k.startswith("branch_")},
            "observed_first_branch": dict(Counter(e["observed_first_branch"] for e in B_events)),
            "branch_changed_true": sum(1 for e in B_events if e["branch_changed"] is True),
            "branch_changed_null_uncertified": sum(1 for e in B_events if e["branch_changed"] is None),
            "fwd_old_rule": len(ver["fwd"]),
        },
        "counterexamples": {
            "note": "反例覆盖 '先 RECLAIMED_AT_S 后 S1': RECLAIMED 不行动不抢占; "
                    "若误将 RECLAIMED 计入 min(可决策) 会错判 DIP_REPAIR (旧 61 口径的错误来源).",
            "reclaimed_first_then_s1s2": ver["reclaimed_first"],
            "forward_look_fixed_sample": ver["fwd"][:5],
            "forward_look_fixed_n": len(ver["fwd"]),
            "same_bar_priority": ver["same_bar"],
        },
        "events": B_events,
    })

    C = dict(common)
    C.update({
        "phase": "R1_GAP_DEPENDENCY_AUDIT_V0_1_2",
        "C1_exit_dependencies": {
            "note": "原 65 对 (r1 gap_freeze); 仅退出路径且仅原 187 条人口; 不是最终补采清单, "
                    "补采须待执行器跑出实际依赖终点后再定; 同源重取一致仅证明缺口可复现, 不据此恢复样本.",
            "classification_rule": "有复盘: SIM_COMPLETE->NOT_BLOCKED_VERIFIED; label_end 比较->BLOCKS/NO_IMPACT; "
                                   "无复盘->PENDING_EXECUTOR",
            "n": len(C1_gaps), "distribution": dict(c1_dist), "gaps": C1_gaps,
        },
        "C2_decision_dependencies": {
            "note": "E 日决策依赖; 含 v3 未扫 S1/S2 的 295 条补扫结果; "
                    "IRRELEVANT_NO_TRADE = 认证 NO_ACTION 且缺槽全在监控窗外 (无交易无依赖).",
            "n": len(C2_events),
            "distribution": dict(Counter(e["decision_class"] for e in C2_events)),
            "events": C2_events,
        },
        "C3_physical_gap_inventory": {
            "note": "物理 (code, date) 去重; C1 退出日与 C2 E 日分开来源标注, 不混为一个判定.",
            "n": len(C3_entries),
            "in_both": sum(1 for e in C3_entries if e["in_C1"] and e["in_C2"]),
            "entries": C3_entries,
        },
    })

    # ---- 预算自检 + 写盘 ----
    base_before = dir_size(RUN)
    docs = {"A": (OUT_A, A), "B": (OUT_B, B), "C": (OUT_C, C)}
    texts = {name: json.dumps(doc, ensure_ascii=False, indent=1)
             for name, (path, doc) in docs.items()}
    total_new = sum(len(t.encode("utf-8")) for t in texts.values())
    if total_new > CAP_NEW_BYTES:  # 超预算则整体降为紧凑格式重算
        texts = {name: json.dumps(doc, ensure_ascii=False, separators=(",", ":"))
                 for name, (path, doc) in docs.items()}
        total_new = sum(len(t.encode("utf-8")) for t in texts.values())
    if total_new > CAP_NEW_BYTES:
        raise RuntimeError(f"新增产物 {total_new} B 超本步 1 MiB 上限, 停机不写")
    if base_before + total_new > CAP_BASE_BYTES:
        raise RuntimeError(f"累计 {base_before}+{total_new} B 超 55 MiB 上限, 停机不写")

    for name, (path, doc) in docs.items():
        path.write_text(texts[name], encoding="utf-8")
        print(f"写入 {path.name}: {len(texts[name].encode('utf-8'))} B")
    base_after = dir_size(RUN)

    # ---- 报告 ----
    print("\n=== 验证 (用户口径) ===")
    for k, v in verification.items():
        if isinstance(v, dict):
            print(f"  {k}: expected={v.get('expected')} found={v.get('found')} match={v.get('match')}")
    print("\n=== 身份迁移 ===")
    for k, v in sorted(A["identity_distribution"].items()):
        print(f"  {k}: {v}")
    print("=== branch_at_decision ===")
    for k, v in sorted(A["branch_at_decision_distribution"].items()):
        print(f"  {k}: {v}")
    print("=== 认证 ===")
    for k, v in sorted(A["certification_distribution"].items()):
        print(f"  {k}: {v}")
    print("=== gap_status ===")
    for k, v in sorted(A["gap_status_distribution"].items()):
        print(f"  {k}: {v}")
    print(f"=== 补扫 === 重扫 {A['s1_s2_rescan']['n_rescanned']} 条, 新发现 S1/S2 {len(rescan_found_log)} 条")
    for r in rescan_found_log[:10]:
        print(f"  {r['code']}@{r['E']} mutual={r['mutual']} s1={r['s1_ts']} s2={r['s2_ts']} dip={r['dip_ts']} ({r['dip_view']})")
    print(f"=== C1 === {dict(c1_dist)}")
    print(f"=== C2 === {C['C2_decision_dependencies']['distribution']}")
    print(f"=== C3 === n={len(C3_entries)}, 双源重叠 {C['C3_physical_gap_inventory']['in_both']}")
    print(f"\n预算: 前 {base_before} B, 新增 {total_new} B, 后 {base_after} B "
          f"({base_after/1048576:.3f} MiB / 上限 55 MiB)")
    print(f"耗时 {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
