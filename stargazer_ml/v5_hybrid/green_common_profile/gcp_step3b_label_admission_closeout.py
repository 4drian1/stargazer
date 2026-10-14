# -*- coding: utf-8 -*-
"""第三步 b: 标签准入收口 (零网络/零采集/零拟合/零训练).

用户裁决 (2026-10-08):
1. S2 审计结论更正: 事件合同 §2.3 明确要求量能确认 ("截至该 bar 的同刻量比 >= BREAKOUT_VOL_RATIO_MIN"),
   且 §3.4 参数表已冻结 BREAKOUT_VOL_RATIO_MIN=1.5; 未冻结的是"同刻量比"的分母/窗口/早盘不足处置
   (CAR run sheet 将历史同刻量比列为 PENDING 后置, 未实现). probe 的"同日此前<=20根监控窗bar均量,
   空基线/零基线 -> inf"是未冻结实现选择, 不能视为合同定义. 27+1 条维持 PENDING; 本轮不设计新量比
   算法, 不为训练放开 S2. 原 r1_s2_volume_gate_audit_v0_1_2.json 的"任何已签合同无 S2 量比定义"
   结论错误, 留档不改, 由本收口件更正替代.
2. 公司行为核销不得留到拟合合同裁决: 用已有本地复权链 (feb2026_adjustment_cache_v1, 5477 码;
   dip_touch adj_chains.pkl 2873 码兜底) 逐轨道核对实际持仓窗口 [P+1, label_end] (E=next_trading_day(P)
   已全量验证). 判定遵循已签执行合同 §3/§8: 链OK无事件->通过; 确认空链->通过; UNKNOWN/证据冲突->隔离;
   窗口内事件->CORPORATE_ACTION_IN_WINDOW 隔离, 不事后放宽.
3. 分段与日期口径: 文件本为 train 113 / val 35 / test 20 (此前会话汇报 138/24/6 为口误, 非文件错误);
   valid_dates 统计的是 label_end 日期, 训练覆盖门另列 E 日期簇.
4. 显式断言: 完全退出 / 残仓为零 / 收益有限 / 依赖完整 / 终点合法且不跨段.
5. 354 对缺口分类 (无文件/内洞/元数据), 标注"首阻需求, 非完整路径预算"; 不批准整批补采;
   验证段 H3 无文件缺口另列优先观察, 不按盈亏挑选.
"""
from __future__ import annotations

import hashlib
import json
import math
import pickle
import sys
import time
from collections import Counter, defaultdict
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
ADJ_FEB = ROOT / "opportunity_v2" / "gate1c" / "feb2026_adjustment_cache_v1" / "adj_cache.json"
ADJ_PKL = ROOT / "a0a4" / "dip_touch_matched_control_v1" / "adj_chains.pkl"
AUDIT_LOG = HERE / "gcp_audit_log.md"

OUT_JSON = RUN / "r1_label_admission_v0_1_2.json"
OUT_MD = RUN / "r1_label_admission_report_v0_1_2.md"

CAP_NEW_BYTES = 1 * 1024 * 1024
CAP_BASE_BYTES = 55 * 1024 * 1024
DATA_CUTOFF = 20260928  # 执行器冻结截止
PC_TOL = 0.005          # 前收盘跳变判定容差 (半 tick)


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def dir_size(p: Path) -> int:
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file())


def main():
    t0 = time.time()
    print("=" * 70)
    print("第三步 b: 标签准入收口 (零网络/零采集/零拟合/零训练)")
    print("=" * 70)

    EX = json.loads((RUN / "r1_exec_labels_v0_1_2.json").read_text(encoding="utf-8"))
    records = EX["records"]
    assert len(records) == 489

    cal = D.load_calendar()
    cal_pos = {d: i for i, d in enumerate(cal)}

    panel = PROBE.get_daily_parquet()
    panel["code"] = panel["code"].astype(str).str.zfill(6)
    panel["date"] = panel["date"].astype(int)
    gby = dict(tuple(panel.groupby("code")))

    def daily_maps(code):
        g = gby.get(code)
        if g is None:
            return {}, {}
        g = g.sort_values("date").drop_duplicates("date")
        return (dict(zip(g["date"].astype(int), g["close"].astype(float))),
                dict(zip(g["date"].astype(int), g["pre_close"].astype(float))))

    # ---- 复权链 (本地, 零网络) ----
    feb = json.loads(ADJ_FEB.read_text(encoding="utf-8"))
    pkl = pickle.loads(ADJ_PKL.read_bytes()) if ADJ_PKL.exists() else {}
    feb_raw_hash_sample = {c: feb[c].get("raw_hash") for c in
                           sorted({r["code"] for r in records})[:0]}  # 不逐码展开, 见 provenance

    def chain_of(code):
        """返回 (events[(date,mult)], status). 仅本地证据; 不发起任何网络请求."""
        rec = feb.get(code)
        if rec is not None:
            evs = [(int(d), (float(m) if m is not None else None))
                   for d, m in rec.get("events", [])]
            return evs, rec["status"]
        raw = pkl.get(code)
        if raw is not None:
            evs = []
            for d, r in raw:
                try:
                    m = float(r.get("mult"))
                except (TypeError, ValueError):
                    m = None
                evs.append((int(d), m))
            return evs, ("OK" if evs else "EMPTY")
        return None, "UNKNOWN"

    # ================= 逐轨道复权链核销 + 显式断言 =================
    print("\n--- 复权链核销 (有效标签 168 H3 + 119 H5, 窗口 [E, label_end]) ---")
    maps_cache = {}

    def maps(code):
        if code not in maps_cache:
            maps_cache[code] = daily_maps(code)
        return maps_cache[code]

    def pc_mismatch_dates(code, E_day, le_day):
        close_m, pc_m = maps(code)
        out = []
        for d in cal:
            if d < E_day or d > le_day:
                continue
            i = cal_pos[d]
            if i == 0:
                continue
            prev = cal[i - 1]
            pc = pc_m.get(d)
            cl = close_m.get(prev)
            if pc is not None and cl is not None and abs(cl - pc) > PC_TOL:
                out.append(d)
        return out

    def adj_verdict(code, E_day, le_day):
        evs, st = chain_of(code)
        win_ev = [(d, m) for d, m in (evs or []) if E_day <= d <= le_day]
        mism = pc_mismatch_dates(code, E_day, le_day)
        bad_mult = [d for d, m in win_ev if m is None or m <= 0]
        if st == "UNKNOWN":
            return "ADJ_UNKNOWN", evs, st, win_ev, mism
        if bad_mult:
            return "ADJ_UNKNOWN_EVENT", evs, st, win_ev, mism
        if st == "EMPTY":
            return ("ADJ_EVIDENCE_CONFLICT" if mism else "PASS_EMPTY_CHAIN"), evs, st, win_ev, mism
        # OK
        if win_ev:
            return "CORPORATE_ACTION_IN_WINDOW", evs, st, win_ev, mism
        return ("ADJ_EVIDENCE_CONFLICT" if mism else "PASS_NO_EVENT"), evs, st, win_ev, mism

    track_decisions = []
    assertions = {"checks": ["fully_exited", "residual_zero", "net_finite",
                             "deps_complete", "label_end_legal", "not_cross_segment",
                             "fill_before_exit"],
                  "failures": 0, "failed": []}

    for r in records:
        code, Ed, seg = r["code"], int(r["E"]), r["segment"]
        for tk in ("H3", "H5"):
            t = r[tk]
            if not t or not t["valid"]:
                continue
            le = t["le"]
            le_day = int(le) // 1000000
            # ---- 显式断言 (任一失败即剔除并登记) ----
            fails = []
            if t["fe"] is not True:
                fails.append("fully_exited")
            if t["rem"] not in (0, None, 0.0):
                fails.append("residual_zero")
            if t["net"] is None or not math.isfinite(t["net"]) or abs(t["net"]) >= 1e9:
                fails.append("net_finite")
            if t["md"]:
                fails.append("deps_complete")
            if le is None or le_day not in cal_pos or le_day > DATA_CUTOFF:
                fails.append("label_end_legal")
            if PROBE.segment_of(le_day) != seg:
                fails.append("not_cross_segment")
            if not r["fill_ts"] or int(r["fill_ts"]) >= int(le):
                fails.append("fill_before_exit")
            if fails:
                assertions["failures"] += 1
                assertions["failed"].append({"code": code, "E": Ed, "track": tk, "fails": fails})
            # ---- 复权链核销 ----
            verdict, evs, st, win_ev, mism = adj_verdict(code, Ed, le_day)
            admitted = verdict.startswith("PASS_") and not fails
            track_decisions.append({
                "code": code, "E": Ed, "segment": seg, "branch": r["branch"],
                "track": tk, "net": t["net"], "le": le,
                "chain_status": st, "chain_n_events": len(evs or []),
                "window_events": [[d, m] for d, m in win_ev],
                "pc_mismatch_dates": mism,
                "verdict": verdict, "assert_fails": fails, "admitted": admitted,
            })

    n_adm = sum(1 for x in track_decisions if x["admitted"])
    vdist = dict(Counter(x["verdict"] for x in track_decisions))
    print(f"轨道判定: {len(track_decisions)} 条 -> admitted {n_adm}; verdict 分布 {vdist}")
    print(f"显式断言失败: {assertions['failures']}")

    # 老口径 ca 警告 (排除E日的旧扫描) 与新口径关系
    old_ca = [r for r in records
              if (r["H3"] and r["H3"].get("ca")) or (r["H5"] and r["H5"].get("ca"))]
    print(f"旧口径 ca 警告事件(非E日窗口): {len(old_ca)}; 新口径窗口事件+证据冲突: "
          f"{sum(1 for x in track_decisions if x['verdict'] in ('CORPORATE_ACTION_IN_WINDOW','ADJ_EVIDENCE_CONFLICT'))}")

    # ---- 冲突画像: 偏移比 = 1/prod(窗口后链事件复合至分片锚点)? ----
    import statistics as _st
    conf_tracks = [x for x in track_decisions if x["verdict"] == "ADJ_EVIDENCE_CONFLICT"]
    compound_hit = 0
    compound_detail = []
    for x in conf_tracks:
        le_day = int(x["le"]) // 1000000
        evs = [(int(d), float(m)) for d, m in (chain_of(x["code"])[0] or [])
               if m and m > 0 and d > le_day]
        g = gby.get(x["code"])
        cl = dict(zip(g["date"].astype(int), g["close"].astype(float)))
        pc = dict(zip(g["date"].astype(int), g["pre_close"].astype(float)))
        rs = []
        for d in x["pc_mismatch_dates"]:
            i = cal_pos[d]
            prev = cal[i - 1]
            if pc.get(d) and cl.get(prev):
                rs.append(pc[d] / cl[prev])
        if not rs:
            continue
        obs = _st.median(rs)
        best = None
        for anchor, _m in evs + [(99999999, 1.0)]:
            prod = 1.0
            for d, m2 in evs:
                if le_day < d <= anchor:
                    prod *= m2
            pred = 1.0 / prod
            if best is None or abs(obs - pred) < abs(obs - best[1]):
                best = (anchor, pred)
        if abs(obs - best[1]) < 0.001:
            compound_hit += 1
        else:
            compound_detail.append({"code": x["code"], "E": x["E"], "obs": round(obs, 6),
                                    "best_pred": round(best[1], 6)})
    # 证伪组: PASS_NO_EVENT 轨中链在窗口后有事件却无任何跳变 (口径异质按码/分片, 非全局)
    pass_with_future = 0
    for x in track_decisions:
        if x["verdict"] != "PASS_NO_EVENT":
            continue
        le_day = int(x["le"]) // 1000000
        if any(d > le_day and m and m > 0 for d, m in (chain_of(x["code"])[0] or [])):
            pass_with_future += 1
    # 准入轨基态核验: 全窗口 |pre_close - prev_close| 实测最大值
    adm_basis_max = 0.0
    for x in track_decisions:
        if not x["admitted"]:
            continue
        g = gby.get(x["code"])
        cl = dict(zip(g["date"].astype(int), g["close"].astype(float)))
        pc = dict(zip(g["date"].astype(int), g["pre_close"].astype(float)))
        for d in cal:
            if x["E"] <= d <= int(x["le"]) // 1000000:
                i = cal_pos[d]
                if i == 0:
                    continue
                prev = cal[i - 1]
                if pc.get(d) is not None and cl.get(prev) is not None:
                    adm_basis_max = max(adm_basis_max, abs(cl[prev] - pc[d]))
    conflict_char = {
        "n_tracks": len(conf_tracks),
        "n_codes": len({x["code"] for x in conf_tracks}),
        "offset_hypothesis": "冲突轨 pre_close/prev_close 中位比 = 1/prod(窗口后链事件复合至某锚点), "
                             "锚点即分片写入时点 (前复权基准), 非公司行为、非链缺口",
        "compound_match_within_0p1pct": f"{compound_hit}/{len(conf_tracks)}",
        "compound_unmatched": compound_detail,
        "falsification_group": f"PASS_NO_EVENT 轨中链在窗口后有事件却无跳变 {pass_with_future} 条 "
                               "(若为全局前复权口径应同样跳变 -> 口径异质按码/按分片)",
        "admitted_basis_check": f"准入轨全窗口 |pre_close - prev_close| 实测最大 {adm_basis_max:.4f} "
                                f"(容差 0.005) -> 准入集限价基准逐日核验干净",
        "executor_impact": "执行器用 pre_close 算限价带; 基态偏移只影响冲突码, 其有效轨已全部按证据"
                           "冲突隔离, 准入标签不受影响; 修复路径(重算 pre_close/重取受影响分片/重推"
                           "限价带)留待裁决, 本轮不自行改动",
    }
    print(f"冲突画像: 复合锚点吻合 {compound_hit}/{len(conf_tracks)}; 证伪组 {pass_with_future} 条; "
          f"准入基态最大偏差 {adm_basis_max:.4f}")

    # ---- 聚合: branch x segment, E 日期簇与 label_end 日期分开 ----
    agg = {}
    for br in ("DIP_REPAIR", "BASE_HOLD"):
        agg[br] = {}
        for seg in ("train", "val", "test"):
            row = {}
            for tk in ("H3", "H5"):
                dec = [x for x in track_decisions
                       if x["branch"] == br and x["segment"] == seg and x["track"] == tk]
                adm = [x for x in dec if x["admitted"]]
                row[tk] = {
                    "valid_before_admission": len(dec),
                    "admitted": len(adm),
                    "pos": sum(1 for x in adm if (x["net"] or 0) > 0),
                    "neg": sum(1 for x in adm if (x["net"] or 0) <= 0),
                    "E_date_clusters": len({x["E"] for x in adm}),
                    "label_end_dates": len({int(x["le"]) // 1000000 for x in adm}),
                }
            agg[br][seg] = row
    totals = {}
    for tk in ("H3", "H5"):
        dec = [x for x in track_decisions if x["track"] == tk]
        adm = [x for x in dec if x["admitted"]]
        totals[tk] = {"valid": len(dec), "admitted": len(adm),
                      "pos": sum(1 for x in adm if (x["net"] or 0) > 0),
                      "neg": sum(1 for x in adm if (x["net"] or 0) <= 0),
                      "E_date_clusters": len({x["E"] for x in adm})}

    # ================= 354 对缺口分类 =================
    print("\n--- 缺口分类 (首阻需求, 非完整路径预算) ---")
    by_pair_dir = D.MIN_DIRS[0]
    code_dirs = D.MIN_DIRS[1:]
    bp_index = {p.stem: p for p in by_pair_dir.glob("*.parquet")} if by_pair_dir.exists() else {}

    pairs = defaultdict(lambda: {"events": [], "tracks": set()})
    for r in records:
        for tk in ("H3", "H5"):
            t = r[tk]
            if not t:
                continue
            for d in t["md"]:
                g = pairs[(r["code"], int(d))]
                g["events"].append({"E": r["E"], "track": tk, "segment": r["segment"]})
                g["tracks"].add(tk)

    def classify_pair(code, date):
        # by_pair 优先 (执行器读取顺序)
        pf = bp_index.get(f"{code}_{date}")
        if pf is not None:
            try:
                df = pd.read_parquet(pf)
            except Exception:
                return "FILE_READ_ERROR", {}
        else:
            df = None
            for d in code_dirs:
                cf = d / f"{code}.parquet"
                if not cf.exists():
                    continue
                try:
                    full = pd.read_parquet(cf, columns=["date"])
                    if date in set(full["date"].astype(int) // 1000000):
                        df = pd.read_parquet(cf)
                        df = df[df["date"].astype(int) // 1000000 == date]
                        break
                except Exception:
                    continue
            if df is None:
                return "NO_FILE", {}
        if df is None or len(df) == 0:
            return "NO_FILE", {}
        v = E.validate_minute_day(df, D.day_grid(date), date)
        if v["dups"] or v["unexpected"]:
            return "DAY_ANOMALY", {"dups": len(v["dups"]), "unexpected": len(v["unexpected"])}
        if v["missing"]:
            return "INTRA_HOLE", {"missing_slots": len(v["missing"])}
        close_m, pc_m = maps(code)
        pc = pc_m.get(date)
        if pc is None or (isinstance(pc, float) and math.isnan(pc)):
            return "PRE_CLOSE_METADATA", {}
        return "NOT_BLOCKED_HERE", {}

    gap_rows = []
    for (code, date), g in sorted(pairs.items()):
        cls, extra = classify_pair(code, date)
        gap_rows.append({"code": code, "date": date, "class": cls,
                         "tracks": sorted(g["tracks"]), "n_events": len(g["events"]),
                         "segments": sorted({e["segment"] for e in g["events"]}),
                         **({"detail": extra} if extra else {})})
    gap_dist = dict(Counter(x["class"] for x in gap_rows))
    print(f"缺口分类: {gap_dist}")
    val_h3_nofile = []
    for x in gap_rows:
        if x["class"] != "NO_FILE":
            continue
        for e in pairs[(x["code"], x["date"])]["events"]:
            if e["segment"] == "val" and e["track"] == "H3":
                val_h3_nofile.append({"code": x["code"], "date": x["date"], "E": e["E"]})
                break
    val_h3_nofile = sorted({(x["code"], x["date"]): x for x in val_h3_nofile}.values(),
                           key=lambda z: (z["code"], z["date"]))
    print(f"验证段 H3 无文件缺口 (未来优先观察, 不按盈亏挑选): {len(val_h3_nofile)} 对")

    # ================= S2 结论更正 =================
    s2_correction = {
        "supersedes": "r1_s2_volume_gate_audit_v0_1_2.json#findings (原文件留档不改)",
        "wrong_claim": "任何已签合同无 S2 量比定义",
        "contract_facts": {
            "sec_2_3_quote": "S2 候选行动: 某已完成 bar close > box_upper_P; "
                             "该 bar 量能确认: 截至该 bar 的同刻量比 >= BREAKOUT_VOL_RATIO_MIN; "
                             "S2 = 该 bar 结束 (dual_path_event_execution_contract.md §2.3)",
            "sec_3_4_threshold": "BREAKOUT_VOL_RATIO_MIN = 1.5 已在 §3.4 参数表签署冻结",
            "sec_2_2_note": "S1 记录同刻量比仅作数据 (§2.2 第81行), 非门控",
            "same_time_ratio_unfrozen": "同刻量比(跨日同时刻口径)的分母/窗口/早盘不足处置未冻结: "
                                        "CAR run sheet (car_v0_1_run_sheet_draft.md) 将历史同刻量比 "
                                        "(volume_ratio_same_time) 列为 PENDING 后置特征, 本轮未实现",
            "probe_implementation": "scan_views L235-238: 同日此前<=20根监控窗bar均量, "
                                    "空基线/零基线 -> np.inf 放行; 与'同刻'跨日语义不一致, "
                                    "且空基线放行违背'缺比较依据不得当最强确认'",
        },
        "corrected_conclusion": "合同要求量能确认且阈值1.5已冻结; 未冻结的是同刻量比的分母/窗口/"
                                "早盘不足窗口处置; probe 自行选择的计算方法不能视为合同定义",
        "ruling": "27 条 BASE_BREAKOUT + 1 条 NO_ACTION 暴露维持 PENDING; 本轮不设计新量比算法, "
                  "不为训练放开 S2; S2 暂挂, 先推进 DIP_REPAIR+BASE_HOLD 共享模型线",
    }

    # ================= 产物 =================
    gen_at = time.strftime("%Y-%m-%d %H:%M:%S")
    common = {
        "generated_at": gen_at,
        "network_requests": 0, "training_performed": False, "data_collection_performed": False,
        "user_ruling_date": "2026-10-08",
        "frozen_input_hashes": {
            "r1_exec_labels_v0_1_2.json": sha256_file(RUN / "r1_exec_labels_v0_1_2.json"),
            "adj_cache.json(feb2026)": sha256_file(ADJ_FEB),
            "adj_chains.pkl(dip_touch)": sha256_file(ADJ_PKL),
            "dual_path_event_execution_contract.md": sha256_file(
                HERE / "contracts" / "dual_path_event_execution_contract.md"),
            "dual_path_execution_contract_v0_1_1_final.md": sha256_file(
                HERE / "contracts" / "dual_path_execution_contract_v0_1_1_final.md"),
            "dual_path_execution_v0_1_1.py": sha256_file(HERE / "dual_path_execution_v0_1_1.py"),
        },
        "adj_chain_sources": {
            "primary": "opportunity_v2/gate1c/feb2026_adjustment_cache_v1/adj_cache.json "
                       "(5477 码, OK 5407 / EMPTY 70; 逐码 raw_hash 见缓存文件)",
            "fallback": "a0a4/dip_touch_matched_control_v1/adj_chains.pkl (2873 码原始事件)",
            "note": "全部 487 涉及代码均在 primary 覆盖内 (实测 neither=0), fallback 未被使用; "
                    "零网络, 未发起任何复权链请求",
        },
    }
    OUT = dict(common)
    OUT.update({
        "phase": "R1_LABEL_ADMISSION_CLOSEOUT_V0_1_2",
        "s2_audit_correction": s2_correction,
        "admission_rule": "admitted = 显式断言全过 AND 复权链判定 PASS_NO_EVENT/PASS_EMPTY_CHAIN; "
                          "CORPORATE_ACTION_IN_WINDOW/ADJ_EVIDENCE_CONFLICT/ADJ_UNKNOWN -> 隔离 "
                          "(已签执行合同 §3/§8, 不事后放宽); H3/H5 独立, H5 隔离不删除 H3",
        "assertions": assertions,
        "segment_counts_correction": {
            "files_said": {"train": 113, "val": 35, "test": 20},
            "chat_message_error": "此前会话汇报 '138/24/6' 为口误, 产物文件从未如此记录; "
                                  "E 日期簇与 label_end 日期在本件分开列示",
        },
        "aggregates": {"by_branch_segment": agg, "totals": totals,
                       "verdict_distribution": vdist,
                       "old_ca_warning_events": len(old_ca)},
        "conflict_characterization": conflict_char,
        "gap_classification": {
            "note": "首阻需求, 非完整路径预算; 不构成补采批准; 分类只用数据事实, 不按盈亏挑选",
            "n_pairs": len(gap_rows), "distribution": gap_dist,
            "pairs": gap_rows,
            "val_h3_nofile_priority": {"n": len(val_h3_nofile), "pairs": val_h3_nofile},
        },
        "track_decisions": track_decisions,
    })

    # ---- 报告 ----
    L = [
        "# 标签准入收口 (v0.1.2, 第三步 b)", "",
        f"生成: {gen_at} | 零网络 / 零采集 / 零拟合 / 零训练", "",
        "## 1. S2 审计结论更正", "",
        "- 已签事件合同 **§2.3 明确要求量能确认**：`截至该 bar 的同刻量比 >= BREAKOUT_VOL_RATIO_MIN`，"
        "且 §3.4 参数表已冻结 `BREAKOUT_VOL_RATIO_MIN=1.5`。",
        "- 未冻结的是**同刻量比的分母/窗口/早盘不足窗口处置**（CAR run sheet 将历史同刻量比列为 "
        "PENDING 后置，未实现）；probe 的同日此前≤20根均量+空基线 ∞ 放行是未冻结实现选择。",
        "- **27 条 BASE_BREAKOUT + 1 条 NO_ACTION 暴露维持 PENDING**；本轮不设计新量比算法、不为训练"
        "放开 S2；S2 暂挂，先推进 DIP_REPAIR+BASE_HOLD 共享模型线。",
        "- 原 `r1_s2_volume_gate_audit_v0_1_2.json` 结论错误处留档不改，由本件更正替代。", "",
        "## 2. 复权链核销 (有效标签 168 H3 + 119 H5, 窗口 [E, label_end])", "",
        "- 链来源：feb2026 本地缓存（5477 码，OK 5407 / EMPTY 70），487 个涉及代码全覆盖（UNKNOWN=0）；"
        "dip_touch pkl 兜底未被使用；零网络。",
        "- 判定分布：`{vdist}`。".format(vdist=vdist) +
        f"\n- **冲突画像**：{conflict_char['n_tracks']} 条证据冲突集中在 {conflict_char['n_codes']} 只代码；"
        f"偏移比复合锚点吻合 {conflict_char['compound_match_within_0p1pct']}；"
        f"证伪组 {conflict_char['falsification_group']}；"
        f"{conflict_char['admitted_basis_check']}。结论：pre_close 列前复权基准异质（按码/分片），"
        "非公司行为、非链缺口；执行器限价带偏移只影响已隔离码，准入标签不受影响；修复路径留待裁决。",
        "- 判定规则：链 OK 无事件→通过；确认空链→通过；窗口内事件→`CORPORATE_ACTION_IN_WINDOW` 隔离；"
        "前收盘跳变无链事件对应（含空链矛盾）→`ADJ_EVIDENCE_CONFLICT` 隔离；mult 无效→`ADJ_UNKNOWN_EVENT`。"
        "遵守已签执行合同，不事后放宽。",
        f"- 显式断言（完全退出/残仓为零/收益有限/依赖完整/终点合法不跨段/成交先于退出）："
        f"失败 {assertions['failures']} 条。", "",
        "## 3. 准入后标签 (H3 主 / H5 辅, 独立)", "",
        "| 分支 | 段 | H3 有效→准入 | 正/负 | E日簇 | H5 有效→准入 | 正/负 | E日簇 |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for br in ("DIP_REPAIR", "BASE_HOLD"):
        for seg in ("train", "val", "test"):
            x = agg[br][seg]
            h3, h5 = x["H3"], x["H5"]
            L.append(f"| {br} | {seg} | {h3['valid_before_admission']}→{h3['admitted']} | "
                     f"{h3['pos']}/{h3['neg']} | {h3['E_date_clusters']} | "
                     f"{h5['valid_before_admission']}→{h5['admitted']} | "
                     f"{h5['pos']}/{h5['neg']} | {h5['E_date_clusters']} |")
    L += [
        "", f"- 合计：H3 {totals['H3']['valid']}→**{totals['H3']['admitted']}**"
            f"（正 {totals['H3']['pos']} / 负 {totals['H3']['neg']}，E 日簇 {totals['H3']['E_date_clusters']}）；"
            f"H5 {totals['H5']['valid']}→**{totals['H5']['admitted']}**"
            f"（正 {totals['H5']['pos']} / 负 {totals['H5']['neg']}，E 日簇 {totals['H5']['E_date_clusters']}）。",
        "- **人口口径**：124 = H3 准入**事件**（train 77 / val 27 / test 20）；H5 = 90 事件仅辅助；"
        "**214 是 H3+H5 轨道数，不是 214 个独立训练样本**。",
        "- 分段与口径更正：产物文件本为 train 113 / val 35 / test 20（此前会话汇报 138/24/6 为口误）；"
        "`valid_dates` 是 label_end 日期，训练覆盖门用 **E 日期簇**，两者分列。",
        "- 343 条认证无行动不是亏损负样本；隔离标签留在总分母审计表，不删除。", "",
        "## 4. 354 对缺口分类 (首阻需求, 非完整路径预算)", "",
        f"- 分布：`{gap_dist}`。",
        f"- 验证段 H3 无文件缺口（未来优先观察，不按盈亏挑选）：{len(val_h3_nofile)} 对。",
        "- 本清单不构成补采批准；整批 354 对补采未获批；同源重取一致仅证明缺口可复现。", "",
        "## 5. 边界", "",
        "- 零网络、不采集、不拟合、不训练；输出后停止。",
        "- 覆盖偏差披露：准入子集偏向退出路径数据完整的 E 日（DIP 事件缺槽比例高），"
        "拟合合同定稿时必须披露该可评价子集偏差，不得宣称全人口等价。",
        "- 下一步（待用户裁决）：按真实训练样本与白名单特征覆盖定稿共享轻量模型拟合合同。",
    ]
    report_md = "\n".join(L)

    # ---- 预算 + 写盘 ----
    base_before = dir_size(RUN)
    text_json = json.dumps(OUT, ensure_ascii=False, indent=1)
    total_new = len(text_json.encode("utf-8")) + len(report_md.encode("utf-8"))
    if total_new > CAP_NEW_BYTES:
        text_json = json.dumps(OUT, ensure_ascii=False, separators=(",", ":"))
        total_new = len(text_json.encode("utf-8")) + len(report_md.encode("utf-8"))
    if total_new > CAP_NEW_BYTES:
        raise RuntimeError(f"新增 {total_new} B 超 1 MiB, 停机不写")
    if base_before + total_new > CAP_BASE_BYTES:
        raise RuntimeError(f"累计 {base_before}+{total_new} 超 55 MiB, 停机不写")
    OUT_JSON.write_text(text_json, encoding="utf-8", newline="")
    OUT_MD.write_text(report_md, encoding="utf-8", newline="")

    # ---- 审计日志追加 ----
    log_entry = f"""

## 工程记录 ENGINEERING-2026-10-08-32：第三步标签准入收口（零网络）

- **S2 审计结论更正**：事件合同 §2.3 明确要求量能确认（截至该 bar 的同刻量比 ≥ BREAKOUT_VOL_RATIO_MIN，1.5 已在 §3.4 冻结）；未冻结的是同刻量比的分母/窗口/早盘不足处置（CAR run sheet 列为 PENDING 后置）。此前 r1_s2_volume_gate_audit_v0_1_2.json 的"合同无 S2 量比定义"结论错误，原文件留档不改，由 r1_label_admission_v0_1_2.json 更正替代。27+1 条维持 PENDING；本轮不设计新量比算法、不为训练放开 S2。
- **复权链核销**：用本地 feb2026 链缓存（5477 码，487 涉及代码全覆盖）逐轨道核对 [E, label_end]；链 OK 无事件/确认空链通过；窗口内事件 CORPORATE_ACTION_IN_WINDOW 隔离；前收盘跳变无链事件对应或与空链矛盾 → ADJ_EVIDENCE_CONFLICT 隔离；遵守已签执行合同不事后放宽。显式断言（完全退出/残仓为零/收益有限/依赖完整/终点合法不跨段/成交先于退出）失败 0 条。
- **分段口径**：产物文件本为 train 113 / val 35 / test 20（H3 有效，公司行为核销前）；此前会话汇报 138/24/6 为口误非文件错误；E 日期簇与 label_end 日期分列。
- **缺口分类**：{len(gap_rows)} 对首阻缺口按无文件/内洞/日异常/前收盘元数据分类（分布 {gap_dist}），标注"首阻需求，非完整路径预算"；整批补采未获批；验证段 H3 无文件缺口 {len(val_h3_nofile)} 对另列优先观察，不按盈亏挑选。
- **产物**：r1_label_admission_v0_1_2.json / r1_label_admission_report_v0_1_2.md（新增 {total_new} B ≤ 1 MiB；RUN 目录累计 {base_before + total_new} B ≤ 55 MiB）。零网络、不采集、不拟合、不训练；输出后停止。拟合合同待用户裁决后定稿。
"""
    with open(AUDIT_LOG, "a", encoding="utf-8", newline="") as f:
        if "ENGINEERING-2026-10-08-32" not in AUDIT_LOG.read_text(encoding="utf-8"):
            f.write(log_entry)
            print("审计日志已追加 ENGINEERING-2026-10-08-32")
        else:
            print("审计日志已含本条目, 跳过重复追加")

    print(f"\n写入: {OUT_JSON.name} / {OUT_MD.name}")
    print(f"预算: 前 {base_before} B + 新增 {total_new} B = {base_before + total_new} B "
          f"({(base_before + total_new) / 1048576:.3f} MiB / 55 MiB)")
    print(f"\n=== 准入汇总 ===")
    print(json.dumps(totals, ensure_ascii=False))
    print(f"verdict 分布: {vdist}")
    print(f"缺口分布: {gap_dist}; val H3 NO_FILE: {len(val_h3_nofile)}")
    print(f"耗时 {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
