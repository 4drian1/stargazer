"""R0-V1 逐候选外部证据核验 v1（只读、零网络、可复跑）。

对 DUAL_PATH_1013 人口逐条产出 PASS / 隔离原因表：
  1. 独立日历比对：akshare 自带官方交易日历 vs 日线推导日历（差集双向）。
  2. 逐候选：P 绑定（fetch list）、资格来源（pool_a/pool_b 冻结记录）、规则版本（fetch list sha256）、
     决策可用时刻（无证据 -> UNKNOWN）、W=P 后 5 交易日映射、复权状态与 [P,W末日] 覆盖、
     P/W 日线行在场与有效性。
  3. FAIL 日志对账：按码对照最终缓存，区分历史失败与当前失败（不与 OK/EMPTY 重复计数）。
  4. 价格尺度跨日对拍：除权日 close[d-1]/pre_close[d] 对照复权乘子 mult（RAW vs qfq 判别）。
"""
from __future__ import annotations
import glob, os, sys, json, hashlib, time
from collections import Counter
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
BASE = HERE.parent
EV_JSON = BASE / "green_common_profile/gcp_round_b_small_sample_v1/r1_branch_at_decision_v0_1_2.json"
FETCH_LIST = BASE / "green_common_profile/contracts/dual_path_e_day_fetch_list.txt"
POOL_DIR = BASE / "green_common_profile/round_a_v0_1_impl_fix"
CACHE = BASE / "opportunity_v2/gate1c/feb2026_adjustment_cache_v1/adj_cache.json"
FAILLOG = CACHE.with_suffix(CACHE.suffix + ".failures.jsonl")
DAILY = r"D:\a股数据\data_daily_parts"
AK_CAL = Path(r"D:\stargazer\_dev_venv\Lib\site-packages\akshare\file_fold\calendar.json")
# AkShare 包内日历工件登记（独立对照用，非交易所官方认证）
AK_CAL_VERSION = "akshare 1.18.94"
AK_CAL_SHA256 = "f1f1ce33c5cceb5c530c3271fc951405cb5624d2f30becec85dc7a85fd47e656"
# 候选生成算法脚本（算法版本，区别于输入版本 fetch list 的哈希）
GEN_SCRIPT = BASE / "green_common_profile/gcp_dual_path_e_day_task_order.py"
GEN_SHA256 = "4b879aad31e18b8e9cd2386bcf00b8d12d044080e774a76ad016181b66fafeb6"

import pandas as pd  # noqa: E402
import pyarrow.dataset as ds  # noqa: E402

FETCH_EPOCH_HINT = None  # 动态取 max(fetched_at)；仅作信息，不用于覆盖判定


def sha256_file(p) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def load_calendar():
    files = sorted(glob.glob(os.path.join(DAILY, "part_*.parquet")))
    dates = ds.dataset(files, format="parquet").to_table(columns=["date"]).column("date").to_pylist()
    return sorted({int(d) for d in dates})


def calendar_diff(cal):
    ak = [int(d) for d in json.loads(AK_CAL.read_text(encoding="utf-8"))]
    lo, hi = cal[0], cal[-1]
    ak_in = {d for d in ak if lo <= d <= hi}
    cal_set = set(cal)
    # 独立对照来源登记（包内工件，非交易所官方认证；据仓库本地源码注释相关工具来源为新浪财经）
    return {"independent_source": "akshare 包内 file_fold/calendar.json",
            "independent_version": AK_CAL_VERSION,
            "independent_sha256": AK_CAL_SHA256,
            "source_note": "非交易所官方认证；据本地源码注释相关工具来源为新浪财经，随 akshare 包版本分发；"
                          "与其零差集仅证明日线推导日历在本区间与之逐日一致，不等同交易所官方背书",
            "akshare_calendar_n": len(ak), "akshare_range": [ak[0], ak[-1]],
            "compare_range": [lo, hi],
            "independent_trading_days_in_range": len(ak_in),
            "data_calendar_days": len(cal_set),
            "missing_in_data(independent_day_no_bars)": sorted(ak_in - cal_set),
            "extra_in_data(bars_on_non_trading_day)": sorted(cal_set - ak_in)}


def classify_scale_per_code(cache, by_code, tol=0.02, disc_thr=0.05):
    """逐码判别日线 OHLC 尺度（RAW / QFQ / AMBIGUOUS / UNVERIFIED）。

    仅用判别力足够的公司行为事件（|mult-1|>=disc_thr）对拍：
    ratio = close[d-1] / pre_close[d]
      - RAW_LIKE : |ratio/mult - 1| <= tol  （OHLC 为原始价，pre_close 携带除权参考）
      - QFQ_LIKE : |ratio - 1| <= tol        （OHLC 已前复权）
      - 两者同判  : AMBIGUOUS（小乘子容差内同判，不归入 RAW）
    无判别事件 / 状态非 OK / 事件日无 bar -> UNVERIFIED（不得套复权公式）。
    """
    out = {}
    for c, g in by_code.items():
        rec = cache.get(c, {})
        if rec.get("status") != "OK":
            out[c] = ("UNVERIFIED", f"status={rec.get('status', 'MISSING')}")
            continue
        events = rec.get("events", [])
        disc = [(int(d), m) for d, m in events if m and abs(m - 1) >= disc_thr]
        if not disc:
            out[c] = ("UNVERIFIED", "no_discriminating_event")
            continue
        votes, detail = [], []
        for d, m in disc:
            prev = g[g["date"] < d]
            cur = g[g["date"] == d]
            if prev.empty or cur.empty:
                continue
            close_prev = float(prev["close"].iloc[-1])
            pre = float(cur["pre_close"].iloc[0])
            if not pre or pre <= 0:
                continue
            ratio = close_prev / pre
            raw_like = abs(ratio / m - 1) <= tol
            qfq_like = abs(ratio - 1) <= tol
            cls = ("AMBIG" if raw_like and qfq_like else
                   "RAW_LIKE" if raw_like else
                   "QFQ_LIKE" if qfq_like else "OTHER")
            votes.append(cls)
            detail.append((d, round(m, 4), round(ratio, 4), cls))
        if not votes:
            out[c] = ("UNVERIFIED", "event_dates_no_bars")
            continue
        cc = Counter(votes)
        if cc.get("RAW_LIKE", 0) > 0 and cc.get("QFQ_LIKE", 0) == 0 and cc.get("OTHER", 0) == 0:
            out[c] = ("RAW", f"votes={dict(cc)}; e.g.{detail[:2]}")
        elif cc.get("QFQ_LIKE", 0) > 0 and cc.get("RAW_LIKE", 0) == 0 and cc.get("OTHER", 0) == 0:
            out[c] = ("QFQ", f"votes={dict(cc)}; e.g.{detail[:2]}")
        else:
            out[c] = ("AMBIGUOUS", f"votes={dict(cc)}; e.g.{detail[:2]}")
    return out


def classify_window(g, win_events, tol=0.02, disc_thr=0.05):
    """窗口级跨日比诊断（非门禁）：仅用目标窗口 [P, W_end] 内的判别力足够事件（|mult-1|>=disc_thr）。

    注意（用户 2026-10-13 纠正）：窗口级尺度**不要求**每 5 日窗口都出现可用于辨别尺度的公司行为。
    跨日比法仅是历史验证手段之一；窗口内无判别事件时它**无法判定**尺度（非事件日 RAW/QFQ 不可区分），
    这是**方法局限，非数据不可用**。正确的窗口级尺度证据来自：
      (1) 明确源接口价格口径 + 每行来源/转换溯源；
      (2) 与独立、口径明确的价格源比对 P 日参考价与窗口内必要价格；
      (3) 公司行为案例用于验证转换方法，不要求每条候选窗口都有案例。
    因此本函数输出为 METHOD_CANNOT_DETERMINE（方法无法判定），**不得解读为数据不可用**，
    也**不**作为标签门禁；真实窗口级尺度改由 window_scale_bounded_validation.py 建立。
    """
    if g is None or getattr(g, "empty", True) or not win_events:
        return "METHOD_CANNOT_DETERMINE", "no_in_window_discriminating_event"
    votes, detail = [], []
    for d, m in win_events:
        prev = g[g["date"] < d]
        cur = g[g["date"] == d]
        if prev.empty or cur.empty:
            continue
        close_prev = float(prev["close"].iloc[-1])
        pre = float(cur["pre_close"].iloc[0])
        if not pre or pre <= 0:
            continue
        ratio = close_prev / pre
        raw_like = abs(ratio / m - 1) <= tol
        qfq_like = abs(ratio - 1) <= tol
        cls = ("AMBIG" if raw_like and qfq_like else
               "RAW_LIKE" if raw_like else
               "QFQ_LIKE" if qfq_like else "OTHER")
        votes.append(cls)
        detail.append((d, round(m, 4), round(ratio, 4), cls))
    if not votes:
        return "METHOD_CANNOT_DETERMINE", "in_window_event_days_no_bars"
    cc = Counter(votes)
    if cc.get("RAW_LIKE", 0) > 0 and cc.get("QFQ_LIKE", 0) == 0 and cc.get("OTHER", 0) == 0:
        return "RAW", f"votes={dict(cc)}; e.g.{detail[:2]}"
    if cc.get("QFQ_LIKE", 0) > 0 and cc.get("RAW_LIKE", 0) == 0 and cc.get("OTHER", 0) == 0:
        return "QFQ", f"votes={dict(cc)}; e.g.{detail[:2]}"
    return "AMBIGUOUS", f"votes={dict(cc)}; e.g.{detail[:2]}"


def coverage_chain_ok(cache, c):
    """复权链是否可在 [P,W末日] 构建（状态层面）；完整性须独立证据，不在此判。"""
    st = cache.get(c, {}).get("status")
    if st is None or st == "MISSING":
        return False, "ADJ_MISSING(链不可用)"
    # OK/EMPTY：链可构建；但源是否返回完整公司行为历史不可本地证
    return True, f"链可构建 status={st}（完整性未独立核销）"


def main():
    t0 = time.time()
    # ---------- 日历 ----------
    cal = load_calendar()
    cal_set = set(cal)
    cal_idx = {d: i for i, d in enumerate(cal)}
    cal_diff = calendar_diff(cal)

    # ---------- 候选：1,013 events + fetch list P 绑定 ----------
    events = json.loads(EV_JSON.read_text(encoding="utf-8"))["events"]
    fl_rows = [l.split(",") for l in FETCH_LIST.read_text(encoding="utf-8").strip().splitlines()]
    fl_map = {(r[0], int(r[1])): (int(r[2]), r[3], r[4]) for r in fl_rows}
    fetch_sha = sha256_file(FETCH_LIST)

    # ---------- 冻结资格池 ----------
    pool_a = pd.read_parquet(POOL_DIR / "pool_a_sdp_members.parquet", columns=["date", "code"])
    pool_b = pd.read_parquet(POOL_DIR / "pool_b_lvb_members.parquet", columns=["date", "code"])
    elig_a = {(str(c).zfill(6), int(d)) for c, d in zip(pool_a["code"], pool_a["date"])}
    elig_b = {(str(c).zfill(6), int(d)) for c, d in zip(pool_b["code"], pool_b["date"])}

    # ---------- 复权缓存（最终状态为准；FAIL 仅留痕不落盘） ----------
    cache = json.loads(CACHE.read_text(encoding="utf-8"))
    fats = [v.get("fetched_at", 0) for v in cache.values()]
    fetch_epoch = int(time.strftime("%Y%m%d", time.localtime(max(fats)))) if fats else 0
    fails = []
    if FAILLOG.exists():
        for line in FAILLOG.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                try:
                    fails.append(json.loads(line))
                except Exception:
                    pass
    fail_codes = {r.get("code") for r in fails if r.get("code")}
    fail_hist = sorted(c for c in fail_codes if c in cache)       # 历史失败，最终已 OK/EMPTY
    fail_curr = sorted(c for c in fail_codes if c not in cache)   # 当前失败（未落盘）

    # ---------- 日线面板切片（全部候选码，用于逐码尺度判别） ----------
    cand_codes = sorted({str(e["code"]).zfill(6) for e in events})
    # 尺度对拍样本（报告补充证据）：候选码中 2025+ 的 OK 事件，取前 40
    scale_cases = []
    for c in cand_codes:
        rec = cache.get(c)
        if rec and rec.get("status") == "OK":
            for d, m in rec.get("events", []):
                d, m = int(d), m
                if 20250101 <= d <= fetch_epoch and m and m > 0 and abs(m - 1) > 1e-6:
                    scale_cases.append((c, d, m))
    scale_cases = scale_cases[:40]
    need_codes = cand_codes
    files = sorted(glob.glob(os.path.join(DAILY, "part_*.parquet")))
    panel_tbl = ds.dataset(files, format="parquet") \
        .to_table(filter=ds.field("code").isin(need_codes),
                  columns=["code", "date", "low", "close", "pre_close", "pct_chg"])
    panel = panel_tbl.to_pandas()
    panel["code"] = panel["code"].astype(str).str.zfill(6)
    panel["date"] = panel["date"].astype(int)
    row_idx = {(c, d): r for (c, d), r in
               panel.set_index(["code", "date"])[["low", "close", "pre_close", "pct_chg"]].iterrows()}
    by_code = {c: g.sort_values("date") for c, g in panel.groupby("code")}
    # 逐码尺度判别（全候选码，非样本代替）
    scale_map = classify_scale_per_code(cache, by_code)

    # ---------- 逐候选核验 ----------
    rows = []
    for e in events:
        c = str(e["code"]).zfill(6)
        E = int(e["E"])
        rec = {"code": c, "E": E, "branch": e.get("branch_at_decision"),
               "decision_available_ts": "UNKNOWN", "time_evidence": "UNKNOWN",
               "window_scale_crossday_diag": "METHOD_CANNOT_DETERMINE",
               "window_scale_crossday_note": "no_window",
               "window_scale_evidence_method": "PENDING: provenance-tracing + independent-price-comparison (bounded validation, not yet run)"}
        reasons = []  # 仅结构性缺数据 -> 隔离（保留完整人口，软缺口不隔离）
        # P 绑定
        if (c, E) not in fl_map:
            reasons.append("P_BINDING_MISSING")
            rec.update({"P": "", "setup": "", "fetch_status": ""})
            rec.update({"input_version": "", "algorithm_version": GEN_SHA256[:16],
                        "membership_binding": "UNBOUND", "eligibility_known_at_time": "UNVERIFIED",
                        "eligibility_ref": "", "adj_status": "MISSING", "W": "", "W_end": "",
                        "coverage_verified": "UNVERIFIED", "coverage_chain_ok": False,
                        "scale_class": "UNVERIFIED", "scale_evidence": "no_P_binding",
                        "scale_verified": "UNVERIFIED"})
        else:
            P, setup, fstat = fl_map[(c, E)]
            # 版本分列：input_version = 输入快照(fetch list 哈希)；algorithm_version = 候选生成算法脚本哈希
            rec.update({"P": P, "setup": setup, "fetch_status": fstat,
                        "input_version": fetch_sha[:16], "algorithm_version": GEN_SHA256[:16]})
            # 资格：成员命中（历史成员关系，可证） vs 当时已知（不可由成员命中独自证明）
            if setup == "A_SDP" and (c, P) in elig_a:
                rec["eligibility_ref"] = f"pool_a_sdp_members.parquet#(code={c},P={P})"
                rec["membership_binding"] = "BOUND"
            elif setup == "B_LVB" and (c, P) in elig_b:
                rec["eligibility_ref"] = f"pool_b_lvb_members.parquet#(code={c},P={P})"
                rec["membership_binding"] = "BOUND"
            elif (c, P) in elig_a or (c, P) in elig_b:
                reasons.append("ELIGIBILITY_SETUP_MISMATCH")
                rec["eligibility_ref"] = "setup_mismatch"
                rec["membership_binding"] = "SETUP_MISMATCH"
            else:
                reasons.append("ELIGIBILITY_UNBOUND")
                rec["eligibility_ref"] = ""
                rec["membership_binding"] = "UNBOUND"
            rec["eligibility_known_at_time"] = "UNVERIFIED"  # 成员命中≠当时已知
            # 日历 W = P 后 5 交易日
            if P not in cal_set:
                reasons.append("P_NOT_IN_CALENDAR")
                rec.update({"W": "", "W_end": ""})
            else:
                i = cal_idx[P]
                W = cal[i + 1:i + 6]
                rec["W"] = "|".join(str(x) for x in W)
                rec["W_end"] = W[-1] if len(W) == 5 else ""
                if len(W) < 5:
                    reasons.append("CALENDAR_TAIL")
                if W and W[0] != E:
                    reasons.append("CALENDAR_W0_NE_E")
                # 复权状态（最终缓存，可证）；[P,W] 覆盖需独立证据 -> 不据此判 PASS
                st = cache.get(c, {}).get("status", "MISSING")
                rec["adj_status"] = st
                if st == "MISSING":
                    reasons.append("ADJ_MISSING")
                rec["coverage_verified"] = "UNVERIFIED"  # 完整性须独立证据，不据抓取时间判
                cok, cnote = coverage_chain_ok(cache, c)
                rec["coverage_chain_ok"] = cok
                if len(W) == 5 and W[-1] > fetch_epoch:
                    rec["coverage_note"] = (cnote + f"；W_end {W[-1]} 超出抓取时点 {fetch_epoch}"
                                            f"（刷新只补状态/内容，不能证完整覆盖）")
                else:
                    rec["coverage_note"] = cnote + f"；W_end {rec['W_end']}"
                # 日线数据在场
                if len(W) == 5:
                    missing = [d for d in [P] + W if (c, d) not in row_idx]
                    badlow = [d for d in [P] + W
                              if (c, d) in row_idx and not (row_idx[(c, d)]["low"] > 0)]
                    if missing:
                        reasons.append(f"DATA_MISSING({','.join(map(str, missing))})")
                    if badlow:
                        reasons.append(f"DATA_INVALID_LOW({','.join(map(str, badlow))})")
                    # 窗口级跨日比诊断（非门禁）：仅用 [P, W_end] 内的判别事件；方法无法判定≠数据不可用
                    win_events = [(int(d), m) for d, m in cache.get(c, {}).get("events", [])
                                  if m and abs(m - 1) >= 0.05 and P <= int(d) <= W[-1]]
                    wc, wev = classify_window(by_code.get(c), win_events)
                    rec["window_scale_crossday_diag"] = wc
                    rec["window_scale_crossday_note"] = wev
            # 逐码尺度判别（全候选码；RAW 才可套复权公式，否则标签 NULL）
            sc, sev = scale_map.get(c, ("UNVERIFIED", "not_in_panel"))
            rec["scale_class"] = sc
            rec["scale_evidence"] = sev
            rec["scale_verified"] = "RAW_VERIFIED" if sc == "RAW" else "UNVERIFIED"
        rec["isolation_reasons"] = ";".join(reasons)
        rec["structural_verdict"] = "PASS" if not reasons else "ISOLATED"
        rows.append(rec)
    df = pd.DataFrame(rows)

    # ---------- 标签准入判定（工程标签） ----------
    # 三层次不可混淆（用户 2026-10-13 纠正）：
    #   RAW              : 价格尺度有支持证据（逐码，采样事件级）
    #   coverage_chain_ok: 现有链能算因子
    #   coverage_verified: 能证明目标窗口无遗漏公司行为  ← 第二项不能替代第三项
    # 仅三者皆满足才 ENGINEERING_OK；RAW+链可用但覆盖未核销 -> SCALE_SUPPORTED_COVERAGE_PENDING
    # （正式 MAE/Y_tail 一律 NULL，待覆盖核销）；其余 LABEL_NULL。training_eligible 全 False。
    # 注：单码某次大比例除权对拍支持 RAW，不自动证明该码所有分片/窗口同尺度（SDK/HTTP 双路径），
    #     故 SCALE_SUPPORTED 仍是"受支持"而非"全量核销"。
    def label_elig(r):
        # 注：窗口级尺度由来源溯源+独立价格比对建立（window_scale_bounded_validation.py），
        # 跨日比法仅历史验证手段且方法无法判定时不得误判数据不可用，故门禁仍用码级 scale_class。
        if r["structural_verdict"] == "ISOLATED":
            return "LABEL_NULL", "structural_isolated"
        if r["coverage_verified"] == "VERIFIED" and r["scale_class"] == "RAW" and r["coverage_chain_ok"]:
            return "ENGINEERING_OK", ""
        if r["scale_class"] == "RAW" and r["coverage_chain_ok"]:
            return "SCALE_SUPPORTED_COVERAGE_PENDING", "scale_raw_supported_but_coverage_unverified"
        if r["scale_class"] != "RAW":
            return "LABEL_NULL", f"scale_unverified({r['scale_class']})"
        return "LABEL_NULL", "coverage_chain_missing"
    le = df.apply(label_elig, axis=1, result_type="expand")
    df["label_eligibility"], df["label_reason"] = le[0], le[1]
    df["training_eligible"] = False  # 历史到达时刻 UNKNOWN，全部不进拟合
    # 码级 RAW：分列"候选记录数"与"唯一股票数"（保留 147 条历史支持证据，不因新方法不适用而撤销）
    raw_mask = df["scale_class"] == "RAW"
    n_raw_records = int(raw_mask.sum())
    n_raw_unique = int(df.loc[raw_mask, "code"].nunique())

    # ---------- 价格尺度跨日对拍（报告补充证据，样本） ----------
    scale_results = []
    for c, d, m in scale_cases:
        g = by_code.get(c)
        if g is None:
            continue
        prev = g[g["date"] < d]
        cur = g[g["date"] == d]
        if prev.empty or cur.empty:
            continue
        close_prev = float(prev["close"].iloc[-1])
        pre = float(cur["pre_close"].iloc[0])
        close_d = float(cur["close"].iloc[0])
        pct = cur["pct_chg"].iloc[0]
        if not pre or pre <= 0:
            continue
        ratio = close_prev / pre
        raw_like = abs(ratio / m - 1) <= 0.02
        qfq_like = abs(ratio - 1) <= 0.02
        pct_chk = (pct is not None and abs((close_d / pre - 1) * 100 - pct) <= 0.5)
        # 小乘子案例 RAW 与 QFQ 在 2% 容差内同时成立 -> 歧义，不应归入 RAW
        if raw_like and qfq_like:
            cls = "AMBIGUOUS"
        elif raw_like:
            cls = "RAW_LIKE"
        elif qfq_like:
            cls = "QFQ_LIKE"
        else:
            cls = "OTHER"
        scale_results.append({"code": c, "ex_date": d, "mult": round(m, 6),
                              "close_prev": close_prev, "pre_close": pre,
                              "ratio": round(ratio, 6), "classify": cls,
                              "pct_chg_consistent": bool(pct_chk)})
    scale_summary = {
        "n_cases": len(scale_results),
        "classify_counts": dict(Counter(r["classify"] for r in scale_results)),
        "pct_chg_consistent_n": sum(1 for r in scale_results if r["pct_chg_consistent"]),
        "note": "仅样本对拍证据，不能推广到全人口；生成链存在 SDK(qfq) 与 HTTP(原始) 双路径，"
                "尺度口径未全量核销；小乘子案例容差内 RAW/QFQ 歧义已单列",
        "cases": scale_results,
    }

    # ---------- 汇总（六个维度分别判定，保留完整人口） ----------
    structural_counts = df["structural_verdict"].value_counts().to_dict()
    reason_counts = Counter()
    for r in df["isolation_reasons"]:
        if r:
            for x in r.split(";"):
                reason_counts[x.split("(")[0]] += 1
    dims = {
        "membership_binding(历史成员关系可证)": dict(Counter(df["membership_binding"].fillna("MISSING"))),
        "eligibility_known_at_P_day_decision(须独立证据)": {"UNVERIFIED": int((df["eligibility_known_at_time"] == "UNVERIFIED").sum())},
        "calendar_compare(W映射一致)": {"structural_fail": int(df["isolation_reasons"].str.contains("CALENDAR").sum()),
                                        "ok": int((~df["isolation_reasons"].str.contains("CALENDAR")).sum())},
        "adj_status_verified(最终缓存状态)": dict(Counter(df["adj_status"].fillna("MISSING"))),
        "coverage_verified([P,W末日]完整性独立证据)": {"UNVERIFIED": len(df)},
        "coverage_chain_ok(链可构建_状态层面)": {str(k): int(v) for k, v in
                                              Counter(df["coverage_chain_ok"]).items()},
        "scale_class(码级判别_全事件)": dict(Counter(df["scale_class"])),
        "scale_verified(码级RAW才可套复权)": {"RAW_VERIFIED": int((df["scale_class"] == "RAW").sum()),
                                              "UNVERIFIED": int((df["scale_class"] != "RAW").sum())},
        "window_scale_crossday_diag(跨日比法_方法受限_非数据判定_非门禁)": dict(Counter(df["window_scale_crossday_diag"])),
        "window_scale_evidence_method(待建_来源溯源+独立价格比对)": dict(Counter(df["window_scale_evidence_method"])),
        "raw_scale_codelevel_records_vs_unique_codes(147条历史支持证据保留)": {"raw_records": n_raw_records, "raw_unique_codes": n_raw_unique},
        "label_eligibility(工程标签准入)": dict(Counter(df["label_eligibility"])),
        "label_note": "ENGINEERING_OK 须 RAW+链可用+覆盖核销三者皆满足（当前覆盖全 UNVERIFIED，故 0）；"
                      "RAW+链可用但覆盖未核销=SCALE_SUPPORTED_COVERAGE_PENDING（正式 MAE/Y_tail 仍 NULL）；"
                      "其余 LABEL_NULL。training_eligible 全 False。",
        "training_eligible": {"False": int((~df["training_eligible"]).sum())},
        "time_evidence(历史到达时刻)": {"UNKNOWN": len(df)},
    }
    summary = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "network_requests": 0,
        "population": {"events": len(df), "unique_code": int(df["code"].nunique()),
                       "unique_codeE": int(df.groupby(["code", "E"]).ngroups),
                       "version": "DUAL_PATH_1013"},
        "fetch_epoch(抓取时点_仅信息_非覆盖终点)": fetch_epoch,
        "six_dimension_verification": dims,
        "structural_verdict(仅结构性缺数据隔离)": structural_counts,
        "isolation_reason_counts": dict(reason_counts),
        "fail_reconciliation": {"fail_log_distinct": len(fail_codes),
                                "historical(now_in_cache)": fail_hist,
                                "current(not_in_cache)": fail_curr,
                                "note": "最终状态以缓存为准；历史失败不与 OK/EMPTY 重复计数"},
        "decision_available_ts": {"value": "UNKNOWN", "n": len(df),
                                  "note": "无原始日志，今天的数据管线无法追溯补造历史到达时刻；"
                                          "建议另签'历史标签工程用途'范围，保留全部 training_eligible=False；"
                                          "前瞻运行再真实记录到达时刻。不制造时间证据。"},
        "calendar_diff_vs_independent": cal_diff,
        "price_scale_crossday": scale_summary,
        "generation_chain_evidence": {
            "build_full_market_daily": "fetch_daily_sdk(..., fq='qfq') 主路径前复权；HTTP 兜底为原始日k",
            "build_increment": "同口径 SDK qfq",
            "risk": "生成链双路径（SDK qfq / HTTP 原始）并存，data_daily_parts 存在混合尺度风险；"
                    "OHLC 尺度未全量核销；pre_close 已实证为 qfq 锚定（危险，禁用）",
        },
    }
    df.to_csv(HERE / "per_candidate_verification_v1.csv", index=False, encoding="utf-8-sig")
    (HERE / "per_candidate_verification_summary_v1.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=1, default=str), encoding="utf-8")

    # ---------- 打印 ----------
    print("==== PER-CANDIDATE VERIFICATION v1 (六维度分别判定) ====")
    print(f"population: events={len(df)} unique_code={summary['population']['unique_code']} "
          f"unique(code,E)={summary['population']['unique_codeE']}")
    print(f"structural_verdict: {structural_counts}  (仅结构性缺数据隔离；软缺口不隔离)")
    print(f"isolation reasons: {dict(reason_counts)}")
    print("six-dimension:", json.dumps(dims, ensure_ascii=False))
    print(f"scale_class(码级): {dict(Counter(df['scale_class']))}")
    print(f"window_scale_crossday_diag(跨日比法_方法受限_非数据判定_非门禁): {dict(Counter(df['window_scale_crossday_diag']))}")
    print(f"raw_scale_codelevel(147条历史支持证据保留): records={n_raw_records} unique_codes={n_raw_unique}")
    print(f"label_eligibility: {dict(Counter(df['label_eligibility']))}  "
          f"(ENGINEERING_OK须覆盖核销; SCALE_SUPPORTED_COVERAGE_PENDING=RAW+链可用但覆盖未核销; 正式标签全NULL)")
    print(f"training_eligible: all False (历史到达时刻 UNKNOWN)")
    print(f"adj status on candidates: {dict(Counter(df['adj_status']))}")
    print(f"fail reconciliation: historical={fail_hist} current={fail_curr}")
    cd = cal_diff
    print(f"calendar vs independent(akshare {AK_CAL_VERSION}): independent_in_range="
          f"{cd['independent_trading_days_in_range']} data_days={cd['data_calendar_days']} "
          f"missing_in_data={len(cd['missing_in_data(independent_day_no_bars)'])} "
          f"extra_in_data={len(cd['extra_in_data(bars_on_non_trading_day)'])}")
    if cd["missing_in_data(independent_day_no_bars)"]:
        print("  missing days:", cd["missing_in_data(independent_day_no_bars)"][:20])
    if cd["extra_in_data(bars_on_non_trading_day)"]:
        print("  extra days:", cd["extra_in_data(bars_on_non_trading_day)"][:20])
    ss = scale_summary
    print(f"price_scale crossday: n={ss['n_cases']} classify={ss['classify_counts']} "
          f"pct_chg_consistent={ss['pct_chg_consistent_n']} (样本证据；全人口 RAW 未核销)")
    for r in ss["cases"][:12]:
        print(f"  {r['code']} ex={r['ex_date']} mult={r['mult']} ratio={r['ratio']} -> {r['classify']}")
    print(f"\nwrote per_candidate_verification_v1.csv / per_candidate_verification_summary_v1.json "
          f"({time.time()-t0:.1f}s)")


if __name__ == "__main__":
    main()
