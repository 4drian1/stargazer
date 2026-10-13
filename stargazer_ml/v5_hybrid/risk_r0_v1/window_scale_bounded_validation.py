"""窗口级尺度核销 — 有界验证（BOUNDED VALIDATION）后端实现与案例选择。

目的（用户 2026-10-13 裁定 + 本轮落地）：
  跨日比法被证明"无法判定"窗口尺度（短窗口极少含判别事件），且不能因之误判数据不可用。
  正确的窗口级尺度证据来自两条独立路径之一：
    (A) 来源口径溯源：明确源接口（stockdb 日k）价格口径，追踪每行来源/转换过程；
    (B) 独立价格比对：与独立、口径明确的价格源比对 P 日参考价与窗口内必要价格。
  公司行为案例（大比例除权）用于**验证转换方法**，不要求每条候选窗口都有案例；
  这些案例**不加入训练人口**。

本轮落地（零网络、不写训练标签）：
  - 已实现一个可运行后端 method_transformation_continuity：用已知公司行为（mult）检验本数据集
    价格流在除权日的**连续性方向**——QFQ 应连续（比值≈1），RAW 应在除权日跳变≈mult。
    这是"转换方向验证"（同一事件链内部一致性），是公司行为案例的允许用途；
    **不是**对 1,013 人口的独立尺度证据（独立证据须来源溯源或独立价格比对，见后端 B/C）。
  - 判定分级（用户 2026-10-13 收紧）：比值方向命中但带真实市场涨跌噪声者标
    `RAW_LIKE_DIRECTION_SUPPORTED` / `QFQ_LIKE_DIRECTION_SUPPORTED`（方向支持，**非 RAW 确认**），
    不得据少数案例宣告整库 OHLC 为 RAW；缺覆盖者标 `NOT_TESTED_DATA_MISSING`（非通过、非方法失败）。
  - 后端 B method_independent_compare（独立价格比对，优先核验 P 日参考收盘与后五日**各自** low）与
    后端 A method_provenance_trace（来源溯源）仍为 STUB，需独立来源授权/网络，**未运行、不计 PASS**。
  - 跨分片 / 跨 SDK-HTTP 案例仍待独立来源证据，**不把未测试项算 PASS**。

运行：python window_scale_bounded_validation.py
  -> 打印并写 bounded_validation_cases.json（案例）、bounded_validation_evidence_v1.json（已运行后端证据）
"""
from __future__ import annotations

import json
import glob
from collections import defaultdict
from pathlib import Path

import pyarrow.dataset as ds

HERE = Path(__file__).resolve().parent
BASE = HERE.parent
EV_JSON = BASE / "green_common_profile/gcp_round_b_small_sample_v1/r1_branch_at_decision_v0_1_2.json"
FETCH_LIST = BASE / "green_common_profile/contracts/dual_path_e_day_fetch_list.txt"
CACHE = BASE / "opportunity_v2/gate1c/feb2026_adjustment_cache_v1/adj_cache.json"
DAILY = r"D:\a股数据\data_daily_parts"
DAILY_FILES = sorted(glob.glob(f"{DAILY}\\part_*.parquet"))

# ---- 容差（写死；但须如实定性）----
# 重要定性：TOL_RAW_REL=0.15 是**看过此前案例后设定的粗粒度方向诊断容差**，
# 属"诊断方法修订"，**不是**这些案例的独立预注册验证，也**不能**用于正式价格尺度准入。
# 它仅用于 transformation_continuity 的粗筛方向（RAW_LIKE / QFQ_LIKE），正式尺度准入须由
# method_independent_compare 用独立源 + 执行时冻结的容差（按价格精度+语义）判定。
# 除权日前后比值同时含"真实市场涨跌"，故本后端只支持"转换方向"，不能据此确认尺度。
TOL_QFQ = 0.05          # QFQ 连续性粗筛容差（诊断用，非准入）
TOL_RAW_REL = 0.15      # RAW 跳变相对 mult 粗筛容差（诊断用，看过案例后设定，非预注册/非准入）
MIN_GAP = 0.20          # 仅选 |mult-1|>=0.20 的大比例事件，确保 RAW/QFQ 方向可分离


def load_calendar():
    dates = ds.dataset(DAILY_FILES, format="parquet").to_table(columns=["date"]) \
        .column("date").to_pylist()
    return sorted({int(d) for d in dates})


def load_close_series(code: str):
    """取某 code 的 (date->close) 映射（零网络，仅本地 parquet）。"""
    t = ds.dataset(DAILY_FILES, format="parquet").to_table(
        columns=["code", "date", "close"], filter=ds.field("code") == code)
    df = t.to_pandas()
    return {int(d): float(c) for d, c in zip(df["date"], df["close"])}


# ---------------- 案例选择（零网络） ----------------
def select_cases(max_per_category: int = 6, max_corp_fixed: int = 12) -> dict:
    cal = load_calendar()
    cal_set = set(cal)
    cal_idx = {d: i for i, d in enumerate(cal)}

    events = json.loads(EV_JSON.read_text(encoding="utf-8"))["events"]
    fl_rows = [l.split(",") for l in FETCH_LIST.read_text(encoding="utf-8").strip().splitlines()]
    fl_map = {(r[0], int(r[1])): (int(r[2]), r[3], r[4]) for r in fl_rows}
    cache = json.loads(CACHE.read_text(encoding="utf-8"))

    # (1) 训练人口内跨分片案例（来自 1,013 候选）
    shard_of = {}
    for p in DAILY_FILES:
        t = ds.dataset(p, format="parquet").to_table(columns=["code", "date"]).to_pandas()
        for c, d in zip(t["code"].astype(str).str.zfill(6), t["date"].astype(int)):
            shard_of[(c, d)] = Path(p).name

    cases = defaultdict(list)
    for e in events:
        c = str(e["code"]).zfill(6)
        E = int(e["E"])
        if (c, E) not in fl_map:
            continue
        P, setup, fstat = fl_map[(c, E)]
        if P not in cal_set:
            continue
        i = cal_idx[P]
        W = cal[i + 1:i + 6]
        if len(W) != 5:
            continue
        dates = [P] + list(W)
        shards = {shard_of.get((c, d)) for d in dates if (c, d) in shard_of}
        if len(shards) > 1:
            cases["cross_shard"].append(
                {"code": c, "E": E, "P": P, "W_end": W[-1], "shards": sorted(shards),
                 "status": "PENDING_INDEPENDENT_SOURCE"})  # 待独立来源证据，不计 PASS

    # (2) 公司行为边界固定工程案例（来自完整 cache，非训练人口）
    corp_fixed = []
    for code, blob in cache.items():
        code = str(code).zfill(6)
        for d_s, m in blob.get("events", []):
            d, m = int(d_s), float(m)
            if not m or abs(m - 1) < MIN_GAP:
                continue
            if d not in cal_idx:
                continue
            corp_fixed.append({"code": code, "event_date": d, "mult": m})
    corp_fixed.sort(key=lambda x: -abs(x["mult"] - 1))
    for cf in corp_fixed[:max_corp_fixed]:
        cf["in_training_population"] = False  # 公司行为固定案例不加入训练人口
        cases["corp_action_fixed"].append(cf)

    out = {k: v[:max_per_category] if k in ("cross_shard", "corp_action_boundary") else v
           for k, v in cases.items()}
    out["_meta"] = {
        "total_training_candidates": len(events),
        "selection_rule": "cross_shard=跨 part 文件(训练人口内,待独立来源); "
                          "corp_action_fixed=完整cache中|mult-1|>=%.2f的大比例事件(非训练人口)" % MIN_GAP,
        "independent_source": "NOT_AVAILABLE (零网络)；method_independent_compare / method_provenance_trace 仍为 STUB",
        "note": "公司行为固定案例验证转换方向(同一事件链内部一致性)，非对人口的独立尺度证据；"
                "跨分片/跨SDK-HTTP 待独立来源，未测试项不计 PASS",
    }
    return out


# ---------------- 方法后端 ----------------
class ScaleVerdict:
    def __init__(self, verdict, method, needs, assumptions, detail=""):
        self.verdict = verdict          # "RAW_LIKE_DIRECTION_SUPPORTED" / "QFQ_LIKE_DIRECTION_SUPPORTED" /
        self.method = method            # "AMBIGUOUS" / "UNDETERMINED" / "NOT_TESTED_DATA_MISSING"
        self.needs = needs
        self.assumptions = assumptions
        self.detail = detail

    def to_dict(self):
        return {"verdict": self.verdict, "method": self.method,
                "needs": self.needs, "assumptions": self.assumptions, "detail": self.detail}


def method_transformation_continuity(code: str, event_date: int, mult: float) -> ScaleVerdict:
    """已运行后端：用已知公司行为检验本数据集价格流在除权日的连续性方向。

    原理（同一事件链内部一致性，公司行为案例的允许用途）：
      - QFQ：复权使序列连续，除权日 close_{D-1}/close_D ≈ 1（仅正常日波动）。
      - RAW：除权日价格跳变≈mult，close_{D-1}/close_D ≈ mult。
    注意：除权日前后比值**同时含真实市场涨跌**，故本后端只支持"转换方向"，
    **不能据比值接近乘子确认尺度**；尺度确认须 method_independent_compare 独立价格比对。
    判定分级（预先冻结容差，非观感）：
      - 方向命中且相对 mult 偏差在 TOL_RAW_REL 内 → `RAW_LIKE_DIRECTION_SUPPORTED`（方向支持，非 RAW 确认）
      - 方向命中且≈1 在 TOL_QFQ 内 → `QFQ_LIKE_DIRECTION_SUPPORTED`
      - 两者皆可分离 → `AMBIGUOUS`；皆否 → `UNDETERMINED`
      - 缺覆盖（日历/价格缺失）→ `NOT_TESTED_DATA_MISSING`（非通过、非方法失败）
    """
    series = load_close_series(code)
    cal = load_calendar()
    ci = {d: i for i, d in enumerate(cal)}
    if event_date not in ci:
        return ScaleVerdict("NOT_TESTED_DATA_MISSING", "transformation_continuity",
                            needs="event_date 须在交易日历内",
                            assumptions="", detail="event_date_not_in_calendar")
    i = ci[event_date]
    pre, post = cal[i - 1], cal[i + 1]
    c_pre, c_ev, c_post = series.get(pre), series.get(event_date), series.get(post)
    if None in (c_pre, c_ev, c_post):
        return ScaleVerdict("NOT_TESTED_DATA_MISSING", "transformation_continuity",
                            needs="", assumptions="",
                            detail=f"missing_close pre={pre}:{c_pre} ev={event_date}:{c_ev} post={post}:{c_post}")
    r_pre_ev = c_pre / c_ev          # 除权日比值
    r_ev_post = c_ev / c_post        # 除权后首日比值
    raw_dev = abs(r_pre_ev - mult) / max(mult, 1.0)     # 相对 mult 的偏差（含真实市场涨跌）
    qfq_dev = abs(r_pre_ev - 1.0)                        # 相对 1 的偏差
    qfq_ok = qfq_dev <= TOL_QFQ and abs(r_ev_post - 1) <= TOL_QFQ
    raw_ok = raw_dev <= TOL_RAW_REL and abs(r_ev_post - 1) <= TOL_QFQ
    if qfq_ok and not raw_ok:
        v = "QFQ_LIKE_DIRECTION_SUPPORTED"
    elif raw_ok and not qfq_ok:
        v = "RAW_LIKE_DIRECTION_SUPPORTED"
    elif qfq_ok and raw_ok:
        v = "AMBIGUOUS"
    else:
        v = "UNDETERMINED"
    return ScaleVerdict(v, "transformation_continuity", needs="",
                        assumptions="大比例除权使 RAW/QFQ 方向可分离",
                        detail=f"r_pre_ev={r_pre_ev:.4f} r_ev_post={r_ev_post:.4f} mult={mult:.4f} "
                               f"raw_dev={raw_dev:.4f} qfq_dev={qfq_dev:.4f}")


def method_provenance_trace(code, P, W_end) -> ScaleVerdict:
    """(A) 来源口径溯源 STUB：需 stockdb 日k 口径查询 / data_daily_parts 构建溯源日志（独立授权）。"""
    return ScaleVerdict("UNDETERMINED", "provenance_trace",
                        needs="stockdb 日k 口径查询 / data_daily_parts 构建溯源日志（需授权）",
                        assumptions="源接口返回单一明确口径且可追溯每行转换")


def method_independent_compare(code, P, W_end) -> ScaleVerdict:
    """(B) 独立价格比对 STUB（**优先核验真正影响标签的价格**，需独立来源授权/网络）。

    目标字段（与标签契约对齐，见 R0_V1_DEVELOPMENT.md）：
      - P 日参考收盘 R_P（= data_daily_parts 中 P 日 close，必须同口径）
      - **后五日各自的 low**：须取 P+1..P+5 共五个交易日的**每日 low**（逐日保留比对），
        不能只比一个窗口最小值——否则可能漏掉某天的尺度异常或日期错位。
    含既有跨分片案例（如 000600）与公司行为边界案例；逐候选、逐日比对并冻结差异容差。
    注意：
      - 再次读取同一个 stockdb 只能证明一致性，不能算独立核验；独立源须与 stockdb 不同源。
      - 差异容差按**价格精度及数据语义冻结**（如最小变动价位/四舍五入粒度），不用宽泛百分比"凑匹配"；
        差异须**留因**（记录来源、口径、可能的复权/停牌/集合竞价差异），**不静默覆盖原数据**。
      - 独立价格匹配**不自动证明复权事件完整**：尺度核销与覆盖核销**分别出结论**，两者各自通过后
        才可能批准正式标签构建（见 risk_r0_v1_coverage_acceptance_rules.md）。
    """
    return ScaleVerdict("UNDETERMINED", "independent_compare",
                        needs="独立价格源 API（与 stockdb 不同源，须登记来源/覆盖/完整性/口径）+ 网络授权 + 单独采集批准",
                        assumptions="独立源口径明确且与本数据集可比；差异容差按价格精度+语义冻结、差异留因不覆盖")


METHOD_STATUS = {
    "transformation_continuity": "IMPLEMENTED_AND_RUN (转换方向验证；非人口独立证据)",
    "provenance_trace": "STUB_PENDING_AUTHORIZATION",
    "independent_compare": "STUB_PENDING_AUTHORIZATION",
}


def main():
    cases = select_cases()
    print("==== BOUNDED VALIDATION (有界验证) ====")
    for k, v in cases.items():
        if k == "_meta":
            continue
        print(f"  {k}: {len(v)} 例  e.g. {v[0] if v else None}")
    print("  _meta:", json.dumps(cases["_meta"], ensure_ascii=False))

    # 运行已就绪后端：公司行为固定案例 → 转换方向验证
    evidence = []
    for cf in cases.get("corp_action_fixed", []):
        v = method_transformation_continuity(cf["code"], cf["event_date"], cf["mult"])
        evidence.append({"code": cf["code"], "event_date": cf["event_date"], "mult": cf["mult"],
                         "verdict": v.to_dict()})
    n = len(evidence)
    from collections import Counter
    vc = Counter(e["verdict"]["verdict"] for e in evidence)
    print(f"\n[已运行后端] transformation_continuity on {n} 公司行为固定案例: {dict(vc)}")
    for e in evidence[:6]:
        print(f"  {e['code']} D={e['event_date']} mult={e['mult']:.3f} -> {e['verdict']['verdict']} ({e['verdict']['detail']})")

    print("\n方法后端状态:", METHOD_STATUS)
    print("未运行/不计 PASS: 跨分片/跨SDK-HTTP 案例（待独立来源）；provenance_trace/independent_compare（STUB）")
    print("注意：本后端只给方向支持（RAW_LIKE_DIRECTION_SUPPORTED），**不能据 7 例宣告整库 OHLC 为 RAW**；"
          "缺覆盖者记为 NOT_TESTED_DATA_MISSING（非通过、非方法失败）。")
    print("下一步（用户指定优先级，本轮不采集）：① 先定独立源与有界请求单（vendor/口径/独立性/固定案例/字段(P日收盘+后五日最低)/"
          "日期/请求数/重试/预算），单独授权取数；② 独立价格比对优先核验真正影响标签的 P 日参考收盘与后五日最低价，"
          "差异容差按价格精度+语义冻结、差异留因不覆盖原数据；③ 尺度与覆盖分别出结论，各自核销后才可能批准正式标签构建。")

    (HERE / "bounded_validation_cases.json").write_text(
        json.dumps(cases, ensure_ascii=False, indent=1), encoding="utf-8")
    (HERE / "bounded_validation_evidence_v1.json").write_text(
        json.dumps({"corp_action_fixed_evidence": evidence,
                    "method_status": METHOD_STATUS,
                    "cross_shard_status": "PENDING_INDEPENDENT_SOURCE"},
                   ensure_ascii=False, indent=1), encoding="utf-8")
    print("wrote bounded_validation_cases.json + bounded_validation_evidence_v1.json")


if __name__ == "__main__":
    main()
