# -*- coding: utf-8 -*-
"""行动过滤模型 第〇阶段盘点 (合同 v1 FROZEN, 第〇阶段随签署生效; 零网络零采集零拟合).

样本框 = 触发记录件 187 条. 逐段 (train/val/test) 报告: 触发事件数/成交且闭环数(标签分母)/
主标签正负样本数/左尾标签数/标签覆盖率(未成交另列)/白名单四字段缺失率/有效E日期数;
五项训练样本门逐段判定 (train=拟合门, val/test 仅描述不移动);
不变性核验: ①逐行闭环隔离 (label_end 后分钟截断标签不变) ②四字段特征未来扰动不变性
(特征只用 P 及以前日线, 扰动 P 后日线特征不变).
标签来源=逐事件逐轨道记录 (入场价/净PnL/实际投入成本/label_end_ts/有效性原因), 禁聚合反推.
产物: gcp_247_action_filter_phase0_v1.json/.md (RUN 新文件; 写前查预算).
"""
from __future__ import annotations
import sys, json, time, hashlib, glob
from pathlib import Path
from collections import Counter, defaultdict

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import gcp_dual_path_coverage_probe as PROBE
import dual_path_execution_v0_1_1 as D

BASE = HERE / "gcp_round_b_small_sample_v1"
OUT_JSON = BASE / "gcp_247_action_filter_phase0_v1.json"
OUT_MD = BASE / "gcp_247_action_filter_phase0_v1.md"
BUDGET = 50 * 1024 * 1024
LEFT_TAIL = -0.05
GATES = {"pos": 30, "neg": 30, "dates": 40, "coverage": 0.80, "feat_missing": 0.10}


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def dir_size(p: Path) -> int:
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file())


def write_checked(path: Path, text: str):
    b = len(text.encode("utf-8"))
    if dir_size(BASE) + b > BUDGET:
        raise RuntimeError(f"budget exceeded before writing {path.name} (stop)")
    path.write_text(text, encoding="utf-8")
    print(f"  written {path.name} (+{b} B; dir now {dir_size(BASE)} B)")


def main():
    t0 = time.time()
    print(f"[预算] dir={dir_size(BASE)} B / {BUDGET} B")
    trig_recs = json.loads(open(BASE / "gcp_247_ruleA_trigger_records_v1.json", encoding="utf-8").read())["records"]
    sample = [r for r in trig_recs if r["mutual_status"] == "1_有行动"]
    print(f"样本框: {len(sample)} 触发事件")

    panel = PROBE.get_daily_parquet()
    panel["code"] = panel["code"].astype(str).str.zfill(6)
    panel["date"] = panel["date"].astype(int)
    cal = sorted(int(d) for d in panel["date"].unique())
    daily_by_code = {c: g.sort_duplicates("date").sort_values("date").reset_index(drop=True)
                     if hasattr(g, "sort_duplicates") else g.sort_values("date").reset_index(drop=True)
                     for c, g in panel.groupby("code")}
    D.MIN_DIRS.insert(0, BASE / "deferral_collect_v2" / "by_pair")
    D.MIN_DIRS.insert(1, BASE / "p2b1_collect" / "by_pair")
    mbd_cache = {}
    def get_minute(code):
        if code not in mbd_cache:
            mbd_cache[code] = D.load_minute_code(code)[0]
        return mbd_cache[code]
    cov = {(r["code"], int(r["E"])): r["board"] for r in
           json.loads(open(BASE / "gcp_247_coverage_verify.json", encoding="utf-8").read())["rows"]}
    def board_of(c, E):
        if (c, E) in cov:
            return cov[(c, E)]
        if c.startswith(("300", "301")):
            return "GEM"
        if c.startswith("688"):
            return "STAR"
        return "SH_MAIN" if c.startswith("6") else "SZ_MAIN"

    def features(code, Pd, daily_df):
        """白名单特征: contraction/vol_cv10 (P 及以前日线); setup/branch 取自触发记录."""
        if daily_df is None:
            return {"contraction": None, "vol_cv10": None}
        loc = int(daily_df["date"].searchsorted(Pd, side="right"))
        if loc < 20:
            return {"contraction": None, "vol_cv10": None}
        d_sub = daily_df.iloc[:loc]
        atr_ref = PROBE.E.build_atr_ref(d_sub)
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
        return {"contraction": (round(float(contraction), 6) if np.isfinite(contraction) else None),
                "vol_cv10": (round(cv, 6) if np.isfinite(cv) else None)}

    chains = json.loads(open(r"D:\stargazer\stargazer_ml\v5_hybrid\opportunity_v2\gate1c\feb2026_adjustment_cache_v1\adj_cache.json", encoding="utf-8").read())
    def corp_unknown(code, E_, exit_day):
        ch = chains.get(code)
        if ch and ch.get("status") in ("OK", "EMPTY"):
            return False
        return True

    rows = []
    for r in sample:
        code, Ee, Pd = r["code"], int(r["E"]), int(r["P"])
        dd = daily_by_code.get(code)
        feat = features(code, Pd, dd)
        mbd = get_minute(code)
        res = D.execute_event(code, Ee, board_of(code, Ee), None, r["setup"],
                              signal_ts=int(r["first_action_ts"]), minute_by_day=mbd,
                              daily_df=dd, cal=cal)
        ent = res.get("entry") or {}
        t = res.get("H3") or {}
        full = t.get("status") == "SIM_COMPLETE" and t.get("fully_exited") and t.get("net") is not None
        valid = False
        reason = t.get("status") or "NONE"
        if full:
            exit_day = int(t["last_fill_ts"] // 1000000)
            if r["segment"] in ("train", "val") and exit_day >= (20260309 if r["segment"] == "train" else 20260610):
                reason = "CROSS_SEGMENT_LABEL"
            elif corp_unknown(code, Ee, exit_day):
                reason = "CORP_UNKNOWN"
            else:
                valid = True
                reason = "COMPLETE"
        label_main = (1 if float(t["net"]) > 0 else 0) if valid else None
        label_tail = (1 if float(t["net"]) <= LEFT_TAIL else 0) if valid else None
        rows.append({"code": code, "E": Ee, "P": Pd, "segment": r["segment"],
                     "setup": r["setup"], "branch": r["action_kind"], "first_action_ts": r["first_action_ts"],
                     "feat_contraction": feat["contraction"], "feat_vol_cv10": feat["vol_cv10"],
                     "entry_status": ent.get("status"),
                     "traded_closed": bool(valid),
                     "entry_price": ent.get("sim_fill_price"), "C_buy": ent.get("amount"),
                     "pnl": t.get("pnl"), "net": t.get("net"),
                     "label_end_ts": t.get("label_end_ts"),
                     "validity_reason": reason,
                     "label_main": label_main, "label_tail": label_tail})

    # ---------- 分段盘点 ----------
    seg_report = {}
    for seg in ("train", "val", "test"):
        rs = [r for r in rows if r["segment"] == seg]
        n_trig = len(rs)
        n_closed = sum(1 for r in rs if r["traded_closed"])
        n_pos = sum(1 for r in rs if r["label_main"] == 1)
        n_neg = sum(1 for r in rs if r["label_main"] == 0)
        n_tail = sum(1 for r in rs if r["label_tail"] == 1)
        n_nofill = sum(1 for r in rs if r["entry_status"] == "NO_FILL_EOD")
        n_null = n_trig - n_closed - n_nofill
        coverage = round(n_closed / n_trig, 4) if n_trig else None
        dates = len({r["E"] for r in rs})
        dates_labeled = len({r["E"] for r in rs if r["traded_closed"]})
        feat_miss = {}
        for f in ("feat_contraction", "feat_vol_cv10"):
            feat_miss[f] = round(sum(1 for r in rs if r[f] is None) / n_trig, 4) if n_trig else None
        feat_miss["setup"] = 0.0
        feat_miss["branch"] = 0.0
        gates = {
            "pos>=30": (n_pos >= GATES["pos"]) if n_trig else None,
            "neg>=30": (n_neg >= GATES["neg"]) if n_trig else None,
            "dates>=40": (dates_labeled >= GATES["dates"]) if n_trig else None,
            "coverage>=80%": (coverage is not None and coverage >= GATES["coverage"]),
            "feat_missing<=10%": all(v is not None and v <= GATES["feat_missing"] for v in feat_miss.values()),
        }
        seg_report[seg] = {"triggered": n_trig, "traded_closed": n_closed,
                           "label_pos": n_pos, "label_neg": n_neg, "label_tail": n_tail,
                           "confirmed_no_fill": n_nofill, "null_or_invalid": n_null,
                           "label_coverage": coverage, "n_dates_all": dates,
                           "n_dates_labeled": dates_labeled,
                           "feature_missing_rate": feat_miss, "gates": gates,
                           "gate_pass_all": all(gates.values())}
        print(seg, seg_report[seg])

    # ---------- 不变性核验 ----------
    labeled = [r for r in rows if r["traded_closed"]][:5]
    iso_ok, iso_detail = 0, []
    for r in labeled:
        mbd_full = get_minute(r["code"])
        end_day = int(r["label_end_ts"] // 1000000)
        mbd_trunc = {d: df for d, df in mbd_full.items() if d <= end_day}
        res2 = D.execute_event(r["code"], r["E"], board_of(r["code"], r["E"]), None, r["setup"],
                               signal_ts=int(r["first_action_ts"]),
                               minute_by_day=mbd_trunc, daily_df=daily_by_code.get(r["code"]), cal=cal)
        t2 = res2.get("H3") or {}
        same = (t2.get("label_end_ts") == r["label_end_ts"] and t2.get("net") == r["net"]
                and t2.get("status") == "SIM_COMPLETE")
        iso_ok += bool(same)
        iso_detail.append({"code": r["code"], "E": r["E"], "unchanged": bool(same)})
    feat_ok, feat_detail = 0, []
    for r in labeled:
        dd = daily_by_code.get(r["code"])
        dd_pert = dd.copy()
        mask = dd_pert["date"].astype(int) >= int(r["E"])
        dd_pert.loc[mask, "close"] = dd_pert.loc[mask, "close"] * 1.5 + 3.0
        dd_pert.loc[mask, "volume"] = dd_pert.loc[mask, "volume"] * 2.0
        f1 = features(r["code"], int(r["P"]), dd)
        f2 = features(r["code"], int(r["P"]), dd_pert)
        same = (f1["contraction"] == f2["contraction"] and f1["vol_cv10"] == f2["vol_cv10"])
        feat_ok += bool(same)
        feat_detail.append({"code": r["code"], "E": r["E"], "unchanged": bool(same)})

    out = {"phase": "GCP_247_ACTION_FILTER_PHASE0_V1",
           "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
           "network_requests": 0, "no_fitting": True,
           "contract": "contracts/action_filter_model_research_contract_v1.md (FROZEN; 第〇阶段随签署生效)",
           "sample_frame": {"n": len(sample), "source": "gcp_247_ruleA_trigger_records_v1.json (f6c4c997…)"},
           "labels": {"main": "成交且闭环 H3 净收益>0→1 否则 0; 未成交/无行动/NULL 不入训练",
                      "left_tail_aux": "净收益<=-5% (本轮提案阈值, 未验证最优)",
                      "source": "逐事件逐轨道记录 (禁聚合反推)"},
           "segment_report": seg_report,
           "invariance_checks": {
               "closed_loop_isolation": {"n_checked": len(labeled), "n_unchanged": iso_ok, "detail": iso_detail,
                                          "rule": "label_end_ts 后分钟截断, 标签/net/终点不变"},
               "feature_future_perturbation": {"n_checked": len(labeled), "n_unchanged": feat_ok,
                                                "detail": feat_detail,
                                                "rule": "P 后日线 close×1.5+3 / volume×2 扰动, 特征不变 (特征只用 P 及以前)"}},
           "gates_note": "train=拟合门(过门后拟合仍须单独授权); val/test 仅描述, 不移动边界/门槛/参数; 门=有界拟合门, 非统计有效性保证",
           "rows": rows,
           "elapsed_s": round(time.time() - t0, 1)}
    write_checked(OUT_JSON, json.dumps(out, ensure_ascii=False, indent=1))

    L = ["# 行动过滤模型 第〇阶段盘点 v1 (零网络零拟合, 合同 v1 FROZEN)", "",
         f"- 生成: {out['generated_at']} | 样本框 {len(sample)} 触发事件 | 用时 {out['elapsed_s']}s", "",
         "## 分段盘点 (门: 盈利≥30/非盈利≥30/有效日期≥40/覆盖≥80%/特征缺失≤10%)", "",
         "| 段 | 触发 | 成交闭环 | 正/负 | 左尾 | 未成交 | NULL/无效 | 覆盖 | 日期(有标) | 特征缺失 | 门判定 |",
         "|---|---|---|---|---|---|---|---|---|---|---|"]
    for seg in ("train", "val", "test"):
        s = seg_report[seg]
        fm = s["feature_missing_rate"]
        L.append(f"| {seg} | {s['triggered']} | {s['traded_closed']} | {s['label_pos']}/{s['label_neg']} | "
                 f"{s['label_tail']} | {s['confirmed_no_fill']} | {s['null_or_invalid']} | "
                 f"{s['label_coverage']*100:.1f}% | {s['n_dates_labeled']}/{s['n_dates_all']} | "
                 f"c={fm['feat_contraction']},v={fm['feat_vol_cv10']} | "
                 f"{'**过门**' if s['gate_pass_all'] else '未过: ' + ','.join(k for k,v in s['gates'].items() if not v)} |")
    L += ["", "## 不变性核验", "",
          f"- 闭环隔离: {iso_ok}/{len(labeled)} 不变 (label_end 后分钟截断)",
          f"- 特征未来扰动: {feat_ok}/{len(labeled)} 不变 (P 后日线扰动)", "",
          "## 声明", "",
          "- train 门判定=拟合门; 拟合仍须用户单独授权; val/test 仅描述不移动; 328 挂起.", ""]
    write_checked(OUT_MD, "\n".join(L))
    print(f"phase0 done ({time.time()-t0:.1f}s); dir={dir_size(BASE)} B")


if __name__ == "__main__":
    main()
