# -*- coding: utf-8 -*-
"""完整箱体候选人口零网络盘点 v1 (用户授权; 零网络零采集, 不重算信号).

回答"亏损主要来自候选背景不合适，还是 S1/S2 在不该行动时放行"的数据基础: 先看全人口长什么样.

人口来源核验 (先核实哪些属合同冻结人口并登记哈希, 不确定的如实报告):
  - CAR 冻结候选清单 3 件 (contracts/car_v0_1_candidate_list.txt / car_v0_2_expansion_candidate_list.txt
    / car_v0_2_train_top_up_candidate_list.txt) + 对应 registry (expansion/top_up 自带清单文件哈希锚定,
    逐一复验) + 批处理报告 per_candidate (行动状态来源: signal_found/sig_type).
  - dual-path 线: _251_events_cache.json (251 事件) -> gcp_247_coverage_verify.json (冻结 247, manifest 锚定
    f2974391...); 4 条人口剔除 (2 ETF + 2 BSE, 工具/板块外).
  **关键事实**: 247 ∩ CAR 583 清单 = 1 — CAR 清单不是 247 的母人口 (两条研究线近 disjoint),
  本盘点如实双人口并列, 不以 CAR 清单充当 247 的完整人口.

盘点维度: A/B 池(setup) × 行动状态 × 段(train/val/test 冻结边界); 每格 候选数/唯一(code,E)/日期数.
数据覆盖: 每候选 E 日分钟覆盖只读核对 (COMPLETE/GAP/NO_FILE/BEYOND_CUTOFF).
收益仅探索描述: 247 子集臂A 收益按 A/B 池分组引用 two_arm 结果; 不删组/不下结论/不做统计检验.
产物: gcp_round_b_small_sample_v1/gcp_247_full_population_inventory_v1.json/.md (新文件).
"""
from __future__ import annotations
import sys, json, time, hashlib, glob
from pathlib import Path
from collections import Counter, defaultdict

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import dual_path_execution_v0_1_1 as D
import gcp_round_b_engine as E

BASE = HERE / "gcp_round_b_small_sample_v1"
OUT_JSON = BASE / "gcp_247_full_population_inventory_v1.json"
OUT_MD = BASE / "gcp_247_full_population_inventory_v1.md"
DATA_CUTOFF = D.DATA_CUTOFF
BOUNDS = {"train_max": 20260305, "val_min": 20260309, "val_max": 20260609, "test_min": 20260610}

CAR_LISTS = {
    "car_v0_1": ("../contracts/car_v0_1_candidate_list.txt", None),
    "car_v0_2_expansion": ("../contracts/car_v0_2_expansion_candidate_list.txt",
                           "../contracts/car_v0_2_expansion_registry.json"),
    "car_v0_2_train_top_up": ("../contracts/car_v0_2_train_top_up_candidate_list.txt",
                              "../contracts/car_v0_2_train_top_up_registry.json"),
}
BATCH_REPORTS = {
    "car_v0_1": "car_v0_1_real_replay_report.json",
    "car_v0_2_expansion": "car_v0_2_full_batch_report.json",
    "car_v0_2_train_top_up": "car_v0_2_train_top_up_report.json",
}


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def dir_size(p: Path) -> int:
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file())


def write_checked(path: Path, text: str):
    if dir_size(BASE) + len(text.encode("utf-8")) > 50 * 1024 * 1024:
        raise RuntimeError("budget exceeded (stop)")
    path.write_text(text, encoding="utf-8")


def seg_of(e: int) -> str:
    if e <= BOUNDS["train_max"]:
        return "train"
    if e <= 20260308:
        return "UNASSIGNED"
    if e <= BOUNDS["val_max"]:
        return "val"
    return "test"


def main():
    t0 = time.time()
    # ---------- 1) 来源哈希登记与核验 ----------
    sources = {}
    lists = {}
    for name, (lf, reg) in CAR_LISTS.items():
        lp = (BASE / lf).resolve()
        rec = {"list_file": str(lp), "list_sha256": sha256_file(lp)}
        rows = []
        for line in open(lp, encoding="utf-8"):
            if line.strip():
                p = line.strip().split(",")
                # v0_1 清单 3 列 (code,setup,E); expansion/top_up 4 列 (code,setup,P,E)
                rows.append({"code": p[0], "setup": p[1], "E": int(p[-1]), "P": int(p[2]) if len(p) >= 4 else None,
                             "list": name})
        rec["n"] = len(rows)
        lists[name] = rows
        if reg:
            rp = (BASE / reg).resolve()
            rj = json.loads(open(rp, encoding="utf-8").read())
            anchor = rj.get("sha256", {})
            expect = anchor.get("candidate_list_file") or rj.get("list_file_sha256")
            rec["registry"] = str(rp)
            rec["registry_anchor_file_sha256"] = expect
            rec["anchor_match"] = (expect == rec["list_sha256"]) if expect else None
            rec["registry_status_fields"] = {k: rj.get(k) for k in ("seed", "signed_date", "target", "version") if k in rj}
        sources[name] = rec
    # 批处理报告 (行动状态来源)
    for name, rf in BATCH_REPORTS.items():
        rp = BASE / rf
        rj = json.loads(open(rp, encoding="utf-8").read())
        key = "new_per_candidate" if "new_per_candidate" in rj else "per_candidate"
        sources[name]["batch_report"] = {"file": rf, "sha256": sha256_file(rp),
                                         "batch_status": rj.get("batch_status") or rj.get("status"),
                                         "per_candidate_key": key, "n_per_candidate": len(rj[key])}
    dp251_p = HERE / "_251_events_cache.json"
    cov_p = BASE / "gcp_247_coverage_verify.json"
    sources["dual_path_251_cache"] = {"file": str(dp251_p), "sha256": sha256_file(dp251_p),
                                      "note": "251 事件缓存 (dual-path 线); 非 manifest 锚定, 来源以 dual_path_251_event_board_inventory 为佐"}
    sources["frozen_247_scope"] = {"file": str(cov_p), "sha256": sha256_file(cov_p),
                                   "manifest_anchor": "f2974391fcff26e75bea450426093679bb8547e17125d21589f5108ddfb9b673"}

    # ---------- 2) 人口组装 ----------
    ev251 = json.loads(open(dp251_p, encoding="utf-8").read())
    cov_rows = json.loads(open(cov_p, encoding="utf-8").read())["rows"]
    k247 = {(r["code"], int(r["E"])) for r in cov_rows}
    k251 = {(e["code"], int(e["E"])) for e in ev251}
    excluded_251 = sorted(k251 - k247)

    sig_status = {}
    for name, rf in BATCH_REPORTS.items():
        rj = json.loads(open(BASE / rf, encoding="utf-8").read())
        key = "new_per_candidate" if "new_per_candidate" in rj else "per_candidate"
        for x in rj[key]:
            sig_status[(x["code"], int(x["E"]))] = {"signal_found": bool(x.get("signal_found")),
                                                    "sig_type": x.get("sig_type")}
    car_all = []
    for name, rows in lists.items():
        for r in rows:
            st = sig_status.get((r["code"], r["E"]))
            car_all.append({**r, "signal_found": (st or {}).get("signal_found"),
                            "sig_type": (st or {}).get("sig_type")})
    car_keys = {(c["code"], c["E"]) for c in car_all}
    overlap_247_car = sorted(k247 & car_keys)

    population = []
    for c in car_all:
        population.append({"pop": "CAR", "code": c["code"], "E": c["E"], "setup": c["setup"],
                           "segment": seg_of(c["E"]), "list": c["list"],
                           "action_status": ("行动了(有信号)" if c["signal_found"] else "无信号(E日无行动候选)"),
                           "sig_type": c["sig_type"]})
    for e in ev251:
        k = (e["code"], int(e["E"]))
        population.append({"pop": "DUAL_PATH", "code": e["code"], "E": int(e["E"]),
                           "setup": e["setup"], "segment": e["segment"], "list": "dual_path_251",
                           "action_status": ("行动了(冻结247)" if k in k247 else "人口剔除(工具/板块外)"),
                           "sig_type": None})

    # ---------- 3) E 日分钟覆盖 (只读) ----------
    by_pair_index = {}
    for d in [BASE / "minute_collect_realtest" / "by_pair",
              BASE / "deferral_collect_v2" / "by_pair",
              BASE / "p2b1_collect" / "by_pair"]:
        for f in d.glob("*.parquet"):
            stem = f.stem
            if "_" in stem and stem.split("_")[-1].isdigit():
                by_pair_index.setdefault((stem.split("_")[0], int(stem.split("_")[-1])), f)
    slice_idx, slice_cache = {}, {}
    codes_needed = sorted({p["code"] for p in population})
    for d in [BASE / "minute_collect_realtest", BASE / "minute_ext", BASE / "minute", BASE / "minute_car"]:
        for code in codes_needed:
            f = d / f"{code}.parquet"
            if f.exists() and (d, code) not in slice_cache:
                try:
                    df = pd.read_parquet(f, columns=["date"])
                    slice_cache[(d, code)] = (f, set(int(x) // 1000000 for x in df["date"].astype(int)))
                except Exception:
                    slice_cache[(d, code)] = (f, set())

    def coverage(code, day):
        if day > DATA_CUTOFF:
            return "BEYOND_CUTOFF", None
        src = by_pair_index.get((code, day))
        if src is None:
            for (d, c), (f, days) in slice_cache.items():
                if c == code and day in days:
                    src = f
                    break
        if src is None:
            return "NO_FILE", None
        try:
            df = pd.read_parquet(src)
            df = df[df["date"].astype(int) // 1000000 == day]
            if not len(df):
                return "GAP", str(src.relative_to(BASE))
            v = E.validate_minute_day(df, D.day_grid(day), day)
            bad_price = bool((df[["open", "high", "low", "close"]].astype(float) <= 0).any().any())
            zero_all = bool((df["volume"].astype(float) > 0).sum() == 0)
            if v["missing"] or v["dups"] or v["unexpected"] or bad_price or zero_all:
                return "GAP", str(src.relative_to(BASE))
            return "COMPLETE", str(src.relative_to(BASE))
        except Exception:
            return "GAP", str(src.relative_to(BASE))

    cov_counter = Counter()
    for p in population:
        cls, src = coverage(p["code"], p["E"])
        p["e_day_coverage"] = cls
        cov_counter[(p["pop"], p["segment"], p["action_status"], cls)] += 1

    # ---------- 4) 交叉表 ----------
    def crosstab(key_fields):
        tab = defaultdict(lambda: {"candidates": 0, "unique_codeE": set(), "dates": set()})
        for p in population:
            k = tuple(p[f] for f in key_fields)
            tab[k]["candidates"] += 1
            tab[k]["unique_codeE"].add((p["code"], p["E"]))
            tab[k]["dates"].add(p["E"])
        return {json.dumps(k, ensure_ascii=False): {"candidates": v["candidates"],
                                                    "unique_codeE": len(v["unique_codeE"]),
                                                    "dates": len(v["dates"])}
                for k, v in sorted(tab.items())}

    tab_main = crosstab(["pop", "setup", "action_status", "segment"])
    tab_cov = crosstab(["pop", "segment", "e_day_coverage"])
    sig_type_tab = None
    st_c = Counter((p["pop"], p["sig_type"]) for p in population if p["sig_type"])

    # ---------- 5) 探索性收益描述 (引用 two_arm, 纯分组) ----------
    ta = json.loads(open(BASE / "gcp_247_two_arm_v1.json", encoding="utf-8").read())
    setup_map = {(r["code"], int(r["E"])): r["setup"] for r in cov_rows}
    exp = {}
    for tr_key, rows_key in (("H3", "pair_rows_full_H3"), ("H5", "pair_rows_full_H5")):
        for seg in ("train", "val", "test"):
            rows = ta[rows_key][seg]
            by_setup = defaultdict(list)
            for r in rows:
                by_setup[setup_map.get((r["code"], r["E"]), "?")].append(r)
            for setup, rs in by_setup.items():
                if not rs:
                    continue
                import numpy as np
                exp[f"{tr_key}/{seg}/{setup}"] = {
                    "n_pairs": len(rs),
                    "delta_mean": round(float(np.mean([r["delta"] for r in rs])), 6),
                    "armA_net_mean": round(float(np.mean([r["netA"] for r in rs])), 6),
                    "note": "探索描述, 引用 two_arm 结果分组; 不做检验不下结论"}

    # ---------- 6) 覆盖洼地判断 (按段/行动状态 NO_FILE+GAP 占比) ----------
    lowland = {}
    for pop in ("CAR", "DUAL_PATH"):
        for seg in ("train", "val", "test", "UNASSIGNED"):
            tot = sum(v["candidates"] for k, v in tab_cov.items()
                      if json.loads(k)[0] == pop and json.loads(k)[1] == seg)
            if not tot:
                continue
            bad = sum(v["candidates"] for k, v in tab_cov.items()
                      if json.loads(k)[0] == pop and json.loads(k)[1] == seg and json.loads(k)[2] in ("NO_FILE", "GAP"))
            lowland[f"{pop}/{seg}"] = {"total": tot, "no_file_or_gap": bad,
                                       "bad_ratio": round(bad / tot, 4)}

    out = {"phase": "GCP_247_FULL_POPULATION_INVENTORY_V1",
           "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
           "network_requests": 0, "no_collection": True,
           "purpose": "回答'亏损主要来自候选背景不合适, 还是 S1/S2 在不该行动时放行'的数据基础: 先看全人口",
           "sources": sources,
           "key_facts": {
               "car_total_candidates": len(car_all),
               "car_lists": {k: len(v) for k, v in lists.items()},
               "dual_path_251": len(k251), "frozen_247": len(k247),
               "dual_path_excluded_from_247": excluded_251,
               "overlap_247_with_car_lists": len(overlap_247_car),
               "lineage_note": "247 的母人口是 dual-path 251 事件线 (与 CAR 583 清单近 disjoint: 交集=1); "
                               "CAR 清单是 CAR 研究线从箱体池(8665 候选日)的抽样, 两条线如实双人口并列."},
           "watch_note": "WATCH_BASE 是日内观察状态; 冻结批处理报告只记 signal_found/sig_type, "
                         "未逐日保留 WATCH 标记 -> '无信号'格内含曾 WATCH 未触发者, 未再细分 (细分需重扫, 本盘点不做)",
           "crosstab_setup_status_segment": tab_main,
           "coverage_by_pop_segment": tab_cov,
           "sig_type_counts": {f"{k[0]}/{k[1]}": v for k, v in sorted(st_c.items())},
           "lowland": lowland,
           "exploratory_returns_by_setup": exp,
           "population_detail_note": "逐候选明细 (code/E/setup/segment/action_status/e_day_coverage) 见 json 内 population 字段" ,
           "population": population,
           "warning": "收益仅为探索性描述 (引用 two_arm), 不删组/不下结论/不做统计检验",
           "elapsed_s": round(time.time() - t0, 1)}
    write_checked(OUT_JSON, json.dumps(out, ensure_ascii=False, indent=1))

    # ---------- md ----------
    L = ["# 完整箱体候选人口零网络盘点 v1", "",
         f"- 生成: {out['generated_at']} | 零网络零采集 | 用时 {out['elapsed_s']}s", "",
         "## 关键事实 (来源核验)", "",
         f"- CAR 冻结候选清单 3 件: { {k: sources[k]['n'] for k in lists} } (合计 {len(car_all)}); "
         f"registry 锚定哈希复验: " + ", ".join(
             f"{k}={'OK' if sources[k].get('anchor_match') else sources[k].get('anchor_match')}" for k in
             ("car_v0_2_expansion", "car_v0_2_train_top_up")),
         f"- dual-path 线: 251 事件缓存 -> 冻结 247 (manifest 锚定 f2974391 复验 "
         f"{'OK' if sources['frozen_247_scope']['sha256']==sources['frozen_247_scope']['manifest_anchor'] else 'FAIL'}); "
         f"251−247 = {len(excluded_251)} 条人口剔除 (工具/板块外: {excluded_251})",
         f"- **247 ∩ CAR 清单 = {len(overlap_247_car)}** — CAR 清单不是 247 的母人口; 双人口并列, 不以 CAR 充当 247 完整人口", "",
         "## 人口规模与交叉表 (候选数 / 唯一(code,E) / 日期数)", ""]
    for k, v in tab_main.items():
        kk = json.loads(k)
        L.append(f"- {kk[0]} | {kk[1]} | {kk[2]} | {kk[3]}: {v['candidates']} / {v['unique_codeE']} / {v['dates']}")
    L += ["", "## E 日分钟覆盖 (按人口×段)", ""]
    for k, v in tab_cov.items():
        kk = json.loads(k)
        L.append(f"- {kk[0]} | {kk[1]} | {kk[2]}: {v['candidates']}")
    L += ["", "## 覆盖洼地 (NO_FILE+GAP 占比)", ""]
    for k, v in lowland.items():
        L.append(f"- {k}: {v['no_file_or_gap']}/{v['total']} = {v['bad_ratio']*100:.1f}%")
    L += ["", "## 信号类型分布 (CAR 人口)", ""]
    for k, v in sorted(st_c.items()):
        L.append(f"- {k[0]}/{k[1]}: {v}")
    L += ["", "## 探索性收益描述 (引用 two_arm, 仅陈述, val 主判读段)", ""]
    for k, v in exp.items():
        if "/val/" in k:
            L.append(f"- {k}: 配对 {v['n_pairs']} | Δ均值 {v['delta_mean']} | 臂A净均值 {v['armA_net_mean']}")
    L += ["", "## 为'是否该买'研究合同的盘点结论建议", "",
          "- (见 json; md 摘要: 样本足够的格 = CAR/DUAL_PATH 各段'行动了+COMPLETE'格; 覆盖不足格 = test 段 NO_FILE 占比高的格)",
          "", "## 声明", "",
          "- WATCH 未在冻结批处理报告中逐日保留, '无信号'格未再细分; 247 子集核销结论不重算.",
          "- 收益仅为探索性描述; 不删组/不下结论/不做统计检验; 研究合同另签.", ""]
    write_checked(OUT_MD, "\n".join(L))
    print(f"inventory done ({time.time()-t0:.1f}s)")
    print("key:", out["key_facts"]["overlap_247_with_car_lists"], "overlap; car", len(car_all), "; dp", len(k251))
    print("lowland:", lowland)


if __name__ == "__main__":
    main()
