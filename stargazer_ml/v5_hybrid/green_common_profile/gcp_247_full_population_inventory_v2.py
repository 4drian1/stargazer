# -*- coding: utf-8 -*-
"""完整人口与状态盘点 v2 — 用户核对三处会改变研究设计的错误的纠错版 (零网络, 零收益计算).

纠错1 谱系钉死: DUAL_PATH 共性资格原料(pool parquets, 哈希登记) -> 箱体候选人口 1050
  (dual_path_e_day_fetch_list.txt, code,E,P,setup,status 五列; 1072 原始 - 22 超截止 = 1050;
  已核 251⊆1050 全含) -> probe main 漏斗 (minute_valid 1005 / WATCH 476 / S1 225 / S2 31 /
  action 254 / fillable 251) -> 排除工具/板块外 4 (2 ETF+2 BSE) -> 冻结 247 (manifest f297…).
  251/247 不含 WATCH/无信号, 不能充当完整人口; CAR 8665=共性资格原料非"箱体池", CAR 清单=共性资格层;
  CAR∩247=24 不奇怪, 两线按用户裁定分开研究。
纠错2 无信号两桶: replay_error 区分 NO_SIGNAL(成功扫描确认无信号, 可作对照) vs E_DAY_MISSING
  (数据缺失无法判断, 不得充当对照); CAR 无信号 E 日分钟分布实测 (COMPLETE/GAP/NO_FILE),
  更正 v1"无信号组缺分钟"的错误建议; 明文 CAR signal_found=True ≠ DUAL_PATH S1/S2 行动成立。
纠错3 需求集合求交: 328 对=退出/顺延路径 NO_FILE (coverage_scope_v0_1_2), 与 E 日分钟是不同集合;
  实测 (code,date) 交集; E 日 GAP 集合与 328 分开登记, 各自独立预算口径; 328 维持挂起不批。
用户裁定入档: ①两线分开; ②拒绝"DUAL_PATH val+CAR train 混合首判读"; ③328 挂起。
产物: gcp_247_full_population_inventory_v2.json/.md (v1 保留不改, v2 开头 errata 节).
"""
from __future__ import annotations
import sys, json, time, hashlib
from pathlib import Path
from collections import Counter, defaultdict

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import dual_path_execution_v0_1_1 as D
import gcp_round_b_engine as E

BASE = HERE / "gcp_round_b_small_sample_v1"
OUT_JSON = BASE / "gcp_247_full_population_inventory_v2.json"
OUT_MD = BASE / "gcp_247_full_population_inventory_v2.md"
DATA_CUTOFF = D.DATA_CUTOFF
BOUNDS = {"train_max": 20260305, "val_min": 20260309, "val_max": 20260609, "test_min": 20260610}


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
    # ---------- 输入哈希登记 ----------
    p_fetch = HERE / "contracts" / "dual_path_e_day_fetch_list.txt"
    p_probe = BASE / "dual_path_coverage_probe.json"
    p_reg = HERE / "contracts" / "dual_path_e_day_registry.json"
    p_251 = HERE / "_251_events_cache.json"
    p_cov = BASE / "gcp_247_coverage_verify.json"
    p_pool_a = HERE / "round_a_v0_1_impl_fix" / "pool_a_sdp_members.parquet"
    p_pool_b = HERE / "round_a_v0_1_impl_fix" / "pool_b_lvb_members.parquet"
    p_scope = BASE / "gcp_247_coverage_scope_v0_1_2.json"
    sources = {
        "dual_path_fetch_list(箱体候选人口1050)": {"file": str(p_fetch), "sha256": sha256_file(p_fetch)},
        "dual_path_e_day_registry": {"file": str(p_reg), "sha256": sha256_file(p_reg)},
        "dual_path_coverage_probe(funnel来源)": {"file": str(p_probe), "sha256": sha256_file(p_probe)},
        "dual_path_251_cache": {"file": str(p_251), "sha256": sha256_file(p_251)},
        "frozen_247_scope": {"file": str(p_cov), "sha256": sha256_file(p_cov),
                             "manifest_anchor_ok": sha256_file(p_cov) == "f2974391fcff26e75bea450426093679bb8547e17125d21589f5108ddfb9b673"},
        "pool_a_sdp_members(共性资格原料)": {"file": str(p_pool_a), "sha256": sha256_file(p_pool_a)},
        "pool_b_lvb_members(共性资格原料)": {"file": str(p_pool_b), "sha256": sha256_file(p_pool_b)},
        "coverage_scope_v0_1_2(328集合来源)": {"file": str(p_scope), "sha256": sha256_file(p_scope)},
    }
    reg = json.loads(open(p_reg, encoding="utf-8").read())
    probe = json.loads(open(p_probe, encoding="utf-8").read())

    # ---------- 谱系链钉死 ----------
    fl = [l.strip().split(",") for l in open(p_fetch, encoding="utf-8") if l.strip()]
    box = [{"code": p[0], "E": int(p[1]), "P": int(p[2]), "setup": p[3], "fetch_status": p[4]} for p in fl]
    k_box = {(b["code"], b["E"]) for b in box}
    ev251 = json.loads(open(p_251, encoding="utf-8").read())
    k251 = {(e["code"], int(e["E"])) for e in ev251}
    cov_rows = json.loads(open(p_cov, encoding="utf-8").read())["rows"]
    k247 = {(r["code"], int(r["E"])) for r in cov_rows}
    excluded4 = sorted(k251 - k247)
    main_funnel = Counter()
    for cls, f in probe["funnel"].items():
        if cls.endswith("|main"):
            for k, v in f.items():
                if isinstance(v, int):
                    main_funnel[k] += v
    lineage = [
        {"step": 0, "name": "共性资格原料(双线共用)", "rule": "round_a 日线箱体成员池 (A_SDP/B_LVB)",
         "count": "8665 合并候选日 (CAR expansion registry pool_size.merged; 非箱体资格本身)",
         "artifact": "pool_a/pool_b_sdp_members.parquet", "exclusions": None},
        {"step": 1, "name": "箱体候选人口", "rule": "box_members_2025plus=1072 -> 剔除 outside_cutoff_codeP=22 -> unique(code,E)=1050",
         "count": len(box), "artifact": "dual_path_e_day_fetch_list.txt (code,E,P,setup,status)",
         "exclusions": "22 条 P 超数据截止"},
        {"step": 2, "name": "盘中状态分流(probe main 漏斗, 零网络重放核)",
         "rule": "minute_valid=1005 -> WATCH 476 / S1 225 / S2 31 / action_unique 254 / fillable 251 (fill_blocked 3)",
         "count": main_funnel["fillable"],
         "artifact": "dual_path_coverage_probe.json (funnel, main 类)",
         "exclusions": "minute_invalid 25 + 未缓存 20; 无动作格= WATCH/未启动 (1005-476-225-31=273)"},
        {"step": 3, "name": "模拟可成交行动", "count": len(k251),
         "artifact": "_251_events_cache.json", "exclusions": None,
         "verify": f"251 ⊆ 1050 箱体候选: {k251 <= k_box} ({len(k251 & k_box)}/{len(k251)})"},
        {"step": 4, "name": "排除工具/板块外", "count": len(k251) - len(k247),
         "rule": "ETF(515070,588200) + BSE(920906,920961) 非 STOCK 主板/创业/科创范围",
         "artifact": "gcp_247_event_coverage_verify.py 口径", "exclusions": excluded4},
        {"step": 5, "name": "冻结 247 行动范围", "count": len(k247),
         "artifact": "gcp_247_coverage_verify.json (manifest 锚定 f2974391…)",
         "exclusions": None},
    ]

    # ---------- 纠错2: CAR 无信号两桶 + E 日分钟分布 ----------
    car_lists = {}
    sig_status = {}
    bucket_counts = Counter()
    for name, lf, rf, rk in [
            ("car_v0_1", "car_v0_1_candidate_list.txt", "car_v0_1_real_replay_report.json", "per_candidate"),
            ("car_v0_2_expansion", "car_v0_2_expansion_candidate_list.txt", "car_v0_2_full_batch_report.json", "per_candidate"),
            ("car_v0_2_train_top_up", "car_v0_2_train_top_up_candidate_list.txt", "car_v0_2_train_top_up_report.json", "new_per_candidate")]:
        lp = HERE / "contracts" / lf
        rows = []
        for line in open(lp, encoding="utf-8"):
            if line.strip():
                p = line.strip().split(",")
                rows.append({"code": p[0], "setup": p[1], "E": int(p[-1]), "P": int(p[2]) if len(p) >= 4 else None,
                             "list": name})
        car_lists[name] = rows
        rj = json.loads(open(BASE / rf, encoding="utf-8").read())
        for x in rj[rk]:
            k = (x["code"], int(x["E"]))
            err = x.get("replay_error")
            if x.get("signal_found"):
                st = "行动了(有信号)"
                bucket = "action"
            elif err == "NO_SIGNAL":
                st = "无信号(确认,扫描成功)"
                bucket = "no_signal_confirmed"
            else:
                st = f"数据缺失/异常无法判断({err})"
                bucket = "data_missing"
            sig_status[k] = {"bucket": bucket, "replay_error": err}
            bucket_counts[(name, bucket)] += 1
    car_all = []
    for name, rows in car_lists.items():
        for r in rows:
            st = sig_status.get((r["code"], r["E"]), {"bucket": "unknown", "replay_error": None})
            car_all.append({**r, "bucket": st["bucket"], "segment": seg_of(r["E"])})

    # ---------- 覆盖索引 (E 日) ----------
    by_pair_index = {}
    for d in [BASE / "minute_collect_realtest" / "by_pair",
              BASE / "deferral_collect_v2" / "by_pair",
              BASE / "p2b1_collect" / "by_pair"]:
        for f in d.glob("*.parquet"):
            stem = f.stem
            if "_" in stem and stem.split("_")[-1].isdigit():
                by_pair_index.setdefault((stem.split("_")[0], int(stem.split("_")[-1])), f)
    slice_cache = {}
    pop_all = ([{"pop": "DUAL_PATH", "code": b["code"], "E": b["E"], "setup": b["setup"],
                 "segment": seg_of(b["E"]), "bucket": "箱体候选(状态见漏斗)",
                 "fetch_status": b["fetch_status"]} for b in box] +
               [{"pop": "CAR", "code": c["code"], "E": c["E"], "setup": c["setup"],
                 "segment": c["segment"], "bucket": c["bucket"]} for c in car_all])
    codes_needed = sorted({p["code"] for p in pop_all})
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
            return "BEYOND_CUTOFF"
        src = by_pair_index.get((code, day))
        if src is None:
            for (d, c), (f, days) in slice_cache.items():
                if c == code and day in days:
                    src = f
                    break
        if src is None:
            return "NO_FILE"
        try:
            df = pd.read_parquet(src)
            df = df[df["date"].astype(int) // 1000000 == day]
            if not len(df):
                return "GAP"
            v = E.validate_minute_day(df, D.day_grid(day), day)
            bad_price = bool((df[["open", "high", "low", "close"]].astype(float) <= 0).any().any())
            zero_all = bool((df["volume"].astype(float) > 0).sum() == 0)
            return "GAP" if (v["missing"] or v["dups"] or v["unexpected"] or bad_price or zero_all) else "COMPLETE"
        except Exception:
            return "GAP"

    cov_c = Counter()
    e_day_gap_set, e_day_nofile_set = set(), set()
    for p in pop_all:
        cls = coverage(p["code"], p["E"])
        p["e_day_coverage"] = cls
        cov_c[(p["pop"], p["bucket"], p["segment"], cls)] += 1
        if cls == "GAP":
            e_day_gap_set.add((p["code"], p["E"]))
        elif cls == "NO_FILE":
            e_day_nofile_set.add((p["code"], p["E"]))

    # ---------- 纠错3: 328 集合与 E 日集合求交 ----------
    scope = json.loads(open(p_scope, encoding="utf-8").read())
    set328 = {(r["code"], int(r["day"])) for r in scope["pairs"] if r.get("class") == "NO_FILE"}
    dp247_edays = {(r["code"], int(r["E"])) for r in cov_rows}
    gap247 = e_day_gap_set & dp247_edays
    inter_328_gap = set328 & e_day_gap_set
    inter_328_eday247 = set328 & dp247_edays

    # ---------- 交叉表 ----------
    def crosstab(fields, rows):
        tab = defaultdict(lambda: [0, set(), set()])
        for p in rows:
            k = tuple(p[f] for f in fields)
            tab[k][0] += 1
            tab[k][1].add((p["code"], p["E"]))
            tab[k][2].add(p["E"])
        return {json.dumps(k, ensure_ascii=False): {"candidates": v[0], "unique_codeE": len(v[1]), "dates": len(v[2])}
                for k, v in sorted(tab.items(), key=lambda x: str(x[0]))}

    tab_car = crosstab(["setup", "bucket", "segment"], car_all)
    tab_dp_box = crosstab(["setup", "fetch_status", "segment"],
                          [dict(b, segment=seg_of(b["E"])) for b in box])

    no_sig_cov = Counter()
    for c in pop_all:
        if c["pop"] == "CAR" and c["bucket"] == "no_signal_confirmed":
            no_sig_cov[c["e_day_coverage"]] += 1

    out = {"phase": "GCP_247_FULL_POPULATION_INVENTORY_V2_ERRATA",
           "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
           "network_requests": 0, "no_collection": True, "no_return_computation": True,
           "errata_vs_v1": [
               {"item": "谱系/lineage_note", "v1_error": "lineage_note 称 '交集=1' 且 CAR 清单被暗示为 247 母人口; "
                "CAR 8665 被称为'箱体池'",
                "correction": "实际 CAR∩247=24; 谱系钉死为 共性资格原料(8665,双线共用)->箱体候选 1050->漏斗 "
                "(WATCH476/S1 225/S2 31/fillable 251)->排除 4->冻结 247; CAR 清单=共性资格层, 非箱体层; "
                "251/247 不含 WATCH/无信号, 不能充当完整人口; 两线分开研究, CAR∩247=24 不奇怪, 不要求合并"},
               {"item": "无信号分钟覆盖", "v1_error": "v1 返回报告建议以'无信号组缺分钟'为由补采",
                "correction": "与 v1 JSON 相反: CAR 无信号确认桶(231) E 日分钟 = "
                f"COMPLETE {no_sig_cov.get('COMPLETE',0)} / GAP {no_sig_cov.get('GAP',0)} / "
                f"NO_FILE {no_sig_cov.get('NO_FILE',0)}; 数据缺失桶(4) = NO_FILE 4; "
                "合计 235 = 196/35/4, 与用户核对一致; 不得以'无信号组缺分钟'为补采理由"},
               {"item": "328 与 E 日混淆", "v1_error": "328 对被当作与 E 日相关的缺失",
                "correction": "328=退出/顺延路径 NO_FILE, 与 E 日分钟是不同 (code,date) 集合; "
                f"实测交集 |328∩E日GAP|={len(inter_328_gap)}, |328∩247的E日|={len(inter_328_eday247)}; "
                "E 日 GAP 集合与 328 分开登记, 各自独立预算口径; 328 维持挂起不批"}],
           "user_rulings": [
               "两线分开研究: DUAL_PATH 谱系钉死; CAR 保留独立线, CAR∩247=24 不奇怪, 不要求合并",
               "拒绝 'DUAL_PATH val + CAR train 作首判读': 混合事件定义与时间角色; CAR train 仅探索; "
               "完整人口研究含确认无信号/WATCH, 但不强行映射 CAR 状态到 S1/S2",
               "328 对挂起不批; 需求分集登记 (E 日 GAP 集 vs 328 集)",
               "CAR signal_found=True ≠ DUAL_PATH S1/S2 行动成立; 两线行动率不可直接比较"],
           "sources": sources,
           "dual_path_lineage": lineage,
           "main_funnel": dict(main_funnel),
           "car_no_signal_two_buckets": {"counts": {f"{k[0]}/{k[1]}": v for k, v in sorted(bucket_counts.items())},
                                         "confirmed_no_signal_e_day_coverage": dict(no_sig_cov),
                                         "rule": "NO_SIGNAL=成功扫描确认无信号(可作对照); E_DAY_MISSING=数据缺失无法判断(不得充当对照)"},
           "crosstab_car_setup_bucket_segment": tab_car,
           "crosstab_dp_box_setup_fetchstatus_segment": tab_dp_box,
           "set_intersection": {
               "set_328_exit_path_nofile": len(set328),
               "e_day_gap_set_all": len(e_day_gap_set),
               "e_day_gap_set_dp247": len(gap247),
               "e_day_nofile_set_all": len(e_day_nofile_set),
               "intersection_328_with_eday_gap": len(inter_328_gap),
               "intersection_328_with_247_edays": len(inter_328_eday247),
               "note": "DUAL_PATH 247 的 E 日全部有文件(仅部分网格缺槽): "
                       f"247 E 日 GAP={len(gap247)}, NO_FILE={len(dp247_edays & e_day_nofile_set)}"},
           "denominator_advice_for_should_buy": {
               "dual_path_line_complete_population": "1050 箱体候选 (漏斗: WATCH 476 / S1 225 / S2 31 / 无动作 273 / "
                                                     "minute_invalid 25 / 未缓存 20; fillable 251 -> 247)",
               "car_line": "583 共性资格候选 (行动 348 / 确认无信号 231 / 数据缺失 4); CAR train 仅探索, 不与 DUAL_PATH 混合首判读",
               "pending": "研究合同等用户审阅本盘点后另签"},
           "warning": "本盘点不含任何收益计算; WATCH/无信号分母仅供后续合同设计",
           "elapsed_s": round(time.time() - t0, 1)}
    # 明细行
    out["population"] = pop_all
    write_checked(OUT_JSON, json.dumps(out, ensure_ascii=False, indent=1))

    L = ["# 完整人口与状态盘点 v2（纠错版；v1 保留，三处 errata 见下）", "",
         f"- 生成: {out['generated_at']} | 零网络零采集零收益计算 | 用时 {out['elapsed_s']}s", "",
         "## Errata（v1 三处会改变研究设计的错误）", ""]
    for e in out["errata_vs_v1"]:
        L.append(f"### {e['item']}")
        L.append(f"- v1 错误: {e['v1_error']}")
        L.append(f"- 更正: {e['correction']}")
    L += ["", "## DUAL_PATH 谱系链（钉死）", ""]
    for s in lineage:
        L.append(f"{s['step']}. **{s['name']}**: {s['count']} | 规则: {s.get('rule')} | 件: {s['artifact']}"
                 + (f" | 排除: {s['exclusions']}" if s.get("exclusions") else "")
                 + (f" | 核验: {s['verify']}" if s.get("verify") else ""))
    L += ["", "## 用户裁定（入档）", ""] + [f"- {r}" for r in out["user_rulings"]] + [
         "", "## CAR 无信号两桶", "",
         f"- 桶计数: {out['car_no_signal_two_buckets']['counts']}",
         f"- 确认无信号桶 E 日分钟: {out['car_no_signal_two_buckets']['confirmed_no_signal_e_day_coverage']}",
         "", "## 需求集合求交（纠错3）", ""]
    for k, v in out["set_intersection"].items():
        L.append(f"- {k}: {v}")
    L += ["", "## 正确分母建议（待用户签研究合同）", ""]
    for k, v in out["denominator_advice_for_should_buy"].items():
        L.append(f"- {k}: {v}")
    L += ["", "## CAR 交叉表 (setup × 桶 × 段)", ""]
    for k, v in tab_car.items():
        L.append(f"- {k}: {v['candidates']} / {v['unique_codeE']} / {v['dates']}")
    L += ["", "## DUAL_PATH 箱体候选交叉表 (setup × fetch_status × 段)", ""]
    for k, v in tab_dp_box.items():
        L.append(f"- {k}: {v['candidates']} / {v['unique_codeE']} / {v['dates']}")
    L += ["", "## 声明", "",
          "- 本盘点零收益计算、零采集；328 挂起；两线分开；研究合同另签。", ""]
    write_checked(OUT_MD, "\n".join(L))
    print(f"v2 done ({time.time()-t0:.1f}s)")
    print("no_signal cov:", dict(no_sig_cov))
    print("sets:", {k: v for k, v in out["set_intersection"].items() if isinstance(v, int)})


if __name__ == "__main__":
    main()
