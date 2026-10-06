# -*- coding: utf-8 -*-
"""完整人口与状态盘点 v3 — 互斥状态 + 人口层 ETF/BSE 排除 + 覆盖双口径 (零网络核对修正).

修正1 互斥状态: WATCH/S1/S2 是独立标志非互斥类别. 对 1,050 箱体人口(人口层排除 ETF/BSE 后)
按冻结规则(scan_views, probe 同码)逐候选生成四类互斥状态:
  1 有行动(记首次 S1/S2; 双触发归属与记录) / 2 WATCH但无行动 / 3 未WATCH且无行动 / 4 数据不足无法判断.
  保留 WATCH/S1/S2 独立标志供审计; 逐行计数并核验总和=格人口;
  去重行动数 vs S1+S2 并集核验 (probe 冻结口径 254 vs 256, 双触发 2 条逐条列;
  v3 为 p2b1 采集后缓存实况重算, 与 probe 预采集聚合的差异如实解释).
修正2 ETF/BSE 排除发生在人口层 (非行动层): 命名空间规则 (BSE ^[48]|^920; ETF ^15|^51|^52|^56|^58),
  全量排查 1,050, 登记排除清单, 重算各格分母. (4 条已知含于其中)
修正3 覆盖双口径: 行动识别覆盖 (E 日分钟, 逐互斥状态格) vs 收益路径覆盖 (E+1..E+5,
  仅登记不 replay, 有 E 日分钟 ≠ 有收益路径覆盖); 两口径不得混用;
  "不依赖补采即可研究"仅限行动识别层.
用户判读限定写入"研究设计警示": ①"整天没行动"收盘后才可确定, 不能当早盘已知买入条件,
  未来合同应比较两套可执行规则, 不得把行动组与整日无行动组的均值差解释成放行效果;
  ②"没行动"对照必须区分第 2 类(WATCH 无行动)与第 3 类(未 WATCH 无行动), 时间可判定性不同.
零网络零采集零收益计算零训练; 产物 v3 json/md (v1/v2 保留); 审计 -13.
"""
from __future__ import annotations
import sys, json, time, hashlib
from pathlib import Path
from collections import Counter, defaultdict

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "step9_preflight"))
import gcp_dual_path_coverage_probe as PROBE
import dual_path_execution_v0_1_1 as D
import gcp_round_b_engine as E

BASE = HERE / "gcp_round_b_small_sample_v1"
OUT_JSON = BASE / "gcp_247_full_population_inventory_v3.json"
OUT_MD = BASE / "gcp_247_full_population_inventory_v3.md"
DATA_CUTOFF = D.DATA_CUTOFF
BOUNDS = {"train_max": 20260305, "val_max": 20260609}


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


def is_excluded_instrument(code: str) -> str | None:
    if code[0] in "48" or code.startswith("92"):
        return "BSE(北交所)"
    if code[:2] in ("15", "51", "52", "56", "58"):
        return "ETF"
    return None


def main():
    t0 = time.time()
    p_fetch = HERE / "contracts" / "dual_path_e_day_fetch_list.txt"
    p_probe = BASE / "dual_path_coverage_probe.json"
    p_251 = HERE / "_251_events_cache.json"
    p_cov = BASE / "gcp_247_coverage_verify.json"
    probe = json.loads(open(p_probe, encoding="utf-8").read())
    k251 = {(e["code"], int(e["E"])) for e in json.loads(open(p_251, encoding="utf-8").read())}
    k247 = {(r["code"], int(r["E"])) for r in json.loads(open(p_cov, encoding="utf-8").read())["rows"]}

    fl = [l.strip().split(",") for l in open(p_fetch, encoding="utf-8") if l.strip()]
    box = [{"code": p[0], "E": int(p[1]), "P": int(p[2]), "setup": p[3], "fetch_status": p[4]} for p in fl]

    # ---------- 修正2: 人口层 ETF/BSE 排除 ----------
    excluded, population = [], []
    for b in box:
        why = is_excluded_instrument(b["code"])
        if why:
            excluded.append({**b, "reason": why, "in_251": (b["code"], b["E"]) in k251})
        else:
            population.append(b)
    known4 = {("515070", 20260915), ("588200", 20260915), ("920906", 20250325), ("920961", 20251211)}
    excl_keys = {(x["code"], x["E"]) for x in excluded}
    assert known4 <= excl_keys, "已知 4 条未全部命中排除规则"

    # ---------- 数据装载 (probe 同管线) ----------
    panel = PROBE.get_daily_parquet()
    panel["code"] = panel["code"].astype(str).str.zfill(6)
    panel["date"] = panel["date"].astype(int)
    cal = sorted(int(d) for d in panel["date"].unique())
    cal_idx = {d: i for i, d in enumerate(cal)}
    daily_by_code = {c: g.sort_values("date").reset_index(drop=True)
                     for c, g in panel.groupby("code")}

    # 广义分钟索引 (全部已登记目录 + p2b1/deferral/realtest by_pair + code 切片)
    minute_idx = {}
    for d in [BASE / "minute_collect_realtest" / "by_pair",
              BASE / "deferral_collect_v2" / "by_pair",
              BASE / "p2b1_collect" / "by_pair"]:
        for f in d.glob("*.parquet"):
            stem = f.stem
            if "_" in stem and stem.split("_")[-1].isdigit():
                minute_idx.setdefault((stem.split("_")[0], int(stem.split("_")[-1])), str(f))
    slice_days = defaultdict(list)   # code -> [(file, days)] 全目录保留
    codes_needed = sorted({b["code"] for b in population})
    for d in [BASE / "minute_collect_realtest", BASE / "minute_ext", BASE / "minute", BASE / "minute_car",
              HERE.parent / "pullback_train" / "minute_cache_clean"]:
        for code in codes_needed:
            f = d / f"{code}.parquet"
            if f.exists():
                try:
                    df = pd.read_parquet(f, columns=["date"])
                    slice_days[code].append((str(f), set(int(x) // 1000000 for x in df["date"].astype(int))))
                except Exception:
                    pass

    def load_day(code, day):
        f = minute_idx.get((code, day))
        if f is None:
            for sf, days in slice_days.get(code, []):
                if day in days:
                    f = sf
                    break
        if f is None:
            return None, None
        try:
            df = pd.read_parquet(f)
            df = df[df["date"].astype(int) // 1000000 == day].sort_values("date").reset_index(drop=True)
            return df, f
        except Exception:
            return None, f

    def day_class(code, day):
        """COMPLETE / GAP / NO_FILE / BEYOND_CUTOFF (严格网格口径, 同 v0.1.1)."""
        if day > DATA_CUTOFF:
            return "BEYOND_CUTOFF"
        df, _ = load_day(code, day)
        if df is None or not len(df):
            return "NO_FILE"
        v = E.validate_minute_day(df, D.day_grid(day), day)
        bad_price = bool((df[["open", "high", "low", "close"]].astype(float) <= 0).any().any())
        zero_all = bool((df["volume"].astype(float) > 0).sum() == 0)
        return "COMPLETE" if not (v["missing"] or v["dups"] or v["unexpected"] or bad_price or zero_all) else "GAP"

    # ---------- 修正1: 逐候选互斥状态 ----------
    recs = []
    for b in population:
        code, Pd, Ee, setup = b["code"], b["P"], b["E"], b["setup"]
        seg = seg_of(Ee)
        dc = daily_by_code.get(code)
        rec = {"code": code, "E": Ee, "P": Pd, "setup": setup, "segment": seg,
               "in_251": (code, Ee) in k251, "in_247": (code, Ee) in k247}
        # 箱体/指标评估 (probe 同函数)
        loc = int(dc["date"].searchsorted(Pd, side="right")) if dc is not None else 0
        if dc is None or loc < 20:
            rec.update(mutual="4_数据不足无法判断", reason="日线历史不足20日", watch=None, s1_ts=None, s2_ts=None)
            recs.append(rec)
            continue
        d_sub = dc.iloc[:loc]
        atr_ref = PROBE.E.build_atr_ref(d_sub)
        calibs = PROBE.evaluate_calibers(d_sub, atr_ref)
        cb = calibs.get("main", {})
        if not cb.get("valid"):
            rec.update(mutual="4_数据不足无法判断", reason="main口径箱体无效", watch=None, s1_ts=None, s2_ts=None)
            recs.append(rec)
            continue
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
        # E 日分钟 (行动识别覆盖)
        e_cls = day_class(code, Ee)
        rec["e_day_coverage"] = e_cls
        if e_cls != "COMPLETE":
            reason = {"GAP": "E日分钟缺槽/非法", "NO_FILE": "E日无分钟文件",
                      "BEYOND_CUTOFF": "E日超数据截止"}.get(e_cls, e_cls)
            rec.update(mutual="4_数据不足无法判断", reason=reason, watch=None, s1_ts=None, s2_ts=None)
            recs.append(rec)
            continue
        e_df, _ = load_day(code, Ee)
        views = PROBE.scan_views(e_df, cb["upper"], cb["lower"], atr_ref, c_ok, v_ok)
        watch = bool(views["watch"])
        s1, s2 = views["s1_ts"], views["s2_ts"]
        rec.update(watch=watch, s1_ts=s1, s2_ts=s2)
        if s1 is not None or s2 is not None:
            first = min(x for x in (s1, s2) if x is not None)
            both = (s1 is not None) and (s2 is not None)
            rec.update(mutual="1_有行动", reason=("双触发S1+S2" if both else ("S1" if s1 is not None else "S2")),
                       first_action=first, double_trigger=both)
        elif watch:
            rec.update(mutual="2_WATCH但无行动", reason=None)
        else:
            rec.update(mutual="3_未WATCH且无行动", reason=None)
        # 修正3b: 收益路径覆盖 (仅登记, 不 replay)
        i_e = cal_idx.get(Ee)
        path = {}
        for off in (1, 2, 3, 5):
            d = cal[i_e + off] if (i_e is not None and i_e + off < len(cal)) else None
            path[f"E+{off}"] = day_class(code, d) if d else "BEYOND_CALENDAR"
        rec["exit_path_coverage"] = path
        recs.append(rec)

    # 无日线/箱体/E日问题行的 exit path 不登记 (本身 class4)
    for r in recs:
        r.setdefault("e_day_coverage", None)
        r.setdefault("exit_path_coverage", None)

    # ---------- 交叉表 + 核验 ----------
    def cell(key_fields):
        tab = defaultdict(lambda: {"n": 0, "codeE": set(), "dates": set()})
        for r in recs:
            k = tuple(r[f] for f in key_fields)
            tab[k]["n"] += 1
            tab[k]["codeE"].add((r["code"], r["E"]))
            tab[k]["dates"].add(r["E"])
        return {json.dumps(k, ensure_ascii=False): {"n": v["n"], "unique_codeE": len(v["codeE"]),
                                                    "dates": len(v["dates"])}
                for k, v in sorted(tab.items(), key=lambda x: str(x[0]))}

    tab_main = cell(["mutual", "segment", "setup"])
    # 总和核验: 每格 (segment,setup) 四互斥类之和 = 该格人口
    pop_by_cell = defaultdict(int)
    for r in recs:
        pop_by_cell[(r["segment"], r["setup"])] += 1
    sum_check = {}
    for (seg, setup), total in sorted(pop_by_cell.items()):
        s = sum(v["n"] for k, v in tab_main.items()
                if json.loads(k)[1] == seg and json.loads(k)[2] == setup)
        sum_check[f"{seg}/{setup}"] = {"population": total, "mutual_sum": s, "ok": s == total}

    n1 = sum(1 for r in recs if r["mutual"] == "1_有行动")
    s1_n = sum(1 for r in recs if r.get("s1_ts") is not None)
    s2_n = sum(1 for r in recs if r.get("s2_ts") is not None)
    doubles = [{"code": r["code"], "E": r["E"], "segment": r["segment"], "setup": r["setup"],
                "s1_ts": r["s1_ts"], "s2_ts": r["s2_ts"]} for r in recs if r.get("double_trigger")]

    # ---------- 修正3a: 行动识别覆盖 (逐互斥状态格) ----------
    cov_tab = cell(["mutual", "e_day_coverage"])
    # 修正3b: 收益路径覆盖 (class1 格)
    path_tab = defaultdict(lambda: Counter())
    for r in recs:
        if r["mutual"] == "1_有行动" and r.get("exit_path_coverage"):
            p = r["exit_path_coverage"]
            full_h3 = all(p[f"E+{o}"] == "COMPLETE" for o in (1, 2, 3))
            full_h5 = all(p[f"E+{o}"] == "COMPLETE" for o in (1, 2, 3, 5))
            key = f"{r['segment']}/{r['setup']}"
            path_tab[key]["n_action"] += 1
            path_tab[key]["Eplus123_complete" if full_h3 else "Eplus123_incomplete"] += 1
            path_tab[key]["Eplus12345_complete" if full_h5 else "Eplus12345_incomplete"] += 1

    v3_vs_probe = {
        "probe_frozen_pre_p2b1": {"action_unique": 254, "s1": 225, "s2": 31, "watch": 476,
                                  "minute_valid": 1005,
                                  "note": "probe 聚合冻结于 p2b1 采集前; 口径含 missing<=2 容忍与 242 平网格"},
        "v3_recompute_post_p2b1": {"action_unique": n1, "s1": s1_n, "s2": s2_n,
                                   "watch": sum(1 for r in recc if recc.get("watch")) if False else
                                   sum(1 for r in recs if r.get("watch")),
                                   "minute_strict_complete": sum(1 for r in recs if r.get("e_day_coverage") == "COMPLETE"),
                                   "note": "v3 按现有缓存实况重算; E 日采用严格网格口径 (v0.1.1 语义), "
                                           "GAP/缺文件日入第 4 类; 与 probe 差异来源=p2b1 新缓存+口径从严, 如实并列"},
        "double_trigger_rule_check": {"s1_plus_s2": s1_n + s2_n, "union": n1,
                                      "double_count": s1_n + s2_n - n1, "rows": doubles}}

    out = {"phase": "GCP_247_FULL_POPULATION_INVENTORY_V3_MUTEX",
           "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
           "network_requests": 0, "no_collection": True, "no_return_computation": True, "set_328_suspended": True,
           "errata_vs_v2": [
               {"item": "互斥状态算法", "v2_error": "把 WATCH/S1/S2 当互斥类, 用 1005-476-225-31=273 推算'无动作'",
                "correction": "三者是独立标志; v3 逐候选按冻结规则生成四类互斥状态并保留独立标志; "
                              f"合法分钟无行动 = strict_complete({v3_vs_probe['v3_recompute_post_p2b1']['minute_strict_complete']})"
                              f" - 行动({n1}) 逐行核; probe 口径 1005-254=751"},
               {"item": "ETF/BSE 排除层级", "v2_error": "4 条 ETF/BSE 排除发生在行动子集层",
                "correction": f"排除上移至人口层: 1,050 全量按命名空间规则排查, 排除 {len(excluded)} 条"
                              "(含已知 4 条), 各格分母重算"},
               {"item": "覆盖口径混用", "v2_error": "E 日覆盖与收益路径覆盖未区分",
                "correction": "行动识别覆盖(E日)与收益路径覆盖(E+1..E+5)双口径分开登记; "
                              "有 E 日分钟 != 有收益路径覆盖; '不依赖补采即可研究'仅限行动识别层"}],
           "sources": {
               "fetch_list_1050": {"file": str(p_fetch), "sha256": sha256_file(p_fetch)},
               "probe_funnel": {"file": str(p_probe), "sha256": sha256_file(p_probe)},
               "frozen_247": {"file": str(p_cov), "sha256": sha256_file(p_cov)},
               "scan_engine": {"module": "gcp_dual_path_coverage_probe.scan_views/evaluate_calibers (冻结规则同码复用)"}},
           "instrument_exclusion_population_layer": {
               "rule": "BSE: ^[48]|^920 ; ETF: ^15|^51|^52|^56|^58 (命名空间规则 CN_A_SHARE_CODE_NAMESPACE_2026)",
               "excluded_n": len(excluded), "excluded_list": excluded,
               "known4_covered": sorted(known4) <= sorted(excl_keys)},
           "study_population_n": len(recs),
           "mutual_crosstab": tab_main,
           "mutual_sum_check": sum_check,
           "action_identification_coverage_by_cell": cov_tab,
           "exit_path_coverage_by_cell": {k: dict(v) for k, v in sorted(path_tab.items())},
           "v3_vs_probe": v3_vs_probe,
           "mutex_recompute_273_to_751": {
               "v2_wrong_no_action": 273,
               "v3_correct_no_action_class23": sum(1 for r in recs if r["mutual"] in ("2_WATCH但无行动", "3_未WATCH且无行动")),
               "decomposition": {
                   "class2_watch_no_action": sum(1 for r in recs if r["mutual"] == "2_WATCH但无行动"),
                   "class3_no_watch_no_action": sum(1 for r in recs if r["mutual"] == "3_未WATCH且无行动"),
                   "double_trigger_recount": len(doubles),
                   "class4_data_insufficient": sum(1 for r in recs if r["mutual"] == "4_数据不足无法判断"),
                   "note": "273->751 口径差 = 互斥算法纠正(同日可既 WATCH 又行动) + 严格网格口径(p2b1 后) + "
                           "人口层排除; probe 冻结口径无行动=1005-254=751 与 v3 第2+3类同构可比"}},
           "research_design_warnings": [
               "'整天没行动'是收盘后才可确定的状态, 不能当作早盘已知买入条件; 未来合同应比较两套可执行规则"
               "(如'早盘可判定规则 A vs B'), 不得把行动组与整日无行动组的均值差解释成放行效果",
               "行动状态互斥化后, 1,050 人口的'没行动'对照必须区分第 2 类(WATCH 无行动)与第 3 类(未 WATCH 无行动), "
               "两者时间可判定性不同"],
           "denominator_advice_for_executable_comparison": {
               "line": "DUAL_PATH 独立线 (1,050 人口层排除后 %d 为研究人口)" % len(recs),
               "class1": "有行动 (可执行规则 A 的落点; 首触发 S1/S2 逐条在案)",
               "class2_vs_3": "第 2 类(WATCH 无行动, 盘中可判定 WATCH) 与 第 3 类(未 WATCH, 早盘即知无行动)"
                              " 是两套不同时间可判定性的对照, 合同须分开",
               "class4": "数据不足无法判断 (不得混入任何对照组)",
               "coverage_note": "行动识别层多数格不依赖补采即可研究; 收益比较层的 E+1..E+5 缺口如实单列(见 exit_path_coverage)",
               "pending": "待用户审 v3 后起草 DUAL_PATH 独立研究合同"},
           "warning": "零收益计算; WATCH/S1/S2 独立标志与互斥状态并存供审计",
           "elapsed_s": round(time.time() - t0, 1)}
    out["records"] = recs
    write_checked(OUT_JSON, json.dumps(out, ensure_ascii=False, indent=1))

    L = ["# 完整人口与状态盘点 v3（互斥状态纠错版；v1/v2 保留）", "",
         f"- 生成: {out['generated_at']} | 零网络零采集零收益计算 | 用时 {out['elapsed_s']}s", "",
         "## Errata（v3 对 v2 的三处修正）", ""]
    for e in out["errata_vs_v2"]:
        L.append(f"### {e['item']}")
        L.append(f"- v2 错误: {e['v2_error']}")
        L.append(f"- 更正: {e['correction']}")
    L += ["", "## 人口层 ETF/BSE 排除（修正2）", "",
          f"- 规则: {out['instrument_exclusion_population_layer']['rule']}",
          f"- 1,050 中排除 {len(excluded)} 条 (含已知 4 条 ETF/BSE, 全部命中): "
          + ", ".join(f"{x['code']}@{x['E']}({x['reason']}{',在251' if x['in_251'] else ''})" for x in excluded),
          f"- 研究人口 = {len(recs)}", "",
          "## 互斥状态交叉表 (×段×池) + 总和核验", ""]
    for k, v in tab_main.items():
        L.append(f"- {k}: {v['n']} / {v['unique_codeE']} / {v['dates']}")
    bad = [k for k, v in sum_check.items() if not v["ok"]]
    L += ["", f"- 总和核验: {'全部格通过 (互斥类之和=格人口)' if not bad else 'FAIL ' + str(bad)}", "",
          "## 行动数并集核验 (probe 254 vs 256)", "",
          f"- probe(冻结, p2b1 前): action 254 = S1 225 + S2 31 − 双触发 2",
          f"- v3(现缓存重算): action {n1} = S1 {s1_n} + S2 {s2_n} − 双触发 {s1_n + s2_n - n1}"]
    for dbl in doubles:
        L.append(f"  - 双触发: {dbl['code']}@{dbl['E']} ({dbl['segment']}/{dbl['setup']}) s1={dbl['s1_ts']} s2={dbl['s2_ts']}")
    L += ["", "## 273→751 构成分解", "",
          f"- {json.dumps(out['mutex_recompute_273_to_751']['decomposition'], ensure_ascii=False)}", "",
          "## 覆盖双口径（修正3）", "",
          "### 行动识别覆盖 (E 日分钟, 逐互斥状态格)", ""]
    for k, v in cov_tab.items():
        L.append(f"- {k}: {v['n']}")
    L += ["", "### 收益路径覆盖 (E+1..E+5, 仅登记未 replay; 有 E 日分钟≠有收益路径覆盖)", ""]
    for k, v in sorted(path_tab.items()):
        L.append(f"- {k}: {v}")
    L += ["", "## 研究设计警示（用户判读限定，入档）", ""] + [f"- {w}" for w in out["research_design_warnings"]] + [
         "", "## 对'可执行规则对照'合同的分母建议", ""]
    for k, v in out["denominator_advice_for_executable_comparison"].items():
        L.append(f"- {k}: {v}")
    L += ["", "## 声明", "",
          "- 328 挂起; 零收益计算; 两线分开 (CAR 线见 v2 未变); 研究合同待用户审 v3 后起草.", ""]
    write_checked(OUT_MD, "\n".join(L))
    print(f"v3 done ({time.time()-t0:.1f}s)")
    print("excluded:", len(excluded), "| population:", len(recs))
    print("mutex:", Counter(r["mutual"] for r in recs))
    print("sum check bad:", bad)
    print("doubles:", doubles)


if __name__ == "__main__":
    main()
