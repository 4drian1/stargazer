# -*- coding: utf-8 -*-
"""第三步 c: 拟合就绪有界核验 (零网络/零采集/零拟合).

依据用户 2026-10-08 裁定:
1. 32 只冲突码继续隔离, 不修复全部历史数据作为训练前置.
2. 人口厘清: 214 = H3+H5 轨道数; H3 主研究=124 事件 (train 77/val 27/test 20);
   H5=90 事件, 仅辅助, 不能救 H3.
3. 旧 80% 覆盖门未过: train 执行入口 226, 准入 H3 train 77 -> 34%; 本轮显式接受此覆盖限制,
   定位为小样本研究性拟合, 不声称继承旧门且达标.
4. 有界 RAW 尺度核验: 准入事件 (code,E日) 的本地分钟末bar收盘 vs 日线 close, 不一致即缺尺度证据.
5. 白名单四字段覆盖: contraction/vol_cv10 (P及以前日线, 决策时可见) + setup + branch; 缺失率+不变性.
6. 输出就绪件后停止, 拟合须单独授权.
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "step9_preflight"))
sys.path.insert(0, str(HERE))

import gcp_dual_path_coverage_probe as PROBE  # noqa: E402
import dual_path_execution_v0_1_1 as D  # noqa: E402
import gcp_round_b_engine as E  # noqa: E402

RUN = HERE / "gcp_round_b_small_sample_v1"
EX_PATH = RUN / "r1_exec_labels_v0_1_2.json"
ADM_PATH = RUN / "r1_label_admission_v0_1_2.json"
AUDIT_LOG = HERE / "gcp_audit_log.md"
OUT_JSON = RUN / "r1_fit_readiness_v0_1_1.json"
OUT_MD = RUN / "r1_fit_readiness_report_v0_1_1.md"

CAP_NEW_BYTES = 1 * 1024 * 1024
CAP_BASE_BYTES = 55 * 1024 * 1024


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def dir_size(p: Path) -> int:
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file())


def main():
    t0 = time.time()
    print("=" * 70)
    print("第三步 c: 拟合就绪有界核验 (零网络/零采集/零拟合)")
    print("=" * 70)

    EX = json.loads(EX_PATH.read_text(encoding="utf-8"))
    ADM = json.loads(ADM_PATH.read_text(encoding="utf-8"))
    records = EX["records"]
    td = {f"{x['code']}|{x['E']}|{x['track']}": x for x in ADM["track_decisions"]}

    cal = D.load_calendar()
    cal_pos = {d: i for i, d in enumerate(cal)}

    panel = PROBE.get_daily_parquet()
    panel["code"] = panel["code"].astype(str).str.zfill(6)
    panel["date"] = panel["date"].astype(int)
    gby = {c: g.sort_values("date").drop_duplicates("date")
           for c, g in panel.groupby("code")}

    # ---------- 人口厘清 ----------
    n_input = EX["aggregates"]["n_input"]                # 489
    train_exec = EX["aggregates"]["by_segment"]["train"]["n"]  # 226
    # 准入事件 (来自收口件)
    adm_h3 = [x for x in ADM["track_decisions"] if x["track"] == "H3" and x["admitted"]]
    adm_h5 = [x for x in ADM["track_decisions"] if x["track"] == "H5" and x["admitted"]]
    n_tracks = len(adm_h3) + len(adm_h5)
    h3_by_seg = Counter(x["segment"] for x in adm_h3)
    h5_by_seg = Counter(x["segment"] for x in adm_h5)
    h3_train = h3_by_seg["train"]
    coverage = h3_train / train_exec
    print(f"输入 {n_input} | train 执行入口 {train_exec} | 准入 H3 事件 {len(adm_h3)} "
          f"({dict(h3_by_seg)}) / H5 {len(adm_h5)} / 轨道 {n_tracks}")
    print(f"train H3 覆盖 = {h3_train}/{train_exec} = {coverage:.1%}  (旧80%门未过, 研究性接受)")

    # ---------- RAW 尺度有界核验 ----------
    print("\n--- RAW 尺度核验: 本地分钟末bar收盘 vs 日线 close (E日) ---")
    bp_dir = D.MIN_DIRS[0]
    code_dirs = D.MIN_DIRS[1:]
    bp_index = {p.stem: p for p in bp_dir.glob("*.parquet")} if bp_dir.exists() else {}

    def minute_close(code, day):
        pf = bp_index.get(f"{code}_{day}")
        df = None
        if pf is not None:
            try:
                df = pd.read_parquet(pf)
            except Exception:
                df = None
        if df is None:
            for d in code_dirs:
                cf = d / f"{code}.parquet"
                if not cf.exists():
                    continue
                try:
                    full = pd.read_parquet(cf, columns=["date"])
                    if day in set(int(x) // 1000000 for x in full["date"].tolist()):
                        df = pd.read_parquet(cf)
                        df = df[df["date"].astype(int) // 1000000 == day]
                        break
                except Exception:
                    continue
        if df is None or len(df) == 0:
            return None
        df = df.sort_values("date")
        return float(df["close"].iloc[-1])

    scale_results = {"H3": {}, "H5": {}}
    scale_fail = []
    for tk_label, adm in (("H3", adm_h3), ("H5", adm_h5)):
        cnt = {"checked": 0, "match": 0, "mismatch": 0, "no_minute_file": 0,
               "le_day_unavailable": 0}
        for x in adm:
            code, Eday = x["code"], int(x["E"])
            mc = minute_close(code, Eday)
            g = gby.get(code)
            if g is None:
                cnt["no_minute_file"] += 1
                continue
            dc = dict(zip(g["date"].astype(int), g["close"].astype(float)))
            if mc is None:
                cnt["no_minute_file"] += 1
                continue
            dclose = dc.get(Eday)
            if dclose is None:
                cnt["no_minute_file"] += 1
                continue
            tol = max(0.01, 0.005 * dclose)
            cnt["checked"] += 1
            if abs(mc - dclose) > tol:
                cnt["mismatch"] += 1
                scale_fail.append({"code": code, "E": Eday, "track": tk_label,
                                   "minute_close": round(mc, 4),
                                   "daily_close": round(dclose, 4)})
            else:
                cnt["match"] += 1
            # le-day 仅记录可得性
            le_day = int(x["le"]) // 1000000
            if minute_close(code, le_day) is None:
                cnt["le_day_unavailable"] += 1
        scale_results[tk_label] = cnt
    print("尺度核验:", scale_results)
    print(f"H3+H5 尺度不一致 (建议隔离证据不足): {len(scale_fail)}",
          scale_fail[:8])

    # ---------- 白名单覆盖 + 不变性 ----------
    print("\n--- 白名单四字段覆盖 (contraction/vol_cv10/setup/branch) ---")

    def whitelist(code, Eday):
        P = cal[cal_pos[Eday] - 1] if cal_pos[Eday] > 0 else None
        g = gby.get(code)
        if g is None or P is None:
            return {"contraction": None, "vol_cv10": None, "setup": None, "branch": None}
        d_sub = g[g["date"].astype(int) <= P]
        if len(d_sub) < 20:
            return {"contraction": None, "vol_cv10": None, "setup": None, "branch": None}
        atr_ref = E.build_atr_ref(d_sub)
        h = d_sub["high"].astype(float).to_numpy()
        l = d_sub["low"].astype(float).to_numpy()
        c = d_sub["close"].astype(float).to_numpy()
        pc = np.concatenate([[c[0]], c[:-1]])
        tr = np.maximum(h - l, np.maximum(np.abs(h - pc), np.abs(l - pc)))
        atr14 = pd.Series(tr).rolling(14).mean().to_numpy()
        amed60 = float(np.nanmedian(atr14[-60:])) if len(atr14) >= 14 else np.nan
        contraction = atr_ref / amed60 if (np.isfinite(amed60) and amed60 > 0) else np.nan
        vol = d_sub["volume"].astype(float).to_numpy()
        last10 = vol[-10:]
        cv = float(np.std(last10, ddof=1) / np.mean(last10)) if np.mean(last10) > 0 else np.nan
        return {"contraction": float(contraction) if np.isfinite(contraction) else None,
                "vol_cv10": cv if np.isfinite(cv) else None}

    wl = {"H3": {"contraction": 0, "vol_cv10": 0, "setup": 0, "branch": 0,
                 "n": 0, "missing": {"contraction": 0, "vol_cv10": 0}},
          "H5": {"contraction": 0, "vol_cv10": 0, "setup": 0, "branch": 0,
                 "n": 0, "missing": {"contraction": 0, "vol_cv10": 0}}}
    for tk_label, adm in (("H3", adm_h3), ("H5", adm_h5)):
        for x in adm:
            wl[tk_label]["n"] += 1
            rec = next(r for r in records
                       if r["code"] == x["code"] and r["E"] == x["E"]
                       and r["branch"] == x["branch"])
            w = whitelist(x["code"], int(x["E"]))
            for fld in ("contraction", "vol_cv10", "setup", "branch"):
                if fld in ("setup", "branch"):
                    if rec.get(fld):
                        wl[tk_label][fld] += 1
                else:
                    if w[fld] is not None:
                        wl[tk_label][fld] += 1
                    else:
                        wl[tk_label]["missing"][fld] += 1
    miss_rate = {}
    for tk in ("H3", "H5"):
        n = wl[tk]["n"]
        miss_rate[tk] = {f: (wl[tk]["missing"][f] / n if n else 0)
                         for f in ("contraction", "vol_cv10")}
    print("白名单覆盖:", {tk: {k: wl[tk][k] for k in ("contraction", "vol_cv10", "setup", "branch", "n")}
                          for tk in ("H3", "H5")})
    print("缺失率 (contraction/vol_cv10):", miss_rate)

    # 不变性抽检: 扰动 P 后日线, 重算 contraction/vol_cv10 应不变
    rng = np.random.default_rng(20261008)
    sample = list(rng.choice(adm_h3, size=min(10, len(adm_h3)), replace=False))
    inv_fail = 0
    for x in sample:
        code, Eday = x["code"], int(x["E"])
        g = gby[code].copy()
        w0 = whitelist(code, Eday)
        P = cal[cal_pos[Eday] - 1]
        mask = g["date"].astype(int) > P
        g.loc[mask, "close"] = g.loc[mask, "close"] * 1.5
        g.loc[mask, "volume"] = g.loc[mask, "volume"] * 2.0
        gby[code] = g
        w1 = whitelist(code, Eday)
        if (w1["contraction"] != w0["contraction"]) or (w1["vol_cv10"] != w0["vol_cv10"]):
            inv_fail += 1
        gby[code] = panel[panel["code"] == code].sort_values("date").drop_duplicates("date")
    print(f"不变性抽检 {len(sample)} 条, 扰动后特征变化: {inv_fail} (期望 0)")

    # ---------- 输出 ----------
    base_before = dir_size(RUN)
    OUT = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "network_requests": 0, "training_performed": False, "data_collection_performed": False,
        "population_clarification": {
            "note": "214 = H3+H5 轨道数, 非独立训练样本; H3 主研究=124事件, H5=90事件仅辅助",
            "n_input": n_input,
            "train_execution_entries": train_exec,
            "admitted_H3_events": len(adm_h3),
            "admitted_H3_by_segment": dict(h3_by_seg),
            "admitted_H5_events": len(adm_h5),
            "admitted_H5_by_segment": dict(h5_by_seg),
            "admitted_tracks": n_tracks,
        },
        "coverage_gate": {
            "old_gate": "ACTION_FILTER_MODEL_RESEARCH_CONTRACT_V1 §5: 标签覆盖>=80% (分母=train全部触发事件)",
            "denominator_train_exec_entries": train_exec,
            "admitted_H3_train": h3_train,
            "coverage": round(coverage, 4),
            "gate_passed": False,
            "acceptance": "显式接受覆盖限制, 定位小样本研究性拟合, 不声称继承旧门且达标",
        },
        "raw_scale_check": {
            "method": "本地分钟末bar收盘 vs 日线close(E日), tol=max(0.01,0.005*price) — 仅验证 E 日收盘尺度一致",
            "by_track": scale_results,
            "mismatch_events": scale_fail,
            "recommendation": "仅验证 E 日收盘尺度一致; 尺度不一致事件缺该日同尺度证据, 拟合前继续隔离. 注意: 这不证明全部历史日线或整条持仓路径同尺度",
        },
        "whitelist_coverage": {
            "fields": ["contraction", "vol_cv10", "setup", "branch"],
            "decision_time_visible": "contraction/vol_cv10 用 P及以前日线(触发前已知); setup 候选生成时已知; branch 首次可行动bar完成后(决策触发时刻)已知",
            "coverage_by_track": {tk: {k: wl[tk][k] for k in ("contraction", "vol_cv10", "setup", "branch", "n")}
                                  for tk in ("H3", "H5")},
            "missing_rate": miss_rate,
            "invariance_sample": {"n": len(sample), "feature_changed": inv_fail,
                                  "note": "扰动P后日线, contraction/vol_cv10 不受影响 (仅用P及以前); branch 由执行器在 signal_ts(首次可行动bar)决定, 对 signal_ts 后分钟截断不变(冻结设计属性)"},
        },
        "inputs_sha256": {
            "r1_exec_labels_v0_1_2.json": sha256_file(EX_PATH),
            "r1_label_admission_v0_1_2.json": sha256_file(ADM_PATH),
            "gcp_dual_path_coverage_probe.py": sha256_file(
                HERE / "gcp_dual_path_coverage_probe.py"),
            "dual_path_execution_v0_1_1.py": sha256_file(
                HERE / "dual_path_execution_v0_1_1.py"),
        },
    }
    md = [
        "# 拟合就绪有界核验 (v0.1.1, 第三步 c)", "",
        f"生成: {OUT['generated_at']} | 零网络 / 零采集 / 零拟合", "",
        "## 1. 人口厘清 (防误读)", "",
        f"- **214 = H3+H5 轨道数**, 不是 214 个独立训练样本。",
        f"- **H3 主研究 = {len(adm_h3)} 事件**：train {h3_by_seg['train']} / val {h3_by_seg['val']} / test {h3_by_seg['test']}；H5 = {len(adm_h5)} 事件仅辅助，不能救 H3。",
        f"- 输入 {n_input} 条；train 执行入口 {train_exec}。", "",
        "## 2. 覆盖门 (旧 80% 门未过，显式接受)", "",
        f"- train 准入 H3 = {h3_train} / 执行入口 {train_exec} = **{coverage:.1%}**。",
        "- 不声称继承 ACTION_FILTER 合同 §5 的 80% 覆盖门；本轮定位为小样本研究性拟合，覆盖偏差如实披露。", "",
        "## 3. RAW 尺度有界核验", "",
        f"- 方法：本地分钟末 bar 收盘 vs 日线 close（E 日），容差 max(0.01, 0.005×价)。",
        f"- 结果：{scale_results}。",
        f"- 尺度不一致事件 {len(scale_fail)} 条（仅验证 E 日收盘尺度一致；缺该日同尺度证据者拟合前继续隔离；不声称全部历史或整条持仓路径已验证同尺度）。", "",
        "## 4. 白名单四字段覆盖", "",
        "- 字段：`contraction`/`vol_cv10`（P 及以前日线，触发前已知）＋ `setup`（候选生成时已知）＋ `branch`（首次可行动 bar 完成后、决策触发时刻已知）。",
        f"- 覆盖：{ {tk: {k: wl[tk][k] for k in ('contraction','vol_cv10','setup','branch','n')} for tk in ('H3','H5')} }。",
        f"- 缺失率（contraction/vol_cv10）：{miss_rate}。",
        f"- 不变性抽检 {len(sample)} 条扰动 P 后日线：特征变化 {inv_fail}（期望 0，仅用 P 及以前→未来扰动不影响）。", "",
        "## 5. 结论", "",
        "- 准入人口、尺度证据、白名单覆盖与不变性均已核验；唯覆盖门未过已显式接受。",
        "- 32 只冲突码继续隔离；不修复全部历史数据、不补采 354 对、不训练逐股模型。",
        "- 拟合合同草案见 `contracts/dual_path_fit_contract_v0_1_draft.md`；**拟合须单独授权**。",
    ]
    text_json = json.dumps(OUT, ensure_ascii=False, indent=1)
    text_md = "\n".join(md)
    total_new = len(text_json.encode("utf-8")) + len(text_md.encode("utf-8"))
    if total_new > CAP_NEW_BYTES:
        raise RuntimeError(f"新增 {total_new} B 超 1 MiB, 停机")
    if base_before + total_new > CAP_BASE_BYTES:
        raise RuntimeError(f"累计超 55 MiB, 停机")
    OUT_JSON.write_text(text_json, encoding="utf-8", newline="")
    OUT_MD.write_text(text_md, encoding="utf-8", newline="")

    log = f"""

## 工程记录 ENGINEERING-2026-10-08-33：拟合就绪有界核验（零网络）

- **人口厘清**：214 = H3+H5 轨道数（非独立样本）；H3 主研究 124 事件（train {h3_by_seg['train']}/val {h3_by_seg['val']}/test {h3_by_seg['test']}），H5 90 事件仅辅助。32 只冲突码继续隔离；不修全部历史数据、不补采 354 对。
- **覆盖门**：train 准入 H3 {h3_train} / 执行入口 {train_exec} = {coverage:.1%}；旧 ACTION_FILTER 合同 §5 的 80% 门未过，显式接受覆盖限制，定为小样本研究性拟合，不声称继承旧门达标。
- **RAW 尺度核验**：本地分钟末 bar 收盘 vs 日线 close（E 日），{scale_results}；不一致 {len(scale_fail)} 条（缺同尺度证据，拟合前继续隔离）。
- **白名单覆盖**：contraction/vol_cv10（P及以前日线，决策时可见）+ setup/branch；覆盖与缺失率已列；不变性抽检 {len(sample)} 条扰动 P 后日线特征变化 {inv_fail}（仅用 P 及以前→未来扰动不影响）。
- **产物**：r1_fit_readiness_v0_1_1.json/.md（新增 {total_new} B；RUN 累计 {base_before+total_new} B / 55 MiB）。拟合合同草案 dual_path_fit_contract_v0_1_draft.md 待用户签署；拟合须单独授权。
"""
    if "ENGINEERING-2026-10-08-33" not in AUDIT_LOG.read_text(encoding="utf-8"):
        with open(AUDIT_LOG, "a", encoding="utf-8", newline="") as f:
            f.write(log)
    print(f"\n写入: {OUT_JSON.name} / {OUT_MD.name}; 审计日志 ENGINEERING-2026-10-08-33")
    print(f"预算: {base_before} + {total_new} = {base_before + total_new} B "
          f"({(base_before + total_new) / 1048576:.3f} MiB / 55 MiB)")
    print(f"耗时 {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
