# -*- coding: utf-8 -*-
"""第三步 d: 冻结逐事件拟合特征表 (零网络/零采集/零拟合).

物料化已准入人口的白名单四字段 + 已核验现金账本标签, 作为拟合(待单独授权)的冻结输入.
- 特征 = contraction/vol_cv10 (P及以前日线, 见 gcp_dual_path_coverage_probe.py / gcp_round_b_engine.build_atr_ref)
        + setup/branch (执行标签件).
- 标签 = 执行标签件 H3.net (现金账本 C_sell/C_buy-1, 含部分卖出; 见 dual_path_execution_v0_1_1.py).
- 不重算单退出价收益; 不训练; 仅物料化.
- 产物: r1_fit_features_v0_1_1.json (逐事件特征表) + r1_fit_manifest_v0_1_1.json (全输入+生成代码哈希).
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
FEAT_JSON = RUN / "r1_fit_features_v0_1_1.json"
MANI_JSON = RUN / "r1_fit_manifest_v0_1_1.json"

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
    print("第三步 d: 冻结逐事件拟合特征表 (零网络/零采集/零拟合)")
    print("=" * 70)

    EX = json.loads(EX_PATH.read_text(encoding="utf-8"))
    ADM = json.loads(ADM_PATH.read_text(encoding="utf-8"))
    records = EX["records"]
    rec_by_key = {(r["code"], int(r["E"]), r["branch"]): r for r in records}

    cal = D.load_calendar()
    cal_pos = {d: i for i, d in enumerate(cal)}

    panel = PROBE.get_daily_parquet()
    panel["code"] = panel["code"].astype(str).str.zfill(6)
    panel["date"] = panel["date"].astype(int)
    gby = {c: g.sort_values("date").drop_duplicates("date")
           for c, g in panel.groupby("code")}

    # 入场锚点 P = E 的前一个交易日
    def whitelist(code, Eday):
        P = cal[cal_pos[Eday] - 1] if cal_pos[Eday] > 0 else None
        g = gby.get(code)
        if g is None or P is None:
            return {"contraction": None, "vol_cv10": None}
        d_sub = g[g["date"].astype(int) <= P]
        if len(d_sub) < 20:
            return {"contraction": None, "vol_cv10": None}
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

    def build(track):
        rows = []
        for x in ADM["track_decisions"]:
            if x["track"] != track or not x["admitted"]:
                continue
            code, Eday, branch, seg = x["code"], int(x["E"]), x["branch"], x["segment"]
            rec = rec_by_key.get((code, Eday, branch))
            if rec is None:
                continue
            w = whitelist(code, Eday)
            net = rec.get(track, {}).get("net")
            rows.append({
                "code": code, "E": Eday, "segment": seg, "branch": branch,
                "setup": rec.get("setup"),
                "contraction": w["contraction"], "vol_cv10": w["vol_cv10"],
                "net": net,
                "label": (1 if (net is not None and net > 0) else 0),
                "le": int(x["le"]) if x.get("le") is not None else None,
            })
        return rows

    h3 = build("H3")
    h5 = build("H5")
    print(f"特征表: H3 {len(h3)} 事件 / H5 {len(h5)} 事件 (仅辅助)")

    feat = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "network_requests": 0, "training_performed": False, "data_collection_performed": False,
        "note": "冻结逐事件拟合特征表; 标签=执行标签件 H3.net 现金账本口径(C_sell/C_buy-1); 不重算单退出价; 不训练",
        "feature_spec": {
            "contraction": "atr_ref/median(ATR14[-60:]) 取 P及以前日线; gcp_round_b_engine.build_atr_ref",
            "vol_cv10": "volume[-10:] 变异系数 (std/mean, ddof=1) 取 P及以前日线",
            "setup": "执行标签件 setup 字段 (A_SDP/B_LVB)",
            "branch": "执行标签件 branch 字段 (DIP_REPAIR/BASE_HOLD), 决策触发时刻已知",
            "label": "H3.net>0 -> 1 否则 0; net 来自执行标签件 (现金账本 C_sell/C_buy-1, 含部分卖出)",
        },
        "generating_code": "gcp_step3d_fit_features.py",
        "population_H3_by_segment": dict(Counter(r["segment"] for r in h3)),
        "population_H5_by_segment": dict(Counter(r["segment"] for r in h5)),
        "H3_primary": h3,
        "H5_auxiliary": h5,
    }

    # 哈希: 先写特征表再哈希, 再写 manifest (含特征表哈希)
    base_before = dir_size(RUN)
    feat_text = json.dumps(feat, ensure_ascii=False, indent=1)
    if len(feat_text.encode("utf-8")) > CAP_NEW_BYTES:
        raise RuntimeError("特征表超 1 MiB, 停机")
    if base_before + len(feat_text.encode("utf-8")) > CAP_BASE_BYTES:
        raise RuntimeError("累计超 55 MiB, 停机")
    FEAT_JSON.write_text(feat_text, encoding="utf-8", newline="")

    inputs = {
        "r1_exec_labels_v0_1_2.json": EX_PATH,
        "r1_label_admission_v0_1_2.json": ADM_PATH,
        "gcp_dual_path_coverage_probe.py": HERE / "gcp_dual_path_coverage_probe.py",
        "gcp_round_b_engine.py": HERE / "gcp_round_b_engine.py",
        "dual_path_execution_v0_1_1.py": HERE / "dual_path_execution_v0_1_1.py",
        "gcp_adj_cache_acceptance.json": RUN / "gcp_adj_cache_acceptance.json",
        "gcp_step3c_fit_readiness.py": HERE / "gcp_step3c_fit_readiness.py",
        "r1_fit_readiness_v0_1_1.json": RUN / "r1_fit_readiness_v0_1_1.json",
        "gcp_step3d_fit_features.py": HERE / "gcp_step3d_fit_features.py",
        "r1_fit_features_v0_1_1.json": FEAT_JSON,
    }
    manifest = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "note": "拟合(待单独授权)的冻结输入清单: 逐事件特征表 + 其生成代码 + 全部上游输入哈希",
        "sha256": {k: sha256_file(v) for k, v in inputs.items()},
        "feature_table_path": "gcp_round_b_small_sample_v1/r1_fit_features_v0_1_1.json",
        "feature_generating_code": "gcp_step3d_fit_features.py",
    }
    mani_text = json.dumps(manifest, ensure_ascii=False, indent=1)
    total_new = len(feat_text.encode("utf-8")) + len(mani_text.encode("utf-8"))
    if total_new > CAP_NEW_BYTES:
        raise RuntimeError(f"新增 {total_new} B 超 1 MiB, 停机")
    if base_before + total_new > CAP_BASE_BYTES:
        raise RuntimeError(f"累计超 55 MiB, 停机")
    MANI_JSON.write_text(mani_text, encoding="utf-8", newline="")

    log = f"""

## 工程记录 ENGINEERING-2026-10-08-34：冻结逐事件拟合特征表（第三步 d）

- **产物**：r1_fit_features_v0_1_1.json（H3 {len(h3)} 事件主表 + H5 {len(h5)} 事件辅助表），含白名单四字段 + 现金账本标签 net（执行标签件 H3.net，C_sell/C_buy-1，含部分卖出）。
- **标签口径**：直接采用已核验 net，不重算单退出价公式；主标签 = net>0 → 1。
- **冻结输入清单**：r1_fit_manifest_v0_1_1.json 含全部上游输入 + 特征表 + 生成代码（gcp_step3d_fit_features.py）的 SHA256（要约填入拟合合同 §6，无占位符）。
- **范围**：零网络/零采集/零拟合；不修 32 只冲突码、不补采 354 对、不训练逐股模型。
- **预算**：新增 {total_new} B；RUN 累计 {base_before + total_new} B / 55 MiB。合同 §5/§6/§8 已据用户 5 点收紧意见改写待签；拟合仍须单独授权。
"""
    if "ENGINEERING-2026-10-08-34" not in AUDIT_LOG.read_text(encoding="utf-8"):
        with open(AUDIT_LOG, "a", encoding="utf-8", newline="") as f:
            f.write(log)
    print(f"写入: {FEAT_JSON.name} / {MANI_JSON.name}; 审计 ENGINEERING-2026-10-08-34")
    print(f"预算: {base_before} + {total_new} = {base_before + total_new} B "
          f"({(base_before + total_new) / 1048576:.3f} MiB / 55 MiB)")
    print(f"耗时 {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
