# -*- coding: utf-8 -*-
"""3d/5d 账本验证：dual_head_labels 的 5d 账本 vs minute_samples 引擎字段。

对两年数据 join (code,dp_date,dp) 后比对：
- o_t1_bt_5d == (outcome_t1.0=="TARGET")
- o_net_ret_5d ≈ net_ret_t2.0（kt=2.0 主场景，容忍 1e-9）
- o_mae_5d_atr*atr_pct ≈ mae；o_mfe_5d_atr*atr_pct ≈ mfe
- o_stop_5d == (outcome_t2.0=="STOP")
不一致率必须 <0.1%，否则 3d 账本同样不可信，需停下来修。
另报告 o_ties_5d 同bar双触事件数（口径审计：引擎同bar双触记TARGET，
与"悲观STOP"文档不一致，数量级供评审决定是否重生成样本）。
"""
import os
import sys
import time

import numpy as np
import pandas as pd

OUT_DIR = "D:/stargazer/stargazer_ml/pullback_train"
PAIRS = (("minute_samples_2025.parquet", "dual_head_labels_2025.parquet"),
         ("minute_labels/minute_samples_2026.parquet",
          "dual_head_labels_2026.parquet"))


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def check(sp_path, lb_path):
    sp = pd.read_parquet(os.path.join(OUT_DIR, sp_path))
    sp = sp[(sp["entry_type"] == "vwap") & (sp["censored"] == 0)]
    sp["code"] = sp["code"].astype(str)
    # 同(code,dp_date,dp)可能有多事件（不同group_id）；账本侧 groupby first
    # 取首行，这里同样 keep="first" 对齐，避免 join 扇出造成假性不一致
    sp = sp.drop_duplicates(subset=["code", "dp_date", "dp"], keep="first")
    lb = pd.read_parquet(os.path.join(OUT_DIR, lb_path))
    lb["code"] = lb["code"].astype(str)
    m = lb.merge(sp[["code", "dp_date", "dp", "outcome_t1.0", "outcome_t2.0",
                     "net_ret_t2.0", "mfe", "mae", "atr_pct"]],
                 on=["code", "dp_date", "dp"], how="inner")
    log(f"{lb_path}: join {len(m):,}/{len(lb):,}")
    r = {}
    r["t1_match"] = float((m["o_t1_bt_5d"] == (m["outcome_t1.0"] == "TARGET"))
                          .mean())
    r["stop_match"] = float((m["o_stop_5d"] == (m["outcome_t2.0"] == "STOP"))
                            .mean())
    d_nr = (m["o_net_ret_5d"] - m["net_ret_t2.0"]).abs()
    r["nr_exact"] = float((d_nr < 1e-9).mean())
    r["nr_maxdiff"] = float(d_nr.max())
    d_mfe = (m["o_mfe_5d_atr"] * m["atr_pct"] - m["mfe"]).abs()
    d_mae = (m["o_mae_5d_atr"] * m["atr_pct"] - m["mae"]).abs()
    r["mfe_match"] = float((d_mfe < 1e-9).mean())
    r["mae_match"] = float((d_mae < 1e-9).mean())
    r["ties_rows"] = int((m["o_ties_5d"] > 0).sum())
    r["ties_events"] = int(m["o_ties_5d"].sum())
    r["n"] = len(m)
    # 3d 账本分布
    r["nr3_mean"] = float(m["o_net_ret_3d"].mean())
    r["nr5_mean"] = float(m["o_net_ret_5d"].mean())
    r["t1_3d"] = float(m["o_t1_bt_3d"].mean())
    r["t1_5d"] = float(m["o_t1_bt_5d"].mean())
    r["pos3"] = float((m["o_net_ret_3d"] > 0).mean())
    r["pos5"] = float((m["o_net_ret_5d"] > 0).mean())
    r["rbl3"] = float(m["o_repair_but_loss_3d"].mean())
    r["rbl5"] = float(m["o_repair_but_loss_5d"].mean())
    return r


def main():
    all_r = {}
    for sp, lb in PAIRS:
        all_r[lb] = check(sp, lb)
    for lb, r in all_r.items():
        log(f"== {lb} ==")
        for k, v in r.items():
            log(f"  {k}: {v:,.6f}" if isinstance(v, float) else f"  {k}: {v}")
    ok = all(r["t1_match"] > 0.999 and r["nr_exact"] > 0.999
             and r["stop_match"] > 0.999 and r["mfe_match"] > 0.999
             and r["mae_match"] > 0.999 for r in all_r.values())
    log(f"VALIDATION {'PASS' if ok else 'FAIL'}")


if __name__ == "__main__":
    main()
