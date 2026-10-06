"""GREEN-COMMON-PROFILE-v0.1 Round B 登记范围内 P0/P1 限定诊断 (2026-09-29).

范围 (用户 2026-09-29 裁定):
  - 仅登记清单 (16股×E日) 内限定回放, 不扩大抓取; 全部输入来自已缓存快照/复权链/池文件
  - 场景定向小样本只检查行为与机制, 不用于宣布盈利或全市场有效
  - FILLED 为工程假设成交事件 (1%参与率研究近似), 非 P0/P1 策略真实成交

内容:
  轨道2 (决策函数行为): 同一预注册决策函数 enter=1[mu_hat>0 AND p_fail_hat<=0.50],
    P0 用组统计 mu_g/p_g, P1 用收缩 mu_1/p_1; 统计分支/一致率/名义账户口径
  轨道1 (信息诊断): mu_1-mu_g 差与标签关联 (本样本 P1 分支 n=0 时如实报告不可检验)
  数据可用性: 逐 (股,E) 的 250 日回看窗内池资格日数 (离线可算), 评估 P1 分支
    (个股同格 n>=10) 在扩大抓取下的可达性, 为下一轮范围讨论提供依据
运行: python gcp_round_b_p0p1_diagnosis.py
  (内部复用 acceptance.stage_real 同输入重放; 快照已缓存, 无新增分钟请求)
"""
from __future__ import annotations

import bisect
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "step9_preflight"))
sys.path.insert(0, str(HERE))
from data_source_provenance import get_daily_parquet  # noqa: E402
import gcp_round_b_engine as E  # noqa: E402
import gcp_round_b_acceptance as ACC  # noqa: E402

OUT = HERE / "gcp_round_b_small_sample_v1"
POOL_DIR = HERE / "round_a_v0_1_impl_fix"


def main():
    grid = json.loads((OUT / "minute_grid.json").read_text(encoding="utf-8"))["grid"]
    panel = get_daily_parquet()
    cal = sorted(int(d) for d in panel["date"].unique())
    didx = {d: i for i, d in enumerate(cal)}

    # ---- 同输入重放 (复用验收驱动; 全缓存, 无新增分钟请求) ----
    print("== 同输入重放 (复用 stage_real, 全缓存) ==")
    all_events, real_recs = ACC.stage_real(grid, panel, True)

    # ---- 轨道2: 决策函数 P0 vs P1 (名义账户口径, 工程近似) ----
    print("\n== 轨道2: P0/P1 决策函数行为 ==")
    rows = []
    for r in real_recs:
        row = {"code": r["code"], "date": r["date"], "session": r.get("session"),
               "status": r["status"], "signal_type": r.get("signal_type")}
        if r.get("status") != "FILLED" or not r.get("zone_sources"):
            row.update({"branch": "NO_EVENT", "enter_p0": False, "reason_p0": r["status"],
                        "enter_p1": False, "reason_p1": r["status"],
                        "pnl_p0": 0.0, "pnl_p1": 0.0, "y": None})
            rows.append(row)
            continue
        est = E.estimate(all_events, r["code"], r["setup"], r["zone_sources"],
                         r["session"], r["date"], didx)
        d0 = E.decide(est["mu_g"], est["p_g"])
        d1 = E.decide(est["mu_1"], est["p_1"])
        branch = ("NO_ESTIMATE" if est["no_estimate"]
                  else "FALLBACK" if est["fallback"] else "P1")
        pnl = (r.get("fill_amount") or 0.0) * (r.get("y") or 0.0)
        row.update({"branch": branch, "n_g": est["n_g"], "n": est["n"],
                    "cell": "-".join([str(r["setup"]),
                                      "+".join(r["zone_sources"]), r["session"]]),
                    "mu_g": est["mu_g"], "p_g": est["p_g"],
                    "mu_1": est["mu_1"], "p_1": est["p_1"],
                    "enter_p0": d0["enter"], "reason_p0": d0["reason"],
                    "enter_p1": d1["enter"], "reason_p1": d1["reason"],
                    "y": r.get("y"), "fill_amount": r.get("fill_amount"),
                    "pnl_p0": pnl if d0["enter"] else 0.0,
                    "pnl_p1": pnl if d1["enter"] else 0.0})
        rows.append(row)
    df = pd.DataFrame(rows)
    filled = df[df["status"] == "FILLED"]
    br = filled["branch"].value_counts().to_dict()
    n_e0 = int(filled["enter_p0"].sum())
    n_e1 = int(filled["enter_p1"].sum())
    agree = int((filled["enter_p0"] == filled["enter_p1"]).mean() * len(filled))
    pnl0 = float(filled["pnl_p0"].sum())
    pnl1 = float(filled["pnl_p1"].sum())
    inv0 = float(filled.loc[filled["enter_p0"], "fill_amount"].sum())
    inv1 = float(filled.loc[filled["enter_p1"], "fill_amount"].sum())
    y0 = filled.loc[filled["enter_p0"], "y"]
    y1 = filled.loc[filled["enter_p1"], "y"]
    months = filled.assign(m=filled["date"] // 100).groupby("m").agg(
        n=("y", "size"), e0=("enter_p0", "sum"), e1=("enter_p1", "sum"))
    print(f"  E日FILLED事件={len(filled)}  分支={br}")
    print(f"  P0: enter={n_e0} 名义投入={inv0/1e4:.1f}万 PnL={pnl0:+.0f}元 "
          f"(n={len(y0)}, mean_y={y0.mean() if len(y0) else float('nan'):+.4f})")
    print(f"  P1: enter={n_e1} 名义投入={inv1/1e4:.1f}万 PnL={pnl1:+.0f}元 "
          f"(n={len(y1)}, mean_y={y1.mean() if len(y1) else float('nan'):+.4f})")
    print(f"  P0/P1 决策一致率={agree}/{len(filled)}; 月度:\n{months.to_string()}")

    # ---- 轨道1: 信息诊断 (mu_1 - mu_g 与标签) ----
    print("\n== 轨道1: 信息诊断 ==")
    has_est = filled[filled["branch"].isin(["FALLBACK", "P1"])]
    if len(has_est):
        d_mu = (has_est["mu_1"] - has_est["mu_g"]).abs()
        n_identical = int((d_mu < 1e-15).sum())
        corr_g = (float(np.corrcoef(has_est["mu_g"], has_est["y"])[0, 1])
                  if len(has_est) >= 3 else None)
        print(f"  可估计事件={len(has_est)}  mu_1==mu_g(纯回退)={n_identical} "
              f"|mu_1-mu_g|max={d_mu.max():.6f}")
        print(f"  mu_g 与同期标签 y 的相关={corr_g if corr_g is None else round(corr_g, 3)} "
              f"(n={len(has_est)}, 仅机制描述, 无绩效含义)")
        if n_identical == len(has_est):
            print("  结论: 本样本 P1 分支(n>=10)为 0, mu_1≡mu_g → 个性增量不可检验")
    else:
        print("  无可估计事件 (全部 NO_ESTIMATE)")

    # ---- 数据可用性: 250 日回看窗内池资格日数 (离线可算, 无需分钟数据) ----
    print("\n== 数据可用性: P1 分支可达性 (上界估计) ==")
    pool = pd.concat([
        pd.read_parquet(POOL_DIR / "pool_a_sdp_members.parquet")[["date", "code"]],
        pd.read_parquet(POOL_DIR / "pool_b_lvb_members.parquet")[["date", "code"]],
    ])
    pool["code"] = pool["code"].astype(str).str.zfill(6)
    pool_days = pool.groupby("code")["date"].apply(lambda s: sorted(set(int(x) for x in s))).to_dict()
    look, qual = [], []
    for r in real_recs:
        days = pool_days.get(r["code"], [])
        i = bisect.bisect_left(cal, r["date"])
        window = cal[max(0, i - 250):i]                      # E 前 250 交易所日
        q = sum(1 for d in window
                if d in days and bisect.bisect_left(pool_days[r["code"]], d) >= 0
                and d + 1 in didx and True)                   # 池资格日 (D 日, 次日才监控)
        # 注意: 池 date=D(选股日), 监控日=D+1; 统计窗内可作为监控日的池资格日
        q_monitored = sum(1 for d in window if d in set(days))
        look.append(q_monitored)
    look = np.array(look)
    print(f"  E日事件={len(look)}  250日回看窗内池资格日(D)分布: "
          f"p10={np.percentile(look, 10):.0f} p50={np.percentile(look, 50):.0f} "
          f"p90={np.percentile(look, 90):.0f} max={look.max():.0f}")
    print(f"  >=10 个池资格日的 E 日占比={(look >= 10).mean()*100:.1f}% "
          f"(同格 setup+zone_sources+session 匹配后会更低, 此为上界)")
    cell_hist = [e for e in all_events
                 if e.get("status") == "FILLED" and e.get("setup") is not None]
    print(f"  16股全部池资格 FILLED 事件={len(cell_hist)} "
          f"(组单元格样本来源; 扩大抓取后按池密度外推)")

    # ---- 落盘 ----
    diag = {
        "diagnosis_version": "GCP_ROUND_B_P0P1_LIMITED_DIAGNOSIS_V1",
        "date": "2026-09-29",
        "scope": "登记清单内限定诊断; 工程假设成交; 不作盈利/全市场有效性结论",
        "track2": {"n_e_day_filled": int(len(filled)), "branches": br,
                   "p0": {"enter": n_e0, "invested_yuan": inv0, "pnl_yuan": pnl0,
                          "mean_y_entered": (float(y0.mean()) if len(y0) else None)},
                   "p1": {"enter": n_e1, "invested_yuan": inv1, "pnl_yuan": pnl1,
                          "mean_y_entered": (float(y1.mean()) if len(y1) else None)},
                   "decision_agreement": f"{agree}/{len(filled)}",
                   "monthly": months.reset_index().to_dict(orient="records")},
        "track1": {"n_estimable": int(len(has_est)),
                   "mu1_equals_mug": bool(len(has_est) and
                                          ((has_est["mu_1"] - has_est["mu_g"]).abs()
                                           < 1e-15).all()),
                   "note": "P1分支n=0时个性增量不可检验, 如实报告"},
        "data_availability": {
            "lookback_pool_days_p10_p50_p90_max":
                [float(np.percentile(look, p)) for p in (10, 50, 90, 100)],
            "share_e_days_with_ge10_pool_days": float((look >= 10).mean()),
            "pool_qualified_filled_events_16stocks": len(cell_hist),
        },
        "events": rows,
    }
    rp = OUT / "p0p1_diagnosis_v1.json"
    rp.write_text(json.dumps(diag, ensure_ascii=False, indent=1, default=str),
                  encoding="utf-8")
    print(f"\n诊断已写入: {rp}")


if __name__ == "__main__":
    main()
