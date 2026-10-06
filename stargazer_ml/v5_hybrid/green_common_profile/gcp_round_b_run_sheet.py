"""GREEN-COMMON-PROFILE-v0.1 Round B 小样本 run sheet 生成器 (REV3 §8).

冻结 <=20 股票 × <=15 监控日 E 的显式登记清单 (确定性种子 + 场景定向), 含:
  - 每股真实回看 60 交易日封顶 (不静默扩到 250)
  - 九类场景覆盖意图 (实际场景在回放后确认)
  - 分钟请求预算核算 (每股 E 日 + 回看日)
生成后写 JSON + sha256, 未列入清单的请求不允许发出。
"""
import hashlib
import json
import sys
from bisect import bisect_left, bisect_right
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).parent
ROOT = HERE.parent                    # stargazer_ml/v5_hybrid
sys.path.insert(0, str(ROOT / "step9_preflight"))
from data_source_provenance import get_daily_parquet  # noqa: E402

SEED = 20260928
MAX_CODES = 20
MAX_E_PER_CODE = 15
LOOKBACK_REAL_CAP = 60
MIN_MINUTE_DATE = 20250102      # 分钟表覆盖起始 (探针 2026-09-28 核实, 20241231 空)
MAX_MINUTE_DATE = 20260917      # 日线 parquet 末日渐即分钟末
POOL_DIR = HERE / "round_a_v0_1_impl_fix"
OUT = HERE / "gcp_round_b_small_sample_v1" / "run_sheet.json"


def main():
    rng = np.random.default_rng(SEED)
    panel = get_daily_parquet()
    need = {"date", "code", "open", "high", "low", "close"}
    assert need <= set(panel.columns), f"日线 parquet 缺列: {need - set(panel.columns)}"
    cal = sorted(int(d) for d in panel["date"].unique())
    cal_idx = {d: i for i, d in enumerate(cal)}

    def nxt(d):
        i = bisect_right(cal, d)
        return cal[i] if i < len(cal) else None

    def prev(d):
        i = bisect_left(cal, d)
        return cal[i - 1] if i > 0 else None

    stocks = []
    for pool_file, setup in (("pool_a_sdp_members.parquet", "A_SDP"),
                             ("pool_b_lvb_members.parquet", "B_LVB")):
        df = pd.read_parquet(POOL_DIR / pool_file)
        # E = D 的下一交易日; 去掉 E 超出日历的
        cand = {}
        for d, c in zip(df["date"].astype(int), df["code"].astype(str).str.zfill(6)):
            e = nxt(d)
            if e is not None and MIN_MINUTE_DATE <= e <= MAX_MINUTE_DATE:
                cand.setdefault(c, set()).add(e)   # E 必须落在分钟覆盖窗内
        n_cand = {c: len(v) for c, v in cand.items()}

        # 每股日线子表 (场景定向用)
        def scenarios(code, e_days):
            sub = panel[panel["code"].astype(str).str.zfill(6) == code].sort_values("date")
            sub = sub.assign(cd=sub["date"].astype(int))
            s = sub.set_index("cd")
            tags = {}
            for e in e_days:
                p = prev(e)
                if p is None or p not in s.index or e not in s.index:
                    continue
                row_p, row_e = s.loc[p], s.loc[e]
                t = []
                if float(row_e["open"]) < float(row_p["low"]):
                    t.append("gap_down")
                # severe: low_E <= close_P - 1.5*ATR (ATR 用 TR 14 日均值近似)
                if len(s) > 15:
                    h, l, c_ = (s["high"].astype(float), s["low"].astype(float),
                                s["close"].astype(float))
                    tr = np.maximum(h - l, np.maximum((h - c_.shift()).abs(),
                                                      (l - c_.shift()).abs()))
                    atr = tr.rolling(14).mean()
                    if p in atr.index and np.isfinite(atr.loc[p]):
                        if float(row_e["low"]) <= float(row_p["close"]) - 1.5 * atr.loc[p]:
                            t.append("severe_dip")
                i_exit = cal_idx[e] + 3
                if i_exit < len(cal) and (e // 100) != (cal[i_exit] // 100):
                    t.append("cross_month")
                tags[e] = t
            return tags

        # ---- 选股: 4 随机(>=10 候选) + 1 少候选(<=3, 无画像回退) + 3 场景定向 ----
        rich = sorted([c for c, n in n_cand.items() if n >= 10])
        poor = sorted([c for c, n in n_cand.items() if 1 <= n <= 3])
        picked = list(rng.choice(rich, size=min(4, len(rich)), replace=False))
        if poor:
            picked.append(str(rng.choice(poor)))
        # 场景定向: 扫描有候选的股, 找 gap_down / severe / cross_month 各 1 只
        scan_pool = [c for c in sorted(cand) if c not in picked][:400]
        for want in ("gap_down", "severe_dip", "cross_month"):
            found = None
            for c in scan_pool:
                if c in picked:
                    continue
                e_days = sorted(cand[c])[:MAX_E_PER_CODE]
                tags = scenarios(c, e_days)
                if any(want in t for t in tags.values()):
                    found = (c, e_days, tags)
                    break
            if found and len(picked) < 10:
                picked.append(found[0])
        picked = list(dict.fromkeys(picked))[:10]   # 去重保序

        for c in picked:
            e_days = sorted(cand[c])
            rng2 = np.random.default_rng(SEED + hash(c) % 100000)
            if len(e_days) > MAX_E_PER_CODE:
                e_days = sorted(rng2.choice(e_days, size=MAX_E_PER_CODE, replace=False))
            tags = scenarios(c, e_days)
            # 场景优先排序
            e_days = sorted(e_days, key=lambda e: -len(tags.get(e, [])))
            earliest = min(e_days)
            lb_start = max(MIN_MINUTE_DATE,
                           int(cal[max(0, cal_idx[earliest] - LOOKBACK_REAL_CAP)]))
            fetch_days = sorted({int(e) for e in e_days} |
                                {int(d) for d in cal
                                 if lb_start <= d < earliest})   # 均在覆盖窗内
            e_days = [int(e) for e in e_days]
            stocks.append({"code": c, "setup": setup, "E_days": e_days,
                           "scenario_tags": {str(e): tags.get(e, []) for e in e_days},
                           "lookback_start": lb_start,
                           "minute_fetch_days": fetch_days,
                           "n_minute_requests": len(fetch_days)})

    total_req = sum(s["n_minute_requests"] for s in stocks)
    assert len(stocks) <= MAX_CODES, f"股票数超限 {len(stocks)}"
    assert total_req <= 1700, f"分钟请求预算超限 {total_req}"

    sheet = {"version": "GCP-ROUNDB-SMALL-SAMPLE-V1", "seed": SEED,
             "spec": "GREEN-COMMON-PROFILE-v0.1-ROUNDB-REV3",
             "created": "2026-09-28",
             "constraints": {"max_codes": MAX_CODES, "max_E_per_code": MAX_E_PER_CODE,
                             "real_lookback_cap": LOOKBACK_REAL_CAP,
                             "minute_window": [MIN_MINUTE_DATE, MAX_MINUTE_DATE]},
             "coverage_intents": ["gap_down", "severe_dip", "cross_month",
                                  "no_profile_fallback", "approach_only", "am_fill",
                                  "pm_fill", "fail_event", "invalid_day"],
             "coverage_note": "am/pm/approach/fail/invalid 在回放后按实际确认; "
                              "合成数据已覆盖 invalid/回退/分支 (验收 B)",
             "stocks": stocks, "total_minute_requests": total_req}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(sheet, sort_keys=True, ensure_ascii=False)
    sheet["run_sheet_sha256"] = hashlib.sha256(payload.encode()).hexdigest()
    OUT.write_text(json.dumps(sheet, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"run sheet 冻结: {OUT}")
    print(f"股票 {len(stocks)} 只, 总分钟请求预算 {total_req}")
    for s in stocks:
        tag_sum = sum(len(v) for v in s["scenario_tags"].values())
        print(f"  {s['setup']:5s} {s['code']}  E={len(s['E_days'])}  "
              f"请求={s['n_minute_requests']}  场景标签={tag_sum}")


if __name__ == "__main__":
    main()
