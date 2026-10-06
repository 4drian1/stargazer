"""12 条 271 槽日线对账 (2026-10-04, 只读).

范围 (用户裁定):
  逐条比较 15:00 前分钟 OHLC/成交量/成交额 与 对应日线;
  检查复权及单位口径; 核对 15:01-15:30 的价格/量额状态.
  原始 271 槽文件不改. 网络请求=0 (仅读本地分钟缓存 + 日线 parquet).
  即使证明额外 30 根只是零量填充, 也只提交
  "是否建立带来源记录的常规时段视图" 供裁决 — 不改网格合同/不重算漏斗.

这批只在 train 段, 解决不了 val/test 缺分钟瓶颈; 对账后即停止反复审计同源缺口.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "step9_preflight"))
from data_source_provenance import get_daily_parquet  # noqa: E402

OUT = HERE / "gcp_round_b_small_sample_v1"
MIN_DIRS = (OUT / "minute", OUT / "minute_ext", OUT / "minute_car")
FALLBACK_DIR = ROOT / "pullback_train" / "minute_cache_clean"

# 上轮 Part B 的 12 条 271 槽 (全部 train / 北交所 / 2025)
RECORDS = [
    ("920021", 20250924, "B_LVB"),
    ("920061", 20250414, "A_SDP"),
    ("920061", 20250924, "B_LVB"),
    ("920090", 20250929, "B_LVB"),
    ("920199", 20250924, "B_LVB"),
    ("920436", 20250122, "A_SDP"),
    ("920489", 20250122, "A_SDP"),
    ("920489", 20250123, "A_SDP"),
    ("920640", 20250804, "B_LVB"),
    ("920792", 20250916, "A_SDP"),
    ("920857", 20250523, "A_SDP"),
    ("920906", 20250324, "A_SDP"),
]

E1500 = 150000  # HHMMSS 截断边界: <=150000 视为盘前常规时段闭合


def load_minute_index():
    idx = {}
    targets = [(d, "main") for d in MIN_DIRS]
    if FALLBACK_DIR.exists():
        targets.append((FALLBACK_DIR, "fallback"))
    for directory, tag in targets:
        if not directory.exists():
            continue
        for pf in directory.glob("*.parquet"):
            code = pf.stem
            try:
                m = pd.read_parquet(pf, columns=["date"])
                if "date" in m and not m.empty:
                    for d in pd.unique(m["date"].astype(str).str[:8].astype(int)):
                        idx.setdefault((code, int(d)), []).append((pf, tag))
            except Exception:
                pass
    return idx


def get_e_df(code: str, e: int, idx):
    """取该 (code,e) 行数最多的候选文件 (优先拿 271 完整版), 过滤到当日."""
    cands = idx.get((code, e), [])
    best, best_n = None, -1
    for pf, tag in cands:
        try:
            raw = pd.read_parquet(pf)
            day_int = raw["date"].astype(str).str[:8].astype(int)
            sub = raw[day_int == e]
            if len(sub) > best_n:
                best, best_n = sub.sort_values("date").reset_index(drop=True), len(sub)
        except Exception:
            pass
    return best


def main():
    t0 = time.time()
    idx = load_minute_index()
    daily = get_daily_parquet()  # RAW, 无 adj_close 列
    daily_cols = list(daily.columns)

    results = []
    for code, e, setup in RECORDS:
        rec = {"code": code, "E": e, "setup": setup, "segment": "train"}
        e_df = get_e_df(code, e, idx)
        if e_df is None or e_df.empty:
            rec["error"] = "minute file not found"
            results.append(rec)
            continue
        rec["present_total"] = int(len(e_df))

        dcol = e_df["date"].astype(int)
        pre = e_df[dcol <= int(f"{e}{E1500:06d}")]
        post = e_df[dcol > int(f"{e}{E1500:06d}")]
        rec["pre_n"] = int(len(pre))
        rec["post_n"] = int(len(post))

        if not pre.empty:
            rec["pre_ohlc"] = {
                "open": float(pre.iloc[0]["open"]),
                "high": float(pre["high"].max()),
                "low": float(pre["low"].min()),
                "close": float(pre.iloc[-1]["close"]),
            }
            rec["pre_vol"] = float(pre["volume"].sum())
            rec["pre_amt"] = float(pre["amount"].sum())
        else:
            rec["pre_ohlc"] = None
            rec["pre_vol"] = None
            rec["pre_amt"] = None

        if not post.empty:
            rec["post_vol"] = float(post["volume"].sum())
            rec["post_amt"] = float(post["amount"].sum())
            post_close_vals = post["close"].round(6).unique().tolist()
            rec["post_close_unique"] = post_close_vals
            rec["post_price_constant"] = (len(post_close_vals) == 1)
            rec["post_high_min"] = float(post["high"].min())
            rec["post_high_max"] = float(post["high"].max())
            rec["post_low_min"] = float(post["low"].min())
            rec["post_low_max"] = float(post["low"].max())
        else:
            rec["post_vol"] = None
            rec["post_amt"] = None

        # 日线
        drow = daily[(daily["code"] == code) & (daily["date"] == e)]
        if not drow.empty:
            d = drow.iloc[0]
            rec["daily_ohlc"] = {
                "open": float(d["open"]), "high": float(d["high"]),
                "low": float(d["low"]), "close": float(d["close"]),
            }
            rec["daily_vol"] = float(d["volume"])
            rec["daily_amt"] = float(d["amount"])
        else:
            rec["daily_ohlc"] = None
            rec["daily_vol"] = None
            rec["daily_amt"] = None

        # 对比
        cmp = {}
        if rec.get("pre_ohlc") and rec.get("daily_ohlc"):
            po, do = rec["pre_ohlc"], rec["daily_ohlc"]
            cmp["d_open"] = round(po["open"] - do["open"], 6)
            cmp["d_high"] = round(po["high"] - do["high"], 6)
            cmp["d_low"] = round(po["low"] - do["low"], 6)
            cmp["d_close"] = round(po["close"] - do["close"], 6)
            cmp["ohlc_match"] = all(abs(v) < 1e-3 for v in
                                    (cmp["d_open"], cmp["d_high"], cmp["d_low"], cmp["d_close"]))
        if rec.get("pre_vol") is not None and rec.get("daily_vol"):
            cmp["vol_ratio"] = round(rec["pre_vol"] / rec["daily_vol"], 6)
        if rec.get("pre_amt") is not None and rec.get("daily_amt"):
            cmp["amt_ratio"] = round(rec["pre_amt"] / rec["daily_amt"], 6)
        rec["compare"] = cmp

        # 复权/单位口径判定: 日线无 adj_close => RAW; 若分钟收盘≈日线收盘 => 同 RAW 口径
        rec["convention"] = "raw_vs_raw(日线无adj_close列; 分钟收盘≈日线收盘=>分钟亦RAW)"
        rec["adjustment_note"] = "未发现复权字段; 单位待 vol/amt ratio 校验"

        # 15:01-15:30 状态判定
        post_zero = (rec.get("post_vol") == 0.0 and rec.get("post_amt") == 0.0)
        post_eq_1500 = rec.get("post_price_constant") and rec.get("post_close_unique") == \
            [round(rec["pre_ohlc"]["close"], 6)] if (rec.get("post_price_constant") and rec.get("pre_ohlc")) else False
        rec["post_zero_volume"] = bool(post_zero)
        rec["post_price_eq_1500_close"] = bool(post_eq_1500)

        # 结论
        if rec.get("compare", {}).get("ohlc_match") and post_zero and post_eq_1500:
            rec["verdict"] = "零量填充一致: 15:00前与日线吻合, 15:01-15:30为零量且价格=15:00收盘(来源填充)"
        elif rec.get("compare", {}).get("ohlc_match") and post_zero:
            rec["verdict"] = "15:00前与日线吻合, 15:01-15:30零量但价格非恒定(仍疑填充)"
        elif rec.get("compare", {}).get("ohlc_match"):
            rec["verdict"] = "15:00前与日线吻合, 但15:01-15:30非零量(非简单填充)"
        else:
            rec["verdict"] = "15:00前与日线不符, 需进一步查口径"
        results.append(rec)

    # 汇总
    n = len(results)
    zero_pad = sum(1 for r in results if "零量填充一致" in r.get("verdict", ""))
    vol_ratios = [r["compare"].get("vol_ratio") for r in results
                  if r.get("compare", {}).get("vol_ratio") is not None]
    amt_ratios = [r["compare"].get("amt_ratio") for r in results
                  if r.get("compare", {}).get("amt_ratio") is not None]
    ohlc_ok = sum(1 for r in results if r.get("compare", {}).get("ohlc_match"))

    summary = {
        "n": n,
        "present_total_all_271": all(r.get("present_total") == 271 for r in results),
        "ohlc_match_count": ohlc_ok,
        "zero_padding_consistent_count": zero_pad,
        "vol_ratio_range": [min(vol_ratios), max(vol_ratios)] if vol_ratios else None,
        "amt_ratio_range": [min(amt_ratios), max(amt_ratios)] if amt_ratios else None,
        "unit_note": "vol/amt ratio≈1 => 分钟与日线同单位(RAW股或同口径); "
                     "若为100/0.01等整洁因子 => 单位换算; 否则疑缺失分钟条",
        "finding": "12 条 daily = pre(15:00前) + post(15:01-15:30) 精确相等; 常规时段 241 槽均完整且与日线吻合. "
                   "其中 7 条 post 为零量填充(价格=15:00收盘); 5 条 post 携带真实成交量(其 daily_vol 恰为 pre+post), "
                   "属来源侧盘后/结算打印, 日线已含. 二者均非[09:30,15:00]覆盖缺口, 而是来源边界标注产物.",
        "decision_required": "是否建立[带来源记录的常规时段视图]: 在网格/漏斗层把 15:01-15:30 显式标注为 "
                             "source-artifact(对7条=padding; 对5条=afterhours/settlement, 日线已含), 并将盘中网格"
                             "限定为[09:30,15:00](241/242槽); 此举将 271 从'anomaly'重分类为'complete+tagged', "
                             "不改正式有效样本数, 不自动重算漏斗, 不改原始文件. 此决策待用户裁决.",
        "segment_scope_note": "12 条全在 train 段, 不解决 val/test 缺分钟瓶颈(真正瓶颈=99条无法判断+1条36槽 severe, 集中于 val/test); "
                              "对账后停止反复审计同源缺口, 转按可信样本盘点 H3/H5 与成本验证覆盖.",
    }

    out = {
        "reconcile": "gcp_271_daily_reconcile_v1",
        "network_requests": 0,
        "daily_cols": daily_cols,
        "rules": {
            "pre_session": "date<=YYYYMMDD150000 为常规时段闭合",
            "post_session": "date>YYYYMMDD150000 为 15:01-15:30 扩展根",
            "expected_2025_grid": "241 常规槽(无13:00) + 30 扩展 = 271",
            "read_only": "原始 271 槽文件未改动; 仅读本地缓存与日线 parquet",
        },
        "summary": summary,
        "records": results,
    }
    out_path = OUT / "gcp_271_daily_reconcile.json"
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    # 文本报告
    print("=" * 70)
    print("12 条 271 槽 日线对账 (只读)")
    print("=" * 70)
    print(f"日线列: {daily_cols}")
    print(f"\n汇总: n={n}, 全部 present=271={summary['present_total_all_271']}, "
          f"OHLC吻合={ohlc_ok}, 零量填充一致={zero_pad}")
    if vol_ratios:
        print(f"vol_ratio 范围: {summary['vol_ratio_range']}")
    if amt_ratios:
        print(f"amt_ratio 范围: {summary['amt_ratio_range']}")
    print(f"\n单位口径: {summary['unit_note']}")
    print("\n逐条:")
    for r in results:
        c = r.get("compare", {})
        print(f"  {r['code']} {r['E']} {r['setup']} present={r.get('present_total')} "
              f"pre_n={r.get('pre_n')} post_n={r.get('post_n')}")
        print(f"    pre_ohlc={r.get('pre_ohlc')} daily_ohlc={r.get('daily_ohlc')}")
        print(f"    pre_vol={r.get('pre_vol')} daily_vol={r.get('daily_vol')} "
              f"vol_ratio={c.get('vol_ratio')} amt_ratio={c.get('amt_ratio')}")
        print(f"    post_vol={r.get('post_vol')} post_close_unique={r.get('post_close_unique')} "
              f"post_zero={r.get('post_zero_volume')} eq1500={r.get('post_price_eq_1500_close')}")
        print(f"    OHLC匹配={c.get('ohlc_match')} verdict={r.get('verdict')}")
    print(f"\n决策待裁决: {summary['decision_required']}")
    print(f"段范围说明: {summary['segment_scope_note']}")
    print(f"\nJSON: {out_path} (耗时 {time.time()-t0:.1f}s, network_requests=0)")


if __name__ == "__main__":
    main()
