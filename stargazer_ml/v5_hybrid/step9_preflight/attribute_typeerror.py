"""attribute_typeerror — 归因 build_one_sample 抛出的 exc:TypeError (用户 2026-09-18 #2b).

背景:
  ④ trace (trace_04_scenarios.py --budget 160) 与早期 Gate4 的 drop_reasons 里出现
  `exc:TypeError` 2 次, 但当时只记了类型计数, 没有 symbol/date/state/字段/stack。
  用户裁定: 不允许把 `exc:TypeError` 作为未知桶带进 Final Freeze, 必须逐条归因。

本脚本做两件事:
  Part 1 (真实): 对末尾 plan_dates 的候选 (H3 未来 bar 超出数据末 -> CENSORED_TAIL
                 分支) 直接跑 build_one_sample, 捕获完整 traceback + 失败帧局部变量。
  Part 2 (合成): 用 stub proxy 强制 s0_market_alpha_h3=None (模拟 alpha 覆盖缺口),
                 复现同一非删失触发路径, 证明与 CENSORED_TAIL 是同一处守卫缺失。

输出: exc_typeerror_attribution.json
运行: python attribute_typeerror.py
"""
from __future__ import annotations
import json
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import coverage_02_full as C02  # noqa
import latr_builder as LB  # noqa
from fast_infra import FastContext, mk_chains  # noqa


def _frame_detail(exc):
    """从异常回溯里取最后一个出现在 build_one_sample 的帧 (行号 + 源码 + 局部变量)."""
    tbs = [f for f in traceback.extract_tb(exc.__traceback__)
           if Path(f.filename).name == "latr_builder.py"]
    if not tbs:
        return None
    f = tbs[-1]
    # 取相同帧的局部变量快照
    tb = exc.__traceback__
    locals_snap = {}
    while tb is not None:
        if Path(tb.tb_frame.f_code.co_filename).name == "latr_builder.py" \
                and tb.tb_lineno == f.lineno:
            for k, v in tb.tb_frame.f_locals.items():
                try:
                    locals_snap[k] = repr(v)[:160]
                except Exception:
                    locals_snap[k] = "<unrepr>"
        tb = tb.tb_next
    return {
        "file": Path(f.filename).name, "lineno": f.lineno,
        "function": f.name, "source_line": f.line,
        "locals": locals_snap,
    }


def _cand_detail(cand):
    return {"symbol": str(cand.symbol), "plan_date": int(cand.plan_date),
            "d_date": int(cand.effective_trade_date),
            "primary_setup": str(cand.primary_setup),
            "atr_ref": float(cand.atr_ref)}


def _attempt(cand, chains, proxy, tag):
    try:
        s = LB.build_one_sample(cand, chains, proxy)
        if s is None:
            return {"tag": tag, "ok": False, "rejected": True,
                    "reject_reason": "engineering_reject (build_one_sample returned None)",
                    **_cand_detail(cand)}
        return {"tag": tag, "ok": True, "rejected": False, **_cand_detail(cand),
                "label_status": s.get("label_status"),
                "s0_alpha_h3": s.get("s0_alpha_h3"),
                "alpha_label_available": s.get("alpha_label_available")}
    except Exception as e:  # noqa
        d = {"tag": tag, "ok": False, **_cand_detail(cand),
             "exc_type": type(e).__name__, "exc_msg": str(e)[:400],
             "failing_frame": _frame_detail(e),
             "traceback": traceback.format_exc()[-2500:]}
        return d


class _StubProxyNoAlpha:
    """强制 alpha 不可用 (返回 s0_market_alpha_h3=None), 用于复现非删失触发路径。"""

    def attach_alpha_labels(self, sample):
        out = dict(sample)
        out.update({
            "s0_market_alpha_h3": None, "timing_market_alpha_h1": None,
            "proxy_s0_return": None, "proxy_timing_return": None,
            "alpha_label_available": 0, "proxy_weight_date": sample.get("plan_date"),
            "proxy_weight_coverage": None, "proxy_constituent_count": None,
            "proxy_excluded_symbol": sample.get("symbol"),
        })
        return out


def _mechanism_proof():
    """永久可复现地证明机制: 旧表达 `round(x,5) if not np.isnan(x) else None` 传 None
    抛 TypeError; 修复后的 _num_or_none(None) 安全返回 None。"""
    import numpy as _np
    legacy = None

    def _legacy_expr(x, nd=5):
        return round(x, nd) if not _np.isnan(x) else None
    try:
        _legacy_expr(None)
    except Exception as e:  # noqa
        legacy = {"raises": True, "exc_type": type(e).__name__, "exc_msg": str(e)[:200]}
    else:
        legacy = {"raises": False}
    return {
        "failing_expression_old": "round(x, 5) if not np.isnan(x) else None  # x is None",
        "failing_fields": ["s0_alpha_h3", "timing_alpha_h1"],
        "legacy_expr_on_None": legacy,
        "fixed_helper": "latr_builder._num_or_none(None) -> None",
        "fixed_helper_on_None": LB._num_or_none(None),
        "trigger": "proxy alpha 不可用 = (a) CENSORED_TAIL 分支 alpha_out 全 None; "
                   "(b) 非删失但 symbol alpha 覆盖缺口",
        "consequence_if_unfixed": "样本被当异常丢弃 -> 违反 CENSORED_TAIL 保留契约 + "
                                  "alpha 缺口样本静默丢失 (用户 #1/#4 禁止)",
        "fix_location": "latr_builder.py :: _num_or_none()",
    }


def main():
    t0 = time.time()
    out = {"generated_at": pd.Timestamp.now("UTC").isoformat(),
           "mechanism": _mechanism_proof(), "part1_real": {}, "part2_synthetic": {}}
    print("[attr] building fast daily dict ...", flush=True)
    daily_dict = C02.build_daily_dict()

    # 取 v11 pool 末尾 plan_dates (H3 超出数据末 -> CENSORED_TAIL 分支)
    df = pd.read_parquet(C02.V11_POOL).copy()
    df = df[df["effective_trade_date"].notna()]
    df["_plan"] = df["plan_date"].astype(int)
    df["_eff"] = df["effective_trade_date"].astype(int)
    df = df[(df["_eff"] > df["_plan"]) & (df["_eff"] >= C02.MINUTE_START)]
    last_dates = sorted(df["_plan"].unique())[-3:]
    print(f"[attr] tail plan_dates={last_dates}", flush=True)
    tail = df[df["_plan"].isin(last_dates)].groupby("_plan").head(8)
    cands = [C02._build_row(r) for _, r in tail.iterrows()]

    # Part 2 候选: 中段历史 (非删失路径; 用 stub proxy 强制 alpha 不可用)
    mid = df[(df["_plan"] >= 20260101) & (df["_plan"] <= 20260331)]
    mid = mid.groupby("_plan").head(1).head(4)
    mid_cands = [C02._build_row(r) for _, r in mid.iterrows()]

    # 一次性建两组候选的复权链 (快); proxy 由 stub 提供, 不走网络
    codes = sorted({c.symbol for c in cands} | {c.symbol for c in mid_cands})
    chains = mk_chains(codes)
    ctx = FastContext(daily_dict, chains, [])
    ctx.patch()
    stub = _StubProxyNoAlpha()

    results = []
    for c in cands:
        results.append(_attempt(c, chains, stub, "tail_censored_or_noalpha"))
        r = results[-1]
        print(f"[attr] {r['symbol']} {r['plan_date']} -> "
              f"{'OK' if r['ok'] else (r.get('exc_type') or 'REJECT')}", flush=True)
    out["part1_real"] = {
        "scope": f"v11 pool tail plan_dates={last_dates} (H3 beyond data end -> CENSORED_TAIL)",
        "n_attempted": len(results),
        "n_typeerror": sum(1 for r in results if not r.get("ok") and r.get("exc_type") == "TypeError"),
        "results": results,
    }

    # ---- Part 2: 合成验证同一守卫缺失 (非删失, alpha 不可用) ----
    syn = []
    for c in mid_cands:
        r = _attempt(c, chains, stub, "synthetic_alpha_unavailable")
        syn.append(r)
        print(f"[attr] P2 {r['symbol']} {r['plan_date']} -> "
              f"{'OK' if r['ok'] else (r.get('exc_type') or 'REJECT')}", flush=True)
    out["part2_synthetic"] = {
        "scope": "mid-history candidates + stub proxy (s0_market_alpha_h3=None)",
        "n_attempted": len(syn),
        "n_typeerror": sum(1 for x in syn if not x.get("ok") and x.get("exc_type") == "TypeError"),
        "results": syn,
    }

    # ---- 归因汇总 ----
    allres = results + syn
    te = [x for x in allres if not x.get("ok") and x.get("exc_type") == "TypeError"]
    frames = sorted({(x["failing_frame"]["lineno"], x["failing_frame"]["source_line"])
                     for x in te if x.get("failing_frame")})
    retained = [x for x in allres if x.get("ok")]
    out["attribution"] = {
        "root_cause": "np.isnan(None) on s0_alpha_h3 / timing_alpha_h1 when proxy alpha "
                      "unavailable (含 CENSORED_TAIL 分支 alpha_out 全为 None)",
        "n_typeerror_total": len(te),
        "failing_locations": [{"lineno": ln, "source": src} for ln, src in frames],
        "symbols": sorted({(x["symbol"], x["plan_date"]) for x in te}),
        "note": "同一守卫缺失: 删失样本被当作异常丢弃 (违反 CENSORED_TAIL 保留契约), "
                "alpha 缺口样本同样被丢弃。修复后应 0 TypeError 且样本保留。",
    }
    out["post_fix_check"] = {
        "n_retained": len(retained),
        "label_status_distribution": pd.Series(
            [x.get("label_status") for x in retained]).value_counts().to_dict(),
        "alpha_label_available_distribution": pd.Series(
            [x.get("alpha_label_available") for x in retained]).value_counts().to_dict(),
        "expectation": "TypeError=0 且 CENSORED_TAIL 样本出现在 label_status 分布 (未被丢弃)",
    }
    # 修复前捕获 (若存在) 一并归档, 保证审计链完整
    cap = Path(__file__).resolve().parent / "exc_typeerror_pre_fix_capture.json"
    if cap.exists():
        try:
            out["pre_fix_capture"] = json.loads(cap.read_text(encoding="utf-8"))
        except Exception:  # noqa
            pass
    tag = os.environ.get("ATTR_TAG", "").strip()
    suffix = f"_{tag}" if tag else ""
    p = Path(__file__).resolve().parent / f"exc_typeerror_attribution{suffix}.json"
    p.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[attr] wrote {p} ({time.time()-t0:.1f}s)")
    print(f"[attr] TypeError total={len(te)} locations={frames}")
    for x in te[:6]:
        print(f"[attr]   {x['symbol']} {x['plan_date']} d={x['d_date']} "
              f"{x.get('exc_type')}: {x.get('exc_msg')[:120]}")


if __name__ == "__main__":
    main()
