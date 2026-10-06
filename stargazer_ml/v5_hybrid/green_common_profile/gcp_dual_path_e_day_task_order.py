"""双路径 E 日定向抓取任务单生成 (2026-10-01; 仅起草, 不抓取).

依据已签署 dual_path_event_execution_contract.md v0.1:
  - 从 2025+ main(Swing≤2.5) 箱体成员推导 E 日, 按 (code,E) 精确去重
  - 逐 (code,E) 核验缓存: HIT_VALID / HIT_INVALID / NEED
  - 输出逐月、A/B、已用/未用、时间切分分布与文件预算
  - 双轨 SHA256: 规范化内容哈希 + 文件字节哈希
本脚本 0 网络请求; 任务单须另行签署才可执行抓取.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "step9_preflight"))
sys.path.insert(0, str(HERE))
from data_source_provenance import get_daily_parquet, DAILY_PARTS  # noqa: E402
import gcp_round_b_engine as E  # noqa: E402
import gcp_dual_path_coverage_probe as P  # noqa: E402

CONTRACTS = HERE / "contracts"
LIST_FILE = CONTRACTS / "dual_path_e_day_fetch_list.txt"
SUBSET_FILE = CONTRACTS / "dual_path_e_day_request_subset.txt"
OUTSIDE_FILE = CONTRACTS / "dual_path_e_day_outside_cutoff.txt"
TASK_MD = CONTRACTS / "dual_path_e_day_task_order.md"
REGISTRY = CONTRACTS / "dual_path_e_day_registry.json"

TRAIN_E_MAX = 20260305
VAL_E_MIN, VAL_E_MAX = 20260309, 20260609
TEST_E_MIN = 20260610


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def segment_of(e_date: int) -> str:
    if e_date <= TRAIN_E_MAX:
        return "train"
    if VAL_E_MIN <= e_date <= VAL_E_MAX:
        return "val"
    if e_date >= TEST_E_MIN:
        return "test"
    return "gap"


def main():
    t0 = time.time()
    print("=" * 60)
    print("E 日定向抓取任务单生成 (仅起草)")
    print("=" * 60)

    panel = get_daily_parquet()
    panel["code"] = panel["code"].astype(str).str.zfill(6)
    panel["date"] = panel["date"].astype(int)
    cal = sorted(int(d) for d in panel["date"].unique())
    cal_max = cal[-1]
    cal_idx = {d: i for i, d in enumerate(cal)}
    daily_by_code = {c: g.sort_values("date").reset_index(drop=True)
                     for c, g in panel.groupby("code")}

    # ---- 日线数据快照标识 (用于可复现核验) ----
    import glob as _glob
    parts = sorted(_glob.glob(os.path.join(DAILY_PARTS, "part_*.parquet")))
    _h = hashlib.sha256()
    _total = 0
    last_part = None
    for p in parts:
        b = Path(p).read_bytes()
        _h.update(b)
        _total += len(b)
        last_part = (Path(p).name, os.path.getsize(p), int(os.path.getmtime(p)))
    daily_snapshot = {
        "parts_dir": DAILY_PARTS,
        "num_parts": len(parts),
        "total_bytes": _total,
        "agg_sha256": _h.hexdigest(),
        "last_part_name": last_part[0] if last_part else None,
        "last_part_size": last_part[1] if last_part else None,
        "last_part_mtime": last_part[2] if last_part else None,
        "calendar_max_date": cal_max,
        "calendar_trading_days": len(cal),
    }
    print(f"日线快照: parts={len(parts)} total={_total} cal_max={cal_max} agg_sha={daily_snapshot['agg_sha256'][:16]}")

    pool_a = pd.read_parquet(P.POOL_DIR / "pool_a_sdp_members.parquet")
    pool_b = pd.read_parquet(P.POOL_DIR / "pool_b_lvb_members.parquet")
    pool_a["code"] = pool_a["code"].astype(str).str.zfill(6)
    pool_b["code"] = pool_b["code"].astype(str).str.zfill(6)
    members = sorted({(c, int(p), "A_SDP") for c, p in pool_a[["code", "date"]].itertuples(index=False)}
                     | {(c, int(p), "B_LVB") for c, p in pool_b[["code", "date"]].itertuples(index=False)})
    used_keys = P.load_used_keys()
    minute_idx = P.load_minute_index()
    grid = json.loads((P.OUT / "minute_grid.json").read_text(encoding="utf-8"))["grid"]

    # ---- main 箱体成员 (2025+), 聚合到 (code,E) ----
    grouped: dict = {}
    outside: list = []  # E 不在已冻结数据日历 (超出截止范围), 不冒充未来日期
    n_box_members = 0
    for code, p, setup in members:
        if p < P.MINUTE_ERA_START:
            continue
        dc = daily_by_code.get(code)
        if dc is None:
            continue
        loc = int(dc["date"].searchsorted(p, side="right"))
        if loc < 20:
            continue
        d_sub = dc.iloc[:loc]
        atr_ref = E.build_atr_ref(d_sub)
        calibs = P.evaluate_calibers(d_sub, atr_ref)
        cb = calibs.get("main", {})
        if not cb.get("valid"):
            continue
        n_box_members += 1
        i_p = cal_idx.get(p)
        if i_p is None or i_p + 1 >= len(cal):
            outside.append({"code": code, "P": p, "setup": setup})
            continue
        e = cal[i_p + 1]
        key = (code, e)
        g = grouped.setdefault(key, {"P": p, "setups": set()})
        g["setups"].add(setup)
    print(f"main 箱体成员(2025+)={n_box_members}; 去重 (code,E)={len(grouped)}; "
          f"超出数据日历截止={len(outside)}")

    # ---- 缓存逐键分类 ----
    records = []
    for (code, e), g in sorted(grouped.items()):
        mkeys = minute_idx.get((code, e))
        cache_state = "NEED"
        cache_source = None
        cache_reason = None
        if mkeys:
            cache_source = mkeys[0][1]
            try:
                raw = pd.read_parquet(mkeys[0][0])
                di = raw["date"].astype(str).str[:8].astype(int)
                sub = raw[di == e]
                vd = E.validate_minute_day(sub, grid, e)
                cache_state = "HIT_VALID" if vd["ok"] else "HIT_INVALID"
                if not vd["ok"]:
                    cache_reason = vd.get("reason", "UNKNOWN")
            except Exception as exc:
                cache_state = "HIT_INVALID"
                cache_reason = f"READ_ERROR:{type(exc).__name__}"
        records.append({
            "code": code, "E": e, "P": g["P"],
            "setups": sorted(g["setups"]),
            "cache": cache_state, "cache_source": cache_source,
            "cache_reason": cache_reason,
            "used_before": (code, e) in used_keys,
            "segment": segment_of(e),
            "n_setups": len(g["setups"]),
        })

    n = len(records)
    n_need = sum(r["cache"] == "NEED" for r in records)
    n_hit_valid = sum(r["cache"] == "HIT_VALID" for r in records)
    n_hit_invalid = sum(r["cache"] == "HIT_INVALID" for r in records)
    n_dual_setup = sum(r["n_setups"] == 2 for r in records)
    print(f"缓存: HIT_VALID={n_hit_valid}, HIT_INVALID={n_hit_invalid}, NEED={n_need}")
    print(f"A&B 双池同 (code,E): {n_dual_setup}")

    # ---- 分布 ----
    by_month = Counter(str(r["E"])[:6] for r in records)
    by_month_need = Counter(str(r["E"])[:6] for r in records if r["cache"] == "NEED")
    by_setup: dict = {"A_SDP": Counter(), "B_LVB": Counter()}
    for r in records:
        for s in r["setups"]:
            by_setup[s][r["cache"]] += 1
    by_seg = defaultdict(Counter)
    for r in records:
        by_seg[r["segment"]]["total"] += 1
        by_seg[r["segment"]][r["cache"]] += 1
    by_used = Counter(("used" if r["used_before"] else "unused") + ":" + r["cache"]
                      for r in records)

    # ---- 文件预算 (实测均值) ----
    total_bytes = 0
    total_days = 0
    for d in P.MIN_DIRS:
        if not d.exists():
            continue
        for pf in d.glob("*.parquet"):
            try:
                m = pd.read_parquet(pf, columns=["date"])
                nd = len(pd.unique(m["date"].astype(str).str[:8].astype(int)))
                total_bytes += pf.stat().st_size
                total_days += nd
            except Exception:
                pass
    avg_bytes_day = total_bytes / total_days if total_days else 0.0
    est_new_bytes = int(avg_bytes_day * n_need)

    snap_bytes = sum(f.stat().st_size for d in (P.OUT,) for f in d.rglob("*") if f.is_file())
    budget_after = snap_bytes + est_new_bytes
    budget_ok = budget_after <= 50 * 1024 * 1024

    print(f"实测缓存日均 {avg_bytes_day/1024:.1f} KB; 新增估算 {est_new_bytes/1024/1024:.2f} MB")
    print(f"当前 snapshot={snap_bytes/1024/1024:.2f}MB; 抓取后={budget_after/1024/1024:.2f}MB "
          f"(≤50: {budget_ok})")

    # ---- 主清单: 逐行缓存状态 (code,E,P,setups,cache[,reason]) ----
    lines = [f"{r['code']},{r['E']},{r['P']},{'+'.join(r['setups'])},{r['cache']}"
             for r in records]
    content = "\n".join(lines)
    content_hash = sha256_text(content)
    LIST_FILE.write_text(content + "\n", encoding="utf-8")
    file_hash = sha256_file(LIST_FILE)

    # ---- 实际请求子集: 仅 NEED 行, 执行器只认本文件 ----
    subset_lines = [f"{r['code']},{r['E']},{r['P']},{'+'.join(r['setups'])}"
                    for r in records if r["cache"] == "NEED"]
    subset_content = "\n".join(subset_lines)
    subset_content_hash = sha256_text(subset_content)
    SUBSET_FILE.write_text(subset_content + "\n", encoding="utf-8")
    subset_file_hash = sha256_file(SUBSET_FILE)

    # ---- 超出冻结数据日历截止的成员 (单列, 不冒充未来日期) ----
    outside_merged: dict = {}
    for o in outside:
        og = outside_merged.setdefault((o["code"], o["P"]), set())
        og.add(o["setup"])
    outside_lines = [f"{c},{p},{'+'.join(sorted(s))}"
                     for (c, p), s in sorted(outside_merged.items())]
    outside_content = "\n".join(outside_lines)
    outside_content_hash = sha256_text(outside_content)
    OUTSIDE_FILE.write_text(outside_content + "\n", encoding="utf-8")
    outside_file_hash = sha256_file(OUTSIDE_FILE)

    invalid_records = [r for r in records if r["cache"] == "HIT_INVALID"]

    # ---- 任务单 MD ----
    md = []
    md.append("# E 日定向抓取任务单（双路径 v0.1 配套）")
    md.append("")
    md.append("> 状态: **待签署；签署前不启动抓取**；日期: 2026-10-01")
    md.append("> 依据: 已签署 dual_path_event_execution_contract.md v0.1")
    md.append("")
    md.append("## 1. 范围")
    md.append("")
    md.append(f"- 来源: 2025+ main(Swing≤2.5) 箱体成员 {n_box_members} 个 (code,P,setup);")
    md.append(f"- 物理单位: 去重后 **{n} 个 (code,E)**（A&B 双池同键 {n_dual_setup} 个，只抓一次）;")
    md.append(f"- 缓存逐键核验: HIT_VALID **{n_hit_valid}**, HIT_INVALID **{n_hit_invalid}**, "
              f"NEED **{n_need}** → **冻结的实际请求子集 = {n_need} 行**（见 §9 子集文件）;")
    md.append(f"- 另有 **{len(outside_merged)}** 个 (code,P) 成员的 E 日"
              "**超出本次冻结数据日历/截止范围**（见 §8），不纳入本批、不冒充未来日期；")
    md.append("- 源缺（SOURCE_MISSING）只有抓取时才能确认；失败按合同显式保留，不找替补、不扩候选.")
    md.append("")
    md.append("主清单 schema（CSV，表头即列名）: `code,E,P,setups,cache`；"
              "cache ∈ HIT_VALID/HIT_INVALID/NEED，逐行冻结，机器可直接筛 NEED 复核.")
    md.append("")
    md.append("## 2. 逐月分布（全部 / 其中 NEED）")
    md.append("")
    md.append("| 月 | 全部 | NEED |")
    md.append("|---|---|---|")
    for mth in sorted(by_month):
        md.append(f"| {mth} | {by_month[mth]} | {by_month_need.get(mth, 0)} |")
    md.append("")
    md.append("## 3. A/B 分布（setup 成员计数；同键双池两边都计）")
    md.append("")
    md.append("| Setup | HIT_VALID | HIT_INVALID | NEED |")
    md.append("|---|---|---|---|")
    for s in ("A_SDP", "B_LVB"):
        c = by_setup[s]
        md.append(f"| {s} | {c['HIT_VALID']} | {c['HIT_INVALID']} | {c['NEED']} |")
    md.append("")
    md.append("## 4. 时间切分（固定日期边界；与既有研究口径一致，不重切）")
    md.append("")
    md.append("| 段 | 边界 | 去重(code,E) | HIT_VALID | HIT_INVALID | NEED |")
    md.append("|---|---|---|---|---|---|")
    bounds = {"train": "E≤20260305", "val": "20260309–20260609",
              "test": "E≥20260610", "gap": "边界外"}
    for seg in ("train", "val", "test", "gap"):
        c = by_seg[seg]
        md.append(f"| {seg} | {bounds[seg]} | {c['total']} | {c['HIT_VALID']} "
                  f"| {c['HIT_INVALID']} | {c['NEED']} |")
    md.append("")
    md.append("## 5. 已用 / 未用集合")
    md.append("")
    md.append("```")
    for k in sorted(by_used):
        md.append(f"{k}: {by_used[k]}")
    md.append("```")
    md.append("")
    md.append("## 6. 文件预算（实测，非系数估算）")
    md.append("")
    md.append(f"- 现有小样本缓存日均: {avg_bytes_day/1024:.1f} KB/(code,日);")
    md.append(f"- 新增 {n_need} 日估算: **{est_new_bytes/1024/1024:.2f} MB**;")
    md.append(f"- snapshot 当前 {snap_bytes/1024/1024:.2f} MB，抓取后 "
              f"**{budget_after/1024/1024:.2f} MB**，≤50MB: **{budget_ok}**;")
    md.append("- 沿用每请求 ≤3 次有限重试；重试计入请求量；超预算或异常即停并报告.")
    md.append("")
    md.append("## 7. HIT_INVALID 四条明细与处置（冻结）")
    md.append("")
    md.append("| code | E | 已用集合 | 失效原因 | 本批处置 |")
    md.append("|---|---|---|---|---|")
    for r in invalid_records:
        md.append(f"| {r['code']} | {r['E']} | {'used' if r['used_before'] else 'unused'} "
                  f"| {r['cache_reason']} | **不重抓** |")
    md.append("")
    md.append("- **本批一律不重抓这 4 条**，不进入请求子集；同源同日重取不会产生新网格，"
              "且本批请求子集已冻结；")
    md.append("- 它们保留为本地失效缓存并计入 HIT_INVALID，若日后需要，另立\"缓存修复\"任务单，"
              "不占本批请求数、不影响 §6 预算.")
    md.append("")
    md.append("## 8. 超出冻结数据日历截止的成员（22 个 code,P）")
    md.append("")
    md.append("- 事实口径（按交易所 2026-09-15 公告）: 9 月 28 日**已恢复交易**，10 月 1 日起国庆休市；")
    md.append(f"  因此 P=20260928 的 E 日**不是\"尚未来到\"**，而是其后第一个交易日尚未进入"
              f"本次冻结的**日线数据快照（快照止于 {cal_max}，见 §12）**；")
    md.append("- 处置: 保留在本批 1,072 成员口径内备查，但不进入 1,050 物理单位、不产生请求；"
              "待日线快照延展后另批处理，清单见 `dual_path_e_day_outside_cutoff.txt`.")
    md.append("")
    md.append("| code | P | setups |")
    md.append("|---|---|---|")
    for (c, p), s in sorted(outside_merged.items()):
        md.append(f"| {c} | {p} | {'+'.join(sorted(s))} |")
    md.append("")
    md.append("## 9. 实际请求子集（执行器唯一输入）")
    md.append("")
    md.append(f"- 文件 `dual_path_e_day_request_subset.txt`，{n_need} 行，schema `code,E,P,setups`;")
    md.append("- 执行时只允许请求本文件内的 (code,E)；成功/新增/源缺/失败计数之和须等于本文件行数；")
    md.append("- 主清单 1,050 行为登记与复核依据，不直接驱动请求.")
    md.append("")
    md.append("## 10. 能力边界（重要）")
    md.append("")
    md.append("- 本任务单**只补 E 日分钟**，作用是补齐行动漏斗（WATCH/S1/S2 的识别与候选成交检查）;")
    md.append("- **单凭 E 日无法完成 H3/H5 成本后收益与退出验证**——还需 E 之后交易日数据，"
              "届时另列数据需求与预算，不包含在本单；")
    md.append("- 现有 31 个模拟可成交中 24 个来自已用集合、S2 仅 5 个，不作为独立盈利证据；")
    md.append("- 2025–2026 已多轮参与设计，不称新盲测；")
    md.append("- 横盘池 C：\"无需 C\"为待检假设，本单不涉及 A/B 之外横盘占比检验；")
    md.append("- 抓取与训练授权相互独立：签署本单只授权按子集采集，不含训练与买入建议.")
    md.append("")
    md.append("## 11. 日线数据快照核验（只读）")
    md.append("")
    md.append("- 本次实际加载的日线数据快照（35 个 part_*.parquet 聚合）:")
    md.append(f"  - 交易日数: **{daily_snapshot['calendar_trading_days']}**；")
    md.append(f"  - 日历最大日期: **{daily_snapshot['calendar_max_date']}**；")
    md.append(f"  - 分片目录: `{daily_snapshot['parts_dir']}`;")
    md.append(f"  - 最后分片: `{daily_snapshot['last_part_name']}` "
              f"(size={daily_snapshot['last_part_size']}, mtime={daily_snapshot['last_part_mtime']});")
    md.append(f"  - 总字节数: {daily_snapshot['total_bytes']};")
    md.append(f"  - 聚合 SHA256: `{daily_snapshot['agg_sha256']}`.")
    md.append("")
    md.append(f"- **22 条无法映射的原因**: P=20260928 是该快照中的最后一个交易日，"
              f"其后第一个交易日未进入本快照（cal_max={cal_max}），故 `i_p+1 >= len(cal)` 归入截止外；")
    md.append("  该处置不改变执行范围，不影响 952 请求子集；待日线快照延展后另批处理。")
    md.append("")
    md.append("## 12. 登记哈希")
    md.append("")
    md.append("| 文件 | 内容哈希(UTF-8/LF/无末尾换行) | 文件字节哈希 |")
    md.append("|---|---|---|")
    md.append(f"| 主清单 1050 行 | `{content_hash}` | `{file_hash}` |")
    md.append(f"| 请求子集 {n_need} 行 | `{subset_content_hash}` | `{subset_file_hash}` |")
    md.append(f"| 截止外 {len(outside_merged)} 行 | `{outside_content_hash}` | `{outside_file_hash}` |")
    md.append("")
    TASK_MD.write_text("\n".join(md), encoding="utf-8")

    # ---- registry ----
    registry = {
        "version": "dual_path_e_day_registry",
        "date": "2026-10-01",
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "contract_ref": "dual_path_event_execution_contract.md (SIGNED v0.1)",
        "network_requests": 0,
        "calendar_max_date": cal_max,
        "daily_snapshot": daily_snapshot,
        "counts": {"box_members_2025plus": n_box_members,
                   "unique_codeE": n, "dual_setup_keys": n_dual_setup,
                   "hit_valid": n_hit_valid, "hit_invalid": n_hit_invalid,
                   "planned_requests_need": n_need,
                   "outside_cutoff_codeP": len(outside_merged)},
        "by_month": {m: {"total": by_month[m], "NEED": by_month_need.get(m, 0)}
                     for m in sorted(by_month)},
        "by_setup": {s: dict(by_setup[s]) for s in by_setup},
        "by_segment": {s: dict(by_seg[s]) for s in by_seg},
        "by_used_state": dict(by_used),
        "invalid_details": [{"code": r["code"], "E": r["E"],
                             "used_before": r["used_before"],
                             "reason": r["cache_reason"],
                             "disposition": "NOT_REFETCHED_IN_BATCH"}
                            for r in invalid_records],
        "outside_cutoff": [{"code": c, "P": p, "setups": sorted(s)}
                           for (c, p), s in sorted(outside_merged.items())],
        "budget": {"avg_bytes_per_day": round(avg_bytes_day, 1),
                   "est_new_bytes": est_new_bytes,
                   "snapshot_bytes_now": snap_bytes,
                   "snapshot_bytes_after": budget_after,
                   "within_50mb": budget_ok},
        "list_content_hash": content_hash,
        "list_file_byte_hash": file_hash,
        "request_subset_content_hash": subset_content_hash,
        "request_subset_file_byte_hash": subset_file_hash,
        "outside_cutoff_content_hash": outside_content_hash,
        "outside_cutoff_file_byte_hash": outside_file_hash,
        "data_hashes": {
            "pool_a_sdp_members.parquet": P.sha256_file(P.POOL_DIR / "pool_a_sdp_members.parquet"),
            "pool_b_lvb_members.parquet": P.sha256_file(P.POOL_DIR / "pool_b_lvb_members.parquet"),
            "minute_grid.json": P.sha256_file(P.OUT / "minute_grid.json"),
        },
        "capabilities": {
            "e_day_only": True,
            "h3_h5_possible_with_e_day_alone": False,
            "training_authorized": False,
            "buy_advice_authorized": False,
        },
        "elapsed_s": round(time.time() - t0, 1),
    }
    REGISTRY.write_text(json.dumps(registry, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n产出:\n  " + "\n  ".join([LIST_FILE.name, SUBSET_FILE.name, OUTSIDE_FILE.name,
                                       TASK_MD.name, REGISTRY.name]))
    print(f"耗时 {registry['elapsed_s']}s")


if __name__ == "__main__":
    main()
