"""CAR v0.2 扩样冻结清单生成 (2026-09-29).

按 v0.2 扩样方案 v2 生成抽样清单:
  - 时间分层: 按月份分桶, 配额 = round(月占比 * 200), 四舍五入;
    合计 > 200 时从最晚月份开始扣减, 合计 < 200 时从最早月份补齐.
  - 同股连续 episode 过滤: 同一 code 相邻 P 日期间隔 < 5 交易日只保留最早一条.
  - 排除已有 33 候选 (code, E).
  - 随机种子 seed=42, 可复现.
  - 登记 SHA256、缓存命中数、请求上限.
"""
from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "step9_preflight"))
sys.path.insert(0, str(HERE))
from data_source_provenance import get_daily_parquet  # noqa: E402

CONTRACTS = HERE / "contracts"
POOL_DIR = HERE / "round_a_v0_1_impl_fix"
OUT = HERE / "gcp_round_b_small_sample_v1"
SNAP_CAR = OUT / "minute_car"
SNAP_OLD = OUT / "minute"
SNAP_EXT = OUT / "minute_ext"
ADJ_CACHE = OUT / "adj_cache.json"

SEED = 42
TARGET_PER_POOL = 200
EPISODE_MIN_GAP = 5  # 同股相邻 P 日最小间隔 (交易日)


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    rng = np.random.RandomState(SEED)

    # ---- 加载冻结池 ----
    pool_a = pd.read_parquet(POOL_DIR / "pool_a_sdp_members.parquet")
    pool_b = pd.read_parquet(POOL_DIR / "pool_b_lvb_members.parquet")

    # 2025+ (分钟覆盖范围)
    pool_a = pool_a[pool_a["date"] >= 20250102].copy()
    pool_b = pool_b[pool_b["date"] >= 20250102].copy()
    n_a_raw, n_b_raw = len(pool_a), len(pool_b)

    # 统一列名
    pool_a["code"] = pool_a["code"].astype(str).str.zfill(6)
    pool_b["code"] = pool_b["code"].astype(str).str.zfill(6)
    pool_a["setup"] = "A_SDP"
    pool_b["setup"] = "B_LVB"

    # 合并两池
    pool = pd.concat([
        pool_a[["date", "code", "setup"]].rename(columns={"date": "P"}),
        pool_b[["date", "code", "setup"]].rename(columns={"date": "P"}),
    ], ignore_index=True)
    pool = pool.drop_duplicates(subset=["code", "P", "setup"]).sort_values(["P", "code"]).reset_index(drop=True)
    n_merged = len(pool)

    # 交易日历
    panel = get_daily_parquet()
    cal = sorted(int(d) for d in panel["date"].unique())
    cal_idx = {d: i for i, d in enumerate(cal)}

    # E = P 的下一交易日
    pool["E"] = pool["P"].map(lambda p: cal[cal_idx[p] + 1] if p in cal_idx and cal_idx[p] + 1 < len(cal) else None)
    n_before_e_map = len(pool)
    pool = pool.dropna(subset=["E"]).copy()
    pool["E"] = pool["E"].astype(int)
    n_e_map_dropped = n_before_e_map - len(pool)

    # 排除已有 33 候选
    sampling = json.loads((CONTRACTS / "car_v0_1_sampling_summary.json").read_text(encoding="utf-8"))
    existing = set((str(c["code"]).zfill(6), int(c["E"])) for c in sampling["selected"])
    n_before_exclude = len(pool)
    pool = pool[~pool.apply(lambda r: (r["code"], r["E"]) in existing, axis=1)].copy()
    n_after_exclude = len(pool)
    n_excluded_33 = n_before_exclude - n_after_exclude

    # ---- Episode 过滤 ----
    pool = pool.sort_values(["code", "P"]).reset_index(drop=True)
    keep_mask = pd.Series(True, index=pool.index)
    last_p = {}  # code -> last P date kept
    for idx, row in pool.iterrows():
        code = row["code"]
        p = int(row["P"])
        if code in last_p:
            gap = cal_idx.get(p, 0) - cal_idx.get(last_p[code], 0)
            if gap < EPISODE_MIN_GAP:
                keep_mask.iloc[idx] = False
                continue
        last_p[code] = p
    pool = pool[keep_mask].reset_index(drop=True)
    n_episode_filtered = len(pool)

    # ---- 时间分层抽样 ----
    pool["month"] = pool["P"].astype(str).str[:6]
    month_counts = pool.groupby("month").size().sort_index()
    total = len(pool)

    selected = {"A_SDP": [], "B_LVB": []}

    for setup in ("A_SDP", "B_LVB"):
        sub = pool[pool["setup"] == setup].copy()
        sub_month = sub.groupby("month").size().sort_index()
        sub_total = len(sub)

        # 月份配额
        quotas = {}
        for m, cnt in sub_month.items():
            quotas[m] = round(cnt / sub_total * TARGET_PER_POOL)

        # 合计修正
        q_total = sum(quotas.values())
        if q_total > TARGET_PER_POOL:
            # 从最晚月份开始扣减
            months_sorted = sorted(quotas.keys(), reverse=True)
            excess = q_total - TARGET_PER_POOL
            for m in months_sorted:
                deduct = min(excess, quotas[m])
                quotas[m] -= deduct
                excess -= deduct
                if excess <= 0:
                    break
        elif q_total < TARGET_PER_POOL:
            # 从最早月份补齐
            months_sorted = sorted(quotas.keys())
            deficit = TARGET_PER_POOL - q_total
            for m in months_sorted:
                quotas[m] += deficit
                break  # 全部补给最早月

        # 按配额抽样
        for m, quota in quotas.items():
            if quota <= 0:
                continue
            m_sub = sub[sub["month"] == m]
            n_pick = min(quota, len(m_sub))
            if n_pick <= 0:
                continue
            idx = rng.choice(m_sub.index, size=n_pick, replace=False)
            selected[setup].extend(m_sub.loc[idx].to_dict("records"))

    # ---- 合并并去重 ----
    all_selected = []
    for setup in ("A_SDP", "B_LVB"):
        for r in selected[setup]:
            all_selected.append({
                "code": r["code"], "P": int(r["P"]), "E": int(r["E"]), "setup": setup,
            })

    # 去重 (code, E)
    seen = set()
    unique = []
    for r in all_selected:
        key = (r["code"], r["E"])
        if key not in seen:
            seen.add(key)
            unique.append(r)

    # 按 E 排序
    unique.sort(key=lambda x: (x["E"], x["code"]))

    # ---- 缓存命中统计 (按 (code,date) 核验) ----
    # 分钟缓存: 逐文件检查实际覆盖日期
    minute_cache_dates = {}  # code -> set of dates
    for snap_dir in (SNAP_OLD, SNAP_EXT, SNAP_CAR):
        if not snap_dir.exists():
            continue
        for pf in snap_dir.glob("*.parquet"):
            code = pf.stem
            try:
                m = pd.read_parquet(pf, columns=["date"])
                if "date" in m and not m.empty:
                    days = m["date"].astype(str).str[:8].astype(int).unique()
                    minute_cache_dates.setdefault(code, set()).update(days)
            except Exception:
                pass

    adj_data = json.loads(ADJ_CACHE.read_text(encoding="utf-8")) if ADJ_CACHE.exists() else {}

    # 去重 (code, date) 分钟请求
    minute_requests = set()
    for r in unique:
        code = r["code"]
        e_date = r["E"]
        minute_requests.add((code, e_date))
        # 下一交易日
        i_e = cal_idx.get(e_date, -1)
        if i_e >= 0 and i_e + 1 < len(cal):
            minute_requests.add((code, cal[i_e + 1]))

    # 缓存命中: 按 (code,date) 精确核验
    cached_minute = sum(1 for (c, d) in minute_requests if d in minute_cache_dates.get(c, set()))
    new_minute = len(minute_requests) - cached_minute

    # 复权请求: 检查 ok 状态
    adj_requests = set(r["code"] for r in unique)
    cached_adj = sum(1 for c in adj_requests if c in adj_data and adj_data[c].get("ok"))
    new_adj = len(adj_requests) - cached_adj

    # ---- 时间切分 ----
    e_dates = sorted(set(r["E"] for r in unique))
    n_dates = len(e_dates)
    train_end_idx = int(n_dates * 0.6)
    val_end_idx = int(n_dates * 0.8)
    train_dates = set(e_dates[:train_end_idx])
    val_dates = set(e_dates[train_end_idx:val_end_idx])
    test_dates = set(e_dates[val_end_idx:])

    # ---- 输出 ----
    print("=" * 60)
    print("CAR v0.2 扩样冻结清单生成")
    print("=" * 60)
    print(f"源池: A={n_a_raw}, B={n_b_raw}, 合并去重={n_merged}")
    print(f"E映射失败(无下一交易日): {n_e_map_dropped} 条")
    print(f"排除已有33候选: {n_excluded_33} 条 ({n_before_exclude} → {n_after_exclude})")
    print(f"Episode过滤(间隔<{EPISODE_MIN_GAP}日): {n_after_exclude} → {n_episode_filtered} (过滤{n_after_exclude - n_episode_filtered}条)")
    print(f"抽样: A={len(selected['A_SDP'])}, B={len(selected['B_LVB'])}, 去重后={len(unique)}")
    print(f"唯一股票: {len(adj_requests)} 股")
    print(f"日期覆盖: {e_dates[0]} ~ {e_dates[-1]} ({n_dates} 个交易日)")
    print()

    print("[时间切分]")
    print(f"  训练: {len(train_dates)} 日 ({min(train_dates)} ~ {max(train_dates)})")
    print(f"  验证: {len(val_dates)} 日 ({min(val_dates)} ~ {max(val_dates)})")
    print(f"  测试: {len(test_dates)} 日 ({min(test_dates)} ~ {max(test_dates)})")
    print()

    print("[请求预算]")
    print(f"  分钟请求 (去重 code,date): {len(minute_requests)}")
    print(f"    缓存命中: {cached_minute}, 新增: {new_minute}")
    print(f"  复权请求 (独立 code): {len(adj_requests)}")
    print(f"    缓存命中: {cached_adj}, 新增: {new_adj}")
    print(f"  磁盘估算: {new_minute * 15 / 1024:.1f} MB (分钟) + 报告 < 50MB")
    print()

    # 写清单 (内容哈希 = 规范化文本, 不含末尾换行)
    list_text = "\n".join(
        f"{r['code']},{r['setup']},{r['P']},{r['E']}" for r in unique
    )
    list_content_hash = sha256_text(list_text)  # 内容哈希: 无末尾换行
    list_file = CONTRACTS / "car_v0_2_expansion_candidate_list.txt"
    list_file.write_text(list_text + "\n", encoding="utf-8")
    list_file_hash = sha256_file(list_file)  # 文件字节哈希: 含实际写入字节

    # 分钟请求清单
    minute_list = sorted(minute_requests)
    minute_text = "\n".join(f"{c},{d}" for c, d in minute_list)
    minute_content_hash = sha256_text(minute_text)
    minute_file = CONTRACTS / "car_v0_2_expansion_minute_requests.txt"
    minute_file.write_text(minute_text + "\n", encoding="utf-8")
    minute_file_hash = sha256_file(minute_file)

    # 复权请求清单
    adj_list = sorted(adj_requests)
    adj_text = "\n".join(adj_list)
    adj_content_hash = sha256_text(adj_text)
    adj_file = CONTRACTS / "car_v0_2_expansion_adj_requests.txt"
    adj_file.write_text(adj_text + "\n", encoding="utf-8")
    adj_file_hash = sha256_file(adj_file)

    # 登记
    registry = {
        "seed": SEED,
        "target_per_pool": TARGET_PER_POOL,
        "episode_min_gap_days": EPISODE_MIN_GAP,
        "pool_size": {
            "A_SDP": n_a_raw, "B_LVB": n_b_raw, "merged": n_merged,
            "e_map_dropped": n_e_map_dropped,
            "excluded_33": n_excluded_33,
            "after_exclude_33": n_after_exclude,
        },
        "after_episode_filter": n_episode_filtered,
        "sampled": {"A_SDP": len(selected["A_SDP"]), "B_LVB": len(selected["B_LVB"])},
        "unique_candidates": len(unique),
        "unique_codes": len(adj_requests),
        "date_range": [e_dates[0], e_dates[-1]],
        "n_dates": n_dates,
        "time_split": {
            "train": [min(train_dates), max(train_dates), len(train_dates)],
            "val": [min(val_dates), max(val_dates), len(val_dates)],
            "test": [min(test_dates), max(test_dates), len(test_dates)],
        },
        "requests": {
            "minute_total": len(minute_requests),
            "minute_cached": cached_minute,
            "minute_new": new_minute,
            "adj_total": len(adj_requests),
            "adj_cached": cached_adj,
            "adj_new": new_adj,
        },
        "sha256": {
            "candidate_list_content": list_content_hash,
            "candidate_list_file": list_file_hash,
            "minute_requests_content": minute_content_hash,
            "minute_requests_file": minute_file_hash,
            "adj_requests_content": adj_content_hash,
            "adj_requests_file": adj_file_hash,
            "note": "content=规范化文本(UTF-8/LF/无末尾换行); file=实际文件字节(含CRLF转换)",
        },
        "files": {
            "candidate_list": list_file.name,
            "minute_requests": minute_file.name,
            "adj_requests": adj_file.name,
        },
        "note": "重试计入预算; 测速20条来自本清单前20条; 不根据标签结果修改抽样",
    }
    registry_file = CONTRACTS / "car_v0_2_expansion_registry.json"
    registry_file.write_text(json.dumps(registry, ensure_ascii=False, indent=2), encoding="utf-8")

    print("[SHA256]")
    print(f"  候选清单内容: {list_content_hash[:16]}...")
    print(f"  候选清单文件: {list_file_hash[:16]}...")
    print(f"  分钟请求内容: {minute_content_hash[:16]}...")
    print(f"  分钟请求文件: {minute_file_hash[:16]}...")
    print(f"  复权请求内容: {adj_content_hash[:16]}...")
    print(f"  复权请求文件: {adj_file_hash[:16]}...")
    print()
    print(f"清单文件: {list_file}")
    print(f"登记文件: {registry_file}")


if __name__ == "__main__":
    main()
