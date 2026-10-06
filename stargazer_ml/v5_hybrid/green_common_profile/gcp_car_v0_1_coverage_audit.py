"""CAR v0.1 覆盖盘点脚本 (2026-09-29).

用现有缓存统计 A/B 池事件数、完整标签数、正负样本、日期覆盖.
不增加请求, 不修改数据, 只读现有回放报告和候选清单.
"""
import json
from pathlib import Path

HERE = Path(__file__).parent
replay = json.loads((HERE / "gcp_round_b_small_sample_v1/car_v0_1_real_replay_report.json").read_text())
candidates = json.loads((HERE / "contracts/car_v0_1_sampling_summary.json").read_text())

# 候选到 setup 的映射
setup_map = {}
for c in candidates["selected"]:
    key = (str(c["code"]).zfill(6), int(c["E"]))
    setup_map[key] = c["setup"]

results = replay["per_candidate"]
stats = {
    "A_SDP": {"total": 0, "signal": 0, "reclaim_at_s": 0, "already_above": 0,
              "eligible": 0, "no_signal": 0, "reclaimed": 0, "not_reclaimed": 0},
    "B_LVB": {"total": 0, "signal": 0, "reclaim_at_s": 0, "already_above": 0,
              "eligible": 0, "no_signal": 0, "reclaimed": 0, "not_reclaimed": 0},
}
date_ranges = {"A_SDP": [], "B_LVB": []}

for r in results:
    key = (str(r["code"]).zfill(6), int(r["E"]))
    setup = setup_map.get(key, "UNKNOWN")
    if setup not in stats:
        continue
    s = stats[setup]
    s["total"] += 1
    date_ranges[setup].append(int(r["E"]))

    if not r.get("signal_found"):
        s["no_signal"] += 1
        continue
    s["signal"] += 1

    status = r.get("reclaim_status", "")
    if status == "RECLAIMED_AT_S":
        s["reclaim_at_s"] += 1
    elif status == "ALREADY_ABOVE":
        s["already_above"] += 1
    else:
        # ELIGIBLE 主标签 (60m)
        if r.get("window_complete_60m") and not r.get("window_truncated_60m"):
            s["eligible"] += 1
        if r.get("reclaim_60m") == 1:
            s["reclaimed"] += 1
        elif r.get("reclaim_60m") == 0:
            s["not_reclaimed"] += 1

# 输出
print("=" * 60)
print("CAR v0.1 覆盖盘点 (33 候选, 冻结缓存)")
print("=" * 60)
for setup in ["A_SDP", "B_LVB"]:
    s = stats[setup]
    dates = sorted(set(date_ranges[setup]))
    print(f"\n[{setup}]")
    print(f"  候选总数: {s['total']}")
    print(f"  有信号: {s['signal']}, 无信号: {s['no_signal']}")
    print(f"  分流: RECLAIMED_AT_S={s['reclaim_at_s']}, ALREADY_ABOVE={s['already_above']}, ELIGIBLE={s['eligible']}")
    print(f"  主标签: 收回={s['reclaimed']}, 未收回={s['not_reclaimed']}")
    print(f"  日期覆盖: {dates[0]} ~ {dates[-1]} ({len(dates)} 个交易日)")
    print(f"  日期列表: {dates}")

print(f"\n[合计]")
total = sum(s["total"] for s in stats.values())
sig = sum(s["signal"] for s in stats.values())
ras = sum(s["reclaim_at_s"] for s in stats.values())
aa = sum(s["already_above"] for s in stats.values())
elig = sum(s["eligible"] for s in stats.values())
nosig = sum(s["no_signal"] for s in stats.values())
rec = sum(s["reclaimed"] for s in stats.values())
nrec = sum(s["not_reclaimed"] for s in stats.values())
print(f"  候选: {total} (A={stats['A_SDP']['total']}, B={stats['B_LVB']['total']})")
print(f"  有信号: {sig}, 无信号: {nosig}")
print(f"  分流: RECLAIMED_AT_S={ras}, ALREADY_ABOVE={aa}, ELIGIBLE={elig}")
print(f"  主标签: 收回={rec}, 未收回={nrec}")
print(f"  互斥合计: {ras + aa + elig + nosig} (应=33)")

# v0.2 状态机下的预期变化
print(f"\n[v0.2 预期影响]")
print(f"  v0.1: APPROACH 触发 → {ras} 条 RECLAIMED_AT_S")
print(f"  v0.2: APPROACH 仅观察 → 这些候选若后续未跌破则变为 NO_SIGNAL")
print(f"  v0.2 预计 ELIGIBLE 减少: {ras} 条可能变为无事件 (需逐一检查是否后续跌破)")
