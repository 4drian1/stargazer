"""CAR v0.2 覆盖盘点 (2026-09-29).

读取 v0.2 提取器报告, 统计 A/B 池事件数、完整标签数、正负样本、日期覆盖.
与 v0.1 对账差异已在上一步输出.
"""
import json
from pathlib import Path

HERE = Path(__file__).parent
report = json.loads((HERE / "gcp_round_b_small_sample_v1/car_v0_2_replay_report.json").read_text())

results = report["per_candidate"]
stats = {
    "A_SDP": {"total": 0, "signal": 0, "no_signal": 0,
              "RECLAIMED_AT_S": 0, "ALREADY_BELOW": 0, "BREAK": 0,
              "eligible": 0, "reclaimed": 0, "not_reclaimed": 0,
              "incomplete": 0, "no_window": 0},
    "B_LVB": {"total": 0, "signal": 0, "no_signal": 0,
              "RECLAIMED_AT_S": 0, "ALREADY_BELOW": 0, "BREAK": 0,
              "eligible": 0, "reclaimed": 0, "not_reclaimed": 0,
              "incomplete": 0, "no_window": 0},
}
date_ranges = {"A_SDP": [], "B_LVB": []}

for r in results:
    setup = r["setup"]
    s = stats[setup]
    s["total"] += 1
    date_ranges[setup].append(int(r["E"]))

    if not r.get("signal_found"):
        s["no_signal"] += 1
        continue
    s["signal"] += 1

    sig_t = r.get("sig_type", "")
    if sig_t == "RECLAIMED_AT_S":
        s["RECLAIMED_AT_S"] += 1
    elif sig_t in ("ALREADY_BELOW", "BREAK"):
        s[sig_t] += 1
        # 主标签判定
        status = r.get("reclaim_status", "")
        if status == "RECLAIMED":
            s["eligible"] += 1
            s["reclaimed"] += 1
        elif status == "NOT_RECLAIMED":
            s["eligible"] += 1
            s["not_reclaimed"] += 1
        elif status == "INCOMPLETE_WINDOW":
            s["incomplete"] += 1
        elif status == "NO_WINDOW_BARS":
            s["no_window"] += 1

# 输出
print("=" * 60)
print("CAR v0.2 覆盖盘点 (33 候选, 新状态机)")
print("=" * 60)
for setup in ["A_SDP", "B_LVB"]:
    s = stats[setup]
    dates = sorted(set(date_ranges[setup]))
    print(f"\n[{setup}]")
    print(f"  候选总数: {s['total']}")
    print(f"  有信号: {s['signal']}, 无信号: {s['no_signal']}")
    print(f"  分流: RECLAIMED_AT_S={s['RECLAIMED_AT_S']}, ALREADY_BELOW={s['ALREADY_BELOW']}, BREAK={s['BREAK']}")
    print(f"  主标签: ELIGIBLE={s['eligible']} (收回={s['reclaimed']}, 未收回={s['not_reclaimed']})")
    print(f"  窗口状态: INCOMPLETE={s['incomplete']}, NO_WINDOW={s['no_window']}")
    print(f"  日期覆盖: {dates[0]} ~ {dates[-1]} ({len(dates)} 个交易日)")

print(f"\n[合计]")
total = sum(s["total"] for s in stats.values())
sig = sum(s["signal"] for s in stats.values())
ras = sum(s["RECLAIMED_AT_S"] for s in stats.values())
ab = sum(s["ALREADY_BELOW"] for s in stats.values())
br = sum(s["BREAK"] for s in stats.values())
elig = sum(s["eligible"] for s in stats.values())
rec = sum(s["reclaimed"] for s in stats.values())
nrec = sum(s["not_reclaimed"] for s in stats.values())
nosig = sum(s["no_signal"] for s in stats.values())
incomp = sum(s["incomplete"] for s in stats.values())
nw = sum(s["no_window"] for s in stats.values())
print(f"  候选: {total} (A={stats['A_SDP']['total']}, B={stats['B_LVB']['total']})")
print(f"  有信号: {sig}, 无信号: {nosig}")
print(f"  分流: RECLAIMED_AT_S={ras}, ALREADY_BELOW={ab}, BREAK={br}")
print(f"  主标签: ELIGIBLE={elig} (收回={rec}, 未收回={nrec})")
print(f"  窗口状态: INCOMPLETE={incomp}, NO_WINDOW={nw}")
print(f"  互斥合计: {ras + ab + br + nosig} (应=33)")

# 与 v0.1 对比
print(f"\n[与 v0.1 对比]")
v01 = {"RECLAIMED_AT_S": 18, "ALREADY_ABOVE": 1, "ELIGIBLE": 8, "NO_SIGNAL": 6}
print(f"  v0.1: RECLAIMED_AT_S={v01['RECLAIMED_AT_S']}, ALREADY_ABOVE={v01['ALREADY_ABOVE']}, ELIGIBLE={v01['ELIGIBLE']}, NO_SIGNAL={v01['NO_SIGNAL']}")
print(f"  v0.2: RECLAIMED_AT_S={ras}, ALREADY_BELOW={ab}, BREAK={br}, NO_SIGNAL={nosig}")
print(f"  变化: RECLAIMED_AT_S {v01['RECLAIMED_AT_S']}→{ras}, ELIGIBLE {v01['ELIGIBLE']}→{elig}, NO_SIGNAL {v01['NO_SIGNAL']}→{nosig}")
print(f"  正样本: v0.1 收回2 → v0.2 收回{rec}")
print(f"  负样本: v0.1 未收回6 → v0.2 未收回{nrec}")

# 样本充足性评估
print(f"\n[样本充足性评估]")
min_pos = 5
min_neg = 5
print(f"  最低要求: 正样本≥{min_pos}, 负样本≥{min_neg}")
print(f"  当前: 正样本={rec} ({'满足' if rec >= min_pos else '不足'}), 负样本={nrec} ({'满足' if nrec >= min_neg else '不足'})")
if rec < min_pos or nrec < min_neg:
    print(f"  结论: 样本不足, 不启动训练; 如实归档, 扩大样本须另签方案")
else:
    print(f"  结论: 样本满足最低要求, 可进入统计合同签署")
