"""阶段2.1：归档旧标签(SUPERSEDED) -> 重建 -> 新旧对比报告。

旧版 = 当前磁盘上的 exit_labels_*.parquet（2.1 修复前）。
新版 = 本次 run_exit 重建结果。

对比输出：
  - 退出原因变化比例
  - 平均净收益变化
  - STOP/TRAIL/TIME_EXIT 占比（新）
  - 受同bar顺序/成交约束影响的事件数（原因或价格变化）
  - 被涨跌停阻塞的事件数

运行：python phase21_rebuild.py
"""
from __future__ import annotations
import sys, os, shutil
from datetime import datetime
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import pandas as pd

from build_exit_labels import build, ROOT

LABELS = {
    "DAILY_APPROX": ROOT / "labels" / "exit_labels_daily_approx.parquet",
    "MINUTE_EXACT": ROOT / "labels" / "exit_labels_minute_exact.parquet",
}
ARCHIVE = ROOT / "labels" / "SUPERSEDED"
STAMP = datetime.now().strftime("%Y%m%d")


def archive_old():
    ARCHIVE.mkdir(parents=True, exist_ok=True)
    archived = []
    for name, path in LABELS.items():
        if path.exists():
            dst = ARCHIVE / f"{path.stem}_SUPERSEDED_{STAMP}.parquet"
            shutil.copy(path, dst)
            archived.append(str(dst))
    return archived


def load_all():
    frames = [pd.read_parquet(p) for p in LABELS.values() if p.exists()]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def main():
    print("== 归档旧标签 ==")
    arch = archive_old()
    for a in arch:
        print("  archived:", a)
    print("\n== 重建标签（run_exit 2.1）==")
    build()
    new = load_all()
    old = pd.concat([pd.read_parquet(ARCHIVE / f) for f in os.listdir(ARCHIVE)
                     if f.endswith(".parquet") and f"SUPERSEDED_{STAMP}" in f],
                    ignore_index=True) if arch else pd.DataFrame()
    if old.empty or new.empty:
        print("旧/新标签缺失，无法对比")
        return
    key = ["code", "signal_date", "strategy_variant", "label_precision"]
    m = old.merge(new, on=key, suffixes=("_old", "_new"))
    n = len(m)
    reason_chg = (m["exit_reason_old"] != m["exit_reason_new"]).sum()
    price_chg = ((m["exit_price_old"] - m["exit_price_new"]).abs() > 1e-6).sum()
    net_old = m["net_return_old"].mean()
    net_new = m["net_return_new"].mean()
    print("\n========== 阶段2.1 新旧对比 ==========")
    print(f"匹配事件数: {n}")
    print(f"退出原因变化: {reason_chg} ({reason_chg/n*100:.2f}%)")
    print(f"退出价/净收变化(>1e-6): {price_chg} ({price_chg/n*100:.2f}%)  <- 受同bar顺序/成交约束影响估计")
    print(f"平均净收益: 旧 {net_old:+.4f} -> 新 {net_new:+.4f} (Δ {net_new-net_old:+.4f})")
    print("\n新标签退出原因占比:")
    for r in ["STOP", "TRAIL1", "TRAIL2", "REBOUND_FAILURE", "REBOUND_FAILURE_STAGED", "TIME_EXIT"]:
        c = (new["exit_reason"] == r).sum()
        print(f"  {r:22s} {c:6d} ({c/len(new)*100:.1f}%)")
    print(f"\n被涨跌停/停牌阻塞(exit_blocked=1): {int(new['exit_blocked'].sum())}")
    # 旧版阻塞应为0（修复前未实现）
    print(f"旧版阻塞事件数: {int(old['exit_blocked'].sum())}")

    rep = ROOT / "reports" / f"phase21_report_{STAMP}.txt"
    rep.parent.mkdir(parents=True, exist_ok=True)
    with open(rep, "w", encoding="utf-8") as f:
        f.write(f"matched={n}\nreason_change={reason_chg}\nprice_change={price_chg}\n"
                f"net_old={net_old}\nnet_new={net_new}\nblocked_new={int(new['exit_blocked'].sum())}\n")
    print(f"\n报告已写: {rep}")


if __name__ == "__main__":
    main()
