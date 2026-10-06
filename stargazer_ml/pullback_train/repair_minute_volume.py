# -*- coding: utf-8 -*-
"""分钟缓存量价单位修复（一次性数据修复）。

2026-02~09 轮下载的数据中，部分 bar 的 volume 被源端放大了 100 倍
（r = amount/(volume*close) ≈ 0.01；2025 与 202601 数据抽样验证 0% 污染，
此前所有 2025 实验不受影响）。污染的 bar 会使 bar-VWAP=amount/volume 失真
约 100 倍，腐蚀 vwap 口径入场价与一切 VWAP 类特征（2026 复核曾出现
净收益 +540% 的荒谬值即由此而来）。

修复规则：仅当 amount>0、volume>0、close>0 且 r<0.5 时，
volume := amount/close（金额与收盘价一致的口径）；r>=0.5 的 bar 不动。
按 (code,day) 统计污染率，输出 repair_volume_report.csv，供下游决定
是否整日活动剔除（当前选择：修复后保留，修复已使量价自洽）。

幂等：修复后 r 落于 [0.5,2]（≈1），重跑不会再改任何 bar。
"""
import os
import time

import numpy as np
import pandas as pd

OUT_DIR = "D:/stargazer/stargazer_ml/pullback_train"
CACHE = os.path.join(OUT_DIR, "minute_cache")
REPORT = os.path.join(OUT_DIR, "repair_volume_report.csv")


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main():
    t0 = time.time()
    files = [f for f in os.listdir(CACHE) if f.endswith(".parquet")]
    stats = []
    total_bad = total_bars = 0
    for i, f in enumerate(files):
        p = os.path.join(CACHE, f)
        try:
            df = pd.read_parquet(p)
        except Exception:  # noqa: BLE001
            continue
        if not len(df) or "volume" not in df.columns:
            continue
        v = pd.to_numeric(df["volume"], errors="coerce")
        a = pd.to_numeric(df["amount"], errors="coerce")
        c = pd.to_numeric(df["close"], errors="coerce")
        ok = (v > 0) & (a > 0) & (c > 0)
        r = pd.Series(np.nan, index=df.index)
        r[ok] = a[ok] / (v[ok] * c[ok])
        bad = ok & (r < 0.5)
        n_bad, n = int(bad.sum()), int(len(df))
        total_bad += n_bad
        total_bars += n
        if n_bad:
            df.loc[bad, "volume"] = (a[bad] / c[bad]).round().astype(np.float64)
            df.to_parquet(p, index=False)
            day = (df.loc[bad, "date"].astype(np.int64) // 1000000)
            for d, g in day.groupby(day):
                stats.append({"code": f[:-7], "day": int(d),
                              "bad_bars": int((day == d).sum())})
        if (i + 1) % 500 == 0:
            log(f"{i+1}/{len(files)} 修复bar累计={total_bad} "
                f"{time.time()-t0:.0f}s")
    rep = pd.DataFrame(stats)
    rep.to_csv(REPORT, index=False)
    log(f"done: {len(files)} files, 修复bar {total_bad}/{total_bars} "
        f"= {total_bad/max(total_bars,1):.3%}；受影响(code,day)={len(rep)}；"
        f"报告 {REPORT}；用时 {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
