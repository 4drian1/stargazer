"""tarot_labels.py：冻结 Tarot 日级风险标签（外围模型训练闸门）。

规格（用户冻结）：
- 三套池口径分开，绝不混成一个标签：
    tarot.hermit.a2@v1    <- pullback_train/events_a1_cooled.parquet（主研究池）
    tarot.hermit.legacy@v1 <- events_legacy_cooled.parquet（旧模型审计对照）
    tarot.b0@v1           <- events_b_b0_cooled.parquet（小容量辅助，只作描述）
- 一日一行：当天池子的聚合前瞻结果（不把几千只候选当几千个独立市场样本）。
- 横盘埋伏/盘中 Wheel 不混入（历史长度不足，后续单独建标签头）。

冻结口径 cost_version = costs_v2_frozen（照抄 stargazer_ml/labels_v2.py 常量，不 import 防漂移）：
    commission=0.0003（双边，按名义） stamp=0.0005（卖出） slip_entry=slip_exit=0.001
入场：t+1 开盘价（不使用 t+1 之后信息决定入场）；entry_eff = open*(1+slip_entry+commission)
止损触碰：min(low, t+1..t+3) <= entry_eff*(1-0.03)（简化为触碰口径，不模拟封板顺延，
与 labels_v3 的成交口径不同，本标签服务"池子集体风险"相对比较，口径已冻结）
净收益 h 日：close_{t+h}*(1-slip_exit-commission-stamp) / entry_eff - 1
尾部：池内 net_ret_5d 的 5% 分位。
末尾不足前瞻窗口的交易日标 CENSORED_TAIL（embargo 在划分层处理，不在此删行）。

输出：SQLite corvus.tarot_day_labels + parquet 导出。
用法：python -m corvus.tarot_labels [--strategy tarot.hermit.a2@v1] [--start 20190101] [--end 20260901]
"""
import argparse
import hashlib
import json
import sqlite3
from datetime import datetime
from pathlib import Path

import pandas as pd

from .db import get_conn, init_db
from .timeutil import now_iso

ML = Path(r"D:\stargazer\stargazer_ml\pullback_train")
PARTS = sorted(Path(r"D:\a股数据\data_daily_parts").glob("part_*.parquet"))

POOLS = {
    "tarot.hermit.a2@v1": ("events_a1_cooled.parquet", "events_a1_cooled"),
    "tarot.hermit.legacy@v1": ("events_legacy_cooled.parquet", "events_legacy_cooled"),
    "tarot.b0@v1": ("events_b_b0_cooled.parquet", "events_b_b0_cooled"),
}

COMMISSION = 0.0003
STAMP = 0.0005
SLIP_ENTRY = 0.001
SLIP_EXIT = 0.001
STOP = 0.03
HORIZONS = (3, 5)
COST_VERSION = "costs_v2_frozen"
LABEL_VERSION = "tarot_day_labels@v1"

COLS = ["date", "code", "open", "low", "close"]


def load_daily(start, end):
    """日线 parts -> 只含前瞻计算所需列，按 code/date 排序。"""
    dfs = []
    for p in PARTS:
        df = pd.read_parquet(p, columns=COLS)
        dfs.append(df)
    daily = pd.concat(dfs, ignore_index=True)
    daily = daily[(daily["date"] >= start) & (daily["date"] <= end)]
    daily = daily.sort_values(["code", "date"]).reset_index(drop=True)
    return daily


def add_forward(daily):
    """每个 (code,date) 加前瞻字段：open_1 / minlow_3 / close_3 / minlow_5 / close_5。"""
    g = daily.groupby("code", sort=False)
    daily["open_1"] = g["open"].shift(-1)
    daily["close_3"] = g["close"].shift(-3)
    daily["close_5"] = g["close"].shift(-5)

    def _fwd_min(s, h):
        return s.iloc[::-1].rolling(h, min_periods=h).min().iloc[::-1]

    daily["minlow_3"] = g["low"].transform(_fwd_min, 3)
    daily["minlow_5"] = g["low"].transform(_fwd_min, 5)
    return daily


def build_strategy_labels(strategy_id, daily_fwd, start, end):
    fname, gen_name = POOLS[strategy_id]
    events = pd.read_parquet(ML / fname, columns=["code", "date"])
    events = events[(events["date"] >= start) & (events["date"] <= end)].copy()
    events["code"] = events["code"].astype(str).str.zfill(6)

    gen_version = (f"{gen_name}@rows{len(events)}@{events['date'].min()}-{events['date'].max()}")

    fwd = daily_fwd[["code", "date", "open_1", "close_3", "close_5", "minlow_3", "minlow_5"]]
    m = events.merge(fwd, on=["code", "date"], how="left")

    entry_ok = m["open_1"].notna() & (m["open_1"] > 0)
    entry_eff = m["open_1"] * (1 + SLIP_ENTRY + COMMISSION)
    for h in HORIZONS:
        close_h = m[f"close_{h}"]
        m[f"net_ret_{h}d"] = (close_h * (1 - SLIP_EXIT - COMMISSION - STAMP)) / entry_eff - 1
        m.loc[~entry_ok | close_h.isna(), f"net_ret_{h}d"] = None
    m["stop_3d"] = entry_ok & m["minlow_3"].notna() \
        & (m["minlow_3"] <= entry_eff * (1 - STOP))

    rows = []
    for d, grp in m.groupby("date"):
        n = len(grp)
        membership = "\n".join(sorted(f"{c}:{d}" for c in grp["code"]))
        uh = hashlib.sha256(membership.encode("utf-8")).hexdigest()[:16]
        v3 = grp["net_ret_3d"].dropna()
        v5 = grp["net_ret_5d"].dropna()
        censored = len(v5) < max(1, int(0.8 * n))  # 尾部交易日前瞻窗口不足
        rows.append({
            "trade_date": str(d),
            "snapshot_type": "PREMARKET",
            "strategy_id": strategy_id,
            "candidate_generator_version": gen_version,
            "candidate_universe_hash": uh,
            "candidate_count": n,
            "entry_valid_rate": round(float(entry_ok.loc[grp.index].mean()), 4),
            "stop_rate_3d": round(float(grp["stop_3d"].mean()), 4) if len(grp) else None,
            "loss_rate_3d": round(float((v3 < 0).mean()), 4) if len(v3) else None,
            "loss_rate_5d": round(float((v5 < 0).mean()), 4) if len(v5) else None,
            "mean_net_return_3d": round(float(v3.mean()), 6) if len(v3) else None,
            "mean_net_return_5d": round(float(v5.mean()), 6) if len(v5) else None,
            "left_tail_return_5d": round(float(v5.quantile(0.05)), 6) if len(v5) else None,
            "n_valid_3d": int(len(v3)),
            "n_valid_5d": int(len(v5)),
            "cost_version": COST_VERSION,
            "label_version": LABEL_VERSION,
            "label_status": "CENSORED_TAIL" if censored else "OK",
            "label_frozen_at": now_iso(),
        })
    return pd.DataFrame(rows)


def save(df, conn):
    cols = ("trade_date,snapshot_type,strategy_id,candidate_generator_version,"
            "candidate_universe_hash,candidate_count,entry_valid_rate,stop_rate_3d,"
            "loss_rate_3d,loss_rate_5d,mean_net_return_3d,mean_net_return_5d,"
            "left_tail_return_5d,n_valid_3d,n_valid_5d,cost_version,label_version,"
            "label_status,label_frozen_at")
    placeholders = ",".join("?" * 19)
    conn.executemany(
        f"INSERT OR REPLACE INTO tarot_day_labels ({cols}) VALUES ({placeholders})",
        df.itertuples(index=False, name=None))
    conn.commit()


def main(argv=None):
    ap = argparse.ArgumentParser(description="冻结 Tarot 日级风险标签")
    ap.add_argument("--strategy", default="all")
    ap.add_argument("--start", default="20190101")
    ap.add_argument("--end", default="20260901")
    ap.add_argument("--db", default=None)
    args = ap.parse_args(argv)

    init_db(args.db)
    conn = get_conn(args.db)
    try:
        print(f"[1/3] 加载日线 parts（{len(PARTS)} 个文件）...", flush=True)
        daily = load_daily(int(args.start) - 10000, int(args.end) + 10)
        print(f"      {len(daily)} 行", flush=True)
        print("[2/3] 计算前瞻窗口...", flush=True)
        daily_fwd = add_forward(daily)
        del daily

        strategies = list(POOLS) if args.strategy == "all" else [args.strategy]
        out_frames = []
        for sid in strategies:
            print(f"[3/3] 冻结 {sid}...", flush=True)
            df = build_strategy_labels(sid, daily_fwd, int(args.start), int(args.end))
            save(df, conn)
            out_frames.append(df)
            ok = df[df["label_status"] == "OK"]
            print(f"      {len(df)} 个交易日 | 平均池大小 {df['candidate_count'].mean():.0f} | "
                  f"5日亏损率均值 {ok['loss_rate_5d'].mean():.3f} | "
                  f"5日均净收益 {ok['mean_net_return_5d'].mean():.4f}", flush=True)
        out = pd.concat(out_frames, ignore_index=True)
        path = Path(__file__).resolve().parent.parent / "data" / "tarot_day_labels.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        out.to_parquet(path, index=False)
        print(f"导出 {path}（{len(out)} 行）", flush=True)
    finally:
        conn.close()


if __name__ == "__main__":
    main()
