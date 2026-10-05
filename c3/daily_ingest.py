"""daily_ingest.py — Corvus C4 路线 A：盘后公告事实增量摄取（出证据，不出概率）。

定位（开发路线 #12.2/#12.6）：盘后选股总结的信息交叉验证段。
只产出**当日可见的事实证据**（公告/类目/tier/关键词/链接/重叠），
**不打分**（p_materiality 只在 T+1 收盘后的正式链产出，见 drift_audit 裁决）。

流程：
  fetch_window(每票, [date-W, date])            ← 复用 backfill_pipeline（akshare 巨潮）
  → 与历史 event_document_link.parquet 按 dedup_key 去重（幂等）
  → 新公告聚类 event_cluster_id=sha(code|date)（与 canonical 同口径）
  → taxonomy 路由（_taxonomy 冻结词表）→ label_start（calendar.json 保守盘后语义）
  → 与历史 event_cluster.parquet 算 overlap_count（同码 ±5 交易日）
  → 写 out/daily_events/events_{date}.jsonl（确定性重写，幂等）
  → 为请求票写 evidence_{date}.json（按 code 分组，draft_gen 消费）

模式：
  --pool   读 green_watch pool_latest.json 的候选票（默认，秒级~分钟级）
  --codes  逗号分隔指定票
  --all    全市场 universe（~7500 票 × 限速 ≈ 1.5h，后台慢速用）

红线：本脚本不加载模型、不写 prediction/realization 表、不改任何冻结参数。
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from backfill_pipeline import fetch_window, resolve_label_start, sha, ymd  # noqa: E402
from _taxonomy import KEYWORD_DIRECTION, map_canonical, tier_of  # noqa: E402

OUT = ROOT / "out"
EVENTS_DIR = OUT / "daily_events"
POOL_PATH = Path(r"D:/stargazer/stargazer_ml/v5_hybrid/green_watch_v0/pool_latest.json")
EVIDENCE_VERSION = "c4_route_a_facts_v1"
HISTORY_LINK = OUT / "event_document_link.parquet"
HISTORY_CLUSTER = OUT / "event_cluster.parquet"
CALENDAR = OUT / "calendar.json"


def log(m: str):
    print(f"[daily_ingest {dt.datetime.now():%H:%M:%S}] {m}", flush=True)


def load_history_keys() -> set:
    if not HISTORY_LINK.exists():
        log(f"WARN: {HISTORY_LINK.name} 不存在，历史去重退化为仅当日（首次运行）")
        return set()
    df = pd.read_parquet(HISTORY_LINK, columns=["dedup_key"])
    return set(df["dedup_key"].tolist())


def load_history_clusters() -> pd.DataFrame | None:
    if not HISTORY_CLUSTER.exists():
        return None
    return pd.read_parquet(HISTORY_CLUSTER, columns=["code", "announce_date"])


def load_calendar() -> list[int]:
    if not CALENDAR.exists():
        raise SystemExit(f"无 {CALENDAR}（先跑过回填或放入日历）")
    return json.loads(CALENDAR.read_text(encoding="utf-8"))


def pick_codes(args) -> list[str]:
    if args.codes:
        return [c.strip().zfill(6) for c in args.codes.split(",") if c.strip()]
    if args.all:
        uni = OUT / "universe.txt"
        if not uni.exists():
            raise SystemExit(f"无 {uni}（全市场模式需 universe 缓存）")
        return [l.strip() for l in uni.read_text(encoding="utf-8").splitlines() if l.strip()]
    if not POOL_PATH.exists():
        raise SystemExit(f"池文件不存在: {POOL_PATH}（--pool 模式）")
    pool = json.loads(POOL_PATH.read_text(encoding="utf-8"))
    codes = sorted({str(c["code"]).zfill(6) for c in pool.get("candidates") or []})
    log(f"--pool: pool_date={pool.get('pool_date')} next={pool.get('next_trading_date')} "
        f"候选 {len(codes)} 票")
    return codes


def overlap_count(code: str, announce_date: int, cal: list[int],
                  hist: pd.DataFrame | None) -> int:
    """同码 ±5 交易日历史簇数（HEAVY 判定的信息性近似，facts 展示用）。"""
    if hist is None or hist.empty:
        return 0
    pos = {d: i for i, d in enumerate(cal)}
    if announce_date not in pos:
        return 0
    h = hist
    if not pd.api.types.is_integer_dtype(h["announce_date"]):
        h = h.assign(announce_date=pd.to_numeric(h["announce_date"], errors="coerce"))
    i = pos[announce_date]
    lo, hi = cal[max(0, i - 5)], cal[min(len(cal) - 1, i + 5)]
    g = h[(h["code"] == code) & (h["announce_date"] >= lo) & (h["announce_date"] <= hi)]
    return int(len(g))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", required=True, help="公告日 YYYYMMDD（盘后跑=当日）")
    ap.add_argument("--window-days", type=int, default=7, help="回查窗口（防漏拉）")
    ap.add_argument("--codes", default=None)
    ap.add_argument("--pool", action="store_true")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--request-interval", type=float, default=0.7)
    ap.add_argument("--out", default=str(EVENTS_DIR))
    args = ap.parse_args()

    date = int(ymd(args.date))
    codes = pick_codes(args)
    cal = load_calendar()
    hist_keys = load_history_keys()
    hist_cl = load_history_clusters()
    log(f"date={date} 票={len(codes)} 历史键={len(hist_keys)} 日历至 {cal[-1]}")

    w0 = (dt.date(date // 10000, date // 100 % 100, date % 100)
          - dt.timedelta(days=args.window_days)).strftime("%Y%m%d")
    w1 = str(date)

    EVENTS_DIR.mkdir(parents=True, exist_ok=True)
    raw_new: list[dict] = []
    errors: dict[str, str] = {}
    t0 = time.time()
    for i, code in enumerate(codes):
        recs, err = fetch_window(code, w0, w1)
        if err:
            errors[code] = err
        for r in recs:
            if r["dedup_key"] not in hist_keys:
                raw_new.append(r)
        if (i + 1) % 20 == 0:
            log(f"fetch {i+1}/{len(codes)} 新公告 {len(raw_new)} ({time.time()-t0:.0f}s)")
        time.sleep(args.request_interval)

    df = pd.DataFrame(raw_new)
    if df.empty:
        log("无新公告（全部命中历史或当日无公告）")
        df = pd.DataFrame(columns=["code", "name", "title", "announce_date", "raw_url",
                                   "title_hash", "dedup_key"])
    else:
        df = df.drop_duplicates("dedup_key")
        df["announce_date"] = df["announce_date"].astype(int)  # fetch 为 str，统一 int
        cats = df["title"].map(map_canonical)
        df["category"] = [c for c, _ in cats]
        df["tier"] = df["category"].map(tier_of)
        df.loc[df["category"] == "UNMAPPED", "tier"] = "P2"
        df["matched_keyword"] = [k for _, k in cats]
        df["event_cluster_id"] = [sha(f"{c}|{d}")[:12]
                                  for c, d in zip(df["code"], df["announce_date"])]
        df = df[df["announce_date"] == date]  # 窗口回查只留当日（历史日已被去重，双保险）
    log(f"当日新公告 {len(df)} 条 / 簇 {df['event_cluster_id'].nunique() if len(df) else 0} 个"
        f"｜错误 {len(errors)} 票")

    # 聚类级行（同码同日多公告合并，与 canonical 同口径）
    rows = []
    for (code, adate), g in df.groupby(["code", "announce_date"], sort=True):
        cats = g["category"].tolist()
        tier = "P0" if "P0" in [tier_of(c) for c in cats] else \
               ("P1" if "P1" in [tier_of(c) for c in cats] else "P2")
        kw = next((k for k in g["matched_keyword"] if isinstance(k, str)), None)
        start, basis = resolve_label_start(int(adate), cal, assume_premarket=False)
        ov = overlap_count(code, int(adate), cal, hist_cl)
        # 方向提示：规则粗表（KEYWORD_DIRECTION），仅人工参考，不作任何决策输入
        direction_hint = KEYWORD_DIRECTION.get(kw) if kw else None
        rows.append({
            "evidence_version": EVIDENCE_VERSION,
            "event_cluster_id": g["event_cluster_id"].iloc[0],
            "code": code, "name": g["name"].iloc[0],
            "announce_date": int(adate),
            "titles": g["title"].tolist(),
            "n_announcements_in_cluster": int(len(g)),
            "category": cats[0] if len(set(cats)) == 1 else "MULTI",
            "categories_all": sorted(set(cats)),
            "tier": tier,
            "matched_keyword": kw,
            "direction_hint": direction_hint,          # +1/-1/None，规则粗表
            "direction_hint_basis": "KEYWORD_DIRECTION_RULE_BASELINE",
            "raw_urls": g["raw_url"].tolist(),
            "label_start": start, "label_start_basis": basis,
            "overlap_count": ov, "heavy_flag": ov >= 3,
            "eligible_fact": bool(tier == "P0" and "DISTRIBUTION" not in cats),
            "eligible_fact_basis": "P0 & non-DISTRIBUTION（事实层；HEAVY/特征完整性属正式评分链）",
            "p_materiality": None,                      # 路线 A：盘后永不出概率
            "ingest_ts": dt.datetime.now().isoformat(timespec="seconds"),
        })

    # 确定性重写（幂等：同 date 重跑结果一致，仅 ingest_ts 变）
    out_path = EVENTS_DIR / f"events_{date}.jsonl"
    with out_path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # 按请求票分组的证据文件（draft_gen 消费）
    req = set(codes)
    evidence = {}
    for r in rows:
        if r["code"] in req:
            evidence.setdefault(r["code"], []).append(r)
    ev_path = EVENTS_DIR / f"evidence_{date}.json"
    ev_path.write_text(json.dumps({"date": date, "version": EVIDENCE_VERSION,
                                   "by_code": evidence}, ensure_ascii=False, indent=1),
                       encoding="utf-8")
    n_ev = sum(len(v) for v in evidence.values())
    log(f"完成: {out_path.name}（{len(rows)} 簇）, {ev_path.name}（{len(evidence)} 票 {n_ev} 簇）"
        f" ({time.time()-t0:.0f}s)")
    if errors:
        err_path = EVENTS_DIR / f"errors_{date}.json"
        err_path.write_text(json.dumps(errors, ensure_ascii=False, indent=1), encoding="utf-8")
        log(f"错误票 {len(errors)} → {err_path.name}")


if __name__ == "__main__":
    main()
