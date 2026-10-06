# -*- coding: utf-8 -*-
"""T5 冻结版盘后批量打分（t5_batch_score.py）。

批量包装 intraday_t5_features.py 的冻结打分逻辑（加载/特征/四头推理全部
import 复用，不改其内部）：

  .venv\\python -X utf8 t5_batch_score.py --date YYYYMMDD \
      --codes-file codes.json [--out scores.jsonl] [--dp 1500] [--workers 6]

- 输入 codes.json = ["600207","300155",...]（6 位代码列表）。
- 取数优先 stockdb 分钟源（STOCKDB_ROOT；本机默认发现路径不存在时自动
  回退到 D:/a股数据/stockdb 等候选）；stockdb 无数据时回退
  minute_cache_clean / data_daily_parts parquet。
- 决策点统一 dp=1500（尾盘决策点 = 盘后批口径）。dp>=1200 走 PM 原生头
  （PM_R2/PM_O6）；同一特征再喂 AM 头（AM_R1/AM_O4）出第二行——
  AM/PM 头各出一份，均为 "out-of-native-regime approximation"
  （AM 头原生 dp∈{1000,1030}，盘后 1500 特征打 AM 头是近似口径）。
- 输出 jsonl 每行（冻结 schema，两个代理共用）：
  {code,date,dp,head,p_risk,p_opp,quadrant,ood,model_version,model_hash}
- 模型文件缺失 / torch 不可用 / 冻结 manifest 哈希不一致 → exit 2。

== 冻结纪律 ==
- model_version 永久钉死 2026-09-09 四头打分器；本脚本只读冻结 pkl，
  不训练、不调参、不反哺。
- green_v2/t5_freeze_manifest.json 记录冻结 sha256；每日运行校验，
  不一致即拒绝输出。
"""
import argparse
import hashlib
import json
import os
import sys
import time
import traceback
from multiprocessing import Pool
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

PKL = _HERE / "shadow_models" / "shadow_t5_frozen.pkl"
MANIFEST = _HERE.parent / "v5_hybrid" / "green_v2" / "t5_freeze_manifest.json"

# stockdb 服务目录发现：intraday_t5_features 默认找
# D:/a股数据/free-stockdb-win/stockdb（本机不存在），这里在 import 前补候选。
def _fix_stockdb_root():
    if os.environ.get("STOCKDB_ROOT"):
        return
    for cand in (r"D:/a股数据/stockdb",
                 r"D:/a股数据/free-stockdb-main",
                 r"D:/a股数据/free-stockdb--0.3.1"):
        if (Path(cand) / "pybao" / "stock_sdk.py").is_file():
            os.environ["STOCKDB_ROOT"] = cand
            return

_fix_stockdb_root()


def model_hash(pkl=PKL):
    h = hashlib.sha256()
    with open(pkl, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:12]


def check_or_create_manifest(pkl=PKL, manifest=MANIFEST):
    """manifest 存在 → 哈希不一致 raise；不存在 → 以当前冻结文件创建。"""
    mh = model_hash(pkl)
    if manifest.is_file():
        doc = json.loads(manifest.read_text(encoding="utf-8"))
        want = doc.get("sha256_12")
        if want != mh:
            raise SystemExit(
                f"FROZEN_MODEL_HASH_MISMATCH: 权重文件 sha256 前12位={mh} "
                f"与冻结清单 {manifest} 记录 {want} 不一致；按纪律拒绝输出。"
                f"（若确为合法换模，须人工更新 manifest 并登记 deviation）")
        return mh, False
    manifest.parent.mkdir(parents=True, exist_ok=True)
    doc = {
        "model_file": str(pkl),
        "model_version": "2026-09-09",
        "frozen_list": "AM-R1 / PM-R2 / AM-O4 / PM-O6 (T7=OOD only)",
        "sha256_12": mh,
        "frozen_at": "2026-09-09",
        "note": "T5 冻结版钉死；每日运行校验，不一致即拒绝输出。",
    }
    manifest.write_text(json.dumps(doc, ensure_ascii=False, indent=2),
                        encoding="utf-8")
    return mh, True


# ---------------------------------------------------------------- worker
_W = {"t5": None, "bundle": None, "mhash": None, "dp": 1500}


def _worker_init(dp, date):
    import intraday_t5_features as t5
    if not PKL.is_file():
        raise SystemExit(f"FROZEN_MODEL_MISSING: {PKL}")
    _W["t5"] = t5
    _W["bundle"] = t5.load_bundle()
    _W["mhash"] = model_hash()
    _W["dp"] = int(dp)
    _W["date"] = str(date)


def _minutes_for(t5, code, trade_date):
    """stockdb 分钟（≥20 前交易日历史 + 当日未截断）；空则回退分钟缓存。"""
    import numpy as np
    import pandas as pd
    if t5.service_available():
        start = (pd.Timestamp(str(trade_date)) - pd.Timedelta(days=75)) \
            .strftime("%Y%m%d")
        df = t5.fetch_minute(code, start=start, end=str(trade_date), fq=None)
        if len(df):
            return df, "stockdb"
    cm, day_i = t5.replay_cm(code, trade_date, 1500)
    if cm is None:
        return None, "none"
    dates = cm["t"] + (cm["days"][cm["day_idx"]] * 1000000).astype(np.int64)
    df = pd.DataFrame({"date": dates, "open": cm["o"], "high": cm["h"],
                       "low": cm["l"], "close": cm["c"],
                       "volume": cm["v"], "amount": cm["a"]})
    return df, "minute_cache"


def _daily_for(t5, code, trade_date):
    """stockdb 日线（400 自然日）；空则回退 data_daily_parts parquet。"""
    import pandas as pd
    if t5.service_available():
        start = (pd.Timestamp(str(trade_date)) - pd.Timedelta(days=500)) \
            .strftime("%Y%m%d")
        df = t5.fetch_daily(code, start=start, end=str(trade_date), fq=None)
        if len(df):
            df["code"] = code
            if "pre_close" not in df or df["pre_close"].isna().all():
                df["pre_close"] = df["close"].shift(1)
            return df, "stockdb"
    df = t5.load_daily_hist(code)
    return (df, "parquet") if len(df) else (None, "none")


def score_one(code):
    """单票 dp=1500 双头打分。返回 (code, rows, err)；err 非空表示失败。"""
    import numpy as np
    t5, bundle, dp, trade_date = \
        _W["t5"], _W["bundle"], _W["dp"], _W["date"]
    code = str(code).zfill(6)
    try:
        minutes, _src_m = _minutes_for(t5, code, trade_date)
        if minutes is None or not len(minutes):
            return code, None, "no_minute_data"
        daily, _src_d = _daily_for(t5, code, trade_date)
        if daily is None or not len(daily):
            return code, None, "no_daily_data"
        asof = t5.daily_asof(code, daily, trade_date)
        pre_close, atr_px = asof["pre_close"], asof["atr_ref"]
        if not (np.isfinite(pre_close) and np.isfinite(atr_px)):
            return code, None, "insufficient_daily_history"
        feats = t5.compute_features(code, trade_date, dp, minutes, pre_close,
                                    atr_px=atr_px, daily_hist=daily)
        if feats is None:
            return code, None, "no_bars_before_dp"
        ood = t5.attach_ood(asof)
        mv = str(bundle["meta"].get("_frozen_at") or "")
        rows = []
        for head_dp, head_name in ((dp, "PM"), (1000, "AM")):
            sc = t5.score_features(bundle, head_dp, feats)
            rows.append({
                "code": code, "date": int(trade_date), "dp": int(dp),
                "head": head_name,
                "p_risk": (None if not np.isfinite(sc["P_risk"])
                           else round(float(sc["P_risk"]), 6)),
                "p_opp": (None if not np.isfinite(sc["P_opp"])
                          else round(float(sc["P_opp"]), 6)),
                "quadrant": sc["quadrant"],
                "ood": bool(ood["ood"]),
                "model_version": mv,
                "model_hash": _W["mhash"],
            })
        return code, rows, None
    except Exception:
        return code, None, traceback.format_exc().strip().splitlines()[-1]


ARGS_DATE = None  # 仅主进程用；worker 通过 _worker_init 拿 date（spawn 不继承全局）


def main():
    global ARGS_DATE
    ap = argparse.ArgumentParser(description="T5 冻结版盘后批量打分")
    ap.add_argument("--date", required=True, help="交易日 YYYYMMDD")
    ap.add_argument("--codes-file", required=True,
                    help="json 列表路径（6 位代码）")
    ap.add_argument("--out", default=None,
                    help="jsonl 输出路径（默认 stdout）")
    ap.add_argument("--dp", type=int, default=1500, help="决策点（默认 1500）")
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()

    if not PKL.is_file():
        print(f"FATAL: 冻结模型文件缺失 {PKL}", file=sys.stderr)
        return 2
    try:
        import torch  # noqa: F401  契约要求 torch 可用（.venv）
    except Exception as e:
        print(f"FATAL: torch 不可用：{e}", file=sys.stderr)
        return 2
    try:
        mh, created = check_or_create_manifest()
    except SystemExit as e:
        print(f"FATAL: {e}", file=sys.stderr)
        return 2
    if created:
        print(f"[manifest] 已创建冻结清单 {MANIFEST}（sha256_12={mh}）")

    codes = json.loads(Path(a.codes_file).read_text(encoding="utf-8"))
    codes = sorted({str(c).zfill(6) for c in codes})
    if not codes:
        print("FATAL: codes-file 为空", file=sys.stderr)
        return 2
    ARGS_DATE = a.date

    t0 = time.time()
    out_rows, fails = [], {}
    with Pool(a.workers, initializer=_worker_init,
              initargs=(a.dp, a.date)) as pool:
        for code, rows, err in pool.imap_unordered(score_one, codes):
            if err:
                fails[code] = err
            else:
                out_rows.extend(rows)
    el = time.time() - t0

    payload = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in out_rows)
    if a.out:
        Path(a.out).write_text(payload, encoding="utf-8")
    else:
        sys.stdout.write(payload)
    n_codes = len({r["code"] for r in out_rows})
    print(f"[batch] date={a.date} dp={a.dp} 打分 {n_codes}/{len(codes)} 只 "
          f"（双头行 {len(out_rows)}），失败 {len(fails)}，耗时 {el:.1f}s "
          f"（{el / max(n_codes, 1):.2f}s/只）", file=sys.stderr)
    for c, e in sorted(fails.items()):
        print(f"  FAIL {c}: {e}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
