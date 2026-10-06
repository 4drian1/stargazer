# -*- coding: utf-8 -*-
"""首轮 247 事件退出窗口分钟采集脚本 v2 (加固版; 默认 dry-run 不抓取).

加固点 (2026-10-04 修复 dry-run 验收失效):
  1. dry-run 与真实进度彻底分离: 真实用 progress_real.jsonl, dry-run 用 progress_dryrun.jsonl.
     旧 progress.jsonl 为遗留 dry-run 产物(DRYRUN_NO_FETCH), 本脚本不再读取, 避免 --real 误跳全部.
  2. 真实 done 必须同时有可读文件 + 匹配哈希: 续跑时校验文件存在且 sha256==记录, 否则重抓.
  3. 预算按"实际将写入字节数"守住: 真实模式用 len(parquet_bytes) 累加, 超 50MB 硬线即停(不写入超限文件);
     运行结束再核对 by_pair 目录实际占用.
  4. 连续连接错误自动停机: 连续 error 达 CONN_ERR_LIMIT 即中止(不批量记完成), 留待人工/重启;
     source_missing(源确无数据)记终态留因, 不找替补, 不计入连接错误.
两层预算: 基础层 1006 冻结; 顺延层待 §6.4 另算, 默认不含.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "step9_preflight"))

os.environ.setdefault("GATE1C_MINUTE_SOURCE", "stockdb")

VERIFY_JSON = HERE / "gcp_round_b_small_sample_v1" / "gcp_247_coverage_verify.json"
OUT2 = HERE / "gcp_round_b_small_sample_v1" / "minute_collect_v1"

MIN_DIRS = [OUT2.parent / "minute", OUT2.parent / "minute_ext", OUT2.parent / "minute_car"]
FALLBACK_DIR = ROOT / "pullback_train" / "minute_cache_clean"

HARD_STOP_BYTES_DEFAULT = 50 * 1024 * 1024
RETRY = 2
CONN_ERR_LIMIT = 3
TERMINAL = {"done", "source_missing"}   # error / budget_hold 非终态, 续跑重试
DATA_END = 20260928


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def load_minute_index():
    """用于 dry-run 字节估算: 返回 (idx_dict, total_bytes)."""
    idx = {}
    total_bytes = 0
    for d in MIN_DIRS + ([FALLBACK_DIR] if FALLBACK_DIR.exists() else []):
        if not d.exists():
            continue
        for pf in d.glob("*.parquet"):
            try:
                total_bytes += pf.stat().st_size
                import pandas as pd
                m = pd.read_parquet(pf, columns=["date"])
                if "date" in m and not m.empty:
                    for dd in pd.unique(m["date"].astype(str).str[:8].astype(int)):
                        idx.setdefault((pf.stem, int(dd)), []).append(pf)
            except Exception:
                pass
    return idx, total_bytes


def load_frozen_base():
    v = json.loads(VERIFY_JSON.read_text(encoding="utf-8"))
    return sorted(set(tuple(x) for x in v["requests"]["minute_missing_pairs"]))


def load_rollover(path):
    if not path:
        return set()
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(data, dict):
        data = data.get("rollover_pairs", data.get("minute_missing_pairs", []))
    return set(tuple(x) for x in data)


def progress_paths(out2: Path, real: bool):
    return out2 / ("progress_real.jsonl" if real else "progress_dryrun.jsonl")


def load_progress(path: Path, real: bool, out2: Path):
    done = {}
    if not path.exists():
        return done
    for line in path.read_text(encoding="utf-8").strip().split("\n"):
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line)
        if real:
            # 真实 done: 必须文件可读且哈希匹配, 否则视为未完成(重抓)
            if rec.get("status") == "done":
                fp = out2 / "by_pair" / f"{rec['code']}_{rec['date']}.parquet"
                if fp.exists():
                    try:
                        if sha256_bytes(fp.read_bytes()) == rec.get("sha256"):
                            done[(rec["code"], rec["date"])] = rec
                    except Exception:
                        pass
        else:
            if rec.get("status") == "sim_done":
                done[(rec["code"], rec["date"])] = rec
    return done


def precheck(dry_run, hard_stop_bytes, out2):
    notes = []
    if dry_run:
        notes.append("DRY-RUN: 小批量测速/磁盘实测/stockdb 健康 均跳过 (不向源发请求)")
        return notes
    import shutil
    from data_source_provenance import get_minute_http
    t0 = time.time()
    try:
        sample = get_minute_http("600519", str(DATA_END))
        notes.append(f"stockdb 健康: 采样耗时 {round(time.time()-t0,2)}s, 行数={len(sample) if sample is not None else 0}")
    except Exception as ex:
        notes.append(f"stockdb 健康: 失败 ({type(ex).__name__}) -> 中止采集")
        raise SystemExit("stockdb 预检失败, 终止")
    try:
        free = shutil.disk_usage(out2).free
        notes.append(f"磁盘可用: {round(free/1024/1024,1)}MB (硬停止线={round(hard_stop_bytes/1024/1024,1)}MB)")
    except Exception:
        notes.append("磁盘实测: 不可用")
    try:
        t1 = time.time()
        get_minute_http("000001", str(DATA_END))
        notes.append(f"小批量测速: 1 请求耗时 {round(time.time()-t1,2)}s")
    except Exception as ex:
        notes.append(f"小批量测速: 失败 ({type(ex).__name__})")
    return notes


def fetch_one_minute(code, date, dry_run, est_bytes):
    """返回 (status, payload, sha256, note).
    dry-run: ('sim_done', est_bytes, 'DRYRUN_NO_FETCH', ...) 不向源发请求.
    real:    'done'  -> (buf_bytes, h); 'source_missing' -> (None,''); 'error' -> (None,'').
    """
    if dry_run:
        return "sim_done", est_bytes, "DRYRUN_NO_FETCH", "dry-run 模拟命中"
    from data_source_provenance import get_minute_http
    last = None
    for attempt in range(RETRY + 1):
        try:
            df = get_minute_http(code, str(date))
            if df is None or len(df) == 0:
                return "source_missing", None, "", f"源缺(空) attempt={attempt}"
            import io
            bio = io.BytesIO()
            df.to_parquet(bio)
            buf = bio.getvalue()
            return "done", buf, sha256_bytes(buf), "ok"
        except Exception as ex:
            last = ex
    return "error", None, "", f"重试耗尽: {type(last).__name__}: {last}"


def append_progress(path: Path, rec: dict):
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(rec, ensure_ascii=False) + "\n")


def run(dry_run, hard_stop_bytes, rollover_approved, rollover_path,
        out2=OUT2, limit=None):
    real = not dry_run
    out2.mkdir(parents=True, exist_ok=True)
    (out2 / "by_pair").mkdir(exist_ok=True)
    PROGRESS = progress_paths(out2, real)
    REPORT = out2 / ("dryrun_report.json" if dry_run else "real_report.json")

    base = load_frozen_base()
    base_set = set(base)
    rollover = load_rollover(rollover_path) if (rollover_approved and rollover_path) else set()
    if rollover_approved and not rollover_path:
        raise SystemExit("已 --approve-rollover 但未提供 --rollover-pairs, 拒绝混入顺延层")
    pairs = sorted(base_set | rollover)
    rollover_included = bool(rollover)
    rollover_exceed = sorted({p for p in rollover if p[1] > DATA_END}) if rollover else []

    _, total_bytes = load_minute_index()
    n_idx = len(load_minute_index()[0])
    est_per = max(1, total_bytes // max(1, n_idx)) if total_bytes else 8192

    done = load_progress(PROGRESS, real, out2)
    remaining = [p for p in pairs if p not in done]
    skipped_resume = len(pairs) - len(remaining)

    extra = [p for p in pairs if p not in base_set and not rollover_included]
    assert not extra, f"出现非冻结清单对: {extra[:3]}"

    precheck(dry_run, hard_stop_bytes, out2)

    processed = 0
    cum_bytes = 0
    stopped_by_budget = False
    auto_stop = False
    consecutive_errors = 0
    per_status = {}
    t0 = time.time()
    for (code, date) in remaining:
        if limit is not None and processed >= limit:
            break
        status, payload, h, note = fetch_one_minute(code, date, dry_run, est_per)
        if status == "sim_done":
            if cum_bytes + payload > hard_stop_bytes:
                append_progress(PROGRESS, {"code": code, "date": date, "status": "budget_hold",
                                           "bytes": 0, "sha256": "", "ts": time.time(),
                                           "note": "超预算, 未写入(dry-run 估算)"})
                stopped_by_budget = True
                break
            append_progress(PROGRESS, {"code": code, "date": date, "status": "sim_done",
                                       "bytes": payload, "sha256": h, "ts": time.time(), "note": note})
            cum_bytes += payload
            per_status[status] = per_status.get(status, 0) + 1
            processed += 1
            consecutive_errors = 0
        elif status == "done":
            nbytes = len(payload)
            if cum_bytes + nbytes > hard_stop_bytes:
                append_progress(PROGRESS, {"code": code, "date": date, "status": "budget_hold",
                                           "bytes": 0, "sha256": "", "ts": time.time(),
                                           "note": "超预算, 未写入"})
                stopped_by_budget = True
                break
            out = out2 / "by_pair" / f"{code}_{date}.parquet"
            out.write_bytes(payload)
            cum_bytes += nbytes
            append_progress(PROGRESS, {"code": code, "date": date, "status": "done",
                                       "bytes": nbytes, "sha256": h, "ts": time.time(), "note": note})
            per_status[status] = per_status.get(status, 0) + 1
            processed += 1
            consecutive_errors = 0
        elif status == "source_missing":
            append_progress(PROGRESS, {"code": code, "date": date, "status": "source_missing",
                                       "bytes": 0, "sha256": "", "ts": time.time(), "note": note})
            per_status[status] = per_status.get(status, 0) + 1
            processed += 1
            consecutive_errors = 0
        elif status == "error":
            append_progress(PROGRESS, {"code": code, "date": date, "status": "error",
                                       "bytes": 0, "sha256": "", "ts": time.time(), "note": note})
            per_status[status] = per_status.get(status, 0) + 1
            processed += 1
            consecutive_errors += 1
            if consecutive_errors >= CONN_ERR_LIMIT:
                auto_stop = True
                print(f"[自动停机] 连续连接错误达 {CONN_ERR_LIMIT}, 中止本轮 (剩余未标记, 待重启重试)")
                break

    actual_occ = 0
    bp = out2 / "by_pair"
    if bp.exists():
        actual_occ = sum(f.stat().st_size for f in bp.glob("*.parquet"))

    report = {
        "version": "gcp_minute_collect_v2",
        "run_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "dry_run": dry_run,
        "progress_file": str(PROGRESS),
        "legacy_progress_jsonl_ignored": True,
        "frozen_base_n": len(base_set),
        "rollover_included": rollover_included,
        "rollover_pairs_n": len(rollover),
        "rollover_exceed_data_end_n": len(rollover_exceed),
        "total_pairs": len(pairs),
        "skipped_resume": skipped_resume,
        "processed_this_run": processed,
        "stopped_by_budget": stopped_by_budget,
        "auto_stop_connection_errors": auto_stop,
        "hard_stop_bytes": hard_stop_bytes,
        "est_bytes_per_file": est_per,
        "cum_bytes_recorded": cum_bytes,
        "actual_dir_occupancy_bytes": actual_occ,
        "per_status": per_status,
        "frozen_only_assert": (len(extra) == 0),
    }
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[采集 v2 dry-run={dry_run}] 冻结基础={len(base_set)} 顺延纳入={rollover_included}")
    print(f"  进度文件={PROGRESS.name} 待处理={len(pairs)} 续跑跳过={skipped_resume} 本轮={processed}")
    print(f"  预算停机={stopped_by_budget} 连接错误停机={auto_stop}")
    print(f"  记录累计字节={round(cum_bytes/1024/1024,2)}MB 实际目录占用={round(actual_occ/1024/1024,2)}MB "
          f"/ 硬线={round(hard_stop_bytes/1024/1024,1)}MB")
    print(f"  状态分布={per_status} 只认冻结清单={report['frozen_only_assert']}")
    print(f"  报告: {REPORT.name}")
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--real", action="store_true", help="真实采集(向源发请求); 默认 dry-run")
    ap.add_argument("--hard-stop-bytes", type=int, default=HARD_STOP_BYTES_DEFAULT)
    ap.add_argument("--approve-rollover", action="store_true",
                    help="明确批准纳入顺延层 (需同时给 --rollover-pairs)")
    ap.add_argument("--rollover-pairs", type=str, default="")
    ap.add_argument("--out2", type=str, default=str(OUT2),
                    help="独立缓存/进度目录 (隔离用)")
    ap.add_argument("--limit", type=int, default=None,
                    help="本轮最多处理 N 对 (有界试跑用)")
    args = ap.parse_args()
    run(not args.real, args.hard_stop_bytes, args.approve_rollover,
        args.rollover_pairs, Path(args.out2), args.limit)


if __name__ == "__main__":
    main()
