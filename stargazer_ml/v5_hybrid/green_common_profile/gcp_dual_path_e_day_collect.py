"""双路径 E 日定向采集执行器 (SIGNED task order, 2026-10-01).

只读取冻结请求子集 dual_path_e_day_request_subset.txt (952 行, schema code,E,P,setups),
可续跑, 严守 50MB 上限.

启动硬门 (用户指定):
  1. 子集规范化内容哈希 + 文件字节哈希 + 唯一键数 全部匹配登记 -> 否则停止, 不采集;
  2. stockdb 状态预检 (控制样本可取) -> 否则停止.
范围纪律:
  - 4 条 HIT_INVALID 与 22 条截止外均不在子集内, 本执行器只认子集 952 行, 不扩围、不替补;
  - 缓存命中 (已有 (code,E)) 跳过重抓; 失败按合同显式保留 (source_missing / grid_anomaly).
输出:
  gcp_round_b_small_sample_v1/minute_ext/{code}.parquet  (追加新快照, 只读旧缓存不覆盖)
  gcp_round_b_small_sample_v1/dual_path_e_day_collect_report.json
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

import pandas as pd

HERE = Path(__file__).parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "step9_preflight"))
from data_source_provenance import get_minute_http  # noqa: E402

CONTRACTS = HERE / "contracts"
SUBSET = CONTRACTS / "dual_path_e_day_request_subset.txt"
OUT = HERE / "gcp_round_b_small_sample_v1"
SNAP_DIRS = (OUT / "minute", OUT / "minute_ext", OUT / "minute_car")  # 只读已有缓存, 命中跳过
WRITE_DIR = OUT / "minute_ext"   # 新 E 日快照追加于此 (覆盖盘点探针会扫描本目录)
REPORT = OUT / "dual_path_e_day_collect_report.json"

EXPECTED_SUBSET_CONTENT_SHA = "f0795871a3c11376bd58f182f6d95678b04b38b29d2d2d8fd39e1d2c248641d8"
EXPECTED_SUBSET_BYTE_SHA = "4832798ab562e18e7fd4346b2a434578f8193084e6b6060e99a4c80ca93cbdba"
EXPECTED_N = 952
MAX_RETRY = 3
SNAPSHOT_CAP = 50 * 1024 * 1024   # 50MB 硬上限 (任务单 §6)

TRAIN_E_MAX = 20260305
VAL_E_MIN, VAL_E_MAX = 20260309, 20260609
TEST_E_MIN = 20260610


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def segment_of(e_date: int) -> str:
    if e_date <= TRAIN_E_MAX:
        return "train"
    if VAL_E_MIN <= e_date <= VAL_E_MAX:
        return "val"
    if e_date >= TEST_E_MIN:
        return "test"
    return "gap"


def verify_subset_gate():
    raw = SUBSET.read_bytes()
    byte_sha = sha256_bytes(raw)
    norm = raw.decode("utf-8", "replace").replace("\r\n", "\n").replace("\r", "\n").rstrip("\n")
    content_sha = sha256_bytes(norm.encode("utf-8"))
    lines = [ln for ln in norm.split("\n") if ln.strip()]
    uniq = sorted(set(lines))
    n = len(uniq)
    ok = (content_sha == EXPECTED_SUBSET_CONTENT_SHA
          and byte_sha == EXPECTED_SUBSET_BYTE_SHA
          and n == EXPECTED_N)
    return ok, content_sha, byte_sha, n, uniq


def grid_ok(df: pd.DataFrame) -> tuple[bool, str]:
    n = len(df)
    if n not in (241, 242):
        return False, f"slots={n} (非241/242)"
    for col in ("open", "close", "high", "low"):
        if col in df and (df[col] <= 0).any():
            return False, f"{col}<=0"
    if "volume" in df and (df["volume"] < 0).any():
        return False, "volume<0"
    if "amount" in df and (df["amount"] < 0).any():
        return False, "amount<0"
    return True, "OK"


def snapshot_bytes() -> int:
    return sum(f.stat().st_size for d in (OUT,) for f in d.rglob("*") if f.is_file())


def main():
    t0 = time.time()
    WRITE_DIR.mkdir(parents=True, exist_ok=True)

    # ============ 0. 子集哈希门 ============
    ok, csha, bsha, n, uniq = verify_subset_gate()
    print("=" * 60)
    print("[子集哈希门]")
    print(f"  规范化内容哈希: {csha}")
    print(f"  期望          : {EXPECTED_SUBSET_CONTENT_SHA}")
    print(f"  文件字节哈希  : {bsha}")
    print(f"  期望          : {EXPECTED_SUBSET_BYTE_SHA}")
    print(f"  唯一行数      : {n} (期望 {EXPECTED_N})")
    if not ok:
        print("  >> 门控失败, 停止采集 (子集被改动或计数不符)")
        REPORT.write_text(json.dumps({
            "status": "SUBSET_GATE_FAILED", "content_sha": csha,
            "byte_sha": bsha, "n_unique": n, "expected_n": EXPECTED_N,
        }, ensure_ascii=False, indent=1), encoding="utf-8")
        sys.exit(1)
    print("  >> 门控通过")

    # ============ 1. stockdb 状态预检 ============
    print("[stockdb 状态预检]")
    try:
        probe = get_minute_http("600519", "20260423")
        if probe is None or probe.empty:
            print("  >> stockdb 返回空, 状态异常, 停止采集")
            sys.exit(1)
        print(f"  >> stockdb 可达, 控制样本 600519/20260423 返回 {len(probe)} 行")
    except Exception as e:
        print(f"  >> stockdb 不可达: {e}; 停止采集")
        sys.exit(1)

    # ============ 2. 已有缓存索引 (只读, 命中跳过) ============
    have: dict = {}
    for d in SNAP_DIRS:
        if not d.exists():
            continue
        for pf in d.glob("*.parquet"):
            code = pf.stem
            try:
                m = pd.read_parquet(pf, columns=["date"])
                days = set(int(x) for x in m["date"].astype(str).str[:8].astype(int).tolist())
                have.setdefault(code, set()).update(days)
            except Exception:
                pass

    # ============ 3. 50MB 上限预检 ============
    if snapshot_bytes() > SNAPSHOT_CAP:
        print(f"  >> 快照已超 50MB ({snapshot_bytes()}), 停止采集")
        sys.exit(1)
    print(f"[快照当前 {snapshot_bytes()/1024/1024:.2f}MB / 上限 50MB]")

    # ============ 4. 逐行采集 (按 code 聚合, 可续跑) ============
    by_code: dict = {}
    for line in uniq:
        parts = line.split(",")
        code, e_str, p_str, setups = parts[0], parts[1], parts[2], parts[3]
        by_code.setdefault(code, []).append((int(e_str), int(p_str), setups))

    summary = {
        "status": "RUNNING",
        "subset_gate": {"passed": True, "content_sha": csha, "byte_sha": bsha, "n_unique": n},
        "stockdb_preflight": "OK",
        "snapshot_cap_bytes": SNAPSHOT_CAP,
        "totals": {"requested": 0, "cache_hit": 0, "fetched_success": 0,
                   "source_missing": 0, "fetch_error": 0, "grid_anomaly": 0},
        "by_setup": {},       # A_SDP / B_LVB
        "by_year": {},        # P 年
        "by_segment": {},     # E 时段 train/val/test/gap
        "per_stock": {},
        "source_missing_detail": [],
        "grid_anomaly_detail": [],
        "fetch_error_detail": [],
        "incomplete_detail": [],  # 续跑中断或截断遗留
    }
    summary["totals"]["requested"] = sum(len(v) for v in by_code.values())

    def bump(d, key, field):
        d.setdefault(key, {"requested": 0, "cache_hit": 0, "fetched_success": 0,
                           "source_missing": 0, "fetch_error": 0, "grid_anomaly": 0})
        d[key][field] += 1

    for code in sorted(by_code):
        rows = sorted(by_code[code])
        s = {"requested": len(rows), "cache_hit": 0, "fetched_success": 0,
             "source_missing": 0, "fetch_error": 0, "grid_anomaly": 0, "valid_dates": 0}
        old_set = have.get(code, set())
        new_rows = []

        for e, p, setups in rows:
            for st in setups.split("+"):
                bump(summary["by_setup"], st, "requested")
            bump(summary["by_year"], str(p)[:4], "requested")
            bump(summary["by_segment"], segment_of(e), "requested")

            # 缓存命中 (已有 (code,E)) -> 跳过, 不重抓 (含 4 条 HIT_INVALID 本就不在子集)
            if e in old_set:
                s["cache_hit"] += 1
                summary["totals"]["cache_hit"] += 1
                for st in setups.split("+"):
                    bump(summary["by_setup"], st, "cache_hit")
                bump(summary["by_year"], str(p)[:4], "cache_hit")
                bump(summary["by_segment"], segment_of(e), "cache_hit")
                continue

            # 请求 (有限重试)
            df = pd.DataFrame()
            err = None
            for _ in range(MAX_RETRY):
                try:
                    df = get_minute_http(code, str(e))
                    break
                except Exception as e2:
                    err = str(e2)
                    time.sleep(0.3)
            if df is None or df.empty:
                if err is not None:
                    s["fetch_error"] += 1
                    summary["totals"]["fetch_error"] += 1
                    summary["fetch_error_detail"].append({"code": code, "E": e, "error": err})
                else:
                    s["source_missing"] += 1
                    summary["totals"]["source_missing"] += 1
                    summary["source_missing_detail"].append({"code": code, "E": e, "P": p})
                for st in setups.split("+"):
                    bump(summary["by_setup"], st, "source_missing" if err is None else "fetch_error")
                bump(summary["by_year"], str(p)[:4], "source_missing" if err is None else "fetch_error")
                bump(summary["by_segment"], segment_of(e), "source_missing" if err is None else "fetch_error")
                continue

            # 质检
            gok, gmsg = grid_ok(df)
            df["code"] = code
            new_rows.append(df)
            if gok:
                s["fetched_success"] += 1
                summary["totals"]["fetched_success"] += 1
                for st in setups.split("+"):
                    bump(summary["by_setup"], st, "fetched_success")
                bump(summary["by_year"], str(p)[:4], "fetched_success")
                bump(summary["by_segment"], segment_of(e), "fetched_success")
            else:
                s["grid_anomaly"] += 1
                summary["totals"]["grid_anomaly"] += 1
                summary["grid_anomaly_detail"].append({"code": code, "E": e, "reason": gmsg})
                for st in setups.split("+"):
                    bump(summary["by_setup"], st, "grid_anomaly")
                bump(summary["by_year"], str(p)[:4], "grid_anomaly")
                bump(summary["by_segment"], segment_of(e), "grid_anomaly")
            old_set.add(e)  # 即便异常也标记已抓 (保留证据, 续跑不重抓)

        # 写新快照 (追加到 minute_ext/{code}.parquet)
        ext_path = WRITE_DIR / f"{code}.parquet"
        if new_rows:
            new_df = pd.concat(new_rows, ignore_index=True) if len(new_rows) > 1 else new_rows[0]
            if ext_path.exists():
                try:
                    old_ext = pd.read_parquet(ext_path)
                    new_df = pd.concat([old_ext, new_df], ignore_index=True)
                except Exception:
                    pass
            new_df = new_df.sort_values("date").drop_duplicates(subset=["date"]).reset_index(drop=True)
            new_df.to_parquet(ext_path, index=False)

        s["valid_dates"] = len(old_set)
        summary["per_stock"][code] = s
        print(f"  {code}: req={s['requested']} hit={s['cache_hit']} ok={s['fetched_success']} "
              f"src_miss={s['source_missing']} err={s['fetch_error']} anom={s['grid_anomaly']} "
              f"valid_dates={s['valid_dates']}")

        # 50MB 上限运行期强制
        if snapshot_bytes() > SNAPSHOT_CAP:
            print(f"  >> 快照超 50MB ({snapshot_bytes()}), 中止后续请求 (已落盘部分可续跑)")
            summary["status"] = "ABORTED_50MB_CAP"
            break

    summary["elapsed_s"] = round(time.time() - t0, 1)
    if summary["status"] == "RUNNING":
        # 判定是否全部完成
        done = (summary["totals"]["cache_hit"] + summary["totals"]["fetched_success"]
                + summary["totals"]["source_missing"] + summary["totals"]["fetch_error"]
                + summary["totals"]["grid_anomaly"])
        summary["status"] = "COMPLETED" if done == summary["totals"]["requested"] else "INCOMPLETE"

    REPORT.write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    t = summary["totals"]
    print("\n" + "=" * 60)
    print("[采集汇总]")
    print(f"  请求总数       : {t['requested']}")
    print(f"  缓存命中跳过  : {t['cache_hit']}")
    print(f"  请求成功       : {t['fetched_success']}")
    print(f"  源缺           : {t['source_missing']}")
    print(f"  请求异常       : {t['fetch_error']}")
    print(f"  网格异常(已留): {t['grid_anomaly']}")
    print(f"  状态           : {summary['status']}")
    print(f"  快照           : {snapshot_bytes()/1024/1024:.2f}MB / 50MB")
    print(f"  报告: {REPORT}")
    print("  >> 本轮为 E 日分钟补抓 (合同 §10 能力边界: 仅 E 日, 不含 H3/H5/训练)")


if __name__ == "__main__":
    main()
