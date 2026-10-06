"""GCP v0.2 有限补抓执行 (REV2-F-exec, 2026-09-29).

用户放行: 按冻结清单串行补抓一次, 不必等 865 全成功; 失败记录完整、范围不变即完成.
本本轮仅放行有限数据补抓, 非正式训练或策略晋级.

启动硬门: 哈希核验 (用户指定)
  - 规范化文本哈希 = 86a210a2da559eca... (UTF-8, LF, 末尾换行) — 冻结清单的权威哈希
  - 当前文件字节哈希 ≈ 23081540... (Windows CRLF 致异) — 仅报告, 不用于门控, 不混用
  - 启动时重新核验 865 个唯一 (code,date) 键 + 规范化哈希; 不一致即停止, 不补抓.

措辞约束 (用户): 上市日期代理 / 零量占位 / 冻结研究请求清单 — 不用"新上市/已确认停牌/权威化"
  证明措辞; 本补抓不用于证明交易规则有效.

执行范围:
  1. 按冻结清单串行补抓, 缓存命中不重复请求
  2. 失败有限重试, 源缺显式记录; 不插值、不自动扩围
  3. 新快照追加保存 (minute_ext/), 旧输入 (minute/) 保留; 逐日质量检查
  4. 输出请求成功/缓存命中/源缺/异常数量 + 各股可用日期数
  5. (后续) 再运行固定 60m 主检验的 v3 稳定性分析 — 本脚本不含, 另行

输出: gcp_round_b_small_sample_v1/v02_backfill_execute_report.json
      + minute_ext/{code}.parquet  (新快照, 追加, 可断点续抓)
"""
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

OUT = HERE / "gcp_round_b_small_sample_v1"
LIST_TXT = OUT / "v02_backfill_request_list.txt"
SNAP_OLD = OUT / "minute"            # 旧输入, 只读
SNAP_NEW = OUT / "minute_ext"        # 新快照, 追加
REPORT = OUT / "v02_backfill_execute_report.json"

EXPECTED_NORM_SHA = "86a210a2da559ecaecfca47b34108195d7720b46124096a0b0174bfdb99c9654"
EXPECTED_N = 865
MAX_RETRY = 3


def sha256_hex(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def verify_hash_gate():
    """启动硬门: 规范化哈希 + 唯一键数. 返回 (ok, norm_sha, byte_sha, unique_lines, n_unique)."""
    raw = LIST_TXT.read_bytes()
    byte_sha = sha256_hex(raw)
    # 规范化: UTF-8 解码, CRLF->LF, 保证末尾换行
    norm = raw.decode("utf-8", "replace").replace("\r\n", "\n").replace("\r", "\n")
    if not norm.endswith("\n"):
        norm += "\n"
    norm_sha = sha256_hex(norm.encode("utf-8"))
    # 唯一键
    lines = [ln for ln in norm.rstrip("\n").split("\n") if ln.strip()]
    unique = sorted(set(lines))
    n_unique = len(unique)
    ok = (norm_sha == EXPECTED_NORM_SHA) and (n_unique == EXPECTED_N)
    return ok, norm_sha, byte_sha, unique, n_unique


def grid_ok(df: pd.DataFrame) -> tuple[bool, str]:
    """逐日质量检查: 槽位 241/242, OHLC>0, volume>=0, amount>=0. 返回 (ok, reason)."""
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


def main():
    SNAP_NEW.mkdir(parents=True, exist_ok=True)

    # ============ 0. 哈希门 ============
    ok, norm_sha, byte_sha, unique, n_unique = verify_hash_gate()
    print("=" * 60)
    print("[哈希门]")
    print(f"  规范化哈希 (UTF-8/LF/末尾换行): {norm_sha}")
    print(f"  期望规范化哈希              : {EXPECTED_NORM_SHA}")
    print(f"  当前文件字节哈希 (CRLF,仅报告): {byte_sha}")
    print(f"  唯一 (code,date) 键数: {n_unique} (期望 {EXPECTED_N})")
    if not ok:
        print(f"  >> 门控失败, 停止补抓 (norm_match={norm_sha==EXPECTED_NORM_SHA}, n_match={n_unique==EXPECTED_N})")
        REPORT.write_text(json.dumps({
            "status": "HASH_GATE_FAILED",
            "norm_sha": norm_sha, "expected_norm_sha": EXPECTED_NORM_SHA,
            "byte_sha": byte_sha, "n_unique": n_unique, "expected_n": EXPECTED_N,
        }, ensure_ascii=False, indent=1), encoding="utf-8")
        sys.exit(1)
    print("  >> 门控通过, 进入补抓")
    print("=" * 60)

    # 预加载旧缓存的 (code, date) 索引, 用于缓存命中判断
    old_have = {}   # code -> set(date_int)
    if SNAP_OLD.exists():
        for pf in SNAP_OLD.glob("*.parquet"):
            code = pf.stem
            try:
                df = pd.read_parquet(pf, columns=["date"])
                df["day"] = df["date"].astype(str).str[:8].astype(int)
                old_have[code] = set(df["day"].unique().tolist())
            except Exception:
                old_have[code] = set()

    # 断点续抓: 预加载新快照已有 (code, date)
    new_have = {}
    for pf in SNAP_NEW.glob("*.parquet"):
        code = pf.stem
        try:
            df = pd.read_parquet(pf, columns=["date"])
            df["day"] = df["date"].astype(str).str[:8].astype(int)
            new_have[code] = set(df["day"].unique().tolist())
        except Exception:
            new_have[code] = set()

    # ============ 1. 串行补抓 ============
    # 按股聚合, 便于写 per-code parquet
    by_code = {}
    for line in unique:
        code, dstr = line.split(",")
        by_code.setdefault(code, []).append(int(dstr))

    summary = {
        "hash_gate": {"passed": True, "norm_sha": norm_sha, "byte_sha": byte_sha,
                      "n_unique": n_unique},
        "totals": {"requested": 0, "cache_hit_old": 0, "cache_hit_new": 0,
                   "fetched_success": 0, "source_missing": 0, "fetch_error": 0,
                   "grid_anomaly": 0},
        "per_stock": {},
        "source_missing_detail": [],
        "grid_anomaly_detail": [],
        "fetch_error_detail": [],
    }
    summary["totals"]["requested"] = sum(len(v) for v in by_code.values())

    for code in sorted(by_code):
        dates = sorted(by_code[code])
        s = {"requested": len(dates), "cache_hit_old": 0, "cache_hit_new": 0,
             "fetched_success": 0, "source_missing": 0, "fetch_error": 0,
             "grid_anomaly": 0, "valid_dates": 0}
        old_set = old_have.get(code, set())
        new_set = new_have.get(code, set())
        new_rows = []   # 本轮新抓的行 (追加到新快照)

        for d in dates:
            # 缓存命中 (旧或新)
            if d in old_set:
                s["cache_hit_old"] += 1
                summary["totals"]["cache_hit_old"] += 1
                continue
            if d in new_set:
                s["cache_hit_new"] += 1
                summary["totals"]["cache_hit_new"] += 1
                continue
            # 请求 (有限重试)
            df = pd.DataFrame()
            err = None
            for attempt in range(MAX_RETRY):
                try:
                    df = get_minute_http(code, str(d))
                    break
                except Exception as e:
                    err = str(e)
                    time.sleep(0.3)
            if df is None or df.empty:
                # 源缺 (零量占位也算空? — 零量占位 HTTP 返回 241 行 volume=0, 非空)
                if err is not None:
                    s["fetch_error"] += 1
                    summary["totals"]["fetch_error"] += 1
                    summary["fetch_error_detail"].append({"code": code, "date": d, "error": err})
                else:
                    s["source_missing"] += 1
                    summary["totals"]["source_missing"] += 1
                    summary["source_missing_detail"].append({"code": code, "date": d})
                continue
            # 质检
            gok, gmsg = grid_ok(df)
            df["code"] = code
            new_rows.append(df)
            if gok:
                s["fetched_success"] += 1
                summary["totals"]["fetched_success"] += 1
            else:
                s["grid_anomaly"] += 1
                summary["totals"]["grid_anomaly"] += 1
                summary["grid_anomaly_detail"].append({"code": code, "date": d, "reason": gmsg})
            new_set.add(d)   # 标记已抓 (即便异常也写入, 保留证据)

        # 写新快照 (追加到已有 minute_ext/{code}.parquet)
        ext_path = SNAP_NEW / f"{code}.parquet"
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

        # 可用日期数 = 旧缓存有效 + 新抓有效 (同日不重计)
        valid_dates = len(old_set | new_set)
        s["valid_dates"] = valid_dates
        summary["per_stock"][code] = s
        print(f"  {code}: req={s['requested']} old_hit={s['cache_hit_old']} "
              f"new_hit={s['cache_hit_new']} fetch_ok={s['fetched_success']} "
              f"src_miss={s['source_missing']} err={s['fetch_error']} "
              f"anom={s['grid_anomaly']} valid_dates={valid_dates}")

    # ============ 2. 汇总 ============
    t = summary["totals"]
    REPORT.write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    print("\n" + "=" * 60)
    print("[补抓汇总]")
    print(f"  请求总数       : {t['requested']}")
    print(f"  缓存命中(旧)  : {t['cache_hit_old']}")
    print(f"  缓存命中(新续): {t['cache_hit_new']}")
    print(f"  请求成功       : {t['fetched_success']}")
    print(f"  源缺           : {t['source_missing']}")
    print(f"  请求异常       : {t['fetch_error']}")
    print(f"  网格异常(已留): {t['grid_anomaly']}")
    print(f"  报告: {REPORT}")
    print(f"  新快照: {SNAP_NEW}")
    print("  >> 本轮为有限数据补抓, 非训练/策略晋级; 失败已记录, 范围不变")


if __name__ == "__main__":
    main()
