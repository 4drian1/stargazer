"""CAR v0.1 独立请求清单生成 (2026-09-29, 用户审批阶段).

读取冻结候选清单 (car_v0_1_candidate_list.txt), 生成:
  - car_v0_1_minute_request_list.txt:  分钟去重 (code,date), 注明 E日/未来窗 + 缓存命中
  - car_v0_1_adj_request_list.txt:     复权链独立股票清单
  - car_v0_1_request_manifest.json:    SHA256 + 数据源 + 代码版本 + 数量 + 重试 + 磁盘上限

本轮只做 30m/60m/收回后10m; 不加 R 历史基线、不做真实 5d/10d、不训练.
失败保持 UNKNOWN, 不当空链. 有限重试 (3次), 源缺显式记录, 不插值、不自动扩围.

授权边界: 本清单审批后才可执行真实回放. 现在只生成清单待签.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "step9_preflight"))
from data_source_provenance import get_daily_parquet  # noqa: E402

CONTRACTS = HERE / "contracts"
SNAP_OLD = HERE / "gcp_round_b_small_sample_v1" / "minute"
SNAP_EXT = HERE / "gcp_round_b_small_sample_v1" / "minute_ext"
ADJ_CACHE = HERE / "gcp_round_b_small_sample_v1" / "adj_cache.json"

# 有限重试 + 磁盘上限 (合同 §7)
MAX_RETRY = 3
DISK_CAP_MB = 50  # 84 请求 * ~0.4MB/股日 ≈ 34MB; 上限 50MB
DATA_SOURCE = "canonical stockdb HTTP (data_source_provenance.get_minute_http)"
CODE_VERSION = "gcp_car_v0_1_labels.py@2026-09-29 + gcp_car_v0_1_request_manifest.py@2026-09-29"


def sha256_hex(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def main():
    # 读候选清单
    cand_text = (CONTRACTS / "car_v0_1_candidate_list.txt").read_text(encoding="utf-8")
    cand_lines = [l for l in cand_text.strip().split("\n") if l]
    candidates = []
    for l in cand_lines:
        code, setup, e = l.split(",")
        candidates.append({"code": code, "setup": setup, "E": int(e)})
    print(f"[候选清单] {len(candidates)} 行, {len(set(c['code'] for c in candidates))} 股")

    # 交易日历
    panel = get_daily_parquet()
    cal = sorted(int(d) for d in panel["date"].unique())
    cal_idx = {d: i for i, d in enumerate(cal)}

    # 缓存命中
    cache_days = {}
    for d in (SNAP_OLD, SNAP_EXT):
        if not d.exists():
            continue
        for pf in d.glob("*.parquet"):
            code = pf.stem
            m = pd.read_parquet(pf, columns=["date"])
            days = set(m["date"].astype(str).str[:8].astype(int).unique().tolist())
            cache_days.setdefault(code, set()).update(days)

    adj_existing = set()
    if ADJ_CACHE.exists():
        adj_existing = set(json.loads(ADJ_CACHE.read_text(encoding="utf-8")).keys())

    # ---- 分钟请求: 去重 (code, date) ----
    # E日 + 未来窗可能触及的下一交易日 (下午信号 60m + 持续性 10m 可能跨日)
    minute_req = {}  # (code,date) → {role, cache_hit}
    for c in candidates:
        i_e = cal_idx[c["E"]]
        future_days = {c["E"]}
        if i_e + 1 < len(cal):
            future_days.add(cal[i_e + 1])
        for d in future_days:
            key = (c["code"], d)
            if key not in minute_req:
                hit = d in cache_days.get(c["code"], set())
                role = "E_DAY" if d == c["E"] else "FUTURE_WINDOW_DAY"
                minute_req[key] = {"role": role, "cache_hit": hit, "setup": c["setup"]}

    # 按字典序输出
    min_lines = []
    n_hit = n_miss = 0
    for (code, date) in sorted(minute_req.keys()):
        info = minute_req[(code, date)]
        if info["cache_hit"]:
            n_hit += 1
        else:
            n_miss += 1
        min_lines.append(f"{code},{date},{info['role']},cache={'HIT' if info['cache_hit'] else 'MISS'}")

    min_text = "\n".join(min_lines) + "\n"
    min_f = CONTRACTS / "car_v0_1_minute_request_list.txt"
    min_f.write_bytes(min_text.encode("utf-8"))
    min_byte_sha = sha256_hex(min_f.read_bytes())
    min_norm_sha = sha256_hex(min_text.encode("utf-8"))

    # ---- 复权链请求: 独立股票清单 ----
    adj_codes = sorted(set(c["code"] for c in candidates))
    adj_lines = []
    n_adj_hit = n_adj_miss = 0
    for code in adj_codes:
        hit = code in adj_existing
        if hit:
            n_adj_hit += 1
        else:
            n_adj_miss += 1
        adj_lines.append(f"{code},cache={'HIT' if hit else 'MISS'}")
    adj_text = "\n".join(adj_lines) + "\n"
    adj_f = CONTRACTS / "car_v0_1_adj_request_list.txt"
    adj_f.write_bytes(adj_text.encode("utf-8"))
    adj_byte_sha = sha256_hex(adj_f.read_bytes())
    adj_norm_sha = sha256_hex(adj_text.encode("utf-8"))

    # ---- 清单摘要 ----
    manifest = {
        "status": "PENDING_APPROVAL — 审批后才可执行真实回放",
        "date": "2026-09-29",
        "scope": "20股 / 34候选日 (candidate_list); 30m/60m/收回后10m; 不加R组/不做5d10d/不训练",
        "code_version": CODE_VERSION,
        "data_source": DATA_SOURCE,
        "minute_request": {
            "file": "car_v0_1_minute_request_list.txt",
            "format": "code,date,role,cache=HIT/MISS  (role=E_DAY/FUTURE_WINDOW_DAY)",
            "unique_code_date_pairs": len(minute_req),
            "cache_hit": n_hit,
            "cache_miss": n_miss,
            "new_requests": n_miss,
            "byte_sha256": min_byte_sha,
            "normalized_sha256_utf8_lf_trailing_nl": min_norm_sha,
        },
        "adj_request": {
            "file": "car_v0_1_adj_request_list.txt",
            "format": "code,cache=HIT/MISS",
            "unique_codes": len(adj_codes),
            "cache_hit": n_adj_hit,
            "cache_miss": n_adj_miss,
            "new_requests": n_adj_miss,
            "byte_sha256": adj_byte_sha,
            "normalized_sha256_utf8_lf_trailing_nl": adj_norm_sha,
            "failure_policy": "失败保持 UNKNOWN, 不当空链; 不插值",
        },
        "retry_policy": f"有限重试 {MAX_RETRY} 次; 源缺显式记录; 不插值、不自动扩围",
        "disk_cap_mb": DISK_CAP_MB,
        "estimated_storage_mb": round(n_miss * 0.4, 1),  # ~0.4MB/股日
        "candidate_list_sha256": sha256_hex(
            (CONTRACTS / "car_v0_1_candidate_list.txt").read_bytes()),
        "real_replay_guardrails": [
            "按分窗口状态决定有效性: hold 缺失不删除已完整有效的 60m 主标签",
            "核验闭环终点: hold 提前失败时修改其记录终点之后的数据不应改变已闭环输出",
            "5d/10d 真实回放不做; window_complete 一律 False (未抓取)",
            "R组历史基线本轮不加",
        ],
    }
    manifest_f = CONTRACTS / "car_v0_1_request_manifest.json"
    manifest_f.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    # 打印
    print(f"\n[分钟请求] 去重 (code,date):")
    print(f"  唯一对数: {len(minute_req)}")
    print(f"  缓存命中: {n_hit}  缓存缺失(新请求): {n_miss}")
    print(f"  字节SHA256: {min_byte_sha}")
    print(f"  规范化SHA256 (UTF-8/LF/末尾换行): {min_norm_sha}")
    print(f"\n[复权链请求] 独立股票清单:")
    print(f"  股票数: {len(adj_codes)}")
    print(f"  缓存命中: {n_adj_hit}  缓存缺失(新请求): {n_adj_miss}")
    print(f"  字节SHA256: {adj_byte_sha}")
    print(f"  规范化SHA256: {adj_norm_sha}")
    print(f"\n[策略]")
    print(f"  有限重试: {MAX_RETRY} 次")
    print(f"  磁盘上限: {DISK_CAP_MB}MB")
    print(f"  估计存储: {manifest['estimated_storage_mb']}MB")
    print(f"\n[审批状态] {manifest['status']}")
    print(f"  清单已落盘: {min_f.name}, {adj_f.name}, {manifest_f.name}")


if __name__ == "__main__":
    main()
