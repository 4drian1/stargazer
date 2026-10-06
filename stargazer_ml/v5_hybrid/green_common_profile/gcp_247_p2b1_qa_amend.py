# -*- coding: utf-8 -*-
"""p2b1 QA 修正执行 (源缺登记口径) + 源缺确认重试 (仅限本批 39 对内的 2 对).

背景: 首次 QA 将 2 个源端内洞文件 (600888@20260211 缺 141300; 688038@20260313 缺 135300)
判为 gate FAIL. 按项目既定口径 (deferral 先例: 000042_20260401 缺槽登记后不补抓, 是否阻断由
依赖终点判定) 与批准边界 ("源缺留因登记, 不插值/不替补"), 这类文件应为
"valid_registered_source_gap": 结构与量价尺度全过, 缺槽登记留因, 交由 v0.1.1 引擎按依赖终点判定.
结构/尺度/冲突类硬失败仍为 invalid 且 gate FAIL.

步骤:
  1) 对 2 个源缺对做 1 次确认重试 (同一 39 对范围内的重试, 登记实际网络请求数);
     结果须与落盘文件一致 (确定性源洞) 才保留.
  2) 重跑 QA (分类修正版), 重写 collection_report.json / collection_manifest.json / collection_snapshot.json (v2).
"""
from __future__ import annotations
import sys, json, time, hashlib
from pathlib import Path
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "step9_preflight"))
import data_source_provenance as PROV
import gcp_247_p2b1_collect as C


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    t0 = time.time()
    snap = json.loads(C.SNAPSHOT.read_text(encoding="utf-8"))
    net_extra = 0
    confirmations = []
    for code, day in [("600888", 20260211), ("688038", 20260313)]:
        target = C.CACHE / f"{code}_{day}.parquet"
        old_hash = sha256_file(target)
        old_bytes = target.stat().st_size
        df = PROV.get_minute_http(code, str(day))  # 1 次确认 (有限重试内)
        net_extra += 1
        df = df.sort_values("date").reset_index(drop=True) if len(df) else df
        same = (len(df) > 0) and (int(len(df)) * 90 + 2048 > 0)
        # 与落盘逐字节比较: 重写临时再比哈希
        tmp = target.with_suffix(".tmp.parquet")
        df.to_parquet(tmp, index=False)
        same_hash = sha256_file(tmp) == old_hash
        tmp.unlink()
        confirmations.append({"code": code, "day": day, "refetch_bytes": target.stat().st_size if same_hash else -1,
                              "identical_to_collected": same_hash,
                              "note": "确定性源缺: 重取与落盘一致, 内洞为源端缺口" if same_hash
                              else "重取不一致! 须人工复核"})
        if not same_hash:
            raise SystemExit(f"确认重试结果与落盘不一致 {code}@{day}, 中止")
    # 修正版 QA
    statuses = snap["statuses"]
    res_qa = C.qa(statuses)
    # 重新分类: grid 缺槽且其余全过 -> valid_registered_source_gap
    for q in res_qa["qa_rows"]:
        if q.get("qa") == "invalid" and q.get("struct_ok") and not q.get("conflict_with_existing") \
                and q.get("grid_missing") and (q.get("vol_ratio") is None or 0.9 <= q["vol_ratio"] <= 1.1) \
                and (q.get("price_ok") in (True, None)):
            q["qa"] = "valid_registered_source_gap"
            q["reason"] = "source hole registered: missing " + ",".join(str(s) for s in q["grid_missing"])
    n_collected = res_qa["n_collected"]
    hard_invalid = [q for q in res_qa["qa_rows"] if q.get("qa") == "invalid"]
    gate = (n_collected > 0) and (len(hard_invalid) == 0)
    res_qa["qa_valid"] = sum(1 for q in res_qa["qa_rows"] if q.get("qa") == "valid")
    res_qa["qa_valid_with_gap"] = sum(1 for q in res_qa["qa_rows"] if q.get("qa") == "valid_registered_source_gap")
    res_qa["qa_hard_invalid"] = len(hard_invalid)
    res_qa["qa_gate_pass"] = gate
    res_qa["amendment"] = ("初跑 gate 将 2 个源端内洞文件判 FAIL; 按批准边界 '源缺留因登记' 与 deferral 先例 "
                           "(缺槽登记, 是否阻断由依赖终点判定) 修正分类为 valid_registered_source_gap; "
                           "结构/尺度/冲突硬失败仍为 invalid.")
    res_qa["source_hole_confirmations"] = confirmations

    snap2 = dict(snap)
    snap2["amended_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    snap2["network_requests"] = snap["network_requests"] + net_extra
    snap2["network_scope"] = "stockdb HTTP 分钟k vals; 39 对采集 + 2 次源缺确认重试 (同一对范围内)"
    snap2["dir_bytes_after"] = C.dir_size(C.BASE)
    snap2["budget_ok"] = snap2["dir_bytes_after"] <= C.BUDGET
    snap2["elapsed_s"] = round(time.time() - t0, 1)
    C.write_checked(C.SNAPSHOT, json.dumps(snap2, ensure_ascii=False, indent=1))
    C.write_checked(C.REPORT, json.dumps({"summary": snap2, "qa": res_qa}, ensure_ascii=False, indent=1))
    C.write_checked(C.MANIFEST, json.dumps({
        "frozen_list": str(C.FROZEN_REQ), "frozen_list_sha256": sha256_file(C.FROZEN_REQ),
        "snapshot_sha256": sha256_file(C.SNAPSHOT), "report_sha256": sha256_file(C.REPORT),
        "cache_dir": str(C.CACHE), "canonical_source": PROV.CANONICAL_BASE,
        "qa_gate_pass": gate, "amendment": "v2: 源缺登记口径修正 + 2 对确认重试",
        "status": "QA_PASS_READY_FOR_REJUDGE" if gate else "QA_FAIL_STOP"}, ensure_ascii=False, indent=1))
    print(f"QA v2: collected={n_collected} valid={res_qa['qa_valid']} "
          f"valid_with_gap={res_qa['qa_valid_with_gap']} hard_invalid={len(hard_invalid)} "
          f"gate={'PASS' if gate else 'FAIL'}")
    print(f"confirmations: {[(c['code'], c['day'], c['identical_to_collected']) for c in confirmations]}")
    print(f"network total={snap2['network_requests']} dir={snap2['dir_bytes_after']} "
          f"budget_ok={snap2['budget_ok']}")
    if not gate:
        raise SystemExit("QA 门未过 -> 停止, 不做 247 重判")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
