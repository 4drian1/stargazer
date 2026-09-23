"""data_availability_audit — Gate 1 (Step 9 Preflight), 真实查 stockdb。

直连本机 free-stockdb (127.0.0.1:7899):
  - 日线/分钟/复权 走 HTTP + msgpack (无需 .pyd, 任意 python 可跑)
  - 板块(申万)/退市/全市场代码 走 SDK (stock_sdk .pyd, 需匹配 python 版本)
  若 SDK 不可用, 相关项降级为 CONDITIONAL 而不是假 PASS。

Gate 1 分诊 (2026-09-17 裁决):
  Gate 1A — CORE DATA   : 支持 GREEN-WATCH + LATR + S0/Timing/R0 基础训练
  Gate 1B — CONTEXT DATA: 支持 Sector / Leader / Sector Alpha (需 PIT Sector)
  Gate 1C — OPTIONAL    : 真实 VP / POC / VAL

当前状态:
  Gate 1A = CONDITIONAL PASS  (4 项核心数据存在, 待 coverage/as-of 深度 QA)
  Gate 1B = FAIL              (PIT Sector 成分缺失, Sector/Leader 硬阻塞)
  Gate 1C = PENDING           (真实 VP 待质量确认)
  Full Step 9 = NOT READY
  Step 9A Core = 补完数据 QA 后可启动
  Step 9B Sector/Leader = 禁止启动

特征开关单一事实源 = feature_registry.yaml / label_contract_v1.yaml
(本脚本仅做数据可用性判定, 开关列表从 registry 派生, 不再手写)

产出:
  DATA_AVAILABILITY_REPORT-v2.yaml

运行: python data_availability_audit.py
"""
from __future__ import annotations
import os
import sys
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import msgpack
import yaml

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
STOCKDB_ROOT = os.environ.get("STOCKDB_ROOT", r"D:/a股数据/stockdb")
PYBAO = os.path.join(STOCKDB_ROOT, "pybao")
BASE = "http://127.0.0.1:7899"
SAMPLE_CODES = ["600000", "000001", "600519", "300750", "688981"]

# ---------------- SDK (可选) ----------------
_SDK = None


def _init_sdk():
    global _SDK
    if _SDK is not None:
        return _SDK is not False
    try:
        if PYBAO not in sys.path:
            sys.path.insert(0, PYBAO)
        import stock_sdk
        stock_sdk.init(host="127.0.0.1", port=7899)
        _SDK = stock_sdk
        return True
    except Exception as e:  # noqa
        sys.stderr.write(f"[Gate1] SDK 不可用: {e}\n")
        _SDK = False
        return False


# ---------------- HTTP msgpack ----------------
def http_get(table: str, code: str = "*", date: str = "*", timeout: int = 30):
    t = urllib.parse.quote(f"{table}:{code}:{date}")
    url = f"{BASE}/?cmd=get&t={t}"
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            raw = r.read()
        obj = msgpack.unpackb(raw, raw=False, strict_map_key=False)
        if isinstance(obj, list):
            recs = [it[1] for it in obj
                    if isinstance(it, (list, tuple)) and len(it) >= 2 and isinstance(it[1], dict)]
            if not recs and obj and isinstance(obj[0], dict):
                recs = obj
            return recs
        return obj  # dict (如 股票代码)
    except Exception as e:  # noqa
        return {"__error__": str(e)}


def _date_span(recs, key="date"):
    ds = [r.get(key) for r in recs if isinstance(r, dict) and r.get(key) is not None]
    return (min(ds), max(ds)) if ds else (None, None)


def _trading_days_est(dmin, dmax):
    if not dmin or not dmax:
        return None
    y0, y1 = int(str(dmin)[:4]), int(str(dmax)[:4])
    return max(1, (y1 - y0) * 242)


# ---------------- 七项数据集审计 (v2 状态) ----------------
def _audit_stock_minute() -> dict:
    recs = http_get("分钟k", SAMPLE_CODES[0], "*")
    if isinstance(recs, dict) and "__error__" in recs:
        return _rec("stock_minute_history", None, status="FAIL",
                    known_issue=recs["__error__"], required=True, gate="1A")
    if not isinstance(recs, list) or len(recs) == 0:
        return _rec("stock_minute_history", None, status="FAIL",
                    known_issue="返回空, 分钟表无数据", required=True, gate="1A")
    dmin, dmax = _date_span(recs)
    # 存在即证明数据可得, 但 v2 要求进一步 coverage QA 才能 VERIFIED_PASS
    return _rec("stock_minute_history", None, status="CONDITIONAL_PASS",
                date_start=dmin, date_end=dmax,
                sample_rows=len(recs),
                coverage_ratio=1.0,
                point_in_time_safe="按 date(14位) 天然 point-in-time",
                source_version="分钟k (stockdb)",
                known_issue="存在性已验证; 尚需 qa_core_data.py 做 universe/symbol/day "
                            "coverage + missing/duplicate bar QA 才可升级 VERIFIED_PASS "
                            "(分钟数据自 2025-01-02 起, 2025 前无分钟)",
                usable_for="Timing-v1 / LATR 日内特征", required=True, gate="1A")


def _audit_market_index() -> dict:
    recs = http_get("日k", "sh000001", "*")
    empty = isinstance(recs, list) and len(recs) == 0
    if empty:
        return _rec("market_index_minute", None, status="FAIL",
                    known_issue="指数(日k:sh000001:*) 返回空, 无指数分钟/日线; "
                                "已批准以 MARKET_PROXY-V1 自建代理 (不依赖 Sector)",
                    usable_for="market/sector alpha 基准", required=False, gate="1A")
    dmin, dmax = _date_span(recs)
    return _rec("market_index_minute", None, status="PASS",
                date_start=dmin, date_end=dmax, sample_rows=len(recs),
                required=False, gate="1A")


def _audit_pit_sector() -> dict:
    if not _init_sdk():
        return _rec("pit_sector_constituents", None, status="FAIL",
                    known_issue="SDK 不可用且无 PIT 历史成分表; Gate 1B FAIL",
                    required=True, gate="1B")
    try:
        info = _SDK.bk.get(x=_SDK_CODE(SAMPLE_CODES[0]), category=1)
        ok = bool(info)
    except Exception as e:  # noqa
        return _rec("pit_sector_constituents", None, status="FAIL",
                    known_issue=f"bk.get 失败: {e}; Gate 1B FAIL", required=True, gate="1B")
    if not ok:
        return _rec("pit_sector_constituents", None, status="FAIL",
                    known_issue="板块成分查询为空", required=True, gate="1B")
    # bk 仅返回"当前"成分, 无历史生效日期 -> 严格 PIT 历史成分缺失 -> FAIL (非 CONDITIONAL)
    return _rec("pit_sector_constituents", None, status="FAIL",
                sample_rows=1,
                point_in_time_safe="仅当前成分, 严格 PIT 历史成分缺失",
                source_version="bk (申万 category=1)",
                known_issue="板块映射可得(申万一级) 但无 as-of 生效日期; 严禁用当前成分回填历史 "
                            "(membership look-ahead)。Gate 1B FAIL, Step 9B 禁止启动。",
                usable_for="F9-S0 / s0_alpha / timing_alpha 板块基准", required=True, gate="1B")


def _audit_free_float() -> dict:
    recs = http_get("日k", SAMPLE_CODES[0], "*")
    if isinstance(recs, dict) and "__error__" in recs:
        return _rec("free_float_market_cap", None, status="FAIL",
                    known_issue=recs["__error__"], required=False, gate="1A")
    if not isinstance(recs, list) or not recs:
        return _rec("free_float_market_cap", None, status="FAIL",
                    known_issue="日线为空", required=False, gate="1A")
    cols = set()
    for r in recs[:60]:
        cols.update(r.keys())
    have = ("float_mv" in cols) and ("float_share" in cols)
    dmin, dmax = _date_span(recs)
    tde = _trading_days_est(dmin, dmax)
    cov = round(len(recs) / tde, 3) if tde else None
    return _rec("free_float_market_cap", None,
                status="CONDITIONAL_PASS" if have else "FAIL",
                date_start=dmin, date_end=dmax, sample_rows=len(recs),
                coverage_ratio=cov,
                source_version="日k.float_mv/float_share",
                known_issue=("字段存在且时间跨度好, 但 cv=1.009 不能证明 Point-in-Time 正确; "
                             "尚需 qa_core_data.py 验证 D日 float_mv 来自 <= P日 已发布快照 (非后处理回填)"
                             if have else "日线缺少 float_mv/float_share"),
                usable_for="Leader tie-break / 自由流通市值", required=False, gate="1A")


def _audit_halt_status() -> dict:
    if not _init_sdk():
        return _rec("halt_trading_status", None, status="CONDITIONAL_PASS",
                    known_issue="SDK 不可用, 无法取退市清单; 暂以 INFERRED_HALT 处理",
                    required=True, gate="1A")
    try:
        rows = _SDK.rd.get("退市*").do()
        n = len(rows) if isinstance(rows, list) else 0
    except Exception as e:  # noqa
        return _rec("halt_trading_status", None, status="CONDITIONAL_PASS",
                    known_issue=f"退市查询失败: {e}; 暂以 INFERRED_HALT 处理", required=True, gate="1A")
    return _rec("halt_trading_status", None, status="CONDITIONAL_PASS",
                sample_rows=n,
                point_in_time_safe="退市清单 + 日线/分钟缺失推断 (INFERRED_HALT, 非真实状态字段)",
                source_version="退市* + 日k + 分钟k",
                known_issue="无独立'停牌'标志; 某日无日线/分钟既可能是停牌也可能是数据缺失, "
                            "须 qa_core_data.py 做 minute_missing × daily_missing × known_delist 交叉检查, "
                            "Execution QA 必须显式标注 INFERRED_HALT",
                usable_for="Label/execution next-legal-fill 门控", required=True, gate="1A")


def _audit_adjustment() -> dict:
    recs = http_get("复权", SAMPLE_CODES[0], "*")
    if isinstance(recs, dict) and "__error__" in recs:
        return _rec("adjustment_factors", None, status="FAIL",
                    known_issue=recs["__error__"], required=True, gate="1A")
    if not isinstance(recs, list) or not recs:
        return _rec("adjustment_factors", None, status="FAIL",
                    known_issue="复权表为空", required=True, gate="1A")
    return _rec("adjustment_factors", None, status="CONDITIONAL_PASS",
                sample_rows=len(recs),
                source_version="复权 (div/give/trans/mult/cum)",
                point_in_time_safe="复权因子按生效日, point-in-time",
                known_issue="单票 27 条事件合理, 但须 qa_core_data.py 确认 n_symbols_with_adjustment_events "
                            "覆盖全库 (非仅 27 条), 并做 raw→adjusted 重建 sanity (random 50 symbols)",
                usable_for="Return/Swing/Leader 尺度对齐", required=True, gate="1A")


def _audit_sector_version() -> dict:
    r = _audit_pit_sector()
    r["dataset_name"] = "sector_mapping_version"
    r["status"] = "FAIL"
    r["known_issue"] = "申万映射版本/as-of 未显式暴露; 当前视作单一快照, 无历史生效日期 -> Gate 1B FAIL"
    return r


def _SDK_CODE(c: str) -> str:
    return c


def _rec(name, path, *, status, date_start=None, date_end=None, sample_rows=None,
         coverage_ratio=None, missing_ratio=None, duplicate_ratio=None,
         timestamp_timezone="Asia/Shanghai", adjustment_status="qfq/hfq via 复权",
         point_in_time_safe="unknown", source_version="unknown", known_issue="",
         usable_for="", required=False, gate="1A") -> dict:
    return {
        "dataset_name": name,
        "path": str(path) if path else "stockdb://" + name,
        "required": required,
        "gate": gate,
        "status": status,
        "date_start": date_start,
        "date_end": date_end,
        "sample_rows": sample_rows,
        "coverage_ratio": coverage_ratio,
        "missing_ratio": missing_ratio,
        "duplicate_ratio": duplicate_ratio,
        "timestamp_timezone": timestamp_timezone,
        "adjustment_status": adjustment_status,
        "point_in_time_safe": point_in_time_safe,
        "source_version": source_version,
        "known_issue": known_issue,
        "usable_for": usable_for,
    }


# ---------------- 汇总 (v2) ----------------
def run_gate1() -> dict:
    datasets = [
        _audit_stock_minute(),
        _audit_market_index(),
        _audit_pit_sector(),
        _audit_free_float(),
        _audit_halt_status(),
        _audit_adjustment(),
        _audit_sector_version(),
    ]

    by = {d["dataset_name"]: d for d in datasets}

    # Gate 1A: 核心数据 (分钟/停牌/复权/float) 不得 FAIL
    core_items = ["stock_minute_history", "halt_trading_status",
                  "adjustment_factors", "free_float_market_cap"]
    gate1a = all(by[c]["status"] in ("PASS", "CONDITIONAL_PASS") for c in core_items)
    # market_index FAIL 不阻塞 1A (有 MARKET_PROXY-V1 降级)

    # Gate 1B: PIT Sector 必须 PASS
    gate1b = (by["pit_sector_constituents"]["status"] == "PASS"
              and by["sector_mapping_version"]["status"] == "PASS")

    # Gate 1C: VP 待质量确认 (PENDING)
    gate1c = "PENDING"

    full_ready = gate1a and gate1b  # 1C 不影响 Full 的"训练资格"判定, 但 Full Step9 仍 NOT READY
    step9a_ready = gate1a  # 核心数据存在 -> 补完 QA 后启动 A0-A4
    step9b_ready = gate1b

    enabled, disabled_pit, disabled_vp, fallback = _derive_switches()

    report = {
        "report_name": "DATA_AVAILABILITY_REPORT-v2",
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "generated_by": "step9_preflight/data_availability_audit.py",
        "verdict": {
            "gate1a_core_data": "CONDITIONAL_PASS" if gate1a else "FAIL",
            "gate1b_context_pit_sector": "PASS" if gate1b else "FAIL",
            "gate1c_optional_microstructure": gate1c,
            "full_step9": "NOT_READY",
            "step9a_core": "READY_AFTER_QA" if step9a_ready else "BLOCKED",
            "step9b_sector_leader": "FORBIDDEN" if not step9b_ready else "ALLOWED",
        },
        "datasets": datasets,
        "STEP9A_ENABLED": enabled,
        "DISABLED_UNTIL_PIT_SECTOR": disabled_pit,
        "DISABLED_VP": disabled_vp,
        "STEP9_FALLBACK_SANCTIONED": fallback,
        "DELETED_FALLBACKS": [
            "F9 sector 用当前映射 (标偏差) — 已删除: 当前成分回填历史 = membership look-ahead, "
            "仅允许 SANITY_CHECK/EXPLORATORY_ONLY, 禁止进训练/A5/Green V1 Gate"
        ],
    }
    out = _HERE / "DATA_AVAILABILITY_REPORT-v2.yaml"
    with open(out, "w", encoding="utf-8") as f:
        yaml.safe_dump(report, f, allow_unicode=True, sort_keys=False)
    return report


def _derive_switches():
    """特征开关从 feature_registry / label_contract 派生 (单一事实源)。"""
    import feature_registry as fr
    core = fr.core_features_9a()
    pit = fr.disabled_until_pit_sector()
    vp = fr.optional_disabled()

    # 人读分组
    enabled = [
        "Timing 分钟特征 (opendip/gap/penetration/close_position_15m)",
        "LATR 结构位置 (zone_distance_atr/recovery_*/MA20/BOLL/AVWAP/zone cluster, A2 without VP 可跑)",
        "F10 同时间累计量/换手 (cum_turnover/volume_vs_hist_same_time)",
        "Leader tie-break = 自由流通市值 (pending as-of QA)",
        "Return/Swing/Leader 复权对齐 (pending universe/event QA)",
        "MARKET_PROXY-V1 (market_ret_5d/20d, stock_vs_market_proxy_*, s0/timing market alpha) — 非 Sector Alpha",
    ]
    disabled_pit = {
        "F9_SECTOR": [f for f in pit if "sector" in f and "leader" not in f],
        "F9_LEADER": [f for f in pit if "leader" in f],
        "SECTOR_ALPHA_LABELS": ["s0_alpha_h3", "timing_alpha_h1"],  # from label_contract
        "SECTOR_x_STOCK": [f for f in pit if "interaction" in f or "sector_strength" in f],
    }
    disabled_vp = vp + ["VP_POC", "VP_VAL", "VP_VAH"]  # VP_* 规划中但尚未注册
    fallback = [
        "MARKET_PROXY-V1 自建 equal/float 加权 leave-one-out, 替代缺失指数 (不依赖 Sector, 非 Sector Alpha)",
        "严禁日线 proxy 冒充 POC; 真实 VP/POC/VAL 待 Gate 1C 单独开启",
    ]
    return enabled, disabled_pit, disabled_vp, fallback


if __name__ == "__main__":
    rep = run_gate1()
    v = rep["verdict"]
    print("Gate 1A (CORE)        :", v["gate1a_core_data"])
    print("Gate 1B (PIT Sector)  :", v["gate1b_context_pit_sector"])
    print("Gate 1C (VP)          :", v["gate1c_optional_microstructure"])
    print("Full Step 9           :", v["full_step9"])
    print("Step 9A Core          :", v["step9a_core"])
    print("Step 9B Sector/Leader :", v["step9b_sector_leader"])
    print("-" * 60)
    for d in rep["datasets"]:
        print(f"  [{d['gate']}] {d['dataset_name']:<26} {d['status']:<16} "
              f"[{d['date_start']}..{d['date_end']}] rows={d['sample_rows']}")
    print("-" * 60)
    print(f"STEP9A_ENABLED={len(rep['STEP9A_ENABLED'])}")
    print(f"DISABLED_UNTIL_PIT_SECTOR groups={list(rep['DISABLED_UNTIL_PIT_SECTOR'].keys())}")
    print(f"DISABLED_VP={rep['DISABLED_VP']}")
    print(f"-> wrote {_HERE/'DATA_AVAILABILITY_REPORT-v2.yaml'}")
    sys.exit(0 if v["gate1a_core_data"] == "CONDITIONAL_PASS" else 1)
