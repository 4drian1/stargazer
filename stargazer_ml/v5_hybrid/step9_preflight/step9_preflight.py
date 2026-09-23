"""step9_preflight — Step 9 Preflight Gate (9.0) 编排器。

Gate 0: Route / Label Contract Freeze
Gate 1: DATA_AVAILABILITY_REPORT
Gate 2: Machine-readable feature_registry
Gate 3: L0 + GREEN-WATCH-v1 Freeze
Gate 4: Dry Replay (20 交易日端到端泄漏核验)

STEP9_READY = Gate0 AND Gate1 AND Gate2 AND Gate3 AND (Gate4 == PASS)
任一 FAIL 不允许开始 A0-A8 / S0 / Timing / R0 训练。

运行: python step9_preflight.py
"""
from __future__ import annotations
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

ROADMAP = Path(r"C:\Users\22107\Desktop\交易\开发路线.txt")
ROUTE_MARK = "Step 9 Label Contract Alignment"


def gate0() -> dict:
    label_contract = (_HERE / "label_contract_v1.yaml").exists()
    route_ok = ROADMAP.exists() and ROUTE_MARK in ROADMAP.read_text(encoding="utf-8")
    # build_step9_labels 可导入即视为 Gate0 代码落地
    try:
        import build_step9_labels  # noqa
        code_ok = True
    except Exception as e:  # noqa
        code_ok = False
        sys.stderr.write(f"build_step9_labels import error: {e}\n")
    ok = label_contract and route_ok and code_ok
    return {"gate0": ok, "label_contract": label_contract,
            "route_record": route_ok, "label_code": code_ok}


def gate1() -> dict:
    import data_availability_audit as m
    rep = m.run_gate1()
    v = rep["verdict"]
    return {
        "gate1a": v["gate1a_core_data"],
        "gate1b": v["gate1b_context_pit_sector"],
        "gate1c": v["gate1c_optional_microstructure"],
        "full_step9": v["full_step9"],
        "step9a_core": v["step9a_core"],
        "step9b_sector_leader": v["step9b_sector_leader"],
        "hard_fail": [d["dataset_name"] for d in rep["datasets"]
                      if d["status"] == "FAIL" and d["required"]],
    }


def gate2() -> dict:
    import feature_registry as m
    res = m.run_gate2_tests()
    return {"gate2": res["gate2"], "tests": res["tests"]}


def gate3() -> dict:
    import freeze as m
    l0 = _HERE / "L0_freeze.yaml"
    gw = _HERE / "green_watch_v1_frozen.yaml"
    if not (l0.exists() and gw.exists()):
        return {"gate3": False, "reason": "freeze manifests 缺失, 先运行 freeze.py"}
    ok = m.verify()
    return {"gate3": ok}


def gate4() -> dict:
    import dry_replay as m
    r = m.run_dry_replay()
    return {"gate4": r["gate4"], "reason": r["reason"]}


def main() -> int:
    g0 = gate0()
    g1 = gate1()
    g2 = gate2()
    g3 = gate3()
    g4 = gate4()

    # Full Step 9: 需 Gate1B(PIT Sector) PASS + 其余闸门
    full_ready = (g0["gate0"] and g2["gate2"] and g3["gate3"]
                  and g4["gate4"] == "PASS" and g1["gate1b"] == "PASS")
    if g4["gate4"] == "CONDITIONAL":
        full_ready = False

    # Step 9A Core: 仅需 Gate1A(核心数据) + Gate2/3/4 (不要求 PIT Sector)
    step9a_ready = (g0["gate0"] and g2["gate2"] and g3["gate3"]
                    and g4["gate4"] == "PASS" and g1["gate1a"] in ("PASS", "CONDITIONAL_PASS"))

    print("=" * 60)
    print("Step 9 Preflight Gate (9.0) — v2 staged")
    print("=" * 60)
    print(f"Gate0 Route/Label Freeze : {'PASS' if g0['gate0'] else 'FAIL'}  {g0}")
    print(f"Gate1A CORE DATA         : {g1['gate1a']}")
    print(f"Gate1B PIT SECTOR        : {g1['gate1b']}")
    print(f"Gate1C VP MICROSTRUCTURE : {g1['gate1c']}")
    print(f"Gate2 Feature Registry   : {'PASS' if g2['gate2'] else 'FAIL'}  {g2['tests']}")
    print(f"Gate3 L0/GW Freeze       : {'PASS' if g3['gate3'] else 'FAIL'}  {g3.get('reason','')}")
    print(f"Gate4 Dry Replay         : {g4['gate4']}  {g4['reason']}")
    print("-" * 60)
    print(f"STEP9A_READY (A0-A4)     : {step9a_ready}   [{g1['step9a_core']}]")
    print(f"STEP9B (A5-A8)           : {g1['step9b_sector_leader']}")
    print(f"STEP9_FULL_READY         : {full_ready}   [{g1['full_step9']}]")
    if step9a_ready:
        print("=> Step 9A: 可进入 Gate2->Gate3->Gate4-> A0-A4 重训 (补完 qa_core_data 后)")
    else:
        print("=> Step 9A 不允许启动; 先关闭 Gate1A/2/3/4 FAIL")
    if not full_ready:
        print("=> Step 9B (A5-A8) 禁止: 等待 Gate 1B PIT Sector 解锁")
    return 0 if step9a_ready else 1


if __name__ == "__main__":
    sys.exit(main())
