"""run_component_audit — 冻结前组件契约测试 + 源文件 SHA256 (用户 2026-09-18 #5)。

用户裁定:
  - 允许对 build_pool.py / market_proxy_v1.py 做「审计范围内」的改动 (Gate 分解 /
    PriorStrength / StructureIntact / dip_stage / 质量 / 审计字段), 禁止改
    BOTTOM / 25-100亿 / ATR_ref / tier 阈值 / Label / Execution;
  - 不得回滚代码去迎合旧 manifest; 正确做法 = 重跑契约测试 + 刷新哈希。

本脚本:
  1) 子进程运行各组件自测 / 契约测试, 收集 stdout 与退出码;
  2) 计算全部冻结组件 (freeze.COMPONENT_FILES) 的 SHA256;
  3) 覆盖写 audit_component_tests.log (供 build_gate3_evidence.py 引用其 SHA256)。

运行: python run_component_audit.py
"""
from __future__ import annotations
import hashlib
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

TESTS = [
    ("market_proxy_v1", ["market_proxy_v1.py"]),
    ("execution_contract", ["execution_contract_test.py"]),
    ("zone_state_contract", ["zone_state_contract_test.py"]),
    ("gate4_harness_synthetic", ["gate4_harness.py"]),
    ("data_regimes_smoke",
     ["-c", "import data_regimes as dr; assert dr.FLOAT_MV_DATA_REGIME_CHANGE==20260701; "
            "print('FLOAT_MV_DATA_REGIME_CHANGE=', dr.FLOAT_MV_DATA_REGIME_CHANGE); "
            "print(dr.forced_slices())"]),
    ("latr_num_or_none_regression",
     ["-c", "import latr_builder as LB; "
            "assert LB._num_or_none(None) is None; "
            "assert LB._num_or_none(float('nan')) is None; "
            "assert LB._num_or_none(float('inf')) is None; "
            "assert LB._num_or_none(0.123456789) == 0.12346; "
            "print('num_or_none regression OK')"]),
]


def _sha(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def main():
    import freeze as F
    lines = [f"# component contract audit @ {datetime.now(timezone.utc).isoformat(timespec='seconds')}",
             f"# python = {sys.executable}", ""]
    all_ok = True
    for name, argv in TESTS:
        env = {**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1"}
        r = subprocess.run([sys.executable, *argv], cwd=str(_HERE), capture_output=True,
                           text=True, encoding="utf-8", errors="replace", env=env)
        ok = r.returncode == 0
        all_ok = all_ok and ok
        lines.append(f"===== [{name}] exit={r.returncode} {'PASS' if ok else 'FAIL'} =====")
        out = (r.stdout or "").strip()
        err = (r.stderr or "").strip()
        lines.append(out[-4000:] if out else "(no stdout)")
        if err:
            lines.append("--- stderr ---")
            lines.append(err[-2000:])
        lines.append("")
    lines.append("===== COMPONENT SHA256 (freeze.COMPONENT_FILES) =====")
    for k, p in sorted(F.COMPONENT_FILES.items()):
        lines.append(f"{_sha(p) if p.exists() else '<MISSING>':64s}  {k:28s} {p}")
    lines.append("===== TEST SCRIPT SHA256 =====")
    for p in (_HERE / "run_component_audit.py", _HERE / "attribute_typeerror.py",
              _HERE / "gate4_run_real.py", _HERE / "gate4_add_scenario_cases.py",
              _HERE / "trace_04b_scenario_scan.py", _HERE / "gate4_harness.py"):
        if p.exists():
            lines.append(f"{_sha(p):64s}  {p.name}")
    lines.append("")
    lines.append(f"# OVERALL: {'ALL_TESTS_PASS' if all_ok else 'HAS_FAILURE'}")
    (_HERE / "audit_component_tests.log").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[audit] wrote audit_component_tests.log; ALL_TESTS_PASS={all_ok}")
    if not all_ok:
        sys.exit(2)


if __name__ == "__main__":
    main()
