"""run_pipeline_after02 — 等 ② 全量 coverage 完成后, 后台串行执行 ④ 与 ⑤.

仅供本轮 QA 使用: 轮询 coverage_02_summary.json 出现后, 依次跑
trace_04_scenarios.py 与 gate4_run_real.py, 日志写 pipeline_045.log。
FINAL Gate3 Freeze 不在此自动执行 (需人工确认 ②/④/⑤ 结论)。
"""
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
PY = r"d:/stargazer/_dev_venv/Scripts/python.exe"
LOG = HERE / "pipeline_045.log"


def log(m):
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(m + "\n")
    print(m, flush=True)


def wait_for(path: Path, name: str, timeout: int = 7200):
    t0 = time.time()
    while not path.exists():
        if time.time() - t0 > timeout:
            log(f"[orch] TIMEOUT waiting for {name} ({path.name})")
            sys.exit(1)
        time.sleep(20)
    time.sleep(15)
    log(f"[orch] {name} ready: {path.name}")


def run(args, name):
    log(f"[orch] START {name} @ {time.strftime('%H:%M:%S')}: {' '.join(args)}")
    r = subprocess.run(args, cwd=str(HERE))
    log(f"[orch] DONE {name} rc={r.returncode} @ {time.strftime('%H:%M:%S')}")
    return r.returncode


def main():
    log(f"[orch] waiting for ② (coverage_02_summary.json) @ {time.strftime('%H:%M:%S')}")
    wait_for(HERE / "coverage_02_summary.json", "②")
    rc4 = run([PY, str(HERE / "trace_04_scenarios.py"), "--budget", "160"], "④")
    rc5 = run([PY, str(HERE / "gate4_run_real.py"), "--budget", "200"], "⑤")
    log(f"[orch] PIPELINE DONE ④ rc={rc4} ⑤ rc={rc5}")


if __name__ == "__main__":
    main()
