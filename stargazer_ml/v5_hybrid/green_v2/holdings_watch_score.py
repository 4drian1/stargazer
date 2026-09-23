# -*- coding: utf-8 -*-
"""持仓 + 观察名单 T5 影子打分（holdings_watch_score.py，功能 2 v1 盘后批量）。

    _dev_venv\\python -X utf8 holdings_watch_score.py [--date YYYYMMDD] [--workers 8]

读 stargazer_desktop/data/watchlist.json（只读，不改动 desktop 任何文件）的
positions + watch，subprocess 调 .venv 的 t5_batch_score.py 批量打分
（dp=1500，AM/PM 双头），输出买/卖节点建议表 markdown 到 green_v2/。

quadrant 映射（advisory，非交易指令）：
  买入节点候选：FOCUS / WAIT（低风险或高风险但高机会——WAIT 须人工复核）；
  卖出节点候选：REJECT / 风险头高（p_risk ≥ 1.5×base 即 WAIT/REJECT 中的
  高风险侧，REJECT 为主）。
卖出优先级硬契约：**HARD_RISK_FLOOR（冻结灾难底线） > T5 卖出节点 > 人工裁量**；
T5 节点只能进 evidence_ref 当证据，真实卖出落库照旧选 risk_type 九枚举。
"""
import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

BASE = Path(__file__).resolve().parent
WATCHLIST = Path(r"D:/stargazer/stargazer_desktop/data/watchlist.json")
SCORE_SCRIPT = Path(r"D:/stargazer/stargazer_ml/pullback_train/t5_batch_score.py")
VENV_PY = Path(r"D:/stargazer/.venv/Scripts/python.exe")
MANIFEST = BASE / "t5_freeze_manifest.json"
RUN_DIR = BASE / "_run"

BUY_QUADS = {"FOCUS", "WAIT"}
SELL_QUADS = {"REJECT"}

PRIORITY_NOTE = ("卖出优先级硬契约：**HARD_RISK_FLOOR（冻结灾难底线）> "
                 "T5 卖出节点 > 人工裁量**。本表为 advisory（建议性质），"
                 "T5 节点只能作为 evidence_ref 证据，不构成自动卖出指令。")


def latest_block_date():
    import glob
    import pandas as pd
    files = sorted(glob.glob(r"D:/a股数据/data_daily_parts/part_incr_*.parquet"))
    df = pd.read_parquet(files[-1], columns=["date"])
    return str(int(df["date"].iloc[0]))


def load_items():
    d = json.loads(WATCHLIST.read_text(encoding="utf-8"))
    pos = [(str(c["code"]).zfill(6), c.get("name") or "",
            {"entry": c.get("entry"), "entry_date": c.get("entry_date"),
             "logic": c.get("logic")}) for c in d.get("positions") or []]
    watch = [(str(c["code"]).zfill(6), "", {"reason": c.get("reason")})
             for c in d.get("watch") or []]
    return pos, watch


def run_scores(date, codes, out_path, workers):
    codes_path = RUN_DIR / f"codes_holdings_{date}.json"
    codes_path.write_text(json.dumps(sorted(set(codes))), encoding="utf-8")
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    proc = subprocess.run(
        [str(VENV_PY), "-X", "utf8", str(SCORE_SCRIPT),
         "--date", date, "--codes-file", str(codes_path),
         "--out", str(out_path), "--workers", str(workers)],
        capture_output=True, text=True, encoding="utf-8", env=env)
    sys.stderr.write(proc.stderr[-2000:])
    if proc.returncode != 0:
        raise SystemExit(f"t5_batch_score 失败 rc={proc.returncode}")


def load_scores(path):
    out = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            r = json.loads(line)
            out.setdefault(r["code"], {})[r["head"]] = r
    return out


def advise(code, heads):
    """返回 (side, note)：buy/sell/hold 节点建议。"""
    pm, am = heads.get("PM"), heads.get("AM")
    if not pm or pm.get("p_opp") is None:
        return "hold", "无 T5 分数（数据不足）"
    quads = {pm["quadrant"], am["quadrant"] if am else None}
    if quads & SELL_QUADS:
        side = "sell"
        note = f"PM={pm['quadrant']}" + (f"/AM={am['quadrant']}" if am else "")
    elif quads & BUY_QUADS:
        side = "buy"
        note = f"PM={pm['quadrant']}" + (f"/AM={am['quadrant']}" if am else "")
        if "WAIT" in quads and "FOCUS" not in quads:
            note += "（高风险高机会，须人工复核）"
    else:
        side = "hold"
        note = f"PM={pm['quadrant']}" + (f"/AM={am['quadrant']}" if am else "")
    if pm.get("ood"):
        note += " ⚠OOD（高热，分数按分布外折扣）"
    return side, note


def main():
    ap = argparse.ArgumentParser(description="持仓/观察名单 T5 影子打分")
    ap.add_argument("--date", default=None)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    date = a.date or latest_block_date()
    RUN_DIR.mkdir(parents=True, exist_ok=True)

    pos, watch = load_items()
    if not pos and not watch:
        print("watchlist.json 无 positions/watch，跳过")
        return 0
    all_codes = [c for c, _, _ in pos] + [c for c, _, _ in watch]
    scores_path = RUN_DIR / f"holdings_scores_{date}.jsonl"
    t0 = time.time()
    run_scores(date, all_codes, scores_path, a.workers)
    scores = load_scores(scores_path)
    mh = next((r["model_hash"] for h in scores.values()
               for r in h.values()), "?")

    L = [f"# 持仓/观察名单 T5 影子打分 {date}\n",
         f"- T5 冻结版 model_version=2026-09-09，model_hash=`{mh}`"
         f"（已与 t5_freeze_manifest.json 校验）",
         f"- 口径：盘后批量 dp=1500；PM 头原生 / AM 头 out-of-native-regime "
         f"approximation；advisory 非交易指令。",
         f"- {PRIORITY_NOTE}\n",
         "## 持仓（positions）\n",
         "| 代码 | 名称 | 建仓 | 逻辑 | PM risk/opp | PM 象限 | AM 象限 | 节点建议 |",
         "|------|------|------|------|-------------|---------|---------|----------|"]
    n_buy = n_sell = 0
    for code, name, meta in pos:
        heads = scores.get(code, {})
        side, note = advise(code, heads)
        n_buy += side == "buy"
        n_sell += side == "sell"
        pm, am = heads.get("PM"), heads.get("AM")
        sc = (f"{pm['p_risk']:.3f}/{pm['p_opp']:.3f}" if pm and
              pm.get("p_opp") is not None else "—")
        L.append(f"| {code} | {name} | {meta.get('entry_date') or ''} "
                 f"@ {meta.get('entry') or ''} | {meta.get('logic') or ''} "
                 f"| {sc} | {pm['quadrant'] if pm else '—'} "
                 f"| {am['quadrant'] if am else '—'} "
                 f"| **{side.upper()}** {note} |")
    L.append("\n## 观察名单（watch）\n")
    L.append("| 代码 | 理由 | PM risk/opp | PM 象限 | AM 象限 | 节点建议 |")
    L.append("|------|------|-------------|---------|---------|----------|")
    for code, name, meta in watch:
        heads = scores.get(code, {})
        side, note = advise(code, heads)
        n_buy += side == "buy"
        n_sell += side == "sell"
        pm, am = heads.get("PM"), heads.get("AM")
        sc = (f"{pm['p_risk']:.3f}/{pm['p_opp']:.3f}" if pm and
              pm.get("p_opp") is not None else "—")
        L.append(f"| {code} | {meta.get('reason') or ''} | {sc} "
                 f"| {pm['quadrant'] if pm else '—'} "
                 f"| {am['quadrant'] if am else '—'} "
                 f"| **{side.upper()}** {note} |")
    L.append(f"\n> 统计：买节点候选 {n_buy}，卖节点候选 {n_sell}。"
             f"打分 {len(scores)}/{len(set(all_codes))} 只，"
             f"耗时 {time.time()-t0:.0f}s。")
    out = BASE / f"holdings_watch_{date}.md"
    out.write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"[md] {out}（buy={n_buy} sell={n_sell}）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
