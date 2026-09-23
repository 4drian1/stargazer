"""green_watch 每日池 → 计划草案落库（draft_gen.py，规则层 v1）。

读取 stargazer_ml/v5_hybrid/green_watch_v0/pool_latest.json（green_watch 模块生成，
schema 见该目录），把 BOTTOM_SETUP / INTRADAY_DIP_SETUP 候选翻译成 draft.json
三条件口径，写入 data/drafts/draft_{next_trading_date}.json，并 subprocess 调
`python -m cli create_draft <path>` 落库（cwd=本脚本所在目录），打印 session_id。

选取口径：
    BOTTOM_SETUP      tier ∈ {WATCH, HIGH_PRIORITY_WATCH}，按 score 降序取前 N=5
    INTRADAY_DIP_SETUP 按 score 降序取前 M=3

幂等与节奏：create_draft 的 upsert 锚是 UNIQUE(trading_date, session_tag)，重复
运行不产生重复 session/候选；**每日只跑一次**——当日已 freeze 后再重跑只会原样
返回既有 version 1；若当日确需变更，人工用 revise 通道追加 version 2+（既有行为，
本脚本不自动追加版本）。池文件不存在时报清晰错误退出。
"""
import json
import re
import subprocess
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
POOL_PATH = Path(r"D:/stargazer/stargazer_ml/v5_hybrid/green_watch_v0/pool_latest.json")
DRAFT_DIR = BASE_DIR / "data" / "drafts"
SESSION_TAG = "green_watch_daily"

N_BOTTOM = 5          # BOTTOM_SETUP 取前 N
M_DIP = 3             # INTRADAY_DIP_SETUP 取前 M
BOTTOM_TIERS = {"WATCH", "HIGH_PRIORITY_WATCH"}


def _load_pool(path=POOL_PATH):
    if not path.exists():
        raise SystemExit(
            f"池文件不存在: {path}\n"
            "（green_watch 模块尚未生成 pool_latest.json，请先跑池构建流程）")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise SystemExit(f"池文件 JSON 解析失败: {path}: {e}")


def _pick_candidates(pool):
    """按 setup/tier/score 选取候选，返回 (bottom_list, dip_list)（均已按 score 降序）。"""
    cands = pool.get("candidates") or []
    bottom = [c for c in cands
              if c.get("setup") == "BOTTOM_SETUP"
              and c.get("tier") in BOTTOM_TIERS]
    dips = [c for c in cands if c.get("setup") == "INTRADAY_DIP_SETUP"]

    def _score(c):
        try:
            return float(c.get("score"))
        except (TypeError, ValueError):
            return float("-inf")

    bottom.sort(key=_score, reverse=True)
    dips.sort(key=_score, reverse=True)
    return bottom[:N_BOTTOM], dips[:M_DIP]


def build_draft(pool):
    """池 JSON → draft.json dict（trading_date=next_trading_date）。"""
    next_day = str(pool.get("next_trading_date") or "").strip()
    if not re.fullmatch(r"\d{8}", next_day):
        raise SystemExit(f"pool_latest.json 的 next_trading_date 非法: {next_day!r}")
    bottom, dips = _pick_candidates(pool)
    if not bottom and not dips:
        raise SystemExit("池中没有符合口径的候选（BOTTOM_SETUP tier∈WATCH/"
                         "HIGH_PRIORITY_WATCH 或 INTRADAY_DIP_SETUP）")

    out, dialogue_lines = [], [
        f"【green_watch_daily 自动草案】next_trading_date={next_day}，"
        f"rule_version={pool.get('rule_version')}",
        f"入选 {len(bottom) + len(dips)} 只"
        f"（BOTTOM_SETUP {len(bottom)} / INTRADAY_DIP_SETUP {len(dips)}）",
    ]
    order = 0
    for c in bottom:
        order += 1
        trigger_price = float(c["trigger_price"])
        pre_close = float(c["pre_close"])
        out.append({
            "sort_order": order,
            "code": str(c["code"]).zfill(6),
            "name": c.get("name") or "",
            "direction": "BUY",
            "trigger": {"desc": "回踩触发价=昨收−0.25×ATR_ref（v1 过渡口径）",
                        "dsl": f"price <= {trigger_price:.2f}"},
            "invalidate": {"desc": "未回踩且翻红，或尾盘，当日失效",
                           "dsl": f"price >= {pre_close:.2f} or time >= 14:30"},
            "observe": {"desc": "跌幅进入监控区", "dsl": "pct_chg <= -2.0"},
        })
    for c in dips:
        order += 1
        pre_close = float(c["pre_close"])
        out.append({
            "sort_order": order,
            "code": str(c["code"]).zfill(6),
            "name": c.get("name") or "",
            "direction": "BUY",
            "trigger": {"desc": "v1 不出 BUY trigger，仅观察"
                                "（承接修复特征组未实现）",
                        "dsl": None},
            "invalidate": {"desc": "未回踩且翻红，或尾盘，当日失效",
                           "dsl": f"price >= {pre_close:.2f} or time >= 14:30"},
            "observe": {"desc": "跌幅进入监控区", "dsl": "pct_chg <= -2.0"},
        })
    for c in bottom + dips:
        dialogue_lines.append(
            f"- {str(c['code']).zfill(6)} {c.get('name') or ''} "
            f"rank={c.get('rank')} setup={c.get('setup')} "
            f"score={c.get('score')} tier={c.get('tier')}")
        reasons = (c.get("selection_reasons") or "").strip()
        risks = (c.get("risk_reasons") or "").strip()
        if reasons:
            dialogue_lines.append(f"  入选理由: {reasons}")
        if risks:
            dialogue_lines.append(f"  风险: {risks}")
    return {
        "trading_date": next_day,
        "session_tag": SESSION_TAG,
        "candidates": out,
        "dialogues": [{"message_ref": "auto", "speaker": "AGENT",
                       "content": "\n".join(dialogue_lines)}],
    }


def main(argv=None):
    pool = _load_pool()
    draft = build_draft(pool)
    DRAFT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = DRAFT_DIR / f"draft_{draft['trading_date']}.json"
    out_path.write_text(json.dumps(draft, ensure_ascii=False, indent=2),
                        encoding="utf-8")
    print(f"draft 已写出: {out_path}（候选 {len(draft['candidates'])} 条）")
    import os
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    proc = subprocess.run(
        [sys.executable, "-m", "cli", "create_draft", str(out_path)],
        cwd=str(BASE_DIR), capture_output=True, text=True, encoding="utf-8",
        env=env)
    sys.stdout.write(proc.stdout)
    if proc.returncode != 0:
        sys.stderr.write(proc.stderr)
        raise SystemExit(f"create_draft 落库失败 rc={proc.returncode}")
    m = re.search(r"session_id=(\d+)", proc.stdout)
    if m:
        print(f"落库成功 session_id={m.group(1)}（trading_date={draft['trading_date']} "
              f"tag={SESSION_TAG}）——每日只跑一次，已 freeze 后请勿重跑")


if __name__ == "__main__":
    main()
