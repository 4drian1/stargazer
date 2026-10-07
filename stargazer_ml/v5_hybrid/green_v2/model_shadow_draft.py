# -*- coding: utf-8 -*-
"""T5 影子双名单：model_shadow 名单生成 + 落库 + 双名单 markdown。

    _dev_venv\\python -X utf8 model_shadow_draft.py [--date YYYYMMDD]
        [--trading-date YYYYMMDD] [--top-k 20] [--workers 10]
        [--universe auto|full|degraded] [--scores path.jsonl] [--no-db]

流程：
  1. universe = data_daily_parts 最新日期块（非ST/volume>0/close>1）。
  2. 可行性：50 只探针实测耗时外推全量；>20 分钟则降级（登记 deviation）：
     universe = 池 top300（pool_daily.parquet 按 primary_score）
              ∪ 用户 watchlist/持仓（stargazer_desktop/data/watchlist.json）
              ∪ 规则名单候选（pool_latest.json）。
  3. subprocess 调 .venv 的 t5_batch_score.py 批量打分（dp=1500，AM/PM 双头）。
  4. PM 头（dp=1500 原生头）排序取 top-K → model_shadow 名单；
     subprocess 调 stargazer_plan `python -m cli create_draft`
     （session_tag=model_shadow_daily，trigger dsl=null 纯观察）落库。
  5. 输出双名单 markdown：green_v2/daily_dual_list_<date>.md
     （规则名单带 T5 分数列、模型名单三 caveat 头、重叠高亮）。

冻结纪律：权重 sha256 与 green_v2/t5_freeze_manifest.json 每日校验，
不一致即拒绝输出。T5 输出仅供观察，不进冻结计划、不接监控推送。
"""
import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

BASE = Path(__file__).resolve().parent           # green_v2/
GREEN_V2 = BASE
_REPO_ROOT = BASE.parents[2]                     # 仓库根（stargazer_ml/v5_hybrid/green_v2 → 根）


def _root_path(env: str, rel: str, dev_default: str) -> Path:
    """env 覆盖 → 仓库相对布局 → 本机 dev 缺省（监控机/本机同构部署）。"""
    v = os.environ.get(env)
    if v:
        return Path(v)
    p = _REPO_ROOT / rel
    return p if p.exists() else Path(dev_default)


POOL_JSON = _root_path("STARGAZER_POOL_JSON",
                       "stargazer_ml/v5_hybrid/green_watch_v0/pool_latest.json",
                       r"D:/stargazer/stargazer_ml/v5_hybrid/green_watch_v0/pool_latest.json")
POOL_DAILY = _root_path("STARGAZER_POOL_DAILY",
                        "stargazer_ml/v5_hybrid/green_watch_v0/pool_daily.parquet",
                        r"D:/stargazer/stargazer_ml/v5_hybrid/green_watch_v0/pool_daily.parquet")
WATCHLIST = _root_path("STARGAZER_WATCHLIST",
                       "stargazer_desktop/data/watchlist.json",
                       r"D:/stargazer/stargazer_desktop/data/watchlist.json")
SCORE_SCRIPT = _root_path("STARGAZER_SCORE_SCRIPT",
                          "stargazer_ml/pullback_train/t5_batch_score.py",
                          r"D:/stargazer/stargazer_ml/pullback_train/t5_batch_score.py")
VENV_PY = _root_path("STARGAZER_VENV_PY", ".venv/Scripts/python.exe",
                     r"D:/stargazer/.venv/Scripts/python.exe")
if os.environ.get("STARGAZER_ML_PYTHON") and not os.environ.get("STARGAZER_VENV_PY"):
    VENV_PY = Path(os.environ["STARGAZER_ML_PYTHON"])  # 监控机既有逃生口约定
PLAN_DIR = _root_path("STARGAZER_PLAN_DIR", "stargazer_plan",
                      r"D:/stargazer/stargazer_plan")
DEVENV_PY = _root_path("STARGAZER_DEVENV_PY", "_dev_venv/Scripts/python.exe",
                       r"D:/stargazer/_dev_venv/Scripts/python.exe")
MANIFEST = GREEN_V2 / "t5_freeze_manifest.json"
RUN_DIR = GREEN_V2 / "_run"

SESSION_TAG = "model_shadow_daily"
TOP_K = 20
PROBE_N = 50
BUDGET_SEC = 20 * 60

CAVEATS = (
    "1. **口径偏移**：T5 原生为盘中决策点打分器（AM/PM 双头），本名单为盘后 "
    "dp=1500 批量口径——PM 头为原生，AM 头为 out-of-native-regime "
    "approximation；评估“训练是否符合预期”时计入折扣。",
    "2. **无 edge 声明**：与规则名单并排 ≠ 被验证；模型名单只是另一个待观察"
    "排序器（STEP1 COVERAGE_ONLY 对一切排序器名单适用）。",
    "3. **人工从模型名单挑票买入 = 自由**（人工轨迹），落库 source 标人工；"
    "T5 分数不构成任何买卖理由代码。",
)

QUAD_RANK = {"FOCUS": 0, "WAIT": 1, "WATCH": 2, "REJECT": 3, "UNSCORED": 9}


# ---------------------------------------------------------------- 基础
def _py_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def latest_block_date():
    import glob
    import pandas as pd
    files = sorted(glob.glob(r"D:/a股数据/data_daily_parts/part_incr_*.parquet"))
    if not files:
        raise SystemExit("未找到 data_daily_parts/part_incr_*.parquet")
    df = pd.read_parquet(files[-1], columns=["date"])
    return str(int(df["date"].iloc[0]))


def available_trade_dates():
    """日线 parts 全部已覆盖交易日（全量+增量块，仅读 date 列）。"""
    import glob
    import pandas as pd
    dates = set()
    for pat in ("part_*.parquet", "part_incr_*.parquet"):
        for f in glob.glob(rf"D:/a股数据/data_daily_parts/{pat}"):
            dates.update(str(int(d)) for d in
                         pd.read_parquet(f, columns=["date"])["date"].unique())
    return dates


def build_universe(date):
    """最新日期块 → 股票 universe：非ST/volume>0/close>1，
    剔除基金（5xxxxx/15xxxx/16xxxx/18xxxx——T5 为个股回撤打分器）。"""
    import glob
    import pandas as pd
    files = sorted(glob.glob(r"D:/a股数据/data_daily_parts/part_incr_*.parquet"))
    df = pd.read_parquet(files[-1])
    df = df[(~df["is_st"].fillna(False)) & (df["volume"] > 0) & (df["close"] > 1)]
    codes = df["code"].astype(str).str.zfill(6)
    df = df[codes.str.startswith(("60", "68", "00", "30", "8", "4", "92"))]
    return sorted(df["code"].astype(str).str.zfill(6).unique())


def pool_top_codes(n=300):
    import pandas as pd
    df = pd.read_parquet(POOL_DAILY, columns=["date", "code", "primary_score"])
    dmax = df["date"].max()
    top = df[df["date"] == dmax].dropna(subset=["primary_score"])
    top = top.sort_values("primary_score", ascending=False)["code"]
    return sorted({str(c).zfill(6) for c in top.head(n)})


def watch_codes():
    if not WATCHLIST.is_file():
        return set()
    d = _py_json(WATCHLIST)
    out = {str(c["code"]).zfill(6) for c in d.get("positions") or []}
    out |= {str(c["code"]).zfill(6) for c in d.get("watch") or []}
    return out


def rule_candidates():
    pool = _py_json(POOL_JSON)
    return pool, [(str(c["code"]).zfill(6), c) for c in pool.get("candidates") or []]


def check_freeze():
    pkl = SCORE_SCRIPT.parent / "shadow_models" / "shadow_t5_frozen.pkl"
    h = hashlib.sha256(pkl.read_bytes()).hexdigest()[:12]
    if MANIFEST.is_file():
        want = _py_json(MANIFEST).get("sha256_12")
        if want != h:
            raise SystemExit(
                f"FROZEN_MODEL_HASH_MISMATCH: 实测 {h} vs 清单 {want}；拒绝输出。")
    return h


def run_scores(date, codes, out_path, workers):
    codes_path = RUN_DIR / f"codes_{out_path.stem}.json"
    codes_path.write_text(json.dumps(sorted(set(codes))), encoding="utf-8")
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    t0 = time.time()
    proc = subprocess.run(
        [str(VENV_PY), "-X", "utf8", str(SCORE_SCRIPT),
         "--date", date, "--codes-file", str(codes_path),
         "--out", str(out_path), "--workers", str(workers)],
        capture_output=True, text=True, encoding="utf-8", env=env)
    sys.stderr.write(proc.stderr[-3000:])
    if proc.returncode != 0:
        raise SystemExit(f"t5_batch_score 失败 rc={proc.returncode}: "
                         f"{proc.stdout[-500:]}")
    return time.time() - t0


def load_scores(path):
    """jsonl → {code: {"PM": row, "AM": row}}"""
    out = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        out.setdefault(r["code"], {})[r["head"]] = r
    return out


# ---------------------------------------------------------------- 名单
def rank_model_list(scores, top_k, universe=None):
    """PM 头（dp=1500 原生）排序：象限优先 FOCUS>WAIT>WATCH>REJECT，
    同象限 p_opp 降序、p_risk 升序；剔除 unscored/NaN 与 universe 外代码。"""
    rows = []
    for code, h in scores.items():
        if universe is not None and code not in universe:
            continue
        pm = h.get("PM")
        if not pm or pm.get("p_opp") is None or pm.get("p_risk") is None:
            continue
        rows.append((code, pm, h.get("AM")))
    rows.sort(key=lambda x: (QUAD_RANK.get(x[1]["quadrant"], 9),
                             -x[1]["p_opp"], x[1]["p_risk"]))
    return rows[:top_k]


def _fmt_pm(pm, am=None):
    am_q = f"/{am['quadrant']}" if am else ""
    ood = " ⚠OOD" if pm.get("ood") else ""
    return (f"{pm['p_risk']:.3f} / {pm['p_opp']:.3f} "
            f"**{pm['quadrant']}**{am_q}{ood}")


def write_dual_md(date, trading_date, universe_note, mhash, pool, rule,
                  model_top, scores, md_path):
    model_codes = {c for c, _, _ in model_top}
    rule_rows = []
    for code, c in rule:
        sc = scores.get(code, {})
        t5s = _fmt_pm(sc["PM"], sc.get("AM")) if sc.get("PM") else "—（未打分）"
        rule_rows.append((code, c, t5s, code in model_codes))
    overlap = [code for code, c, t5s, ov in rule_rows if ov]

    L = []
    L.append(f"# T5 影子双名单 {date}（交易日 {trading_date}）\n")
    L.append(f"- T5 冻结版：model_version=2026-09-09，model_hash=`{mhash}`"
             f"（sha256 前 12 位，已与 t5_freeze_manifest.json 校验一致）")
    L.append(f"- 打分口径：盘后批量 dp=1500；**PM 头 = dp1500 原生，"
             f"AM 头 = out-of-native-regime approximation**")
    L.append(f"- universe：{universe_note}")
    L.append(f"- 生成时间：{time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    L.append("## 模型名单三条 caveat（必读）\n")
    L.extend(CAVEATS)
    L.append("")
    L.append(f"## 双名单重叠：{len(overlap)} 只"
             f"（{('、'.join(overlap)) if overlap else '无'}）\n")
    L.append("## 规则层名单（green_watch_daily，正式候选）\n")
    L.append("| # | 代码 | 名称 | setup | tier | 规则分 | T5 (risk/opp 象限 AM) | 重叠 |")
    L.append("|---|------|------|-------|------|--------|------------------------|------|")
    for code, c, t5s, ov in rule_rows:
        L.append(f"| {c.get('rank')} | {code} | {c.get('name') or ''} "
                 f"| {c.get('setup')} | {c.get('tier')} | {c.get('score')} "
                 f"| {t5s} | {'**★**' if ov else ''} |")
    L.append("")
    L.append("## 模型名单（model_shadow_daily，**纯观察**，未验证模型输出）\n")
    L.append("| # | 代码 | PM risk/opp | PM 象限 | AM 象限 | OOD |")
    L.append("|---|------|-------------|---------|---------|-----|")
    for i, (code, pm, am) in enumerate(model_top, 1):
        L.append(f"| {i} | **{code}** | {pm['p_risk']:.3f} / {pm['p_opp']:.3f} "
                 f"| **{pm['quadrant']}** | {am['quadrant'] if am else '—'} "
                 f"| {'⚠' if pm.get('ood') else ''} |")
    L.append("")
    L.append("> 模型名单候选一律 dsl=null 纯观察（trigger 不进监控）；"
             "人工挑票买入落库 source 标人工，T5 分数不构成买卖理由代码。")
    md_path.write_text("\n".join(L) + "\n", encoding="utf-8")
    return overlap


def build_draft(trading_date, model_top, date, mhash, universe_note):
    lines = [
        f"【model_shadow_daily 自动草案】date={date}，"
        f"trading_date={trading_date}，T5 冻结版 model_hash={mhash}",
        f"universe：{universe_note}",
        "纯观察名单：候选 trigger 一律 dsl=null；未验证模型输出，"
        "不进冻结计划、不接监控推送（三 caveat 见 daily_dual_list md）。",
    ]
    cands = []
    for i, (code, pm, am) in enumerate(model_top, 1):
        cands.append({
            "sort_order": i,
            "code": code,
            "name": "",
            "direction": "BUY",
            "trigger": {"desc": f"MODEL_SHADOW 纯观察；T5 PM {pm['quadrant']} "
                                f"(risk={pm['p_risk']:.3f}, opp={pm['p_opp']:.3f})"
                                + (f"；AM {am['quadrant']}" if am else ""),
                        "dsl": None},
            "invalidate": {"desc": "纯观察，无失效条件", "dsl": None},
            "observe": {"desc": "纯观察（未验证模型输出）", "dsl": None},
        })
        lines.append(f"- #{i} {code} PM={pm['quadrant']} "
                     f"risk={pm['p_risk']:.3f} opp={pm['p_opp']:.3f}"
                     + (f" AM={am['quadrant']}" if am else ""))
    return {
        "trading_date": trading_date,
        "session_tag": SESSION_TAG,
        "candidates": cands,
        "dialogues": [{"message_ref": "auto", "speaker": "AGENT",
                       "content": "\n".join(lines)}],
    }


def create_draft_pg(draft):
    out = PLAN_DIR / "data" / "drafts" / \
        f"draft_model_shadow_{draft['trading_date']}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(draft, ensure_ascii=False, indent=2),
                   encoding="utf-8")
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    proc = subprocess.run(
        [str(DEVENV_PY), "-X", "utf8", "-m", "cli", "create_draft", str(out)],
        cwd=str(PLAN_DIR), capture_output=True, text=True,
        encoding="utf-8", env=env)
    sys.stdout.write(proc.stdout)
    if proc.returncode != 0:
        sys.stderr.write(proc.stderr)
        raise SystemExit(f"create_draft 落库失败 rc={proc.returncode}")
    m = re.search(r"session_id=(\d+)", proc.stdout)
    return (int(m.group(1)) if m else None), out


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description="T5 影子双名单生成器")
    ap.add_argument("--date", default=None, help="打分日（默认最新数据块）")
    ap.add_argument("--trading-date", default=None,
                    help="落库交易日（默认 pool_latest.json 的 next_trading_date）")
    ap.add_argument("--top-k", type=int, default=TOP_K)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--universe", choices=("auto", "full", "degraded"),
                    default="auto")
    ap.add_argument("--scores", default=None,
                    help="复用已有打分 jsonl（跳过取数与打分）")
    ap.add_argument("--no-db", action="store_true", help="不落库（只出 md）")
    a = ap.parse_args()

    date = a.date or latest_block_date()
    requested_date = date  # 输出文件名按请求日（桌面端按请求日找文件）；内容注明实际评分日
    avail = available_trade_dates()
    if date not in avail:
        fb = max(avail)
        print(f"[date-guard] {date} 无交易数据（节假日或数据未更新），"
              f"自动改用最近有数据的交易日 {fb}", file=sys.stderr)
        date = fb
    mhash = check_freeze()
    pool, rule = rule_candidates()
    trading_date = a.trading_date or str(pool.get("next_trading_date") or "")
    if not re.fullmatch(r"\d{8}", trading_date):
        raise SystemExit(f"trading_date 非法: {trading_date!r}（可用 "
                         f"--trading-date 覆盖）")
    RUN_DIR.mkdir(parents=True, exist_ok=True)

    universe = build_universe(date)
    wcodes = watch_codes()
    rule_codes = {c for c, _ in rule}
    degraded_set = sorted(set(pool_top_codes(300)) | wcodes | rule_codes)

    universe_note = ""
    if a.scores:
        scores_path = Path(a.scores)
        elapsed = 0.0
        n_scored = len(load_scores(scores_path))
        universe_note = (f"复用打分 {scores_path.name}"
                         f"（universe 股票 {len(universe)} 只，"
                         f"打分文件 {n_scored} 只）")
    else:
        if a.universe == "full":
            final, universe_note = universe, \
                f"全量 {len(universe)} 只（--universe full）"
        elif a.universe == "degraded":
            final, universe_note = degraded_set, (
                f"降级池 {len(degraded_set)} 只（池top300 ∪ watchlist "
                f"{len(wcodes)} ∪ 规则名单 {len(rule_codes)}）"
                f"（--universe degraded）")
        else:
            probe = universe[::max(1, len(universe) // PROBE_N)][:PROBE_N]
            t0 = time.time()
            run_scores(date, probe, RUN_DIR / "probe_scores.jsonl", a.workers)
            per = (time.time() - t0) / len(probe)
            est = per * len(universe)
            print(f"[probe] {len(probe)} 只 {time.time()-t0:.1f}s → "
                  f"{per:.3f}s/只，外推全量 {len(universe)} 只 ≈ {est/60:.1f} min")
            if est > BUDGET_SEC:
                final, universe_note = degraded_set, (
                    f"**降级**：外推全量 {est/60:.0f} min > 20 min（50 只探针实测）；"
                    f"universe = 池 top300 ∪ watchlist/持仓 {len(wcodes)} 只 "
                    f"∪ 规则名单 {len(rule_codes)} 只 = {len(degraded_set)} 只"
                    f"【deviation 已登记】")
            else:
                final, universe_note = universe, \
                    f"全量 {len(universe)} 只（探针外推 {est/60:.1f} min ≤ 20 min）"
        final = sorted(set(final) | rule_codes)   # 规则名单必须有 T5 分数
        scores_path = RUN_DIR / f"universe_scores_{date}.jsonl"
        t0 = time.time()
        elapsed = run_scores(date, final, scores_path, a.workers)
        n_scored = len(load_scores(scores_path))
        universe_note += f"；实际打分 {n_scored} 只，耗时 {elapsed:.0f}s"

    scores = load_scores(scores_path)
    model_top = rank_model_list(scores, a.top_k, set(universe))
    if not model_top:
        raise SystemExit("模型名单为空（全部 unscored）——检查打分输出")

    if requested_date != date:
        universe_note = (f"**日期回退**：请求打分日 {requested_date} 无交易数据，"
                         f"实际评分日 {date}。") + universe_note
    md_path = GREEN_V2 / f"daily_dual_list_{requested_date}.md"
    overlap = write_dual_md(date, trading_date, universe_note, mhash,
                            pool, rule, model_top, scores, md_path)
    print(f"[md] 双名单: {md_path}（实际评分日 {date}，重叠 {len(overlap)} 只）")

    session_id = None
    if not a.no_db:
        draft = build_draft(trading_date, model_top, date, mhash,
                            universe_note)
        session_id, draft_path = create_draft_pg(draft)
        print(f"[db] model_shadow session_id={session_id} "
              f"trading_date={trading_date}（draft: {draft_path}）")
    print(json.dumps({"date": date, "trading_date": trading_date,
                      "model_hash": mhash, "universe": universe_note,
                      "top_k": len(model_top), "overlap": len(overlap),
                      "session_id": session_id}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
