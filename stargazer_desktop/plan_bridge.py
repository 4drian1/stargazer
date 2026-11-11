r"""计划桥（桌面进程 → stargazer_plan）：「盘后总结」settle + 「交易计划页」封装。

2026-09-17 方案回退说明：原「计划闭环」的盘后跑批/候选池/DRAFT/冻结/成交录入/
外部长驻 monitor_loop 进程方案已整体回退——盘中规则监控改由 intraday_monitor
线程内完成（规则源 data/plan_frozen.json，DSL 语义同 stargazer_plan.plan_db），
不再启动任何外部监控进程。

2026-09-23（T5 双名单设计 §5）新增「交易计划页」（本机专属）：
- dual_list_run/state：子进程跑 green_v2\model_shadow_draft.py --date <YYYYMMDD>
  （锁+线程+日志+broadcast，照 ml_bridge 模式），产出 daily_dual_list_<date>.md；
- rule_pool：读 green_watch_v0\pool_latest.json（规则名单结构化数据，勾选/改价用；
  模型名单纯观察不进冻结计划，故只有规则名单需要结构化读取）；
- create_draft_from_picks：按勾选结果组装 draft.json（dialogues 带用户修改理由）
  → ``python -m cli create_draft``（cwd stargazer_plan）→ session_id；
- freeze_and_export：cli freeze + export_frozen → 直接写出 plan_file
  （data\plan_frozen.json），前端提示用户拷监控机；
- plan_ui_visible：plan.enabled=false 或全链（脚本/解释器/plan 目录）不存在时
  隐藏页面（监控机形态）。

监控机无 PG：子进程连接失败 → 返回错误文案由前端展示，不崩。
"""
from __future__ import annotations

import glob
import json
import os
import re
import subprocess
import threading
from datetime import datetime

import notifier
import paths


def _settings(cfg):
    plan = cfg.get("plan") or {}
    root = os.path.dirname(paths.APP_DIR)
    return {
        "rule_python": plan.get("rule_python") or os.path.join(
            root, "_dev_venv", "Scripts", "python.exe"),
        "plan_dir": plan.get("plan_dir") or os.path.join(root, "stargazer_plan"),
    }


def _decode_output(raw: bytes) -> str:
    """子进程输出可能是 UTF-8（PYTHONIOENCODING）或 GBK（Windows 默认控制台）。"""
    for enc in ("utf-8", "gbk"):
        try:
            return raw.decode(enc)
        except (UnicodeDecodeError, LookupError):
            continue
    return raw.decode("utf-8", errors="replace")


def _env():
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    return env


def _cli(cfg, args, timeout=300):
    settings = _settings(cfg)
    cmd = [settings["rule_python"], "-m", "cli"] + [str(a) for a in args]
    try:
        completed = subprocess.run(
            cmd, cwd=settings["plan_dir"], env=_env(), capture_output=True,
            timeout=timeout,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0), check=False)
    except Exception as exc:
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
    out = _decode_output(completed.stdout or b"")
    err = _decode_output(completed.stderr or b"")
    full = out + ("\n[stderr]\n" + err if err else "")
    return {"ok": completed.returncode == 0, "output": full[-500:], "full": full,
            "returncode": completed.returncode}


_SESSION_RE = re.compile(r"^session\s+(\d+)\s*\|", re.M)
_VERSION_RE = re.compile(r"^\s+v(\d+)\s+\[(\w+)\]", re.M)


def settle_latest(cfg):
    """盘后总结：最新 session 的最新 FROZEN 版 → `python -m cli settle`（子进程）。

    返回 {ok, output/full, session, version}；监控机无 PG 时 show_draft 输出为空，
    返回 ok=False + 错误文案（前端 addMsg 展示，不崩）。
    """
    r = _cli(cfg, ["show_draft"], timeout=120)
    text = r.get("full") or ""
    m = _SESSION_RE.search(text)
    if not m:
        return {"ok": False,
                "error": "未找到最新 session 的 FROZEN 版本（本机 plan 库不可达，"
                         "或今日尚无冻结计划）",
                "raw": text[-500:]}
    latest = int(m.group(1))
    session = version = None
    for sid in range(latest, max(latest - 15, 0), -1):
        if sid != latest:
            r = _cli(cfg, ["show_draft", "--session", sid], timeout=120)
            text = r.get("full") or ""
        frozen = [int(v.group(1)) for v in _VERSION_RE.finditer(text)
                  if v.group(2) == "FROZEN"]
        if frozen:
            session, version = sid, max(frozen)
            break
    if session is None:
        return {"ok": False,
                "error": f"最新 {latest} 及之前共 15 个 session 均无 FROZEN 版本"
                         f"（先冻结计划再结算）",
                "raw": text[-500:]}
    notifier.log(f"盘后总结：settle session {session} v{version}（子进程）")
    r2 = _cli(cfg, ["settle", "--session", session, "--version", version],
              timeout=900)
    r2["session"] = session
    r2["version"] = version
    return r2


# ---------------------------------------------------------------- 交易计划页
# 双名单子进程状态（锁+线程+日志+broadcast，照 ml_bridge 模式）
_DUAL_STATE = {"running": False, "error": None, "date": None, "md": None,
               "md_date": None, "started_at": None, "finished_at": None,
               "log_tail": ""}
_DUAL_LOCK = threading.Lock()


def _ml_root(cfg):
    plan = cfg.get("plan") or {}
    if plan.get("ml_root"):
        return plan["ml_root"]  # config 绝对路径优先（本机/源码形态）
    return os.path.join(os.path.dirname(paths.APP_DIR), "stargazer_ml")


def _dual_script(cfg):
    return os.path.join(_ml_root(cfg), "v5_hybrid", "green_v2",
                        "model_shadow_draft.py")


def _dual_md_path(cfg, date):
    return os.path.join(_ml_root(cfg), "v5_hybrid", "green_v2",
                        f"daily_dual_list_{date}.md")


def plan_ui_visible(cfg):
    """交易计划页可见性：plan.enabled=false（监控机配置）或全链不存在时隐藏。"""
    plan = cfg.get("plan") or {}
    if not plan.get("enabled", True):
        return {"visible": False, "reason": "plan.enabled=false（监控机形态）"}
    settings = _settings(cfg)
    missing = []
    if not os.path.isfile(_dual_script(cfg)):
        missing.append("model_shadow_draft.py")
    if not os.path.isfile(settings["rule_python"]):
        missing.append("rule_python")
    if not os.path.isdir(settings["plan_dir"]):
        missing.append("plan_dir")
    if missing:
        return {"visible": False, "reason": "全链不存在：" + ", ".join(missing)}
    return {"visible": True}


def dual_list_state(cfg):
    """轮询用：running/error/md（本次跑的或磁盘上最新一份 daily_dual_list_*.md）。"""
    state = dict(_DUAL_STATE)
    if not state.get("md"):
        # 本次会话没跑过（或刚启动）：磁盘上最新一份也可展示
        pattern = os.path.join(_ml_root(cfg), "v5_hybrid", "green_v2",
                               "daily_dual_list_*.md")
        files = sorted(glob.glob(pattern), key=os.path.getmtime, reverse=True)
        for p in files:
            try:
                with open(p, encoding="utf-8") as f:
                    state["md"] = f.read()
                state["md_date"] = re.search(r"(\d{8})", os.path.basename(p))
                state["md_date"] = (state["md_date"].group(1)
                                    if state["md_date"] else None)
                break
            except OSError:
                continue
    state["script_exists"] = os.path.isfile(_dual_script(cfg))
    return state


def dual_list_run(cfg, date):
    """启动双名单子进程（model_shadow_draft.py --date <YYYYMMDD>），立即返回。

    前端轮询 dual_list_state()；完成后 md 填入状态并 broadcast 一条 SSE。
    脚本不存在（监控机/另一代理尚未交付）→ 直接返回错误文案，前端展示不崩。
    """
    date = str(date or "").strip()
    if not re.fullmatch(r"\d{8}", date or ""):
        return {"ok": False, "error": f"日期格式须为 YYYYMMDD，收到 {date!r}"}
    settings = _settings(cfg)
    script = _dual_script(cfg)
    if not os.path.isfile(script):
        return {"ok": False, "error": f"双名单脚本不存在：{script}"
                                     f"（green_v2 全链未就绪，稍后再试）"}
    if not os.path.isfile(settings["rule_python"]):
        return {"ok": False, "error": f"解释器不存在：{settings['rule_python']}"}
    with _DUAL_LOCK:
        if _DUAL_STATE.get("running"):
            return {"ok": True, "running": True, "message": "双名单生成已在进行中"}
        _DUAL_STATE.update(running=True, error=None, date=date, md=None,
                           md_date=None, log_tail="",
                           started_at=datetime.now().isoformat(timespec="seconds"),
                           finished_at=None)

    def _work():
        refresh = os.path.join(os.path.dirname(script), "refresh_pool.py")
        try:
            notifier.log("交易计划：数据增量+池刷新子进程启动（refresh_pool.py）")
            rp = subprocess.run(
                [settings["rule_python"], "-X", "utf8", refresh],
                cwd=os.path.dirname(script), env=_env(), capture_output=True,
                timeout=1800,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0), check=False)
            rp_out = _decode_output(rp.stdout or b"") + _decode_output(rp.stderr or b"")
            notifier.log(f"交易计划：池刷新结束 rc={rp.returncode}")
            if rp.returncode != 0:
                raise RuntimeError("refresh_pool 失败：" + rp_out[-400:])
        except Exception as exc:
            notifier.log(f"交易计划：refresh_pool 异常 {exc}")
        cmd = [settings["rule_python"], "-X", "utf8", script, "--date", date]
        try:
            notifier.log(f"交易计划：盘后选股子进程启动（date={date}）")
            completed = subprocess.run(
                cmd, cwd=os.path.dirname(script), env=_env(), capture_output=True,
                timeout=3600,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0), check=False)
            out = _decode_output(completed.stdout or b"")
            err = _decode_output(completed.stderr or b"")
            log_path = paths.data_file(f"data/plan_dual_list_{date}.log",
                                       seed_from_bundle=False)
            try:
                os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
                with open(log_path, "w", encoding="utf-8") as f:
                    f.write(out)
                    if err:
                        f.write("\n[stderr]\n" + err)
            except OSError:
                pass
            tail = (out + err)[-500:]
            with _DUAL_LOCK:
                if completed.returncode:
                    _DUAL_STATE.update(running=False, error=(
                        f"子进程退出码 {completed.returncode}：{tail}"), log_tail=tail)
                else:
                    md_path = _dual_md_path(cfg, date)
                    try:
                        with open(md_path, encoding="utf-8") as f:
                            md_text = f.read()
                    except OSError:
                        md_text = None
                    if md_text:
                        _DUAL_STATE.update(running=False, error=None, md=md_text,
                                           md_date=date, log_tail=tail)
                    else:
                        _DUAL_STATE.update(running=False, error=(
                            f"子进程跑完但未找到 {md_path}"), log_tail=tail)
                _DUAL_STATE["finished_at"] = datetime.now().isoformat(timespec="seconds")
            notifier.log(f"交易计划：盘后选股子进程结束（date={date}，"
                         f"{'成功' if not _DUAL_STATE.get('error') else '失败'}）")
        except Exception as exc:
            with _DUAL_LOCK:
                _DUAL_STATE.update(running=False,
                                   error=f"{type(exc).__name__}: {exc}")
        notifier.broadcast({"type": "plan_dual_list", "date": date,
                            "ok": not _DUAL_STATE.get("error")})

    threading.Thread(target=_work, daemon=True, name="plan-dual-list").start()
    return {"ok": True, "running": True,
            "started_at": _DUAL_STATE.get("started_at")}


def rule_pool(cfg):
    r"""规则名单池（green_watch_v0\pool_latest.json）——勾选/改价的结构化数据源。

    draft_gen.py 同路径读取；文件不存在（池未构建）时返回错误文案，前端展示不崩。
    """
    p = os.path.join(_ml_root(cfg), "v5_hybrid", "green_watch_v0",
                     "pool_latest.json")
    if not os.path.isfile(p):
        return {"ok": False,
                "error": f"规则名单池不存在：{p}（先跑 build_pool/draft_gen 链路）"}
    try:
        with open(p, encoding="utf-8") as f:
            pool = json.load(f)
    except (OSError, ValueError) as exc:
        return {"ok": False, "error": f"pool_latest.json 读取失败: {exc}"}
    return {"ok": True, "pool": pool, "path": p}


def create_draft_from_picks(cfg, body):
    """勾选/改价结果 → draft.json（dialogues 带用户修改理由）→ cli create_draft。

    body: {trading_date, candidates: [{code, name, trigger_price|null, pre_close,
           reason}]}；三条件口径照 draft_gen.build_draft（§5 流程第 4 步）。
    """
    settings = _settings(cfg)
    date = str((body or {}).get("trading_date") or "").strip()
    if not re.fullmatch(r"\d{8}", date or ""):
        return {"ok": False, "error": f"trading_date 须为 YYYYMMDD，收到 {date!r}"}
    picks = (body or {}).get("candidates") or []
    if not picks:
        return {"ok": False, "error": "至少勾选一个候选"}
    cands, user_lines = [], []
    for i, c in enumerate(picks, 1):
        try:
            code = str(c["code"]).strip().zfill(6)
            pre_close = float(c["pre_close"])
            trigger = (float(c["trigger_price"])
                       if c.get("trigger_price") not in (None, "") else None)
        except (KeyError, TypeError, ValueError) as exc:
            return {"ok": False, "error": f"第 {i} 个候选字段非法: {exc}"}
        reason = str(c.get("reason") or "").strip()
        cands.append({
            "sort_order": i, "code": code, "name": str(c.get("name") or ""),
            "direction": "BUY",
            "trigger": ({"desc": "回踩触发价（用户在交易计划页确认/改价）",
                         "dsl": f"price <= {trigger:.2f}"} if trigger is not None
                        else {"desc": "v1 不出 BUY trigger，仅观察"
                                      "（承接修复特征组未实现）", "dsl": None}),
            "invalidate": {"desc": "未回踩且翻红，或尾盘，当日失效",
                           "dsl": f"price >= {pre_close:.2f} or time >= 14:30"},
            "observe": {"desc": "跌幅进入监控区", "dsl": "pct_chg <= -2.0"},
        })
        user_lines.append(f"- {code} {c.get('name') or ''} "
                          f"触发价={'%.2f' % trigger if trigger is not None else '仅观察'} "
                          f"昨收={pre_close:.2f}" + (f"｜用户理由: {reason}" if reason else ""))
    draft = {
        # 照 draft_gen：8 位数字串（PG date 列可接受；export_frozen 输出亦 8 位）
        "trading_date": date,
        "session_tag": "green_watch_daily",
        "candidates": cands,
        "dialogues": [
            {"message_ref": "ui", "speaker": "AGENT",
             "content": f"【交易计划界面草案】trading_date={date}，"
                        f"来源=双名单规则层勾选（模型名单纯观察不进计划），"
                        f"入选 {len(cands)} 只"},
            {"message_ref": "ui-user", "speaker": "USER",
             "content": str(body.get("note") or "").strip() or "（无总备注）"},
            {"message_ref": "ui-picks", "speaker": "USER",
             "content": "勾选/改价明细：\n" + "\n".join(user_lines)},
        ],
    }
    draft_dir = os.path.join(settings["plan_dir"], "data", "drafts")
    try:
        os.makedirs(draft_dir, exist_ok=True)
        draft_path = os.path.join(draft_dir, f"draft_ui_{date}.json")
        with open(draft_path, "w", encoding="utf-8") as f:
            json.dump(draft, f, ensure_ascii=False, indent=2)
    except OSError as exc:
        return {"ok": False, "error": f"draft.json 写出失败: {exc}"}
    notifier.log(f"交易计划：create_draft（{len(cands)} 只，draft_ui_{date}.json）")
    r = _cli(cfg, ["create_draft", draft_path], timeout=300)
    if not r.get("ok"):
        r.setdefault("error", r.get("output") or f"create_draft 退出码 {r.get('returncode')}")
        return r
    m = re.search(r"session_id=(\d+)", r.get("full") or "")
    r["session_id"] = int(m.group(1)) if m else None
    r["draft_path"] = draft_path
    return r


def freeze_and_export(cfg, session_id, trading_date):
    r"""冻结并导出：cli freeze --version 1 → export_frozen 直接写 plan_file。

    导出目标 = cfg plan.plan_file（data\plan_frozen.json），前端提示用户拷监控机。
    """
    session_id = str(session_id or "").strip()
    date = str(trading_date or "").strip()
    if not session_id.isdigit():
        return {"ok": False, "error": "session_id 必填（先点「生成计划」）"}
    if not re.fullmatch(r"\d{8}", date or ""):
        return {"ok": False, "error": f"trading_date 须为 YYYYMMDD，收到 {date!r}"}
    notifier.log(f"交易计划：freeze session {session_id} v1")
    r1 = _cli(cfg, ["freeze", "--session", session_id, "--version", "1"],
              timeout=300)
    if not r1.get("ok"):
        r1.setdefault("error", r1.get("output") or f"freeze 退出码 {r1.get('returncode')}")
        r1["step"] = "freeze"
        return r1
    rel = (cfg.get("plan") or {}).get("plan_file") or "data/plan_frozen.json"
    out_path = rel if os.path.isabs(rel) else paths.data_file(rel, seed_from_bundle=False)
    notifier.log(f"交易计划：export_frozen --date {date} → {out_path}")
    r2 = _cli(cfg, ["export_frozen", "--date", date, "--out", out_path],
              timeout=600)
    r2["step"] = "export_frozen"
    r2["path"] = out_path
    r2["frozen"] = r1.get("output") or ""
    if not r2.get("ok"):
        r2.setdefault("error", r2.get("output") or f"export_frozen 退出码 {r2.get('returncode')}")
        return r2
    # W2a：单机形态导出成功后自动同步运行目录+版本戳并触发准备；
    # 同步/准备失败不推翻导出结果，只回报警告。
    try:
        from postclose_prepared_v1 import on_plan_frozen
        r2["prepared"] = on_plan_frozen(cfg, out_path, date)
        notifier.log(f"交易计划：运行目录同步+准备清单已登记（roster="
                     f"{r2['prepared'].get('roster_size')}）")
    except Exception as exc:
        r2["prepare_warning"] = f"{type(exc).__name__}: {exc}"
        notifier.log(f"交易计划：导出后同步/准备失败（不推翻导出）：{exc}")
    return r2
