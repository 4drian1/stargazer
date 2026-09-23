"""盘中监控循环（monitor_loop.py，规则层 v1）。

两种模式：

PG 模式（本机，默认）：
    python monitor_loop.py --date 20260916 [--interval 180] [--once] [--push-observe]
    对 trading_date=--date 的全部 FROZEN 版本候选，用 cli.load_minute(force=True)
    重拉分钟线跑 plan_db.machine 状态机，幂等落 trigger_event；"新事件"
    （event_id > data/monitor_state.json 的 last_event_id，首轮=当前最大 id）经
    PushPlus 推送（TRIGGER/INVALIDATE 必推，OBSERVE 仅 --push-observe）。

独立模式（监控机，无 PG/psycopg）：
    python monitor_loop.py --plan-file frozen_20260918.json \
            [--events-out events.jsonl] [--date 20260918] [--once] [--push-observe]
    顶层与运行时不 import cli / plan_db / psycopg；用 mquote 拉分钟线、按文件路径
    加载 plan_db/dsl.py 求值，失效抑制语义与 plan_db/machine.py run_rules 逐位一致
    （最早 INVALIDATE 为 cutoff，>=cutoff 的 TRIGGER 作废，OBSERVE 原样保留）。
    inputs_hash 与 machine.compute_inputs_hash 同口径（盘后 merge_events 靠它幂等）。
    事件追加写 jsonl（UTF-8），本地去重键 (code, rule_type, fired_at)——游标即 jsonl
    自身。盘后由本机 `python -m cli merge_events --file events.jsonl` 并入 PG。

时间控制（两模式共用）：09:25 前每轮 sleep 到 09:25；15:05 后跑完最后一轮退出；
盘中 sleep --interval 秒；--once 只跑一轮（跳过时间门控）。
PushPlus token 读取链：环境变量 PUSHPLUS_TOKEN 优先 → 本文件同目录 config.yaml
的 push.pushplus_token（监控机便携配置）→ stargazer_desktop/config.yaml
（本机兼容），单文件解析 pyyaml 优先、正则兜底；为空只写日志不推送；
失败只写日志不重试。

日志：data/monitor_loop.log。
"""
import argparse
import datetime as dt
import hashlib
import json
import logging
import os
import re
import sys
import time
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

STATE_PATH = BASE_DIR / "data" / "monitor_state.json"
LOG_PATH = BASE_DIR / "data" / "monitor_loop.log"
DESKTOP_CFG = Path(r"D:\stargazer\stargazer_desktop\config.yaml")
PUSH_URL = "https://www.pushplus.plus/send"
LABEL_VERSION = "prod_v1"

OPEN_TIME = dt.time(9, 25)
CLOSE_TIME = dt.time(15, 5)

log = logging.getLogger("monitor_loop")


# ---------------------------------------------------------------- push（两模式共用）
def _read_token_from_cfg(cfg_path):
    """从单个 yaml 解析 push.pushplus_token；pyyaml 优先、正则兜底，读不到返回 None。"""
    if not cfg_path.exists():
        return None
    try:
        import yaml
        cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
        tok = ((cfg or {}).get("push") or {}).get("pushplus_token")
        if tok:
            return str(tok).strip() or None
    except ImportError:
        pass
    except Exception as e:
        log.warning("pyyaml 解析 %s 失败，改正则解析: %s", cfg_path, e)
    try:
        m = re.search(r"pushplus_token:\s*['\"]?([^'\"\s#]+)",
                      cfg_path.read_text(encoding="utf-8"))
        if m and m.group(1).strip():
            return m.group(1).strip()
    except Exception as e:
        log.warning("正则解析 %s 失败: %s", cfg_path, e)
    return None


def load_pushplus_token():
    """token 读取链：环境变量 PUSHPLUS_TOKEN 优先 → 本文件同目录 config.yaml
    （监控机便携配置）→ stargazer_desktop/config.yaml（本机兼容）。
    都读不到返回 None。"""
    env_tok = (os.environ.get("PUSHPLUS_TOKEN") or "").strip()
    if env_tok:
        return env_tok
    for cfg_path in (BASE_DIR / "config.yaml", DESKTOP_CFG):
        tok = _read_token_from_cfg(cfg_path)
        if tok:
            return tok
    return None


def _post_json(url, payload, timeout):
    """POST JSON；优先 requests，缺失时用 urllib（监控机最小依赖也能推）。"""
    try:
        import requests
        r = requests.post(url, json=payload, timeout=timeout)
        return "http={} {}".format(r.status_code, r.text[:120])
    except ImportError:
        import urllib.request
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        req = urllib.request.Request(
            url, data=data, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return "http={} {}".format(resp.status, resp.read()[:120])


def push_plus(token, title, content):
    """PushPlus 推送；失败只写日志不重试。"""
    try:
        result = _post_json(PUSH_URL, {"token": token, "title": title,
                                       "content": content, "template": "txt"},
                            timeout=10)
        log.info("PushPlus 已发送 [%s] %s", title, result)
    except Exception as e:
        log.error("PushPlus 推送失败（不重试）[%s]: %s", title, e)


def _notify(token, title, content, push_observe, rule_type):
    """按规则类型决定推送或仅记录（两模式共用）。"""
    if rule_type in ("TRIGGER", "INVALIDATE") or (rule_type == "OBSERVE" and push_observe):
        if token:
            push_plus(token, title, content)
        else:
            log.info("PushPlus token 未配置/为空，只写日志不推送: %s | %s",
                     title, content.replace("\n", " / "))
    else:
        log.info("OBSERVE 事件（未开 --push-observe，不推送）: %s", title)


# ---------------------------------------------------------------- 时间控制
def _sleep_until_open():
    now = dt.datetime.now()
    secs = max((dt.datetime.combine(now.date(), OPEN_TIME) - now).total_seconds(), 1.0)
    log.info("未到 09:25，sleep %.0f 秒后开盘监控", secs)
    time.sleep(secs)


# ---------------------------------------------------------------- PG 模式
def _load_state():
    try:
        return json.loads(STATE_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _save_state(state):
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATE_PATH.write_text(json.dumps(state, ensure_ascii=False, indent=2),
                          encoding="utf-8")


def run_round_pg(conn, day, push_observe, token, state):
    """PG 模式一轮：状态机（force 重拉分钟线）→ 查新事件 → 推送 → 更新游标。
    返回 False 表示当日无 FROZEN 版本（调用方应退出循环）。"""
    import cli  # 延迟导入：独立模式（监控机）绝不触碰
    from plan_db import machine
    day_iso = f"{day[:4]}-{day[4:6]}-{day[6:]}"
    vers = conn.execute(
        "SELECT v.version_id, v.version_no, s.session_id "
        "FROM stargazer_plan.plan_version v "
        "JOIN stargazer_plan.plan_session s ON s.session_id=v.session_id "
        "WHERE s.trading_date=%s AND v.status='FROZEN' "
        "ORDER BY s.session_id, v.version_no", (day_iso,)).fetchall()
    if not vers:
        log.warning("交易日 %s 没有 FROZEN 版本，退出监控", day_iso)
        return False
    cands = []
    for vid, vno, sid in vers:
        for cid, code, name in conn.execute(
                "SELECT candidate_id, code, name FROM stargazer_plan.plan_candidate "
                "WHERE version_id=%s ORDER BY sort_order", (vid,)).fetchall():
            cands.append((cid, code, name, vno, sid))
    log.info("本轮监控 %d 个候选（session %s 的 %d 个 FROZEN 版本）",
             len(cands), sorted({s for *_, s in cands}), len(vers))

    for cid, code, name, vno, sid in cands:
        minute = cli.load_minute(code, day, force=True)
        if minute is None or minute.empty:
            log.info("  %s: DATA_MISSING（分钟数据不可得，不猜测）", code)
            continue
        pre_close = cli.load_pre_close(code, day)
        rules = machine.load_rules(conn, cid)
        try:
            evs, ih = machine.run_minute_machine(
                conn, cid, code, day, minute, rules, pre_close=pre_close,
                label_version=LABEL_VERSION)
        except Exception as e:
            log.exception("  %s 状态机运行失败（本轮跳过）: %s", code, e)
            continue
        fired = [e for e in evs if e[1] != "OBSERVE"]
        log.info("  %s (session=%s v%s): 分钟 %d 条，新触发 %d / 观察 %d"
                 "（inputs_hash=%s）", code, sid, vno, len(minute),
                 len(fired), len(evs) - len(fired), str(ih)[:12])

    cur_max = conn.execute(
        "SELECT COALESCE(MAX(event_id),0) FROM stargazer_plan.trigger_event"
    ).fetchone()[0]
    last = int(state.get("last_event_id") or 0)
    ids = [c[0] for c in cands]
    rows = conn.execute(
        "SELECT e.event_id, e.candidate_id, e.fired_at, e.rule_type, c.code, c.name "
        "FROM stargazer_plan.trigger_event e "
        "JOIN stargazer_plan.plan_candidate c ON c.candidate_id=e.candidate_id "
        "WHERE e.event_id > %s AND e.candidate_id = ANY(%s) "
        "ORDER BY e.event_id", (last, ids)).fetchall()
    for eid, cid, fired_at, rt, code, name in rows:
        title = f"[Stargazer] {rt} {code} {name or ''}".strip()
        content = (f"交易日 {day} 候选 #{cid} {code} {name or ''}\n"
                   f"事件: {rt}\n时刻: {fired_at}\nlabel_version: {LABEL_VERSION}")
        _notify(token, title, content, push_observe, rt)
    state["last_event_id"] = cur_max
    state["date"] = day
    state["updated_at"] = dt.datetime.now().isoformat(timespec="seconds")
    _save_state(state)
    log.info("本轮结束：游标 last_event_id=%d（新事件 %d 条）", cur_max, len(rows))
    return True


# ---------------------------------------------------------------- 独立模式
def _load_dsl():
    """按文件路径加载 plan_db/dsl.py——不经过 plan_db 包（其 __init__ 会 import
    psycopg，监控机没有）。dsl.py 自身只依赖 re/pandas。"""
    import importlib.util
    path = BASE_DIR / "plan_db" / "dsl.py"
    spec = importlib.util.spec_from_file_location("stargazer_dsl", str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _eval_candidate_events(dsl, code, day, minute, pre_close, rules):
    """纯函数求事件序列 + inputs_hash；语义与 plan_db/machine.py 逐位一致：
    边缘触发（dsl.edge_fired_times）+ 失效抑制（最早 INVALIDATE 为 cutoff，
    >=cutoff 的 TRIGGER 作废，OBSERVE 原样保留）。inputs_hash 复刻
    machine.compute_inputs_hash（分钟区间指纹 + 规则 DSL 规范化串的 sha256），
    保证盘后 merge_events 能按同幂等锚去重。"""
    import pandas as pd
    minute = minute.sort_values("dt").reset_index(drop=True)
    rule_texts = [(rt, (rules.get(rt) or {})["dsl"])
                  for rt in ("INVALIDATE", "OBSERVE", "TRIGGER")
                  if (rules.get(rt) or {}).get("dsl")]
    events = []
    for rule_type, text in rule_texts:
        ast = dsl.parse(text)
        cond = dsl.evaluate(ast, minute, pre_close)
        for t in dsl.edge_fired_times(cond, minute):
            events.append((pd.Timestamp(t), rule_type))
    inv_times = [t for t, rt in events if rt == "INVALIDATE"]
    if inv_times:
        cutoff = min(inv_times)
        events = [e for e in events
                  if not (e[1] == "TRIGGER" and e[0] >= cutoff)]
    events.sort(key=lambda e: (e[0], e[1]))
    fp = {
        "code": str(code),
        "trade_day": str(day),
        "n_bars": int(len(minute)),
        "dt_first": str(minute["dt"].iloc[0]),
        "dt_last": str(minute["dt"].iloc[-1]),
        "pre_close": None if pre_close is None else float(pre_close),
        "rules": sorted({"{}|{}".format(rt, dsl.canonical(dsl.parse(text)))
                         for rt, text in rule_texts}),
    }
    raw = json.dumps(fp, ensure_ascii=False, sort_keys=True)
    return events, hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _load_seen_events(events_out):
    """jsonl 即游标：读已有行，返回去重键集合 (session_tag, code, rule_type, fired_at)。"""
    seen = set()
    if events_out.exists():
        for line in events_out.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                ev = json.loads(line)
                seen.add((ev.get("session_tag"), ev.get("code"),
                          ev.get("rule_type"), ev.get("fired_at")))
            except json.JSONDecodeError:
                continue
    return seen


def run_round_standalone(plan, events_out, day, push_observe, token):
    """独立模式一轮：mquote 拉分钟线 → dsl 求事件（含失效抑制）→ jsonl 去重
    追加 → 推送。返回 False 表示计划里没有候选。"""
    import mquote
    dsl = _load_dsl()
    cands = [(s, c) for s in plan.get("sessions") or []
             for c in s.get("candidates") or []]
    if not cands:
        log.warning("冻结计划里没有候选，退出监控")
        return False
    seen = _load_seen_events(events_out)
    n_new = 0
    new_events = []
    for sess, c in cands:
        code = c["code"]
        rules = c.get("rules") or {}
        minute = mquote.load_minute(code, day, force=True)
        if minute is None or minute.empty:
            log.info("  %s: DATA_MISSING（分钟数据不可得，不猜测）", code)
            continue
        pre_close = c.get("pre_close")
        try:
            events, ih = _eval_candidate_events(dsl, code, day, minute,
                                                pre_close, rules)
        except Exception as e:
            log.exception("  %s 规则求值失败（本轮跳过）: %s", code, e)
            continue
        fired = [e for e in events if e[1] != "OBSERVE"]
        log.info("  %s (session_tag=%s): 分钟 %d 条，新触发 %d / 观察 %d"
                 "（inputs_hash=%s）", code, sess.get("session_tag"),
                 len(minute), len(fired), len(events) - len(fired), str(ih)[:12])
        for t, rt in events:
            ev = {
                "day": day,
                "session_tag": sess.get("session_tag"),
                "code": code,
                "sort_order": c.get("sort_order"),
                "rule_type": rt,
                "fired_at": str(t),
                "dsl": (rules.get(rt) or {}).get("dsl"),
                "inputs_hash": ih,
                "minute_rows": len(minute),
                "label_version": LABEL_VERSION,
            }
            key = (ev["session_tag"], code, rt, ev["fired_at"])
            if key in seen:
                continue
            seen.add(key)
            new_events.append(ev)
            n_new += 1
    if new_events:
        events_out.parent.mkdir(parents=True, exist_ok=True)
        with events_out.open("a", encoding="utf-8") as f:
            for ev in new_events:
                f.write(json.dumps(ev, ensure_ascii=False) + "\n")
    for ev in new_events:
        title = (f"[Stargazer] {ev['rule_type']} {ev['code']}").strip()
        content = (f"交易日 {day} {ev['session_tag']} #{ev['sort_order']} "
                   f"{ev['code']}\n事件: {ev['rule_type']}\n时刻: {ev['fired_at']}\n"
                   f"label_version: {LABEL_VERSION}")
        _notify(token, title, content, push_observe, ev["rule_type"])
    log.info("本轮结束：新事件 %d 条（jsonl 累计去重键 %d 个，文件 %s）",
             n_new, len(seen), events_out)
    return True


# ---------------------------------------------------------------- main loop
def main(argv=None):
    ap = argparse.ArgumentParser(description="盘中监控循环（规则层 v1，PG/独立双模式）")
    ap.add_argument("--date", default=dt.date.today().strftime("%Y%m%d"),
                    help="交易日 YYYYMMDD（默认今日）")
    ap.add_argument("--interval", type=int, default=180, help="盘中轮询间隔秒")
    ap.add_argument("--once", action="store_true", help="只跑一轮即退出（跳过时间门控）")
    ap.add_argument("--push-observe", action="store_true", dest="push_observe",
                    help="OBSERVE 事件也推送（默认只 TRIGGER/INVALIDATE）")
    ap.add_argument("--plan-file", default=None, dest="plan_file",
                    help="独立模式：export_frozen 导出的冻结计划 JSON（不连 PG）")
    ap.add_argument("--events-out", default=None, dest="events_out",
                    help="独立模式：events.jsonl 输出路径（默认 plan-file 同目录 "
                         "events_<date>.jsonl）")
    args = ap.parse_args(argv)

    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        encoding="utf-8",
        handlers=[logging.FileHandler(LOG_PATH, encoding="utf-8"),
                  logging.StreamHandler()])
    if not re.fullmatch(r"\d{8}", args.date):
        raise SystemExit(f"--date 格式应为 YYYYMMDD: {args.date!r}")

    token = load_pushplus_token()
    if token:
        log.info("PushPlus token 已加载（%s****）", token[:4])
    else:
        log.warning("PushPlus token 为空或读不到（PUSHPLUS_TOKEN / 同目录 config.yaml / "
                    "%s push.pushplus_token），本轮起只写日志不推送", DESKTOP_CFG)

    if args.plan_file:
        # ---------------- 独立模式：全程不 import cli / plan_db / psycopg
        plan_path = Path(args.plan_file)
        if not plan_path.exists():
            raise SystemExit(f"冻结计划文件不存在: {plan_path}")
        plan = json.loads(plan_path.read_text(encoding="utf-8"))
        if str(plan.get("trading_date")) != args.date:
            raise SystemExit(f"plan-file 交易日 {plan.get('trading_date')} 与 "
                             f"--date {args.date} 不一致")
        events_out = (Path(args.events_out) if args.events_out
                      else plan_path.with_name(f"events_{args.date}.jsonl"))
        log.info("独立模式：plan=%s events_out=%s（无 PG，事件盘后 merge_events）",
                 plan_path, events_out)
        while True:
            if not args.once and dt.datetime.now().time() < OPEN_TIME:
                _sleep_until_open()
                continue
            ok = run_round_standalone(plan, events_out, args.date,
                                      args.push_observe, token)
            if args.once or not ok:
                break
            if dt.datetime.now().time() >= CLOSE_TIME:
                log.info("已过 15:05，最后一轮跑完，退出")
                break
            log.info("盘中 sleep %d 秒", args.interval)
            time.sleep(args.interval)
        return 0

    # ---------------- PG 模式（本机）
    from plan_db.db import get_pg_conn
    state = _load_state()
    if state.get("date") != args.date or not isinstance(
            state.get("last_event_id"), int):
        conn = get_pg_conn()
        try:
            cur_max = conn.execute(
                "SELECT COALESCE(MAX(event_id),0) FROM stargazer_plan.trigger_event"
            ).fetchone()[0]
        finally:
            conn.close()
        state = {"date": args.date, "last_event_id": cur_max}
        log.info("首轮：游标初始化为当前最大 event_id=%d（历史事件不推送）", cur_max)

    conn = get_pg_conn()
    try:
        while True:
            if not args.once and dt.datetime.now().time() < OPEN_TIME:
                _sleep_until_open()
                continue
            ok = run_round_pg(conn, args.date, args.push_observe, token, state)
            if args.once or not ok:
                break
            if dt.datetime.now().time() >= CLOSE_TIME:
                log.info("已过 15:05，最后一轮跑完，退出")
                break
            log.info("盘中 sleep %d 秒", args.interval)
            time.sleep(args.interval)
    finally:
        conn.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
