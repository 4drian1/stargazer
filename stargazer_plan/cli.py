"""第 4.1 步：真实计划生产入口 CLI（cli.py）。

用法（d:/stargazer/stargazer_plan 目录下，PYTHONIOENCODING=utf-8）：
    python -m cli create_draft draft.json        # 录入对话证据 + 结构化草案（DRAFT）
    python -m cli show_draft [--session 1]       # 展示（默认最新 session）
    python -m cli freeze --session 1 --version 1 # 人工确认冻结（唯一冻结通道）
    python -m cli revise draft2.json --session 1 # 追加新版本（旧版不动）
    python -m cli run_triggers --session 1 --version 1 --date 20260916
                                                 # 状态机跑冻结版计划，写 trigger_event
    python -m cli settle --session 1 --version 1 [--label-version prod_v1]
                                                 # 盘后五结局打标 + 未执行反事实
    python -m cli record_buy --session 1 --candidate 10 --price 5.80 --volume 100
                                                 # 盘台录入：批准买入 + 成交 BUY
    python -m cli record_sell --session 1 --candidate 10 --price 6.20 --volume 100 \
            --risk-type TRAIL_STOP [--evidence-ref 截图/备注]
                                                 # 盘台录入：卖出 + 风险类型（冻结枚举）

draft.json 结构见 spec/plan_db_v1.md 与 create_draft 的字段映射。
规则 DSL 语法见 plan_db/dsl.py（price>=X / price<=X / pct_chg<=-Y% / volume_ratio>=Z /
time>=HH:MM / and/or/not）。

每日草案生成走 draft_gen.py（green_watch 池 → draft.json → create_draft 落库）；
盘中监控走 monitor_loop.py：PG 模式（--date/--interval/--once/--push-observe）或
独立模式（--plan-file frozen_XXX.json [--events-out events.jsonl]，无 PG/psycopg）。
监控机事件由本机 `python -m cli merge_events --file events.jsonl` 幂等并回 PG；
冻结计划用 `python -m cli export_frozen --date YYYYMMDD --out frozen.json` 导出。
"""
import argparse
import datetime as dt
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))  # 保证 plan_db 可导入

from plan_db import ddl, repo
from plan_db.db import get_pg_conn as connect

RISK_TYPES = ("E0_HARD_STOP", "TRAIL_STOP", "REBOUND_FAIL", "TIME_EXIT",
              "RISK_FLOOR", "SECTOR_WEAK", "MARKET_RISK", "NEWS_EVENT",
              "MANUAL_OTHER")

MINUTE_CACHE = Path(__file__).resolve().parent / "data" / "minute_cache"


def _daily_parts_root():
    """日线分区目录：环境变量 > 本机覆盖文件 stargazer_plan/data/daily_parts_root.txt
    （已 gitignore）> 兼容旧缺省。与桌面版同一回退口径（STARGAZER_TDX_ROOT 不直接
    指向日线分区，故单独覆盖文件）。"""
    import os
    env = os.environ.get('STARGAZER_DAILY_PARTS')
    if env:
        return env
    override = Path(__file__).resolve().parent / 'data' / 'daily_parts_root.txt'
    if override.is_file():
        text = override.read_text(encoding='utf-8').strip()
        if text:
            return text
    return r'D:/a股数据/data_daily_parts'


DAILY_PARTS = Path(_daily_parts_root())


def load_minute(code, day, *, force=False):
    """生产分钟线：实现已移至 mquote.py（同签名），此处保持包装以便既有调用方
    与文档不变；force=True 忽略 parquet 缓存直接重拉（盘中监控用）。"""
    import mquote
    return mquote.load_minute(code, day, force=force)


def load_pre_close(code, day):
    """事件日前一交易日收盘（日线 parquet）；实现已移至 mquote.py。"""
    import mquote
    return mquote.load_pre_close(code, day)


# ---------------------------------------------------------------- create / revise
def _apply_draft(conn, draft, session_id, version_no):
    """把草案 JSON 落到指定版本（幂等）。返回 (version_id, status)。"""
    cands = draft.get("candidates") or []
    if not cands:
        raise SystemExit("草案至少需要一个 candidate")
    # 规范化成 compute_content_hash 期望的键（desc 字段进入三条件 JSON）
    def _cond(key, c):
        d = c.get(key) or {}
        return {"desc": d.get("desc"), "dsl": d.get("dsl")}
    norm = [{
        "sort_order": c.get("sort_order", 1),
        "code": str(c["code"]).zfill(6),
        "trigger_json": _cond("trigger", c),
        "invalidate_json": _cond("invalidate", c),
        "observe_json": _cond("observe", c),
    } for c in cands]
    content_hash = repo.compute_content_hash(session_id, version_no, norm)
    version_id, status = repo.ensure_version(conn, session_id, version_no, content_hash)
    for c, n in zip(cands, norm):
        cid = repo.ensure_candidate(
            conn, version_id, n["sort_order"], n["code"],
            name=c.get("name"), direction=c.get("direction", "BUY"),
            trigger_json=n["trigger_json"],
            invalidate_json=n["invalidate_json"],
            observe_json=n["observe_json"])
        for rule_type, key in (("TRIGGER", "trigger"), ("INVALIDATE", "invalidate"),
                               ("OBSERVE", "observe")):
            dsl = (c.get(key) or {}).get("dsl")
            if dsl:
                repo.ensure_trigger_rule(conn, cid, rule_type, dsl,
                                         condition_json={"desc": (c.get(key) or {}).get("desc")})
    refs = {"version_id": version_id, "version_no": version_no}
    for i, d in enumerate(draft.get("dialogues") or [], start=1):
        repo.ensure_dialogue_evidence(
            conn, session_id, d.get("message_ref", f"m{i}"),
            speaker=d.get("speaker"), content=d.get("content"),
            structured_refs=refs,
            message_ts=d.get("message_ts"))
    return version_id, status


def cmd_create_draft(args):
    draft = json.loads(Path(args.file).read_text(encoding="utf-8"))
    conn = connect()
    try:
        session_id = repo.upsert_session(
            conn, draft["trading_date"], draft.get("session_tag", "premarket"),
            dialogue_ref=draft.get("dialogue_ref"), is_demo=False)
        vid, st = _apply_draft(conn, draft, session_id, 1)
        print(f"session_id={session_id} version_id={vid} status={st}（待人工确认）")
        print(f"候选 {len(draft.get('candidates') or [])} 条；用 show_draft 查看后用 freeze 冻结")
    finally:
        conn.close()


def cmd_revise(args):
    draft = json.loads(Path(args.file).read_text(encoding="utf-8"))
    conn = connect()
    try:
        row = conn.execute(
            "SELECT COALESCE(MAX(version_no),0)+1 FROM stargazer_plan.plan_version "
            "WHERE session_id=%s", (args.session,)).fetchone()
        vn = row[0]
        vid, st = _apply_draft(conn, draft, args.session, vn)
        print(f"session {args.session} 追加 version {vn}（version_id={vid}，{st}）"
              f"——旧版本保持冻结不动")
    finally:
        conn.close()


# ---------------------------------------------------------------- show / freeze
def cmd_show_draft(args):
    conn = connect()
    try:
        sid = args.session or conn.execute(
            "SELECT MAX(session_id) FROM stargazer_plan.plan_session").fetchone()[0]
        s = conn.execute(
            "SELECT session_id, trading_date, session_tag, is_demo "
            "FROM stargazer_plan.plan_session WHERE session_id=%s", (sid,)).fetchone()
        print(f"session {s[0]} | 交易日 {s[1]} | {s[2]} | demo={s[3]}")
        vers = conn.execute(
            "SELECT version_id, version_no, status, frozen_at, content_hash "
            "FROM stargazer_plan.plan_version WHERE session_id=%s ORDER BY version_no", (sid,))
        for v in vers:
            print(f"  v{v[1]} [{v[2]}] frozen_at={v[3]} hash={str(v[4])[:12]}")
            cands = conn.execute(
                "SELECT candidate_id, sort_order, code, name, direction "
                "FROM stargazer_plan.plan_candidate WHERE version_id=%s ORDER BY sort_order",
                (v[0],)).fetchall()
            for cd in cands:
                print(f"    #{cd[1]} {cd[4]} {cd[2]} {cd[3] or ''} (candidate_id={cd[0]})")
                rules = conn.execute(
                    "SELECT rule_type, rule_dsl FROM stargazer_plan.trigger_rule "
                    "WHERE candidate_id=%s ORDER BY rule_type", (cd[0],)).fetchall()
                for rt, dsl in rules:
                    print(f"      {rt}: {dsl}")
        ev = conn.execute(
            "SELECT message_ref, speaker, left(content,40) FROM stargazer_plan.dialogue_evidence "
            "WHERE session_id=%s ORDER BY message_ref", (sid,)).fetchall()
        for e in ev:
            print(f"  证据[{e[0]}] {e[1] or ''}: {e[2]}")
    finally:
        conn.close()


def cmd_freeze(args):
    conn = connect()
    try:
        row = conn.execute(
            "SELECT version_id, status FROM stargazer_plan.plan_version "
            "WHERE session_id=%s AND version_no=%s", (args.session, args.version)).fetchone()
        if not row:
            raise SystemExit("版本不存在")
        if row[1] != "DRAFT":
            raise SystemExit(f"当前状态 {row[1]}，只有 DRAFT 可冻结")
        repo.freeze_version(conn, row[0])
        v = repo.get_version(conn, row[0])
        print(f"已冻结 session {args.session} version {args.version} "
              f"(version_id={v[0]}) frozen_at={v[4]} hash={str(v[5])[:12]}")
        print("冻结后任何业务字段不可修改；变更请用 revise 追加新版本")
    finally:
        conn.close()


# ---------------------------------------------------------------- run_triggers
def cmd_run_triggers(args):
    from plan_db import machine
    conn = connect()
    try:
        v = conn.execute(
            "SELECT version_id, status FROM stargazer_plan.plan_version "
            "WHERE session_id=%s AND version_no=%s", (args.session, args.version)).fetchone()
        if not v:
            raise SystemExit("版本不存在")
        if v[1] != "FROZEN":
            raise SystemExit(f"版本状态 {v[1]}，只有 FROZEN 版本可跑状态机")
        cands = conn.execute(
            "SELECT candidate_id, code FROM stargazer_plan.plan_candidate "
            "WHERE version_id=%s ORDER BY sort_order", (v[0],)).fetchall()
        pre_close = None
        for cid, code in cands:
            minute = load_minute(code, args.date)
            if minute is None or minute.empty:
                print(f"  {code}: DATA_MISSING（分钟数据不可得，不猜测）")
                continue
            if pre_close is None:
                pre_close = load_pre_close(code, args.date)
            rules = machine.load_rules(conn, cid)
            evs, ih = machine.run_minute_machine(conn, cid, code, args.date, minute, rules,
                                                 pre_close=pre_close,
                                                 label_version=args.label_version)
            fired = [e for e in evs if e[1] != "OBSERVE"]
            print(f"  {code}: 分钟 {len(minute)} 条，触发 {len(fired)} / 观察 "
                  f"{len(evs) - len(fired)}（inputs_hash={str(ih)[:12]} 已落库可重放）")
    finally:
        conn.close()


# ---------------------------------------------------------------- settle / record
def _validate_candidate(conn, session_id, candidate_id):
    """candidate 必须属于该 session（经 plan_version 关联），否则退出。"""
    row = conn.execute(
        "SELECT 1 FROM stargazer_plan.plan_candidate c "
        "JOIN stargazer_plan.plan_version v ON v.version_id=c.version_id "
        "WHERE c.candidate_id=%s AND v.session_id=%s", (candidate_id, session_id)).fetchone()
    if not row:
        raise SystemExit(f"candidate {candidate_id} 不属于 session {session_id}")


def _assumed_trigger_price(conn, candidate_id):
    """从 TRIGGER 规则 DSL 解析假设触发价（如 'price <= 5.80' → 5.80）；
    未触发计划的反事实 entry 用，basis 里显式标注假设。"""
    rows = conn.execute(
        "SELECT rule_dsl FROM stargazer_plan.trigger_rule "
        "WHERE candidate_id=%s AND rule_type='TRIGGER' ORDER BY rule_dsl",
        (candidate_id,)).fetchall()
    for (text,) in rows:
        m = re.search(r"price\s*(?:>=|<=)\s*(-?\d+(?:\.\d+)?)", text or "")
        if m:
            return float(m.group(1))
    return None


def _close_at(minute, fired_at):
    """分钟 DataFrame 里 fired_at 当分钟的收盘价（时区归一后精确匹配）。"""
    if minute is None or minute.empty:
        return None
    import pandas as pd
    t = pd.Timestamp(fired_at)
    if t.tzinfo is not None:
        t = t.tz_convert("Asia/Shanghai").tz_localize(None)
    sub = minute[minute["dt"] == t]
    return float(sub["close"].iloc[0]) if len(sub) else None


def cmd_settle(args):
    """盘后结算：版本全部候选五结局打标 + 未执行反事实（outcome.write_outcome）。

    结局与库内已有 trigger_event 一致：有 TRIGGER 事件即 triggered；已有任何
    trigger_event 即证明当日分钟输入存在（minute_present）；无事件的候选
    现场 load_minute 判定，缺失 → DATA_MISSING（绝不猜）。"""
    import pandas as pd
    from plan_db import daily as daily_mod
    from plan_db import outcome
    conn = connect()
    try:
        v = conn.execute(
            "SELECT v.version_id, v.status, s.trading_date FROM stargazer_plan.plan_version v "
            "JOIN stargazer_plan.plan_session s ON s.session_id=v.session_id "
            "WHERE v.session_id=%s AND v.version_no=%s", (args.session, args.version)).fetchone()
        if not v:
            raise SystemExit("版本不存在")
        version_id, status, trading_date = v
        day = trading_date.strftime("%Y%m%d")
        day_dt = pd.Timestamp(trading_date)
        cands = conn.execute(
            "SELECT candidate_id, code, name FROM stargazer_plan.plan_candidate "
            "WHERE version_id=%s ORDER BY sort_order", (version_id,)).fetchall()
        if not cands:
            raise SystemExit("该版本没有候选")
        overlap = "同日多候选重叠，只标'事件后反应'，不声称因果" if len(cands) > 1 else None
        print(f"settle session {args.session} version {args.version} "
              f"(version_id={version_id}, {status}) trading_date={trading_date} "
              f"label_version={args.label_version}")
        for cid, code, name in cands:
            evs = conn.execute(
                "SELECT fired_at, rule_type FROM stargazer_plan.trigger_event "
                "WHERE candidate_id=%s AND label_version=%s ORDER BY fired_at",
                (cid, args.label_version)).fetchall()
            trig = [t for t, rt in evs if rt == "TRIGGER"]
            minute = None
            if not evs:
                minute = load_minute(code, day)
            minute_present = bool(evs) or (minute is not None and len(minute) > 0)
            acts = {r[0] for r in conn.execute(
                "SELECT DISTINCT action FROM stargazer_plan.human_decision "
                "WHERE candidate_id=%s", (cid,)).fetchall()}
            has_exec = conn.execute(
                "SELECT 1 FROM stargazer_plan.execution_log e "
                "JOIN stargazer_plan.human_decision d ON d.decision_id=e.decision_id "
                "WHERE d.candidate_id=%s AND e.action <> 'NONE' LIMIT 1",
                (cid,)).fetchone() is not None
            oc = outcome.classify_outcome(minute_present, bool(trig), acts, has_exec)
            # entry：EXECUTED 取真实 BUY 成交价；触发未执行取触发分钟收盘价；
            # 未触发取假设触发价（basis 显式标注）；DATA_MISSING 为 NULL
            entry = None
            if oc == "EXECUTED":
                row = conn.execute(
                    "SELECT e.price FROM stargazer_plan.execution_log e "
                    "JOIN stargazer_plan.human_decision d ON d.decision_id=e.decision_id "
                    "WHERE d.candidate_id=%s AND e.action='BUY' "
                    "ORDER BY e.executed_at LIMIT 1", (cid,)).fetchone()
                entry = float(row[0]) if row else None
                basis = "真实人工成交，无反事实（EXECUTED）"
            elif oc == "DATA_MISSING":
                basis = "分钟数据缺失，无法判定（绝不猜测）"
            else:
                if trig:
                    if minute is None:
                        minute = load_minute(code, pd.Timestamp(trig[0]).strftime("%Y%m%d"))
                    entry = _close_at(minute, trig[0])
                if entry is None:
                    entry = _assumed_trigger_price(conn, cid)
                if trig and entry is not None:
                    basis = (f"would_fill_at=trigger_price_if_fired={entry:.2f}"
                             f"（触发分钟收盘价，反事实假设）")
                elif entry is not None:
                    basis = (f"would_fill_at=assumed_trigger_price={entry:.2f}"
                             f"（计划全天未触发，假设价，非真实成交）")
                else:
                    basis = "未触发且无法解析假设触发价（反事实不可得）"
            if oc in ("NOT_TRIGGERED", "DECLINED", "NO_DECISION") and entry is not None:
                cf = outcome.counterfactual_from_daily(
                    daily_mod.load_daily(code), day_dt, entry)
            else:
                cf = {"return_1d": None, "return_3d": None, "return_5d": None,
                      "mfe": None, "mae": None, "label_status": "OK"}
            if oc == "DATA_MISSING":
                cf["label_status"] = "CENSORED_TAIL"
            if cf["label_status"] == "CENSORED_TAIL" and oc != "DATA_MISSING":
                basis += "；CENSORED_TAIL：日线增量窗口未走完，缺失字段绝不填 0"
            outcome.write_outcome(
                conn, cid, args.label_version, outcome=oc,
                triggered_at=trig[0] if trig else None, entry_price=entry,
                basis=basis, overlap_note=overlap, label_status=cf["label_status"],
                return_1d=cf["return_1d"], return_3d=cf["return_3d"],
                return_5d=cf["return_5d"], mfe=cf["mfe"], mae=cf["mae"])
            print(f"  #{code} {name or ''}: {oc}"
                  f"（entry={entry} label_status={cf['label_status']}）")
    finally:
        conn.close()


def cmd_record_buy(args):
    """盘台录入：human_decision(DESK, APPROVE) + execution_log(BUY)。"""
    conn = connect()
    try:
        _validate_candidate(conn, args.session, args.candidate)
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO stargazer_plan.human_decision "
                "(candidate_id, decided_at, source, action, reason) "
                "VALUES (%s, now(), 'DESK','APPROVE',%s) RETURNING decision_id",
                (args.candidate, f"record_buy 盘台批准买入（session {args.session}）"))
            decision_id = cur.fetchone()[0]
            ref = f"cli-buy-{args.candidate}-{dt.datetime.now():%Y%m%d%H%M%S}"
            cur.execute(
                "INSERT INTO stargazer_plan.execution_log "
                "(decision_id, source, source_ref, action, price, volume, executed_at) "
                "VALUES (%s,'DESK',%s,'BUY',%s,%s, now())",
                (decision_id, ref, args.price, args.volume))
        conn.commit()
        print(f"record_buy OK: candidate={args.candidate} decision_id={decision_id} "
              f"price={args.price} volume={args.volume}")
    finally:
        conn.close()


def cmd_record_reject(args):
    """盘台录入：human_decision(CHAT, REJECT, reason)。用于"否决但留档"的候选——
    候选保留在计划中（observe/失效照跑攒尺子数据），但不出 BUY 触发，
    日后归因 Override Regret / Human Alpha。"""
    conn = connect()
    try:
        _validate_candidate(conn, args.session, args.candidate)
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO stargazer_plan.human_decision "
                "(candidate_id, decided_at, source, action, reason) "
                "VALUES (%s, now(), 'CHAT','REJECT',%s) RETURNING decision_id",
                (args.candidate, args.reason))
            decision_id = cur.fetchone()[0]
        conn.commit()
        print(f"record_reject OK: candidate={args.candidate} decision_id={decision_id}")
    finally:
        conn.close()


def cmd_record_sell(args):
    """盘台录入：human_decision(DESK, APPROVE, risk_type, evidence_ref)
    + execution_log(SELL, risk_type)。risk_type 为冻结枚举，MANUAL_OTHER 必须给
    --evidence-ref 说明。"""
    if args.risk_type == "MANUAL_OTHER" and not args.evidence_ref:
        raise SystemExit("risk_type=MANUAL_OTHER 时 --evidence-ref 必填（说明卖出理由）")
    conn = connect()
    try:
        _validate_candidate(conn, args.session, args.candidate)
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO stargazer_plan.human_decision "
                "(candidate_id, decided_at, source, action, reason, risk_type, evidence_ref) "
                "VALUES (%s, now(), 'DESK','APPROVE',%s,%s,%s) RETURNING decision_id",
                (args.candidate,
                 f"record_sell 盘台确认卖出（session {args.session}）",
                 args.risk_type, args.evidence_ref))
            decision_id = cur.fetchone()[0]
            ref = f"cli-sell-{args.candidate}-{dt.datetime.now():%Y%m%d%H%M%S}"
            cur.execute(
                "INSERT INTO stargazer_plan.execution_log "
                "(decision_id, source, source_ref, action, price, volume, risk_type, executed_at) "
                "VALUES (%s,'DESK',%s,'SELL',%s,%s,%s, now())",
                (decision_id, ref, args.price, args.volume, args.risk_type))
        conn.commit()
        print(f"record_sell OK: candidate={args.candidate} decision_id={decision_id} "
              f"price={args.price} volume={args.volume} risk_type={args.risk_type}")
    finally:
        conn.close()


# ---------------------------------------------------------------- export / merge
def cmd_export_frozen(args):
    """导出某日全部 FROZEN 版本为冻结计划 JSON（给无 PG 的监控机盘后/盘前使用）。

    结构：{"trading_date","exported_at","sessions":[{session_id,session_tag,
    version_no,content_hash,candidates:[{sort_order,code,name,direction,
    pre_close,pre_close_missing,rules:{TRIGGER/INVALIDATE/OBSERVE:{dsl,desc}}}]}]}。
    pre_close 逐候选用 load_pre_close 加载，缺失记 null 并标 pre_close_missing。"""
    conn = connect()
    try:
        day_iso = f"{args.date[:4]}-{args.date[4:6]}-{args.date[6:]}"
        sessions = conn.execute(
            "SELECT session_id, session_tag FROM stargazer_plan.plan_session "
            "WHERE trading_date=%s ORDER BY session_id", (day_iso,)).fetchall()
        out = {"trading_date": args.date,
               "exported_at": dt.datetime.now().isoformat(timespec="seconds"),
               "sessions": []}
        for sid, tag in sessions:
            vers = conn.execute(
                "SELECT version_id, version_no, content_hash "
                "FROM stargazer_plan.plan_version "
                "WHERE session_id=%s AND status='FROZEN' ORDER BY version_no",
                (sid,)).fetchall()
            for vid, vno, chash in vers:
                cands = []
                for cid, so, code, name, direction in conn.execute(
                        "SELECT candidate_id, sort_order, code, name, direction "
                        "FROM stargazer_plan.plan_candidate "
                        "WHERE version_id=%s ORDER BY sort_order", (vid,)).fetchall():
                    rules = {"TRIGGER": None, "INVALIDATE": None, "OBSERVE": None}
                    for rt, dsl_text, cj in conn.execute(
                            "SELECT rule_type, rule_dsl, condition_json "
                            "FROM stargazer_plan.trigger_rule "
                            "WHERE candidate_id=%s ORDER BY rule_type", (cid,)).fetchall():
                        desc = cj.get("desc") if isinstance(cj, dict) else None
                        if rt in rules:
                            rules[rt] = {"dsl": dsl_text, "desc": desc}
                    pre_close = load_pre_close(code, args.date)
                    cands.append({
                        "sort_order": so, "code": code, "name": name,
                        "direction": direction,
                        "pre_close": pre_close,
                        "pre_close_missing": pre_close is None,
                        "rules": rules,
                    })
                out["sessions"].append({
                    "session_id": sid, "session_tag": tag, "version_no": vno,
                    "content_hash": chash, "candidates": cands,
                })
        n_c = sum(len(s["candidates"]) for s in out["sessions"])
        n_v = len(out["sessions"])
        if not n_v:
            print(f"警告：{args.date} 没有 FROZEN 版本（导出为空）")
        else:
            miss = [f"{c['code']}(s{s['session_id']})" for s in out["sessions"]
                    for c in s["candidates"] if c["pre_close_missing"]]
            print(f"导出 {args.date}: {n_v} 个 FROZEN 版本 / {n_c} 个候选"
                  + (f"；pre_close 缺失: {', '.join(miss)}" if miss else ""))
        Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=2),
                                  encoding="utf-8")
        print(f"已写出: {args.out}")
    finally:
        conn.close()


def cmd_merge_events(args):
    """把监控机写出的 events.jsonl 幂等并入 PG trigger_event（可重复执行）。

    定位链：事件 (day, session_tag) → plan_session → 该 session 最新 FROZEN 版本
    → candidate（先 sort_order+code，fallback 只按 code）。INSERT ... ON CONFLICT
    DO NOTHING；统计 inserted/duplicate/unknown，有 unknown 退出码 1。"""
    import pandas as pd
    conn = connect()
    inserted = duplicate = unknown = 0
    unknown_lines = []
    try:
        path = Path(args.file)
        if not path.exists():
            raise SystemExit(f"事件文件不存在: {path}")
        with conn.cursor() as cur:
            for lineno, line in enumerate(
                    path.read_text(encoding="utf-8").splitlines(), start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    ev = json.loads(line)
                except json.JSONDecodeError:
                    unknown += 1
                    unknown_lines.append(f"第{lineno}行 JSON 解析失败")
                    continue
                day, tag = str(ev.get("day", "")), str(ev.get("session_tag", ""))
                fired_at = pd.Timestamp(str(ev.get("fired_at"))).to_pydatetime()
                day_iso = f"{day[:4]}-{day[4:6]}-{day[6:]}" if len(day) == 8 else day
                cur.execute(
                    "SELECT session_id FROM stargazer_plan.plan_session "
                    "WHERE trading_date=%s AND session_tag=%s", (day_iso, tag))
                sid = cur.fetchone()
                if not sid:
                    unknown += 1
                    unknown_lines.append(f"第{lineno}行 session 不存在({day},{tag})")
                    continue
                cur.execute(
                    "SELECT version_id FROM stargazer_plan.plan_version "
                    "WHERE session_id=%s AND status='FROZEN' "
                    "ORDER BY version_no DESC LIMIT 1", (sid[0],))
                ver = cur.fetchone()
                if not ver:
                    unknown += 1
                    unknown_lines.append(f"第{lineno}行 session {sid[0]} 无 FROZEN 版本")
                    continue
                cur.execute(
                    "SELECT candidate_id FROM stargazer_plan.plan_candidate "
                    "WHERE version_id=%s AND sort_order=%s AND code=%s",
                    (ver[0], ev.get("sort_order"), str(ev.get("code", "")).zfill(6)))
                cand = cur.fetchone()
                if not cand:  # fallback：只按 code
                    cur.execute(
                        "SELECT candidate_id FROM stargazer_plan.plan_candidate "
                        "WHERE version_id=%s AND code=%s ORDER BY sort_order LIMIT 1",
                        (ver[0], str(ev.get("code", "")).zfill(6)))
                    cand = cur.fetchone()
                if not cand:
                    unknown += 1
                    unknown_lines.append(
                        f"第{lineno}行 candidate 不存在(v{ver[0]},"
                        f"#{ev.get('sort_order')},{ev.get('code')})")
                    continue
                cur.execute(
                    "INSERT INTO stargazer_plan.trigger_event "
                    "(candidate_id, fired_at, rule_type, inputs_hash, label_version) "
                    "VALUES (%s,%s,%s,%s,%s) "
                    "ON CONFLICT (candidate_id, fired_at, rule_type, inputs_hash) "
                    "DO NOTHING",
                    (cand[0], fired_at, ev.get("rule_type"),
                     ev.get("inputs_hash"), ev.get("label_version", "prod_v1")))
                if cur.rowcount == 1:
                    inserted += 1
                else:
                    duplicate += 1
        conn.commit()
        print(f"merge_events: inserted={inserted} duplicate={duplicate} "
              f"unknown={unknown}（文件 {path.name}）")
        for msg in unknown_lines[:20]:
            print(f"  unknown: {msg}")
        if unknown:
            raise SystemExit(1)
    finally:
        conn.close()


def main(argv=None):
    ap = argparse.ArgumentParser(description="stargazer_plan 生产入口（第 4.1 步）")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("create_draft", help="录入对话证据+结构化草案")
    p.add_argument("file", help="draft.json 路径")
    p.set_defaults(fn=cmd_create_draft)

    p = sub.add_parser("show_draft", help="展示草案（默认最新 session）")
    p.add_argument("--session", type=int, default=None)
    p.set_defaults(fn=cmd_show_draft)

    p = sub.add_parser("freeze", help="人工确认冻结（唯一冻结通道）")
    p.add_argument("--session", type=int, required=True)
    p.add_argument("--version", type=int, required=True)
    p.set_defaults(fn=cmd_freeze)

    p = sub.add_parser("revise", help="追加新版本（旧版本不可变）")
    p.add_argument("file")
    p.add_argument("--session", type=int, required=True)
    p.set_defaults(fn=cmd_revise)

    p = sub.add_parser("run_triggers", help="跑冻结版计划的状态机")
    p.add_argument("--session", type=int, required=True)
    p.add_argument("--version", type=int, required=True)
    p.add_argument("--date", required=True, help="交易日 YYYYMMDD")
    p.add_argument("--label-version", default="prod_v1", dest="label_version")
    p.set_defaults(fn=cmd_run_triggers)

    p = sub.add_parser("settle", help="盘后五结局打标 + 未执行反事实")
    p.add_argument("--session", type=int, required=True)
    p.add_argument("--version", type=int, required=True)
    p.add_argument("--label-version", default="prod_v1", dest="label_version")
    p.set_defaults(fn=cmd_settle)

    p = sub.add_parser("record_buy", help="盘台录入：批准买入 + 成交 BUY")
    p.add_argument("--session", type=int, required=True)
    p.add_argument("--candidate", type=int, required=True, help="candidate_id")
    p.add_argument("--price", type=float, required=True)
    p.add_argument("--volume", type=int, required=True)
    p.set_defaults(fn=cmd_record_buy)

    p = sub.add_parser("record_sell", help="盘台录入：卖出 + 风险类型")
    p.add_argument("--session", type=int, required=True)
    p.add_argument("--candidate", type=int, required=True, help="candidate_id")
    p.add_argument("--price", type=float, required=True)
    p.add_argument("--volume", type=int, required=True)
    p.add_argument("--risk-type", required=True, choices=RISK_TYPES, dest="risk_type",
                   help="冻结枚举: " + "/".join(RISK_TYPES))
    p.add_argument("--evidence-ref", default=None, dest="evidence_ref",
                   help="证据引用（MANUAL_OTHER 时必填）")
    p.set_defaults(fn=cmd_record_sell)

    p = sub.add_parser("record_reject", help="盘台录入：否决候选（留档不删，不退出 BUY 触发由 draft 控制）")
    p.add_argument("--session", type=int, required=True)
    p.add_argument("--candidate", type=int, required=True, help="candidate_id")
    p.add_argument("--reason", required=True, help="否决理由（进 human_decision.reason）")
    p.set_defaults(fn=cmd_record_reject)

    p = sub.add_parser("export_frozen",
                       help="导出某日全部 FROZEN 版本为冻结计划 JSON（监控机用）")
    p.add_argument("--date", required=True, help="交易日 YYYYMMDD")
    p.add_argument("--out", required=True, help="输出 JSON 路径")
    p.set_defaults(fn=cmd_export_frozen)

    p = sub.add_parser("merge_events",
                       help="监控机 events.jsonl 幂等并入 PG trigger_event")
    p.add_argument("--file", required=True, help="events.jsonl 路径")
    p.set_defaults(fn=cmd_merge_events)

    args = ap.parse_args(argv)
    conn = connect()
    try:
        ddl.ensure_columns(conn)   # 规则层 v1 迁移（幂等，2026-09-17）
    finally:
        conn.close()
    args.fn(args)


if __name__ == "__main__":
    main()
