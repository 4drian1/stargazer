"""Flask 后端：聊天 API + SSE 推送通道 + 状态/配置接口。"""
import json
import os
import queue
import threading
import time
import datetime as dt

import yaml
from flask import Flask, jsonify, request, send_from_directory, Response

import paths
import services
import notifier
import ml_bridge
import plan_bridge
import intraday_monitor
import fmt
import pentacles_demo
import assistant_workflow
import deepseek_api
from agent.brain import Brain, load_config

# sector_pullback（→numpy+pandas）延迟到路由首次调用时才导入

def _sector_pullback():
    import sector_pullback
    return sector_pullback

app = Flask(__name__, static_folder=None)
WEB = paths.resource("web")


@app.before_request
def retire_legacy_mutations():
    if request.method == 'POST' and request.path in (
        '/api/plan/settle', '/api/plan/dual_list/run', '/api/plan/draft', '/api/plan/freeze_export'):
        return jsonify({'ok': False, 'error': '旧交易计划入口已停用；历史记录与已冻结计划的风控保留'}), 410

_cfg_lock = threading.Lock()
_brain_lock = threading.RLock()   # RLock: chat() 与 get_brain() 会嵌套获取
_brain = None


def get_cfg():
    with _cfg_lock:
        cfg = load_config()
        cfg['stable_shell'] = True
        return cfg


def get_brain():
    global _brain
    with _brain_lock:
        if _brain is None:
            _brain = Brain()
        return _brain


notifier.LOG_FILE = paths.data_file("data/push.log")
if notifier.LOG_FILE:
    os.makedirs(os.path.dirname(notifier.LOG_FILE) or ".", exist_ok=True)
notifier_thread = notifier.Notifier(get_cfg)
model_monitor_thread = intraday_monitor.IntradayMonitor(get_cfg)


# ---------- 斜杠命令（确定性操作不走LLM，输出中文化） ----------
def _name_of(code):
    """尽力取股票名用于确认文案，失败就返回空串。"""
    try:
        from tools import market
        q = market.quotes([code]).get(code)
        return (q or {}).get("name", "")
    except Exception:
        return ""


def handle_slash(text, cfg):
    if text.split()[0] in ('/scan', '/score', '/model'):
        return {'text': '旧选股入口已关闭；请到星币侍从查看当前规则观察名单。历史研究与持仓风控保留。'}
    parts = text.split()
    op = parts[0]
    try:
        if op == "/scan" and len(parts) > 1:
            include_st = len(parts) > 2 and parts[2].lower() in ("st", "含st")
            return {"text": fmt.scan(services.scan_sector(parts[1], include_st=include_st))}
        if op == "/score" and len(parts) > 1:
            return {"text": fmt.score(services.score_stocks(parts[1].replace("，", ",").split(",")))}
        if op == "/check":
            return {"text": fmt.check(services.check_portfolio(cfg))}
        if op == "/model":
            payload = ml_bridge.advice(cfg, limit=10)
            status = payload["status"]
            if not status.get("cache_ready"):
                return {"text": "本地模型尚无建议缓存。点击左侧“生成建议”，或调用 POST /api/model/run。"}
            lines = [f"本地模型数据日：{status.get('data_date', '-')}"
                     + ("（已过期，仅供回看）" if status.get("stale") else ""),
                     f"横盘候选：{status.get('horizontal_candidates', 0)}｜Kimi调用：否"]
            market_rows = payload.get("market") or []
            if market_rows:
                m = market_rows[0]
                lines.append(f"大盘诊断：预期横盘收益 {float(m.get('expected_horizontal_return') or 0):.2%}｜"
                             f"风险概率 {float(m.get('market_risk_probability') or 0):.1%}（诊断级）")
            lines.append("\n横盘 Top10（只供人工复核）：")
            for row in payload.get("horizontal") or []:
                lines.append(f"- {row.get('code') or row.get('symbol')}｜排名 {row.get('advisory_rank')}｜"
                             f"建议分 {float(row.get('advisory_score') or 0):.3f}")
            g0_rows = payload.get("g0_shadow") or []
            if g0_rows:
                lines.append("\n绿盘 G0 Top10（价格触发后仍需人工复核，不自动买入）：")
                for row in g0_rows[:10]:
                    lines.append(f"- {row.get('code')}｜排名 {row.get('advisory_rank', '-')}｜"
                                 f"T+{row.get('recommended_entry_offset', '-')}优先｜"
                                 f"回撤 {float(row.get('buy_trigger_pct') or 0):.1f}% 触发｜"
                                 f"参考价 {float(row.get('buy_trigger_price') or 0):.2f}")
            t5_rows = payload.get("t5_shadow") or []
            if t5_rows:
                lines.append("\nT5 影子双头（只记录不推荐；OOD=超出训练分布仅观察）：")
                for row in t5_rows[:10]:
                    lines.append(
                        f"- {row.get('code')}｜{row.get('dp')}｜{row.get('quadrant')}"
                        + ("｜OOD" if row.get("ood") else "")
                        + f"｜P(接飞刀){float(row.get('P_risk') or 0):.0%}｜"
                          f"P(先止盈){float(row.get('P_opp') or 0):.0%}")
            return {"text": "\n".join(lines)}
        if op == "/flow":
            return {"text": fmt.flow(services.market_flow())}
        if op in ("/t5buy", "/t5skip") and len(parts) > 1:
            # 影子决策人工录入（评审第2点：必须绑定具体决策点，禁止批量记全部 dp）：
            # /t5buy 003040 10:30 板块承接良好
            code = parts[1]
            dp_arg = parts[2] if len(parts) > 2 and ":" in (parts[2] or "") else None
            if not dp_arg:
                return {"text": "必须带决策点：/t5buy 003040 10:30 原因。"
                                "禁止对全部决策点批量记买（会毁掉 dp 可靠性分析）"}
            reason = " ".join(parts[3:]) or ""
            action = "buy" if op == "/t5buy" else "skip"
            dp = int(dp_arg.replace(":", ""))
            shadow_log = _shadow_log()
            db = intraday_monitor._db_path(cfg)
            import sqlite3
            conn = sqlite3.connect(db, timeout=20)
            row = conn.execute(
                "SELECT sig_date FROM shadow_decisions WHERE code=? AND dp=? "
                "ORDER BY sig_date DESC LIMIT 1", (code, dp)).fetchone()
            conn.close()
            if not row:
                return {"text": f"{code} 在 dp {dp_arg} 没有影子决策行（先生成建议留痕）"}
            sig_date = row[0]
            fill_price = None
            if action == "buy":
                try:
                    from tools import market
                    fill_price = ((market.quotes([code]) or {}).get(code) or {}).get("price")
                except Exception:
                    fill_price = None
            shadow_log.record_action(db, sig_date, code, dp, action, reason,
                                     fill_price=fill_price)
            return {"text": f"已记录 {code}（信号日 {sig_date}，dp {dp_arg}）：{action}"
                             + (f" @ {fill_price}" if fill_price else "")
                             + (f"，原因：{reason}" if reason else "")}
        if op == "/t5":
            date = parts[1] if len(parts) > 1 else None
            import sqlite3
            conn = sqlite3.connect(intraday_monitor._db_path(cfg), timeout=20)
            q = ("SELECT code, dp, quadrant, p_risk, p_opp, ood, action FROM "
                 "shadow_decisions WHERE sig_date=? ORDER BY p_opp DESC LIMIT 30"
                 if date else
                 "SELECT sig_date, code, dp, quadrant, p_risk, p_opp, ood, action "
                 "FROM shadow_decisions ORDER BY sig_date DESC, p_opp DESC LIMIT 30")
            rows = conn.execute(q, (int(date),) if date and str(date).isdigit()
                                else (date,) if date else ()).fetchall()
            conn.close()
            if not rows:
                return {"text": "暂无影子决策记录（先生成建议）"}
            head = ("信号日 代码 决策点 象限 P接飞刀 P先止盈 OOD 操作"
                    if not date else "代码 决策点 象限 P接飞刀 P先止盈 OOD 操作")
            lines = [head]
            for r in rows:
                lines.append(" ".join(str(x if x is not None else "-") for x in r))
            return {"text": "\n".join(lines)}
        if op == "/news":
            kw = parts[1:] or None
            return {"text": fmt.news(services.news_digest(keywords=kw))}
        if op == "/quote" and len(parts) > 1:
            return {"text": fmt.quote(parts[1], services.stock_check(parts[1]))}
        if op == "/watch" and len(parts) > 1:
            r = services.wl_add_watch(cfg, parts[1], " ".join(parts[2:]))
            if not r.get("ok"):
                return {"text": f"{parts[1]} 已经在观察名单里了"}
            reason = " ".join(parts[2:])
            return {"text": f"已加入观察名单：{_name_of(parts[1]) or ''} {parts[1]}"
                            + (f"\n理由：{reason}" if reason else "")
                            + f"\n观察名单共 {r['watch_n']} 只。单日跌超 5% 会推送预警，止损盯梢每 5 分钟盯一次。"}
        if op == "/del" and len(parts) > 1:
            services.wl_remove(cfg, parts[1])
            return {"text": f"已移除 {parts[1]}（持仓和观察名单里的记录都删掉了）"}
        if op == "/pos" and len(parts) >= 6:
            r = services.wl_add_position(cfg, parts[1], parts[2], float(parts[3]), parts[4],
                                         float(parts[5]), " ".join(parts[6:]))
            return {"text": f"已登记持仓：{parts[2]} {parts[1]}\n"
                            f"成本 {parts[3]}｜买入日 {parts[4]}｜结构止损线 {parts[5]}"
                            + (f"\n逻辑：{' '.join(parts[6:])}" if len(parts) > 6 else "")
                            + f"\n当前共 {r['positions_n']} 只持仓。每 30 分钟巡检 + 每 5 分钟止损盯梢已覆盖它。"}
        if op in ("/buy", "/sell") and len(parts) >= 4:
            code, name, price = parts[1], parts[2], parts[3]
            reason = " ".join(parts[4:])
            r = services.trade_add(cfg, action=("buy" if op == "/buy" else "sell"),
                                   code=code, name=name, price=price, reason=reason, source="slash")
            if not r.get("ok"):
                return {"text": f"记录失败：{r.get('error', '未知错误')}"}
            return {"text": f"已记录{'买入' if op == '/buy' else '卖出'}：{name} {code} @ {price}"
                            + (f"\n逻辑：{reason}" if reason else "")
                            + "\n（已计入交易数据库，可在左侧「交易记录」查看，或导出 CSV 供回测）"}
    except Exception as e:
        return {"text": f"命令执行失败: {e}", "error": True}
    return {"text": ("可用命令：\n"
                     "- `/scan 板块名 [st]` 四层漏斗扫描（加 st 参数则连ST股一起扫）\n"
                     "- `/score 代码1,代码2` 垃圾度打分\n"
                     "- `/check` 持仓+观察名单检查\n- `/flow` 板块资金流\n"
                     "- `/news [关键词]` 快讯(+密度)\n- `/quote 代码` 个股体检\n"
                     "- `/model` 查看本地模型建议（不调用Kimi）\n"
                     "- `/watch 代码 [理由]` 加观察\n- `/del 代码` 移除\n"
                     "- `/pos 代码 名称 买价 日期 止损线 [逻辑]` 登记持仓")}


# ---------- 路由 ----------
@app.route("/")
def index():
    return send_from_directory(WEB, "index.html")


@app.route("/<path:fname>")
def static_files(fname):
    return send_from_directory(WEB, fname)


@app.route("/api/chat", methods=["POST"])
def chat():
    msg = (request.json or {}).get("message", "").strip()
    if not msg:
        return jsonify({"error": "空消息"}), 400
    cfg = get_cfg()
    if msg.startswith("/"):
        return jsonify({"reply": handle_slash(msg, cfg)["text"], "tools": []})
    tools_seen = []

    def on_tool(name, args):
        tools_seen.append({"name": name, "args": args})
        notifier.broadcast({"type": "tool", "name": name, "args": args,
                            "time": dt.datetime.now().strftime("%H:%M:%S")})

    try:
        with _brain_lock:
            body = request.get_json() or {}
            from conversation_intent import plan_work
            if plan_work(msg, drafting=body.get('mode') == 'next_day_plan'):
                draft_id = body.get('draft_id')
                if not draft_id and not any(word in msg for word in ('制定', '生成', '重选')):
                    previous = assistant_workflow.latest('plan')
                    draft_id = previous['id'] if previous else None
                result = assistant_workflow.plan_chat(cfg, msg, draft_id)
                if result['draft'].get('awaiting_count'):
                    return jsonify({**result, 'tools': []})
                from assistant_runtime import gate
                try:
                    result['graph'] = gate('plan', result['draft'])
                except ValueError as exc:
                    result['graph'] = {'ok': False, 'error': str(exc)}
                return jsonify({**result, 'tools': []})
            reply = get_brain().chat(msg, on_tool=on_tool)
    except Exception as e:
        reply = f"对话失败: {e}" if isinstance(e, ValueError) else f"对话失败（{type(e).__name__}）"
    return jsonify({"reply": reply, "tools": tools_seen})


@app.route("/api/events")
def events():
    q = notifier.subscribe()

    def stream():
        try:
            yield "data: {\"type\": \"hello\"}\n\n"
            while True:
                try:
                    ev = q.get(timeout=25)
                    yield f"data: {json.dumps(ev, ensure_ascii=False, default=str)}\n\n"
                except queue.Empty:
                    yield ": ping\n\n"
        finally:
            notifier.unsubscribe(q)

    return Response(stream(), mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.route("/api/state")
def state():
    cfg = get_cfg()
    wl = services.wl_load(cfg)
    p = cfg.get("push", {})
    positions = wl.get("positions", [])
    watch = wl.get("watch", [])
    # 注入实时行情（持仓+观察共用一次 quotes 调用）
    codes = [p["code"] for p in positions] + [w["code"] for w in watch]
    quotes = {}
    if codes:
        try:
            from tools import market
            quotes = market.quotes(codes) or {}
        except Exception:
            quotes = {}
    def _live(item):
        q = quotes.get(item["code"], {})
        price = q.get("price")
        pct = q.get("pct")
        return {"live_price": float(price) if price not in (None, "") else None,
                "live_pct": float(pct) if pct not in (None, "") else None,
                "name": q.get("name") or item.get("name", "")}
    return jsonify({
        "positions": [{**p, **_live(p)} for p in positions],
        "watch": [{**w, **_live(w)} for w in watch],
        "market_open": notifier.market_open(dt.datetime.now()),
        "push": {"enabled": p.get("enabled", True), "provider": p.get("provider", "serverchan"),
                 "has_sendkey": bool(p.get("serverchan_sendkey", "").strip()),
                 "has_pushplus_token": bool(p.get("pushplus_token", "").strip()),
                 "interval_minutes": p.get("interval_minutes", 30),
                 "close_time": p.get("close_time", "15:05"),
                 "morning_report": p.get("morning_report", True),
                 "stop_watch": p.get("stop_watch", True),
                 "auction_check": p.get("auction_check", True)},
        "last_check_at": notifier_thread.last_check_at,
        "last_report": notifier_thread.last_report,
        "has_tavily": bool((cfg.get("tavily") or {}).get("api_key", "").strip()),
        "plan_ui": plan_bridge.plan_ui_visible(cfg),
    })


@app.route("/api/config", methods=["POST"])
def update_config():
    body = request.json or {}
    cfg_path = paths.data_file("config.yaml")
    with _cfg_lock:
        with open(cfg_path, encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        push = cfg.setdefault("push", {})
        for k in ("enabled", "provider", "serverchan_sendkey", "pushplus_token",
                  "interval_minutes", "close_time", "in_app_notify",
                  "morning_report", "morning_time", "stop_watch",
                  "auction_check", "auction_time"):
            if k in body:
                push[k] = body[k]
        kimi = body.get("kimi") or {}
        if kimi.get("api_key"):
            cfg["kimi"]["api_key"] = kimi["api_key"].strip()
            global _brain
            with _brain_lock:
                _brain = None  # 换 key 后重建
        if "tavily_key" in body:
            cfg.setdefault("tavily", {})["api_key"] = (body["tavily_key"] or "").strip()
        if body.get('deepseek_key'):
            deepseek_api.save_key(body['deepseek_key'])
            # complete() reads the protected key each call. Do not wait for a
            # running chat while holding _cfg_lock: quote monitors need config.
        with open(cfg_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)
    return jsonify({"ok": True})


@app.route("/api/push/test", methods=["POST"])
def push_test():
    cfg = get_cfg()
    ok, msg = notifier.send_wechat(cfg, "Stargazer 推送测试",
                                   "这是一条测试消息。\n\n如果你在微信看到了它，说明推送配置成功。")
    return jsonify({"ok": ok, "msg": msg})


@app.route("/api/check", methods=["POST"])
def manual_check():
    text, msg = notifier_thread.run_once("manual")
    return jsonify({"report": text or "无信号", "push_msg": msg})


# ---------- T5 影子决策留痕（只记录不推荐；stargazer_ml/pullback_train） ----------
def _shadow_log():
    """惰性导入影子决策留痕模块（stargazer_ml/pullback_train/shadow_log.py）。

    路径从 paths 推导，不再硬编码 D:\\stargazer\\...，兼容打包后环境。
    """
    import sys, os
    ml_dir = os.path.join(os.path.dirname(paths.APP_DIR),
                          "stargazer_ml", "pullback_train")
    if not os.path.isdir(ml_dir):
        ml_dir = paths.resource(os.path.join("stargazer_ml", "pullback_train"))
    if ml_dir not in sys.path:
        sys.path.insert(0, ml_dir)
    import shadow_log
    return shadow_log


@app.route("/api/t5/action", methods=["POST"])
def api_t5_action():
    """人工录入候选×决策点上的操作：{sig_date, code, dp, action: buy|skip,
    reason?, fill_price?}。

    评审约束：①action 只允许 buy/skip，sell 是账户事件走 /api/trade/add；
    ②买入必须绑定具体 sig_date+dp（前端按钮所在行=记录行），不支持批量；
    ③buy 时 fill_price 缺省则服务端取点击瞬间实时价。
    """
    body = request.get_json(force=True) or {}
    action = str(body.get("action") or "skip")
    if action not in ("buy", "skip"):
        return jsonify({"ok": False,
                        "error": "action 只允许 buy/skip；卖出请用持仓卡卖出按钮（trades 体系）"}), 400
    try:
        sig_date, code = body["sig_date"], str(body["code"])
        dp = int(body["dp"])
    except (KeyError, TypeError, ValueError):
        return jsonify({"ok": False,
                        "error": "sig_date/code/dp 必填，dp 为决策点整数（如 1030）"}), 400
    fill_price = body.get("fill_price")
    if action == "buy" and fill_price in (None, ""):
        try:
            from tools import market
            fill_price = ((market.quotes([code]) or {}).get(code) or {}).get("price")
        except Exception:
            fill_price = None
    try:
        shadow_log = _shadow_log()
        shadow_log.record_action(
            intraday_monitor._db_path(get_cfg()),
            str(sig_date), code, dp, action,
            str(body.get("reason") or ""), fill_price=fill_price)
        return jsonify({"ok": True, "fill_price": fill_price})
    except Exception as exc:
        return jsonify({"ok": False, "error": f"{type(exc).__name__}: {exc}"})


@app.route("/api/t5/decisions")
def api_t5_decisions():
    """影子决策记录查询：?date=YYYYMMDD 或全部最近 200 条。"""
    shadow_log = _shadow_log()
    conn = shadow_log.connect(intraday_monitor._db_path(get_cfg()))
    try:
        date = request.args.get("date")
        if date:
            rows = conn.execute(
                "SELECT * FROM shadow_decisions WHERE sig_date=? "
                "ORDER BY p_opp DESC",
                (int(date) if str(date).isdigit() else str(date),)).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM shadow_decisions ORDER BY sig_date DESC, p_opp DESC "
                "LIMIT 200").fetchall()
        cols = [d[0] for d in conn.execute(
            "SELECT * FROM shadow_decisions LIMIT 0").description]
    finally:
        conn.close()
    return jsonify({"ok": True, "rows": [dict(zip(cols, r)) for r in rows]})


# ---------- 本地 v5.9 建议（完全不走 Kimi） ----------
@app.route("/api/model/state")
def api_model_state():
    result = ml_bridge.status(get_cfg())
    result["last_snapshot_at"] = model_monitor_thread.last_snapshot_at
    result["monitor_error"] = model_monitor_thread.last_error
    return jsonify(result)


@app.route("/api/model/advice")
def api_model_advice():
    return jsonify(ml_bridge.advice(get_cfg(), limit=request.args.get("limit", 20)))


@app.route("/api/model/run", methods=["POST"])
def api_model_run():
    return jsonify({'ok': False, 'error': '旧选股入口已停用，请使用星币侍从；历史结果保留'}), 410


@app.route("/api/model/refresh", methods=["POST"])
def api_model_refresh():
    return jsonify({'ok': False, 'error': '旧选股入口已停用，请使用星币侍从'}), 410


# ---------- 计划盘后总结（stargazer_plan：settle 子进程封装，plan_bridge） ----------
@app.route("/api/plan/settle", methods=["POST"])
def api_plan_settle():
    try:
        return jsonify(plan_bridge.settle_latest(get_cfg()))
    except Exception as exc:
        return jsonify({"ok": False, "error": f"{type(exc).__name__}: {exc}"})


# ---------- 交易计划页（双名单 + 生成计划 + 冻结导出；本机专属，T5 设计 §5） ----------
# 页面可见性见 plan_bridge.plan_ui_visible（plan.enabled=false 或全链不存在时
# 前端隐藏侧栏入口）；以下为页面全部后端接口，监控机上不可见亦不注册业务影响。
@app.route("/api/plan/dual_list/state")
def api_plan_dual_state():
    return jsonify(plan_bridge.dual_list_state(get_cfg()))


@app.route("/api/plan/dual_list/run", methods=["POST"])
def api_plan_dual_run():
    body = request.json or {}
    try:
        return jsonify(plan_bridge.dual_list_run(get_cfg(), body.get("date")))
    except Exception as exc:
        return jsonify({"ok": False, "error": f"{type(exc).__name__}: {exc}"})


@app.route("/api/plan/pool")
def api_plan_pool():
    return jsonify(plan_bridge.rule_pool(get_cfg()))


@app.route("/api/plan/draft", methods=["POST"])
def api_plan_draft():
    try:
        return jsonify(plan_bridge.create_draft_from_picks(get_cfg(),
                                                           request.json or {}))
    except Exception as exc:
        return jsonify({"ok": False, "error": f"{type(exc).__name__}: {exc}"})


@app.route("/api/plan/freeze_export", methods=["POST"])
def api_plan_freeze_export():
    body = request.json or {}
    try:
        return jsonify(plan_bridge.freeze_and_export(get_cfg(),
                                                     body.get("session_id"),
                                                     body.get("trading_date")))
    except Exception as exc:
        return jsonify({"ok": False, "error": f"{type(exc).__name__}: {exc}"})


# ---------- 星币侍从 Demo：独立观察，不触及旧监控 ----------
@app.route('/api/pentacles/state')
def api_pentacles_state():
    return jsonify(pentacles_demo.state())


@app.route('/api/pentacles/monitor', methods=['POST'])
def api_pentacles_monitor():
    try:
        body = request.get_json(silent=True)
        if not isinstance(body, dict):
            raise ValueError('请求必须为 JSON 对象')
        return jsonify(pentacles_demo.set_enabled(body.get('enabled')))
    except ValueError as exc:
        return jsonify({'ok': False, 'error': str(exc)}), 400


@app.route('/api/pentacles/journal')
def api_pentacles_journal():
    try:
        return jsonify(pentacles_demo.monitor.journal.report(request.args.get('day')))
    except Exception as exc:
        return jsonify({'ok': False, 'error': str(exc)}), 400


@app.route('/api/pentacles/run', methods=['POST'])
def api_pentacles_run():
    try:
        return jsonify(pentacles_demo.run(get_cfg()))
    except Exception as exc:
        return jsonify({'ok': False, 'error': str(exc)}), 400


# ---------- 影子建议（Stargazer 1.0 接线③：只记录不影响；失败绝不拖垮主进程） ----------
@app.route('/api/shadow_evidence')
def api_shadow_evidence():
    import shadow_ui
    code = (request.args.get('code') or '').strip()
    date = (request.args.get('date') or '').strip()
    if not code and not date:
        return jsonify(shadow_ui.evidence_catalog())
    if not code or not date:
        return jsonify({'status': 'UNSCORABLE', 'reason': 'code/date 必填'}), 400
    try:
        return jsonify(shadow_ui.get_shadow_evidence(code, date))
    except Exception as exc:
        return jsonify({'status': 'ERROR', 'reason': type(exc).__name__})


@app.route('/api/shadow_advice')
def api_shadow_advice():
    """GET ?code=XXXXXX&date=YYYYMMDD → {status: OK|UNSCORABLE|ERROR, reason,
    advice: render_advice 字段或 null, journal_ref}。

    证据包由数据线产出（dist/data/shadow_packs.jsonl），本接口不算特征；
    评分只走 shadow_journal.record_event(trigger='MANUAL') 入影子日记。
    持仓中的候选用 POSITIONED 场景，其余按 UNPOSITIONED。
    """
    import shadow_ui
    code = (request.args.get('code') or '').strip()
    date = (request.args.get('date') or '').strip()
    if not code or not date:
        return jsonify({'status': 'UNSCORABLE', 'reason': 'code/date 必填',
                        'advice': None, 'journal_ref': None}), 400
    if not deepseek_api.status().get('configured'):
        return jsonify({'status': 'ERROR',
                        'reason': '本次建议不可用（DeepSeek 未配置）',
                        'advice': None, 'journal_ref': None})
    try:
        held = {str(p.get('code')) for p in
                (services.wl_load(get_cfg()).get('positions') or [])}
    except Exception:
        held = set()
    try:
        return jsonify(shadow_ui.get_shadow_advice(
            code, date, scenario=('POSITIONED' if code in held
                                  else 'UNPOSITIONED')))
    except Exception as exc:  # 双保险：任何异常不外抛、不 500
        return jsonify({'status': 'ERROR',
                        'reason': f'本次建议不可用（{type(exc).__name__}）',
                        'advice': None, 'journal_ref': None})


@app.route('/api/assistant/state')
def api_assistant_state():
    try:
        return jsonify({**assistant_workflow.state(), 'llm': deepseek_api.status()})
    except Exception as exc:
        return jsonify({'ok': False, 'error': f'助手记录读取失败（{type(exc).__name__}）'}), 400


@app.route('/api/assistant/plan/confirm', methods=['POST'])
def api_assistant_plan_confirm():
    try:
        body = request.get_json() or {}
        if body.get('confirmed') is not True:
            raise ValueError('必须人工确认计划')
        from assistant_runtime import gate
        draft = {'id': body.get('id'), **assistant_workflow.load_draft('plan', body.get('id'))}
        gate('plan', draft, approved=True)
        result = assistant_workflow.confirm_plan(body.get('id'), body.get('confirmed'))
        pentacles_demo.set_enabled(True)
        return jsonify(result)
    except Exception as exc:
        return jsonify({'ok': False, 'error': str(exc) if isinstance(exc, ValueError)
                        else f'计划确认失败（{type(exc).__name__}）'}), 400


@app.route('/api/assistant/review', methods=['POST'])
def api_assistant_review():
    try:
        body = request.get_json() or {}
        return jsonify(assistant_workflow.review_start(get_cfg(), int(body['day'])))
    except Exception as exc:
        return jsonify({'ok': False, 'error': str(exc) if isinstance(exc, ValueError)
                        else '复盘日期无效'}), 400


@app.route('/api/assistant/review/confirm', methods=['POST'])
def api_assistant_review_confirm():
    try:
        body = request.get_json() or {}
        if body.get('confirmed') is not True:
            raise ValueError('必须人工确认学习记录')
        from assistant_runtime import gate
        draft = {'id': body.get('id'), **assistant_workflow.load_draft('review', body.get('id'))}
        gate('review', draft, approved=True)
        return jsonify(assistant_workflow.confirm_review(body.get('id'), body.get('confirmed')))
    except Exception as exc:
        return jsonify({'ok': False, 'error': str(exc) if isinstance(exc, ValueError)
                        else f'复盘保存失败（{type(exc).__name__}）'}), 400


@app.route('/api/shell/state')
def api_shell_state():
    from model_slots import slots
    try:
        value = assistant_workflow.snapshot()
    except (ValueError, OSError):
        value = {'schema': 'UNAVAILABLE', 'rows': []}
    import joint_risk_bridge
    import plan_evidence
    return jsonify({**slots(value), 'display': plan_evidence.status(value),
                    'joint_shadow': joint_risk_bridge.state(get_cfg())})


@app.route('/api/model/detail')
def api_model_detail():
    import plan_evidence
    try:
        return jsonify(plan_evidence.model_view(request.args.get('model', '')))
    except ValueError as exc:
        return jsonify({'error': str(exc)}), 400


@app.route('/api/risk/outlook', methods=['POST'])
def api_risk_outlook():
    import plan_evidence
    import risk_outlook_v1
    view=plan_evidence.model_view('risk')
    if not view.get('items') or not view.get('data_date'):
        return jsonify({'status':'NO_PLAN','reason':'尚无绑定计划证据'}),400
    old=view.get('outlook') or {}
    if old.get('status') not in (None,'NOT_GENERATED'):
        return jsonify(old)
    cutoff=int(view['data_date'])*1000000+150000
    return jsonify(risk_outlook_v1.queue(get_cfg(),view,cutoff,plan_id=risk_outlook_v1.view_id(view)))


@app.route('/api/plan_boards')
def api_plan_boards():
    import plan_board_monitor
    try:
        return jsonify(plan_board_monitor.state())
    except Exception:
        return jsonify({'status': 'UNAVAILABLE', 'rows': []})


@app.route('/api/joint-shadow/state')
def api_joint_shadow_state():
    import joint_risk_bridge
    return jsonify(joint_risk_bridge.state(get_cfg()))


@app.route('/api/postclose-risk/state')
def api_postclose_risk_state():
    import postclose_risk_runtime
    return jsonify(postclose_risk_runtime.state())


@app.route('/api/joint-shadow/run', methods=['POST'])
def api_joint_shadow_run():
    import joint_risk_bridge
    try:
        # Data paths, mode and enabling belong to local config, not the POST body.
        return jsonify(joint_risk_bridge.run(get_cfg(), pentacles_demo.monitoring_plan()))
    except (ValueError, OSError) as exc:
        return jsonify({'ok': False, 'status': 'REQUEST_REFUSED', 'reason': type(exc).__name__}), 400


@app.route('/api/chat/session', methods=['GET', 'POST'])
def api_chat_session():
    global _brain
    with _brain_lock:
        if request.method == 'POST':
            _brain = Brain(fresh=True)
        brain = get_brain()
        return jsonify({'ok': True, 'session': brain.archive.stem,
                        'messages': [m for m in brain.messages[-20:]
                                     if m.get('role') in ('user', 'assistant') and m.get('content')]})


# ---------- 信息面板（Stargazer 1.0 UI 框架：Corvus 信息模型，缺席=未接入） ----------
# 数据契约：信息流后续推送 JSON 到 dist/data/info_feed.jsonl（每行一条，见
# /api/info_feed 路由注释）。文件缺席/为空 → connected=False → 面板显示"信息模型未接入"。
INFO_FEED_FILE = paths.data_file('dist/data/info_feed.jsonl', seed_from_bundle=False)
INFO_FEED_MAX_ITEMS = 20
INFO_FEED_FIELDS = ('ts', 'source', 'kind', 'title', 'summary', 'tags', 'refs')


@app.route('/api/info_feed')
def api_info_feed():
    """最新信息流条目。契约（每行一个 JSON）：
    {ts: 14位时间戳, source: 来源标识, kind: 类别(announcement/news/risk/...),
     title: 标题, summary: 摘要, tags: [..], refs: [代码/证据编号]}
    """
    items, bad_lines = [], 0
    try:
        with open(INFO_FEED_FILE, encoding='utf-8') as f:
            lines = f.read().splitlines()
    except OSError:
        lines = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            bad_lines += 1
            continue
        if isinstance(row, dict) and row.get('ts') and row.get('title'):
            items.append({k: row.get(k) for k in INFO_FEED_FIELDS})
    items = items[-INFO_FEED_MAX_ITEMS:][::-1]
    return jsonify({'connected': bool(items), 'schema': 'STARGAZER_INFO_FEED_V1',
                    'items': items, 'bad_lines': bad_lines,
                    'asof': items[0]['ts'] if items else None})


# ---------- 交易计划（Stargazer 1.0 UI 框架：server 校验后落盘 dist/data/trade_plan.json） ----------
@app.route('/api/trade_plan')
def api_trade_plan():
    """当前交易计划；缺席 → {plan: None, status: 'NO_PLAN'}（UI 显示"暂无计划"）。"""
    import trade_plan_v1 as tp
    try:
        plan = tp.load_plan()
        return jsonify({'ok': True, 'status': 'OK' if plan else 'NO_PLAN',
                        'plan': plan, 'errors': tp.validate_plan(plan) if plan else []})
    except Exception as exc:
        return jsonify({'ok': False, 'status': 'ERROR',
                        'reason': f'{type(exc).__name__}: {exc}', 'plan': None})


@app.route('/api/trade_plan', methods=['POST'])
def api_trade_plan_save():
    """写入正式计划（用户人工确认后）。body=契约 JSON，校验通过才落盘。"""
    import trade_plan_v1 as tp
    body = request.get_json(silent=True)
    errs = tp.validate_plan(body)
    if errs:
        return jsonify({'ok': False, 'status': 'REJECTED',
                        'reason': '计划契约校验失败', 'errors': errs[:8]}), 400
    try:
        path = tp.save_plan(body)
        return jsonify({'ok': True, 'status': 'SAVED', 'path': str(path)})
    except Exception as exc:
        return jsonify({'ok': False, 'status': 'ERROR',
                        'reason': f'{type(exc).__name__}: {exc}'})


# 计划触发推送去重（内存态：进程内每股每计划日只推一次；框架级，模型接入后保留）
_PLAN_PUSHED: set = set()


@app.route('/api/plan_check')
def api_plan_check():
    """计划监控钩子（框架先行）：?code=&price=[&ts=14位] → 结构化状态。

    当前只按计划 JSON 规则判断（NEAR/TRIGGERED/EXPIRED/...）；
    # 模型决策接入点：数据链验收后接 decision_layer（见 trade_plan_v1.plan_check）。
    规则触发事件沿现有通道推送（微信未开启时仅记录）；推送失败不影响返回。
    """
    import trade_plan_v1 as tp
    code = (request.args.get('code') or '').strip()
    price = request.args.get('price')
    ts = (request.args.get('ts') or '').strip() or None
    if not code:
        return jsonify({'status': 'ERROR', 'reason': 'code 必填'}), 400
    try:
        result = tp.plan_check(code, price, ts)
    except Exception as exc:
        return jsonify({'status': 'ERROR', 'reason': f'{type(exc).__name__}'})
    if result.get('status') == 'TRIGGERED':
        key = (code, str((result.get('stock') or {}).get('valid_until', '')))
        if key not in _PLAN_PUSHED:
            _PLAN_PUSHED.add(key)
            try:
                cfg = get_cfg()
                name = (result.get('stock') or {}).get('name') or ''
                zone = result.get('buy_zone') or []
                ok, msg = notifier.send_wechat(
                    cfg, f'计划触发提醒 {name} {code}',
                    f'现价 {price} 进入计划买入区 {zone[0]}–{zone[1]}。\n'
                    f'规则触发，模型决策待接入；买卖由你决定。')
                result['pushed'] = ok
                result['push_msg'] = msg
            except Exception as exc:
                result['pushed'] = False
                result['push_msg'] = f'{type(exc).__name__}'
    return jsonify(result)


@app.route('/api/index_freshness')
def api_index_freshness():
    """指数新鲜度（UI 契约②）：由 index_minutes_v1.get_index_freshness() 提供；
    数据链未接入（模块缺席/异常）→ available=False，前端显示"数据过期"。"""
    try:
        import index_minutes_v1
        value = index_minutes_v1.get_index_freshness()
        if not isinstance(value, dict):
            raise ValueError('FRESHNESS_NOT_DICT')
        return jsonify({'available': True, **value})
    except Exception:
        return jsonify({'available': False, 'reason': 'INDEX_FRESHNESS_NOT_CONNECTED'})


# ---------- 信息面板（路线 #13：SHADOW 展示页，无 LLM，无决策权） ----------
# 纯展示：计划票行情/公告/规则标注/mature p 值 + 申万板块实时 + 隔夜美股金油
# + 快讯资讯（转述源未经核实，如实标注）。采集失败降级为缺口，不拖垮主程序。
INFO_STATE = {"payload": None, "collected_at": 0.0}
INFO_CACHE_TTL = 45
INFO_AUTO_INTERVAL = 1800  # 30min


def _info_collect(force: bool = False) -> dict:
    now = time.time()
    if (not force and INFO_STATE["payload"] is not None
            and now - INFO_STATE["collected_at"] < INFO_CACHE_TTL):
        return INFO_STATE["payload"]
    import info_panel
    try:
        payload = info_panel.collect()
        info_panel.save_snapshot(payload)
    except Exception:
        # 冻结排障：traceback 落 exe 旁 data/info_panel/error.log（响应里同时携带）
        import traceback
        tb = traceback.format_exc()
        try:
            d = info_panel.DATA / "error.log"
            d.parent.mkdir(parents=True, exist_ok=True)
            d.write_text(tb, encoding="utf-8")
        except Exception:
            pass
        raise RuntimeError(f"collect failed: {tb[-600:]}")
    INFO_STATE["payload"] = payload
    INFO_STATE["collected_at"] = now
    return payload


def _info_auto_loop():
    while True:
        time.sleep(60)
        h = dt.datetime.now().hour
        m = dt.datetime.now().minute
        in_window = (h == 8 and m >= 30) or (9 <= h < 15) or (h == 15 and m <= 35)
        if not in_window:
            continue
        if time.time() - INFO_STATE["collected_at"] >= INFO_AUTO_INTERVAL:
            try:
                _info_collect(force=True)
            except Exception:
                pass  # 缺口在下次手动刷新可见


@app.route("/info")
def info_page():
    return send_from_directory(WEB, "info.html")


@app.route("/api/info/snapshot")
def api_info_snapshot():
    try:
        p = dict(_info_collect())
        p["cache_age_s"] = round(time.time() - INFO_STATE["collected_at"], 1)
        p["auto_interval_min"] = INFO_AUTO_INTERVAL // 60
        return jsonify(p)
    except Exception as exc:
        import traceback
        return jsonify({"error": f"{type(exc).__name__}: {exc}",
                        "traceback": traceback.format_exc()[-800:]}), 500


@app.route("/api/info/refresh", methods=["POST"])
def api_info_refresh():
    try:
        return jsonify(_info_collect(force=True))
    except Exception as exc:
        return jsonify({"error": f"{type(exc).__name__}: {exc}"}), 500


@app.route("/api/info/note", methods=["POST"])
def api_info_note():
    text = str((request.json or {}).get("text") or "").strip()
    if not text:
        return jsonify({"error": "empty"}), 400
    try:
        import info_panel
        snap = (INFO_STATE["payload"] or {}).get("snapshot_ts", "N/A")
        f = info_panel.append_note(text, snap)
        return jsonify({"saved": str(f)})
    except Exception as exc:
        return jsonify({"error": f"{type(exc).__name__}: {exc}"}), 500


# ---------- 强势中继低吸（绿盘低吸 v2） ----------
# 2026-09-23（T5 设计 §6）：14:30 尾盘扫描已删除——/scan、/scan_state、/state
# 三个路由下线（run_afternoon_scan 在 sector_pullback.py 保留为死代码，无路由可达）；
# 实时监控板块设置（user_sectors）与盘前助手保留。


@app.route("/api/sector_pullback/intraday", methods=["POST"])
def api_sector_intraday():
    body = request.json or {}
    now = None
    if body.get("now"):
        try:
            now = dt.datetime.strptime(body["now"], "%H:%M")
        except Exception:
            pass
    return jsonify(_sector_pullback().evaluate_intraday(get_cfg(), now=now))


@app.route("/api/sector_pullback/intraday_scan", methods=["POST"])
def api_sector_intraday_scan():
    """盘中主动扫描所选板块内新出现的符合要求个股（测试/手动触发用）。"""
    return jsonify(_sector_pullback().scan_intraday_new(get_cfg()))


@app.route("/api/sector_pullback/user_sectors", methods=["GET", "POST"])
def api_user_sectors():
    """实时监控板块：GET 返回已保存列表+映射；POST 保存（最多20个同花顺概念名）。"""
    cfg = get_cfg()
    if request.method == "POST":
        body = request.json or {}
        names = body.get("names") or []
        if not isinstance(names, list):
            names = [names]
        return jsonify(_sector_pullback().save_user_sectors(cfg, names))
    mapped, missing = _sector_pullback().resolve_user_boards(cfg)
    return jsonify({"names": _sector_pullback().load_user_sectors(cfg),
                    "mapped": mapped, "missing": missing})


@app.route("/api/premarket/state")
def api_premarket_state():
    return jsonify(_sector_pullback().build_premarket_state(get_cfg()))


@app.route("/api/premarket/save", methods=["POST"])
def api_premarket_save():
    return jsonify({'ok': False, 'error': '盘前手填助手已停用；持仓风控与实时板块监控保留'}), 410


# ---------- 交易记录（对话/命令写入的成交，供回测） ----------
@app.route("/api/trades")
def api_trades():
    return jsonify({"trades": services.trade_list(get_cfg())})


@app.route("/api/watch/add", methods=["POST"])
def api_watch_add():
    b = request.json or {}
    r = services.wl_add_watch(get_cfg(), code=b.get("code", ""), reason=b.get("reason", ""),
                              target_buy=b.get("target_buy"), stop_loss=b.get("stop_loss"),
                              strategy=b.get("strategy"), source_date=b.get("source_date"))
    return jsonify(r)


@app.route("/api/watch/update", methods=["POST"])
def api_watch_update():
    b = request.json or {}
    code = b.get("code", "")
    if not code:
        return jsonify({"ok": False, "msg": "code 必填"})
    r = services.wl_update_watch(get_cfg(), code,
                                 target_buy=b.get("target_buy"), stop_loss=b.get("stop_loss"),
                                 strategy=b.get("strategy"), reason=b.get("reason"))
    return jsonify(r)


@app.route("/api/positions/add", methods=["POST"])
def api_positions_add():
    b = request.json or {}
    code, name = b.get("code", ""), b.get("name", "")
    entry = b.get("entry")
    entry_date = b.get("entry_date") or dt.date.today().isoformat()
    low250 = b.get("low250")
    if not code or not name or entry in (None, "") or low250 in (None, ""):
        return jsonify({"ok": False, "msg": "代码/名称/买入价/止损线 必填"})
    try:
        entry = float(entry); low250 = float(low250)
    except (TypeError, ValueError):
        return jsonify({"ok": False, "msg": "买入价/止损线必须是数字"})
    r = services.wl_add_position(get_cfg(), code=code, name=name, entry=entry,
                                 entry_date=entry_date, low250=low250,
                                 logic=b.get("logic", ""))
    return jsonify(r)


@app.route("/api/positions/update", methods=["POST"])
def api_positions_update():
    b = request.json or {}
    code = b.get("code", "")
    if not code:
        return jsonify({"ok": False, "msg": "code 必填"})
    r = services.wl_update_position(get_cfg(), code, name=b.get("name"), entry=b.get("entry"),
                                    entry_date=b.get("entry_date"), low250=b.get("low250"),
                                    logic=b.get("logic"))
    return jsonify(r)


@app.route("/api/trade/add", methods=["POST"])
def api_trade_add():
    b = request.json or {}
    r = services.trade_add(get_cfg(), action=b.get("action"), code=b.get("code"),
                           name=b.get("name"), price=b.get("price"), shares=b.get("shares"),
                           trade_date=b.get("trade_date"), reason=b.get("reason", ""),
                           source=b.get("source", "manual"), raw=b.get("raw", ""))
    # 评审第1点：卖出是账户层事件（trades 体系），决策表不写 action=sell；
    # 仅 best-effort 在影子决策表标记最近一笔未平仓买入的 exit_*，失败不影响成交。
    if r.get("ok") and str(b.get("action") or "").lower() == "sell":
        try:
            _shadow_log().record_exit(
                intraday_monitor._db_path(get_cfg()),
                str(b.get("code") or ""), r.get("price"))
        except Exception:
            pass
    return jsonify(r)


@app.route("/api/trade/export")
def api_trade_export():
    p = services.trade_export_csv(get_cfg())
    return send_from_directory(os.path.dirname(p), os.path.basename(p), as_attachment=True)


def run_server(port=17890):
    app.run(host="127.0.0.1", port=port, threaded=True, use_reloader=False)


def start(port=17890):
    """启动后端+巡检线程，返回实际端口。"""
    import socket
    for p in range(port, port + 20):
        s = socket.socket()
        try:
            s.bind(("127.0.0.1", p))
            s.close()
            port = p
            break
        except OSError:
            s.close()
    notifier_thread.start()
    model_monitor_thread.start()
    pentacles_demo.monitor.start(get_cfg, pentacles_demo.monitoring_plan)
    import index_minutes_v1
    index_minutes_v1.start_source_sync()
    import postclose_risk_runtime
    postclose_risk_runtime.start()
    threading.Thread(target=_info_auto_loop, daemon=True).start()
    t = threading.Thread(target=run_server, args=(port,), daemon=True)
    t.start()
    return port
