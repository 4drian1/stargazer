"""Minute snapshots for human-in-the-loop model validation."""
from __future__ import annotations

import datetime as dt
import json
import os
import sqlite3
import threading
import time

import ml_bridge
import notifier
import paths

# 延迟导入：sector_pullback（→numpy+pandas）和 tools.market（→pandas）
# 是重链，只在首次巡检时才加载，避免拖慢启动


def _db_path(cfg):
    value = (cfg.get("model") or {}).get("intraday_db", "data/model_state.sqlite")
    return value if os.path.isabs(value) else paths.data_file(value, seed_from_bundle=False)


def _conn(cfg):
    path = _db_path(cfg)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    conn = sqlite3.connect(path, timeout=20)
    conn.execute("""CREATE TABLE IF NOT EXISTS minute_snapshots (
        id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT, trade_date TEXT, minute TEXT,
        code TEXT, name TEXT, strategy TEXT, data_date TEXT, advisory_class TEXT,
        advisory_rank INTEGER, advisory_score REAL, live_price REAL, live_pct REAL,
        live_open REAL, live_high REAL, live_low REAL, amount_wan REAL,
        action_hint TEXT, reason TEXT, raw_json TEXT,
        UNIQUE(trade_date, minute, code, strategy))""")
    conn.execute("""CREATE TABLE IF NOT EXISTS trade_context (
        id INTEGER PRIMARY KEY AUTOINCREMENT, trade_id INTEGER, ts TEXT, code TEXT,
        action TEXT, price REAL, nearest_snapshot_id INTEGER, context_json TEXT)""")
    return conn


def _g0_gate_status(row, gate_cfg):
    """根据模型输出的 T+N 预期收益/风险，估算胜率与收益/回撤比，并判断是否通过门控。"""
    offset = int(row.get("recommended_entry_offset") or 1)
    exp_ret = float(row.get(f"expected_return_t{offset}") or 0)
    risk = float(row.get(f"risk_probability_t{offset}") or 1)
    utility = float(row.get("entry_utility") or 0)
    trigger = float(row.get("buy_trigger_pct") or -3)
    win_rate = max(0.0, 1.0 - risk)
    # 收益/回撤比：模型预期收益 / 触发买入所需承受的回撤幅度（绝对值）
    potential_loss = abs(trigger) / 100.0
    reward_risk = exp_ret / potential_loss if potential_loss > 0 else float("inf")
    ok = (
        utility >= gate_cfg.get("min_entry_utility", 0.0)
        and win_rate >= gate_cfg.get("min_win_rate", 0.0)
        and reward_risk >= gate_cfg.get("min_payoff_ratio", 0.0)
        and risk <= gate_cfg.get("max_risk_probability", 1.0)
    )
    return ok, win_rate, reward_risk, exp_ret, risk, offset, utility, trigger


def _hint(row, quote, position, cfg):
    pct = float(quote.get("pct") or 0)
    price = float(quote.get("price") or 0)
    gate_cfg = (cfg.get("model") or {}).get("green_g0_gate", {})
    if position and price <= float(position.get("low250") or 0):
        return "STRUCTURE_STOP_ALERT", "盘中价格触及已登记结构止损线，等待人工确认执行"
    if row.get("strategy") == "t5_shadow":
        # T5 冻结双头影子评分：只提示人工复核，永不自动买入
        if row.get("ood"):
            return "OUT_OF_DISTRIBUTION", (
                f"10日涨幅{float(row.get('runup_10d') or 0):.0%}等高热指标超出训练分布"
                f"（P99.5阈值，命中{int(row.get('ood_hits') or 0)}项）；"
                "模型评分可信度低，仅观察，不生成买入建议")
        pr, po = float(row.get("P_risk") or 0), float(row.get("P_opp") or 0)
        quad = row.get("quadrant") or "REJECT"
        base_note = (
            f"P(接飞刀){pr:.0%}；P(先止盈){po:.0%}；四象限={quad}；"
            f"10日涨幅{float(row.get('runup_10d') or 0):.0%}；"
            f"RSI{float(row.get('rsi14') or 0):.0f}；"
            f"5日换手{float(row.get('turnover_sum_5d') or 0):.0f}%"
        )
        notes = {
            "FOCUS": ("T5_FOCUS",
                      f"{base_note}；低风险高机会(FOCUS)，仅提示人工复核，不自动买入"),
            "WAIT": ("T5_WAIT", f"{base_note}；机会存在但风险未解除，继续等待"),
            "WATCH": ("T5_WATCH", f"{base_note}；风险低但修复不足，观察"),
        }
        if quad in notes:
            return notes[quad]
        return "T5_REJECT", f"{base_note}；高风险低机会，否决"
    if row.get("strategy") == "green_g0":
        timing = int(row.get("recommended_entry_offset") or 1)
        ok, win_rate, payoff, exp_ret, risk, offset, utility, trigger = _g0_gate_status(row, gate_cfg)
        loss_pct = abs(trigger)
        gain_pct = exp_ret * 100
        # 实时买入建议统一展示：触发档、T+N、胜率、收益/回撤比、风险概率、效用
        base_note = (
            f"回撤档{trigger:.1f}%（T+{timing}）；"
            f"胜率{win_rate:.1%}；"
            f"预期收益{gain_pct:.1f}% vs 触发回撤{loss_pct:.1f}%（收益/回撤比{payoff:.2f}）；"
            f"风险概率{risk:.1%}；效用{utility:.3f}"
        )
        if pct <= min(-9.0, trigger - 2.0):
            return "FALLING_KNIFE_RISK", (
                f"{base_note}；已明显跌穿，先检查板块、龙头和承接，禁止自动抄底")
        if pct <= trigger:
            if not ok:
                return "G0_GATE_NOT_MET", (
                    f"{base_note}；门控未通过，只记录，不建议据此买入")
            return "BUY_REVIEW_TRIGGERED", (
                f"{base_note}；已触发，仅提示人工复核，不自动买入")
        return "WAIT_FOR_DIP", f"{base_note}；尚未触发，继续观察"
    if pct >= 5:
        return "DO_NOT_CHASE", "涨幅已超过5%，只观察不追高"
    if pct <= -5:
        return "RISK_REVIEW", "日内跌幅超过5%，先检查板块、龙头与承接"
    if row.get("advisory_class") == "REVIEW_TOP10":
        return "REVIEW_ENTRY", "模型Top10；结合板块、龙头和盘口承接人工复核买点"
    return "WATCH", "模型候选，继续观察"


def record_trade_context(cfg, trade_id, trade):
    conn = _conn(cfg)
    snapshot = conn.execute(
        "SELECT id, raw_json FROM minute_snapshots WHERE code=? ORDER BY id DESC LIMIT 1",
        (str(trade.get("code") or "")[-6:],),
    ).fetchone()
    context = json.loads(snapshot[1]) if snapshot and snapshot[1] else {}
    conn.execute(
        "INSERT INTO trade_context (trade_id,ts,code,action,price,nearest_snapshot_id,context_json) "
        "VALUES (?,?,?,?,?,?,?)",
        (trade_id, dt.datetime.now().isoformat(timespec="seconds"), str(trade.get("code") or "")[-6:],
         trade.get("action"), trade.get("price"), snapshot[0] if snapshot else None,
         json.dumps(context, ensure_ascii=False)),
    )
    conn.commit(); conn.close()


class IntradayMonitor(threading.Thread):
    def __init__(self, get_cfg):
        super().__init__(daemon=True, name="model-minute-monitor")
        self.get_cfg = get_cfg
        self.last_snapshot_at = None
        self.last_error = None
        self._sector_checked = set()
        # 冻结计划规则推送去重（按 交易日|code|rule_type，早盘批量失效合并一条）
        self._plan_day = None
        self._plan_pushed = set()
        self._plan_last = 0.0
        # 心跳与开关状态留痕（2026-09-15：线程静默死亡无日志的教训）
        self._hb_n = 0
        self._hb_start = None
        self._hb_last = dt.datetime.now()
        self._last_enabled_state = True
        # 日终汇总（T5 设计 §7）：今日已跑的轮数 + 已汇总日；收盘后强制一条
        self._rounds = 0
        self._rounds_day = None
        self._summary_day = None
        # W2d：每日校验盘后准备清单并自动补缺；未完成有界续查（15分钟），
        # 全部就绪后才结束当日检查
        self._prepared_done_day = None
        self._prepared_last_at = 0.0

    def snapshot_once(self):
        cfg = self.get_cfg()
        from tools import market
        import services
        payload = ({} if cfg.get('stable_shell') else
                   ml_bridge.advice(cfg, limit=(cfg.get("model") or {}).get("monitor_top_n", 20)))
        model_rows = ((payload.get("horizontal") or [])
                      + (payload.get("g0_shadow") or [])
                      + (payload.get("t5_shadow") or []))
        wl = services.wl_load(cfg)
        positions = {str(p.get("code"))[-6:]: p for p in services.active_positions(cfg, wl.get("positions", []))}
        watches = {str(w.get("code"))[-6:]: w for w in wl.get("watch", [])
                   if w.get("target_buy") or w.get("stop_loss")}
        # 合并需要盯盘的代码：模型候选 + 持仓 + 带目标价的观察名单
        model_codes = [str(r.get("code") or r.get("symbol") or "")[-6:] for r in model_rows]
        codes = list(dict.fromkeys(
            [c for c in model_codes if len(c) == 6]
            + [c for c in positions if len(c) == 6]
            + [c for c in watches if len(c) == 6]
        ))
        if not codes:
            return 0
        quotes = market.quotes(codes)
        now = dt.datetime.now(); minute = now.strftime("%H:%M")
        conn = _conn(cfg); inserted = 0
        # 1) 模型候选快照
        for row in model_rows:
            code = str(row.get("code") or row.get("symbol") or "")[-6:]
            quote = quotes.get(code)
            if not quote:
                continue
            hint, reason = _hint(row, quote, positions.get(code), cfg)
            raw = {"model": row, "quote": quote, "position": positions.get(code),
                   "action_hint": hint, "reason": reason}
            cursor = conn.execute(
                "INSERT OR IGNORE INTO minute_snapshots "
                "(ts,trade_date,minute,code,name,strategy,data_date,advisory_class,advisory_rank,"
                "advisory_score,live_price,live_pct,live_open,live_high,live_low,amount_wan,"
                "action_hint,reason,raw_json) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (now.isoformat(timespec="seconds"), now.date().isoformat(), minute, code,
                 quote.get("name"), row.get("strategy") or "horizontal",
                 (payload.get("status", {}).get("g0_data_date") if row.get("strategy") == "green_g0"
                  else str(row.get("date") or "") if row.get("strategy") == "t5_shadow"
                  else payload.get("status", {}).get("data_date")),
                 row.get("advisory_class"), row.get("advisory_rank"), row.get("advisory_score"),
                 quote.get("price"), quote.get("pct"), quote.get("open"), quote.get("high"),
                 quote.get("low"), quote.get("amount_wan"), hint, reason,
                 json.dumps(raw, ensure_ascii=False)),
            )
            inserted += int(cursor.rowcount > 0)
        # 2) 带目标价的观察名单快照
        for code, w in watches.items():
            quote = quotes.get(code)
            if not quote:
                continue
            price = float(quote.get("price") or 0)
            target = w.get("target_buy")
            stop = w.get("stop_loss")
            if target and price <= float(target):
                hint, reason = "WATCH_BUY_TRIGGER", f"目标买入价 {float(target):.2f} 已触及，现价 {price:.2f}，等待人工确认"
            elif stop and price <= float(stop):
                hint, reason = "WATCH_STOP_BROKEN", f"止损价 {float(stop):.2f} 已跌破，现价 {price:.2f}，建议放弃"
            else:
                # target/stop 均可为空（只登记了其一或仅观察）：None 不得进 float()
                t_s = f"目标买入价 {float(target):.2f}" if target else "未设目标价"
                s_s = f"止损价 {float(stop):.2f}" if stop else "未设止损价"
                hint, reason = "WATCH_WAIT", f"观察中：{t_s}，{s_s}，现价 {price:.2f}"
            raw = {"watch": w, "quote": quote, "action_hint": hint, "reason": reason}
            cursor = conn.execute(
                "INSERT OR IGNORE INTO minute_snapshots "
                "(ts,trade_date,minute,code,name,strategy,data_date,advisory_class,advisory_rank,"
                "advisory_score,live_price,live_pct,live_open,live_high,live_low,amount_wan,"
                "action_hint,reason,raw_json) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (now.isoformat(timespec="seconds"), now.date().isoformat(), minute, code,
                 quote.get("name"), w.get("strategy") or "watch", w.get("source_date"),
                 None, None, None, quote.get("price"), quote.get("pct"), quote.get("open"),
                 quote.get("high"), quote.get("low"), quote.get("amount_wan"), hint, reason,
                 json.dumps(raw, ensure_ascii=False)),
            )
            inserted += int(cursor.rowcount > 0)
        conn.commit(); conn.close()
        self.last_snapshot_at = now.isoformat(timespec="seconds")
        return inserted

    def _check_plan_rules(self, now):
        """按冻结计划 DSL 规则做盘中触发检查（sector_pullback.evaluate_intraday）。

        骨架沿用老检查点（_INTRADAY_CHECKPOINTS 先到先执行，另有 plan 独立轮询
        间隔 cfg plan.interval_seconds）。evaluate_intraday 已做当日 jsonl 去重 +
        事件落盘；这里只负责推送：一轮全部新 TRIGGER/INVALIDATE 各成一个 ### 块
        拼一条消息（notifier 批量合并惯例，早盘批量失效不逐条单发），一次
        send_wechat + 一次 broadcast，标题「Stargazer 计划监控」；
        按 (交易日, code, rule_type) 每日去重；OBSERVE 只落盘不推。"""
        import sector_pullback
        cfg = self.get_cfg()
        pcfg = cfg.get("plan") or {}
        interval = max(60, int(pcfg.get("interval_seconds", 180)))
        today = now.date().isoformat()
        # 检查点（老 _INTRADAY_CHECKPOINTS 骨架）到点必查；平时按 plan 间隔轮询
        force = False
        for t, _is_final in sector_pullback._INTRADAY_CHECKPOINTS:
            tag = f"{today}_{t}"
            if tag in self._sector_checked:
                continue
            if sector_pullback._time_passed(t, now):
                self._sector_checked.add(tag)
                force = True
        if not force and now.timestamp() - self._plan_last < interval:
            return
        self._plan_last = now.timestamp()
        if self._plan_day != today:  # 跨日重置推送去重
            self._plan_day = today
            self._plan_pushed = set()
        res = sector_pullback.evaluate_intraday(cfg, now=now)
        for err in res.get("errors") or []:
            notifier.log(f"plan 规则求值失败: {err}")
        n_obs = len(res.get("observed") or [])
        if res.get("status") == "ok" or n_obs:
            notifier.log(f"plan 规则检查（{res.get('checked_at')}）："
                         f"新事件 触发 {len(res.get('triggered') or [])} / "
                         f"失效 {len(res.get('invalidated') or [])} / 观察 {n_obs}"
                         f"（status={res.get('status')}）")
        blocks = []
        for item in (res.get("triggered") or []) + (res.get("invalidated") or []):
            key = f"{today}|{item['code']}|{item['rule_type']}"
            if key in self._plan_pushed:
                continue
            self._plan_pushed.add(key)
            invalidate = item["rule_type"] == "INVALIDATE"
            tag = "【计划·失效】" if invalidate else "【计划·触发】"
            cond = item.get("desc") or item.get("dsl") or ""
            blocks.append(f"### {tag} {item.get('name', '')} {item['code']}\n"
                          f"- 条件：{cond}\n"
                          f"- 时刻 {item['fired_at']}\n"
                          f"- 冻结计划 {item['session_tag']} "
                          f"v{item.get('version_no')}")
        if not blocks:
            return
        text = "\n\n".join(blocks)
        push_cfg = cfg.get("push", {})
        ok, msg = False, "推送已禁用"
        if push_cfg.get("enabled", True):
            ok, msg = notifier.send_wechat(cfg, "Stargazer 计划监控", text)
        notifier.log(f"plan 规则批量推送 {len(blocks)} 块：微信{'成功' if ok else '未发'} ({msg})")
        if push_cfg.get("in_app_notify", True):
            notifier.broadcast({"type": "push", "kind": "plan",
                                "title": "Stargazer 计划监控", "body": text,
                                "wechat": ok, "wechat_msg": msg,
                                "time": now.strftime("%H:%M:%S")})

    def _check_prepared(self, cfg, now):
        """W2d：校验盘后准备清单，缺项自动增量补缺（幂等，工作线程执行）。
        缺清单从权威名单重建；准备未完成的每15分钟有界续查；全部就绪后
        才结束当日检查。盘前打开程序也会跑，不阻塞巡检循环。"""
        today = now.strftime('%Y%m%d')
        if self._prepared_done_day == today:
            return
        if self._prepared_last_at and now.timestamp() - self._prepared_last_at < 900:
            return
        self._prepared_last_at = now.timestamp()
        try:
            import postclose_prepared_v1 as prepared
            report = prepared.verify_and_backfill(cfg, day=int(today))
            if report.get('status') == 'NO_MANIFEST':
                rebuilt = prepared.rebuild_from_authority(cfg)
                if rebuilt is not None:
                    report = prepared.verify_and_backfill(cfg, day=int(today))
                else:
                    report = dict(status='NO_AUTHORITY', day=today)
            status = report.get('status')
            if status == 'READY':
                self._prepared_done_day = today
                notifier.log(f"盘后准备校验（{today}）：{report.get('items')} 项全部就绪"
                             f"；未完成项={[i.get('id') for i in report.get('open_items') or []]}")
            else:
                backlog = report.get('backlog') or []
                notifier.log(f"盘后准备校验（{today}）：{status}，待补缺 {len(backlog)} 项"
                             f"{backlog[:8]}；15分钟后复查")
        except Exception as exc:
            notifier.log(f"盘后准备校验失败: {type(exc).__name__}: {exc}；15分钟后复查")

    def _daily_summary(self, now):
        """日终汇总（T5 设计 §7）：15:00 收盘后且今日确实监控过，发一条微信汇总 +
        落盘 data/plan_daily_summary_YYYYMMDD.json；今日无触发也要报。

        候选数取冻结计划（plan_frozen.json，与 evaluate_intraday 同源）；事件取
        当日 plan_events jsonl（TRIGGER/INVALIDATE 全列，代码+时刻+类型）。"""
        today = now.date().isoformat()
        if self._summary_day == today:
            return
        if self._rounds <= 0 or now.strftime("%H:%M") < "15:00":
            return
        self._summary_day = today
        try:
            import sector_pullback
            cfg = self.get_cfg()
            day = now.strftime("%Y%m%d")
            n_cand = 0
            rel = (cfg.get("plan") or {}).get("plan_file") or "data/plan_frozen.json"
            plan_path = rel if os.path.isabs(rel) else os.path.join(paths.APP_DIR, rel)
            try:
                plan = sector_pullback.load_plan_file(plan_path)
                if (plan.get("trading_date") or day) == day:
                    n_cand = sum(len(s["candidates"]) for s in plan["sessions"])
            except (OSError, ValueError):
                pass  # 无冻结计划/解析失败：按 0 只候选汇总，不阻断
            events = []
            ev_path = os.path.join(paths.APP_DIR, "data", f"plan_events_{day}.jsonl")
            try:
                with open(ev_path, encoding="utf-8") as f:
                    for line in f:
                        try:
                            events.append(json.loads(line))
                        except ValueError:
                            continue
            except OSError:
                pass
            n_trig = sum(1 for e in events if e.get("rule_type") == "TRIGGER")
            if events:
                lines = [f"- {e.get('code')} {str(e.get('fired_at', ''))[11:]} "
                         f"{e.get('rule_type')}" for e in events]
                text = (f"今日计划监控：候选 {n_cand} 只，触发 {n_trig}：\n"
                        + "\n".join(lines))
            else:
                text = f"今日计划监控：今日无触发，{n_cand} 只候选已全程记录"
            push_cfg = cfg.get("push", {})
            ok, msg = False, "推送已禁用"
            if push_cfg.get("enabled", True):
                ok, msg = notifier.send_wechat(cfg, "Stargazer 计划日终汇总", text)
            summary = {"day": day, "candidates": n_cand, "triggered": n_trig,
                       "events": events, "text": text,
                       "wechat_ok": ok, "wechat_msg": msg,
                       "generated_at": now.isoformat(timespec="seconds")}
            out = paths.data_file(f"data/plan_daily_summary_{day}.json",
                                  seed_from_bundle=False)
            os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
            with open(out, "w", encoding="utf-8") as f:
                json.dump(summary, f, ensure_ascii=False, indent=2)
            notifier.log(f"plan 日终汇总：候选 {n_cand} 只，触发 {n_trig}，"
                         f"微信{'成功' if ok else '未发'} ({msg}) → {out}")
            if push_cfg.get("in_app_notify", True):
                notifier.broadcast({"type": "push", "kind": "plan_summary",
                                    "title": "Stargazer 计划日终汇总", "body": text,
                                    "wechat": ok, "wechat_msg": msg,
                                    "time": now.strftime("%H:%M:%S")})
        except Exception as exc:
            notifier.log(f"plan 日终汇总失败: {type(exc).__name__}: {exc}")

    def run(self):
        while True:
            try:
                cfg = self.get_cfg(); model_cfg = cfg.get("model") or {}
                now = dt.datetime.now()
                enabled = bool(model_cfg.get("enabled", True))
                # 状态变迁必须留痕：开关被关/打开是"影子监控静默"的最大嫌疑犯
                if enabled != self._last_enabled_state:
                    notifier.log(f"model-minute-monitor 开关状态变为 enabled={enabled}"
                                 f"（False 时计划规则检查静默）")
                    self._last_enabled_state = enabled
                self._check_prepared(cfg, now)  # W2d：含盘前，准备不依赖开市
                if enabled and notifier.market_open(now):
                    self._hb_n += 1
                    if self._hb_start is None:
                        self._hb_start = now
                    # 每 30 分钟一条心跳：区分"活着但没事做"与"根本没跑"
                    if (now - self._hb_last).total_seconds() >= 1800:
                        mins = (now - self._hb_start).total_seconds() // 60
                        notifier.log(f"model-minute-monitor 心跳：已连续运行 {mins:.0f} 分钟，"
                                     f"本轮循环 {self._hb_n} 次")
                        self._hb_last = now
                    self.snapshot_once()
                    self._check_plan_rules(now)
                    # 日终汇总（§7）：收盘后一轮触发，跨日重置轮数
                    today = now.date().isoformat()
                    if self._rounds_day != today:
                        self._rounds_day = today
                        self._rounds = 0
                    self._rounds += 1
                    self._daily_summary(now)
                self.last_error = None
            except Exception as exc:
                self.last_error = f"{type(exc).__name__}: {exc}"
                # 异常必须进日志：上次线程死亡无声无息，日志停在 10:00:31 无法归因
                notifier.log(f"model-minute-monitor 异常: {self.last_error}")
            interval = max(60, int((self.get_cfg().get("model") or {}).get(
                "intraday_interval_seconds", 60)))
            time.sleep(interval)
