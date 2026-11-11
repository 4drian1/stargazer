"""定时巡检 + 微信推送（Server酱Turbo / PushPlus）。
- 盘中每 interval 分钟跑一次 check_portfolio，收盘跑一次总结；
- 交易时段每5分钟止损盯梢：持仓盘中触及止损线/观察名单单日-5% 立即推；
- 每个交易日 morning_time（默认09:05）自动发晨报（隔夜美股+昨日情绪+持仓风控，规则化免LLM）；
有信号 → 微信推送 + 窗口内提醒（SSE广播）。
同一股票同一动作 dedup_hours 小时内不重复推；止损盯梢按 (code,方向) 每日去重。

Step2 稳定性修复：
- 去重键一律是可序列化稳定字符串（SHA-256 十六进制），不再使用 Python hash()；
- 去重/回执状态持久化到磁盘（重启去重）；状态损坏按损坏处理（隔离留证 + 明确风险），
  绝不当空状态盲目重推；
- 去重预约写盘失败 → 不得继续发送（fail-closed）；
- 推送回执（成功/失败）逐条落盘，失败明确记风险，不静默当成功。"""
import datetime as dt
import hashlib
import json
import os
import queue
import threading
import time
import urllib.parse
import urllib.request

# 延迟导入：services（→pandas→numpy→tools→strategy）和 tools.market
# 是重链，只在首次巡检/推送时才加载，避免拖慢启动
_subscribers = []
_sub_lock = threading.Lock()
_log_lock = threading.Lock()
LOG_FILE = None  # 由 server 初始化设置


def log(msg):
    line = f"[{dt.datetime.now():%H:%M:%S}] {msg}"
    with _log_lock:
        print(line, flush=True)
        if LOG_FILE:
            try:
                with open(LOG_FILE, "a", encoding="utf-8") as f:
                    f.write(line + "\n")
            except OSError:
                pass


# ---------- 推送去重/回执持久化（Step2） ----------
PUSH_STATE_FILENAME = "push_state_v1.json"
PUSH_RECEIPT_FILENAME = "push_receipts_v1.jsonl"
PUSHED_MAX_AGE_DAYS = 7  # 去重时间戳保留窗口（dedup_hours 远小于此）


def push_state_path():
    """去重/回执状态文件路径。延迟导入 paths，避免拖慢启动。"""
    try:
        import paths
        return paths.data_file(PUSH_STATE_FILENAME)
    except Exception:
        return os.path.join(os.path.dirname(os.path.abspath(__file__)), PUSH_STATE_FILENAME)


def stable_digest_key(text):
    """可序列化稳定去重键：SHA-256 十六进制。

    绝不使用 Python hash()——它按进程加盐，跨进程/重启都不稳定，无法做去重键。
    """
    return hashlib.sha256((text or "").encode("utf-8")).hexdigest()


def load_dedup_state(path):
    """加载去重状态；损坏时隔离留证并按'未知'处理，绝不当空状态盲目重推。

    返回 dict(pushed, stop_fired, fired, corrupt=bool)，并修剪过期项。
    """
    state = {"pushed": {}, "stop_fired": [], "fired": [], "corrupt": False}
    if not path or not os.path.exists(path):
        return state
    try:
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        if not isinstance(raw, dict):
            raise ValueError("not a dict")
        pushed, stop_fired, fired = raw.get("pushed"), raw.get("stop_fired"), raw.get("fired")
        if not isinstance(pushed, dict) or not isinstance(stop_fired, list) or not isinstance(fired, list):
            raise ValueError("bad structure")
        now = time.time()
        max_age = PUSHED_MAX_AGE_DAYS * 86400
        state["pushed"] = {str(k): float(v) for k, v in pushed.items()
                           if isinstance(v, (int, float)) and (now - float(v)) < max_age}
        today = dt.datetime.now().strftime("%Y-%m-%d")
        state["stop_fired"] = [str(x) for x in stop_fired if str(x).startswith(today)]
        state["fired"] = [str(x) for x in fired if str(x).startswith(today)]
    except Exception as e:
        # 损坏：隔离留证，绝不当空状态。调用方据此保守跳过一次推送。
        state = {"pushed": {}, "stop_fired": [], "fired": [], "corrupt": True}
        try:
            quarantined = f"{path}.corrupt-{dt.datetime.now():%Y%m%d_%H%M%S}"
            os.replace(path, quarantined)
            log(f"推送去重状态损坏，已隔离留证: {quarantined} ({type(e).__name__}: {e})")
        except OSError:
            log(f"推送去重状态损坏且隔离失败: {type(e).__name__}: {e}")
    return state


def save_dedup_state(path, state):
    """原子保存去重状态；失败时抛出（调用方必须 fail-closed，不得继续发送）。"""
    payload = {"pushed": state.get("pushed", {}),
               "stop_fired": state.get("stop_fired", []),
               "fired": state.get("fired", [])}
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def record_push_receipt(kind, title, ok, msg, wechat_enabled):
    """逐条推送回执落盘（成功/失败都记录）。写盘失败仅记日志，绝不抛出。"""
    receipt = {"time": dt.datetime.now().isoformat(timespec="seconds"),
               "key": str(kind), "title": str(title), "ok": bool(ok),
               "msg": str(msg), "wechat_enabled": bool(wechat_enabled)}
    try:
        directory = os.path.dirname(push_state_path())
        if directory:
            os.makedirs(directory, exist_ok=True)
        rp = os.path.join(directory, PUSH_RECEIPT_FILENAME)
        with open(rp, "a", encoding="utf-8") as f:
            f.write(json.dumps(receipt, ensure_ascii=False) + "\n")
        return True
    except Exception as e:
        log(f"推送回执写盘失败（风险留痕）: {type(e).__name__}: {e}")
        return False


# ---------- SSE 订阅（窗口内提醒） ----------
def subscribe():
    q = queue.Queue(maxsize=300)
    with _sub_lock:
        _subscribers.append(q)
    return q


def unsubscribe(q):
    with _sub_lock:
        if q in _subscribers:
            _subscribers.remove(q)


def broadcast(event):
    with _sub_lock:
        subs = list(_subscribers)
    for q in subs:
        try:
            q.put_nowait(event)
        except queue.Full:
            pass


# ---------- 微信推送 ----------
def send_wechat(cfg, title, desp):
    """返回 (ok, 说明)。"""
    p = cfg.get("push", {})
    provider = p.get("provider", "serverchan")
    try:
        if provider == "pushplus":
            token = p.get("pushplus_token", "").strip()
            if not token:
                return False, "未配置 pushplus_token"
            body = json.dumps({"token": token, "title": title, "content": desp,
                               "template": "markdown"}).encode()
            req = urllib.request.Request("http://www.pushplus.plus/send", data=body,
                                         headers={"Content-Type": "application/json"})
            r = json.loads(urllib.request.urlopen(req, timeout=15).read().decode())
            return (r.get("code") == 200), r.get("msg", str(r.get("code")))
        if provider == "dingtalk":
            webhook = p.get("dingtalk_webhook", "").strip()
            if not webhook:
                return False, "未配置 dingtalk_webhook"
            secret = p.get("dingtalk_secret", "").strip()
            url = webhook
            if secret:
                # 加签：timestamp + "\n" + secret 做 HMAC-SHA256，base64 后 URL 编码
                import hmac as _hmac
                import base64 as _b64
                import time as _time
                timestamp = str(round(_time.time() * 1000))
                sign = urllib.parse.quote_plus(_b64.b64encode(
                    _hmac.new(secret.encode("utf-8"),
                              (timestamp + "\n" + secret).encode("utf-8"),
                              hashlib.sha256).digest()).decode("utf-8"))
                url += ("&" if "?" in url else "?") + "timestamp=" + timestamp + "&sign=" + sign
            # 钉钉 markdown 会把单个 \n 折叠：段落升级为空行分隔，保留既有空行不倍增
            paragraphs = [seg for seg in desp.replace("\r\n", "\n").split("\n")
                          if seg.strip()]
            text = "\n\n".join(paragraphs)
            body = json.dumps({"msgtype": "markdown",
                               "markdown": {"title": title, "text": text}},
                              ensure_ascii=False).encode("utf-8")
            req = urllib.request.Request(url, data=body,
                                         headers={"Content-Type": "application/json"})
            r = json.loads(urllib.request.urlopen(req, timeout=15).read().decode())
            return (r.get("errcode") == 0), str(r.get("errmsg") or r)
        key = p.get("serverchan_sendkey", "").strip()
        if not key:
            return False, "未配置 serverchan_sendkey"
        body = urllib.parse.urlencode({"title": title, "desp": desp}).encode()
        req = urllib.request.Request(f"https://sctapi.ftqq.com/{key}.send", data=body)
        r = json.loads(urllib.request.urlopen(req, timeout=15).read().decode())
        return (r.get("code") == 0), str(r.get("message") or r)
    except Exception as e:
        return False, f"推送请求失败: {e}"


# ---------- 交易时段 ----------
def _hm(now):
    return now.hour * 60 + now.minute


def in_session(now):
    if now.weekday() >= 5:
        return False
    m = _hm(now)
    return (9 * 60 + 25 <= m <= 11 * 60 + 35) or (12 * 60 + 55 <= m <= 15 * 60 + 10)


def market_open(now):
    if now.weekday() >= 5:
        return False
    m = _hm(now)
    return (9 * 60 + 30 <= m <= 11 * 60 + 30) or (13 * 60 <= m <= 15 * 60)


# ---------- 巡检 ----------
_ACTION_TAG = {"清仓": "【清仓】", "减半": "【减半】", "换股": "【换股】", "持有": "【提醒】"}


def compose_report(report, kind, now):
    """把 check_portfolio 结果整理成 markdown；无任何信号且非收盘总结时返回 None。"""
    lines, has_signal = [], False
    for r in report.get("positions", []):
        if not r.get("signals"):
            continue
        has_signal = True
        tag = _ACTION_TAG.get(r.get("action"), "【提醒】")
        lines.append(f"### {tag} {r.get('name') or ''} {r['code']} — 建议:{r.get('action')}")
        lines.append(f"浮盈 {r.get('gain', 0):+.1f}% | 已持 {r.get('held_days', '?')} 个交易日")
        for s in r["signals"]:
            lines.append(f"- {s}")
        lines.append("")
    for w in report.get("watch_alerts", []):
        has_signal = True
        wtype = w.get("type", "")
        if wtype == "BUY_TRIGGER":
            lines.append(f"### 【观察·买点触发】{w.get('name') or ''} {w['code']}")
            lines.append(f"- {w.get('msg', '')}")
            lines.append(f"- 来源：{w.get('strategy', '观察')} {w.get('source_date', '')}｜{w.get('reason', '')}")
        elif wtype == "STOP_LOSS_BROKEN":
            lines.append(f"### 【观察·止损破位】{w.get('name') or ''} {w['code']}")
            lines.append(f"- {w.get('msg', '')}")
            lines.append(f"- 来源：{w.get('strategy', '观察')} {w.get('source_date', '')}｜{w.get('reason', '')}")
        else:
            lines.append(f"### 【观察预警】{w.get('name') or ''} {w['code']} 今日 {w['pct']:+.1f}%")
            lines.append(f"- 超卖判定: {'触发' if w.get('oversold_trigger') else '未触发'} | {w.get('detail', '')}")
        lines.append("")
    if not has_signal and kind != "close":
        return None
    head = f"## 收盘巡检 {now:%Y-%m-%d %H:%M}" if kind == "close" else f"## 盘中巡检 {now:%H:%M}"
    if not has_signal:
        lines.append("持仓与观察名单：无触发信号，纪律内运行。")
    n_pos = len(report.get("positions", []))
    return head + f"（持仓{n_pos}只）\n\n" + "\n".join(lines)


# ---------- 止损盯梢（盘中每5分钟，轻量报价） ----------
def compose_stop_watch(cfg):
    """只拉实时报价（不跑K线），持仓盘中破止损线 / 观察名单单日-5%。
    返回 [(key, markdown_block), ...]，key=(code, 'stop'|'crash') 供每日去重。"""
    import services
    from tools import market
    wl = services.wl_load(cfg)
    pos, watch = wl.get("positions", []), wl.get("watch", [])
    codes = [p["code"] for p in pos] + [w["code"] for w in watch]
    if not codes:
        return []
    qs = market.quotes(codes)
    alerts = []
    for p in pos:
        q = qs.get(p["code"])
        if not q:
            continue
        line_stop = float(p.get("low250") or 0)
        if line_stop > 0 and q["price"] <= line_stop:
            block = (f"### 【盘中触止损】{p.get('name') or ''} {p['code']}\n"
                     f"现价 {q['price']}（{q['pct']:+.2f}%）≤ 结构止损线 {line_stop}\n"
                     f"- 手册按**收盘确认**：尾盘收不回止损线上方 → 执行清仓；收回则豁免。")
            alerts.append(((p["code"], "stop"), block))
    for w in watch:
        q = qs.get(w["code"])
        if not q:
            continue
        price = q["price"]
        target = w.get("target_buy")
        stop = w.get("stop_loss")
        if target and price <= float(target):
            block = (f"### 【观察·买点触发】{q['name']} {w['code']}\n"
                     f"- 现价 {price:.2f} ≤ 目标买入价 {float(target):.2f}，等待人工确认\n"
                     f"- 来源：{w.get('strategy', '观察')} {w.get('source_date', '')}｜{w.get('reason', '')}")
            alerts.append(((w["code"], "buy_trigger"), block))
        if stop and price <= float(stop):
            block = (f"### 【观察·止损破位】{q['name']} {w['code']}\n"
                     f"- 现价 {price:.2f} ≤ 止损价 {float(stop):.2f}，建议放弃观察\n"
                     f"- 来源：{w.get('strategy', '观察')} {w.get('source_date', '')}｜{w.get('reason', '')}")
            alerts.append(((w["code"], "stop_broken"), block))
        elif q["pct"] <= -5:
            block = (f"### 【观察闪崩】{q['name']} {w['code']} 今日 {q['pct']:+.2f}%\n"
                     f"- 现价 {q['price']}，触发单日-5%预警，可查是否进入超卖伏击区。")
            alerts.append(((w["code"], "crash"), block))
    return alerts


# ---------- 交易日晨报（规则化，免LLM） ----------
def compose_morning(cfg, now):
    """09:05 晨报：隔夜美股 + 昨日情绪 + 持仓风控快报。部分拉取失败不阻塞整体。"""
    import services
    parts = [f"## 晨报 {now:%Y-%m-%d}（交易日自动快报）\n"]
    try:
        us = services.us_market()
        parts.append("### 隔夜美股")
        for r in us["隔夜美股(名称,现价,涨跌幅%,行情时间)"]:
            parts.append(f"- {r['name']}: {r['pct']:+.2f}%")
    except Exception as e:
        parts.append(f"### 隔夜美股\n- 拉取失败: {e}")
    try:
        ov = services.market_overview()
        emo = ov["情绪"]
        parts.append(f"\n### 昨日盘面（收盘数据）")
        parts.append(f"- 两市成交额 {ov['两市成交额(亿,上证+深成)']} 亿 | "
                     f"涨停 {emo['涨停家数']} / 跌停 {emo['跌停家数']} | "
                     f"最高连板 {emo['最高连板']} | 炸板率 {emo['炸板率%']}%")
        idx = " | ".join(f"{i['name']} {i['pct']:+.2f}%" for i in ov["指数(名称,点位,涨跌%,成交额亿)"][:4])
        parts.append(f"- {idx}")
        st = ov.get("大小盘风格")
        if st:
            parts.append(f"- 大小盘风格：{st['判读']}（沪深300 {st['沪深300']:+.2f}% / 国证2000 {st['国证2000']:+.2f}%）")
    except Exception as e:
        parts.append(f"\n### 昨日盘面\n- 拉取失败: {e}")
    try:
        report = services.check_portfolio(cfg)
        n_sig = sum(1 for r in report.get("positions", []) if r.get("signals"))
        parts.append(f"\n### 持仓风控（{len(report.get('positions', []))}只）")
        if n_sig == 0 and not report.get("watch_alerts"):
            parts.append("- 无触发信号，纪律内运行。")
        for r in report.get("positions", []):
            if not r.get("signals"):
                continue
            parts.append(f"- **{r.get('name') or ''} {r['code']}** 建议:{r.get('action')} | "
                         + "；".join(r["signals"][:2]))
        for w in report.get("watch_alerts", []):
            parts.append(f"- 观察预警 **{w.get('name') or ''} {w['code']}** 昨日 {w['pct']:+.1f}%")
    except Exception as e:
        parts.append(f"\n### 持仓风控\n- 检查失败: {e}")
    parts.append("\n---\n数据快报由规则生成。需要 AI 深度解读（题材映射/持仓影响/今日策略）→ 打开窗口点「晨报分析」。")
    return "\n".join(parts)


# ---------- 竞价三条件检查（默认09:26推送，v2.4.3执行协议） ----------
def compose_auction(cfg, now):
    """集合竞价结束后对观察名单逐只检查 ABC 三条件：
    A 板块竞价首板≥1：所属行业板块内竞价涨幅≥9.8%的家数（竞价口径近似值）；
    B 观察票竞价涨幅在 -2% ~ +1%；
    C 大盘竞价平稳：上证指数 |竞价涨跌幅| ≤ 0.5%。
    ABC全过 → 给出挂单价参考=竞价价×0.99（计划仓位50%，人工下单）。
    观察名单为空返回 None。"""
    import services
    from tools import market
    wl = services.wl_load(cfg)
    watch = wl.get("watch", [])
    if not watch:
        return None
    codes = [w["code"] for w in watch]
    qs = market.quotes(codes)
    # 条件C：大盘竞价（9:26时腾讯报价的pct即竞价涨幅）
    c_pct = None
    try:
        for i in market.index_quotes():
            if i["code"] == "000001":
                c_pct = i["pct"]
                break
    except Exception:
        pass
    c_ok = c_pct is not None and abs(c_pct) <= 0.5
    lines = [f"## 竞价检查 {now:%Y-%m-%d %H:%M}（集合竞价结果）\n"]
    if c_pct is None:
        lines.append("**条件C 大盘竞价**：上证指数拉取失败，请人工看一眼大盘。")
    else:
        lines.append(f"**条件C 大盘竞价**：上证指数 {c_pct:+.2f}% → "
                     + ("✅ 平稳（|涨跌|≤0.5%）" if c_ok else "❌ 超出±0.5%，今日整体谨慎"))
    board_cache = {}
    for w in watch:
        code = w["code"]
        q = qs.get(code)
        lines.append("")
        if not q:
            lines.append(f"### {w.get('name') or ''} {code}\n- 竞价报价拉取失败，跳过")
            continue
        name = q.get("name") or w.get("name") or ""
        # 条件B：个股竞价涨幅 ∈ [-2%, +1%]
        b_ok = -2 <= q["pct"] <= 1
        # 条件A：板块竞价首板≥1（行业板块成分竞价涨幅≥9.8%家数）
        a_ok, a_txt = False, "所属板块未知"
        try:
            ind = market.stock_industry(code)
        except Exception:
            ind = None
        if ind:
            if ind not in board_cache:
                entry = (ind, None)
                try:
                    found, bname = services._find_board(ind)
                    if isinstance(found, str):
                        cons = market.board_constituents(found)
                        n_fb = sum(1 for v in cons.values()
                                   if isinstance(v.get("pct"), (int, float)) and v["pct"] >= 9.8)
                        entry = (bname or ind, n_fb)
                except Exception:
                    pass
                board_cache[ind] = entry
            bname, n_fb = board_cache[ind]
            if n_fb is not None:
                a_ok = n_fb >= 1
                a_txt = f"{bname}竞价首板 {n_fb} 家（竞价口径近似值）"
        lines.append(f"### {name} {code} — 竞价 {q['price']}（{q['pct']:+.2f}%）")
        lines.append(f"- 条件A 板块竞价首板≥1：{a_txt} {'✅' if a_ok else '❌'}")
        lines.append(f"- 条件B 个股竞价-2%~+1%：{q['pct']:+.2f}% {'✅' if b_ok else '❌'}")
        lines.append(f"- 条件C 大盘竞价±0.5%：{'✅' if c_ok else '❌'}")
        if a_ok and b_ok and c_ok:
            ref = round(q["price"] * 0.99, 2)
            lines.append(f"→ **ABC全过**：挂单价参考 **{ref}**（竞价价×0.99），"
                         f"计划仓位的50%，人工限价挂单。")
        else:
            lines.append("→ 未全过，今日不追，继续观察。")
    lines.append("\n---\n竞价口径为9:25撮合结果的近似值。"
                 "后续节点按手册人工执行：10:00检查成交与量能，14:30只检查首批成交且缩量的情况。")
    return "\n".join(lines)


class Notifier(threading.Thread):
    def __init__(self, get_cfg):
        super().__init__(daemon=True)
        self.get_cfg = get_cfg
        # 重启去重：从磁盘恢复去重/已触发状态；损坏时保守处理（见 load_dedup_state）。
        persisted = load_dedup_state(push_state_path())
        self._pushed = persisted["pushed"]
        self._stop_fired = set(persisted["stop_fired"])   # 止损盯梢每日去重
        self._fired = set(persisted["fired"])             # 已触发的时刻标记
        self._dedup_corrupt = bool(persisted.get("corrupt"))
        self._dedup_error = None
        self.last_report = None  # 最近一次巡检文本（供界面查看）
        self.last_check_at = None

    def _due(self, now, pcfg):
        if not in_session(now):
            return None
        stamp = now.strftime("%Y-%m-%d %H:%M")
        close_hm = pcfg.get("close_time", "15:05")
        if now.strftime("%H:%M") == close_hm and stamp not in self._fired:
            return "close", stamp
        interval = max(5, int(pcfg.get("interval_minutes", 30)))
        if now.minute % interval == 0 and stamp not in self._fired:
            return "intraday", stamp
        return None

    def _dedup_ok(self, key, hours):
        """去重预约：命中窗口内 → False；否则登记并落盘。

        预约写盘失败 → 回滚登记并返回 False（fail-closed，不得继续发送）。
        """
        key = key if isinstance(key, str) else stable_digest_key(repr(key))
        last = self._pushed.get(key)
        if last and (time.time() - last) < hours * 3600:
            return False
        if self._dedup_corrupt:
            # 状态曾损坏：内容未知，绝不当空状态盲目重推；保守跳过一次并留痕。
            self._dedup_corrupt = False
            self._dedup_error = "DEDUP_STATE_WAS_CORRUPT_SKIPPED_ONCE"
            log("推送去重状态曾损坏，本次保守跳过推送（风险留痕）")
            return False
        self._pushed[key] = time.time()
        try:
            save_dedup_state(push_state_path(),
                             {"pushed": self._pushed,
                              "stop_fired": sorted(self._stop_fired),
                              "fired": sorted(self._fired)})
            self._dedup_error = None
            return True
        except Exception as e:
            self._pushed.pop(key, None)
            self._dedup_error = f"DEDUP_SAVE_FAILED:{type(e).__name__}"
            log(f"推送去重预约写盘失败，跳过本次推送: {e}")
            return False

    def run_once(self, kind="manual"):
        """执行一次巡检；返回 (推送文本 or None, 推送结果说明)。"""
        import services
        cfg = self.get_cfg()
        now = dt.datetime.now()
        try:
            report = services.check_portfolio(cfg)
        except Exception as e:
            log(f"巡检失败: {e}")
            return None, f"巡检执行失败: {e}"
        self.last_check_at = now.strftime("%H:%M:%S")
        text = compose_report(report, kind, now)
        if text is None:
            self.last_report = f"[{now:%H:%M}] 巡检完成，无信号。"
            log("巡检完成，无信号")
            return None, "无信号"
        self.last_report = text
        # 展示与发送资格分开：推送关闭时只在窗口展示，不消耗外部去重资格；
        # 去重只在真实发送前登记（已发送/送达未知仍不盲目重发的规则保留）
        pcfg = cfg.get("push", {})
        wechat_enabled = bool(pcfg.get("enabled", True))
        if wechat_enabled and kind != "close" and \
                not self._dedup_ok(stable_digest_key(text), pcfg.get("dedup_hours", 4)):
            self.last_report = text
            log("信号与上次相同，去重跳过推送")
            return text, "信号重复，已去重未推送"
        title = "Stargazer 风控提醒" if kind != "close" else "Stargazer 收盘巡检"
        ok, msg = False, "推送已禁用"
        if wechat_enabled:
            ok, msg = send_wechat(cfg, title, text)
        log(f"巡检有信号，微信推送: {'成功' if ok else '未发'} ({msg})")
        if wechat_enabled and not ok:
            log(f"推送回执失败，明确记风险: kind={kind} msg={msg}")
        record_push_receipt(kind, title, ok, msg, wechat_enabled)
        if pcfg.get("in_app_notify", True):
            broadcast({"type": "push", "kind": kind, "title": title, "body": text,
                       "wechat": ok, "wechat_msg": msg, "time": now.strftime("%H:%M:%S")})
        return text, msg

    def _deliver(self, cfg, title, text, kind):
        """统一出口：微信推送 + 窗口提醒（回执逐条落盘，失败明确记风险）。"""
        pcfg = cfg.get("push", {})
        ok, msg = False, "推送已禁用"
        wechat_enabled = bool(pcfg.get("enabled", True))
        if wechat_enabled:
            ok, msg = send_wechat(cfg, title, text)
        log(f"{title}: 微信{'成功' if ok else '未发'} ({msg})")
        if wechat_enabled and not ok:
            log(f"推送回执失败，明确记风险: kind={kind} msg={msg}")
        record_push_receipt(kind, title, ok, msg, wechat_enabled)
        if pcfg.get("in_app_notify", True):
            broadcast({"type": "push", "kind": kind, "title": title, "body": text,
                       "wechat": ok, "wechat_msg": msg,
                       "time": dt.datetime.now().strftime("%H:%M:%S")})
        return ok, msg

    def _maybe_stop_watch(self, cfg, now):
        """交易时段每5分钟：持仓盘中触止损线 / 观察名单单日-5% 立即提醒（按 code+方向 每日去重）。"""
        pcfg = cfg.get("push", {})
        if not pcfg.get("stop_watch", True) or not market_open(now) or now.minute % 5 != 0:
            return
        today = now.strftime("%Y-%m-%d")
        self._stop_fired = {k for k in self._stop_fired if k.startswith(today)}
        try:
            alerts = compose_stop_watch(cfg)
        except Exception as e:
            log(f"止损盯梢失败: {e}")
            return
        fresh = [(k, b) for k, b in alerts
                 if f"{today}|{k[0]}|{k[1]}" not in self._stop_fired]
        if not fresh:
            return
        self._stop_fired.update(f"{today}|{k[0]}|{k[1]}" for k, _ in fresh)
        log(f"止损盯梢触发: {[k[0] for k, _ in fresh]}")
        text = "\n\n".join(b for _, b in fresh)
        self._deliver(cfg, "Stargazer 止损盯梢", text, "stop_watch")

    def _maybe_morning(self, cfg, now):
        """每个交易日 morning_time（默认09:05）自动晨报。"""
        pcfg = cfg.get("push", {})
        if not pcfg.get("morning_report", True) or now.weekday() >= 5:
            return
        mt = pcfg.get("morning_time", "09:05")
        stamp = now.strftime("%Y-%m-%d ") + mt
        if now.strftime("%H:%M") != mt or stamp in self._fired:
            return
        self._fired.add(stamp)
        log("触发晨报")
        try:
            text = compose_morning(cfg, now)
        except Exception as e:
            log(f"晨报生成失败: {e}")
            return
        self.last_report = text
        self._deliver(cfg, f"Stargazer 晨报 {now:%m-%d}", text, "morning")

    def _maybe_auction(self, cfg, now):
        """每个交易日 auction_time（默认09:26）竞价三条件检查推送。"""
        pcfg = cfg.get("push", {})
        if not pcfg.get("auction_check", True) or now.weekday() >= 5:
            return
        at = pcfg.get("auction_time", "09:26")
        stamp = now.strftime("%Y-%m-%d ") + at
        if now.strftime("%H:%M") != at or stamp in self._fired:
            return
        self._fired.add(stamp)
        log("触发竞价检查")
        try:
            text = compose_auction(cfg, now)
        except Exception as e:
            log(f"竞价检查失败: {e}")
            return
        if text:
            self._deliver(cfg, f"Stargazer 竞价检查 {now:%m-%d}", text, "auction")

    def _maybe_daily_tarot(self, cfg, now):
        """每个交易日 daily_tarot.time（默认09:00）推送今日塔罗+星象；周末不发，当日仅一次。

        文案层 daily_tarot_v1：抽牌/星象为确定性事实，LLM 仅润色口吻，失败或校验
        不通过静默回退事实核；免责页脚由代码追加。不进交易链路。"""
        tc = (cfg.get("push", {}) or {}).get("daily_tarot", {}) or {}
        if not tc.get("enabled", True) or now.weekday() >= 5:
            return
        at = tc.get("time", "09:00")
        stamp = now.strftime("%Y-%m-%d ") + at
        if now.strftime("%H:%M") != at or stamp in self._fired:
            return
        self._fired.add(stamp)
        log("触发今日塔罗")
        try:
            import daily_tarot_v1
            text = daily_tarot_v1.compose_daily_tarot(cfg, now)
        except Exception as e:
            log(f"今日塔罗生成失败: {type(e).__name__}: {e}")
            return
        self.last_report = text
        self._deliver(cfg, f"今日塔罗 · {now:%m-%d}", text, "daily_tarot")

    def run(self):
        log("定时巡检线程已启动")
        while True:
            try:
                cfg = self.get_cfg()
                pcfg = cfg.get("push", {})
                now = dt.datetime.now()
                if pcfg.get("enabled", True):
                    due = self._due(now, pcfg)
                    if due:
                        kind, stamp = due
                        self._fired.add(stamp)
                        self._fired = {s for s in self._fired if s.startswith(now.strftime("%Y-%m-%d"))}
                        log(f"触发{kind}巡检")
                        self.run_once(kind)
                    self._maybe_morning(cfg, now)
                    self._maybe_auction(cfg, now)
                    self._maybe_daily_tarot(cfg, now)
                    self._maybe_stop_watch(cfg, now)
            except Exception as e:
                log(f"巡检循环异常: {e}")
            time.sleep(15)
