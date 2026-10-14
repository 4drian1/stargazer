/* Stargazer 前端：纯黑 ins 风 · 监控/盘前/对话三页面 · 持仓观察窄卡折叠编辑买入卖出 */
const $ = (id) => document.getElementById(id);
const chatLog = $("chatLog"), input = $("input"), sendBtn = $("sendBtn");
let busy = false, typingEl = null, curBotEl = null;
let planMode = false, planDraftId = null;
let freshPlanSession = false;

marked.setOptions({ breaks: true, gfm: true });
function md(text) {
  const doc = new DOMParser().parseFromString(marked.parse(text || ''), 'text/html');
  doc.querySelectorAll('script,iframe,object,embed,style,link,form,svg,math').forEach(e => e.remove());
  doc.querySelectorAll('*').forEach(e => {
    [...e.attributes].forEach(a => {
      if (a.name.startsWith('on') || a.name === 'style' || a.name === 'srcdoc' ||
          (['href','src'].includes(a.name) && !/^(https?:\/\/|#)/i.test(a.value))) e.removeAttribute(a.name);
    });
  });
  return doc.body.innerHTML;
}

/* ---------- 页面切换 ---------- */
function switchPage(page) {
  document.querySelectorAll(".nav-item").forEach(b => b.classList.toggle("active", b.dataset.page === page));
  if (!["monitor", "context", "chat", "pentacles"].includes(page)) return;
  ["monitor", "context", "chat", "pentacles"].forEach(p => {
    const el = $("page" + p.charAt(0).toUpperCase() + p.slice(1));
    if (el) el.classList.toggle("active", p === page);
  });
}
document.querySelectorAll(".nav-item").forEach(b => b.onclick = () => {
  switchPage(b.dataset.page);
  if (b.dataset.page === "pentacles" && window.refreshPentacles) window.refreshPentacles();
  if (b.dataset.page === "chat") {
    const badge = b.querySelector(".badge");
    if (badge) badge.classList.remove("show");
  }
});

/* ---------- 侧栏页签 ---------- */
const sideTabs = document.querySelectorAll(".side-tabs button");
sideTabs.forEach(b => b.onclick = () => {
  sideTabs.forEach(x => x.classList.toggle("active", x === b));
  document.querySelectorAll(".side-page").forEach(p =>
    p.classList.toggle("active", p.id === "side" + b.dataset.side.charAt(0).toUpperCase() + b.dataset.side.slice(1)));
});

/* ---------- 监控战法标签 ---------- */
const stabs = document.querySelectorAll(".strategy-tabs .stab");
stabs.forEach(b => b.onclick = () => {
  stabs.forEach(x => x.classList.toggle("active", x === b));
  ["green", "horizontal", "t5"].forEach(k =>
    $(k + "Panel").classList.toggle("active", k === b.dataset.tab));
});

/* ---------- 聊天 ---------- */
function addMsg(role, text, extraClass) {
  const el = document.createElement("div");
  el.className = "msg " + role + (extraClass ? " " + extraClass : "");
  if (role === "user") el.textContent = text;
  else el.innerHTML = md(text);
  chatLog.appendChild(el);
  chatLog.scrollTop = chatLog.scrollHeight;
  return el;
}
function addPushMsg(ev) {
  const el = document.createElement("div");
  el.className = "msg push";
  const wx = ev.wechat ? "微信已推" : ("微信未推(" + (ev.wechat_msg || "") + ")");
  el.innerHTML = `<span class="push-tag">风控推送 ${ev.time || ""} · ${wx}</span>` + md(ev.body || "");
  chatLog.appendChild(el);
  chatLog.scrollTop = chatLog.scrollHeight;
  try { new Notification("Stargazer 风控提醒", { body: "有新的风控信号" }); } catch (e) {}
}
function showTyping() {
  typingEl = document.createElement("div");
  typingEl.className = "typing";
  typingEl.textContent = "思考中…";
  chatLog.appendChild(typingEl);
  chatLog.scrollTop = chatLog.scrollHeight;
}
function hideTyping() { if (typingEl) { typingEl.remove(); typingEl = null; curBotEl = null; } }
function toolLine(name, args) {
  if (!typingEl) return;
  typingEl.innerHTML = "";
  const lines = typingEl._tools || [];
  lines.push(`▸ 调用工具 ${name}(${JSON.stringify(args).slice(0, 60)})`);
  typingEl._tools = lines;
  typingEl.innerHTML = lines.map((l) => `<div class="tool-line">${l}</div>`).join("") + "整理回答中…";
  chatLog.scrollTop = chatLog.scrollHeight;
}
async function send(text) {
  text = (text || "").trim();
  if (!text || busy) return;
  busy = true; sendBtn.disabled = true;
  addMsg("user", text); input.value = ""; autoGrow(); showTyping();
  try {
    const r = await fetch("/api/chat", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ message: text, mode: planMode ? 'next_day_plan' : 'chat', draft_id: planDraftId }) });
    const d = await r.json();
    hideTyping();
    addMsg("bot", d.reply || "(空回复)", d.error ? "error" : null);
    if (d.draft) renderPlanDraft(d.draft);
    if (d.graph?.error) addMsg('bot', d.graph.error, 'error');
  } catch (e) { hideTyping(); addMsg("bot", "请求失败: " + e, "error"); }
  busy = false; sendBtn.disabled = false;
  refreshState();
}
sendBtn.onclick = () => send(input.value);
input.addEventListener("keydown", (e) => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); send(input.value); } });
function autoGrow() { input.style.height = "auto"; input.style.height = Math.min(input.scrollHeight, 140) + "px"; }
input.addEventListener("input", autoGrow);
document.querySelectorAll("#quickBar button").forEach((b) => {
  b.onclick = () => {
    if (b.dataset.plan) { window.beginNextDayPlan(); return; }
    if (b.dataset.normal) {
      planMode = false; planDraftId = null;
      document.querySelector('.chat-sub').textContent = 'DeepSeek · 普通对话';
      return;
    }
    if (b.dataset.prompt) { send(b.dataset.prompt); return; }
    const c = b.dataset.cmd;
    if (c.endsWith(" ")) { input.value = c; input.focus(); } else send(c);
  };
});

function renderPlanDraft(draft) {
  planDraftId = draft.id;
  const box = document.createElement('div');
  box.className = 'msg bot';
  const info = document.createElement('p');
  info.textContent = `待确认 · 观察日 ${draft.snapshot.monitor_date} · ${draft.selected_codes.join('、') || '尚未选择股票'}。区域只作到价提醒，不自动买入。`;
  const button = document.createElement('button');
  button.className = 'tb-btn'; button.textContent = '确认计划并开启次日提醒';
  button.onclick = async () => {
    if (draft.id !== planDraftId || busy) { info.textContent = '请确认最新一版计划'; return; }
    if (!confirm('确认这份名单并开启次日到价提醒？不自动下单。')) return;
    button.disabled = true;
    try {
      const response = await fetch('/api/assistant/plan/confirm', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({id:draft.id, confirmed:true})});
      const data = await response.json();
      if (!response.ok || !data.ok) throw new Error(data.error || '确认失败');
      info.textContent = `已冻结 ${data.count} 只，观察日 ${data.monitor_date}。程序需保持运行；微信需开启推送。`;
    } catch (error) { info.textContent = error.message; button.disabled = false; }
  };
  box.append(info, button); chatLog.appendChild(box);
  chatLog.scrollTop = chatLog.scrollHeight;
}
window.beginNextDayPlan = async () => {
  if (busy) return;
  switchPage('chat'); planMode = true;
  document.querySelector('.chat-sub').textContent = 'DeepSeek · 次日计划讨论 · 确认前不生效';
  try {
    const data = await (await fetch('/api/assistant/state')).json();
    if (data.plan && data.plan_current && !planDraftId && !freshPlanSession) {
      addMsg('bot', data.plan.reply); renderPlanDraft(data.plan);
      input.placeholder = '继续修改这份计划；或输入重新制定'; input.focus(); return;
    }
  } catch (_) {}
  planDraftId = null;
  freshPlanSession = false;
  send('请根据当前星币侍从名单制定次日观察计划，说明关注条件、失效条件和未知风险，先给草案供我讨论。');
};

/* ---------- SSE ---------- */
function connectEvents() {
  const es = new EventSource("/api/events");
  es.onmessage = (m) => {
    let ev;
    try { ev = JSON.parse(m.data); } catch (e) { return; }
    if (ev.type === "tool") toolLine(ev.name, ev.args);
    else if (ev.type === "push") { addPushMsg(ev); refreshState();
      const badge = document.querySelector('.nav-item[data-page="chat"] .badge');
      if (badge) badge.classList.add("show");
    }
  };
  es.onerror = () => { es.close(); setTimeout(connectEvents, 3000); };
}

/* ---------- 状态/持仓/观察/交易 ---------- */
let curState = { positions: [], watch: [] };
async function refreshState() {
  let s;
  try { s = await (await fetch("/api/state")).json(); } catch (e) { return; }
  curState = s;
  $("marketDot").className = "dot" + (s.market_open ? " open" : "");
  $("marketText").textContent = s.market_open ? "交易中" : "已收盘/休市";
  $("lastCheck").textContent = s.last_check_at ? "巡检 " + s.last_check_at : "";

  renderPositions(s.positions);
  renderWatch(s.watch);
  $("posCount").textContent = s.positions.length;
  $("watchCount").textContent = s.watch.length;

  const p = s.push;
  $("pushEnabled").checked = !!p.enabled;
  // 交易计划页入口：plan.enabled=false 或全链不存在时隐藏（监控机形态）
  const configured = p.provider === "pushplus" ? p.has_pushplus_token : p.has_sendkey;
  const psb = $("pushStatusBar");
  psb.className = "sb-item" + (configured ? " ok" : "");
  psb.textContent = configured
    ? `${p.provider === "pushplus" ? "PushPlus" : "Server酱"} 已配置 · ${p.interval_minutes}分钟/次 · 收盘${p.close_time}`
    : "未配置推送 key";

  loadTrades();
  $('overviewPositions').textContent = s.positions.length;
  $('overviewWatches').textContent = s.watch.length;
  loadShell();
}

function fmtPrice(v) { return v == null ? "—" : Number(v).toFixed(2); }
function fmtPct(v) { return v == null ? "—" : (v >= 0 ? "+" : "") + Number(v).toFixed(2) + "%"; }

function renderPositions(list) {
  const pos = $("posList");
  pos.innerHTML = list.length ? "" : '<div class="empty">无持仓，点 + 登记</div>';
  list.forEach((p) => {
    const live = p.live_price, pct = p.live_pct;
    const entry = Number(p.entry || 0);
    const pnl = live ? ((live - entry) / entry * 100) : null;
    const stop = Number(p.low250 || 0);
    const distToStop = live && stop ? ((live - stop) / stop * 100) : null;
    const warn = distToStop != null && distToStop < 5;
    const pnlCls = pnl == null ? "" : (pnl >= 0 ? "up" : "down");
    const pctCls = pct == null ? "" : (pct >= 0 ? "up" : "down");
    const el = document.createElement("div");
    el.className = "card pos-card" + (warn ? " warn" : "");
    el.innerHTML = `<button class="del" data-code="${p.code}">×</button>
      <div class="pos-top">
        <span class="nm">${p.name || ""} ${p.code}</span>
        <span class="pos-day ${pctCls}">${pct == null ? "" : fmtPct(pct)}</span>
        <button class="expand-btn"></button>
      </div>
      <div class="pos-mid">
        <span class="pos-live">${live != null ? fmtPrice(live) : "—"}</span>
        <span class="pos-pnl ${pnlCls}">${pnl == null ? "" : (pnl >= 0 ? "浮盈 " : "浮亏 ") + fmtPct(pnl).replace("+", "")}</span>
      </div>
      <div class="pos-detail">
        <div class="detail-row"><span class="dk">成本</span><span class="dv">${fmtPrice(entry)}</span><span class="dk">买入</span><span class="dv">${p.entry_date || "—"}</span></div>
        ${stop ? `<div class="stop-line"><span class="lbl">止损 ${fmtPrice(stop)}</span><div class="stop-bar"><i style="width:${Math.max(0, Math.min(100, distToStop || 0))}%"></i></div><span class="stop-dist">距 ${distToStop == null ? "—" : distToStop.toFixed(1)}%</span></div>` : ""}
        ${p.logic ? `<div class="detail-row"><span class="dk">逻辑</span><span class="dv" style="font-family:var(--sans)">${p.logic}</span></div>` : ""}
        <div class="card-actions">
          <button class="act-btn" data-action="edit-pos" data-code="${p.code}">编辑</button>
          <button class="act-btn sell" data-action="sell" data-code="${p.code}">卖出</button>
        </div>
      </div>`;
    pos.appendChild(el);
  });
  bindCardActions();
}

function renderWatch(list) {
  const wl = $("watchList");
  wl.innerHTML = list.length ? "" : '<div class="empty">空，点 + 添加或从候选卡片加入</div>';
  list.forEach((w) => {
    const live = w.live_price, pct = w.live_pct;
    const target = w.target_buy ? Number(w.target_buy) : null;
    const stop = w.stop_loss ? Number(w.stop_loss) : null;
    let chipText = "观察中", chipCls = "";
    if (live && target) {
      const d = ((live - target) / target * 100);
      if (d <= 1.5) { chipText = `近触发 ${d.toFixed(1)}%`; chipCls = "near"; }
      else { chipText = `回落 ${d.toFixed(1)}%`; }
    }
    const el = document.createElement("div");
    el.className = "card";
    el.innerHTML = `<button class="del" data-code="${w.code}">×</button>
      <div class="nm">${w.name || ""} ${w.code}</div>
      <div class="w-live"><b>${live != null ? fmtPrice(live) : "—"}</b>${pct != null ? ` <span style="color:${pct >= 0 ? 'var(--up)' : 'var(--down)'}">${fmtPct(pct)}</span>` : ""}<span class="w-chip ${chipCls}">${chipText}</span><button class="expand-btn"></button></div>
      <div class="w-detail">
        ${target ? `<div class="detail-row"><span class="dk">目标买入</span><span class="dv">${fmtPrice(target)}</span></div>` : ""}
        ${stop ? `<div class="detail-row"><span class="dk">止损</span><span class="dv">${fmtPrice(stop)}</span></div>` : ""}
        ${w.strategy ? `<div class="detail-row"><span class="dk">策略</span><span class="dv" style="font-family:var(--sans)">${w.strategy}</span></div>` : ""}
        ${w.source_date ? `<div class="detail-row"><span class="dk">来源</span><span class="dv">${w.source_date}</span></div>` : ""}
        ${w.reason ? `<div class="detail-row"><span class="dk">理由</span><span class="dv" style="font-family:var(--sans)">${w.reason}</span></div>` : ""}
        <div class="card-actions">
          <button class="act-btn" data-action="edit-watch" data-code="${w.code}">编辑</button>
          <button class="act-btn buy" data-action="buy-watch" data-code="${w.code}">买入</button>
        </div>
      </div>`;
    wl.appendChild(el);
  });
  bindCardActions();
}

function bindCardActions() {
  document.querySelectorAll(".expand-btn").forEach(b => b.onclick = (e) => { e.stopPropagation(); b.closest(".card").classList.toggle("expanded"); });
  document.querySelectorAll(".card .del").forEach(d => d.onclick = async (e) => { e.stopPropagation(); if (confirm("删除 " + d.dataset.code + "？")) { await send("/del " + d.dataset.code); } });
  document.querySelectorAll("[data-action]").forEach(b => b.onclick = async (e) => {
    e.stopPropagation();
    const code = b.dataset.code;
    const action = b.dataset.action;
    if (action === "edit-pos") openPosModal(code);
    else if (action === "edit-watch") openWatchModal(code);
    else if (action === "sell") openSellModal(code);
    else if (action === "buy-watch") openBuyModalFromWatch(code);
  });
}

/* ---------- 候选卡片渲染 ---------- */
function buildGreenActionNote(r) {
  const off = r.recommended_entry_offset || 1;
  const exp = Number(r[`expected_return_t${off}`] || 0) * 100;
  const risk = Number(r[`risk_probability_t${off}`] || 0);
  const win = Math.max(0, 1 - risk);
  const rr = exp / Math.abs(Number(r.buy_trigger_pct || -2));
  const gateOk = Number(r.entry_utility || 0) > 0 && win >= 0.35 && risk <= 0.65 && rr >= 3.0;
  const trigger = Number(r.buy_trigger_price || 0).toFixed(2);
  const triggerPct = Number(r.buy_trigger_pct || 0).toFixed(1);
  if (gateOk) return `买点：股价回调至 ${trigger} 元（约 ${triggerPct}%）可介入；盈亏比 ${rr.toFixed(2)} 已过关`;
  const parts = [];
  if (rr < 3.0) parts.push(`盈亏比 ${isFinite(rr) ? rr.toFixed(2) : "—"} < 3.0`);
  if (win < 0.35) parts.push(`胜率 ${(win * 100).toFixed(1)}% < 35%`);
  if (risk > 0.65) parts.push(`风险概率 ${(risk * 100).toFixed(1)}% > 65%`);
  if (Number(r.entry_utility || 0) <= 0) parts.push(`点位效用 ${Number(r.entry_utility || 0).toFixed(3)} ≤ 0`);
  if (parts.length === 0) parts.push(`暂不符合自动门控，观察回调至 ${trigger} 元附近`);
  return parts.join("；") + `；参考买入价 ${trigger} 元`;
}

function renderGreenRow(r) {
  const off = r.recommended_entry_offset || 1;
  const exp = Number(r[`expected_return_t${off}`] || 0) * 100;
  const risk = Number(r[`risk_probability_t${off}`] || 0);
  const win = Math.max(0, 1 - risk);
  const rr = exp / Math.abs(Number(r.buy_trigger_pct || -2));
  const gateOk = Number(r.entry_utility || 0) > 0 && win >= 0.35 && risk <= 0.65 && rr >= 3.0;
  const code = r.code || r.symbol;
  const trigger = Number(r.buy_trigger_price || 0).toFixed(2);
  const stopPrice = (Number(trigger) * 0.97).toFixed(2);
  return `<div class="c-card strat-green">
    <div class="c-head">
      <span class="code">${code}</span>
      <span class="tag ${gateOk ? "ok" : "warn"}">G0 ${gateOk ? "已过门控" : "未过门控"}</span>
      <span class="tag">排名 #${r.advisory_rank || "-"}</span>
      <div class="c-head-actions">
        <button class="watch-btn" data-code="${code}" data-name="${r.name || ''}" data-strategy="green_g0"
          data-target="${trigger}" data-stop="${stopPrice}" data-date="${r.date || ''}"
          data-reason="${(r.factor_analysis || '').replace(/"/g, '&quot;')}">加入观察</button>
        <button class="buy-btn" data-code="${code}" data-name="${r.name || ''}" data-price="${trigger}" data-stop="${stopPrice}" data-strategy="green_g0">买入</button>
      </div>
    </div>
    <div class="stat-grid">
      <div class="stat"><span class="k">推荐买入 T+${off}</span><b class="v">${trigger}</b></div>
      <div class="stat"><span class="k">最大回撤</span><b class="v neg">${Number(r.buy_trigger_pct || 0).toFixed(1)}%</b></div>
      <div class="stat"><span class="k">参考止损</span><b class="v">${stopPrice}</b></div>
      <div class="stat"><span class="k">流通市值</span><b class="v">${Number(r.float_mv_yi || 0).toFixed(1)} 亿</b></div>
      <div class="stat"><span class="k">胜率</span><b class="v">${(win * 100).toFixed(1)}%</b></div>
      <div class="stat"><span class="k">预期收益</span><b class="v pos">${exp.toFixed(1)}%</b></div>
      <div class="stat"><span class="k">风险概率</span><b class="v neg">${(risk * 100).toFixed(1)}%</b></div>
      <div class="stat"><span class="k">收益/回撤比</span><b class="v">${isFinite(rr) ? rr.toFixed(2) : "—"}</b></div>
    </div>
    <div class="c-note">${buildGreenActionNote(r)}；${r.factor_analysis || "等待因子解释"}</div>
  </div>`;
}

function renderHorizontalRow(r) {
  const buyPrice = Number(r.sr_support_price || 0).toFixed(2);
  const targetPrice = Number(r.sr_resist_price || 0).toFixed(2);
  const riskProb = Number(r.deep_risk_probability || 0);
  const oppscore = Number(r.deep_opportunity_score || 0);
  const stopPrice = (Number(buyPrice) * 0.97).toFixed(2);
  const code = r.code || r.symbol;
  const note = `买点：回踩支撑位 ${buyPrice} 元附近关注；目标价 ${targetPrice} 元；跌破 ${buyPrice} 元止损` +
    `（机会分 ${oppscore.toFixed(3)}，风险概率 ${(riskProb * 100).toFixed(1)}%）`;
  return `<div class="c-card strat-horizontal">
    <div class="c-head">
      <span class="code">${code}</span>
      <span class="tag">横盘 #${r.advisory_rank || "-"}</span>
      <span class="tag">评分 ${Number(r.advisory_score || 0).toFixed(3)}</span>
      <div class="c-head-actions">
        <button class="watch-btn" data-code="${code}" data-name="${r.name || ''}" data-strategy="horizontal"
          data-target="${buyPrice}" data-stop="${stopPrice}" data-date="${r.ds || ''}"
          data-reason="${(r.factor_analysis || '').replace(/"/g, '&quot;')}">加入观察</button>
        <button class="buy-btn" data-code="${code}" data-name="${r.name || ''}" data-price="${buyPrice}" data-stop="${stopPrice}" data-strategy="horizontal">买入</button>
      </div>
    </div>
    <div class="stat-grid">
      <div class="stat"><span class="k">推荐买入</span><b class="v">${buyPrice}</b></div>
      <div class="stat"><span class="k">目标价</span><b class="v">${targetPrice}</b></div>
      <div class="stat"><span class="k">参考止损</span><b class="v">${stopPrice}</b></div>
      <div class="stat"><span class="k">流通市值</span><b class="v">${Number(r.float_mv_yi || 0).toFixed(1)} 亿</b></div>
      <div class="stat"><span class="k">pos60</span><b class="v">${Number(r.pos60 || 0).toFixed(1)}</b></div>
      <div class="stat"><span class="k">市场分</span><b class="v">${Number(r.market_score || 0).toFixed(3)}</b></div>
      <div class="stat"><span class="k">机会分</span><b class="v">${oppscore.toFixed(3)}</b></div>
      <div class="stat"><span class="k">风险概率</span><b class="v neg">${(riskProb * 100).toFixed(1)}%</b></div>
    </div>
    <div class="c-note">${note}；${r.factor_analysis || "等待因子解释"}</div>
  </div>`;
}

function bindCandidateActions() {
  document.querySelectorAll(".watch-btn").forEach((b) => {
    b.onclick = async () => {
      const r = await (await fetch("/api/watch/add", {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ code: b.dataset.code, strategy: b.dataset.strategy,
          target_buy: parseFloat(b.dataset.target), stop_loss: parseFloat(b.dataset.stop),
          source_date: b.dataset.date, reason: b.dataset.reason || `${b.dataset.strategy} 候选` }),
      })).json();
      addMsg("bot", r.ok ? `已加入观察名单：${b.dataset.code}｜目标买入 ${b.dataset.target}｜止损 ${b.dataset.stop}。` : `加入观察失败：${r.msg || "未知错误"}`);
      refreshState();
    };
  });
  document.querySelectorAll(".buy-btn").forEach((b) => {
    b.onclick = () => openBuyModal({
      code: b.dataset.code, name: b.dataset.name, price: b.dataset.price,
      stop: b.dataset.stop, strategy: b.dataset.strategy,
    });
  });
}

/* ---------- T5 影子决策行（候选×决策点维度，按钮与行一一对应） ---------- */
function dpLabel(dp) {
  const s = String(dp ?? "");
  return s.length >= 3 ? s.slice(0, -2) + ":" + s.slice(-2) : s;
}
function pctEV(v) {
  const n = Number(v || 0);
  const cls = n > 0 ? "pos" : (n < 0 ? "neg" : "");
  return `<b class="v ${cls}">${n >= 0 ? "+" : ""}${(n * 100).toFixed(2)}%</b>`;
}
function renderT5Row(r) {
  const code = r.code || r.symbol;
  const q = String(r.quadrant || "REJECT");
  const dpl = dpLabel(r.dp);
  const head = r.head || (Number(r.dp) < 1200 ? "AM" : "PM");
  const esc = (s) => String(s ?? "").replace(/"/g, "&quot;");
  const stop = r.entry ? (Number(r.entry) * 0.95).toFixed(2) : "";
  let actions;
  if (r.action === "buy") {
    actions = `<span class="t5-state bought">已买入${r.fill_price != null ? ` @ ${r.fill_price}` : ""}` +
      `${r.exit_price != null ? ` · 已退出 @ ${r.exit_price}` : ""}</span>`;
  } else if (r.action === "skip") {
    actions = `<span class="t5-state skipped">已跳过</span>`;
  } else {
    actions = `<button class="t5-skip-btn" data-sig="${r.date}" data-dp="${r.dp}" data-dp-label="${dpl}"
        data-code="${code}">跳过</button>
      <button class="t5-buy-btn" data-sig="${r.date}" data-dp="${r.dp}" data-dp-label="${dpl}"
        data-code="${code}" data-name="${esc(r.name)}" data-price="${r.entry ?? ""}" data-stop="${stop}">买入</button>`;
  }
  return `<div class="c-card strat-t5">
    <div class="c-head">
      <span class="code">${code}</span>
      <span class="tag q-${q}">${q}</span>
      <span class="tag">${dpl} · ${head === "AM" ? "上午头" : "下午头"}</span>
      ${r.ood ? '<span class="tag warn">OOD</span>' : ""}
      <div class="c-head-actions">${actions}</div>
    </div>
    <div class="stat-grid">
      <div class="stat"><span class="k">P(接飞刀)</span><b class="v neg">${(Number(r.P_risk || 0) * 100).toFixed(0)}%</b></div>
      <div class="stat"><span class="k">P(先止盈)</span><b class="v pos">${(Number(r.P_opp || 0) * 100).toFixed(0)}%</b></div>
      <div class="stat"><span class="k">3日参考EV</span>${pctEV(r.ev3_ref)}</div>
      <div class="stat"><span class="k">5日参考EV</span>${pctEV(r.ev5_ref)}</div>
      <div class="stat"><span class="k">参考入场</span><b class="v">${r.entry != null ? Number(r.entry).toFixed(2) : "—"}</b></div>
      <div class="stat"><span class="k">10日涨幅</span><b class="v">${(Number(r.runup_10d || 0) * 100).toFixed(0)}%</b></div>
    </div>
    <div class="c-note">影子双头只记录不推荐；买入/跳过只绑定本决策点 ${dpl}${r.model_version ? ` · 模型 ${r.model_version}` : ""}</div>
  </div>`;
}
function bindT5Actions() {
  document.querySelectorAll(".t5-skip-btn").forEach((b) => {
    b.onclick = async () => {
      const reason = window.prompt(`跳过 ${b.dataset.code} @ ${b.dataset.dpLabel}？可填原因（点取消放弃记录）`, "");
      if (reason === null) return;
      b.disabled = true;
      try {
        const r = await (await fetch("/api/t5/action", {
          method: "POST", headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ sig_date: b.dataset.sig, code: b.dataset.code,
            dp: parseInt(b.dataset.dp, 10), action: "skip", reason }),
        })).json();
        if (r.ok) { addMsg("bot", `已记录跳过 ${b.dataset.code} @ ${b.dataset.dpLabel}（决策点留痕，不影响账户）。`); loadModel(); }
        else { alert(r.error || "记录失败"); b.disabled = false; }
      } catch (e) { alert("网络错误：" + e); b.disabled = false; }
    };
  });
  document.querySelectorAll(".t5-buy-btn").forEach((b) => {
    b.onclick = () => openBuyModal({
      code: b.dataset.code, name: b.dataset.name, price: b.dataset.price,
      stop: b.dataset.stop, strategy: "t5_shadow",
      sigDate: b.dataset.sig, dp: parseInt(b.dataset.dp, 10), dpLabel: b.dataset.dpLabel,
    });
  });
}

/* ---------- 模型建议 ---------- */
async function loadModel() {
  let d;
  try { d = await (await fetch("/api/model/advice?limit=40")).json(); } catch (e) { return; }
  const s = d.status || {};
  const status = $("modelStatus");
  const rows = (d.horizontal || []).slice(0, 10);
  const green = (d.g0_shadow || []).slice(0, 10);
  const t5 = d.t5_shadow || [];
  if (!s.cache_ready) { status.className = "model-status"; status.textContent = s.running ? "正在生成本地建议…" : "尚无缓存；点击生成建议"; }
  else {
    status.className = "model-status " + (s.stale ? "stale" : "ok");
    status.textContent = `数据日 ${s.data_date} · 横盘${rows.length}只 · 绿盘${green.length}只 · ` + (s.running ? "更新中" : (s.stale ? "历史数据" : "本地就绪"));
  }
  $("modelHint").textContent = s.cache_ready ? "候选详情见监控页" : "暂无模型建议";
  $("navHint").textContent = s.cache_ready ? `数据日 ${s.data_date} · 本地模型` : "本地模型";
  $("greenList").innerHTML = green.length ? green.map(renderGreenRow).join("") : '<div class="empty">暂无绿盘低吸候选</div>';
  $("horizontalList").innerHTML = rows.length ? rows.map(renderHorizontalRow).join("") : '<div class="empty">暂无横盘埋伏候选</div>';
  $("greenCount").textContent = `共 ${green.length} 只 · 数据日 ${s.data_date || "—"}`;
  $("horizontalCount").textContent = `共 ${rows.length} 只 · 数据日 ${s.data_date || "—"}`;
  $("greenStatus").textContent = s.cache_ready ? `数据日 ${s.data_date || "—"}` : "尚未读取";
  $("t5List").innerHTML = t5.length
    ? t5.map(renderT5Row).join("")
    : '<div class="empty">暂无影子决策行（盘后生成建议后自动留痕）</div>';
  $("t5Count").textContent = t5.length
    ? `共 ${t5.length} 行 · 信号日 ${d.t5_as_of || "—"} · 模型 ${d.t5_model_version || "—"}`
    : "只记录不推荐 · 人工复核";
  bindCandidateActions();
  bindT5Actions();
}

/* ---------- 盘前助手 & 监控板块 ---------- */

async function loadUserSectors() {
  let d; try { d = await (await fetch("/api/sector_pullback/user_sectors")).json(); }
  catch (e) { $("swResult").textContent = "状态加载失败（服务端错误）"; return; }
  $("swInput").value = (d.names || []).join(", ");
  const mapped = d.mapped || [], missing = d.missing || [];
  $("swStatus").textContent = mapped.length ? `监控中 ${mapped.length} 个` : "未设置";
  $("swResult").textContent = (mapped.length ? "已映射：" + mapped.map((m) => `${m.name}→${m.em_name}`).join("；") : "") + (missing.length ? " 未找到：" + missing.join("、") : "") + (d.last_signal_at ? ` ｜ 最近信号 ${d.last_signal_at}` : "");
  $("sectorMapBar").textContent = mapped.length ? "已映射：" + mapped.map(m => m.name).join("、") + (d.last_signal_at ? `｜最近信号 ${d.last_signal_at}` : "") : "无监控板块";
}
$("swSaveBtn").onclick = async () => {
  const names = $("swInput").value.split(/[,，;；\s]+/).map(s => s.trim()).filter(Boolean).slice(0, 20);
  if (!names.length) { $("swResult").textContent = "请至少填写一个板块名"; return; }
  const d = await (await fetch("/api/sector_pullback/user_sectors", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ names }) })).json();
  loadUserSectors();
  addMsg("bot", `监控板块已保存（最多20个）：${(d.names || []).join("、")}。${(d.missing || []).length ? "未映射到东财板块：" + d.missing.join("、") : "全部映射成功"}。`);
};

/* ---------- 交易计划页（本机专属；T5 双名单设计 §5） ---------- */
let planSessionId = null, planNextDate = null, planDualTimer = null;

function localYmd(d) {
  d = d || new Date();
  return d.getFullYear() * 10000 + (d.getMonth() + 1) * 100 + d.getDate();
}

async function loadPlanDualState() {
  let d;
  try { d = await (await fetch("/api/plan/dual_list/state")).json(); } catch (e) { return; }
  if (d.running) {
    $("planDualStatus").textContent = "生成中…";
    $("planDualLog").textContent = d.log_tail || "全链子进程运行中（数据增量→规则池→T5 打分），约数分钟…";
    if (planDualTimer) clearTimeout(planDualTimer);
    planDualTimer = setTimeout(loadPlanDualState, 3000);
    return;
  }
  planDualTimer = null;
  if (d.error) {
    $("planDualStatus").textContent = "失败";
    $("planDualLog").textContent = d.error;
    return;
  }
  if (d.md) {
    $("planDualStatus").textContent = `数据日 ${d.md_date || "—"}`;
    $("planDualLog").textContent = "";
    $("planDualMd").innerHTML = md(d.md);
  }
}

$("planRunBtn").onclick = async () => {
  const date = String(localYmd());
  $("planRunBtn").disabled = true;
  $("planDualStatus").textContent = "启动中…";
  let r;
  try { r = await (await fetch("/api/plan/dual_list/run", {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ date }),
  })).json(); }
  catch (e) { $("planDualStatus").textContent = "失败"; $("planDualLog").textContent = "请求失败: " + e; $("planRunBtn").disabled = false; return; }
  if (!r.ok && !r.running) {
    $("planDualStatus").textContent = "失败";
    $("planDualLog").textContent = r.error || "启动失败";
    $("planRunBtn").disabled = false;
    return;
  }
  $("planDualMd").innerHTML = '<div class="empty">正在生成双名单，请稍候（页面会自动刷新结果）…</div>';
  loadPlanPool();
  loadPlanDualState();
  setTimeout(() => { $("planRunBtn").disabled = false; }, 10000);
};

async function loadPlanPool() {
  let d;
  try { d = await (await fetch("/api/plan/pool")).json(); } catch (e) { return; }
  const list = $("planPoolList");
  if (!d.ok) { list.innerHTML = `<div class="empty">${d.error || "规则名单池不可用"}</div>`; return; }
  const pool = d.pool || {};
  planNextDate = pool.next_trading_date || null;
  const cands = (pool.candidates || []).filter(c =>
    c.setup === "BOTTOM_SETUP" || c.setup === "INTRADAY_DIP_SETUP");
  if (!cands.length) { list.innerHTML = '<div class="empty">池中没有符合口径的候选（BOTTOM_SETUP / INTRADAY_DIP_SETUP）</div>'; return; }
  const esc = (s) => String(s ?? "").replace(/"/g, "&quot;");
  list.innerHTML = cands.map(c => {
    const trig = c.trigger_price != null ? Number(c.trigger_price) : null;
    return `<div class="plan-pick" data-code="${c.code}" data-preclose="${Number(c.pre_close).toFixed(2)}">
      <label class="pp-check"><input type="checkbox" class="plan-check"> <b>${c.code}</b> ${esc(c.name || "")}</label>
      <span class="tag">${c.setup === "BOTTOM_SETUP" ? "底部 setup" : "盘中 dip"}</span>
      <span class="tag">${esc(c.tier || "-")}</span>
      <span class="pp-field">触发价 <input type="number" step="0.01" class="plan-trigger"
        value="${trig != null ? trig.toFixed(2) : ""}" ${trig != null ? "" : "disabled placeholder=\"仅观察\""}></span>
      <span class="pp-field">昨收 ${Number(c.pre_close).toFixed(2)}</span>
      <input class="plan-reason" placeholder="改价/否决理由（会写进 dialogues 证据）">
    </div>`;
  }).join("");
}

$("planDraftBtn").onclick = async () => {
  const rows = [...document.querySelectorAll("#planPoolList .plan-pick")].filter(
    row => row.querySelector(".plan-check").checked);
  if (!rows.length) { $("planResult").textContent = "请先勾选至少一个候选"; return; }
  const picks = rows.map(row => {
    const trig = row.querySelector(".plan-trigger");
    return {
      code: row.dataset.code,
      name: row.querySelector("b").nextSibling.textContent.trim(),
      trigger_price: trig && trig.value !== "" ? parseFloat(trig.value) : null,
      pre_close: parseFloat(row.dataset.preclose),
      reason: row.querySelector(".plan-reason").value.trim(),
    };
  });
  $("planDraftBtn").disabled = true;
  $("planResult").textContent = "正在组装 draft.json 并落库…";
  let r;
  try {
    r = await (await fetch("/api/plan/draft", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ trading_date: planNextDate || String(localYmd()), candidates: picks }),
    })).json();
  } catch (e) { $("planResult").textContent = "请求失败: " + e; $("planDraftBtn").disabled = false; return; }
  $("planDraftBtn").disabled = false;
  if (!r.ok) { $("planResult").textContent = "生成计划失败：" + (r.error || r.output || "未知错误"); return; }
  planSessionId = r.session_id;
  $("planDraftStatus").textContent = `session ${r.session_id} · DRAFT`;
  $("planResult").textContent = `草案已落库（session ${r.session_id}，含 ${picks.length} 只，用户改动已写入 dialogues 证据）。确认无误后点「冻结并导出」。`;
  $("planFreezeBtn").disabled = !r.session_id;
};

$("planFreezeBtn").onclick = async () => {
  if (!planSessionId) return;
  if (!confirm(`冻结 session ${planSessionId} v1 并导出 plan_frozen.json？冻结后任何业务字段不可修改。`)) return;
  $("planFreezeBtn").disabled = true;
  $("planResult").textContent = "正在冻结并导出…";
  let r;
  try {
    r = await (await fetch("/api/plan/freeze_export", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ session_id: planSessionId,
                             trading_date: planNextDate || String(localYmd()) }),
    })).json();
  } catch (e) { $("planResult").textContent = "请求失败: " + e; $("planFreezeBtn").disabled = false; return; }
  if (!r.ok) { $("planResult").textContent = `冻结/导出失败（${r.step || "?"}）：${r.error || r.output || "未知错误"}`; $("planFreezeBtn").disabled = false; return; }
  $("planDraftStatus").textContent = `session ${planSessionId} · FROZEN`;
  $("planResult").textContent = `已冻结并导出：${r.path} —— 请把该文件拷贝到监控机 data\\plan_frozen.json（覆盖旧文件），明日盘中规则监控即按新计划执行。`;
};

/* ---------- 交易记录 ---------- */
async function loadTrades() {
  let t; try { t = await (await fetch("/api/trades")).json(); } catch (e) { return; }
  const list = $("tradeList");
  const trades = t.trades || [];
  $("tradeCount").textContent = trades.length;
  if (!trades.length) { list.innerHTML = '<div class="empty">暂无成交</div>'; return; }
  list.innerHTML = "";
  trades.slice(0, 50).forEach((r) => {
    const el = document.createElement("div");
    el.className = "trade-card";
    const act = r.action === "buy" ? "买" : "卖";
    const sub = [r.trade_date, r.shares ? (r.shares + "股") : null, r.reason ? ("「" + r.reason + "」") : null].filter(Boolean).join(" · ");
    el.innerHTML = `<span class="trade-badge ${r.action}">${act}</span>` +
      `<div class="trade-main"><div class="nm">${r.name || ""} ${r.code}</div>` + (sub ? `<div class="sub">${sub}</div>` : "") + `</div>` +
      `<div class="trade-price">${r.price}</div>`;
    list.appendChild(el);
  });
}

/* ---------- 设置 ---------- */
async function loadSettings() {
  const s = await (await fetch("/api/state")).json();
  const p = s.push;
  $("cfgProvider").value = p.provider; $("cfgInterval").value = p.interval_minutes; $("cfgClose").value = p.close_time;
  $("cfgMorning").checked = p.morning_report !== false; $("cfgStopWatch").checked = p.stop_watch !== false; $("cfgAuction").checked = p.auction_check !== false;
  toggleProviderFields();
}
function toggleProviderFields() {
  const v = $("cfgProvider").value;
  $("sendkeyField").classList.toggle("hidden", v !== "serverchan");
  $("pushplusField").classList.toggle("hidden", v !== "pushplus");
}
$("openSettings").onclick = () => { loadSettings(); $("settingsModal").classList.remove("hidden"); };
$("closeSettings").onclick = () => $("settingsModal").classList.add("hidden");
$("cfgProvider").onchange = toggleProviderFields;
$("saveSettings").onclick = async () => {
  const body = { enabled: $("pushEnabled").checked, provider: $("cfgProvider").value, interval_minutes: parseInt($("cfgInterval").value || "30", 10), close_time: $("cfgClose").value || "15:05", morning_report: $("cfgMorning").checked, stop_watch: $("cfgStopWatch").checked, auction_check: $("cfgAuction").checked };
  const sk = $("cfgSendkey").value.trim(); if (sk) body.serverchan_sendkey = sk;
  const pp = $("cfgPushplus").value.trim(); if (pp) body.pushplus_token = pp;
  const tv = $("cfgTavily").value.trim(); if (tv) body.tavily_key = tv;
  const ds = $('cfgDeepseek').value.trim(); if (ds) body.deepseek_key = ds;
  const r = await (await fetch("/api/config", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) })).json();
  $("settingsMsg").textContent = r.ok ? "已保存" : "保存失败";
  $('cfgDeepseek').value = '';
  refreshState();
};
$("pushEnabled").onchange = async () => {
  await fetch("/api/config", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ enabled: $("pushEnabled").checked }) });
  refreshState();
};
$("testPushBtn").onclick = async () => {
  const r = await (await fetch("/api/push/test", { method: "POST" })).json();
  addMsg("bot", r.ok ? "测试推送已发出，去微信看看收到没。" : `推送失败：${r.msg}`);
};
$("manualCheckBtn").onclick = async () => {
  addMsg("bot", "正在巡检持仓与观察名单…");
  const r = await (await fetch("/api/check", { method: "POST" })).json();
  addMsg("bot", r.report + (r.push_msg ? `\n\n> 推送状态：${r.push_msg}` : ""));
  refreshState();
};

/* ---------- 模型运行 ---------- */
async function runModel() {
  let state;
  try { state = await (await fetch("/api/model/state")).json(); } catch (e) { addMsg("bot", "获取模型状态失败"); return; }
  const today = new Date().toISOString().slice(0, 10);
  const dataDate = state.data_date || "";
  if (dataDate && dataDate < today) {
    addMsg("bot", `本地数据日期为 ${dataDate}，正在刷新底层 parquet（预计 5–10 分钟）…`);
    setModelStatus(`数据日期 ${dataDate}，正在刷新…`);
    const rr = await (await fetch("/api/model/refresh", { method: "POST", headers: { "Content-Type": "application/json" }, body: "{}" })).json();
    if (!rr.ok) { setModelStatus("刷新失败：" + rr.error, "bad"); return; }
    await waitForState("refresh_running");
  }
  const r = await (await fetch("/api/model/run", { method: "POST", headers: { "Content-Type": "application/json" }, body: "{}" })).json();
  if (!r.ok) { setModelStatus("启动失败：" + r.error, "bad"); return; }
  addMsg("bot", "已启动本地模型生成建议；不会调用 Kimi，也不会自动下单。");
  setModelStatus("正在生成本地建议…");
  await waitForState("running");
  loadModel();
}
function setModelStatus(text, cls) {
  const short = text && text.length > 80 ? text.slice(0, 80) + "…" : text;
  $("modelStatus").textContent = short;
  $("modelStatus").className = "model-status" + (cls ? " " + cls : "");
}
function waitForState(key, maxPolls = 300, interval = 2000) {
  return new Promise(resolve => {
    let polls = 0;
    const timer = setInterval(async () => {
      polls++;
      try { const s = await (await fetch("/api/model/state")).json(); if (!s[key]) { clearInterval(timer); resolve(s); return; } } catch (e) {}
      if (polls >= maxPolls) { clearInterval(timer); resolve(null); }
    }, interval);
  });
}
$("runModelBtn").onclick = () => { switchPage('pentacles'); window.refreshPentacles?.(); };
$("runModelBtnDash").onclick = runModel;
$("runModelBtnHorizontal").onclick = runModel;

/* ---------- 交易记录：导出/快速登记 ---------- */
$("exportTradesBtn").onclick = () => { addMsg("bot", "正在导出成交记录 CSV（可直接喂给 stargazer_ml 做回测验证）…"); window.location.href = "/api/trade/export"; };
$("addTradeBtn").onclick = () => { input.value = "/buy "; input.focus(); switchPage("chat"); };

/* ---------- 持仓/观察 登记 & 编辑 ---------- */
let posEditCode = null, watchEditCode = null;
function openPosModal(code) {
  posEditCode = code || null;
  $("apTitle").firstChild.textContent = code ? "编辑持仓 " : "登记持仓 ";
  if (code) {
    const p = curState.positions.find(x => x.code === code);
    if (p) { $("apCode").value = p.code; $("apCode").disabled = true; $("apName").value = p.name || ""; $("apEntry").value = p.entry || ""; $("apDate").value = p.entry_date || ""; $("apLow").value = p.low250 || ""; $("apLogic").value = p.logic || ""; }
  } else {
    $("apCode").value = ""; $("apCode").disabled = false; $("apName").value = ""; $("apEntry").value = ""; $("apDate").value = new Date().toISOString().slice(0, 10); $("apLow").value = ""; $("apLogic").value = "";
  }
  $("apMsg").textContent = "";
  $("addPosModal").classList.remove("hidden");
}
function openWatchModal(code) {
  watchEditCode = code || null;
  $("awTitle").firstChild.textContent = code ? "编辑观察 " : "加观察名单 ";
  if (code) {
    const w = curState.watch.find(x => x.code === code);
    if (w) { $("awCode").value = w.code; $("awCode").disabled = true; $("awTarget").value = w.target_buy || ""; $("awStop").value = w.stop_loss || ""; $("awStrategy").value = w.strategy || ""; $("awReason").value = w.reason || ""; }
  } else {
    $("awCode").value = ""; $("awCode").disabled = false; $("awTarget").value = ""; $("awStop").value = ""; $("awStrategy").value = "manual"; $("awReason").value = "";
  }
  $("awMsg").textContent = "";
  $("addWatchModal").classList.remove("hidden");
}
$("addPosBtn").onclick = () => openPosModal(null);
$("addWatchBtn").onclick = () => openWatchModal(null);
$("closeAp").onclick = () => $("addPosModal").classList.add("hidden");
$("closeAw").onclick = () => $("addWatchModal").classList.add("hidden");

$("saveAp").onclick = async () => {
  const code = $("apCode").value.trim(), name = $("apName").value.trim(), entry = $("apEntry").value, date = $("apDate").value.trim(), low = $("apLow").value, logic = $("apLogic").value.trim();
  if (!code || !name || !entry || !date || !low) { $("apMsg").textContent = "代码/名称/买价/日期/止损线必填"; return; }
  const url = posEditCode ? "/api/positions/update" : "/api/positions/add";
  const r = await (await fetch(url, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ code, name, entry: parseFloat(entry), entry_date: date, low250: parseFloat(low), logic }) })).json();
  if (r.ok) { $("addPosModal").classList.add("hidden"); refreshState(); }
  else $("apMsg").textContent = r.msg || "保存失败";
};
$("saveAw").onclick = async () => {
  const code = $("awCode").value.trim(), reason = $("awReason").value.trim(), target = $("awTarget").value, stop = $("awStop").value, strat = $("awStrategy").value.trim() || "manual";
  if (!code) { $("awMsg").textContent = "代码必填"; return; }
  if (watchEditCode) {
    const r = await (await fetch("/api/watch/update", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ code, target_buy: target ? parseFloat(target) : null, stop_loss: stop ? parseFloat(stop) : null, strategy: strat, reason }) })).json();
    if (r.ok) { $("addWatchModal").classList.add("hidden"); refreshState(); } else $("awMsg").textContent = r.msg || "保存失败";
  } else {
    const r = await (await fetch("/api/watch/add", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ code, strategy: strat, target_buy: target ? parseFloat(target) : null, stop_loss: stop ? parseFloat(stop) : null, reason }) })).json();
    if (r.ok) { $("addWatchModal").classList.add("hidden"); refreshState(); } else $("awMsg").textContent = r.msg || "保存失败";
  }
};

/* ---------- 卖出 ---------- */
let sellCtx = null;
function openSellModal(code) {
  const p = curState.positions.find(x => x.code === code);
  if (!p) return;
  sellCtx = { code, name: p.name, entry: p.entry, stop: p.low250, live: p.live_price };
  $("sellInfo").innerHTML = `<b>${p.name || ""} ${code}</b><br>成本 ${fmtPrice(p.entry)} · 现价 ${p.live_price != null ? fmtPrice(p.live_price) : "—"} · 止损 ${fmtPrice(p.low250)}`;
  $("sellPrice").value = p.live_price || p.entry || "";
  $("sellShares").value = "";
  $("sellReason").value = "";
  $("sellMsg").textContent = "";
  $("sellModal").classList.remove("hidden");
}
$("closeSell").onclick = () => $("sellModal").classList.add("hidden");
$("confirmSell").onclick = async () => {
  const price = parseFloat($("sellPrice").value);
  if (!price) { $("sellMsg").textContent = "卖出价必填"; return; }
  const shares = $("sellShares").value ? parseInt($("sellShares").value) : null;
  const reason = $("sellReason").value.trim() || "卖出";
  const r = await (await fetch("/api/trade/add", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ action: "sell", code: sellCtx.code, name: sellCtx.name, price, shares, reason, source: "manual" }) })).json();
  if (r.ok) {
    await fetch("/api/chat", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ message: `/del ${sellCtx.code}` }) });
    $("sellModal").classList.add("hidden");
    addMsg("bot", `已卖出 ${sellCtx.name} ${sellCtx.code} @ ${price}，已从持仓移除并计入交易记录。`);
    refreshState();
  } else {
    $("sellMsg").textContent = r.error || "卖出失败";
  }
};

/* ---------- 买入（从候选/观察） ---------- */
let buyCtx = null;
function openBuyModal(info) {
  buyCtx = { ...info };
  const dpNote = info.dpLabel ? ` · T5决策点 ${info.dpLabel}` : "";
  $("buyInfo").innerHTML = `<b>${info.name || info.code}</b> ${info.code}<br>建议买入价 ${info.price || "—"} · 止损 ${info.stop || "—"} · 策略 ${info.strategy || "—"}${dpNote}`;
  $("buyPrice").value = info.price || "";
  $("buyShares").value = "";
  $("buyLow").value = info.stop || "";
  $("buyLogic").value = info.strategy ? `按 ${info.strategy} 策略` : "";
  $("buyMsg").textContent = "";
  $("buyModal").classList.remove("hidden");
}
function openBuyModalFromWatch(code) {
  const w = curState.watch.find(x => x.code === code);
  if (!w) return;
  openBuyModal({ code, name: w.name, price: w.live_price || w.target_buy, stop: w.stop_loss, strategy: w.strategy });
}
$("closeBuy").onclick = () => $("buyModal").classList.add("hidden");
$("confirmBuy").onclick = async () => {
  const price = parseFloat($("buyPrice").value);
  const low = parseFloat($("buyLow").value);
  if (!price || !low) { $("buyMsg").textContent = "买入价和止损线必填"; return; }
  const shares = $("buyShares").value ? parseInt($("buyShares").value) : null;
  const logic = $("buyLogic").value.trim();
  // 1) 加入持仓
  const pr = await (await fetch("/api/positions/add", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ code: buyCtx.code, name: buyCtx.name || buyCtx.code, entry: price, low250: low, logic }) })).json();
  if (!pr.ok) { $("buyMsg").textContent = pr.msg || "加入持仓失败"; return; }
  // 2) 记录买入成交
  await fetch("/api/trade/add", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ action: "buy", code: buyCtx.code, name: buyCtx.name || buyCtx.code, price, shares, reason: logic, source: "manual" }) });
  // 2.5) 来自 T5 影子决策行：账户层之外，在决策层绑定 sig_date+dp 记录人工买入+成交价
  let t5Note = "";
  if (buyCtx.sigDate != null && buyCtx.dp != null) {
    try {
      const tr = await (await fetch("/api/t5/action", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ sig_date: buyCtx.sigDate, code: buyCtx.code, dp: buyCtx.dp, action: "buy", reason: logic, fill_price: price }) })).json();
      if (tr.ok) t5Note = ` T5 决策点 ${buyCtx.dpLabel || buyCtx.dp} 已标记买入。`;
      else t5Note = `（注意：影子决策留痕失败：${tr.error || "未知错误"}）`;
    } catch (e) { t5Note = "（注意：影子决策留痕网络失败）"; }
  }
  // 3) 如果来自观察，移除观察
  const inWatch = curState.watch.some(w => w.code === buyCtx.code);
  if (inWatch) await fetch("/api/chat", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ message: `/del ${buyCtx.code}` }) });
  $("buyModal").classList.add("hidden");
  addMsg("bot", `已买入 ${buyCtx.name || ""} ${buyCtx.code} @ ${price}，止损 ${low}。已加入持仓并记录成交。${t5Note}`);
  refreshState();
  if (buyCtx.sigDate != null) loadModel();
};

/* ---------- 计划盘后总结（stargazer_plan settle 子进程封装） ---------- */
$("planSettleBtn").onclick = async () => {
  if (!confirm("对最新 session 的最新 FROZEN 版跑盘后总结（五结局打标 + 未执行反事实）？")) return;
  addMsg("bot", "正在执行盘后总结（settle 子进程，需要一两分钟）…");
  $("planSettleBtn").disabled = true;
  let r;
  try { r = await (await fetch("/api/plan/settle", { method: "POST" })).json(); }
  catch (e) { addMsg("bot", "盘后总结请求失败：" + e, "error"); $("planSettleBtn").disabled = false; return; }
  $("planSettleBtn").disabled = false;
  addMsg("bot", r.ok
    ? `盘后总结完成（session ${r.session} v${r.version}）：\n\n\`\`\`\n${r.output || r.full || ""}\n\`\`\``
    : `盘后总结失败：${r.error || r.output || "未知错误"}${r.raw ? "\n\n```\n" + r.raw + "\n```" : ""}`,
    r.ok ? null : "error");
};

/* ---------- 启动 ---------- */
if (window.Notification && Notification.permission === "default") Notification.requestPermission();
connectEvents();
refreshState();
loadUserSectors();

async function loadShell() {
  try {
    const data = await (await fetch('/api/shell/state')).json();
    const selected = data.models.selection.payload;
    const date = selected.data_date || '未知';
    $('modelStatus').textContent = `数据日 ${date} · ${selected.candidate_count || 0} 只`;
    $('modelHint').textContent = '规则观察 · 信息/风险未接入';
    $('navHint').textContent = `观察日 ${selected.monitor_date || '未知'} · 稳定应用外壳`;
    $('slotSelection').textContent = `规则观察 · 数据日 ${date}`;
    $('overviewCandidates').textContent = selected.candidate_count ?? '—';
  } catch (_) { $('slotSelection').textContent = '选股快照不可用 · UNKNOWN'; }
}
$('overviewCheck').onclick = () => { switchPage('chat'); send('/check'); };
$('overviewPicks').onclick = () => { switchPage('pentacles'); window.refreshPentacles?.(); };
$('overviewPlan').onclick = () => window.beginNextDayPlan();
$('reviewOpenBtn').onclick = () => { switchPage('pentacles'); $('pentaclesLearning').click(); };
async function restoreChat(fresh=false) {
  if (busy) return;
  if (fresh && !confirm('开始新会话？旧对话保留归档，已确认的提醒计划不改变。')) return;
  busy = true; sendBtn.disabled = true;
  try {
    const response = await fetch('/api/chat/session', {method:fresh ? 'POST' : 'GET'});
    const data = await response.json();
    if (!response.ok) throw new Error('对话恢复失败，旧原文仍保留');
    planMode = false; planDraftId = null; freshPlanSession = fresh; chatLog.replaceChildren();
    (data.messages || []).forEach(m => addMsg(m.role === 'user' ? 'user' : 'bot', m.content));
    document.querySelector('.chat-sub').textContent = 'DeepSeek · 最近对话已恢复 · 旧原文留在本地';
    if (fresh) addMsg('bot', '已开始新会话。切换“制定次日计划”后生成新草案，已确认提醒不自动改变。');
  } catch (error) { addMsg('bot', error.message, 'error'); }
  finally { busy = false; sendBtn.disabled = false; }
}
$('newChatBtn').onclick = () => restoreChat(true);
$('restoreChatBtn').onclick = () => restoreChat();
