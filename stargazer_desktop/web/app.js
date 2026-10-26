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
const PAGES = ['monitor', 'planTrade', 'chat'];
function switchPage(page) {
  document.querySelectorAll(".nav-item").forEach(b => b.classList.toggle("active", b.dataset.page === page));
  if (!PAGES.includes(page)) return;
  PAGES.forEach(p => {
    const el = $("page" + p.charAt(0).toUpperCase() + p.slice(1));
    if (el) el.classList.toggle("active", p === page);
  });
}
document.querySelectorAll(".nav-item").forEach(b => b.onclick = () => {
  switchPage(b.dataset.page);
  if (b.dataset.page === "planTrade" && window.refreshPentacles) window.refreshPentacles();
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
    if (d.draft?.presentation && !d.draft.awaiting_count && !d.error) renderPlanDiscussion(d.draft);
    else addMsg("bot", d.reply || "(空回复)", d.error ? "error" : null);
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
      return;
    }
    if (b.dataset.prompt) { send(b.dataset.prompt); return; }
    const c = b.dataset.cmd;
    if (c.endsWith(" ")) { input.value = c; input.focus(); } else send(c);
  };
});

function renderPlanDiscussion(draft, stale=false) {
  const view = draft.presentation, box = document.createElement('div');
  box.className = 'msg bot plan-discussion';
  const paragraph = (label, text) => {
    const p = document.createElement('p'), title = document.createElement('strong');
    title.textContent = label + '：'; p.append(title, document.createTextNode(text)); box.append(p);
  };
  paragraph('草案', view.draft);
  if (draft.changes) {
    const labels = {model_added:'新增模型候选', model_removed:'移除模型候选', manual_added:'新增人工观察',
      manual_removed:'移除人工观察', conditions_updated:'更新人工条件原文', boards_added:'新增板块', boards_removed:'移除板块'};
    const changes = Object.entries(labels).filter(([k])=>draft.changes[k]?.length)
      .map(([k,label])=>`${label}：${draft.changes[k].join('、')}`);
    paragraph('本次修改', (changes.join('；') || '名单、板块和人工条件原文无变更') + '。确认前不修改正式提醒。');
  }
  for (const analysis of draft.stock_analyses || []) {
    paragraph('模型查询', `${analysis.code} · 数据日 ${analysis.data_date || '未知'} · ${analysis.status}。` +
      (analysis.reason || '读取已验收Tarot/Pyxis结果，具体幅度与输入见风险证据；不是本次重新推理。'));
  }
  if (view.manual_observations?.length) {
    paragraph('人工观察', `${view.manual_observations.map(r=>r.code).join('、')}（你主动添加，不是模型入选；尚无绑定风险评分）。`);
    const notes = document.createElement('details'), caption = document.createElement('summary');
    caption.textContent = '查看人工条件原文（未转换为自动触发规则）'; notes.append(caption);
    for (const row of view.manual_observations) {
      const p = document.createElement('p'); p.textContent = `${row.code}：${row.user_requests.join('\n')}`; p.style.whiteSpace = 'pre-wrap'; notes.append(p);
    }
    box.append(notes);
    paragraph('条件待核对', '人工条件已记录；价格或跌幅的参考基准、止跌定义未结构化确认前，不执行自动价格/买入提醒。可继续补充或修改，模型候选的原规则不因此改变。');
  }
  if (draft.snapshot.rule_version) {
    paragraph('关注条件', '前日F618仅作执行参照；次日须已完成五分钟K线出现下杀、止跌、有效RSI背离。触价不等于买入信号。');
    paragraph('失效条件', '三项未齐不发触发信号；缺分钟数据为未知，不能当作条件成立。');
  }
  paragraph('风险', stale ? '以下为旧草案绑定证据，已失效，不能当成当前结果。' : '以下为Pyxis未来120交易分钟不利下探预测幅度，由代码读取证据；不是概率，不覆盖整个次日。');
  if (view.market_risk_summary?.length) paragraph('大盘风险', view.market_risk_summary.join(' '));
  for (const row of view.risks || []) {
    const details = document.createElement('details'), summary = document.createElement('summary');
    const lines = row.lines || [];
    summary.textContent = `${row.code} ${row.name}：${(lines[1] || lines[0] || '未知').replace('未来120交易分钟不利下探预测幅度：', '').split('。')[0]}`;
    details.append(summary);
    for (const line of lines.filter((_, i) => i !== (lines[1] ? 1 : 0))) {
      const p = document.createElement('p'); p.textContent = line; details.append(p);
    }
    const comment = draft.model_commentary?.[row.code]?.risk;
    if (comment) { const p = document.createElement('p'); p.textContent = `AI解释（非新增事实）：${comment}`; details.append(p); }
    box.append(details);
  }
  paragraph('待确认', view.confirmation);
  const original = document.createElement('details'), caption = document.createElement('summary');
  caption.textContent = '展开AI讨论原文（含数量、板块与重评说明）';
  const text = document.createElement('div'); text.innerHTML = md(draft.reply || '未生成讨论');
  original.append(caption, text); box.append(original); chatLog.append(box);
}

function renderPlanDraft(draft, readiness=null) {
  planMode = true;
  planDraftId = draft.id;
  if (draft.awaiting_count) { input.placeholder = '希望保留多少只？例如：5只'; input.focus(); return; }
  const box = document.createElement('div');
  box.className = 'msg bot';
  const info = document.createElement('p');
  info.textContent = `待确认 · 观察日 ${draft.snapshot.monitor_date} · ${draft.selected_codes.join('、') || '尚未选择股票'}。区域只作到价提醒，不自动买入。`;
  if (draft.manual_observations?.length) info.textContent += ` 另有人工观察 ${draft.manual_observations.map(r=>r.code).join('、')}，条件仅记录、无模型评分/自动价格提醒。`;
  if (draft.boards?.length) info.textContent += ` 监控板块：${draft.boards.map(b => b.name).join('、')}。`;
  const noZone = draft.snapshot.rows.filter(s => draft.selected_codes.includes(s.code) && s.zone == null).map(s => s.code);
  if (noZone.length) info.textContent += ` 仅观察、无价格提醒：${noZone.join('、')}。`;
  if (draft.commentary_warnings?.length) info.textContent += ' 部分逐股AI解释未采纳，模型详情保留代码事实。';
  const stale = readiness?.current === false;
  const actualEvidence = Object.values(draft.model_evidence || {});
  info.textContent += stale ? ` 旧草案不可确认：${readiness.reason}。当前风险证据 ${readiness.risk_available ?? '未知'}/${readiness.candidate_count ?? draft.selected_codes.length}。`
    : ` 本草案绑定风险证据 ${actualEvidence.filter(e=>e.risk?.status==='AVAILABLE').length}/${actualEvidence.length}；不是明日涨跌概率。`;
  const button = document.createElement('button');
  button.className = 'tb-btn'; button.textContent = '确认计划并开启次日提醒';
  button.disabled = stale;
  button.onclick = async () => {
    if (draft.id !== planDraftId || busy) { info.textContent = '请确认最新一版计划'; return; }
    if (!confirm('确认这份名单并开启次日到价提醒？不自动下单。')) return;
    button.disabled = true;
    try {
      const response = await fetch('/api/assistant/plan/confirm', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({id:draft.id, confirmed:true})});
      const data = await response.json();
      if (!response.ok || !data.ok) throw new Error(data.error || '确认失败');
      info.textContent = `已冻结 ${data.count} 只，观察日 ${data.monitor_date}。程序需保持运行；微信需开启推送。`;
      info.textContent += ' 交易计划与板块已由后端统一保存，可在模型工作台查看依据。';
      planMode = false; planDraftId = null; freshPlanSession = false;
      input.placeholder = '计划已确认，可以继续正常对话';
      addMsg('bot', '计划已确认，已退出计划流程。现在可以直接正常聊天；要修改计划时再进入「制定次日计划」。');
      if (window.refreshPlanTrade) window.refreshPlanTrade();
    } catch (error) { info.textContent = error.message; button.disabled = false; }
  };
  const regenerate = document.createElement('button');
  regenerate.className = 'tb-btn'; regenerate.textContent = '按最新证据重新制定';
  regenerate.onclick = () => {
    if (busy) return;
    planMode = true; planDraftId = draft.id;
    send('请按最新证据重新制定次日观察计划，保留我已明确的数量与板块要求。');
  };
  box.append(info, button, regenerate); chatLog.appendChild(box);
  chatLog.scrollTop = chatLog.scrollHeight;
}
window.beginNextDayPlan = async () => {
  if (busy) return;
  switchPage('chat'); planMode = true;
  try {
    const data = await (await fetch('/api/assistant/state')).json();
    if (data.plan && !planDraftId && !freshPlanSession) {
      if (data.plan.presentation && !data.plan.awaiting_count) renderPlanDiscussion(data.plan, !data.plan_current);
      else addMsg('bot', data.plan_current ? data.plan.reply : `恢复的是 ${data.plan.created_at || '此前'} 的旧草案，证据已经变化；未自动调用AI，旧原文保留归档。请点击“按最新证据重新制定”。`);
      renderPlanDraft(data.plan, data.plan_readiness || {current:data.plan_current, reason:'请重新制定'});
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
  const esc = (s) => String(s ?? "").replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, "&quot;");
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

/* ---------- 实时监控板块（Stargazer 1.0：只看交易计划中的板块和计划中的股票） ---------- */
async function loadPlanSectors() {
  let d;
  try { d = await (await fetch("/api/trade_plan")).json(); } catch (e) { return; }
  const plan = d.plan;
  const pill = $("planSectorStatus"), body = $("planSectorBody");
  if (!plan) {
    pill.textContent = "暂无计划";
    body.textContent = "暂无计划：到「交易计划」页盘后选股、与 Stargazer 讨论并确认计划后，这里只监控计划内板块与计划内股票。";
    $("sectorMapBar").textContent = "无监控板块（暂无计划）";
    return;
  }
  pill.textContent = `计划日 ${plan.monitor_date || "—"}`;
  const stocks = plan.stocks || [];
  const boards = plan.boards || [];
  const lines = [];
  if (boards.length) lines.push("计划板块：" + boards.map(b => b.name || b).join("、"));
  if (stocks.length) lines.push("计划股票：" + stocks.map(s => `${s.name || ""} ${s.code}`).join("、"));
  body.textContent = lines.join("；") || "计划中没有板块或股票条目。";
  if (boards.length) {
    try {
      const monitor = await (await fetch('/api/plan_boards')).json();
      const labels = {UNRESOLVED:'板块映射待解析', UNAVAILABLE:'行情暂不可用',
        STALE:'行情过期', WAIT_SESSION:'等待计划交易时段', LOADING:'正在读取行情'};
      for (const row of monitor.rows || []) {
        const line = document.createElement('p');
        line.textContent = `${row.name} · ${monitor.status === 'EXPIRED' ? '计划已过期' : row.status === 'OK' && typeof row.pct === 'number'
          ? `${row.pct >= 0 ? '+' : ''}${row.pct.toFixed(2)}% · ${row.quote_ts || ''}` : labels[row.status] || row.status}`;
        body.append(line);
      }
    } catch (_) { body.append(document.createTextNode(' 板块行情暂不可用')); }
  }
  $("sectorMapBar").textContent = (boards.length ? "计划板块：" + boards.map(b => b.name || b).join("、") : "无计划板块")
    + (stocks.length ? `｜计划股票 ${stocks.length} 只` : "");
}

/* ---------- 信息面板（Stargazer 1.0：Corvus 信息流框架；缺席=信息模型未接入） ---------- */
async function loadInfoFeed() {
  let d;
  try { d = await (await fetch("/api/info_feed")).json(); } catch (e) { return; }
  const status = $("infoFeedStatus"), items = $("infoFeedItems");
  if (!d.connected) {
    status.textContent = "信息模型未接入";
    items.textContent = "";
    return;
  }
  status.textContent = `已接入 · 更新 ${String(d.asof || "").replace(/(\d{4})(\d{2})(\d{2})(\d{2})(\d{2})/, "$1-$2-$3 $4:$5")}`;
  items.textContent = (d.items || []).slice(0, 3).map(i => `[${i.kind || "-"}] ${i.title || ""}`).join(" ｜ ");
}

async function loadIndexFreshness() {
  const label = $('indexFreshnessStatus');
  try {
    const value = await (await fetch('/api/index_freshness')).json();
    label.textContent = value.available
      ? `指数截止 ${value.last_date || '未知'} · ${value.stale ? '数据过期，实时风险不可用' : '日级更新已核验，盘中截止另验'}`
      : '指数状态不可用，实时风险不可用';
    if (value.source_sync) {
      label.textContent += value.source_sync.status === 'OK'
        ? ' · 通达信直读已连接'
        : ` · 通达信读取失败：${value.source_sync.error || '未知原因'}`;
    }
  } catch (_) { label.textContent = '指数状态不可用，实时风险不可用'; }
}

/* ---------- 交易计划栏（已确认计划 + /api/plan_check 规则状态；模型决策待接入） ---------- */
const PLAN_CHECK_STATUSES = {NO_PLAN: "暂无计划", NOT_IN_PLAN: "不在计划", EXPIRED: "已失效",
  NEAR: "接近买入区", TRIGGERED: "已触发", BELOW_ZONE: "低于买入区", WATCHING: "观察中", OBSERVE_ONLY: "仅观察，无价格提醒", ERROR: "状态未知",
  WAIT_MINUTE_EVIDENCE:'等待完成分钟证据', WAIT_SIGNAL:'等待下杀止跌背离', NON_DOWNSIDE:'非下杀路径，不触发', RULE_TRIGGERED:'黄金规则观察成立，待人工复核', DATA_INSUFFICIENT:'证据不足，不触发'};
let planTradeCache = null;
async function loadPlanTrade() {
  let d;
  try { d = await (await fetch("/api/trade_plan")).json(); } catch (e) { return; }
  planTradeCache = d.plan || null;
  const pill = $("planTradeStatus"), list = $("planTradeList");
  if (!planTradeCache) {
    pill.textContent = "暂无计划";
    list.innerHTML = '<div class="empty">暂无计划：盘后选股 → 交易计划制定 → 讨论确认后，已确认计划的股票与关键条件显示在这里。</div>';
    return;
  }
  const plan = planTradeCache;
  pill.textContent = `观察日 ${plan.monitor_date || "—"} · ${(plan.stocks || []).length} 只`;
  const esc = (s) => String(s ?? "").replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;");
  list.innerHTML = (plan.stocks || []).map(s => `
    <div class="plan-stock" data-code="${esc(s.code)}">
      <b>${esc(s.code)}</b> ${esc(s.name || "")}
      <span class="tag">${s.observe_only && s.buy_zone == null ? '仅观察 · 无价格提醒' : Array.isArray(s.buy_zone) ? `关注区 ${Number(s.buy_zone[0]).toFixed(2)} — ${Number(s.buy_zone[1]).toFixed(2)}` : '关注区无效'}</span>
      ${s.entry_rule ? '<span class="tag">前日F618 · 须完成5分下杀止跌背离，不是报价触价即买入</span>' : ''}
      <span class="tag">有效期至 ${esc(String(s.valid_until).replace(/(\d{4})(\d{2})(\d{2})(\d{2})(\d{2})/, "$1-$2-$3 $4:$5"))}</span>
      <span class="tag plan-status" data-plan-status="${esc(s.code)}">计划状态：加载中…</span>
      <div class="c-note">失效条件：${(s.invalidations || []).map(esc).join("；") || "—"}</div>
      ${s.source === 'USER' ? `<div class="c-note">人工观察 · 未评分 · 自定义条件未执行：${(s.user_requests || []).map(esc).join('；')}</div>` : ''}
    </div>`).join("") || '<div class="empty">计划中没有股票条目。</div>';
  refreshPlanStatuses();
}
async function refreshPlanStatuses() {
  if (!planTradeCache) return;
  for (const s of planTradeCache.stocks || []) {
    const slot = document.querySelector(`[data-plan-status="${s.code}"]`);
    if (!slot) continue;
    let st = null;
    try {
      const q = curState.positions.find(p => p.code === s.code) || curState.watch.find(w => w.code === s.code) || {};
      if (q.live_price != null)
        st = await (await fetch(`/api/plan_check?code=${encodeURIComponent(s.code)}&price=${encodeURIComponent(q.live_price)}`)).json();
    } catch (e) { /* 状态标注失败不影响列表 */ }
    slot.textContent = "计划状态：" + (st && st.status ? (PLAN_CHECK_STATUSES[st.status] || st.status) : "行情不可用");
    slot.className = "tag plan-status" + (st && (st.status === "TRIGGERED" || st.status === "NEAR") ? " warn" : "");
  }
}
window.refreshPlanTrade = async () => { await loadPlanTrade(); await loadPlanSectors(); };
$("planTradeDraft").onclick = () => window.beginNextDayPlan();


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
$("runModelBtn").onclick = () => { switchPage('planTrade'); window.refreshPentacles?.(); window.refreshPlanTrade?.(); };
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
loadPlanTrade();
loadPlanSectors();
loadInfoFeed();
loadIndexFreshness();
// 信息面板轮询：信息流接入后自动刷新；未接入时保持占位（30s）
setInterval(loadInfoFeed, 30000);
setInterval(loadIndexFreshness, 30000);
setInterval(loadPlanSectors, 30000);
// 交易计划栏轮询：只按计划 JSON 规则判断（模型决策接入点在后端）
setInterval(() => {
  if ($("pagePlanTrade").classList.contains("active")) window.refreshPlanTrade();
}, 30000);

async function loadShell() {
  try {
    const data = await (await fetch('/api/shell/state')).json();
    const selected = data.models.selection.payload;
    const date = selected.data_date || '未知';
    $('modelStatus').textContent = `数据日 ${date} · ${selected.candidate_count || 0} 只`;
    $('modelHint').textContent = 'Tarot候选 · Pyxis按同日证据可用性展示';
    $('navHint').textContent = `观察日 ${selected.monitor_date || '未知'} · 稳定应用外壳`;
    $('slotSelection').textContent = `${data.display?.selection || '已接入'} · 数据日 ${date}`;
    $('slotRisk').textContent = data.display?.risk || '模型接口已接入 · 当前证据状态未知';
    $('overviewCandidates').textContent = selected.candidate_count ?? '—';
  } catch (_) { $('slotSelection').textContent = '选股快照不可用 · UNKNOWN'; }
}
document.querySelectorAll('[data-model-detail]').forEach(button => button.onclick = async () => {
  const panel = $('modelDetailPanel'), body = $('modelDetailBody');
  panel.hidden = false; body.textContent = '正在读取计划证据…';
  try {
    const value = await (await fetch(`/api/model/detail?model=${button.dataset.modelDetail}`)).json();
    $('modelDetailTitle').textContent = `${value.name || ''} · 计划分析`;
    body.replaceChildren();
    if (value.reason) { body.textContent = value.reason; return; }
    if (value.source_note) { const p = document.createElement('p'); p.textContent = value.source_note; body.append(p); }
    if (value.market_risk_summary?.length) {
      const p = document.createElement('p');
      p.textContent = `大盘风险：${value.market_risk_summary.join(' ')}`;
      body.append(p);
    }
    if (button.dataset.modelDetail === 'risk') {
      const outlook = document.createElement('p'); outlook.style.whiteSpace = 'pre-wrap'; body.append(outlook);
      const show = item => {
        const trial = Object.values(item?.computed_facts || {}).filter(f => f.status === 'TRIAL').map(f => f.text).join('\n');
        outlook.textContent = item?.text || `DeepSeek走势研判：${item?.status || '等待生成'}${item?.validation_error ? ' · '+item.validation_error : ''}。本地模型事实仍可查看；失败不自动付费重试。\n${trial}`;
      };
      show(value.outlook);
      if (!value.outlook || value.outlook.status === 'NOT_GENERATED') {
        const started = await (await fetch('/api/risk/outlook', {method:'POST'})).json(); show(started);
        if (['QUEUED','RUNNING'].includes(started.status)) {
          let polls = 0;
          const poll = async () => {
            if (panel.hidden || !outlook.isConnected || ++polls > 80) return;
            try {
              const current = await (await fetch('/api/model/detail?model=risk')).json(); show(current.outlook);
              if (['READY','AI_FAILED','PREPARATION_FAILED','EXPIRED'].includes(current.outlook?.status)) return;
            } catch (_) { outlook.textContent = '研判读取暂不可用；本地事实保留，不自动重新付费。'; return; }
            setTimeout(poll, 3000);
          };
          setTimeout(poll, 3000);
        }
      }
    }
    for (const row of value.items || []) {
      const card = document.createElement('div'); card.className = 'section-card';
      const title = document.createElement('h3'); title.textContent = `${row.code} ${row.name || ''}`;
      const text = document.createElement('p'); text.textContent = `解释：${row.commentary || '本条未生成AI解释，以下为代码提供的模型事实。'}`;
      const brief = document.createElement('p'), f = row.facts || {};
      if (button.dataset.modelDetail === 'selection') {
        const setup = {A_SDP:'强势首次分歧', B_LVB:'低位稳量'}[f.setup] || f.setup || '未登记';
        brief.textContent = `代码事实：${setup} · 参考价 ${fmtPrice(f.reference_price)} · RSI ${fmtPrice(f.rsi14)} · 距关注区 ${fmtPrice(f.distance_atr)} ATR`;
        if (f.golden_f618 != null) brief.textContent += ` · 黄金0.618位置 ${fmtPrice(f.golden_f618)}`;
        if (f.entry_rule) brief.textContent += ` · 周期节点距离 ${fmtPrice(f.entry_rule.cycle_distance_atr)} ATR（入围≤0.2）；前日F618 ${fmtPrice(f.entry_rule.f618)}，等待次日下杀止跌背离`;
        if (f.status === 'USER_OBSERVATION') brief.textContent = `${f.reason}\n${(f.user_requests || []).join('\n')}`;
      } else {
        const context = f.context || {};
        brief.textContent = f.status === 'AVAILABLE'
          ? `代码事实：截止 ${f.asof_ts}，未来120交易分钟不利下探幅度 ${fmtPrice(context.amp_232 == null ? null : context.amp_232 * 100)}%；不是明日涨跌概率。`
          : `风险证据缺失：${f.reason || '未绑定同日风险输入'}。无法据此判断明日走势。`;
        if (row.risk_lines?.length) brief.textContent = row.risk_lines.join('\n\n');
        brief.style.whiteSpace = 'pre-wrap';
      }
      const details = document.createElement('details'), caption = document.createElement('summary');
      caption.textContent = '完整模型事实与证据引用'; details.append(caption);
      const facts = document.createElement('pre'); facts.style.whiteSpace = 'pre-wrap';
      facts.textContent = JSON.stringify(row.facts, null, 2);
      details.append(facts); card.append(title, brief, text, details); body.append(card);
    }
    if (value.combined) {
      const summary = document.createElement('p'); summary.textContent = `DeepSeek综合讨论：${value.combined}`;
      body.append(summary);
    }
  } catch (_) { body.textContent = '模型详情暂不可用，原监控不受影响。'; }
});
$('reviewOpenBtn').onclick = () => { switchPage('planTrade'); $('pentaclesLearning').click(); };
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
    if (fresh) addMsg('bot', '已开始新会话。切换“制定次日计划”后生成新草案，已确认提醒不自动改变。');
  } catch (error) { addMsg('bot', error.message, 'error'); }
  finally { busy = false; sendBtn.disabled = false; }
}
$('newChatBtn').onclick = () => restoreChat(true);
$('restoreChatBtn').onclick = () => restoreChat();
