/* Independent demo view; no shared model hints, trading calls or live signals. */
(() => {
  const el = id => document.getElementById(id);
  let state = {}, pool = 'ALL', pending = false;
  const number = (n, digits = 2) => typeof n === 'number' && Number.isFinite(n) ? n.toFixed(digits) : '—';
  function goldenLabel(row) {
    const rule = row.entry_rule || {}, hi = rule.anchor_hi, lo = rule.anchor_lo, node = rule.f618;
    if (rule.policy === 'GOLDEN_RESONANCE_CROSSDAY_V4') return '低位多档共振＋ATR · 下杀4%观察';
    if (![hi, lo, node].every(Number.isFinite) || hi <= lo) return 'P日冻结黄金阶梯';
    const ratio = [.236, .382, .5, .618, .786].find(r => Math.abs(hi-r*(hi-lo)-node) <= .00011);
    return ratio === undefined ? 'P日冻结黄金阶梯' : `波段${number(ratio*100, 1)}%回撤位`;
  }
  function resonanceText(row) {
    const bands = (row.entry_rule?.observation_map?.bands || []).filter(b =>
      Number.isFinite(b.lower) && Number.isFinite(b.upper) && b.lower > 0 && b.upper <= row.reference_price &&
      new Set((b.sources || []).map(s => `${s.low_day}:${s.high_day}`)).size >= 2);
    bands.sort((a,b) => b.node-a.node);
    return bands.map(b => {
      const count = new Set(b.sources.map(s => `${s.low_day}:${s.high_day}`)).size;
      const peaks = new Set(b.sources.map(s => s.high_day)).size;
      const zone = row.entry_rule?.policy === 'GOLDEN_RESONANCE_CROSSDAY_V4'
        ? (b.observable ? ` · 观察${number(b.lower,4)}—${number(b.zone_upper,4)}` : ' · 历史参考，未满足本轮低位观察门') : '';
      return `${number(b.lower,4)}—${number(b.upper,4)} · ${count}源${peaks === 1 ? '（共享高点，非独立投票）' : `／${peaks}组高点`}${zone}`;
    }).join('\n') || '暂无两源共振带；单档不冒充共振';
  }
  const gateNames = {VOLUME_GATE_NOT_PASSED:'量能组合未过门',RISE_NOT_ABOVE_15PCT:'前波涨幅未超15%',
    RETRACE_OUTSIDE_23_6_TO_78_6:'回撤不在23.6%—78.6%',PANIC_VOLUME_BELOW_1_25X:'阴线量未超前8日均量1.25倍',
    FIRST_LEG_VOLUME_NOT_DECREASING:'首段下跌量能未逐日递减'};
  async function api(url, body) {
    const response = await fetch(url, body === undefined ? {} : {
      method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(body)
    });
    const data = await response.json();
    if (!response.ok || data.ok === false) throw new Error(data.error || '请求失败');
    return data;
  }
  function render() {
    const snapshot = state.snapshot;
    const v2 = snapshot?.rule_version === 'TAROT_A_SECOND_CASH_V2';
    if (v2 && pool === 'B_LVB') pool = 'ALL';
    document.querySelectorAll('.pentacles-filters button[data-pool]').forEach(button => {
      button.hidden = button.dataset.pool === 'B_LVB' ? v2 : button.dataset.pool === 'SECOND_PULLBACK' ? !v2 : false;
    });
    el('pentaclesMonitor').checked = Boolean(state.enabled);
    el('pentaclesRun').disabled = Boolean(state.running);
    el('pentaclesRun').textContent = state.running ? '正在分析…' : '盘后总结 · 分析选股';
    const reminder = state.reminder || {};
    const messages = [state.enabled ? '次日观察提醒已开启' : '次日观察提醒关闭', reminder.message || '等待生成名单'];
    if (snapshot) messages.push(`数据日 ${snapshot.data_date}${state.stale ? ' · 名单过期/数据未就绪，不提醒' : ''}`,
      `观察日 ${snapshot.monitor_date || '未登记'}`, snapshot.ranking);
    if (v2) messages.push(`A/二段规则与特征通过 ${snapshot.candidate_population_count ?? '未登记'} 只 → 模型排序展示 ${snapshot.rows.length} 只（最多${snapshot.default_candidate_count ?? 10}只）；另有 ${snapshot.excluded_rows?.length || 0} 只数据/计算异常，未进入评分，明细见下方`);
    else if (snapshot?.rule_version) messages.push(`原共性 ${snapshot.common_candidate_count ?? '未登记'} 只 → 黄金位置入围 ${snapshot.rows.length} 只；剔除 ${snapshot.excluded_rows?.length || 0} 只，明细见下方`);
    if (state.running) messages.push('本地分析中，保留上一份结果');
    if (state.error) messages.push(`分析不可用：${state.error}`);
    if (snapshot?.manual_review_rows?.length) messages.push(`另列人工复核 ${snapshot.manual_review_rows.length} 只，未过正式门，不自动触发`);
    if (state.journal?.error) messages.push(`日记记录停止：${state.journal.error}`);
    el('pentaclesStatus').textContent = messages.join(' ｜ ');
    const excluded = snapshot?.excluded_rows || [];
    el('pentaclesExcludedDetails').hidden = !excluded.length;
    el('pentaclesExcludedSummary').textContent = `查看未入选明细（${excluded.length}只${v2 ? '数据/计算异常，非模型低分' : ''}）`;
    const excludedBody = el('pentaclesExcludedRows');
    excludedBody.replaceChildren();
    const reasons = {
      INVALID_CAUSAL_DAILY_VALUES:'历史日线价格/量值校验失败，形态和模型评分前已排除；需核对原行情',
      INFINITE_FEATURE:'特征出现无穷值，未评分',
      NO_CAUSAL_WAVE_MAP:'没有已确认的有效波段地图，未加入补充名单',
      OUTSIDE_CYCLE_GOLDEN_BAND:'未满足旧版周期黄金位距离要求',
      GOLDEN_INPUT_UNAVAILABLE:'黄金位置计算所需数据不足',
      PREVIOUS_F618_ABOVE_CLOSE:'旧版前日61.8%位置高于收盘，关注区间无效'
    };
    for (const row of excluded) {
      const tr = excludedBody.insertRow();
      tr.insertCell().textContent = `${row.code}${row.name ? ' '+row.name : ''}`;
      tr.insertCell().textContent = reasons[row.reason] || row.reason || '原因未登记';
    }
    const rows = (snapshot?.rows || []).filter(row => pool === 'ALL' || row.setup === pool);
    const manualRows = (snapshot?.manual_review_rows || []).filter(row => pool === 'ALL' || row.setup === pool);
    el('pentaclesCount').textContent = `${rows.length} 只正式候选${manualRows.length ? ` ＋ ${manualRows.length}只人工复核` : ''}`;
    // 选股完成出现「交易计划制定」：名单就绪且观察日在未来（与后端 plan_chat 门禁同口径）
    const snap = state.snapshot || {};
    const today = new Date(Date.now() - new Date().getTimezoneOffset() * 60000).toISOString().slice(0, 10).replaceAll('-', '');
    const ready = Boolean(snap.plan_ready) && Number(snap.monitor_date) > Number(today);
    const draftBtn = el('planTradeDraft');
    if (draftBtn) {
      draftBtn.hidden = !ready;
      el('planTradeHint').textContent = ready
        ? `名单已就绪 · 观察日 ${snap.monitor_date} · 点击与 Stargazer 讨论次日计划`
        : '盘后选股完成后出现「交易计划制定」；与 Stargazer 讨论并人工确认后写入计划。';
    }
    const body = el('pentaclesRows');
    body.replaceChildren();
    if (!rows.length) {
      const tr = body.insertRow(), td = tr.insertCell();
      td.colSpan = 9; td.className = 'empty'; td.textContent = snapshot ? '该形态暂无正式候选。' : '暂无观察快照。点击盘后总结开始本地分析。';
    }
    for (const row of [...rows,...manualRows]) {
      const manual = manualRows.includes(row);
      if (manual && row === manualRows[0]) {
        const heading = body.insertRow().insertCell(); heading.colSpan = 9;
        heading.textContent = '以下未过门，仅人工复核；按冻结模型参考分排序，超出已验证适用范围，不进入正式计划或自动买入触发。';
      }
      const tr = body.insertRow(), zone = row.zone, live = reminder.rows?.[row.code];
      tr.className = 'pentacles-row';
      tr.title = '点击查看影子建议（只记录不影响，买卖由你决定）';
      if (!manual) tr.onclick = () => loadShadowAdvice(row.code);
      else tr.title = row.review_scope;
      const alert = reminder.alerts?.[`${snapshot.monitor_date}:${row.code}`];
      const statuses = {WAIT_PRICE:'等待到价', ALERT_RECORDED:'到价提醒已记录', QUOTE_STALE:'报价过期',
        INVALID_PRICE:'无效价格', PRICE_BASIS_MISMATCH:'前收/复权口径不一致', QUOTE_OR_ZONE_UNKNOWN:'报价/区域未知',
        WAIT_SIGNAL:'等待下杀止跌背离', RULE_TRIGGERED:'规则观察成立，待人工复核', DATA_INSUFFICIENT:'分钟/尺度证据不足', NON_DOWNSIDE:'非下杀路径，不触发'};
      const delivery = {SENT:'微信已发送', FAILED:'微信发送失败', WECHAT_DISABLED:'微信未开启', PENDING:'等待推送'};
      const cells = [manual ? `人工${row.rank}` : row.rank, `${row.code} ${row.name || ''}\n${({A_SDP:'一段强势回调', SECOND_PULLBACK:'二段回踩', B_LVB:'低位稳量底部'})[row.setup] || row.setup || '形态未登记'}`,
        number(row.reference_price), zone ? `${number(zone.zone_lower)} — ${number(zone.zone_upper)}` : '暂无合格区域',
        resonanceText(row),
        zone ? zone.zone_sources.map(s => ({MA20:'MA20',BOLL_L:'BOLL下轨',PREV_LOW:'前日低点',PREVIOUS_DAY_F618:'前日0.618（等待完整触发）',P_FROZEN_GOLDEN_LADDER:goldenLabel(row),V4_LOW_RESONANCE_PLUS_ATR:goldenLabel(row)}[s] || s)).join(' + ') : '—',
        number(row.distance_atr), number(row.rsi14, 1), live ?
          `现价 ${number(live.price)} · ${statuses[live.status] || live.status}\n${alert ? delivery[alert.delivery] || alert.delivery : live.time || ''}` :
          row.status === 'GOLDEN_PLAN_READY' ? '黄金计划已登记，等待次日完成5分条件' : row.status === 'WAIT_RESONANCE_ZONE' ? '候选分数保留；暂无合格低位共振观察区，不自动触发' : row.status === 'REFERENCE_ONLY' ? (state.stale ? '过期参考，不提醒' : '关注区域已登记') : '数据 / 区域不可用'];
      if (manual) cells[cells.length-1] = `未过门：${(row.failed_gates || []).map(k=>gateNames[k] || k).join('；')}\n参考分 ${number(row.selection_score,6)}；不自动触发`;
      for (const value of cells) tr.insertCell().textContent = String(value ?? '—');
    }
  }
  // ---------- 影子建议（Stargazer 1.0：只记录不影响；DeepSeek 未配置/失败仅提示） ----------
  const fmtTs14 = s => /^\d{14}$/.test(String(s || ''))
    ? `${s.slice(0, 4)}-${s.slice(4, 6)}-${s.slice(6, 8)} ${s.slice(8, 10)}:${s.slice(10, 12)}` : (s || '—');
  function shadowLine(label, items) {
    if (!items || !items.length) return null;
    const div = document.createElement('div');
    div.className = 'shadow-advice-line';
    const b = document.createElement('b'); b.textContent = label; div.append(b);
    const ul = document.createElement('ul');
    for (const t of items) { const li = document.createElement('li'); li.textContent = t; ul.append(li); }
    div.append(ul);
    return div;
  }
  function renderShadowAdvice(code, date, data, freshness) {
    const body = el('shadowAdviceBody');
    body.replaceChildren();
    const head = document.createElement('div');
    head.className = 'shadow-advice-sub';
    head.textContent = `${code} · 证据日 ${date}` +
      (data.journal_ref ? ` · 影子日记 ${data.journal_ref}` : '');
    body.append(head);
    if (data.status === 'OK' && data.advice) {
      const a = data.advice;
      const title = document.createElement('div');
      title.className = 'shadow-advice-option';
      title.textContent = `建议：${a.title || a.option}（${a.option}）`;
      body.append(title);
      const hint = document.createElement('div');
      hint.className = 'shadow-advice-line';
      hint.textContent = `下一步：${a.next_hint || '—'}` +
        (a.information_gap ? '；存在信息缺口（Corvus 未接入，缺口≠无利空）' : '');
      body.append(hint);
      const conf = document.createElement('div');
      conf.className = 'shadow-advice-line';
      conf.textContent = `置信度 ${typeof a.confidence === 'number' ? a.confidence.toFixed(2) : '—'}` +
        '（模型自信度，不等于胜率）';
      body.append(conf);
      for (const [label, key] of [['支持', 'supporting'], ['反证', 'counterevidence'],
                                  ['等待条件', 'wait_conditions'], ['重评/失效条件', 'invalidations']]) {
        const line = shadowLine(label, a[key]); if (line) body.append(line);
      }
      const valid = document.createElement('div');
      valid.className = 'shadow-advice-line';
      valid.textContent = `有效期至 ${fmtTs14(a.valid_until_ts)}` +
        (data.expired ? '（已过期，仅供回看）' : '') + (data.reused ? '（复用未过期建议，未重复扣额度）' : '');
      body.append(valid);
      return freshnessAvailable(freshness);
    }
    function freshnessAvailable(freshness) {
      return freshness && freshness.available !== false;
    }
    const box = document.createElement('div');
    if (data.status === 'UNSCORABLE') {
      box.className = 'shadow-advice-unscorable';
      box.textContent = `该候选当前不可评分：${data.reason || '原因未知'}`;
    } else {
      box.className = 'shadow-advice-error';
      box.textContent = data.reason || '本次建议不可用';
    }
    body.append(box);
  }
  async function loadShadowAdvice(code, replayDate = null) {
    const panel = el('shadowAdvicePanel'), body = el('shadowAdviceBody');
    panel.hidden = false;
    body.replaceChildren();
    body.textContent = `正在读取影子建议：${code} …`;
    const date = String(replayDate || state.snapshot?.monitor_date || state.snapshot?.data_date || '');
    if (!date) { body.textContent = '暂无观察快照，无法定位证据包。'; return; }
    let data, freshness = null;
    try {
      const res = await fetch(`/api/shadow_evidence?code=${encodeURIComponent(code)}&date=${encodeURIComponent(date)}`);
      data = await res.json();
    } catch (error) {
      body.textContent = '本次建议不可用（网络或服务不可达）';
      return;
    }
    if (data.status === 'EVIDENCE_READY' || data.mode === 'HISTORICAL_REPLAY') {
      body.replaceChildren();
      const title = document.createElement('div');
      title.className = 'shadow-advice-option';
      title.textContent = `历史回放 ${date} · ${code} · ${fmtTs14(data.asof_ts)}`;
      body.append(title);
      const note = document.createElement('div');
      note.className = 'shadow-advice-sub';
      note.textContent = '冻结模型的历史证据，不代表当前行情建议。DeepSeek 未调用；正式观察未启动。';
      body.append(note);
      if (data.reason) {
        const reason = document.createElement('div');
        reason.textContent = `不可评分：${data.reason}`;
        body.append(reason);
      }
      const candidate = data.candidates?.[0] || {}, risk = data.risk_context || {};
      const summary = document.createElement('div');
      summary.className = 'shadow-advice-line';
      summary.textContent = `盘后选股评分 ${number(candidate.p_close_score, 3)} · 昨日0.618位置 ${number(candidate.f618)} · 距离 ${number(candidate.plan_dist_atr)} ATR`;
      body.append(summary);
      const amplitude = document.createElement('div');
      amplitude.className = 'shadow-advice-line';
      amplitude.textContent = `未来120交易分钟不利下探幅度：主版 ${number(typeof risk.amp_232 === 'number' ? risk.amp_232 * 100 : null, 3)}% · 对照版 ${number(typeof risk.amp_266 === 'number' ? risk.amp_266 * 100 : null, 3)}%（幅度不等于概率）`;
      body.append(amplitude);
      const coverage = document.createElement('div');
      coverage.className = 'shadow-advice-sub';
      coverage.textContent = `特征状态 ${risk.feature_status || '未登记'} · 缺失 ${Array.isArray(risk.feature_missing) ? risk.feature_missing.length : '未知'} 项；原生缺失保留，仅供工程回放。`;
      body.append(coverage);
      const trigger = document.createElement('div');
      trigger.textContent = candidate.trigger ? '触发状态已记录，详见证据' : '止跌／背离触发：本批未重放，状态未知';
      body.append(trigger);
      const details = document.createElement('details'), caption = document.createElement('summary');
      caption.textContent = '查看完整证据与来源';
      details.append(caption);
      const evidence = document.createElement('pre');
      evidence.style.whiteSpace = 'pre-wrap';
      evidence.textContent = JSON.stringify({candidates: data.candidates,
        risk_context: data.risk_context, risk_missing: data.risk_missing,
        corvus_gap: data.corvus_gap, pack_hash: data.pack_hash}, null, 2);
      details.append(evidence);
      body.append(details);
      return;
    }
    // 指数新鲜度仅是本地状态展示，证据可用性仍由同刻构建门判断。
    try { freshness = await (await fetch('/api/index_freshness')).json(); } catch (e) { freshness = null; }
    renderShadowAdvice(code, date, data || {status: 'ERROR', reason: '本次建议不可用'}, freshness);
    if (data && data.status === 'OK' && !(freshness && freshness.available !== false)) {
      const note = document.createElement('div');
      note.className = 'shadow-advice-unscorable';
      note.textContent = '数据过期：指数新鲜度未接入（index_minutes_v1 缺席），以上建议基于未核验的数据时点。';
      el('shadowAdviceBody').append(note);
    }
  }
  el('shadowAdviceClose').onclick = () => { el('shadowAdvicePanel').hidden = true; };
  el('shadowReplayOpen').onclick = () => {
    const [date, code] = el('shadowReplaySelect').value.split(':');
    if (date && code) loadShadowAdvice(code, date);
  };
  async function loadReplayCatalog() {
    try {
      const catalog = await (await fetch('/api/shadow_evidence')).json();
      const select = el('shadowReplaySelect');
      select.replaceChildren();
      for (const item of catalog.items || []) {
        const option = document.createElement('option');
        option.value = `${item.market_date}:${item.code}`;
        option.textContent = `${item.market_date} · ${item.code}`;
        select.append(option);
      }
      if (select.options.length) el('shadowAdvicePanel').hidden = false;
    } catch (_) { /* 缺历史文件不影响实时监控。 */ }
  }
  loadReplayCatalog();
  window.refreshPentacles = async () => {
    if (pending) return;
    pending = true;
    try { state = await api('/api/pentacles/state'); render(); }
    catch (error) { el('pentaclesStatus').textContent = `读取失败：${error.message}`; }
    finally { pending = false; }
  };
  el('pentaclesRun').onclick = async () => {
    el('pentaclesRun').disabled = true;
    try { await api('/api/pentacles/run', {}); await window.refreshPentacles(); }
    catch (error) { el('pentaclesStatus').textContent = error.message; el('pentaclesRun').disabled = false; }
  };
  let reviewId = null, reviewPoll = null;
  const selectedReviewDay = () => Number(el('learningDay').value.replaceAll('-', ''));
  function clearReview(message) {
    reviewId = null;
    el('learningConfirm').disabled = true;
    el('learningPreview').textContent = message;
  }
  el('learningDay').onchange = () => {
    clearReview('尚未生成所选观察日的复盘；已有学习记录未改动。');
    el('learningStatus').textContent = '日期已改变，请重新生成该观察日的预览。';
  };
  el('pentaclesLearning').onclick = async () => {
    const panel = el('learningPanel'); panel.hidden = !panel.hidden;
    const now = new Date(), day = new Date(now.getTime() - now.getTimezoneOffset() * 60000).toISOString().slice(0, 10);
    el('learningDay').value ||= day;
    if (!panel.hidden) {
      try {
        const data = await api('/api/assistant/state');
        if (data.last_review) {
          const draft = data.last_review, matches = draft.day === selectedReviewDay();
          reviewId = matches ? draft.id : null;
          el('learningPreview').textContent = (matches ? '' : `历史预览 · 观察日 ${draft.day}，不是当前所选日期的新结果\n\n`) +
            (draft.report_markdown || draft.commentary);
          el('learningConfirm').disabled = !matches;
          el('learningStatus').textContent = matches ? `已恢复所选观察日复盘 ${draft.day}；保存路径 ${data.learning_path}` :
            `仅展示历史复盘 ${draft.day}；请生成所选日期的新预览，不能确认旧篇。`;
        } else {
          clearReview('尚无已生成的复盘，请生成所选日期的预览。');
        }
      } catch (error) { el('learningStatus').textContent = error.message; }
    }
  };
  async function pollReview() {
    try {
      const data = await api('/api/assistant/state'), job = data.review;
      if (job.running) return;
      clearInterval(reviewPoll); reviewPoll = null; el('learningGenerate').disabled = false;
      if (job.error) throw new Error(job.error);
      if (job.draft) {
        if (job.draft.day !== selectedReviewDay()) throw new Error('生成结果属于其他观察日；请重新生成所选日期的复盘');
        reviewId = job.draft.id;
        el('learningPreview').textContent = job.draft.report_markdown || job.draft.commentary;
        el('learningConfirm').disabled = false;
        const s = job.draft.facts.summary;
        el('learningStatus').textContent = `预览 · 观察日 ${job.draft.day} · 有效日K ${s.valid_daily_rows}/${job.draft.facts.candidate_count} · 事实数字由程序计算，以下解释需人工确认。`;
      } else throw new Error('复盘任务没有生成草案，应用可能已重启；请重新生成');
    } catch (error) {
      clearInterval(reviewPoll); reviewPoll = null;
      clearReview('本次未生成可确认的新复盘，学习记录未追加。请核对上方原因后重新生成。');
      el('learningGenerate').disabled = false; el('learningStatus').textContent = error.message;
    }
  }
  el('learningGenerate').onclick = async () => {
    const day = selectedReviewDay();
    el('learningGenerate').disabled = true; el('learningConfirm').disabled = true;
    clearReview(`正在生成观察日 ${day} 的新预览；旧学习记录保持不变。`);
    el('learningStatus').textContent = '正在读取本机行情并请 DeepSeek 整理，提醒监控照常运行…';
    try {
      await api('/api/assistant/review', {day});
      clearInterval(reviewPoll); reviewPoll = setInterval(pollReview, 1500); pollReview();
    } catch (error) {
      clearReview('本次未生成可确认的新复盘，学习记录未追加。');
      el('learningStatus').textContent = error.message; el('learningGenerate').disabled = false;
    }
  };
  el('learningConfirm').onclick = async () => {
    if (!reviewId || !confirm('确认追加这篇复盘？保留旧篇，AI解释不作为训练标签。')) return;
    el('learningConfirm').disabled = true;
    try {
      const data = await api('/api/assistant/review/confirm', {id:reviewId, confirmed:true});
      el('learningStatus').textContent = `${data.duplicate ? '已保存，不重复追加' : '已追加'}：${data.path}`;
    } catch (error) { el('learningStatus').textContent = error.message; el('learningConfirm').disabled = false; }
  };
  el('pentaclesMonitor').onchange = async event => {
    try { state = await api('/api/pentacles/monitor', {enabled:event.target.checked}); render(); }
    catch (error) { event.target.checked = !event.target.checked; el('pentaclesStatus').textContent = error.message; }
  };
  async function loadJournal(day) {
    const report = await api('/api/pentacles/journal' + (day ? `?day=${encodeURIComponent(day)}` : ''));
    const select = el('pentaclesJournalDay');
    select.replaceChildren();
    for (const date of report.days) {
      const option = document.createElement('option');
      option.value = date; option.textContent = date; option.selected = date === report.day;
      select.append(option);
    }
    el('pentaclesJournalText').textContent = report.markdown;
    if (report.error) el('pentaclesStatus').textContent = `日记记录停止：${report.error}`;
  }
  el('pentaclesJournal').onclick = async () => {
    const panel = el('pentaclesJournalPanel');
    panel.hidden = !panel.hidden;
    if (!panel.hidden) {
      try { await loadJournal(); }
      catch (error) { el('pentaclesJournalText').textContent = error.message; }
    }
  };
  el('pentaclesJournalDay').onchange = async event => {
    try { await loadJournal(event.target.value); }
    catch (error) { el('pentaclesJournalText').textContent = error.message; }
  };
  el('pentaclesJournalDownload').onclick = () => {
    const blob = new Blob([el('pentaclesJournalText').textContent], {type:'text/markdown;charset=utf-8'});
    const url = URL.createObjectURL(blob), link = document.createElement('a');
    link.href = url; link.download = `星币侍从观察日记_${el('pentaclesJournalDay').value || '空'}.md`;
    link.click(); setTimeout(() => URL.revokeObjectURL(url), 1000);
  };
  document.querySelectorAll('.pentacles-filters button[data-pool]').forEach(button => button.onclick = () => {
    pool = button.dataset.pool;
    document.querySelectorAll('.pentacles-filters button').forEach(b => b.classList.toggle('active', b === button));
    render();
  });
  setInterval(() => {
    if (el('pagePlanTrade').classList.contains('active') && (state.running || state.enabled)) window.refreshPentacles();
  }, 3000);
})();
