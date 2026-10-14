/* Independent demo view; no shared model hints, trading calls or live signals. */
(() => {
  const el = id => document.getElementById(id);
  let state = {}, pool = 'ALL', pending = false;
  const number = (n, digits = 2) => typeof n === 'number' && Number.isFinite(n) ? n.toFixed(digits) : '—';
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
    el('pentaclesMonitor').checked = Boolean(state.enabled);
    el('pentaclesRun').disabled = Boolean(state.running);
    el('pentaclesRun').textContent = state.running ? '正在分析…' : '盘后总结 · 分析选股';
    const reminder = state.reminder || {};
    const messages = [state.enabled ? '次日到价提醒已开启' : '次日到价提醒关闭', reminder.message || '等待生成名单'];
    if (snapshot) messages.push(`数据日 ${snapshot.data_date}${state.stale ? ' · 名单过期/数据未就绪，不提醒' : ''}`,
      `观察日 ${snapshot.monitor_date || '未登记'}`, snapshot.ranking);
    if (state.running) messages.push('本地分析中，保留上一份结果');
    if (state.error) messages.push(`分析不可用：${state.error}`);
    if (state.journal?.error) messages.push(`日记记录停止：${state.journal.error}`);
    el('pentaclesStatus').textContent = messages.join(' ｜ ');
    const rows = (snapshot?.rows || []).filter(row => pool === 'ALL' || row.setup === pool);
    el('pentaclesCount').textContent = `${rows.length} 只观察候选`;
    const body = el('pentaclesRows');
    body.replaceChildren();
    if (!rows.length) {
      const tr = body.insertRow(), td = tr.insertCell();
      td.colSpan = 8; td.className = 'empty'; td.textContent = snapshot ? '该形态暂无候选。' : '暂无观察快照。点击盘后总结开始本地分析。';
    }
    for (const row of rows) {
      const tr = body.insertRow(), zone = row.zone, live = reminder.rows?.[row.code];
      const alert = reminder.alerts?.[`${snapshot.monitor_date}:${row.code}`];
      const statuses = {WAIT_PRICE:'等待到价', ALERT_RECORDED:'到价提醒已记录', QUOTE_STALE:'报价过期',
        INVALID_PRICE:'无效价格', PRICE_BASIS_MISMATCH:'前收/复权口径不一致', QUOTE_OR_ZONE_UNKNOWN:'报价/区域未知'};
      const delivery = {SENT:'微信已发送', FAILED:'微信发送失败', WECHAT_DISABLED:'微信未开启', PENDING:'等待推送'};
      const cells = [row.rank, `${row.code} ${row.name || ''}\n${row.setup === 'A_SDP' ? '强势首次分歧' : '低位稳量底部'}`,
        number(row.reference_price), zone ? `${number(zone.zone_lower)} — ${number(zone.zone_upper)}` : '暂无合格区域',
        zone ? zone.zone_sources.map(s => ({MA20:'MA20',BOLL_L:'BOLL下轨',PREV_LOW:'前日低点'}[s] || s)).join(' + ') : '—',
        number(row.distance_atr), number(row.rsi14, 1), live ?
          `现价 ${number(live.price)} · ${statuses[live.status] || live.status}\n${alert ? delivery[alert.delivery] || alert.delivery : live.time || ''}` :
          row.status === 'REFERENCE_ONLY' ? (state.stale ? '过期参考，不提醒' : '关注区域已登记') : '数据 / 区域不可用'];
      for (const value of cells) tr.insertCell().textContent = String(value ?? '—');
    }
  }
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
  el('pentaclesPlan').onclick = () => window.beginNextDayPlan();
  let reviewId = null, reviewPoll = null;
  el('learningDay').onchange = () => {
    reviewId = null; el('learningConfirm').disabled = true;
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
          reviewId = data.last_review.id; el('learningPreview').textContent = data.last_review.commentary;
          el('learningConfirm').disabled = false;
          el('learningStatus').textContent = `已恢复复盘 ${data.last_review.day}；保存路径 ${data.learning_path}`;
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
        reviewId = job.draft.id;
        el('learningPreview').textContent = job.draft.commentary;
        el('learningConfirm').disabled = false;
        const s = job.draft.facts.summary;
        el('learningStatus').textContent = `预览 · 观察日 ${job.draft.day} · 有效日K ${s.valid_daily_rows}/${job.draft.facts.candidate_count} · 事实数字由程序计算，以下解释需人工确认。`;
      }
    } catch (error) {
      clearInterval(reviewPoll); reviewPoll = null;
      el('learningGenerate').disabled = false; el('learningStatus').textContent = error.message;
    }
  }
  el('learningGenerate').onclick = async () => {
    const day = Number(el('learningDay').value.replaceAll('-', ''));
    el('learningGenerate').disabled = true; el('learningConfirm').disabled = true;
    el('learningStatus').textContent = '正在读取本机行情并请 DeepSeek 整理，提醒监控照常运行…';
    try {
      await api('/api/assistant/review', {day});
      clearInterval(reviewPoll); reviewPoll = setInterval(pollReview, 1500); pollReview();
    } catch (error) { el('learningStatus').textContent = error.message; el('learningGenerate').disabled = false; }
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
  document.querySelectorAll('.pentacles-filters button').forEach(button => button.onclick = () => {
    pool = button.dataset.pool;
    document.querySelectorAll('.pentacles-filters button').forEach(b => b.classList.toggle('active', b === button));
    render();
  });
  setInterval(() => {
    if (el('pagePentacles').classList.contains('active') && (state.running || state.enabled)) window.refreshPentacles();
  }, 3000);
})();
