// Isolated actual pentacles.js regression; no browser, server or API requests.
const fs = require('node:fs');
const vm = require('node:vm');
const assert = require('node:assert/strict');
const path = require('node:path');
const elements = new Map(), requests = [], responses = [];
function el(id) {
  if (!elements.has(id)) elements.set(id, {value:'', textContent:'', hidden:true, disabled:false,
    classList:{contains:()=>false, toggle:()=>{}}, replaceChildren:()=>{}});
  return elements.get(id);
}
const environment = {document:{getElementById:el, querySelectorAll:()=>[]}, window:{}, Date,
  setInterval:()=>1, clearInterval:()=>{}, confirm:()=>true,
  fetch:async (url)=>{
    requests.push(url);
    assert.ok(responses.length, `Unexpected request ${url}`);
    const body = responses.shift();
    return {ok:true, json:async()=>body};
  }};
vm.runInNewContext(fs.readFileSync(path.join(__dirname, 'web/pentacles.js'), 'utf8'), environment);
const settle = () => new Promise(resolve=>setImmediate(resolve));
const old = {id:'old', day:20261014, commentary:'OLD_BODY'};
(async()=>{
  el('learningDay').value = '2026-10-15';
  responses.push({last_review:old, learning_path:'local'});
  await el('pentaclesLearning').onclick();
  assert.match(el('learningPreview').textContent, /历史预览/);
  assert.equal(el('learningConfirm').disabled, true);
  const count = requests.length;
  await el('learningConfirm').onclick();
  assert.equal(requests.length, count); // No stale ID available for confirmation.

  responses.push({ok:true}, {review:{running:false,error:'finish_reason=length',draft:null}});
  await el('learningGenerate').onclick();
  await settle();
  assert.match(el('learningStatus').textContent, /length/);
  assert.doesNotMatch(el('learningPreview').textContent, /OLD_BODY/);
  assert.match(el('learningPreview').textContent, /本次未生成/);
  assert.equal(el('learningConfirm').disabled, true);

  responses.push({ok:true}, {review:{running:false, draft:{...old, day:20261014}}});
  await el('learningGenerate').onclick(); await settle();
  assert.match(el('learningStatus').textContent, /其他观察日/);
  assert.equal(el('learningConfirm').disabled, true);

  const fresh = {id:'fresh',day:20261015,commentary:'AI_ONLY',report_markdown:'LOCAL_FACTS_AND_AI',
    facts:{summary:{valid_daily_rows:200},candidate_count:200}};
  responses.push({ok:true}, {review:{running:false,draft:fresh}});
  await el('learningGenerate').onclick(); await settle();
  assert.equal(el('learningPreview').textContent, 'LOCAL_FACTS_AND_AI');
  assert.equal(el('learningConfirm').disabled, false);
  el('learningDay').value = '2026-10-16';
  el('learningDay').onchange();
  assert.equal(el('learningConfirm').disabled, true);
  assert.doesNotMatch(el('learningPreview').textContent, /LOCAL_FACTS_AND_AI/);
  console.log('UI regression PASS: history, truncation, wrong day, fresh preview and date change');
})().catch(error=>{console.error(error);process.exitCode=1;});
