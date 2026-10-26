"""Real projected candidates, old draft restore, mock AI, actual confirm/browser."""
import base64
import json
from pathlib import Path
import sys
import tempfile
import time
from types import SimpleNamespace
from unittest.mock import patch

from flask import request
import accept_handoff_browser as browser
import assistant_workflow as workflow
import golden_rules
import paths
import plan_evidence
import postclose_risk_runtime as runtime
import trade_plan_v1

DESKTOP=Path(__file__).resolve().parent
sys.path.insert(0,str(DESKTOP.parent/'stargazer_ml/v5_hybrid/green_common_profile'))
from gcp_golden_snapshot_v1 import transform


def main():
    out=DESKTOP/'golden_rules_acceptance_20261021_v3'
    out.mkdir(exist_ok=True)
    if (out/'real_flow.json').exists():
        raise ValueError('ACCEPTANCE_ALREADY_COMPLETE')
    parent=json.loads((DESKTOP/'dist/data/pentacles_demo/snapshot.json').read_bytes())
    if parent.get('rule_version'):
        reference=parent['projection_parent_hash']
        options=list((DESKTOP/'dist/data/pentacles_demo').glob('snapshot_before_golden_*.json'))
        parent=next(json.loads(p.read_bytes()) for p in options if workflow.digest(json.loads(p.read_bytes()))==reference)
    snapshot=transform(parent)
    assert golden_rules.is_projection_of(snapshot,parent)
    codes=[r['code'] for r in snapshot['rows']]
    assert codes and len(codes)<len(parent['rows'])
    with (out/'snapshot_new.json').open('x',encoding='utf-8') as stream:
        json.dump(snapshot,stream,ensure_ascii=False,allow_nan=False,indent=2)
    oldfile=next(p for p in (DESKTOP/'dist/data/assistant').glob('plan_*.json')
                 if json.loads(p.read_bytes()).get('snapshot_hash')==workflow.digest(parent))
    old=json.loads(oldfile.read_bytes())
    original=browser.local_acceptance_only
    def routes():
        if request.path in ('/api/chat','/api/assistant/state','/api/assistant/plan/confirm',
                '/api/trade_plan','/api/model/detail','/api/plan_check'):
            return None
        return original()
    browser.server.app.before_request_funcs[None]=[routes]
    browser.OUT=out
    answer=dict(reply='新证据模拟讨论，不是真实供应商建议',selected_codes=codes,boards=[],
                stock_comments=[dict(code=c,selection='观察0.618及RSI14',risk='未来120交易分钟',
                    evidence_refs=[f'tarot:{snapshot["data_date"]}:{c}']) for c in codes])
    response=SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps(answer)))])
    result={}
    def checks(evaluate,call):
        evaluate('window.beginNextDayPlan()')
        text=evaluate("document.getElementById('chatLog').textContent")
        assert '旧草案' in text and '按最新证据重新制定' in text and 'risk_evidence为空' not in text, text
        assert mock.call_count==0
        assert evaluate("[...document.querySelectorAll('#chatLog button')].find(b=>b.textContent==='确认计划并开启次日提醒').disabled")
        evaluate("[...document.querySelectorAll('#chatLog button')].find(b=>b.textContent==='按最新证据重新制定').click()")
        for _ in range(100):
            if evaluate('!busy'): break
            time.sleep(.1)
        text=evaluate("document.getElementById('chatLog').textContent")
        assert '新证据模拟讨论' in text and '对话失败' not in text, text
        assert mock.call_count==1
        context=json.loads(mock.call_args.args[1][1]['content'])
        assert len(context['risk_evidence'])==len(codes) and context['evidence_refresh']
        assert context['current_draft']['reply'] is None and context['rule_version']==golden_rules.VERSION
        evaluate("window.confirm=()=>true; [...document.querySelectorAll('#chatLog button')].filter(b=>b.textContent==='确认计划并开启次日提醒').at(-1).click()")
        for _ in range(100):
            if f'已冻结 {len(codes)} 只' in evaluate("document.getElementById('chatLog').textContent"): break
            time.sleep(.1)
        text=evaluate("document.getElementById('chatLog').textContent")
        assert f'已冻结 {len(codes)} 只' in text,text
        evaluate("document.querySelector('[data-page=planTrade]').click(); loadPlanTrade()")
        text=evaluate("document.getElementById('planTradeList').textContent")
        assert '前日F618' in text and '完成5分' in text,text
        screenshot=call('Page.captureScreenshot',{'format':'png'})
        (out/'golden_plan.png').write_bytes(base64.b64decode(screenshot['data']))
        plan=trade_plan_v1.load_plan()
        assert len(plan['stocks'])==len(codes)
        assert all(s['entry_rule']['version']==golden_rules.VERSION for s in plan['stocks'])
        assert all(e['risk']['status']=='AVAILABLE' for e in plan['model_evidence'].values())
        assert all(e['selection']['personal_score'] is not None for e in plan['model_evidence'].values())
        result.update(status='PASS',original_candidates=len(parent['rows']),admitted=codes,
            excluded=len(snapshot['excluded_rows']),risk_available=len(codes),selection_scores=len(codes),
            old_draft_immutable=True,restore_paid_calls=0,regenerate_mock_calls=1,paid_calls=0,
            latest_rule_in_prompt=True,user_plan_modified=False,actual_next_day_minute_trial='PENDING_NEXT_TRADING_DAY')
    with tempfile.TemporaryDirectory() as tmp,patch.object(paths,'data_file',side_effect=lambda name,**kw:str(Path(tmp)/name)),patch.object(
            runtime,'directory',return_value=DESKTOP/'dist/data/postclose_risk'),patch.object(workflow,'snapshot',return_value=snapshot),patch.object(
            workflow,'complete',return_value=response) as mock,patch.object(browser.server,'get_cfg',return_value={}),patch.object(
            trade_plan_v1,'TRADE_PLAN_FILE',Path(tmp)/'dist/data/trade_plan.json'):
        workflow.store('plan',old)
        evidence=plan_evidence.for_snapshot(snapshot)
        assert all(e['risk']['status']=='AVAILABLE' for e in evidence.values())
        browser.main(extra_checks=checks,only_extra=True)
    with (out/'real_flow.json').open('x',encoding='utf-8') as stream:
        json.dump(result,stream,ensure_ascii=False,indent=2)
    print(json.dumps(result,ensure_ascii=False))


if __name__=='__main__':
    main()
