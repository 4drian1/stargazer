"""Actual user-generated snapshot + failed activity + old accepted publication."""
import copy
import json
from pathlib import Path
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


def main():
    desktop=Path(__file__).resolve().parent
    out=desktop/'risk_publication_fix_acceptance_20261021'
    out.mkdir(exist_ok=True)
    if (out/'real_flow.json').exists():
        raise ValueError('ACCEPTANCE_ALREADY_COMPLETE')
    data=desktop/'dist/data'
    snapshot=json.loads((data/'pentacles_demo/snapshot.json').read_bytes())
    oldpath=max((data/'assistant').glob('plan_*.json'),key=lambda p:p.stat().st_mtime_ns)
    old=json.loads(oldpath.read_bytes())
    original=browser.local_acceptance_only
    def routes():
        if request.path in ('/api/chat','/api/assistant/state','/api/assistant/plan/confirm','/api/trade_plan','/api/model/detail'):
            return None
        return original()
    browser.server.app.before_request_funcs[None]=[routes]
    browser.OUT=out
    codes=[r['code'] for r in snapshot['rows']]
    response=SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps(
        dict(reply='验收模拟：本草案使用实际匹配的风险证据，非供应商真实建议',selected_codes=codes,boards=[],stock_comments=[]))))])
    result={}
    def checks(evaluate,call):
        activity=runtime.state()
        assert activity['status']=='FAILED'
        assert not runtime.needs_rebuild(snapshot)
        accepted=runtime.accepted_for(snapshot)
        parent=json.loads((Path(accepted['run_dir'])/'input_snapshot.json').read_bytes())
        for stamp in ('2026-10-21T21:20:00+08:00','2026-10-21T21:30:00+08:00'):
            repeated=copy.deepcopy(snapshot);repeated['created_at']=stamp
            repeated['projection_parent_hash']=golden_rules.digest({**parent,'created_at':stamp})
            assert not runtime.needs_rebuild(repeated)
            assert all(e['risk']['status']=='AVAILABLE' for e in plan_evidence.for_snapshot(repeated).values())
        with patch.object(runtime,'state',return_value={'status':'RUNNING','data_date':snapshot['data_date'],'stage':'CAPTURING_MINUTES'}):
            evidence=plan_evidence.for_snapshot(snapshot)
            assert all(e['risk']['status']=='AVAILABLE' for e in evidence.values())
        evaluate('window.beginNextDayPlan()')
        text=evaluate("document.getElementById('chatLog').textContent")
        assert '旧草案' in text and '当前风险证据 5/5' in text,text
        assert mock.call_count==0
        evaluate("[...document.querySelectorAll('#chatLog button')].find(b=>b.textContent==='按最新证据重新制定').click()")
        for _ in range(100):
            if evaluate('!busy'):break
            time.sleep(.1)
        text=evaluate("document.getElementById('chatLog').textContent")
        assert '本草案绑定风险证据 5/5' in text and '对话失败' not in text,text
        context=json.loads(mock.call_args.args[1][1]['content'])
        assert len(context['risk_evidence'])==5 and not context['risk_missing_codes']
        assert context['evidence_refresh'] and context['current_draft']['reply'] is None
        evaluate("window.confirm=()=>true; [...document.querySelectorAll('#chatLog button')].filter(b=>b.textContent==='确认计划并开启次日提醒').at(-1).click()")
        for _ in range(100):
            if '已冻结 5 只' in evaluate("document.getElementById('chatLog').textContent"):break
            time.sleep(.1)
        assert '已冻结 5 只' in evaluate("document.getElementById('chatLog').textContent")
        plan=trade_plan_v1.load_plan()
        assert all(e['risk']['status']=='AVAILABLE' for e in plan['model_evidence'].values())
        result.update(status='PASS',actual_user_snapshot_created_at=snapshot['created_at'],
            risk_available=5,repeated_generation=2,background_running_preserves_risk=True,
            background_failed_preserves_risk=True,needs_rebuild=False,old_missing_draft_disabled=True,
            actual_prompt_risk=5,actual_prompt_missing=[],mock_calls=1,paid_calls=0,user_plan_modified=False)
    with tempfile.TemporaryDirectory() as tmp,patch.object(paths,'data_file',side_effect=lambda name,**kw:str(Path(tmp)/name)),patch.object(
            runtime,'directory',return_value=data/'postclose_risk'),patch.object(workflow,'snapshot',return_value=snapshot),patch.object(
            workflow,'complete',return_value=response) as mock,patch.object(browser.server,'get_cfg',return_value={}),patch.object(
            trade_plan_v1,'TRADE_PLAN_FILE',Path(tmp)/'dist/data/trade_plan.json'):
        workflow.store('plan',old)
        browser.main(extra_checks=checks,only_extra=True)
        assert mock.call_count==1
    with (out/'real_flow.json').open('x',encoding='utf-8') as stream:
        json.dump(result,stream,ensure_ascii=False,indent=2)
    print(json.dumps(result,ensure_ascii=False))


if __name__=='__main__':
    main()
