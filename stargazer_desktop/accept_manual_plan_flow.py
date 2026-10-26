"""Screenshot request through real chat, LangGraph confirmation, and readback."""
import base64
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

from flask import request
import accept_handoff_browser as browser
import assistant_workflow as workflow
import paths
import postclose_risk_runtime as runtime
import trade_plan_v1
from test_manual_plan_observations import MESSAGE, BOARDS


def main():
    desktop = Path(__file__).resolve().parent
    out = desktop/'manual_plan_acceptance_20261021_v2'
    out.mkdir(exist_ok=True)
    if (out/'real_flow.json').exists():
        raise ValueError('ACCEPTANCE_ALREADY_COMPLETE')
    data = desktop/'dist/data'
    snapshot = json.loads((data/'pentacles_demo/snapshot.json').read_bytes())
    oldpath = max((data/'assistant').glob('plan_*.json'), key=lambda p: p.stat().st_mtime_ns)
    old = json.loads(oldpath.read_bytes())
    codes = [r['code'] for r in snapshot['rows']]
    response = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps(dict(
        reply='草案：模型候选不变，另加两只人工观察。\n\n风险：人工股未评分。\n\n待核对：跌幅参考基准。',
        selected_codes=codes+['600621', '000712'], boards=BOARDS, stock_comments=[]))))])
    original = browser.local_acceptance_only
    def routes():
        if request.path in ('/api/chat', '/api/assistant/state', '/api/assistant/plan/confirm', '/api/trade_plan', '/api/model/detail'):
            return None
        return original()
    browser.server.app.before_request_funcs[None] = [routes]
    browser.OUT = out
    result = {}
    def checks(evaluate, call):
        evaluate('window.beginNextDayPlan()')
        assert mock.call_count == 0
        evaluate('send('+json.dumps(MESSAGE, ensure_ascii=False)+')')
        text = evaluate("document.getElementById('chatLog').textContent")
        for expected in ('新增人工观察：000712、600621', '13.80', '4%', '600621', '000712', '条件待核对', '本草案绑定风险证据 5/5'):
            assert expected in text, (expected, text)
        assert '对话失败' not in text
        context = json.loads(mock.call_args.args[1][1]['content'])
        assert len(context['manual_observations']) == 2
        draft = workflow.latest('plan')
        assert draft['selected_codes'] == codes
        assert len(draft['boards']) == 7
        assert len(draft['model_evidence']) == 5
        assert not trade_plan_v1.TRADE_PLAN_FILE.exists()
        evaluate("document.querySelectorAll('.plan-discussion').item(1).scrollIntoView({block:'start'})")
        screenshot = call('Page.captureScreenshot', {'format': 'png'})
        (out/'manual_observation_draft.png').write_bytes(base64.b64decode(screenshot['data']))
        evaluate("window.confirm=()=>true; [...document.querySelectorAll('#chatLog button')].filter(b=>b.textContent==='确认计划并开启次日提醒').at(-1).click()")
        evaluate("new Promise(resolve=>setTimeout(resolve,1800))")
        text = evaluate("document.getElementById('chatLog').textContent")
        assert '已冻结 7 只' in text, text
        formal = trade_plan_v1.load_plan()
        assert len(formal['stocks']) == 7 and len(formal['boards']) == 7
        assert all(s['observe_only'] and s['buy_zone'] is None for s in formal['stocks'][-2:])
        assert all(s['source'] == 'USER' for s in formal['stocks'][-2:])
        assert all(s['user_requests'] == [MESSAGE] for s in formal['stocks'][-2:])
        evaluate("send('请分析002310和600621怎么样，不修改计划条件')")
        analysis = json.loads(mock.call_args.args[1][1]['content'])['requested_stock_analysis']
        assert [(r['code'], r['status']) for r in analysis] == [('002310', 'AVAILABLE'), ('600621', 'NOT_COVERED')]
        assert all(r['computed_now'] is False for r in analysis)
        assert workflow.latest('plan')['changes']['conditions_updated'] == []
        assert len(trade_plan_v1.load_plan()['stocks']) == 7
        result.update(status='PASS', actual_candidate_count=5, model_risk_available=5,
                      manual_count=2, boards=7, exact_user_conditions_preserved=True,
                      confirmed_total=7, local_graph_confirmation=True, auto_rule_expansion=False,
                      model_analysis_statuses=[r['status'] for r in analysis],
                      analysis_does_not_edit_conditions=True,
                      mock_calls=2, paid_calls=0, user_production_plan_modified=False)
    with tempfile.TemporaryDirectory() as tmp, patch.object(paths, 'data_file', side_effect=lambda name, **kw: str(Path(tmp)/name)), patch.object(
            runtime, 'directory', return_value=data/'postclose_risk'), patch.object(workflow, 'snapshot', return_value=snapshot), patch.object(
            workflow, 'complete', return_value=response) as mock, patch.object(browser.server, 'get_cfg', return_value={}), patch.object(
            trade_plan_v1, 'TRADE_PLAN_FILE', Path(tmp)/'dist/data/trade_plan.json'):
        workflow.store('plan', old)
        browser.main(extra_checks=checks, only_extra=True)
        assert mock.call_count == 2
    with (out/'real_flow.json').open('x', encoding='utf-8') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == '__main__':
    main()
