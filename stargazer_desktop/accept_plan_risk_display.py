"""Actual five-stock draft displayed before confirmation; mock vendor only."""
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
import plan_evidence
import postclose_risk_runtime as runtime
import trade_plan_v1


def main():
    desktop = Path(__file__).resolve().parent
    out = desktop / 'risk_display_acceptance_20261021_v2'
    out.mkdir(exist_ok=True)
    if (out / 'real_flow.json').exists():
        raise ValueError('ACCEPTANCE_ALREADY_COMPLETE')
    data = desktop / 'dist/data'
    snapshot = json.loads((data / 'pentacles_demo/snapshot.json').read_bytes())
    oldpath = max((data / 'assistant').glob('plan_*.json'), key=lambda p: p.stat().st_mtime_ns)
    old = json.loads(oldpath.read_bytes())
    codes = [r['code'] for r in snapshot['rows']]
    original = browser.local_acceptance_only
    def routes():
        if request.path in ('/api/chat', '/api/assistant/state', '/api/model/detail'):
            return None
        return original()
    browser.server.app.before_request_funcs[None] = [routes]
    browser.OUT = out
    response = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps(dict(
        reply='草案：保留当前名单。\n\n风险讨论：按代码绑定证据查看，非买入指令。\n\n待确认：可继续修改数量和板块。',
        selected_codes=codes, boards=[], stock_comments=[]))))])
    def checks(evaluate, call):
        evaluate('window.beginNextDayPlan()')
        text = evaluate("document.getElementById('chatLog').textContent")
        for label in ('草案：', '风险：', '待确认：', '1.20%', '1.22%', '1.31%', '本草案绑定风险证据 5/5'):
            assert label in text, (label, text)
        assert mock.call_count == 0
        assert evaluate("document.querySelectorAll('.plan-discussion details').length") == 6
        evaluate("document.querySelector('.plan-discussion details').open=true")
        text = evaluate("document.querySelector('.plan-discussion').textContent")
        for label in ('65.21%', '领涨股观测', '沪指指标', '未知不等于', '风险证据缺失'):
            if label in ('未知不等于', '风险证据缺失'):
                assert label not in text
            else:
                assert label in text, label
        evaluate("document.querySelector('[data-page=monitor]').click(); document.querySelector('[data-model-detail=risk]').click()")
        evaluate("new Promise(resolve=>setTimeout(resolve,400))")
        detail = evaluate("document.getElementById('modelDetailBody').textContent")
        assert '未确认草案' in detail and '1.20%' in detail and '领涨股观测' in detail, detail
        assert not trade_plan_v1.TRADE_PLAN_FILE.exists()
        evaluate("document.querySelector('[data-page=chat]').click();document.getElementById('modelDetailPanel').hidden=true; document.querySelector('.plan-discussion').scrollIntoView({block:'start'})")
        screenshot = call('Page.captureScreenshot', {'format': 'png'})
        (out / 'draft_risk_sections.png').write_bytes(base64.b64decode(screenshot['data']))
        evaluate("send('请按最新证据重新制定次日观察计划')")
        assert mock.call_count == 1
        assert evaluate("document.querySelectorAll('.plan-discussion').length") == 2
        assert not trade_plan_v1.TRADE_PLAN_FILE.exists()
    with tempfile.TemporaryDirectory() as tmp, patch.object(paths, 'data_file', side_effect=lambda name, **kw: str(Path(tmp)/name)), patch.object(
            runtime, 'directory', return_value=data/'postclose_risk'), patch.object(workflow, 'snapshot', return_value=snapshot), patch.object(
            workflow, 'complete', return_value=response) as mock, patch.object(browser.server, 'get_cfg', return_value={}), patch.object(
            trade_plan_v1, 'TRADE_PLAN_FILE', Path(tmp)/'dist/data/trade_plan.json'):
        workflow.store('plan', old)
        browser.main(extra_checks=checks, only_extra=True)
        view = plan_evidence.model_view('risk')
        assert view['status'] == 'UNCONFIRMED_DRAFT'
    result = dict(status='PASS', source_draft=str(oldpath), candidate_count=5, risk_available=5,
                  restored_without_paid_calls=True, risk_preview_before_confirmation=True,
                  structured_sections=True, mock_calls=1, paid_calls=0, user_plan_modified=False)
    with (out/'real_flow.json').open('x', encoding='utf-8') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == '__main__':
    main()
