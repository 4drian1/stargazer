"""Today's real 17 candidates + deliberately noncompliant mock prose, no paid calls."""
import base64
import json
from pathlib import Path
import tempfile
import time
from types import SimpleNamespace
from unittest.mock import patch

from flask import request
import accept_handoff_browser as browser
import assistant_workflow as workflow
import paths
import postclose_risk_runtime as runtime
import trade_plan_v1


def main():
    desktop = Path(__file__).resolve().parent
    out = desktop / 'plan_comment_fix_acceptance_20261021_v2'
    out.mkdir(exist_ok=True)
    report = out / 'real_flow.json'
    if report.exists():
        raise ValueError('ACCEPTANCE_ALREADY_EXISTS')
    snapshot = json.loads((desktop / 'dist/data/pentacles_demo/snapshot.json').read_bytes())
    codes = [r['code'] for r in snapshot['rows']]
    answer = {'reply': '验收模拟讨论，非真实供应商建议', 'selected_codes': codes, 'boards': [],
        'stock_comments': [{'code': c, 'selection': '观察0.618黄金位置及RSI14',
            'risk': '未来120交易分钟下探幅度，不是概率',
            'evidence_refs': [f'tarot:{snapshot["data_date"]}:{c}', f'pyxis:{snapshot["data_date"]}:{c}']}
            for c in codes[:10]]}
    response = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps(answer)))])
    original = browser.local_acceptance_only
    def local_routes():
        if request.path in ('/api/chat', '/api/assistant/state', '/api/assistant/plan/confirm',
                '/api/trade_plan', '/api/model/detail', '/api/plan_check'):
            return None
        return original()
    browser.server.app.before_request_funcs[None] = [local_routes]
    browser.OUT = out
    result = {'paid_calls': 0, 'user_plan_modified': False}
    def click_flow(evaluate, call):
        evaluate('window.beginNextDayPlan()')
        for _ in range(100):
            if evaluate('!busy'):
                break
            time.sleep(.1)
        text = evaluate("document.getElementById('chatLog').textContent")
        assert '未通过校验' in text and '待确认' in text and '对话失败' not in text, text
        assert '仅观察、无价格提醒：000629' in text, text
        evaluate("window.confirm=()=>true; [...document.querySelectorAll('#chatLog button')].find(b=>b.textContent==='确认计划并开启次日提醒').click()")
        for _ in range(100):
            if '已冻结' in evaluate("document.getElementById('chatLog').textContent"):
                break
            time.sleep(.1)
        text = evaluate("document.getElementById('chatLog').textContent")
        assert '已冻结 17 只' in text, text
        evaluate("document.querySelector('[data-page=planTrade]').click(); loadPlanTrade()")
        text = evaluate("document.getElementById('planTradeList').textContent")
        assert '000629' in text and '仅观察 · 无价格提醒' in text, text
        screenshot = call('Page.captureScreenshot', {'format': 'png'})
        (out / 'plan_confirmed.png').write_bytes(base64.b64decode(screenshot['data']))
        plan = trade_plan_v1.load_plan()
        assert len(plan['stocks']) == 17 and len(plan['model_evidence']) == 17
        assert all(e['risk']['status'] == 'AVAILABLE' for e in plan['model_evidence'].values())
        assert all(e['selection']['personal_score'] is not None for e in plan['model_evidence'].values())
        assert '0.618' not in json.dumps(plan['model_commentary'])
        result.update(status='PASS', candidates=17, risk_evidence=17, selection_scores=17,
            rejected_comment_fields=len(plan['commentary_warnings']), confirmed_stocks=17,
            observe_only=[s['code'] for s in plan['stocks'] if s.get('observe_only')],
            real_clicks=['plan_draft', 'human_confirm_test_only', 'trade_plan'],
            no_automatic_paid_retry=True)
    with tempfile.TemporaryDirectory() as tmp, patch.object(paths, 'data_file',
            side_effect=lambda name, **kw: str(Path(tmp) / name)), patch.object(
            runtime, 'directory', return_value=desktop / 'dist/data/postclose_risk'), patch.object(
            workflow, 'snapshot', return_value=snapshot), patch.object(
            workflow, 'complete', return_value=response) as mock, patch.object(
            browser.server, 'get_cfg', return_value={}), patch.object(
            trade_plan_v1, 'TRADE_PLAN_FILE', Path(tmp) / 'dist/data/trade_plan.json'):
        browser.main(extra_checks=click_flow, only_extra=True)
        assert mock.call_count == 1
        result['mock_calls'] = mock.call_count
    with report.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == '__main__':
    main()
