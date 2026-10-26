"""Real headless Chrome clicks against local Flask; no notifier/paid API started."""
import base64
import json
import os
import subprocess
import tempfile
import threading
import time
import urllib.request
from pathlib import Path

from flask import jsonify, request
from werkzeug.serving import make_server
from websockets.sync.client import connect

import server
import trade_plan_v1

OUT = Path(os.environ.get('STARGAZER_UI_ACCEPTANCE_OUT', str(Path(__file__).parent / 'handoff_acceptance_20261021')))


@server.app.before_request
def local_acceptance_only():
    # Only static resources and read-only evidence/freshness use real handlers.
    if (not request.path.startswith('/api/') or request.path in ('/api/shadow_evidence', '/api/index_freshness', '/api/model/detail')
            or (request.path == '/api/trade_plan' and os.environ.get('STARGAZER_PLAN_UI_FIXTURE') == '1')):
        return None
    if request.path == '/api/state':
        return jsonify(positions=[], watch=[], push={}, market_open=False)
    if request.path == '/api/pentacles/state':
        return jsonify(snapshot=None, enabled=False, running=False)
    if request.path == '/api/trade_plan':
        return jsonify(plan=None, status='NO_PLAN')
    if request.path == '/api/shell/state':
        return jsonify(models={'selection': {'payload': {}}})
    if request.path == '/api/info_feed':
        return jsonify(connected=False, items=[])
    if request.path == '/api/trades':
        return jsonify(trades=[])
    return jsonify(messages=[], status='DISABLED', items=[])


def main(extra_checks=None, only_extra=False):
    OUT.mkdir(exist_ok=True)
    http = make_server('127.0.0.1', 0, server.app, threaded=True)
    threading.Thread(target=http.serve_forever, daemon=True).start()
    errors = []
    with tempfile.TemporaryDirectory(prefix='stargazer-ui-') as profile:
        if os.environ.get('STARGAZER_PLAN_UI_FIXTURE') == '1':
            trade_plan_v1.TRADE_PLAN_FILE = Path(profile) / 'test_plan.json'
            trade_plan_v1.TRADE_PLAN_FILE.write_text(json.dumps({
                'version': 1, 'data_date': '20260908', 'monitor_date': '20260909',
                'stocks': [{'code': '000001', 'name': '合成界面验收', 'buy_zone': [9, 10], 'valid_until': '20260909150000'}],
                'model_evidence': {'000001': {'selection': {'code': '000001', 'setup': 'A_SDP', 'reference_price': 10},
                    'risk': {'status': 'MISSING', 'reason': '合成缺失反例'}}},
                'model_commentary': {'000001': {'selection': '合成选股解释', 'risk': '风险未知'}},
                'combined_commentary': '合成界面验收，不是交易计划'}, ensure_ascii=False), encoding='utf-8')
        chrome = subprocess.Popen([
            r'C:\Program Files\Google\Chrome\Application\chrome.exe', '--headless=new',
            '--disable-gpu', '--disable-background-networking', '--no-first-run',
            '--no-sandbox', '--disable-gpu-sandbox', '--in-process-gpu',
            '--no-default-browser-check', '--disable-extensions', '--remote-debugging-port=0',
            '--proxy-server=http://127.0.0.1:9', '--proxy-bypass-list=127.0.0.1;localhost',
            '--window-size=1280,1000', '--user-data-dir=' + profile, 'about:blank'],
            stdout=subprocess.DEVNULL, stderr=(OUT / 'chrome_stderr.log').open('w'))
        try:
            portfile = Path(profile) / 'DevToolsActivePort'
            for _ in range(100):
                if portfile.exists():
                    break
                time.sleep(.1)
            port = int(portfile.read_text().splitlines()[0])
            with urllib.request.urlopen(f'http://127.0.0.1:{port}/json/list') as response:
                page = next(item for item in json.load(response) if item.get('type') == 'page')
            with connect(page['webSocketDebuggerUrl'], proxy=None, compression=None) as ws:
                seq = 0

                def call(method, params=None):
                    nonlocal seq
                    seq += 1
                    ws.send(json.dumps(dict(id=seq, method=method, params=params or {})))
                    while True:
                        message = json.loads(ws.recv(timeout=20))
                        if message.get('method') == 'Runtime.exceptionThrown':
                            errors.append(message['params']['exceptionDetails'])
                        if message.get('id') == seq:
                            if 'error' in message:
                                raise RuntimeError(message['error'])
                            return message.get('result', {})

                def evaluate(expression):
                    result = call('Runtime.evaluate', dict(expression=expression,
                                  awaitPromise=True, returnByValue=True))
                    if 'exceptionDetails' in result:
                        raise RuntimeError(result['exceptionDetails'])
                    return result['result'].get('value')

                call('Runtime.enable')
                call('Page.enable')
                call('Page.navigate', {'url': f'http://127.0.0.1:{http.server_port}/'})
                time.sleep(1)
                if extra_checks is not None:
                    extra_checks(evaluate, call)
                    if only_extra:
                        assert not errors, errors
                        (OUT / 'browser_acceptance.json').write_text(json.dumps(
                            dict(status='PASS', js_errors=errors, paid_calls=0), ensure_ascii=False), encoding='utf-8')
                        return
                evaluate("document.querySelector('[data-page=planTrade]').click()")
                for _ in range(50):
                    if evaluate("document.getElementById('shadowReplaySelect').options.length"):
                        break
                    time.sleep(.1)
                count = evaluate("document.getElementById('shadowReplaySelect').options.length")
                assert count == 20, count
                evaluate("document.getElementById('shadowReplayOpen').click()")
                time.sleep(.5)
                text = evaluate("document.getElementById('shadowAdviceBody').textContent")
                assert '历史回放 20260908' in text and 'amp_232' in text, text[:400]
                evaluate("document.getElementById('shadowAdvicePanel').scrollIntoView({block:'center',behavior:'instant'})")
                time.sleep(.1)
                screenshot = call('Page.captureScreenshot', {'format': 'png'})
                (OUT / 'historical_replay.png').write_bytes(base64.b64decode(screenshot['data']))
                evaluate("document.getElementById('shadowReplaySelect').value = '20260908:000000'; document.getElementById('shadowReplaySelect').append(new Option('missing','20260908:000000')); document.getElementById('shadowReplaySelect').value='20260908:000000';document.getElementById('shadowReplayOpen').click()")
                time.sleep(.3)
                missing = evaluate("document.getElementById('shadowAdviceBody').textContent")
                assert '不可评分' in missing and 'PACK_MISSING' in missing, missing
                assert not errors, errors
                result = dict(status='PASS', catalog_count=count, js_errors=errors,
                              real_clicks=['planTrade', 'shadowReplayOpen', 'missing_pack'],
                              missing=missing, paid_calls=0, live_routes_stubbed=True)
                if os.environ.get('STARGAZER_PLAN_UI_FIXTURE') == '1':
                    assert evaluate("['overviewCheck','overviewPicks','overviewPlan'].every(id=>document.getElementById(id)===null)")
                    evaluate("document.querySelector('[data-page=monitor]').click();document.querySelector('[data-model-detail=risk]').click()")
                    time.sleep(.3)
                    risk = evaluate("document.getElementById('modelDetailBody').textContent")
                    assert '000001' in risk and '风险证据缺失' in risk, risk
                    evaluate("document.querySelector('[data-model-detail=selection]').click()")
                    time.sleep(.3)
                    selection = evaluate("document.getElementById('modelDetailBody').textContent")
                    assert '强势首次分歧' in selection and '合成选股解释' in selection, selection
                    evaluate("document.getElementById('modelDetailPanel').scrollIntoView({block:'center',behavior:'instant'})")
                    screenshot = call('Page.captureScreenshot', {'format': 'png'})
                    (OUT / 'model_workspace.png').write_bytes(base64.b64decode(screenshot['data']))
                    result.update(model_views='PASS', duplicate_actions_removed=True, synthetic_plan_fixture=True)
                    assert not errors, errors
                (OUT / 'browser_acceptance.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
                print(json.dumps(result, ensure_ascii=False))
                try:
                    call('Browser.close')
                except Exception:
                    pass  # Chrome may close the socket before acknowledging shutdown.
        finally:
            if chrome.poll() is None:
                chrome.terminate()
            chrome.wait(timeout=15)
            http.shutdown()


if __name__ == '__main__':
    main()
