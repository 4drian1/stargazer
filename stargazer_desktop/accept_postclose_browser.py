"""Real UI shell consumes today's accepted evidence; unrelated live routes stubbed."""
import json
import os
from pathlib import Path

import accept_handoff_browser as browser
import plan_evidence
from flask import request


def main():
    root=Path(__file__).resolve().parent
    out=root/'postclose_risk_acceptance_20261021'
    os.environ['STARGAZER_UI_ACCEPTANCE_OUT']=str(out)
    os.environ['STARGAZER_PLAN_UI_FIXTURE']='1'
    browser.OUT=out
    snapshot=json.loads((root/'dist/data/pentacles_demo/snapshot.json').read_text(encoding='utf-8'))
    browser.server.assistant_workflow.snapshot=lambda:snapshot
    def routes():
        if request.path in ('/api/shell/state','/api/postclose-risk/state'):
            return None
        return browser.local_acceptance_only()
    browser.server.app.before_request_funcs[None]=[routes]
    display=plan_evidence.status(snapshot)
    assert display['risk_available']==17
    browser.main()
    path=out/'browser_acceptance.json'
    result=json.loads(path.read_text(encoding='utf-8'))
    result['actual_current_model_display']=display
    result['actual_snapshot_date']=snapshot['data_date']
    path.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print('Actual same-day UI risk coverage:',display['risk_available'],'/',display['candidate_count'])


if __name__=='__main__':
    main()
