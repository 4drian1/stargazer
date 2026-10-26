"""Read real accepted evidence through the real plan API input, with mock LLM."""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

import assistant_workflow as workflow
import plan_evidence
import postclose_risk_runtime as runtime
import paths


def main():
    desktop=Path(__file__).resolve().parent
    root=desktop/'dist/data/postclose_risk'
    state=json.loads((root/'state.json').read_text(encoding='utf-8'))
    if state.get('acceptance_status')!='PASS':
        raise ValueError('REAL_DATA_ACCEPTANCE_NOT_READY')
    snapshot=json.loads((desktop/'dist/data/pentacles_demo/snapshot.json').read_text(encoding='utf-8'))
    with patch.object(runtime,'directory',return_value=root):
        evidence=plan_evidence.for_snapshot(snapshot)
        available=[c for c,e in evidence.items() if e['risk']['status']=='AVAILABLE']
        assert len(available)==state['available_count'] and available
        with tempfile.TemporaryDirectory() as tmp,patch.object(paths,'APP_DIR',tmp),patch.object(
                workflow,'snapshot',return_value=snapshot):
            answer={'reply':'验收模拟回复，不是真实模型建议','selected_codes':[r['code'] for r in snapshot['rows'][:5]],
                    'boards':[],'stock_comments':[]}
            response=SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps(answer,ensure_ascii=False)))])
            with patch.object(workflow,'complete',return_value=response) as mock:
                draft=workflow.plan_chat({},'请保留5只观察')['draft']
            context=json.loads(mock.call_args.args[1][1]['content'])
            assert len(context['risk_evidence'])==len(available)
            assert len(context['selection_model_outputs'])==len(available)
            assert len(draft['selected_codes'])==5
            for code,risk in context['risk_evidence'].items():
                assert risk['asof_ts']==str(snapshot['data_date'])+'150000'
                assert risk['context']['amp_232']==evidence[code]['risk']['context']['amp_232']
                assert risk['context']['status']=='PARTIAL_INPUT_DISCLOSED'
            prompt_chars=len(json.dumps(mock.call_args.args[1],ensure_ascii=False))
    result=dict(status='PASS',data_date=snapshot['data_date'],population=len(evidence),
        actual_risk_in_prompt=len(available),missing_codes=context['risk_missing_codes'],
        actual_selection_model_outputs=len(context['selection_model_outputs']),
        prompt_chars=prompt_chars,mock_calls=1,paid_calls=0,user_plan_modified=False,
        model_rows=state['rows'])
    out=desktop/'postclose_risk_acceptance_20261021'
    out.mkdir(exist_ok=True)
    with (out/'actual_plan_input_readback_v2.json').open('x',encoding='utf-8') as stream:
        json.dump(result,stream,ensure_ascii=False,indent=2)
    print(json.dumps(result,ensure_ascii=False))


if __name__=='__main__':
    main()
