import gzip
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent))
import assistant_workflow as w
import plan_evidence
import trade_plan_v1


def reply(content):
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content),finish_reason='stop')])


class PlanFixTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.base=Path(self.tmp.name)
        self.app=patch('paths.APP_DIR',str(self.base));self.app.start()

    def tearDown(self):
        self.app.stop();self.tmp.cleanup()

    def test_only_lossless_json_formatting(self):
        value={'reply':'草案\n继续观察，字符串内双引号"和反斜杠\\保留','selected_codes':['002780']}
        raw=json.dumps(value,ensure_ascii=False)
        for content in [raw, '```json\n'+raw+'\n```',raw.replace('\\n','\n'),raw.replace('\\n','\r\n')]:
            expected=dict(value)
            if '\r\n' in content:expected['reply']=expected['reply'].replace('\n','\r\n')
            self.assertEqual(w.plan_answer(reply(content),'snap','old'),expected)

    def test_no_guessing_missing_delimiters_prose_or_truncated_object(self):
        for raw in ['{"reply":"未完成', '{"reply":"a",}', '解释：{"reply":"a"}',
                    '{"reply":"a"}{"reply":"b"}', '{"reply":"bad\x00value"}', '']:
            with self.assertRaisesRegex(ValueError,'失败回执'):
                w.plan_answer(reply(raw),'snapshot','previous')
        files=list(w.root().glob('failed_plan_response_*.json'))
        self.assertEqual(len(files),6)
        for p in files:
            d=json.loads(p.read_bytes());self.assertEqual(d['previous_draft_id'],'previous')
            self.assertEqual(d['snapshot_hash'],'snapshot');self.assertIn('raw_content',d)
        self.assertFalse(list(w.root().glob('plan_*.json')))

    def test_original_real_reply_and_current_plan_path(self):
        raw=json.loads(gzip.decompress((HERE/'REAL_ATTEMPT2.json.gz').read_bytes()))
        current=json.loads((HERE.parent/'dist/data/pentacles_demo/snapshot.json').read_bytes())
        old=json.loads((HERE.parent/'dist/data/assistant/plan_7c19e55d4ea7c31836efd7e4d60fa5cd503072a6d6d33807c0662b4eeb3a863e.json').read_bytes())
        previous=w.store('plan',old)
        before=(w.root()/f'plan_{previous["id"]}.json').read_bytes()
        with patch.object(w,'snapshot',return_value=current),patch.object(w,'complete',return_value=reply('```json\n'+raw['content']+'\n```')) as api:
            new=w.plan_chat({},'请按最新证据重新制定次日观察计划，保留我已明确的数量与板块要求。',previous['id'])['draft']
        self.assertEqual(api.call_count,1)
        self.assertEqual(new['snapshot']['monitor_date'],20261028)
        self.assertEqual(set(new['selected_codes']),{r['code'] for r in current['rows']})
        self.assertEqual(new['manual_observations'],[])
        self.assertEqual([b['name'] for b in new['boards']],[b['name'] for b in old['boards']])
        context=json.loads(api.call_args.args[1][1]['content'])
        self.assertTrue(context['evidence_refresh']);self.assertIsNone(context['current_draft']['reply'])
        self.assertEqual(context['manual_observations'],[])
        self.assertEqual((w.root()/f'plan_{previous["id"]}.json').read_bytes(),before)
        # Confirmation uses the real workflow with isolated D-drive state.
        with patch.object(w,'snapshot',return_value=current):
            self.assertTrue(w.confirm_plan(new['id'],True)['ok'])
        plan=json.loads((self.base/'dist/data/trade_plan.json').read_bytes())
        self.assertEqual(str(plan['monitor_date']),'20261028')
        self.assertEqual({r['code'] for r in plan['stocks']},set(new['selected_codes']))


if __name__=='__main__':unittest.main()
