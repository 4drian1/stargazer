"""Actual oversized snapshot, stale-draft regeneration, compact price rendering."""
import copy
from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
APP = ROOT/'stargazer_desktop'
sys.path.insert(0, str(APP))
import assistant_context as context
import assistant_workflow as workflow
import paths
import plan_evidence
from test_assistant_workflow import response


class Evening(datetime):
    @classmethod
    def now(cls):
        return cls(2026, 10, 26, 20)


class FixTests(unittest.TestCase):
    def test_actual_large_rules_compact_without_losing_execution_bands(self):
        snapshot = json.loads((APP/'dist/data/pentacles_demo/snapshot.json').read_bytes())
        old = {r['code']:r['entry_rule'] for r in snapshot['rows']}
        before = copy.deepcopy(old)
        with self.assertRaises(ValueError):
            context.bounded_messages([{'role':'user','content':context.serialized(old)}], [])
        new = {c:context.plan_entry_rule(r) for c,r in old.items()}
        messages = context.bounded_messages([{'role':'user','content':context.serialized(new)}], [])
        self.assertLess(len(context.serialized(messages)), 10000)
        for code, rule in old.items():
            self.assertEqual({k:v for k,v in new[code].items() if k!='observation_map_reference'},
                             {k:v for k,v in rule.items() if k!='observation_map'})
            self.assertEqual(new[code]['observation_map_reference']['sha256'],
                hashlib.sha256(context.serialized(rule['observation_map']).encode()).hexdigest())
        self.assertEqual(before, old)

    def test_legacy_rule_and_empty_zone_unchanged(self):
        rule = {'policy':'LEGACY','levels':[], 'eligible':False, 'f618':None}
        self.assertEqual(context.plan_entry_rule(rule), rule)

    def test_real_snapshot_regenerates_stale_draft_keeps_user_requirements(self):
        snapshot = json.loads((APP/'dist/data/pentacles_demo/snapshot.json').read_bytes())
        codes = [r['code'] for r in snapshot['rows'][:3]]
        board_names = ['pcb','军工','创新药','房地产','煤炭','存储','cpo']
        boards = [{'name':name,'note':'原关注条件'} for name in board_names]
        manual = [{'code':'600621','source':'USER','name':'人工观察',
            'price_zone':[10.,11.], 'conditions_status':'USER_PRICE_ZONE_ONLY_OTHER_CONDITIONS_UNSTRUCTURED',
            'user_requests':['加入600621，关注区间10-11，原话必须保留']}]
        with tempfile.TemporaryDirectory(dir=HERE/'tmp') as tmp, patch.object(paths,'APP_DIR',tmp), \
                patch.object(workflow,'datetime',Evening), \
                patch.object(workflow,'snapshot',return_value=snapshot), \
                patch.object(plan_evidence.shadow_ui,'find_pack',return_value=(None,'MISSING')), \
                patch('postclose_risk_runtime.find_pack',return_value=(None,'当前日尚无已验收风险证据')):
            old = workflow.store('plan',dict(snapshot_hash='previous-snapshot',snapshot={},
                dialogue=[{'role':'user','content':'保留3只，监控pcb、军工、创新药、房地产、煤炭、存储、cpo。'},
                          {'role':'assistant','content':'旧F618草案文字'*10000}],
                selected_codes=['300225','600593','002878'],requested_count=3,boards=boards,
                manual_observations=manual,reply='旧证据不可沿用',created_at='old'))
            file = workflow.root()/f'plan_{old["id"]}.json'
            old_bytes = file.read_bytes()
            with patch.object(workflow,'complete',return_value=response(dict(reply='按当前V4观察档位制定新草案',
                    selected_codes=codes,boards=boards))) as api:
                result = workflow.plan_chat({},'请按最新证据重新制定次日观察计划，保留我已明确的数量与板块要求。',old['id'])
            self.assertEqual(api.call_count,1)
            messages = api.call_args.args[1]
            self.assertLess(len(context.serialized(messages)),context.MAX_CHARS)
            prompt = json.loads(messages[1]['content'])
            self.assertTrue(prompt['evidence_refresh'])
            self.assertEqual(prompt['requested_count'],3)
            self.assertEqual(prompt['current_boards'],boards)
            self.assertEqual(prompt['manual_observations'],manual)
            self.assertEqual(set(prompt['risk_missing_codes']),{r['code'] for r in snapshot['rows']})
            self.assertFalse(prompt['risk_evidence'])
            self.assertFalse(any(m['role']=='assistant' for m in messages))
            self.assertIn('保留3只',context.serialized(messages))
            draft = result['draft']
            self.assertEqual(draft['snapshot_hash'],workflow.digest(snapshot))
            self.assertEqual(draft['boards'],boards)
            self.assertEqual(draft['manual_observations'],manual)
            self.assertTrue(workflow.plan_readiness(draft,snapshot)['current'])
            self.assertEqual(file.read_bytes(),old_bytes)
            self.assertFalse((Path(tmp)/'dist/data/trade_plan.json').exists())
            self.assertTrue(workflow.confirm_plan(draft['id'],True)['ok'])
            formal = json.loads((Path(tmp)/'dist/data/trade_plan.json').read_bytes())
            self.assertEqual([b['name'] for b in formal['boards']],board_names)
            self.assertEqual(len(formal['stocks']),4)
            for r in formal['stocks']:
                if r['code'] in codes:
                    original=next(s for s in snapshot['rows'] if s['code']==r['code'])
                    self.assertEqual(r['entry_rule'],original['entry_rule'])

    def test_js_three_distinct_pairs_prices_only(self):
        from py_mini_racer import MiniRacer
        text = (APP/'web/pentacles.js').read_text(encoding='utf8')
        function = text[text.index('  function resonanceText'):text.index('  const gateNames')]
        js = MiniRacer()
        js.eval("const number = (n, digits=2) => typeof n==='number' && Number.isFinite(n) ? n.toFixed(digits) : '—';"+function)
        def band(lower,count):
            return dict(lower=lower,upper=lower+.1,node=lower+.05,
                sources=[dict(low_day=i,high_day=1) for i in range(count)])
        row = dict(reference_price=15,entry_rule=dict(observation_map=dict(bands=[
            band(12,2),band(13,3),band(14,4),band(16,4)])))
        self.assertEqual(js.call('resonanceText',row),'14.0000—14.1000\n13.0000—13.1000')
        row['entry_rule']['observation_map']['bands']=[dict(band(12,2),
            sources=[dict(low_day=1,high_day=1)]*5)]
        self.assertEqual(js.call('resonanceText',row),'—')
        self.assertIn('三重及以上共振',(APP/'web/index.html').read_text(encoding='utf8'))


if __name__=='__main__':
    (HERE/'tmp').mkdir(exist_ok=True)
    unittest.main()
