from copy import deepcopy
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent))
import risk_outlook_v1 as o

class FocusTests(unittest.TestCase):
    def setUp(self):
        self.archives=[]
        for p in (HERE.parent/'dist/data/postclose_risk/risk_outlook').glob('*.json'):
            if p.name.startswith('latest'):continue
            d=json.loads(p.read_bytes())
            if d.get('asof') in (20261027110000,20261027113000):self.archives.append((p,d))
        assert len(self.archives)==2
        self.d=self.archives[0][1];self.v=json.loads(self.d['raw_reply']);self.f=self.d['computed_facts']

    def test_two_actual_replies_only_drop_quote_highlights(self):
        for path,d in self.archives:
            before=path.read_bytes();v=json.loads(d['raw_reply'])
            with self.assertRaisesRegex(ValueError,'OUTLOOK_POSITION_REFERENCE'):o.validate(v,d['computed_facts'],d['asof'])
            q,notes=o.qualify_focus(v,d['computed_facts'])
            self.assertEqual(q['focus_ids'],[]);self.assertEqual(len(notes),2)
            self.assertEqual({k:x for k,x in q.items() if k!='focus_ids'},{k:x for k,x in v.items() if k!='focus_ids'})
            text=o.validate(q,d['computed_facts'],d['asof'])
            self.assertNotIn('关注位置：',text);self.assertIn('行情快照：',text)
            self.assertEqual(path.read_bytes(),before)

    def test_invalid_focus_still_rejected(self):
        for focus in [['invented'],['market.sse','market.sse'],[{}],None,['market.sse']*7]:
            with self.subTest(focus=focus),self.assertRaisesRegex(ValueError,'OUTLOOK_POSITION_REFERENCE'):
                o.qualify_focus(dict(self.v,focus_ids=focus),self.f)
        for changed in [dict(status='UNKNOWN'),dict(evidence_kind='OTHER')]:
            facts=deepcopy(self.f);facts['market.sse'].update(changed)
            with self.assertRaisesRegex(ValueError,'OUTLOOK_POSITION_REFERENCE'):o.qualify_focus(self.v,facts)

    def test_real_position_preserved_and_no_basis_not_invented(self):
        facts=deepcopy(self.f);facts['002780.plan.0']=dict(status='KNOWN',value=13.,text='位置')
        q,notes=o.qualify_focus(dict(self.v,focus_ids=['market.sse','002780.plan.0']),facts)
        self.assertEqual(q['focus_ids'],['002780.plan.0']);self.assertEqual(len(notes),1)
        q.update(supporting_ids=[],opposing_ids=[],unknown_ids=[])
        with self.assertRaisesRegex(ValueError,'OUTLOOK_NO_BASIS'):o.validate(q,facts,self.d['asof'])

    def test_cached_failure_remains_failure_no_retry(self):
        with patch('deepseek_api.complete') as api:
            d=self.d
            result=o.generate({},dict(plan_hash=d['plan_hash']),d['asof'],'INTRADAY',plan_id=d['plan_id'])
            self.assertEqual(result['status'],'AI_FAILED');api.assert_not_called()

if __name__=='__main__':unittest.main()
