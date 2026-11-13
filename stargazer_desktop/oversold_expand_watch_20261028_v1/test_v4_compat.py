"""Retain the old rule tests; adapt display and source-qualified reuse fixtures."""
import copy
import gzip
import importlib.util
import json
from pathlib import Path
import sys
import unittest
from datetime import datetime

H=Path(__file__).resolve().parent;R=H.parents[1]
spec=importlib.util.spec_from_file_location('previous_v4_tests',R/'stargazer_desktop/resonance_rule_20261026_v4/test_v4.py')
old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)

class V4Compatibility(old.V4Tests):
    def test_quote_alert_at_ATR_zone_without_touching_raw_node(self):
        row,_,_,_=self.yuan
        plan=dict(monitor_date=20261026,data_date=20261023,rows=[row])
        quotes={'002878':dict(time='20261026093500',price=12.49,prev=13.32,low=12.30)}
        alerts=old.observation_reviews(plan,quotes,datetime(2026,10,26,9,35),{})
        self.assertEqual(len(alerts),1)
        self.assertEqual(alerts[0]['decision'],'OBSERVE')
        self.assertIn('12.21—12.45',alerts[0]['reason'])

    def test_risk_pack_reuse_normalizes_project_relative_paths(self):
        import postclose_risk_runtime as runtime
        original=json.loads((R/'stargazer_desktop/dist/data/pentacles_demo/snapshot.json').read_bytes())
        release=json.loads(gzip.decompress(old.tarot.MANIFEST.read_bytes()))
        parent=copy.deepcopy(original);current=copy.deepcopy(original)
        # Both sides share the same risk-source generation. A prior source change is not a display change.
        parent['rule_inputs']=[dict(path=str((R/p).resolve()),sha256=h) for p,h in release['frozen_inputs'].items()]
        current['rule_inputs']=[dict(path=p,sha256=h) for p,h in release['frozen_inputs'].items()]
        for row in current['rows']:
            rule=row['entry_rule'];rule.update(policy=old.v4.POLICY,model_reference_node=rule.get('model_reference_node',rule['f618']))
        self.assertTrue(runtime.risk_inputs_equivalent(parent,current))
        # Adding the quote-only module does not rebind or change 232/266 model matrices.
        parent['rule_inputs']=[r for r in parent['rule_inputs'] if not r['path'].endswith('golden_watch_groups.py')]
        self.assertTrue(runtime.risk_inputs_equivalent(parent,current))
        changed=copy.deepcopy(current);changed['rows'][0]['selection_score']+=.1
        self.assertFalse(runtime.risk_inputs_equivalent(parent,changed))
        changed=copy.deepcopy(current)
        model=next(x for x in changed['rule_inputs'] if x['path'].endswith('initial_node_cash_fit_20261023_v1/model.pkl'));model['sha256']='0'*64
        self.assertFalse(runtime.risk_inputs_equivalent(parent,changed))
        changed=copy.deepcopy(current)
        source=next(x for x in changed['rule_inputs'] if x['path'].endswith('postclose_index_source.py'));source['sha256']='0'*64
        self.assertFalse(runtime.risk_inputs_equivalent(parent,changed))

if __name__=='__main__':unittest.main()
