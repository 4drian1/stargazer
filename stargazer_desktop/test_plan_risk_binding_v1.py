"""风险证据双入口回归（2026-11-08）：原始选股快照入口 + 已确认计划投影入口。

原始入口：find_pack(code, 原始快照) —— 既有行为，不放宽。
计划投影入口：plan_stock_bindings/find_pack_for_plan —— 以最新已验收运行的冻结
原始快照为绑定基准（input_snapshot 字节SHA校验保留），逐股证明计划 row 与原始
快照 row 完全一致才读包；人工新增、输入变化（价格/特征/来源）、日期不符 → 显式
缺失，不按代码＋日期借包，不落入 shadow 历史回退。
"""
import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import plan_evidence
import postclose_risk_runtime as runtime
from oracles_v1 import EvidencePack
from test_shadow_ui import pack_dict

DAY = 20261106
NEXT = 20261109


def _row(code, name, price):
    return {'code': code, 'name': name, 'date': DAY, 'setup': 'GOLDEN_RESONANCE',
            'stage': 2, 'reference_price': price, 'atr_ref': 0.4, 'rsi14': 55.0,
            'distance_atr': 0.2, 'zone': [price - 0.2, price + 0.2],
            'entry_rule': {'version': 'TAROT_A_SECOND_CASH_V2',
                           'policy': 'GOLDEN_LADDER_STOP_V3', 'eligible': True},
            'personal_score': 0.55}


A, B = '000001', '000002'


class PlanRiskBindingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.run = self.root / 'run_20261106_171522'
        self.run.mkdir()
        self.parent = {'data_date': DAY, 'monitor_date': NEXT,
                       'rows': [_row(A, '原始一', 10.0), _row(B, '原始二', 20.0)]}
        raw = json.dumps(self.parent, sort_keys=True).encode()
        self.run.joinpath('input_snapshot.json').write_bytes(raw)
        packs = []
        for i, code in enumerate((A, B)):
            pack = pack_dict()
            pack['evidence_id'] = f'plan{i}'
            pack['market_date'] = str(DAY)
            pack['asof_ts'] = f'{DAY}150000'
            pack['query_ts'] = f'{DAY}180000'
            pack['candidates'][0]['code'] = code
            pack['risk_context']['asof_ts'] = pack['asof_ts']
            packs.append(json.dumps(pack))
        body = ('\n'.join(packs) + '\n').encode()
        self.run.joinpath('packs.jsonl').write_bytes(body)
        self.record = {'status': 'COMPLETE', 'acceptance_status': 'PASS',
                       'data_date': DAY, 'run_dir': str(self.run),
                       'snapshot_sha256': hashlib.sha256(raw).hexdigest(),
                       'packs_path': str(self.run / 'packs.jsonl'),
                       'packs_sha256': hashlib.sha256(body).hexdigest()}
        self.run.joinpath('independent_acceptance.json').write_text(json.dumps(self.record))
        self.root.joinpath('state.json').write_text(json.dumps(self.record))
        self.patch_dir = patch.object(runtime, 'directory', return_value=self.root)
        self.patch_dir.start()
        runtime._cache.clear()
        # 计划入口的确认草稿：无投影父哈希 + 带附加键（真实确认草稿形状）
        self.draft = {'data_date': DAY, 'monitor_date': NEXT,
                      'rows': copy.deepcopy(self.parent['rows']),
                      'inputs': ['a.parquet'], 'strategy': '策略文本',
                      'daily_source': {'tdx': 'native'}, 'source': 'web',
                      'schema': 'ASSISTANT_PLAN_V1'}

    def tearDown(self):
        self.patch_dir.stop()
        runtime._cache.clear()
        self.temp.cleanup()

    def test_raw_snapshot_entry_binds(self):
        """入口一：原始选股快照直接绑定（既有路径）。"""
        with patch.object(plan_evidence.shadow_ui, 'find_pack',
                          side_effect=AssertionError('NO_HISTORY')):
            ev = plan_evidence.for_snapshot(self.parent)
        self.assertTrue(all(ev[c]['risk']['status'] == 'AVAILABLE' for c in (A, B)))

    def test_confirmed_plan_entry_binds(self):
        """入口二：确认计划草稿（无投影父哈希）按冻结原始快照逐股绑定。"""
        with patch.object(plan_evidence.shadow_ui, 'find_pack',
                          side_effect=AssertionError('NO_BORROW')):
            bindings = runtime.plan_stock_bindings(self.draft)
        self.assertEqual({c: b['status'] for c, b in bindings.items()},
                         {A: 'AVAILABLE', B: 'AVAILABLE'})
        prov = bindings[A]['provenance']
        self.assertEqual(prov['data_date'], str(DAY))
        self.assertEqual(prov['original_snapshot_sha256'], self.record['snapshot_sha256'])
        self.assertEqual(prov['packs_sha256'], self.record['packs_sha256'])
        pack, reason = runtime.find_pack_for_plan(A, self.draft)
        self.assertIsNone(reason)
        self.assertIsNotNone(pack)
        self.assertEqual(pack.candidates[0]['code'], A)

    def test_confirmed_plan_via_for_snapshot_with_provenance(self):
        """for_snapshot 计划入口：AVAILABLE 且附绑定出处；DeepSeek 压缩键含出处。"""
        with patch.object(plan_evidence.shadow_ui, 'find_pack',
                          side_effect=AssertionError('NO_BORROW')):
            ev = plan_evidence.for_snapshot(self.draft)
        risk = ev[A]['risk']
        self.assertEqual(risk['status'], 'AVAILABLE')
        self.assertEqual(risk['binding_provenance']['binding'], 'PLAN_PROJECTION_PER_STOCK')
        self.assertEqual(risk['binding_provenance']['packs_sha256'], self.record['packs_sha256'])
        compact = plan_evidence.prompt_risk(risk)
        self.assertIn('binding_provenance', compact)

    def test_input_change_rejected(self):
        """价格变化 → 该股显式缺失；另一股不受影响。"""
        draft = copy.deepcopy(self.draft)
        for row in draft['rows']:
            if row['code'] == A:
                row['reference_price'] = 10.5
        with patch.object(plan_evidence.shadow_ui, 'find_pack',
                          side_effect=AssertionError('NO_BORROW')):
            bindings = runtime.plan_stock_bindings(draft)
        self.assertEqual(bindings[A]['status'], 'MISSING')
        self.assertIn('输入变化', bindings[A]['reason'])
        self.assertIn('reference_price', bindings[A]['reason'])
        self.assertEqual(bindings[B]['status'], 'AVAILABLE')

    def test_feature_change_rejected(self):
        """模型特征（entry_rule）变化 → 拒绝。"""
        draft = copy.deepcopy(self.draft)
        draft['rows'][0]['entry_rule']['eligible'] = False
        with patch.object(plan_evidence.shadow_ui, 'find_pack',
                          side_effect=AssertionError('NO_BORROW')):
            bindings = runtime.plan_stock_bindings(draft)
        self.assertEqual(bindings[A]['status'], 'MISSING')
        self.assertIn('entry_rule', bindings[A]['reason'])

    def test_manual_addition_rejected(self):
        """人工新增（原始快照无此代码）→ 拒绝，不借包。"""
        draft = copy.deepcopy(self.draft)
        draft['rows'].append(_row('999999', '人工新增', 5.0))
        with patch.object(plan_evidence.shadow_ui, 'find_pack',
                          side_effect=AssertionError('NO_BORROW')):
            bindings = runtime.plan_stock_bindings(draft)
        self.assertEqual(bindings['999999']['status'], 'MISSING')
        self.assertIn('人工新增', bindings['999999']['reason'])

    def test_date_mismatch_rejected(self):
        """数据日不符 → 显式缺失，不用其他日期证据。"""
        draft = copy.deepcopy(self.draft)
        draft['data_date'] = 20261107
        with patch.object(plan_evidence.shadow_ui, 'find_pack',
                          side_effect=AssertionError('NO_BORROW')):
            bindings = runtime.plan_stock_bindings(draft)
        self.assertTrue(all(b['status'] == 'MISSING' for b in bindings.values()))
        self.assertIn('无同日已验收', bindings[A]['reason'])

    def test_pack_hash_mismatch_rejected(self):
        """证据包字节被改 → PACK_HASH_MISMATCH 拒绝（SHA 绑定保留）。"""
        path = self.run / 'packs.jsonl'
        path.write_bytes(path.read_bytes() + b'\n')
        runtime._cache.clear()
        with patch.object(plan_evidence.shadow_ui, 'find_pack',
                          side_effect=AssertionError('NO_BORROW')):
            bindings = runtime.plan_stock_bindings(self.draft)
        self.assertEqual(bindings[A]['status'], 'MISSING')
        self.assertIn('PACK_HASH_MISMATCH', bindings[A]['reason'])

    def test_source_mismatch_rejected(self):
        """row 来源字段（name/来源标识）变化同样算输入变化，不允许借包。"""
        draft = copy.deepcopy(self.draft)
        draft['rows'][1]['name'] = '改名后'
        with patch.object(plan_evidence.shadow_ui, 'find_pack',
                          side_effect=AssertionError('NO_BORROW')):
            bindings = runtime.plan_stock_bindings(draft)
        self.assertEqual(bindings[B]['status'], 'MISSING')
        self.assertIn('输入变化', bindings[B]['reason'])


if __name__ == '__main__':
    unittest.main()
