from datetime import date,timedelta
import json
from pathlib import Path
import tempfile
import unittest

import pyarrow as pa
import pyarrow.parquet as pq

from historical_stream_v1 import bound_rows,write_stream
from price_verification_guarded_run import sha
from historical_risk_core_v1 import STOCK_FIELDS,MARKET_FIELDS


class HistoricalStreamTests(unittest.TestCase):
    def fixture(self,root):
        root=Path(root)
        calendar=[];d=date(2023,1,2)
        while len(calendar)<80:
            if d.weekday()<5: calendar.append(int(d.strftime('%Y%m%d')))
            d+=timedelta(days=1)
        self.put(root/'calendar.json',calendar)
        self.put(root/'disclosure.json',{'scope':'SYNTHETIC_NOT_REAL_SECURITY_EVIDENCE'})
        pq.write_table(pa.Table.from_pylist([dict(code='000001',date=d,open=100.,high=101.,low=99.,close=100.)
                                            for d in calendar]),root/'prices.parquet')
        pq.write_table(pa.Table.from_pylist([dict(code='000001',date=d,factor=1.) for d in calendar]),root/'factors.parquet')
        (root/'index.csv').write_text('date,close\n'+''.join(f'{d},3500\n' for d in calendar),encoding='utf-8')
        self.put(root/'index.csv.binding.json',dict(symbol='SH000300',semantics='CSI300_PRICE_INDEX',
                  source='SYNTHETIC_ONLY',adjustment='NONE',sha256=sha(root/'index.csv')))
        self.put(root/'basis.json',{'type':'SYNTHETIC_EVIDENCE_ONLY'})
        evidence=dict(code='000001',scale='RAW',price_sha256=sha(root/'prices.parquet'),
                      factor_sha256=sha(root/'factors.parquet'),calendar_sha256=sha(root/'calendar.json'),
                      scale_verdict='ACCEPTED',corporate_action_coverage_verdict='ACCEPTED',
                      basis_refs=[self.ref(root/'basis.json')],**{'from':calendar[0],'through':calendar[-1]})
        self.put(root/'evidence.json',evidence)
        binding=dict(schema='HISTORICAL_SOURCE_BINDING_V1',calendar=self.ref(root/'calendar.json'),
                     security_scope_disclosure=self.ref(root/'disclosure.json'),sources=[
                         dict(code='000001',prices=self.ref(root/'prices.parquet'),
                              factors=self.ref(root/'factors.parquet'),evidence=self.ref(root/'evidence.json'))])
        self.put(root/'binding.json',binding)
        return root/'binding.json',root/'index.csv'

    def put(self,path,value): Path(path).write_text(json.dumps(value),encoding='utf-8')
    def ref(self,path): return dict(path=str(path),sha256=sha(path))

    def disk_bytes(self,root): return sum(p.stat().st_size for p in Path(root).rglob('*') if p.is_file())

    def test_real_writer_synthetic_bound_pipeline(self):
        with tempfile.TemporaryDirectory() as folder:
            binding,index=self.fixture(folder)
            output=Path(folder)/'out'
            report=write_stream(bound_rows(binding,index),output,lambda:self.disk_bytes(folder),chunk_rows=9)
            self.assertEqual(report['status'],'completed')
            parts=sorted(output.glob('*.parquet'))
            matrix=pq.read_table(parts).to_pylist()
            self.assertEqual(len(matrix),80)
            self.assertEqual(sum(r['training_eligible'] for r in matrix),15)
            self.assertEqual(sum(r['y_tail_5d'] is None for r in matrix),5)
            self.assertTrue(all(r['y_tail_5d']==0 for r in matrix if r['training_eligible']))
            self.assertIsNone(matrix[0]['features']['ret_1d'])
            self.assertEqual(matrix[65]['features']['ret_1d'],0.)
            with self.assertRaises(ValueError): write_stream([],output,lambda:0)

    def test_hash_drift(self):
        with tempfile.TemporaryDirectory() as folder:
            binding,index=self.fixture(folder)
            (Path(folder)/'prices.parquet').write_bytes(b'drift')
            with self.assertRaisesRegex(ValueError,'SOURCE_HASH_MISMATCH'):
                next(bound_rows(binding,index))

    def test_evidence_interval_not_global_flag(self):
        with tempfile.TemporaryDirectory() as folder:
            binding,index=self.fixture(folder)
            path=Path(folder)/'evidence.json'
            evidence=json.loads(path.read_bytes());evidence['corporate_action_coverage_verdict']='UNVERIFIED'
            self.put(path,evidence)
            doc=json.loads(binding.read_bytes());doc['sources'][0]['evidence']=self.ref(path)
            self.put(binding,doc)
            with self.assertRaisesRegex(ValueError,'EVIDENCE_PENDING'): next(bound_rows(binding,index))

    def test_truncated_evidence_does_not_admit_later_rows(self):
        with tempfile.TemporaryDirectory() as folder:
            binding,index=self.fixture(folder)
            path=Path(folder)/'evidence.json'
            e=json.loads(path.read_bytes());e['through']=20230131;self.put(path,e)
            doc=json.loads(binding.read_bytes());doc['sources'][0]['evidence']=self.ref(path);self.put(binding,doc)
            self.assertFalse(any(r['training_eligible'] for r in bound_rows(binding,index)))

    def test_actual_budget_stop_is_terminal(self):
        with tempfile.TemporaryDirectory() as folder:
            binding,index=self.fixture(folder);out=Path(folder)/'out'
            with self.assertRaisesRegex(ValueError,'BUDGET_STOP'):
                write_stream(bound_rows(binding,index),out,lambda:self.disk_bytes(folder),
                             new_cap=16385,chunk_rows=1)
            self.assertEqual(list(out.glob('*.parquet')),[])
            report=json.loads((out/'stop_record.json').read_bytes())
            self.assertEqual(report['status'],'failed')
            with self.assertRaisesRegex(ValueError,'NO_RESTART'):
                write_stream([],out,lambda:0)

    def test_midstream_failure_preserves_partial_not_complete(self):
        with tempfile.TemporaryDirectory() as folder:
            binding,index=self.fixture(folder);out=Path(folder)/'out'
            def failing():
                for i,row in enumerate(bound_rows(binding,index)):
                    if i==3: raise ValueError('source failure')
                    yield row
            with self.assertRaisesRegex(ValueError,'source failure'):
                write_stream(failing(),out,lambda:self.disk_bytes(folder),chunk_rows=2)
            self.assertEqual(len(list(out.glob('*.parquet'))),1)
            self.assertEqual(json.loads((out/'stop_record.json').read_bytes())['status'],'failed')

    def test_nonfinite_nonadmitted_row_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            binding,index=self.fixture(folder)
            row=next(bound_rows(binding,index));row['mae_5d']=float('nan')
            with self.assertRaisesRegex(ValueError,'NONFINITE'):
                write_stream([row],Path(folder)/'out',lambda:self.disk_bytes(folder))
            self.assertFalse(list((Path(folder)/'out').glob('*.parquet')))

    def test_empty_stream_not_success(self):
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaisesRegex(ValueError,'EMPTY_INPUT'):
                write_stream([],Path(folder)/'out',lambda:self.disk_bytes(folder))

    def test_wrong_code_cannot_use_other_stocks_prices(self):
        from historical_stream_v1 import read_unique
        with tempfile.TemporaryDirectory() as folder:
            self.fixture(folder)
            with self.assertRaisesRegex(ValueError,'CODE_MISMATCH'):
                read_unique(Path(folder)/'prices.parquet',('close',),'600519')


if __name__=='__main__': unittest.main()
