from contextlib import redirect_stdout
from datetime import datetime,timezone,timedelta
import http.client
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock,patch
import urllib.error

import expansion_first30_v1 as app
from test_sina_div_client import FIXTURE


def page(code):
    return f'<html><title>公司({code})分红配股</title></html>'.encode('gbk')+FIXTURE


class Stream:
    def __init__(self,code,body=None,status=200,headers=None):
        self.body=io.BytesIO(page(code) if body is None else body)
        self.status=status
        self.headers=headers or {'Content-Type':'text/html; charset=gb2312'}
        self.url=app.URL.format(code)
        self.read_sizes=[]
    def read(self,n):
        self.read_sizes.append(n)
        return self.body.read(n)
    def geturl(self):
        return self.url
    def close(self):
        self.body.close()


class BatchTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.base=Path(self.tmp.name)
        self.root=self.base/'batch'
        self.now=0.
        self.sleeps=[]
        self.calls=[]
    def sleep(self,n):
        self.now+=n
        self.sleeps.append(n)
    def open(self,url,timeout):
        self.calls.append((url,timeout))
        return Stream(url.split('/')[-1].split('.')[0])
    def batch(self,opener=None,limits=None):
        return app.Batch(self.root,limits or app.LIMITS,{'test_fixture_only':True},opener=opener or self.open,
            budget_used=lambda:sum(p.stat().st_size for p in self.base.rglob('*') if p.is_file()),
            clock=lambda:self.now,sleep=self.sleep)
    def test_success_smoke_reused_throttle_and_raw_roundtrip(self):
        batch=self.batch()
        result=batch.run(['000001','000002','000008'])
        self.assertEqual(result['status'],'completed')
        self.assertEqual(len(self.calls),3)  # No additional smoke requests.
        self.assertEqual(self.sleeps,[1.,1.])
        self.assertTrue(batch.ledger[0]['smoke'])
        self.assertTrue(batch.ledger[1]['smoke'])
        self.assertFalse(batch.ledger[2]['smoke'])
        for r in result['files']:
            self.assertEqual(app.sha(self.root/r['name']),r['sha256'])
        self.assertEqual((self.root/'01_000001.bin').read_bytes(),page('000001'))
        self.assertEqual(batch.actual_bytes(),sum(p.stat().st_size for p in self.root.iterdir()))
    def test_first_http_failure_archived_counted_stops(self):
        def fail(url,timeout):
            self.calls.append(url)
            return Stream('000001',b'SERVER_ERROR',500)
        batch=self.batch(fail)
        result=batch.run(['000001','000002'])
        self.assertEqual(result['status'],'failed')
        self.assertEqual(len(self.calls),1)
        self.assertEqual(result['response_bytes'],12)
        self.assertEqual((self.root/'01_000001.bin').read_bytes(),b'SERVER_ERROR')
    def test_open_error_zero_body_preserved_no_retry(self):
        def fail(url,timeout):
            self.calls.append(url)
            raise TimeoutError('test timeout')
        result=self.batch(fail).run(['000001','000002'])
        self.assertEqual(result['requests'],1)
        self.assertEqual(len(self.calls),1)
        self.assertEqual((self.root/'01_000001.bin').read_bytes(),b'')
        self.assertEqual(json.loads((self.root/'01_000001.json').read_bytes())['raw_bytes'],0)
    def test_rejected_identity_retains_original(self):
        result=self.batch(lambda u,t:Stream('000001',page('600001'))).run(['000001','000002'])
        self.assertEqual(result['status'],'failed')
        self.assertIn('PAGE_CODE_IDENTITY',result['error'])
        self.assertEqual((self.root/'01_000001.bin').read_bytes(),page('600001'))
    def test_partial_read_error_retains_partial_and_counts(self):
        class Partial(Stream):
            def read(self,n):
                raise http.client.IncompleteRead(b'partial-bytes',100)
        result=self.batch(lambda u,t:Partial('000001')).run(['000001'])
        self.assertEqual(result['response_bytes'],13)
        self.assertEqual((self.root/'01_000001.bin').read_bytes(),b'partial-bytes')
        self.assertEqual(result['status'],'failed')
    def test_at_limit_no_probe_original_bytes_counted(self):
        stream=Stream('000001',b'x'*100)
        limits=dict(app.LIMITS,response=100)
        batch=self.batch(lambda u,t:stream,limits)
        result=batch.run(['000001'])
        self.assertEqual(result['response_bytes'],100)
        self.assertEqual(stream.read_sizes,[100])
        self.assertIn('READ_LIMIT_REACHED',result['error'])
        self.assertEqual((self.root/'01_000001.bin').read_bytes(),b'x'*100)
    def test_cumulative_response_budget_not_per_request(self):
        raw=page('000001')
        limits=dict(app.LIMITS,response=len(raw)+10)
        batch=self.batch(lambda u,t:Stream(u.split('/')[-1].split('.')[0],raw),limits)
        result=batch.run(['000001','000002','000008'])
        self.assertEqual(result['requests'],2)
        self.assertEqual(result['response_bytes'],len(raw)+10)
        self.assertEqual((self.root/'02_000002.bin').stat().st_size,10)
    def test_request_cap_and_duplicate_codes_block_without_transport(self):
        batch=self.batch(limits=dict(app.LIMITS,requests=1))
        result=batch.run(['000001','000002'])
        self.assertEqual(result['requests'],0)
        self.assertEqual(self.calls,[])
    def test_terminal_and_same_object_refuse_restart(self):
        batch=self.batch()
        batch.run(['000001'])
        stop=(self.root/'stop_record.json').read_bytes()
        with self.assertRaisesRegex(ValueError,'BATCH_ALREADY_FINAL'):
            batch.run(['000002'])
        with self.assertRaisesRegex(ValueError,'BATCH_ALREADY_FINAL'):
            batch.one('000002')
        with self.assertRaisesRegex(ValueError,'BATCH_DIRECTORY_EXISTS'):
            self.batch()
        self.assertEqual((self.root/'stop_record.json').read_bytes(),stop)
        self.assertEqual(len(self.calls),1)
    def test_incomplete_directory_also_refuses_restart(self):
        self.root.mkdir()
        with self.assertRaisesRegex(ValueError,'BATCH_DIRECTORY_EXISTS'):
            self.batch()
        self.assertEqual(self.calls,[])
    def test_known_collision_refused_before_request(self):
        batch=self.batch()
        (self.root/'01_000001.bin').write_bytes(b'old')
        result=batch.run(['000001'])
        self.assertEqual(result['requests'],0)
        self.assertEqual(self.calls,[])
        self.assertEqual((self.root/'01_000001.bin').read_bytes(),b'old')
    def test_one_byte_disk_budget_creates_nothing(self):
        with self.assertRaisesRegex(ValueError,'DISK_BUDGET_STOP'):
            self.batch(limits=dict(app.LIMITS,disk=1))
        self.assertFalse(self.root.exists())
        self.assertEqual(self.calls,[])
    def test_cumulative_disk_limit_not_reset_for_batch(self):
        (self.base/'other_project_evidence').write_bytes(b'x'*100)
        with self.assertRaisesRegex(ValueError,'DISK_BUDGET_STOP'):
            self.batch(limits=dict(app.LIMITS,cumulative=app.RESERVE+99))
        self.assertFalse(self.root.exists())
    def test_reserved_space_limits_read_before_transport(self):
        cap_seen=[]
        class Observed(Stream):
            def read(self,n):
                cap_seen.append(n)
                return super().read(n)
        batch=self.batch(lambda u,t:Observed('000001',b'x'*50000),limits=dict(app.LIMITS,disk=app.RESERVE+20000))
        result=batch.run(['000001'])
        self.assertEqual(result['status'],'failed')
        self.assertLess(result['response_bytes'],20000)
        self.assertLessEqual(batch.actual_bytes(),app.RESERVE+20000)
        self.assertLess(max(cap_seen),20000)
    def test_post_fetch_hash_failure_finalizes_not_completed(self):
        def bad():
            raise ValueError('INPUT_DRIFT')
        result=self.batch().run(['000001'],verify=bad)
        self.assertEqual(result['status'],'failed')
        self.assertIn('INPUT_DRIFT',result['error'])

    def test_full_30_one_shared_counter_no_extra_smoke_requests(self):
        codes=[f'{i:06d}' for i in range(30)]
        batch=self.batch()
        result=batch.run(codes)
        self.assertEqual(result['status'],'completed')
        self.assertEqual(result['requests'],30)
        self.assertEqual(len(self.calls),30)
        self.assertEqual(result['response_bytes'],sum(len(page(c)) for c in codes))
        self.assertEqual(len(self.sleeps),29)

    def test_response_encoding_rejected_without_decompression(self):
        raw=page('000001')
        stream=Stream('000001',raw,headers={'Content-Encoding':'gzip'})
        result=self.batch(lambda u,t:stream).run(['000001','000002'])
        self.assertEqual(result['status'],'failed')
        self.assertEqual(result['requests'],1)
        self.assertEqual((self.root/'01_000001.bin').read_bytes(),raw)


class ParserTransportTests(unittest.TestCase):
    def test_flattened_header_ambiguity_refused(self):
        parsed=dict(bonus=[],allotment=[],headers=[["('分红', '除权除息日')","('其他', '除权除息日')"]])
        with patch.object(app,'parse_bonus_page',return_value=parsed):
            with self.assertRaisesRegex(ValueError,'AMBIGUOUS_FLATTENED'):
                app.validate_page(page('000001'),'000001')

    def test_dual_tables_and_title(self):
        result=app.validate_page(page('000001'),'000001')
        self.assertIsInstance(result['bonus_rows'],int)
        self.assertFalse(result['event_coverage_verified'])
    def test_pagination_and_missing_tables_refused(self):
        for raw in (page('000001')+'下一页'.encode('gbk'),b'<title>(000001)</title>'):
            with self.assertRaises(ValueError):
                app.validate_page(raw,'000001')
    def test_default_transport_headers_and_redirect_handler(self):
        response=Stream('000001')
        opener=Mock()
        opener.open.return_value=response
        with patch.object(app.urllib.request,'build_opener',return_value=opener) as factory:
            self.assertIs(app.open_response(app.URL.format('000001'),10),response)
        self.assertTrue(any(isinstance(h,app.NoRedirect) for h in factory.call_args.args))
        request=opener.open.call_args.args[0]
        self.assertEqual(request.get_method(),'GET')
        self.assertEqual(request.get_header('Accept'),'text/html')
        self.assertEqual(request.get_header('Accept-encoding'),'identity')
        self.assertEqual(opener.open.call_args.kwargs['timeout'],10)
        self.assertIsNone(app.NoRedirect().redirect_request(None,None,302,'',{},'https://elsewhere'))
    def test_http_error_body_readable_no_retry_or_follow(self):
        response=urllib.error.HTTPError(app.URL.format('000001'),302,'Moved',{},io.BytesIO(b'original-302'))
        opener=Mock()
        opener.open.side_effect=response
        with patch.object(app.urllib.request,'build_opener',return_value=opener):
            got=app.open_response(app.URL.format('000001'),10)
        self.assertIs(got,response)
        self.assertEqual(got.read(100),b'original-302')
        opener.open.assert_called_once()


class AuthorizationTests(unittest.TestCase):
    def test_unexpected_outer_error_does_not_claim_zero_network(self):
        with patch.object(app,'preflight',side_effect=OSError('mock IO failure')),redirect_stdout(io.StringIO()) as captured:
            self.assertEqual(app.main(['fetch']),1)
        result=json.loads(captured.getvalue())
        self.assertIsNone(result['network_started'])
        self.assertIn('inspect claim/attempts',result['note'])

    def test_missing_and_old_scope_and_wrong_hash_refused(self):
        check=dict(request_sha256='r',manifest_sha256='m',contract_sha256='c')
        self.assertFalse(app.authorize(None,check)[0])
        auth=dict(status='AUTHORIZED',scope='INDEPENDENT_FETCH_R0_FIRST30_V1',signer='mock',
            authorized_at=datetime.now(timezone.utc).isoformat(),approval_reference='MOCK_ONLY',**check)
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'auth.json'
            p.write_bytes(app.encoded(auth))
            self.assertTrue(app.authorize(p,check)[0])
            for change in (dict(status='DRAFT'),dict(scope='FIT_R0_CROSSYEAR_V1'),dict(manifest_sha256='changed'),
                           dict(authorized_at='2026-01-01'),dict(signer=''),
                           dict(authorized_at=(datetime.now(timezone.utc)+timedelta(days=1)).isoformat())):
                p.write_bytes(app.encoded(dict(auth,**change)))
                self.assertFalse(app.authorize(p,check)[0])
    def test_check_and_unauthorized_fetch_never_construct_batch_or_transport(self):
        for command in ('check','fetch'):
            with patch.object(app,'preflight',return_value=(['000001'],{},{})), \
                 patch.object(app,'Batch') as batch,patch.object(app,'open_response') as transport, \
                 redirect_stdout(io.StringIO()):
                self.assertEqual(app.main([command]),2)
                batch.assert_not_called()
                transport.assert_not_called()


if __name__=='__main__':
    unittest.main()
