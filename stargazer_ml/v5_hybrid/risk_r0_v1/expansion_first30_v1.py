"""One fixed first-30 evidence batch. check has no network; fetch requires separate approval."""
import argparse
import ast
from datetime import datetime, timedelta, timezone
from hashlib import sha256
import json
from pathlib import Path
import re
import time
import urllib.error
import urllib.request

from price_verification_guarded_run import encoded, sha
from research_budget_v2 import bytes_used, TOTAL_CAP
from sina_div_client import parse_bonus_page

HERE=Path(__file__).resolve().parent
REQUEST=HERE/'contracts/r0_expansion_first30_request_draft_v1.json'
CONTRACT=HERE/'contracts/R0_FIRST30_FETCH_EXECUTION_V1.md'
MANIFEST=HERE/'contracts/r0_first30_fetch_manifest_v2.json'
OUTPUT=HERE/'independent_fetch/expansion_first30_v1'
LIMITS=dict(requests=30,response=8*1024**2,disk=12*1024**2,cumulative=TOTAL_CAP)
RESERVE=256*1024
PER_RESPONSE=512*1024
URL='https://vip.stock.finance.sina.com.cn/corp/go.php/vISSUE_ShareBonus/stockid/{}.phtml'


def require(value,reason):
    if not value:
        raise ValueError(reason)


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None  # urllib raises an HTTPError with the ORIGINAL response body; never follows.


def open_response(url,timeout):
    opener=urllib.request.build_opener(urllib.request.ProxyHandler({}),NoRedirect())
    req=urllib.request.Request(url,method='GET',headers={
        'User-Agent':'Mozilla/5.0','Accept':'text/html','Accept-Encoding':'identity'})
    try:
        return opener.open(req,timeout=timeout)
    except urllib.error.HTTPError as exc:
        return exc  # Still readable: archive rejected HTTP bodies, including 3xx/4xx/5xx.


def validate_page(raw,code):
    text=raw.decode('gbk',errors='replace')
    title=re.search(r'<title\b[^>]*>(.*?)</title>',text,re.I|re.S)
    title=title.group(1).strip()[:256] if title else ''
    require(re.search(r'(?<!\d)'+re.escape(code)+r'(?!\d)',title) is not None,'PAGE_CODE_IDENTITY_UNVERIFIED')
    require(not re.search(r'(?:[?&](?:page|pageno|p)=\d+|下一页|下页|next\s+page)',text,re.I),
            'PAGINATION_UNSUPPORTED_NO_FOLLOW')
    parsed=parse_bonus_page(raw)
    require(parsed['bonus'] is not None and parsed['allotment'] is not None,'REQUIRED_TABLES_NOT_IDENTIFIED')
    # The old parser flattens multi-index headers. Reject ambiguous flattened names rather than keeping one.
    for headers in parsed['headers']:
        if not any('除权' in str(h) or '配股' in str(h) for h in headers):
            continue
        names=[]
        for h in headers:
            if h.startswith('('):
                value=ast.literal_eval(h)
                h=str(value[-1]) if isinstance(value,tuple) else h
            names.append(h)
        require(len(names)==len(set(names)),'AMBIGUOUS_FLATTENED_TABLE_HEADERS')
    return dict(title=title,bonus_rows=len(parsed['bonus']),allotment_rows=len(parsed['allotment']),
                event_coverage_verified=False)


class Batch:
    """Test-injectable engine; production CLI calls it only AFTER authorization and immutable checks."""
    def __init__(self,root,limits,context,opener=open_response,parser=validate_page,
                 budget_used=bytes_used,clock=time.monotonic,sleep=time.sleep):
        self.root,self.limits,self.context=Path(root),limits,context
        self.opener,self.parser,self.budget_used=opener,parser,budget_used
        self.clock,self.sleep=clock,sleep
        self.requests=0
        self.response_bytes=0
        self.ledger=[]
        self.last_start=None
        self.closed=False
        require(not self.root.exists(),'BATCH_DIRECTORY_EXISTS_NO_RESTART')
        claim=encoded(dict(status='CLAIMED_NO_RESTART',context=context,limits=limits,
            created_at=datetime.now(timezone.utc).isoformat()))
        self.check_space(len(claim))
        self.root.mkdir(parents=True,exist_ok=False)
        self.write({'claim.json':claim})

    def actual_bytes(self):
        return sum(p.stat().st_size for p in self.root.iterdir() if p.is_file()) if self.root.exists() else 0

    def check_space(self,increment,final=False):
        reserve=0 if final else RESERVE
        require(self.actual_bytes()+increment+reserve<=self.limits['disk']
                and self.budget_used()+increment+reserve<=self.limits['cumulative'],'DISK_BUDGET_STOP')

    def write(self,blobs,final=False):
        self.check_space(sum(map(len,blobs.values())),final=final)
        require(all(not (self.root/name).exists() for name in blobs),'EVIDENCE_COLLISION')
        for name,data in blobs.items():
            with (self.root/name).open('xb') as f:
                f.write(data)

    def one(self,code):
        require(not self.closed and not (self.root/'stop_record.json').exists(),'BATCH_ALREADY_FINAL')
        require(self.requests<self.limits['requests'],'REQUEST_CAP_STOP')
        remaining=self.limits['response']-self.response_bytes
        require(remaining>0,'RESPONSE_CAP_STOP')
        seq=self.requests+1
        require(all(not (self.root/f'{seq:02d}_{code}{suffix}').exists()
                    for suffix in ('.bin','.json','.attempt.json')),'EVIDENCE_COLLISION')
        url=URL.format(code)
        attempt=dict(seq=seq,code=code,url=url,method='GET',smoke=seq<=2,
                     started_at=datetime.now(timezone.utc).isoformat())
        # All variable textual metadata is bounded. Control characters are the maximal JSON expansion.
        worst=dict(attempt,finished_at='9'*40,status=599,content_type='\x00'*128,
            content_encoding='\x00'*32,error='\x00'*512,outcome='X'*64,
            raw_bytes=PER_RESPONSE,sha256='f'*64,title='\x00'*256,bonus_rows=10**12,
            allotment_rows=10**12,event_coverage_verified=False)
        overhead=len(encoded(attempt))+len(encoded(worst))
        free=min(self.limits['disk']-self.actual_bytes(),self.limits['cumulative']-self.budget_used())
        cap=min(PER_RESPONSE,remaining,free-RESERVE-overhead)
        require(cap>0,'PRE_REQUEST_EVIDENCE_SPACE_STOP')
        self.check_space(cap+overhead)
        if self.last_start is not None:
            delay=1-(self.clock()-self.last_start)
            if delay>0:
                self.sleep(delay)
        attempt['started_at']=datetime.now(timezone.utc).isoformat()
        self.write({f'{seq:02d}_{code}.attempt.json':encoded(attempt)})
        self.requests+=1  # Includes failed opens; persisted attempt precedes the only transport invocation.
        self.last_start=self.clock()
        stream=None
        body=bytearray()
        meta=dict(attempt,status=None,content_type='',content_encoding='',error='',outcome='REJECTED')
        try:
            stream=self.opener(url,10)
            meta['status']=int(stream.status if hasattr(stream,'status') else stream.code)
            meta['content_type']=str(stream.headers.get('Content-Type',''))[:128]
            meta['content_encoding']=str(stream.headers.get('Content-Encoding',''))[:32]
            require(stream.geturl()==url,'UNEXPECTED_RESPONSE_URL')
            while len(body)<cap:
                amount=min(65536,cap-len(body))
                try:
                    chunk=stream.read(amount)
                except Exception as exc:
                    partial=getattr(exc,'partial',b'')
                    require(isinstance(partial,bytes) and len(partial)<=amount,'INVALID_PARTIAL_READ_CONTRACT')
                    body.extend(partial)
                    raise
                require(isinstance(chunk,bytes) and len(chunk)<=amount,'INVALID_STREAM_READ_CONTRACT')
                if not chunk:
                    break
                body.extend(chunk)
            if len(body)==cap:
                meta['outcome']='RESPONSE_LIMIT_UNCERTIFIED'
                raise ValueError('READ_LIMIT_REACHED_NO_PROBE_NO_RETRY')
            require(meta['status']==200,'HTTP_STATUS_'+str(meta['status']))
            require(meta['content_encoding'].lower() in ('','identity'),'UNSUPPORTED_CONTENT_ENCODING')
            meta.update(self.parser(bytes(body),code))
            meta['outcome']='PAGE_MECHANISM_ACCEPTED_NOT_DATA_ADMISSION'
        except (Exception,KeyboardInterrupt) as exc:
            meta['error']=(type(exc).__name__+':'+str(exc))[:512]
        finally:
            if stream is not None:
                try:
                    stream.close()
                except Exception as exc:
                    meta['error']=('CLOSE_ERROR:'+str(exc))[:512]
                    meta['outcome']='REJECTED'
        self.response_bytes+=len(body)  # Error/partial/at-limit/rejected bodies all count.
        meta.update(raw_bytes=len(body),sha256=sha256(body).hexdigest(),
                    finished_at=datetime.now(timezone.utc).isoformat())
        # Archive empty bodies too, so an open error has an explicit zero-byte response artifact.
        self.write({f'{seq:02d}_{code}.bin':bytes(body),f'{seq:02d}_{code}.json':encoded(meta)})
        self.ledger.append(meta)
        return not bool(meta['error'])

    def run(self,codes,verify=lambda:None):
        require(not self.closed and not (self.root/'stop_record.json').exists(),'BATCH_ALREADY_FINAL')
        error=None
        try:
            require(len(codes)<=self.limits['requests'] and len(codes)==len(set(codes)), 'CODE_LIST_SCOPE')
            for code in codes:
                require(isinstance(code,str) and len(code)==6 and code.isdigit(),'INVALID_CODE')
                if not self.one(code):
                    raise ValueError('FIRST_FAILURE_STOP:'+self.ledger[-1]['error'])
            verify()
        except (Exception,KeyboardInterrupt) as exc:
            error=(type(exc).__name__+':'+str(exc))[:1024]
        report=encoded(dict(status='failed' if error else 'completed',error=error,
            request_attempts=self.requests,response_bytes=self.response_bytes,codes_attempted=[r['code'] for r in self.ledger],
            context=self.context,limits=self.limits,network_scope='FIXED_SINA_BONUS_PAGES_ONLY',
            data_admission=False,labels_generated=0,fit_performed=False,ledger=self.ledger))
        index=encoded(dict(records=[dict(code=r['code'],seq=r['seq'],sha256=r['sha256'],
            raw_file=f"{r['seq']:02d}_{r['code']}.bin",meta_file=f"{r['seq']:02d}_{r['code']}.json") for r in self.ledger]))
        files=[dict(name=p.name,bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(self.root.iterdir()) if p.is_file()]
        files.extend(dict(name=n,bytes=len(data),sha256=sha256(data).hexdigest()) for n,data in [('report.json',report),('index.json',index)])
        terminal=encoded(dict(status='failed' if error else 'completed',error=error,requests=self.requests,
            response_bytes=self.response_bytes,files=files,labels_generated=0,fit_performed=False,
            note='Batch is final even on failure. Existing/incomplete directory never restarts or refunds counters.'))
        require(len(report)+len(index)+len(terminal)<=RESERVE,'TERMINAL_RESERVE_EXCEEDED')
        self.closed=True
        self.write({'report.json':report,'index.json':index,'stop_record.json':terminal},final=True)
        return json.loads(terminal)


def verify_inputs(refs):
    require(all(sha(r['path'])==r['sha256'] for r in refs),'FROZEN_INPUT_DRIFT')


def preflight():
    manifest=json.loads(MANIFEST.read_bytes())
    require(manifest['schema']=='R0_FIRST30_FETCH_MANIFEST_V1' and manifest['limits']==LIMITS,'MANIFEST_SCOPE')
    verify_inputs(manifest['inputs'])
    require(manifest['request_sha256']==sha(REQUEST) and manifest['contract_sha256']==sha(CONTRACT),'REQUEST_OR_CONTRACT_DRIFT')
    from importlib.metadata import version
    import platform
    require(platform.python_version()==manifest['python_version'],'PYTHON_VERSION_DRIFT')
    require(all(version(k)==v for k,v in manifest['runtime_versions'].items()),'PARSER_DEPENDENCY_DRIFT')
    request=json.loads(REQUEST.read_bytes())
    verify_inputs([request['parent_selected'],request['parent_summary']])
    parent=json.loads(Path(request['parent_selected']['path']).read_bytes())
    expected=sorted([r for r in parent if not r['archived_external_page_available']],
                    key=lambda r:(r['selection_hash'],r['code']))[:30]
    require(request['codes']==[{k:r[k] for k in ('code','board','selection_hash')} for r in expected], 'PILOT_LIST_DRIFT')
    require(len(expected)==30 and request['external_requests_cap']==30 and request['retries']==0
        and request['smoke_requests']==2 and request['smoke_requests_included'] is True
        and request['response_cap_bytes']==LIMITS['response'] and request['disk_cap_bytes']==LIMITS['disk'], 'REQUEST_LIMITS')
    require(not OUTPUT.exists(),'BATCH_DIRECTORY_EXISTS_NO_RESTART')
    require(bytes_used()+LIMITS['disk']<=TOTAL_CAP,'CUMULATIVE_BUDGET_STOP')
    return [r['code'] for r in expected],manifest,dict(engineering_ready=True,fetch_ready=False,
        fetch_authorized=False,network_requests=0,request_sha256=sha(REQUEST),
        manifest_sha256=sha(MANIFEST),contract_sha256=sha(CONTRACT),limits=LIMITS,
        output=str(OUTPUT),data_admission=False,fit_authorized=False)


def authorize(path,check):
    if path is None:
        return False,'SEPARATE_INDEPENDENT_FETCH_AUTHORIZATION_REQUIRED'
    try:
        auth=json.loads(Path(path).read_bytes())
        require(auth.get('status')=='AUTHORIZED' and auth.get('scope')=='INDEPENDENT_FETCH_R0_FIRST30_V1','AUTHORIZATION_SCOPE')
        require(all(isinstance(auth.get(k),str) and auth[k].strip() for k in ('signer','authorized_at','approval_reference')),
                'AUTHORIZATION_EVIDENCE_REQUIRED')
        ts=datetime.fromisoformat(auth['authorized_at'])
        require(ts.tzinfo is not None and ts<=datetime.now(timezone.utc)+timedelta(minutes=5),'AUTHORIZATION_TIME')
        require(all(auth.get(k)==check[k] for k in ('request_sha256','manifest_sha256','contract_sha256')), 'AUTHORIZATION_BINDING')
    except (ValueError,TypeError,KeyError,OSError) as exc:
        return False,str(exc)
    return True,None


def main(argv=None):
    parser=argparse.ArgumentParser()
    parser.add_argument('command',choices=('check','fetch'))
    parser.add_argument('--authorization',type=Path)
    args=parser.parse_args(argv)
    try:
        codes,manifest,check=preflight()
        allowed,reason=authorize(args.authorization,check)
        check.update(fetch_ready=allowed,fetch_authorized=allowed,blocked=reason)
        if args.command=='check' or not allowed:
            print(json.dumps(check,ensure_ascii=False,indent=2))
            return 0 if allowed else 2
        auth_hash=sha(args.authorization)
        def verify():
            verify_inputs(manifest['inputs'])
            require(sha(MANIFEST)==check['manifest_sha256'] and sha(args.authorization)==auth_hash,'MANIFEST_OR_AUTH_DRIFT')
        verify()
        batch=Batch(OUTPUT,LIMITS,dict(check=check,authorization_path=str(args.authorization.resolve()),
            authorization_sha256=auth_hash))
        result=batch.run(codes,verify=verify)
        print(json.dumps(result,ensure_ascii=False))
        return 0 if result['status']=='completed' else 1
    except Exception as exc:
        print(json.dumps(dict(fetch_ready=False,network_started=None,evidence_directory_exists=OUTPUT.exists(),
            error=f'{type(exc).__name__}:{exc}',note='On an unexpected error, inspect claim/attempts; do not infer zero requests.'),ensure_ascii=False))
        return 1


if __name__=='__main__':
    raise SystemExit(main())
