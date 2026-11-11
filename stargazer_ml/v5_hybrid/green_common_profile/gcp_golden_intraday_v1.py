"""Bounded local stockdb prefix -> unchanged frozen 5m trigger. No orders/LLM."""
from datetime import datetime
import gzip
import hashlib
import json
import math
import msvcrt
from pathlib import Path
import sys
import urllib.parse
import urllib.request

import msgpack

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent/'risk_r0_v1'))
sys.path.insert(0, str(HERE.parents[2]/'stargazer_desktop'))
import joint_shadow_compute_v2 as compute
from golden_rules import VERSION, digest
from public_stock_minutes_v1 import current_minutes, CONTRACT as MINUTE_SOURCE_CONTRACT


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *a, **kw):
        return None


def prefix_result(row, day, cutoff, minute_rows, previous_close, adjustments):
    import pandas as pd
    rule = row.get('entry_rule') or {}
    if rule.get('version') not in (VERSION,'TAROT_A_SECOND_CASH_V2') or not rule.get('eligible'):
        raise ValueError('RULE_NOT_ADMITTED')
    if not math.isfinite(previous_close) or previous_close <= 0 or abs(previous_close-float(row['reference_price'])) > .01:
        raise ValueError('PREVIOUS_CLOSE_SCALE_MISMATCH')
    # Any recorded action on execution day requires a separately verified basis.
    for key, value in adjustments:
        d = value.get('date', str(key).split(':')[-1])
        if str(d).startswith(str(day)):
            raise ValueError('EXECUTION_DAY_CORPORATE_ACTION_UNRESOLVED')
    if rule['version']=='TAROT_A_SECOND_CASH_V2':
        from tarot_live_signal_v2 import evaluate
        return evaluate(row,day,cutoff,minute_rows)
    frame = pd.DataFrame(minute_rows)
    if frame.empty or not set(['date','code','open','high','low','close']).issubset(frame):
        raise ValueError('MINUTE_SCHEMA_OR_PREFIX_MISSING')
    frame = frame[(frame.date >= day*1000000) & (frame.date < cutoff-cutoff%100)].copy()
    if not len(frame) or not frame.code.astype(str).str.zfill(6).eq(row['code']).all():
        raise ValueError('MINUTE_IDENTITY_OR_PREFIX_MISSING')
    for _, group in frame[frame.duplicated('date', keep=False)].groupby('date'):
        if len(group[['open','high','low','close']].drop_duplicates()) != 1:
            raise ValueError('MINUTE_DUPLICATE_CONFLICT')
    frame = frame.drop_duplicates('date').sort_values('date')
    for r in frame.itertuples():
        datetime.strptime(str(int(r.date)), '%Y%m%d%H%M%S')
        prices = [float(getattr(r,k)) for k in ('open','high','low','close')]
        if any(not math.isfinite(v) or v <= 0 for v in prices) or not r.low <= min(r.open,r.close) <= max(r.open,r.close) <= r.high:
            raise ValueError('MINUTE_OHLC_INVALID')
    if int(frame.date.iloc[0]) != day*1000000+93000:
        raise ValueError('OPENING_MINUTE_MISSING')
    kernel = compute.source_module(compute.TRIGGER_SOURCE)
    bars = kernel.to_bars(frame.to_dict('records'))
    five, segments, gaps = kernel.aggregate_5min(bars, day)
    done = [(b,s) for b,s in zip(five,segments) if day*1000000+b['t'] <= cutoff]
    if not done:
        raise ValueError('NO_COMPLETED_5M_BAR')
    f, atr = rule['f618'], rule['atr_ref']
    if float(frame.open.iloc[0]) < f:
        return dict(status='NON_DOWNSIDE', reason='OPEN_BELOW_F_NOT_A_DIP_FROM_ABOVE')
    replay = kernel.replay([b for b,s in done], [s for b,s in done], f, atr)
    fired = replay.get('trigger') is not None
    opening_downside = None
    first = frame[(frame.date >= day*1000000+93000) & (frame.date < day*1000000+94500)]
    if cutoff >= day*1000000+94500 and len(first)==15:
        # Downside release only. Not a new 20%-25% hard gate.
        opening_downside = max(0.,float(frame.open.iloc[0])-float(first.low.min()),
                               previous_close-float(first.low.min()))/atr
    return dict(status='RULE_TRIGGERED' if fired else 'WAIT_SIGNAL', reason=replay['reason'],
        trigger_time=day*1000000+replay['trigger_t'] if fired else None,
        flags=replay['flags'], divergence=replay.get('divergence'), data_gaps=max(s for b,s in done),
        opening_15m_downside_atr=opening_downside,
        scope='冻结下杀止跌背离研究规则，只提醒人工复核；非买入，非风险模型新判断')


def pairs(obj):
    if isinstance(obj, dict) and 'data' in obj:
        obj = obj['data']
    if not isinstance(obj, list):
        raise ValueError('STOCKDB_RESPONSE_SHAPE')
    out = []
    for item in obj:
        if isinstance(item, dict):
            out.append((None,item))
        elif isinstance(item, (list,tuple)) and len(item)==2 and isinstance(item[1],dict):
            out.append(item)
        else:
            raise ValueError('STOCKDB_ROW_SHAPE')
    return out


def bound_history(root,day,code):
    """Reuse one accepted local archive, never fetch another full-market batch."""
    from live_postclose_source_v1 import LocalCapture,CapturedMinutes
    for folder in sorted(root.glob(f'run_{day}_*'),reverse=True):
        terminal=folder/'terminal.json';manifest=folder/'source_manifest.json'
        if not terminal.exists() or not manifest.exists():continue
        if json.loads(terminal.read_bytes()).get('status')!='COMPLETE':continue
        if manifest.stat().st_size>16*1024**2:raise ValueError('SOURCE_MANIFEST_SIZE')
        entry=json.loads(manifest.read_bytes()).get('entries',{}).get(code)
        if entry is None:continue
        capture=LocalCapture(root)
        frame=CapturedMinutes({code:entry},capture,int(day)*1000000+150000)[code]
        refs=[dict(path=str(manifest),sha256=hashlib.sha256(manifest.read_bytes()).hexdigest())]
        if entry.get('seed'):refs.append(dict(path=entry['seed']['path'],sha256=entry['seed']['sha256']))
        refs.extend(dict(path=x['path'],sha256=x['sha256']) for x in entry.get('extensions',[]))
        return frame.to_dict('records'),refs
    return [],[]


def verify_plan_sources(plan):
    """Explicit predecessor bridge; verify every current bound file before legacy plans."""
    from tarot_live_v2 import MANIFEST,_project_root
    manifest=json.loads(gzip.decompress(MANIFEST.read_bytes()))
    def absolute(value):
        path=Path(value)
        return str((path if path.is_absolute() else _project_root()/path).resolve())
    current={absolute(k):v for k,v in manifest['frozen_inputs'].items()}
    previous={absolute(k):v for k,v in manifest.get('approved_predecessor_inputs',{}).items()}
    runtime_updates={absolute(k) for k in manifest.get('approved_runtime_updates',[])}
    for file,sha in current.items():
        if hashlib.sha256(Path(file).read_bytes()).hexdigest()!=sha:
            raise ValueError('CURRENT_RULE_SOURCE_FINGERPRINT_MISMATCH')
    legacy=all((r.get('entry_rule') or {}).get('policy') not in ('GOLDEN_LADDER_STOP_V3','GOLDEN_RESONANCE_CROSSDAY_V4') for r in plan['rows'])
    for item in plan['rule_inputs']:
        path=absolute(item['path']);expected=item['sha256']
        if path not in current:raise ValueError('RULE_SOURCE_UNBOUND')
        runtime_update=path in runtime_updates
        if expected!=current[path] and not ((legacy or runtime_update) and expected in previous.get(path,[])):
            raise ValueError('RULE_SOURCE_FINGERPRINT_MISMATCH')


def run(body):
    plan, cutoff = body['plan'], int(body['cutoff'])
    day = int(plan['monitor_date'])
    if cutoff//1000000 != day or plan.get('rule_version') not in (VERSION,'TAROT_A_SECOND_CASH_V2'):
        raise ValueError('RULE_DAY_OR_VERSION_MISMATCH')
    if not plan.get('rule_inputs'):
        raise ValueError('RULE_SOURCE_BINDING_MISSING')
    if plan.get('rule_version')=='TAROT_A_SECOND_CASH_V2':verify_plan_sources(plan)
    else:
        for item in plan['rule_inputs']:
            if hashlib.sha256(Path(item['path']).read_bytes()).hexdigest()!=item['sha256']:
                raise ValueError('RULE_SOURCE_FINGERPRINT_MISMATCH')
    root = Path(body['runtime'])
    root.mkdir(parents=True, exist_ok=True)
    # The launcher uses the existing 1GiB runtime, not a new unbounded directory.
    from minute_history_v1 import DiskBudget,load_days
    budget=DiskBudget()
    folder = root/'golden_intraday'/str(day)
    folder.mkdir(parents=True, exist_ok=True)
    key = digest(plan)
    target = folder/f'{key}_{cutoff}_{MINUTE_SOURCE_CONTRACT}.json'
    if target.exists():
        return json.loads(target.read_bytes())
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirect())
    counter = folder/'attempts.json'
    count = json.loads(counter.read_bytes())['attempts'] if counter.exists() else 0
    receipts = []
    def fetch(table, code, date, reuse=False):
        nonlocal count
        tag = hashlib.sha256(f'{table}:{code}:{date}:{"fixed" if reuse else cutoff}'.encode()).hexdigest()
        rawfile = folder/(tag+'.msgpack.gz')
        if rawfile.exists():
            compressed = rawfile.read_bytes()
            receipts.append(dict(path=str(rawfile),sha256=hashlib.sha256(compressed).hexdigest()))
            return pairs(msgpack.unpackb(gzip.decompress(compressed), raw=False, strict_map_key=False))
        if count >= 100*max(len(plan['rows']),1):
            raise ValueError('DAILY_REQUEST_LIMIT')
        budget(1024**2)
        count += 1  # Actual attempts, including errors; no automatic retries.
        tmp = counter.with_suffix('.tmp')
        tmp.write_text(json.dumps({'attempts':count}), encoding='utf-8')
        tmp.replace(counter)
        url = 'http://127.0.0.1:7899/?cmd=get&t='+urllib.parse.quote(f'{table}:{code}:{date}')
        request = urllib.request.Request(url, headers={'Accept':'application/x-msgpack'})
        with opener.open(request, timeout=3) as response:
            raw = response.read(1024**2+1)
        if len(raw)>1024**2:
            raise ValueError('SINGLE_RESPONSE_LIMIT')
        compressed = gzip.compress(raw)
        budget(len(compressed))
        with rawfile.open('xb') as stream:
            stream.write(compressed)
        receipts.append(dict(path=str(rawfile),sha256=hashlib.sha256(compressed).hexdigest()))
        return pairs(msgpack.unpackb(raw, raw=False, strict_map_key=False))
    rows = {}
    failed = False
    for row in plan['rows']:
        code = row['code']
        try:

            if rule_policy:=row['entry_rule'].get('policy'):
                if rule_policy!='GOLDEN_RESONANCE_CROSSDAY_V4':
                    raise ValueError('RULE_POLICY_RECONFIRM_REQUIRED')
            days=row['entry_rule'].get('history_days') or [int(plan['data_date'])]
            if (not 1<=len(days)<=4 or days!=sorted(set(days)) or days[-1]!=int(plan['data_date'])):
                raise ValueError('HISTORY_DAY_CONTRACT_INVALID')
            history,refs=load_days(code,days,budget)
            receipts.extend(refs)
            previous = [v for v in history if int(v['date'])//1000000==int(plan['data_date'])]
            previous = sorted((v for v in previous if int(v['date'])//1000000==int(plan['data_date'])),key=lambda v:int(v['date']))
            if not previous or any(str(v.get('code')).zfill(6)!=code for v in previous):
                raise ValueError('PREVIOUS_DAY_RAW_MINUTES_MISSING')
            if int(previous[-1]['date'])%1000000 < 145900:
                raise ValueError('PREVIOUS_DAY_FINAL_CLOSE_MISSING')
            minute = [v for k,v in fetch('分钟k',code,str(day)+'*')]
            minute,minute_source=current_minutes(folder,code,int(plan['data_date']),cutoff,
                float(previous[-1]['close']),minute,budget,receipts)
            if rule_policy=='GOLDEN_RESONANCE_CROSSDAY_V4':

                unique={}
                for item in history+previous+minute:
                    stamp=int(item['date'])
                    if stamp>cutoff:continue
                    if str(item.get('code')).zfill(6)!=code:raise ValueError('HISTORY_IDENTITY_MISMATCH')
                    if stamp in unique and any(unique[stamp].get(k)!=item.get(k) for k in ('open','high','low','close','volume')):
                        raise ValueError('HISTORY_OVERLAP_CONFLICT')
                    unique[stamp]=item
                minute=[unique[k] for k in sorted(unique)]
            from tdx_corporate_events_v1 import checked_events
            adjustments,action_receipt=checked_events(code,day,int(plan['data_date']),root)
            receipts.append(action_receipt)
            rows[code] = prefix_result(row,day,cutoff,minute,float(previous[-1]['close']),adjustments)
            rows[code]['minute_source']=minute_source
        except Exception as exc:
            rows[code] = dict(status='DATA_INSUFFICIENT', reason=str(exc) if isinstance(exc, ValueError) else type(exc).__name__)
            # A single unavailable stock must not suppress other stocks.
        rows[code]['rule_hash'] = digest(row['entry_rule'])
    result = dict(rule_version=plan['rule_version'], plan_hash=key, cutoff=cutoff, minute_source_contract=MINUTE_SOURCE_CONTRACT, rows=rows, raw_evidence=receipts)
    raw = json.dumps(result,ensure_ascii=False,allow_nan=False).encode()
    budget(len(raw))
    with target.open('xb') as stream:
        stream.write(raw)
    return result


if __name__ == '__main__':
    body = json.loads(sys.stdin.read(1024**2))
    root = Path(body['runtime']); root.mkdir(parents=True, exist_ok=True)
    with (root/'golden_worker.lock').open('a+b') as lock:
        if lock.tell()==0:
            lock.write(b'0'); lock.flush()
        lock.seek(0)
        msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
        print(json.dumps(run(body),ensure_ascii=False,allow_nan=False))
