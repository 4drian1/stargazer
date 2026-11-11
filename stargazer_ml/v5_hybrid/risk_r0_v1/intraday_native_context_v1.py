"""Native END_5M inputs for the frozen risk kernel; no network or model fit.

Member return paths are prepared once. Only candidates and actually observed
leaders need full-history technical indicators. Missing members retain their
positions in the frozen comparison universe. This module never invents 1m bars.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from build_joint_risk_features_v1 import basket_features, observed_leaders
from tdx_native_arrays_v1 import build_minute_arrays, build_quality, run_features
from tdx_native_reader_v1 import grid_5min, validate_prefix, validate_day

CONTRACT='INTRADAY_NATIVE_END_5M_CONTEXT_V1'
STOCK_COLUMNS=('stock_close','stock_reference_close_raw','stock_day_return_raw',
    'stock_return5m_raw','stock_return15m_raw','stock_momentum_change5m_raw',
    'stock_momentum_change15m_raw','stock_rsi14_5m_raw','stock_rsi14_15m_raw',
    'stock_rsi_delta5m_raw','stock_rsi_delta15m_raw',
    'stock_volume_ratio_previous_session_raw','stock_intrabar_return15m_raw')


def rows_digest(rows):
    return hashlib.sha256(json.dumps(rows,sort_keys=True,allow_nan=False).encode()).hexdigest()


def prefix_rows(rows,code,day,cutoff):
    """Normalize the explicit native 5m schema, then validate the exact prefix."""
    values=[]
    for r in rows:
        if r.get('source_clock')!='END_5M' or r.get('source_frequency_minutes')!=5:
            raise ValueError('INTRADAY_SOURCE_FREQUENCY_UNBOUND:'+code)
        if str(r.get('code'))!=code:
            raise ValueError('INTRADAY_SOURCE_IDENTITY:'+code)
        ts=int(r['date'])
        if ts//1000000!=day:
            raise ValueError('INTRADAY_SOURCE_DAY:'+code)
        if ts>cutoff:
            continue
        values.append(dict(ts=ts,day=day,minute=ts//10000%100*60+ts//100%100,
            open=r['open'],high=r['high'],low=r['low'],close=r['close'],
            raw_volume=r['volume'],amount=r.get('amount',0)))
    check=validate_prefix(values,day,cutoff)
    if not check['eligible']:
        raise ValueError('INTRADAY_SOURCE_PREFIX:'+code+':'+check.get('reason',''))
    return values


class NativeIntradayCache:
    """feature_cache.stock contract backed by source-verified 5m paths.

    current(code) returns normalized END_5M rows plus a source receipt.
    history(code) returns native rows BEFORE day, plus a source receipt.
    references are last-day closes from a validated, complete previous session.
    Callbacks do not collect network data here; acquisition and quotas are separate.
    """
    def __init__(self,days,cutoff,boards,candidates,references,current,history):
        self.days=list(map(int,days));self.cutoff=int(cutoff);self.day=cutoff//1000000
        if self.days!=sorted(set(self.days)) or len(self.days)<2 or self.days[-1]!=self.day:
            raise ValueError('INTRADAY_CALENDAR')
        if self.cutoff not in grid_5min(self.day)[2::3]:
            raise ValueError('INTRADAY_CUTOFF_NOT_15M')
        self.boards=boards;self.codes=sorted({c for m in boards.values() for c in m})
        if any(not m or len(m)!=len(set(m)) for m in boards.values()):
            raise ValueError('INTRADAY_MEMBERSHIP')
        if not set(candidates)<=set(self.codes):raise ValueError('INTRADAY_CANDIDATE_MEMBERSHIP')
        self.references=references;self.current=current;self.history=history
        self.returns={f:np.full((len(self.codes),240//f),np.nan) for f in (5,15)}
        self.receipts={};self.failures={};self.prefix_shas={};self.detail_rows={}
        self.audit=dict(contract=CONTRACT,declared_members=len(self.codes),
            current_reads=0,history_reads=0,detail_computations=0,missing_members=0)
        self.ci={c:i for i,c in enumerate(self.codes)}
        self._capture_returns()
        members=[[self.ci[c] for c in m] for m in boards.values()]
        baskets=[basket_features(self.returns[15],m) for m in members]
        leaders=observed_leaders(np.stack([b['observed_return_raw'] for b in baskets]),
            np.stack([b['present'] for b in baskets]),np.array([len(m) for m in members]),
            members,self.returns[15],list(boards),self.codes)
        selected={c for c in leaders.leader_observed_stock if isinstance(c,str)}
        self.detail_codes=set(candidates)|selected
        self.audit.update(observed_leaders=sorted(selected),detail_codes=sorted(self.detail_codes),
                          available_members=len(self.receipts),missing_members=len(self.failures))

    @classmethod
    def restore(cls,meta,returns,candidates,current,history):
        """Restore SHA-verified numeric common inputs; no return/leader rebuild.

        Caller verifies the persisted artifacts and source version stamps.
        Candidate additions affect technical details only, not board coverage.
        """
        obj=cls.__new__(cls)
        for name in ('days','cutoff','day','boards','codes','references','receipts','failures','prefix_shas','audit'):
            setattr(obj,name,meta[name])
        if not set(candidates)<=set(obj.receipts):raise ValueError('RESTORED_CANDIDATE_UNAVAILABLE')
        obj.returns=returns;obj.ci={c:i for i,c in enumerate(obj.codes)}
        for f in (5,15):
            if returns[f].shape!=(len(obj.codes),240//f):raise ValueError('RESTORED_CONTEXT_SHAPE')
        obj.current=current;obj.history=history;obj.detail_rows={}
        obj.detail_codes=set(candidates)|set(obj.audit['observed_leaders'])
        obj.audit=dict(obj.audit,current_reads=0,history_reads=0,detail_computations=0,restored_common=True)
        return obj

    def common_metadata(self):
        return {name:getattr(self,name) for name in
                ('days','cutoff','day','boards','codes','references','receipts','failures','prefix_shas','audit')}

    def _capture_returns(self):
        for code in self.codes:
            ref=self.references.get(code)
            if not ref or ref.get('day')!=self.days[-2] or not ref.get('complete'):
                self.failures[code]='PREVIOUS_COMPLETE_SESSION_UNBOUND';continue
            previous=ref.get('close')
            if isinstance(previous,bool) or not isinstance(previous,(int,float)) or not np.isfinite(previous) or previous<=0:
                self.failures[code]='PREVIOUS_CLOSE_INVALID';continue
            try:
                rows,receipt=self.current(code);self.audit['current_reads']+=1
                if receipt.get('clock')!='END_5M' or receipt.get('frequency_minutes')!=5:
                    raise ValueError('CURRENT_RECEIPT_FREQUENCY')
                if not receipt.get('reference') or receipt.get('price_basis')!='RAW':
                    raise ValueError('CURRENT_RECEIPT_UNBOUND')
                prefix=prefix_rows(rows,code,self.day,self.cutoff)
            except (OSError,ValueError,KeyError) as error:
                self.failures[code]=str(error);continue
            close=np.array([r['close'] for r in prefix],dtype=float)
            i=self.ci[code]
            self.returns[5][i,:len(close)]=close/previous-1
            self.returns[15][i,:len(close)//3]=close[2::3]/previous-1
            self.prefix_shas[code]=rows_digest(prefix)
            self.receipts[code]=dict(receipt,prefix_sha256=self.prefix_shas[code],
                                     reference_close=previous,reference_day=self.days[-2])

    def _detail(self,code):
        rows,proof=self.history(code);self.audit['history_reads']+=1
        if proof.get('clock')!='END_5M' or proof.get('price_basis')!='RAW' or not proof.get('reference'):
            raise ValueError('HISTORY_RECEIPT_UNBOUND:'+code)
        raw=[]
        for r in rows:
            if r['day']>=self.day:
                continue
            raw.append(r)
        grouped={}
        for r in raw:grouped.setdefault(r['day'],[]).append(r)
        allowed={d for d,part in grouped.items() if validate_day(part,d)['eligible']}
        previous=self.days[-2]
        if previous not in allowed:raise ValueError('HISTORY_PREVIOUS_DAY_INCOMPLETE:'+code)
        if grouped[previous][-1]['close']!=self.references[code]['close']:
            raise ValueError('HISTORY_REFERENCE_CHANGED:'+code)
        current,receipt=self.current(code)
        prefix=prefix_rows(current,code,self.day,self.cutoff)
        if rows_digest(prefix)!=self.prefix_shas[code]:
            raise ValueError('CURRENT_PREFIX_CHANGED:'+code)
        if receipt.get('reference')!=self.receipts[code]['reference']:
            raise ValueError('CURRENT_RECEIPT_CHANGED:'+code)
        grouped[self.day]=prefix;allowed.add(self.day)
        arrays,rejected=build_minute_arrays(grouped,self.days,allowed_days=allowed)
        frame,ret15,_=run_features(arrays,code,self.days,build_quality(arrays,rejected))
        frame=frame[frame.cutoff_index//16==len(self.days)-1].copy()
        expected=self.returns[15][self.ci[code]]
        if not np.array_equal(np.isnan(ret15[-16:]),np.isnan(expected)) or not np.array_equal(ret15[-16:][np.isfinite(expected)],expected[np.isfinite(expected)]):
            raise ValueError('NATIVE_DAY_RETURN_MISMATCH:'+code)
        self.audit['detail_computations']+=1
        self.detail_rows[code]=frame

    def stock(self,code,days,cutoff):
        if list(days)!=self.days or int(cutoff)!=self.cutoff:
            raise ValueError('INTRADAY_CACHE_BINDING')
        if code not in self.receipts:raise ValueError('CURRENT_MEMBER_UNAVAILABLE:'+code)
        i=self.ci[code]
        if code in self.detail_codes:
            if code not in self.detail_rows:self._detail(code)
            frame=self.detail_rows[code].copy()
        else:
            values=self.returns[15][i];keep=np.flatnonzero(np.isfinite(values))
            frame=pd.DataFrame({n:np.full(len(keep),np.nan) for n in STOCK_COLUMNS})
            frame.insert(0,'code',code)
            frame['cutoff_index']=(len(days)-1)*16+keep
            frame['stock_reference_close_raw']=self.references[code]['close']
            frame['stock_day_return_raw']=values[keep]
            frame['stock_close']=(values[keep]+1)*self.references[code]['close']
            frame['stock_divergence_confirmed']=None
            frame['stock_corporate_action_scale_verified']=False
        return frame,{f:self.returns[f][i].copy() for f in (5,15)}
