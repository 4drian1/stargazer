"""SHA-bound OHLCV aggregates; full-history indicators are never truncated."""
import gzip,hashlib,io,json
from pathlib import Path
import numpy as np
import pandas as pd
import msgpack
from build_joint_risk_features_v1 import aggregate_minutes,features_from_arrays
from joint_shadow_context_v1 import _prefix
from live_postclose_source_v1 import clean_ohlc,FIELDS

def digest(body):return hashlib.sha256(body).hexdigest()

class PrefixFeatureCache:
    def __init__(self,snapshot,capture):
        self.snapshot,self.capture=snapshot,capture
        # Layer-bound identity (#18.199): the reusable math is bound only to
        # math-layer files; capture ingestion has its own version, checked per
        # entry, so capture-path edits do not wipe valid math results.
        from cache_layer_keys_v1 import math_key, capture_key
        from storage_roots_v1 import assets_root
        self.kernel=math_key()
        self.capture_version=capture_key()
        self.root=assets_root()/'history_foundation_v1'
        self.legacy=capture.budget_root/'stock_prefix_cache_v1'
        self.audit=dict(hit=0,incremental=0,cold=0,invalid=0,write_skipped_budget=0,bytes_written=0)

    def _sources(self,code):
        entry=self.snapshot.stock_minutes.entries[code]
        seed=entry.get('seed')
        refs=([seed] if seed else [])+entry['extensions']
        bodies={}
        for ref in refs:
            key=(ref['path'],ref['sha256'])
            if key not in bodies:bodies[key]=self.capture.read(ref['path'],ref['sha256'])
        signature=dict(seed=[seed['path'],seed['sha256']] if seed else None,
                       extensions=[[r['path'],r['sha256']] for r in entry['extensions']])
        return entry,signature,bodies

    def _receipt(self,code,cutoff):
        r=self.snapshot.receipts.get(code) or {};observed=self.snapshot.observed_at or cutoff
        research=self.snapshot.membership_scope=='SNAPSHOT_RECONSTRUCTION_RESEARCH'
        if (not r.get('reference') or r.get('clock')!='END_1M' or r.get('price_basis')!='RAW' or
            (r.get('available_at') is None and not(research and r.get('availability_basis')=='HISTORICAL_RECONSTRUCTION_ASSUMPTION')) or
            (r.get('available_at') is not None and r['available_at']>observed)):
            raise ValueError('FEATURE_CACHE_SOURCE_RECEIPT_INVALID')

    def _delta(self,entry,signature,meta,bodies,after,cutoff):
        frames=[];end=after*1000000+235959
        if entry.get('seed'):
            seed=entry['seed'];frames.append(pd.read_parquet(io.BytesIO(bodies[(seed['path'],seed['sha256'])]),filters=[('date','>',end)]))
        old_count=len(meta['sources']['extensions'])
        for i,ref in enumerate(entry['extensions']):
            raw=gzip.decompress(bodies[(ref['path'],ref['sha256'])])
            if digest(raw)!=ref['raw_sha256']:raise ValueError('RAW_RESPONSE_HASH_MISMATCH')
            try:rows=msgpack.unpackb(raw,raw=False,strict_map_key=False)
            except (ValueError,msgpack.UnpackException):rows=json.loads(raw)
            rows=[r[1] if isinstance(r,list) and len(r)==2 else r for r in rows]
            frame=pd.DataFrame(rows,columns=FIELDS)
            if i>=old_count and len(frame) and (frame.date<=end).any():
                raise ValueError('APPENDED_SOURCE_REVISES_HISTORY')
            frames.append(frame.loc[frame.date>end])
        raw=pd.concat(frames,ignore_index=True) if frames else pd.DataFrame(columns=FIELDS)
        from live_postclose_source_v1 import merge_minutes
        raw=merge_minutes(pd.DataFrame(columns=FIELDS),raw.to_dict('records'),signature['code'],cutoff)
        raw,bad=clean_ohlc(raw)
        if len(raw):raw=_prefix(raw,cutoff,self.snapshot.receipts[signature['code']],stock=True,
            research=self.snapshot.membership_scope=='SNAPSHOT_RECONSTRUCTION_RESEARCH',observed_at=self.snapshot.observed_at)
        return raw,bad

    def _store(self,code,days,cutoff,sources,arrays,quality,bad):
        if cutoff%1000000!=150000:return
        meta=dict(code=code,days=list(map(int,days)),cutoff=cutoff,kernel=self.kernel,
                  capture=self.capture_version,sources=sources,quality=quality,quarantined=bad)
        key=digest(json.dumps(meta,sort_keys=True).encode());folder=self.root/code
        target=folder/(str(cutoff)+'_'+key+'.json')
        if target.exists():return
        sink=io.BytesIO();np.savez_compressed(sink,**{f'{f}_{k}':v for f,fields in arrays.items() for k,v in fields.items()})
        body=sink.getvalue();meta['array_sha256']=digest(body);raw=json.dumps(meta,sort_keys=True).encode()
        # The history foundation lives in the assets zone; assets/records have
        # no default budget — an unauthorized zone refuses new bytes instead
        # of silently consuming a proposal number.
        from storage_roots_v1 import zone_headroom, ZoneWriter
        headroom = zone_headroom('assets')
        if headroom is None:
            self.audit['write_skipped_budget'] += 1
            self.audit['skip_reason'] = 'UNAUTHORIZED_ASSETS_BUDGET'
            return
        if headroom < len(body) + len(raw) + 2 * 1024**2:
            self.audit['write_skipped_budget'] += 1
            return
        try:
            with ZoneWriter(folder):
                with target.with_suffix('.npz').open('xb') as stream:
                    stream.write(body)
                with target.open('xb') as stream:
                    stream.write(raw)
                # Marker covers the WHOLE folder: every version's data files,
                # old ones included (persistent assets are never pruned by a
                # write). Migration coverage then matches reality.
                files = {}
                for f in sorted(folder.iterdir()):
                    if f.suffix == '.json' and f.name != 'COMPLETE.json':
                        files[f.name] = digest(f.read_bytes())
                for f in sorted(folder.glob('*.npz')):
                    files[f.name] = digest(f.read_bytes())
                marker = dict(contract='HISTORY_FOUNDATION_V1', code=code,
                              cutoff=int(cutoff), kernel=self.kernel,
                              capture=self.capture_version,
                              files=files)
                (folder / 'COMPLETE.json').write_text(
                    json.dumps(marker, ensure_ascii=False, sort_keys=True), encoding='utf-8')
        except ValueError:
            self.audit['write_skipped_budget'] += 1
            return
        # Persistent assets are NOT pruned by ordinary writes: old math or
        # capture versions stay until explicit authorization + reference
        # checks dispose of them (ordinary cache cleaning must not delete
        # assets either).
        self.audit['bytes_written'] += len(raw) + len(body)

    def stock(self,code,days,cutoff):
        self._receipt(code,cutoff);entry,signature,bodies=self._sources(code)
        arrays=quality=None;bad=[]
        roots=[self.root/code]
        if (self.legacy/code).is_dir() and (self.legacy/code).resolve()!= (self.root/code).resolve():
            roots.append(self.legacy/code)  # read-through for pre-zone entries
        candidates=[]
        for r in roots:
            if (r.parent/(r.name+'.writing.lock')).exists():
                continue  # a writer owns this folder right now; never read half-written state
            candidates.extend(p for p in sorted(r.glob('*.json'),reverse=True)
                              if p.name != 'COMPLETE.json')
        for path in candidates:
            try:
                meta=json.loads(self.capture.read(path));n=len(meta['days'])
                old=meta['sources'];prior=old['extensions'];current=signature['extensions']
                if (meta['kernel']!=self.kernel or meta.get('capture')!=self.capture_version
                    or meta['code']!=code or meta['cutoff']>cutoff or
                    meta['days']!=list(days[:n]) or old['seed']!=signature['seed'] or current[:len(prior)]!=prior):continue
                blob=self.capture.read(path.with_suffix('.npz'),meta['array_sha256'])
                with np.load(io.BytesIO(blob),allow_pickle=False) as z:
                    arrays={f:{k:z[f'{f}_{k}'].copy() for k in ('open','high','low','close','volume')} for f in (5,15)}
                for f in arrays:
                    if any(v.dtype!=np.float64 or len(v)!=n*(240//f) for v in arrays[f].values()):raise ValueError('CACHE_SHAPE')
                quality=dict(meta['quality']);bad=list(meta['quarantined'])
                if n==len(days) and signature==old:
                    self.audit['hit']+=1;break
                if n>=len(days):arrays=None;continue
                raw,new_bad=self._delta(entry,{**signature,'code':code},meta,bodies,meta['days'][-1],cutoff)
                added,q=aggregate_minutes(raw,code,np.asarray(days[n:]),set())
                arrays={f:{k:np.r_[arrays[f][k],added[f][k]] for k in arrays[f]} for f in arrays}
                quality['invalid_or_excluded_rows']+=q['invalid_or_excluded_rows'];quality['feature_rows']+=q['feature_rows'];bad+=new_bad
                self.audit['incremental']+=1;break
            except (ValueError,KeyError,OSError,EOFError):
                self.audit['invalid']+=1;arrays=None
        if arrays is None:
            raw=_prefix(self.snapshot.stock_minutes[code],cutoff,self.snapshot.receipts[code],stock=True,
                research=self.snapshot.membership_scope=='SNAPSHOT_RECONSTRUCTION_RESEARCH',observed_at=self.snapshot.observed_at)
            arrays,quality=aggregate_minutes(raw,code,np.asarray(days),set())
            bad=list(self.snapshot.stock_minutes.quarantined.get(code,[]));self.audit['cold']+=1
        if bad:self.snapshot.stock_minutes.quarantined[code]=bad
        self._store(code,days,cutoff,signature,arrays,quality,bad)
        rows,_,_=features_from_arrays(arrays,code,np.asarray(days),quality)
        refs=dict(zip(days,rows.groupby(rows.cutoff_index//16).stock_reference_close_raw.first().reindex(range(len(days))).to_numpy()))
        ret={}
        for f in (5,15):
            close=arrays[f]['close'];reference=np.repeat(np.array([refs.get(d,np.nan) for d in days]),240//f)
            values=np.divide(close,reference,out=np.full_like(close,np.nan),where=np.isfinite(reference)&(reference>0))-1
            ret[f]=np.where(np.isfinite(values)&(values>-1),values,np.nan)
        done=self.audit['hit']+self.audit['incremental']+self.audit['cold']
        progress=getattr(self.snapshot.stock_minutes,'progress',None)
        if progress and (done%20==0 or done==len(self.snapshot.stock_minutes)):
            progress(stage='REBUILDING_FROZEN_FEATURES',done=done,total=len(self.snapshot.stock_minutes),
                requests=self.capture.requests,read_bytes=self.capture.read_bytes,disk_bytes=self.capture.disk_bytes,
                feature_cache_hits=self.audit['hit'],feature_cache_incremental=self.audit['incremental'])
        return rows,ret
