"""Verified per-stock feature rows shared across plan IDs at one cutoff.

No model fitting or prediction cache: the existing frozen models still predict
on their exact matrices. No lossy JSON serialization of floating-point inputs.
The postclose job's cross-process worker lock is held for the whole operation.
"""
from pathlib import Path
import io
import json
import numpy as np
import pandas as pd

import tdx_postclose_runtime_v1 as rt
from shared_risk_context_v1 import kernel


def build(snapshot,candidates,capture,feature_cache,builder):
    provenance=capture.output/'common_context_reuse.json'
    if not provenance.is_file():return builder(snapshot,candidates,feature_cache=feature_cache)
    source=json.loads(capture.read(provenance));codes=[c['code'] for c in candidates]
    if len(codes)!=len(set(codes)) or any(c not in snapshot.stock_minutes or
            int(r['cutoff_ts'])!=snapshot.cutoff or int(r['P'])!=snapshot.days[-2] or
            int(r['E'])!=snapshot.days[-1] for c,r in zip(codes,candidates)):
        raise ValueError('CANDIDATE_FEATURE_CACHE_BINDING')
    import rebuild_features_v1 as frozen
    _,columns232,columns266,_=frozen.load_spec()
    accepted=rt.read_json(Path(source['prepared_folder'])/'complete.json')['files']
    numeric_inputs={n:accepted[n] for n in ('stocks.parquet','returns.npz','market.parquet','states.json','shared.parquet')}
    version=rt.digest([source['common_identity'],numeric_inputs,kernel(),rt.native.sha(__file__)])
    root=capture.budget_root/'tdx_candidate_features'/str(snapshot.days[-1])/version
    loaded={};missing=[]
    for code in codes:
        folder=root/code;marker=folder/'complete.json'
        if not marker.exists():missing.append(code);continue
        meta=rt.read_json(marker)
        if (meta.get('code'),meta.get('cutoff'),meta.get('version'))!=(code,snapshot.cutoff,version):
            raise ValueError('CANDIDATE_FEATURE_CACHE_IDENTITY')
        frames=[]
        for name in ('frame','x232','x266'):
            file=folder/(name+'.parquet');raw=file.read_bytes()
            if rt.native.hashlib.sha256(raw).hexdigest()!=meta['files'].get(name):
                raise ValueError('CANDIDATE_FEATURE_CACHE_SHA:'+code)
            frames.append(pd.read_parquet(io.BytesIO(raw)))
        if any(len(f)!=1 for f in frames) or frames[0].code.tolist()!=[code] or list(frames[1])!=columns232 or list(frames[2])!=columns266:
            raise ValueError('CANDIDATE_FEATURE_CACHE_SCHEMA')
        loaded[code]=frames
    if missing:
        new=[r for r in candidates if r['code'] in missing]
        frame,x232,x266,audit=builder(snapshot,new,feature_cache=feature_cache)
        if frame.code.tolist()!=missing:raise ValueError('CANDIDATE_FEATURE_RESULT_ORDER')
        for i,code in enumerate(missing):
            frames=[f.iloc[[i]].reset_index(drop=True) for f in (frame,x232,x266)]
            loaded[code]=frames;folder=root/code
            raw={n:f.to_parquet(index=False) for n,f in zip(('frame','x232','x266'),frames)}
            rt.check_budget(capture.budget_root,sum(map(len,raw.values()))+4096)
            folder.mkdir(parents=True,exist_ok=True)
            if any(folder.iterdir()):raise ValueError('CANDIDATE_FEATURE_PARTIAL_RETAINED')
            for name,body in raw.items():
                with (folder/(name+'.parquet')).open('xb') as stream:stream.write(body)
            meta=dict(code=code,cutoff=snapshot.cutoff,version=version,
                files={n:rt.native.hashlib.sha256(b).hexdigest() for n,b in raw.items()})
            with (folder/'complete.json').open('x',encoding='utf8') as stream:json.dump(meta,stream)
    result=[pd.concat([loaded[c][i] for c in codes],ignore_index=True) for i in range(3)]
    if np.isinf(result[2].to_numpy(dtype=float)).any():raise ValueError('INFINITE_FEATURE_VALUE')
    audit=dict(contract='PER_STOCK_FROZEN_FEATURE_REUSE_V1',cutoff=snapshot.cutoff,
        cache_hits=len(codes)-len(missing),computed_codes=missing,common_identity=source['common_identity'],
        note='逐票已验矩阵复用；名单ID、名称、推送文案不进入键；模型仍按冻结权重推理')
    capture.save('candidate_feature_reuse.json',json.dumps(audit,ensure_ascii=False).encode())
    return (*result,audit)
