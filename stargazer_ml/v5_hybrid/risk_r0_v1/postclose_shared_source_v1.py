"""Reuse an accepted TDX common snapshot across candidate-list changes.

No network, fitting, scope reduction, copies of raw history, or changes to
mathematical kernels. Stock and index stages use separate bounded processes.
"""
from pathlib import Path

import tdx_postclose_runtime_v1 as native


def implementation_key():
    return native.digest({n:native.native.sha(Path(__file__).with_name(n)) for n in
        ('postclose_shared_source_v1.py','tdx_shared_prepare_v1.py','intraday_native_context_v1.py')})


def common_identity(spec):
    # Candidates affect scoring, not the shared inputs. A genuinely new supplied
    # member changes bound/files and therefore DOES change the common identity.
    return native.digest({k:v for k,v in spec.items() if k!='candidates'})


def reusable(runtime,spec):
    expected=common_identity(spec)
    for marker in sorted((Path(runtime)/'tdx_prepared'/str(spec['P'])).glob('*/complete.json')):
        try:
            old=native.read_json(marker.parent/'request.json')
            if common_identity(old)!=expected:continue
            impl=marker.parent/'shared_source_impl.json'
            if impl.is_file() and native.read_json(impl)['sha256']!=implementation_key():continue
            manifest=native.read_json(marker.parent/'source_manifest.json')
            if not set(spec['candidates'])<=set(manifest['entries']):continue
            return marker.parent
        except (OSError,ValueError,KeyError):
            continue
    return None


def produce(day,codes,output,progress,*,cutoff=None,index_source=None):
    output=Path(output);runtime=output.parent
    selection=native.read_json(output/'input_snapshot.json')
    spec=native.settings(selection,runtime)
    if int(day)!=spec['P'] or sorted(codes)!=spec['candidates']:
        raise ValueError('TDX_SELECTION_IDENTITY')
    if cutoff is not None and cutoff!=spec['cutoff']:raise ValueError('TDX_POSTCLOSE_ONLY')
    folder=reusable(runtime,spec);cold=folder is None;new_details=[]
    if folder is None:
        # Cold shared build consumes the SAME full membership/source settings,
        # but full-history indicators only for candidates/observed leaders.
        import subprocess
        import sys
        key=native.digest([spec,implementation_key()])
        folder=runtime/'tdx_prepared'/str(day)/key
        native.check_budget(runtime,48*1024**2);folder.mkdir(parents=True,exist_ok=True)
        if any(folder.iterdir()):raise ValueError('TDX_PREPARATION_INCOMPLETE_RETAINED')
        native.write(folder,'request.json',spec)
        for module,args in [('tdx_shared_prepare_v1.py',[str(folder)]),
                            ('tdx_postclose_runtime_v1.py',['market',str(folder)])]:
            progress(stage='TDX_PREPARE_MARKET' if args[0]=='market' else 'TDX_MEMBER_SESSION_PATHS',
                done=0,total=2 if args[0]=='market' else len(spec['files']))
            result=subprocess.run([sys.executable,'-B','-X','utf8',str(Path(__file__).with_name(module)),*args],
                stdout=sys.stdout,stderr=sys.stderr,timeout=600,
                creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
            if result.returncode:raise ValueError('TDX_SHARED_PREPARATION_FAILED:'+module)
    # loaded() validates the original kernel and EVERY artifact SHA, including
    # returns, row order, index context and the source manifest. Failures remain
    # failures; corrupt evidence is never treated as a warm hit.
    snapshot,capture,failures=native.loaded(folder,output,runtime)
    complete=native.read_json(folder/'stock_complete.json')
    if complete.get('strategy')=='MEMBER_TWO_SESSION_PATHS_PLUS_SELECTIVE_FULL_HISTORY':
        details=set(complete['cache_audit']['detail_codes'])
        # Existing member promoted to candidate: compute that stock ONLY. The
        # common return paths/index context stay immutable and byte-identical.
        for code in set(codes)-details:
            import numpy as np
            import pandas as pd
            import io
            key=native.digest([common_identity(spec),implementation_key(),code])
            detail=runtime/'tdx_candidate_details'/str(day)/(key+'.parquet')
            proof=detail.with_suffix('.sha256')
            old_rows,old_returns=snapshot._prepared_context
            if detail.exists() and proof.exists():
                raw=detail.read_bytes()
                if native.native.hashlib.sha256(raw).hexdigest()!=proof.read_text():raise ValueError('CANDIDATE_DETAIL_SHA')
                tail=pd.read_parquet(io.BytesIO(raw))
            else:
                if detail.exists() or proof.exists():raise ValueError('CANDIDATE_DETAIL_INCOMPLETE_RETAINED')
                info=native.native.read_lc5_file(spec['files'][code]['path'])
                source=dict(path=spec['files'][code]['path'],sha256=info['sha256'])
                native.native.P=spec['P'];native.native.E=spec['E'];native.native.CUTOFF=spec['cutoff']
                cache=native.native.NativeCache({code:source},list(snapshot.days),set())
                rows,ret=cache.stock(code,list(snapshot.days),spec['cutoff'],info=info)
                for f in (5,15):
                    if not np.array_equal(ret[f][-(240//f):],old_returns[f][code],equal_nan=True):
                        raise ValueError('CANDIDATE_MEMBER_RETURN_CHANGED:'+code)
                tail=rows[rows.cutoff_index//16==len(snapshot.days)-1].copy();raw=tail.to_parquet(index=False)
                native.check_budget(runtime,len(raw)+64);detail.parent.mkdir(parents=True,exist_ok=True)
                with detail.open('xb') as stream:stream.write(raw)
                with proof.open('x') as stream:stream.write(native.native.hashlib.sha256(raw).hexdigest())
                new_details.append(code)
            if set(tail.code)!={code} or (tail.cutoff_index//16!=len(snapshot.days)-1).any():
                raise ValueError('CANDIDATE_DETAIL_BINDING')
            replacement=dict(old_rows)
            replacement[code]=tail
            snapshot._prepared_context=(replacement,old_returns)
    snapshot._source_identity=native.digest(spec)
    capture.save('common_context_reuse.json',native.json.dumps(dict(
        contract='TDX_COMMON_INPUTS_ROSTER_INDEPENDENT_V1',
        common_identity=common_identity(spec),prepared_folder=str(folder),
        cutoff=spec['cutoff'],candidates=list(codes),cold_common_build=cold,
        full_history_indicator_reads=(complete.get('cache_audit',{}).get('history_reads',0) if cold else 0)+len(new_details),
        new_candidate_details=new_details,
        note='原已验共享输入复用；当前候选按冻结模型重新评分，不复用旧名单分数'
    ),ensure_ascii=False).encode())
    progress(stage='REUSING_PREPARED_CONTEXT',done=len(codes),total=len(codes))
    return snapshot,capture,failures


produce.native_tdx=True
