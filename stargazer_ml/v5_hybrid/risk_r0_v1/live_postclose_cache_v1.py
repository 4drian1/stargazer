"""Verified prior-day archive references for incremental daily minute capture."""
import json
from pathlib import Path


def prior_entries(runtime, day):
    options=[]
    for terminal in Path(runtime).glob('run_*/terminal.json'):
        row=json.loads(terminal.read_text(encoding='utf-8'))
        previous=row.get('data_date',0)
        if row.get('status')=='COMPLETE' and 0<int(previous)<int(day):
            if Path(row['run_dir']).resolve()!=terminal.parent.resolve():
                raise ValueError('PRIOR_RUNTIME_PATH_MISMATCH')
            options.append((int(previous),terminal.parent/'source_manifest.json'))
    if not options:
        return {}
    _,path=max(options,key=lambda v:v[0])
    if path.stat().st_size>32*1024**2:
        raise ValueError('PRIOR_MANIFEST_SIZE')
    manifest=json.loads(path.read_text(encoding='utf-8'))
    return manifest['entries']


def remaining_periods(last_day, days):
    """Only new trading dates; preserve the old full indicator prefix by reference."""
    return [str(int(day))+'*' for day in days if int(day)>int(last_day)]
