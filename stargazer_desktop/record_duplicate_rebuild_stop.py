"""Record the verified interrupted duplicate; never restart or refund its budget."""
from datetime import datetime
import json
from pathlib import Path


def main():
    runtime=Path(__file__).resolve().parent/'dist/data/postclose_risk'
    current=json.loads((runtime/'state.json').read_bytes())
    run=runtime/'run_20261021_211456_011617'
    if Path(current['run_dir']).resolve()!=run.resolve() or current.get('status')!='RUNNING':
        raise ValueError('ACTIVE_JOB_CHANGED_DO_NOT_OVERWRITE')
    record=dict(previous_activity=current,reason='DUPLICATE_AFTER_GENERATION_TIMESTAMP_CHANGE',
        stopped_verified_processes=[43176,13484],recorded_at=datetime.now().isoformat(),
        quota_reset=False,refunded=False,automatic_resume=False)
    with (run/'interruption_record.json').open('x',encoding='utf-8') as stream:
        json.dump(record,stream,ensure_ascii=False,indent=2)
    failure={**current,'status':'FAILED','reason':'DUPLICATE_REBUILD_INTERRUPTED; published matching evidence preserved',
             'stage':'TERMINATED_DUPLICATE','automatic_resume':False}
    if not (run/'terminal.json').exists():
        with (run/'terminal.json').open('x',encoding='utf-8') as stream:
            json.dump(failure,stream,ensure_ascii=False,indent=2)
    if json.loads((runtime/'state.json').read_bytes()).get('run_dir')!=current['run_dir']:
        raise ValueError('ACTIVE_JOB_CHANGED_DO_NOT_OVERWRITE')
    tmp=runtime/'state.stop.tmp'
    tmp.write_text(json.dumps(failure,ensure_ascii=False),encoding='utf-8')
    tmp.replace(runtime/'state.json')
    print('DUPLICATE_TERMINATED_WITHOUT_QUOTA_RESET; published acceptance untouched')


if __name__=='__main__':
    main()
