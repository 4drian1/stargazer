"""Supported manual daily CLI: completed-date guard before any new-date work."""
import argparse
import json
from datetime import datetime
from pathlib import Path

from postclose_daily_job_v1 import plan,run
from postclose_local_source_bridge_v1 import clock_now


def require_completed_day(day,observed_at):
    now=datetime.fromisoformat(observed_at)
    if now.utcoffset() is None or now.utcoffset().total_seconds()!=8*3600:
        raise ValueError('EXCHANGE_CLOCK_REQUIRED')
    endpoint=datetime.strptime(str(day)+'150000','%Y%m%d%H%M%S').replace(tzinfo=now.tzinfo)
    if endpoint>now:raise ValueError('TARGET_DAILY_BAR_NOT_COMPLETED_NO_FETCH')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--day',type=int,required=True);parser.add_argument('--run',action='store_true')
    parser.add_argument('--calendar',type=Path);parser.add_argument('--background',type=Path)
    args=parser.parse_args()
    try:
        if args.run:require_completed_day(args.day,clock_now())
        result=run(args.day,args.calendar,args.background) if args.run else plan(args.day,args.calendar,args.background)[0]
        print(json.dumps(result,ensure_ascii=False,allow_nan=False))
        return 2 if result.get('ready') is False else 0
    except (ValueError,OSError,KeyError) as exc:
        print(json.dumps(dict(status='REJECTED',error=str(exc)),ensure_ascii=False));return 2


if __name__=='__main__':raise SystemExit(main())
