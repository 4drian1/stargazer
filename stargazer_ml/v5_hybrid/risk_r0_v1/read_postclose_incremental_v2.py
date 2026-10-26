"""Read validated refreshed sidecar, never refetch or change the old D2 baseline."""
import argparse
import json
from pathlib import Path

import pandas as pd

from postclose_research_scorer_v1 import HERE,ResearchScorer,strict_json,sha
from postclose_local_source_bridge_v1 import latest_day_payload

ROOT=HERE/'postclose_incremental_refresh_20261017_v2'


def load_refreshed(root=ROOT):
    root=Path(root)
    accepted=strict_json((root/'readback_acceptance.json').read_bytes())
    if accepted['status']!='PASS_INCREMENTAL_FEATURE_AND_SCORE_READBACK':
        raise ValueError('REFRESH_NOT_ACCEPTED')
    for name,h in accepted['artifact_sha256'].items():
        if sha((root/name).read_bytes())!=h:raise ValueError('REFRESH_ARTIFACT_DRIFT:'+name)
    scorer=ResearchScorer()
    if sha(Path(__file__).read_bytes())!=accepted['reader_sha256']:
        raise ValueError('REFRESH_READER_DRIFT')
    data=pd.read_parquet(root/'feature_sidecar.parquet')
    if (set(data.columns)!={'code','day',*scorer.fields} or data[['code','day']].duplicated().any()
            or sorted(data.day.unique())!=[20261015,20261016]):
        raise ValueError('REFRESH_SCHEMA_OR_KEYS')
    for _,group in data.groupby('day'):
        if sorted(group.code)!=scorer.manifest['research_codes']:
            raise ValueError('REFRESH_POPULATION_DRIFT')
    return data,scorer


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--day',type=int,default=20261016)
    parser.add_argument('--score',action='store_true')
    args=parser.parse_args()
    try:
        data,scorer=load_refreshed()
        if args.day not in data.day.unique():raise ValueError('DAY_NOT_REFRESHED_NO_FALLBACK')
        payload,_=latest_day_payload(data[data.day.eq(args.day)],scorer)
        result=scorer.score(payload) if args.score else payload
        if args.score:
            result.update(source_day=args.day,forward_validation_eligible=False,
                          scope='REFRESHED_RESTRICTED_HISTORICAL_REPLAY_NOT_LIVE')
        print(json.dumps(result,ensure_ascii=False,allow_nan=False))
    except (ValueError,OSError,KeyError) as exc:
        print(json.dumps(dict(status='REJECTED',error=str(exc)),ensure_ascii=False))
        return 2
    return 0


if __name__=='__main__':raise SystemExit(main())
