"""Local-only rule projection of the existing common pool; never fits a model."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent/'risk_r0_v1'))
sys.path.insert(0, str(HERE.parents[2]/'stargazer_desktop'))
from golden_rules import build_rule, cycle_anchors, project
import joint_shadow_compute_v2 as compute
from gcp_demo_snapshot import build, calendar_file, ensure_daily_current


def transform(parent):
    for item in parent.get('inputs', []):
        if hashlib.sha256(Path(item['path']).read_bytes()).hexdigest() != item['sha256']:
            raise ValueError('PARENT_SOURCE_CHANGED:'+Path(item['path']).name)
    calendar = sorted(map(int, json.loads(calendar_file().read_bytes())))
    p = int(parent['data_date'])
    rules = {}
    for row in parent['rows']:
        code = row['code']
        g = compute.daily_prefix(compute.read_daily_prefix(code, p), code, p)
        tr = pd.concat([g.high-g.low, (g.high-g.close.shift(1)).abs(),
                        (g.low-g.close.shift(1)).abs()], axis=1).max(axis=1)
        atr = pd.Series(tr.rolling(14).mean().to_numpy(), index=g.date)
        base = [d for d in calendar if d <= p][-5:]
        if len(base)!=5 or any(d not in atr.index or pd.isna(atr.loc[d]) or atr.loc[d]<=0 for d in base):
            raise ValueError('ATR_BASE_INCOMPLETE:'+code)
        ref = float(atr.loc[base].median())
        daily = {int(r.date): {k: float(getattr(r,k)) for k in ('open','high','low','close')} for r in g.itertuples()}
        anchor = cycle_anchors(daily)
        last = g.iloc[-1]
        if abs(float(last.close)-float(row['reference_price'])) > 1e-8:
            raise ValueError('P_CLOSE_CHANGED:'+code)
        rules[code] = build_rule(float(last.close), float(last.high), float(last.low),
                                ref, anchor['high'], anchor['low'])
    value = project(parent, rules)
    value['rule_inputs'] = [{'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
                            for p in (Path(__file__), HERE.parents[2]/'stargazer_desktop/golden_rules.py',
                                      compute.TRIGGER_SOURCE,HERE/'gcp_golden_intraday_v1.py')]
    return value


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--daily-dir', type=Path, required=True)
    parser.add_argument('--parent', type=Path)
    args = parser.parse_args()
    compute.DAILY_DIR = args.daily_dir
    if args.parent:
        parent = json.loads(args.parent.read_bytes())
    else:
        ensure_daily_current(args.daily_dir)
        parent = build(args.daily_dir)
    print(json.dumps(transform(parent), ensure_ascii=False, allow_nan=False))
