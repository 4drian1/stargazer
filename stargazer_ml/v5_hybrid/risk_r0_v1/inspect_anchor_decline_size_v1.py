"""Describe the user-selected local index period; no thresholds fitted, no HTTP."""
from collections import defaultdict
import hashlib
import json
from pathlib import Path

from research_budget_v2 import bytes_used, TOTAL_CAP
from tdx_index_acceptance_v1 import read_binary, grid


def run():
    root = Path(__file__).resolve().parent
    out = root / 'anchor_decline_size_20261017_v1.json'
    if out.exists():
        raise RuntimeError('OUTPUT_EXISTS_NO_OVERWRITE')
    source = Path(r'D:\a股数据\沪指\sh999999.lc5')
    body = source.read_bytes()
    grouped = defaultdict(list)
    for row in read_binary(body):
        grouped[row['date'] // 1000000].append(row)
    days = []
    for day, rows in sorted(grouped.items()):
        if sorted(r['date'] for r in rows) != grid(day, 5):
            raise ValueError('SOURCE_DAY_INCOMPLETE')
        rows.sort(key=lambda r: r['date'])
        days.append(dict(day=day, open=rows[0]['open'], close=rows[-1]['close']))
    report = []
    for i, bar in enumerate(days):
        if i < 3 or not 20260921 <= bar['day'] <= 20261016:
            continue
        o, c = bar['open'], bar['close']
        report.append(dict(**bar, bearish_body_pct=max(0, (o-c)/o*100),
                           day_decline_pct=(1-c/days[i-1]['close'])*100,
                           three_day_decline_pct=(1-c/days[i-3]['close'])*100,
                           three_lower_closes=all(days[j]['close'] < days[j-1]['close'] for j in range(i-2, i+1))))
    result = dict(source=str(source), source_sha256=hashlib.sha256(body).hexdigest(),
                  code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  period=[20260921, 20261016], rows=report, thresholds_frozen=False,
                  provenance='Daily OHLC reconstructed from local native 5min; END clock remains assumption',
                  purpose='Descriptive body-versus-close-return comparison, not strategy/label admission',
                  http_requests=0, training_ready=False)
    payload = json.dumps(result, ensure_ascii=False, allow_nan=False, indent=2).encode('utf-8')
    if len(payload) > 1024**2 or bytes_used() + len(payload) > TOTAL_CAP:
        raise RuntimeError('BUDGET_STOP')
    with out.open('xb') as stream:
        stream.write(payload)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    run()
