"""freeze_comparison_points_44win.py — 冻结 44 窗价格比较点清单（零网络；可复算）。

从 five_day_window_map_v3.json 枚举窗口，构造比较点：
  每窗 1 个 P 日收盘(close) + 5 个后五日各自 low = 6 引用 → 44×6 = 264 引用；
  按 (code, date, field) 去重 → 唯一价格点（用户读数：225）。
落盘：independent_fetch/batch_d_mainpop_30/price_comparison_points_44win.json
含：输入哈希（map 文件 + 人口/缓存哈希）、代码版本（三脚本 sha256）、按股票合并的日期范围（取数计划用）。
"""
from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path

HERE = Path(__file__).resolve().parent
BATCH = HERE / 'independent_fetch' / 'batch_d_mainpop_30'
MAP_PATH = BATCH / 'five_day_window_map_v3.json'


def dint(s: str) -> int:
    return int(s.replace('-', ''))


def main():
    MAP = json.loads(MAP_PATH.read_text(encoding='utf-8'))
    map_sha = hashlib.sha256(MAP_PATH.read_bytes()).hexdigest()[:16]
    inp = MAP.get('input_hash', {})

    points = []
    seen = set()
    stock_dates = defaultdict(set)
    n_ref = 0
    for c in MAP['codes']:
        code = c['code']
        for w in c['windows']:
            p = int(w['P'])
            n_ref += 1
            # P 日收盘
            key = (code, p, 'close')
            if key not in seen:
                seen.add(key)
                points.append(dict(code=code, date=p, field='close'))
            stock_dates[code].add(p)
            for wd in w['window']:
                d = dint(wd)
                n_ref += 1
                key = (code, d, 'low')
                if key not in seen:
                    seen.add(key)
                    points.append(dict(code=code, date=d, field='low'))
                stock_dates[code].add(d)

    # 按股票合并日期范围（取数计划：每股票一次区间请求）
    stock_plan = {}
    for code, ds in stock_dates.items():
        lo, hi = min(ds), max(ds)
        start = (datetime.strptime(str(lo), '%Y%m%d') - timedelta(days=7)).strftime('%Y%m%d')
        end = (datetime.strptime(str(hi), '%Y%m%d') + timedelta(days=2)).strftime('%Y%m%d')
        stock_plan[code] = dict(start=start, end=end, n_points=len(ds))

    # 代码版本：三脚本 sha256
    code_files = ['label_readiness_inventory.py', 'five_day_window_calendar_map.py',
                  'price_verification_batch_44win.py']
    code_sha = {f: hashlib.sha256((HERE / f).read_bytes()).hexdigest()[:12]
                for f in code_files if (HERE / f).exists()}

    out = dict(
        generated_from=str(MAP_PATH.name),
        map_sha256_16=map_sha,
        map_input_hash=inp,                 # population/cache/reanalysis sha
        code_version=code_sha,
        windows_total=MAP.get('total_candidate_windows'),
        stocks_total=len(stock_dates),
        references_total=n_ref,             # 应 = 264
        unique_points_total=len(points),    # 应 = 225
        per_stock_date_range=stock_plan,
        points=points,
    )
    out_path = BATCH / 'price_comparison_points_44win.json'
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding='utf-8')
    print(f'引用={n_ref} 唯一点={len(points)} 股票={len(stock_dates)}')
    print(f'map_sha={map_sha} pop={inp.get("population")} cache={inp.get("cache")}')
    print(f'written {out_path.name}')


if __name__ == '__main__':
    main()
