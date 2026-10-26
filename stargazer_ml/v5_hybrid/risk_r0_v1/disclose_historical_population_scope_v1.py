"""Disclose visible historical population scope (roadmap Blocker 2 follow-up).

Reads the EXISTING inventory; writes a disclosure. Zero-network, no overwrite of the
inventory. Explicitly: cache-visible scope is NOT a verified delisted population;
no back-filtering with today's stock list; no survivor-bias-free backtest claim;
market index not bound; price scale not verified.
"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / 'historical_stock_day_preparation_v1'
INV = OUT / 'population_inventory.json'

# In-scope prefixes per HISTORICAL_STOCK_DAY_V1 contract. Anything else is excluded.
IN_SCOPE_PREFIXES = ('000', '001', '002', '003', '300', '301',
                     '600', '601', '603', '605', '688')


def main():
    rep = json.loads(INV.read_text(encoding='utf-8'))
    L = []
    w = lambda s='': L.append(s)

    w('# 历史股票日 可见人口范围披露（Blocker 2）')
    w()
    w('- 口径：**缓存可见证券范围，非已证完整退市人口**。来源 = `data_daily_parts` 并集，'
      '未与交易所官方退市清单核销。')
    w(f'- 研究 P 范围：2019-01-01 至 2026-09-30（2018 仅预热）。日历并集 '
      f'{rep["calendar_start"]}–{rep["calendar_end"]}。')
    w(f'- 范围内代码数：**{rep["in_scope_codes"]}**（前缀 '
      f'{"/".join(IN_SCOPE_PREFIXES)}）。')
    w(f'- 物理行：{rep["physical_rows"]:,}；重复键超额={rep["duplicate_excess"]}；'
      f'冲突日={rep["conflicting_dates"]}；无效唯一股票日={rep["invalid_unique_stock_days"]}'
      f'（隔离不填零）。')
    w()
    w('## 分年结构计数（仅结构完整，非已认证训练标签）')
    w()
    w('| 年 | 代码数 | 股票日 | valid_P | 61日历史齐全 | 后5日齐全 | 结构完整 | 独立P |')
    w('|---|---:|---:|---:|---:|---:|---:|---:|')
    for y, s in rep['years'].items():
        w(f"| {y} | {s['codes']} | {s['stock_days']:,} | {s['valid_P']:,} | "
          f"{s['complete_61d_history']:,} | {s['complete_future_5d']:,} | "
          f"{s['structurally_complete']:,} | {s['distinct_P']} |")
    w()
    w('## 分段时间窗（purge 后）')
    w()
    w('| 段 | 股票日 | 结构完整 | purge 后 | 删于边界/截断 |')
    w('|---|---:|---:|---:|---:|')
    for name, s in rep['segments'].items():
        w(f"| {name} | {s['stock_days']:,} | {s['structurally_complete']:,} | "
          f"{s['structural_after_purge']:,} | {s['purged_at_boundary_or_censored']:,} |")
    w()
    w('## 显式排除的前缀（不混入研究人口；其余前缀多为 ETF/基金/北交所 等）')
    w()
    w('| 前缀 | 行数 |')
    w('|---|---:|')
    for pref, n in sorted(rep['excluded_prefix_rows'].items(), key=lambda kv: -kv[1]):
        w(f"| {pref} | {n:,} |")
    w(f'\n（北交所 920 前缀 {rep["excluded_prefix_rows"].get("920", 0):,} 行已排除；'
      '其余为基金/ETF 类前缀，类别标签为指示性，绑定规则是前缀集合而非类别名。）')
    w()
    w('## 未关门声明（诚实边界）')
    w()
    w(f"- `price_scale_verified` = **{rep['price_scale_verified']}**：尺度/复权未验收；"
      '44 窗 225 点匹配仅为局部证据，不作全库 OHLC 准入。')
    w(f"- `corporate_action_coverage_verified` = **{rep['corporate_action_coverage_verified']}**："
      '公司行为覆盖未核销。')
    w(f"- `historical_membership_survivorship_verified` = **"
      f"{rep['historical_membership_survivorship_verified']}**：退市证券覆盖未核销；"
      '**本披露为可见范围，不声称无幸存者偏差全市场回测**。')
    w(f"- `market_index_source_bound` = **{rep['market_index_source_bound']}**："
      '沪深300 价格指数未绑定（数据内无 000300/399300）；市场组 5 字段保持 NULL，不零填充。')
    w()
    w('## 禁止事项（合同约束）')
    w()
    w('- 不得用今天股票名单回筛历史人口。')
    w('- 不得把缺失覆盖说明包装为无幸存者偏差全市场回测。')
    w('- 不得用个股中位数代理冒充沪深300（旧 market_factors）。')
    w('- 不得沿用 pre_close 当原始昨收。')
    w()
    w(f'_生成自 `{INV.name}`；原盘点JSON未改动，本文件为目录内新增披露。_')

    out = OUT / 'SECURITY_COVERAGE_DISCLOSURE.md'
    data='\n'.join(L).encode('utf-8')
    from prepare_historical_risk_v1 import bytes_used
    if bytes_used()+len(data)>128*1024**2:
        raise ValueError('REGISTERED_TOTAL_BUDGET')
    with out.open('xb') as f:
        f.write(data)
    print(f'written {out.name} ({out.stat().st_size:,} B)')


if __name__ == '__main__':
    main()
