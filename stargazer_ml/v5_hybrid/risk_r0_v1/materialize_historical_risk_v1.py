"""Materialize admitted label/feature matrix (roadmap Blocker 4) — gated + size-estimating.

Zero-network, no training, no overwrite of the inventory. Refuses unbound source files.
Prints a compressed/streaming size estimate FIRST so we
never silently switch directories to dodge the budget (roadmap: "先给压缩/流式大小估算").

The writer accepts bound per-code source evidence and a separately bound CSI300 input.
It serializes small Parquet chunks and checks their actual byte cost before writing.
Dataset preparation does not require FIT authorization. Large full-matrix storage
still requires an adequate registered budget; streaming alone does not reduce it.
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / 'historical_stock_day_preparation_v1'

# Per-row admitted schema (after evidence admission):
#   code:6, P:4, label(int8):1, 11 stock float64:88, 5 market float64:40, small flags.
BYTES_PER_ROW_UNCOMPRESSED = 6 + 4 + 1 + 11 * 8 + 5 * 8   # ~139 B
COMPRESSION_FACTOR_LO, COMPRESSION_FACTOR_HI = 3.0, 5.0
NEW_OUTPUT_CAP = 2 * 1024 ** 2
from research_budget_v2 import TOTAL_CAP
ROOT_BUDGET = TOTAL_CAP


def estimate_size(report):
    rows = {name: s['structural_after_purge'] for name, s in report['segments'].items()}
    total = sum(rows.values())
    est = {}
    for name, n in rows.items():
        unc = n * BYTES_PER_ROW_UNCOMPRESSED
        est[name] = dict(rows=n, uncompressed_bytes=unc,
                         compressed_bytes_est=[int(unc / COMPRESSION_FACTOR_HI),
                                              int(unc / COMPRESSION_FACTOR_LO)])
    tot_unc = total * BYTES_PER_ROW_UNCOMPRESSED
    return dict(
        note='ROW-COUNT UPPER BOUND ONLY: all structurally complete rows are assumed admitted. '
             'Byte range is a heuristic 3-5x compression guess; excludes provenance/status '
             'columns and per-chunk overhead, not a guaranteed storage bound.',
        total_rows=total,
        total_uncompressed=tot_unc,
        total_compressed_est=[int(tot_unc / COMPRESSION_FACTOR_HI),
                              int(tot_unc / COMPRESSION_FACTOR_LO)],
        segments=est,
        new_output_cap=NEW_OUTPUT_CAP,
        root_budget=ROOT_BUDGET,
        fits_new_output_cap=(tot_unc / COMPRESSION_FACTOR_HI) <= NEW_OUTPUT_CAP,
        fits_root_budget=(tot_unc / COMPRESSION_FACTOR_HI) <= ROOT_BUDGET)


def run(force=False, source_binding=None, index_path=None, output=None):
    rep = json.loads((OUT / 'population_inventory.json').read_bytes())
    est = estimate_size(rep)
    print('== 物化大小估算（不写入；证据未齐前不物化）==')
    print(json.dumps(est, ensure_ascii=False, indent=1))
    if not est['fits_root_budget']:
        print(f'结论：压缩后约 {est["total_compressed_est"][0]/1024**2:.0f}–'
              f'{est["total_compressed_est"][1]/1024**2:.0f} MiB，超过研究根预算 '
              f'{ROOT_BUDGET/1024**2:.0f} MiB → 需要流式/外部存储方案，'
              '不得换目录暗中写入。')
    if source_binding is None or index_path is None or output is None:
        print('拒绝物化：须提供 --source-binding、--index 和 --output；'
              '旧盘点布尔值和 --force 均不能替代逐码文件/尺度/公司行为证据。')
        return False
    from historical_stream_v1 import bound_rows,write_stream
    from prepare_historical_risk_v1 import bytes_used
    target=Path(output).resolve()
    if target.parent!=HERE.resolve():
        raise ValueError('OUTPUT_MUST_BE_REGISTERED_RISK_ROOT_CHILD')
    # Validate every source binding before creating an output directory.
    rows=bound_rows(source_binding,index_path)
    first=next(rows,None)
    if first is None: raise ValueError('NO_RESEARCH_STOCK_DAYS')
    from itertools import chain
    result=write_stream(chain([first],rows),target,bytes_used,
                        total_cap=ROOT_BUDGET,new_cap=NEW_OUTPUT_CAP)
    print(json.dumps(result,ensure_ascii=False))
    return result['status']=='completed'


if __name__ == '__main__':
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument('--force',action='store_true',help='legacy flag; never bypasses evidence')
    parser.add_argument('--source-binding')
    parser.add_argument('--index')
    parser.add_argument('--output')
    args=parser.parse_args()
    try:
        ok=run(force=args.force,source_binding=args.source_binding,index_path=args.index,
               output=args.output)
        sys.exit(0 if ok else 2)
    except Exception as exc:
        print(json.dumps(dict(materialized=False,error=str(exc),fit_performed=False),ensure_ascii=False))
        sys.exit(2)
