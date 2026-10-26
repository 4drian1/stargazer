"""price_verification_batch_44win.py — 44 窗外部价格交叉核验批次（零网络准备；取数需单独授权）。

定位：复用**已验收的腾讯受控客户端** `independent_fetch_client.TencentDailyClient` 作为
"外部交叉核验源"（与 stockdb 不同口径，但**非强独立源**；不重新宣称强独立性）。
本批只做**价格一致性比对**，不认证公司行为完整性（覆盖核销另走 R1–R4）。

治理（沿用 independent_fetch_run.py 已验收条款，不重新发明）：
- 请求计数 FetchCounter（含冒烟/重试/辅助）；超限即停。
- 字节预算 ByteCounter（响应+落盘）；禁重定向；原始响应留证（.bin/.json/index.jsonl）。
- 冒烟 2 次零重试（本批股票各一）→ 失败即整批终止（REV6 治理）。
- 冻结容差 |diff| ≤ 0.01 元逐点（同口径 close / 逐日 low）。
- 尺度与覆盖**分别报告**；不覆盖旧数据；不扩人口；不生成标签；不训练。

零网络模式（本轮交付）：
- --dry-run：读冻结比较点，扫描既有原始响应能覆盖多少价格点，打印取数计划（按股票合并）。
- --selftest：合成数据验证离线比对器与 MAE≈−8% 边界敏感性逻辑（不触网）。
取数模式（需单独授权，本轮不执行）：
- --fetch：调客户端逐股票取区间日线，离线比对 225 唯一点，输出 MATCH/DIFF/MISS 与边界敏感表。
"""
from __future__ import annotations

import json
import hashlib
import sys
import bisect
import math
from collections import Counter
from decimal import Decimal
from datetime import datetime, timedelta, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / 'step9_preflight'))
from independent_fetch_client import (  # 已验收受控客户端
    TencentDailyClient, FetchCounter, ByteCounter, FetchBudgetExceeded,
    ByteBudgetExceeded,
)

BATCH = HERE / 'independent_fetch' / 'batch_d_mainpop_30'
POINTS = BATCH / 'price_comparison_points_44win.json'
MANIFEST = BATCH / 'price_verification_manifest_v2.json'
REQUEST_CAP = 36
BYTE_CAP = 5 * 1024 * 1024
REPORT_RESERVE = 256 * 1024
TOL = 0.01                      # 冻结容差：|diff| ≤ 0.01 元
BOUNDARY = -0.08                # MAE_5d 标签边界（约 −8%）
BSE_GAP = {'920729', '920808'}  # 北交所覆盖缺口：不请求


def valid_price(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value > 0


# ---- 复用已验收的字段序映射（腾讯 day 行序=[date,open,close,high,low]） ----
def tencent_rows_to_map(rows: list) -> dict:
    out = {}
    for r in rows:
        if not isinstance(r, list) or len(r) < 5:
            raise ValueError('REMOTE_ROW_SHAPE')
        datetime.strptime(str(r[0]), '%Y-%m-%d')
        d = int(str(r[0]).replace('-', ''))
        if any(isinstance(v, bool) for v in r[1:5]):
            raise ValueError('REMOTE_PRICE_INVALID')
        value = dict(open=float(r[1]), close=float(r[2]), high=float(r[3]), low=float(r[4]))
        if not all(valid_price(v) for v in value.values()) or not (
                value['low'] <= min(value['open'], value['close']) <=
                max(value['open'], value['close']) <= value['high']):
            raise ValueError('REMOTE_PRICE_INVALID')
        if d in out and out[d] != value:
            raise ValueError('REMOTE_DUPLICATE_CONFLICT')
        out[d] = value
    return out


# ---- 离线比对器 ----
def compare_point(local_val, remote_val):
    if local_val is None and remote_val is None:
        return 'MISS_BOTH', None
    if local_val is None:
        return 'MISS_LOCAL', None
    if remote_val is None:
        return 'MISS_REMOTE', None
    if any(isinstance(v, bool) or not valid_price(v) for v in (local_val, remote_val)):
        return 'INVALID_PRICE', None
    diff = Decimal(str(local_val)) - Decimal(str(remote_val))
    return ('MATCH' if abs(diff) <= Decimal(str(TOL)) else 'DIFF'), float(diff)


def mae_5d(local, code, p, win):
    rp = local.get((code, int(p)))
    if rp is None or not valid_price(rp['close']):
        return None
    r_p = rp['close']
    vals = []
    for wd in win:
        row = local.get((code, int(wd.replace('-', ''))))
        if row is None or not valid_price(row['low']):
            return None
        vals.append(row['low'] / r_p - 1.0)
    return min(vals)


def boundary_sensitivity(local, code, p, win):
    """MAE≈−8% 边界敏感性：±0.01 元价格误差能否翻转标签方向即单列，不静默判类。"""
    rp = local.get((code, int(p)))
    if rp is None or not valid_price(rp['close']):
        return None
    r_p = rp['close']
    m = mae_5d(local, code, p, win)
    if m is None:
        return None
    rp_d, tol = Decimal(str(r_p)), Decimal(str(TOL))
    low = min(Decimal(str(local[(code, int(w.replace('-', '')))]['low'])) for w in win)
    if rp_d <= tol or low <= tol:
        return {'status': 'PERTURBATION_NONPOSITIVE', 'sensitive': None}
    lower = (low - tol) / (rp_d + tol) - 1
    upper = (low + tol) / (rp_d - tol) - 1
    return dict(diagnostic_mae_5d=m, lower=float(lower), upper=float(upper),
                boundary=BOUNDARY, sensitive=bool(lower <= Decimal(str(BOUNDARY)) <= upper),
                note='未核销价格尺度/公司行为；仅误差敏感性诊断，不是正式MAE或风险标签')


# ---- 本地价格读取（来自 data_daily_parts，与标签契约同口径）----
def load_local_prices(points, files=None):
    import pyarrow.dataset as ds, glob
    if files is None:
        from data_source_provenance import DAILY_PARTS
        files = sorted(glob.glob(str(Path(DAILY_PARTS) / 'part_*.parquet')))
    needed = {(p['code'], p['date'], p['field']) for p in points}
    codes = sorted({p['code'] for p in points})
    dates = sorted({p['date'] for p in points})
    local = {}
    sources = {}
    for file in files:
        tbl = ds.dataset([file], format='parquet').to_table(
            columns=['code', 'date', 'close', 'low'],
            filter=ds.field('code').isin(codes) & ds.field('date').isin(dates))
        for row in tbl.to_pylist():
            code, date = str(row['code']).zfill(6), int(row['date'])
            for field in ('close', 'low'):
                key = (code, date, field)
                if key not in needed:
                    continue
                value = row[field]
                if key in sources and local[(code, date)][field] != value:
                    raise ValueError(f'LOCAL_DUPLICATE_CONFLICT:{key}:{sources[key]}:{Path(file).name}')
                if isinstance(value, bool) or not valid_price(value):
                    raise ValueError(f'LOCAL_PRICE_INVALID:{key}:{Path(file).name}')
                local.setdefault((code, date), {})[field] = value
                sources[key] = Path(file).name
    if any(p['field'] not in local.get((p['code'], p['date']), {}) for p in points):
        raise ValueError('LOCAL_REQUIRED_PRICE_MISSING')
    return local


# ---- 零网络：既有原始响应覆盖扫描 ----
def scan_existing_price_coverage():
    """扫描 batch 下 run_* 目录的 index.jsonl，统计既有的腾讯价格响应（gtimg/qq）。
    注：run_20261013T... 为 Sina 公司行为页，非价格响应 → 价格覆盖应为 0。"""
    n_price_resp = 0
    n_total = 0
    for idx in sorted(BATCH.glob('run_*/index.jsonl')):
        for line in idx.read_text(encoding='utf-8').splitlines():
            if not line.strip():
                continue
            n_total += 1
            try:
                rec = json.loads(line)
            except Exception:
                continue
            url = rec.get('url', '')
            if 'gtimg' in url or 'qq.com' in url:
                n_price_resp += 1
    return n_total, n_price_resp


def plan_from_points(points):
    """按股票合并日期范围（每股票一次区间请求）。"""
    by_stock = {}
    for p in points:
        s = by_stock.setdefault(p['code'], {'dates': set()})
        s['dates'].add(p['date'])
    plan = {}
    for code, s in by_stock.items():
        lo, hi = min(s['dates']), max(s['dates'])
        start = (datetime.strptime(str(lo), '%Y%m%d') - timedelta(days=7)).strftime('%Y%m%d')
        end = (datetime.strptime(str(hi), '%Y%m%d') + timedelta(days=2)).strftime('%Y%m%d')
        plan[code] = dict(start=start, end=end, n_points=len(s['dates']))
    return plan


def dry_run(manifest_path=MANIFEST):
    from price_verification_guarded_run import verify_bundle
    manifest, windows, points, local = verify_bundle(manifest_path, sys.modules[__name__])
    plan = plan_from_points(points)
    n_total, n_price = scan_existing_price_coverage()
    n_stocks = len(plan)
    req_estimate = n_stocks  # Two full-interval smoke responses are reused.
    cap = REQUEST_CAP
    print('==== 44 窗价格核验 取数计划（dry-run，零网络）====')
    print(f'窗口={len(windows)} 唯一点={len(points)} 引用={len(windows)*6} 股票={len(plan)}；冻结哈希复验PASS')
    print(f'既有响应扫描：总 {n_total} 条；其中腾讯价格响应 {n_price} 条 '
          f'（响应条数不是价格点覆盖；未经逐点核验不复用）')
    print(f'按股票合并={n_stocks}；2次冒烟区间结果复用；正常请求={req_estimate}；固定硬顶={cap}；零重试')
    print('北交所缺口（不请求）：', sorted(BSE_GAP & set(plan)))
    print(f'容差冻结 |diff|≤{TOL} 元；边界敏感性针对 MAE≈{BOUNDARY}（非 ±8%）')
    print('注：既有 run_20261013* 为 Sina 公司行为页，非价格响应，不可复用为价格点。')
    print('状态：ZERO_NETWORK_PREFLIGHT_PASS / INDEPENDENT_FETCH_AUTHORIZATION_REQUIRED；非标签/训练就绪')
    return dict(req_estimate=req_estimate, cap=cap, existing_price_response_count=n_price,
                price_points_verified_for_reuse=0, training_ready=False)


# ---- 零网络：合成自测 ----
def selftest():
    results = []
    def check(name, cond, detail=""):
        results.append((name, bool(cond), detail))

    # 本地价（含一个边界附近点）
    local = {
        ('000001', 20260105): dict(close=10.00, low=9.00),
        ('000001', 20260106): dict(close=10.10, low=9.20),
        ('000001', 20260107): dict(close=10.05, low=9.10),
        ('000001', 20260108): dict(close=10.00, low=8.50),   # 低 → 拉低 MAE
        ('000001', 20260109): dict(close=10.00, low=9.00),
        ('000001', 20260110): dict(close=10.00, low=9.00),
    }
    win = ['2026-01-06', '2026-01-07', '2026-01-08', '2026-01-09', '2026-01-10']
    # MATCH / DIFF / MISS 逻辑
    v, d = compare_point(9.20, 9.20)
    check('MATCH 判定', v == 'MATCH' and d == 0.0, f'{v},{d}')
    v, d = compare_point(9.20, 9.25)
    check('DIFF 判定(>0.01)', v == 'DIFF' and abs(d - (-0.05)) <= 1e-6, f'{v},{d}')
    v, _ = compare_point(9.20, None)
    check('MISS_REMOTE 判定', v == 'MISS_REMOTE')
    v, _ = compare_point(None, 9.20)
    check('MISS_LOCAL 判定', v == 'MISS_LOCAL')

    # 边界敏感性：构造 MAE_5d ≈ −0.08 附近（五日 low 全部=9.20 → min=9.20 → -0.08 恰边界）
    local2 = {k: dict(v) for k, v in local.items()}
    for wd in win:
        local2[('000001', int(wd.replace('-', '')))] = dict(close=10.00, low=9.20)
    bs = boundary_sensitivity(local2, '000001', 20260105, win)
    check('边界敏感被标记', bs is not None and bs['sensitive'] is True, str(bs))
    # 远离边界：五日 low 全部=9.90 → min=9.90 → -0.01，远离 −0.08
    local3 = {k: dict(v) for k, v in local.items()}
    for wd in win:
        local3[('000001', int(wd.replace('-', '')))] = dict(close=10.00, low=9.90)
    bs3 = boundary_sensitivity(local3, '000001', 20260105, win)
    check('远离边界不误报', bs3 is not None and bs3['sensitive'] is False, str(bs3))

    print('==== 44 窗价格核验 离线比对器自测 ====')
    npass = sum(1 for _, ok, _ in results if ok)
    for name, ok, detail in results:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}  {detail}")
    print(f"\n{npass}/{len(results)} PASS")
    return npass == len(results)


# ---- Fetch entry: signed artifact + full preflight, then isolated batch ----
def fetch_run(archive_dir, authorization=None, authorization_sha256=None, manifest_path=MANIFEST):
    from price_verification_guarded_run import execute
    return execute(archive_dir, authorization, authorization_sha256, manifest_path,
                   sys.modules[__name__])


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='44窗价格交叉核验；不构建标签、不训练')
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument('--dry-run', action='store_true')
    modes.add_argument('--selftest', action='store_true')
    modes.add_argument('--fetch', metavar='NEW_BATCH_DIRECTORY')
    parser.add_argument('--authorization')
    parser.add_argument('--authorization-sha256')
    parser.add_argument('--manifest', type=Path, default=MANIFEST)
    args = parser.parse_args()
    try:
        if args.selftest:
            raise SystemExit(0 if selftest() else 1)
        elif args.dry_run:
            dry_run(args.manifest)
        else:
            result = fetch_run(args.fetch,args.authorization,args.authorization_sha256,args.manifest)
            print(json.dumps(result,ensure_ascii=False,allow_nan=False))
            raise SystemExit(0 if result['stop_reason'] == 'completed' else 2)
    except (ValueError, OSError, KeyError) as exc:
        print(json.dumps({'status':'BLOCKED','reason':str(exc),'network_authorized':False},
                         ensure_ascii=False))
        raise SystemExit(2)
