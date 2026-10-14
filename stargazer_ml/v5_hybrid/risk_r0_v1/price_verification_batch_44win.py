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
import sys
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from independent_fetch_client import (  # 已验收受控客户端
    TencentDailyClient, FetchCounter, ByteCounter, FetchBudgetExceeded,
    ByteBudgetExceeded,
)
from label_readiness_inventory import valid_price  # 复用有限/正数校验

BATCH = HERE / 'independent_fetch' / 'batch_d_mainpop_30'
POINTS = BATCH / 'price_comparison_points_44win.json'
TOL = 0.01                      # 冻结容差：|diff| ≤ 0.01 元
BOUNDARY = -0.08                # MAE_5d 标签边界（约 −8%）
BSE_GAP = {'920729', '920808'}  # 北交所覆盖缺口：不请求


# ---- 复用已验收的字段序映射（腾讯 day 行序=[date,open,close,high,low]） ----
def tencent_rows_to_map(rows: list) -> dict:
    out = {}
    for r in rows:
        d = int(str(r[0]).replace('-', ''))
        out[d] = dict(open=float(r[1]), close=float(r[2]),
                      high=float(r[3]), low=float(r[4]))
    return out


# ---- 离线比对器 ----
def compare_point(local_val, remote_val):
    if local_val is None and remote_val is None:
        return 'MISS_BOTH', None
    if local_val is None:
        return 'MISS_LOCAL', None
    if remote_val is None:
        return 'MISS_REMOTE', None
    diff = round(local_val - remote_val, 4)
    return ('MATCH' if abs(diff) <= TOL else 'DIFF'), diff


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
    perturb = 0.01 / r_p
    m = mae_5d(local, code, p, win)
    if m is None:
        return None
    cross = (m - perturb <= BOUNDARY) and (m + perturb >= BOUNDARY)
    return dict(mae_5d=round(m, 6), perturb=round(perturb, 6),
                boundary=BOUNDARY, sensitive=bool(cross))


# ---- 本地价格读取（来自 data_daily_parts，与标签契约同口径）----
def load_local_prices(points):
    from data_source_provenance import DAILY_PARTS
    import pyarrow.dataset as ds, glob
    codes = sorted({p['code'] for p in points})
    files = sorted(glob.glob(str(Path(DAILY_PARTS) / 'part_*.parquet')))
    tbl = ds.dataset(files, format='parquet').to_table(
        columns=['code', 'date', 'close', 'low'],
        filter=ds.field('code').isin(codes))
    local = {}
    for code, date, cl, lo in zip(tbl['code'].to_pylist(), tbl['date'].to_pylist(),
                                  tbl['close'].to_pylist(), tbl['low'].to_pylist()):
        key = (str(code).zfill(6), int(date))
        if valid_price(cl) and valid_price(lo):
            local[key] = dict(close=cl, low=lo)
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


def dry_run():
    data = json.loads(POINTS.read_text(encoding='utf-8'))
    points = data['points']
    plan = plan_from_points(points)
    n_total, n_price = scan_existing_price_coverage()
    n_stocks = len(plan)
    req_estimate = n_stocks + 2  # +2 冒烟
    cap = req_estimate + 4        # 硬顶 = 估算 + 余量
    print('==== 44 窗价格核验 取数计划（dry-run，零网络）====')
    print(f'唯一点={data["unique_points_total"]} 引用={data["references_total"]} '
          f'股票={data["stocks_total"]}')
    print(f'既有响应扫描：总 {n_total} 条；其中腾讯价格响应 {n_price} 条 '
          f'→ 价格点可复用覆盖 = {n_price}')
    print(f'按股票合并请求数 = {n_stocks}；+ 2 冒烟 = {req_estimate}；'
          f'建议请求硬顶 = {cap}')
    print('北交所缺口（不请求）：', sorted(BSE_GAP & set(plan)))
    print(f'容差冻结 |diff|≤{TOL} 元；边界敏感性针对 MAE≈{BOUNDARY}（非 ±8%）')
    print('注：既有 run_20261013* 为 Sina 公司行为页，非价格响应，不可复用为价格点。')
    return dict(req_estimate=req_estimate, cap=cap, price_covered=n_price)


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


# ---- 取数模式（需单独授权；本轮不执行）----
def fetch_run(archive_dir: str):  # pragma: no cover - 需授权后启用
    data = json.loads(POINTS.read_text(encoding='utf-8'))
    points = data['points']
    plan = plan_from_points(points)
    local = load_local_prices(points)
    req_cap = len(plan) + 2 + 4
    req = FetchCounter(req_cap)
    nbytes = ByteCounter(5 * 1024 * 1024)
    disk = ByteCounter(5 * 1024 * 1024)
    disk.consume(256 * 1024)  # 报告预留
    client = TencentDailyClient(req, nbytes, rate_limit_s=1.0, timeout_s=10.0,
                                max_retries=3, disk_counter=disk)
    adir = Path(archive_dir)
    # 冒烟 2 次零重试（本批股票各一）
    smoke = list(plan.items())[:2]
    for code, rng in smoke:
        try:
            rows = client.fetch_raw_daily(code, rng['start'], rng['end'],
                                          archive_dir=str(adir), smoke=True)
            _ = tencent_rows_to_map(rows)
        except Exception as exc:
            raise SystemExit(f'冒烟失败即终止：{code} {exc}')
    # 逐股票取区间日线
    remote_maps = {}
    for code, rng in plan.items():
        if code in BSE_GAP:
            continue
        rows = client.fetch_raw_daily(code, rng['start'], rng['end'],
                                      archive_dir=str(adir))
        remote_maps[code] = tencent_rows_to_map(rows)
    # 离线比对 225 唯一点
    out = []
    for p in points:
        code, date, field = p['code'], p['date'], p['field']
        loc = local.get((code, date))
        rem = remote_maps.get(code, {}).get(date)
        lv = loc[field] if loc else None
        rv = rem[field] if rem else None
        verdict, diff = compare_point(lv, rv)
        rec = dict(code=code, date=date, field=field, local=lv, remote=rv,
                   diff=diff, verdict=verdict)
        out.append(rec)
    # 边界敏感性逐窗单列
    # (按 (code,P) 聚合窗口后调用 boundary_sensitivity——此处仅逐点，窗口级另由 map 提供)
    (BATCH / 'price_match_table_44win.json').write_text(
        json.dumps(out, ensure_ascii=False, indent=1), encoding='utf-8')
    cnt = Counter(r['verdict'] for r in out)
    print('价格比对（225 唯一点）：', dict(cnt))
    print(f'请求 {req.used}/{req.hard_cap} 响应 {nbytes.used} B 落盘 {disk.used} B')
    return out


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == '--dry-run':
        dry_run()
    elif len(sys.argv) > 1 and sys.argv[1] == '--selftest':
        raise SystemExit(0 if selftest() else 1)
    elif len(sys.argv) > 1 and sys.argv[1] == '--fetch':
        if len(sys.argv) < 3:
            raise SystemExit('--fetch 需要 archive_dir 参数（须先获授权）')
        fetch_run(sys.argv[2])
    else:
        print('用法: --dry-run | --selftest | --fetch <archive_dir>')
