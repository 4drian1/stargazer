"""全球行情快照收集器：显式调用，免费源（腾讯行情），落隔离快照供 GET 零网络读取。

品种覆盖（第一批）：美股指数（道琼/纳指/标普/费半/中概）、COMEX黄金/白银、WTI/布伦特原油、
日经225、恒生指数。中国/美国10年期国债收益率暂无已核验免费源：如实标 MISSING，
不拼造、不用近似值冒充。缺失/单项失败只影响该项，其余正常落盘。
"""
from datetime import datetime, timezone, timedelta
import json
from pathlib import Path

ZONE = timezone(timedelta(hours=8))

# variety → (代码, 市场, 时区)；hf_=腾讯国际期货（字段版式不同），us/hk=腾讯美股/港股
SOURCES = {
    "道琼斯工业平均指数": ("usDJI", "美国", "America/New_York"),
    "纳斯达克综合指数": ("usIXIC", "美国", "America/New_York"),
    "标普500": ("usINX", "美国", "America/New_York"),
    "COMEX黄金": ("hf_GC", "美国", "America/New_York"),
    "COMEX白银": ("hf_SI", "美国", "America/New_York"),
    "WTI原油": ("hf_CL", "美国", "America/New_York"),
    "布伦特原油": ("hf_OIL", "欧洲", "Europe/London"),
    "恒生指数": ("hkHSI", "香港", "Asia/Hong_Kong"),
}
# 免费源暂未覆盖：如实 MISSING，不用其他品种冒充
UNCOVERED = {}


def _em_index(secid):
    """东财国际指数：返回 (name, price, prev, quote_iso)。偶发断连重试一次。"""
    import json as _json
    import urllib.request as _ur
    url = (f'https://push2.eastmoney.com/api/qt/stock/get?secid={secid}'
           '&fields=f43,f60,f58,f86&fltt=2')
    last = None
    for _ in range(2):
        try:
            req = _ur.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            data = _json.loads(_ur.urlopen(req, timeout=8).read()).get('data') or {}
            price, prev = float(data['f43']), float(data['f60'])
            if not (price > 0 and prev > 0):
                return None
            from datetime import datetime as _dt
            stamp = _dt.fromtimestamp(int(data['f86']), ZONE).isoformat(timespec='seconds')
            return data.get('f58'), price, prev, stamp
        except Exception as exc:
            last = exc
            import time as _time
            _time.sleep(0.6)
    raise last


def _bond_rates():
    """akshare 中美国债收益率：返回 {name: (value, date)}；失败返回 {}（如实 MISSING）。"""
    try:
        import akshare as ak
        df = ak.bond_zh_us_rate()
        out = {}
        for col, label in (('中国国债收益率10年', '中国10年期国债收益率'),
                           ('美国国债收益率10年', '美国10年期国债收益率')):
            series = df[['日期', col]].dropna()
            if series.empty:
                continue
            row = series.iloc[-1]
            out[label] = (float(row[col]), str(row['日期']))
        return out
    except Exception:
        return {}
CATEGORIES = {
    "道琼斯工业平均指数": ("美股指数", "点"), "纳斯达克综合指数": ("美股指数", "点"),
    "标普500": ("美股指数", "点"), "COMEX黄金": ("贵金属", "美元/盎司"),
    "COMEX白银": ("贵金属", "美元/盎司"), "WTI原油": ("能源", "美元/桶"),
    "布伦特原油": ("能源", "美元/桶"), "恒生指数": ("股指", "点"),
}


def _parse_us(line):
    body = line.split("=", 1)[1].strip('"').split("~")
    if len(body) < 45 or not body[1] or not body[3]:
        return None
    try:
        stamp = ''.join(ch for ch in str(body[30]) if ch.isdigit())
        return dict(price=float(body[3]), prev=float(body[4]), pct=float(body[32]), time=stamp[:14])
    except (ValueError, IndexError):
        return None


def _parse_hf(line):
    body = line.split("=", 1)[1].strip('"').split(",")
    if len(body) < 14 or not body[0] or not body[2] or not body[13]:
        return None
    try:
        return dict(price=float(body[0]), prev=float(body[2]), pct=float(body[1]),
                    time=(body[12].replace('-', '') + body[6].replace(':', '')))
    except (ValueError, IndexError):
        return None


def collect(snapshot_path=None, fetch=None):
    """抓取并写快照。返回 (ok, items, errors)。单项失败进 errors，不拖垮整批。"""
    from tools import market as _mkt
    fetch_fn = fetch or _mkt._get
    from ui_backend_config import global_overview_snapshot_path
    path = Path(snapshot_path) if snapshot_path else global_overview_snapshot_path()
    q = ",".join(code for code, _, _ in SOURCES.values())
    now = datetime.now(ZONE).isoformat(timespec="seconds")
    items, errors = [], []
    txt = fetch_fn(f"https://qt.gtimg.cn/q={q}", tries=2, timeout=8, encoding="gbk")
    if not txt:
        return False, [], ["COLLECT_TRANSPORT_FAILED:腾讯行情未返回内容"]
    rows = {}
    for line in txt.strip().split(";"):
        if "=" not in line:
            continue
        key = line.split("=", 1)[0].strip()
        key = key[2:] if key.startswith('v_') else key
        parsed = _parse_hf(line) if key.startswith('hf_') else _parse_us(line)
        if parsed is not None:
            rows[key] = parsed
    for variety, (code, market, tz) in SOURCES.items():
        row = rows.get(code)
        category, unit = CATEGORIES[variety]
        if row is None or not all(v > 0 for v in (row["price"], row["prev"])):
            errors.append(f"{variety}:报价缺失或无效")
            items.append(dict(variety=variety, category=category, unit=unit, market=market,
                              timezone=tz, value=None, change=None, change_unit=None,
                              basis=None, quote_time=None, quote_flag=None,
                              scrape_time=now, sample=False,
                              reason="单项报价缺失或无效（不拼造）"))
            continue
        quote_time = row["time"]
        try:
            from datetime import datetime as _dt
            quote_iso = _dt.strptime(quote_time, "%Y%m%d%H%M%S").replace(tzinfo=ZONE).isoformat(timespec="seconds")
        except (TypeError, ValueError):
            quote_iso = None
        items.append(dict(variety=variety, category=category, unit=unit, market=market,
                          timezone=tz, value=row["price"],
                          change=round(row["pct"], 2), change_unit="%",
                          basis="较前一交易日收盘", quote_time=quote_iso,
                          quote_flag="TENCENT_FREE_QUOTE",
                          scrape_time=now, sample=False))
    for variety, (category, unit, market, tz, reason) in UNCOVERED.items():
        items.append(dict(variety=variety, category=category, unit=unit, market=market,
                          timezone=tz, value=None, change=None, change_unit=None,
                          basis=None, quote_time=None, quote_flag=None,
                          scrape_time=now, sample=False, reason=reason))
    # 日经225 / 欧洲斯托克50：东财国际指数
    for variety, secid, market, tz in (
            ("日经225", "100.N225", "日本", "Asia/Tokyo"),
            ("欧洲斯托克50", "100.SX5E", "欧洲", "Europe/Paris")):
        try:
            result = _em_index(secid)
        except Exception:
            result = None
        if result is None:
            errors.append(f"{variety}:东财报价缺失")
            items.append(dict(variety=variety, category="股指", unit="点", market=market,
                              timezone=tz, value=None, change=None, change_unit=None,
                              basis=None, quote_time=None, quote_flag=None,
                              scrape_time=now, sample=False,
                              reason="东财国际指数报价缺失或无效（不拼造）"))
            continue
        _, price, prev, quote_iso = result
        items.append(dict(variety=variety, category="股指", unit="点", market=market,
                          timezone=tz, value=price,
                          change=round((price / prev - 1) * 100, 2), change_unit="%",
                          basis="较前一交易日收盘", quote_time=quote_iso,
                          quote_flag="EASTMONEY_FREE_QUOTE", scrape_time=now, sample=False))
    # 中美 10 年期国债收益率：akshare（中国国债信息网），失败如实 MISSING
    rates = _bond_rates()
    for variety in ("中国10年期国债收益率", "美国10年期国债收益率"):
        value = rates.get(variety)
        if value is None:
            errors.append(f"{variety}:akshare 未返回")
            items.append(dict(variety=variety, category="国债", unit="%", market=("中国" if variety.startswith("中国") else "美国"),
                              timezone="Asia/Shanghai" if variety.startswith("中国") else "America/New_York",
                              value=None, change=None, change_unit=None,
                              basis=None, quote_time=None, quote_flag=None,
                              scrape_time=now, sample=False,
                              reason="akshare 中美国债收益率未返回（不拼造）"))
            continue
        rate, day = value
        items.append(dict(variety=variety, category="国债", unit="%", market=("中国" if variety.startswith("中国") else "美国"),
                          timezone="Asia/Shanghai" if variety.startswith("中国") else "America/New_York",
                          value=rate, change=None, change_unit=None,
                          basis=f"中国国债信息网最新收益率（{day}）",
                          quote_time=day + "T00:00:00+08:00",
                          quote_flag="AKSHARE_BOND_ZH_US_RATE", scrape_time=now, sample=False))
    snapshot = dict(contract="STARGAZER_GLOBAL_OVERVIEW_SNAPSHOT_V1",
                    generated_at=now, items=items,
                    note="免费源（腾讯行情）；中国/美国10Y国债暂无已核验免费源，保持 MISSING；"
                         "快照仅供展示，决策以正式风险链为准")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(snapshot, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)
    return True, items, errors


def collect_subprocess(snapshot_path=None, timeout_seconds=90):
    """以独立子进程执行采集（akshare 导入/抓取较慢，绝不挂住 server 线程）。
    返回 (ok, note)；超时/失败保持旧快照并返回明确原因。"""
    import subprocess as _sp
    from pathlib import Path as _Path
    import paths as _paths
    # 与 worker 同口径：项目根解析（frozen: dist → 上上级目录）
    app = _Path(_paths.APP_DIR).resolve()
    root = app.parent.parent if app.name == 'dist' else app.parent
    python = root / '_dev_venv' / 'Scripts' / 'python.exe'
    if not python.is_file():
        python = root / 'rule_venv' / 'Scripts' / 'python.exe'
    if not python.is_file():
        import sys
        python = _Path(sys.executable)
    env = dict(__import__('os').environ)
    env['STARGAZER_GLOBAL_OVERVIEW_SNAPSHOT'] = str(
        snapshot_path or __import__('ui_backend_config').global_overview_snapshot_path())
    try:
        proc = _sp.run(
            [str(python), '-B', '-X', 'utf8', '-c',
             'import sys;sys.path.insert(0,%r);from global_market_collect_v1 import collect;ok,items,errors=collect();print(ok,len(items),len(errors))' % str(root / 'stargazer_desktop')],
            capture_output=True, text=True, encoding='utf-8', errors='replace',
            timeout=timeout_seconds, creationflags=getattr(_sp, 'CREATE_NO_WINDOW', 0), env=env)
        if proc.returncode != 0:
            return False, 'COLLECT_WORKER_EXIT:%s:%s' % (proc.returncode, (proc.stderr or '')[-200:])
        last = (proc.stdout or '').strip().splitlines()[-1] if proc.stdout else '?'
        return True, 'COLLECT_WORKER_DONE:%s' % last
    except _sp.TimeoutExpired:
        return False, 'COLLECT_WORKER_TIMEOUT'
