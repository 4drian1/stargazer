"""independent_fetch_client.py — 腾讯日K独立取数受控客户端（请求单 §1/§5 执行件）。

设计约束（对应 bounded_independent_verification_request_order.md 修订条款）：
- **直连** web.ifzq.gtimg.cn fqkline 接口，**不使用 akshare 封装**（封装有隐藏起点查询
  且无超时，违反"一次调用≠一次请求"的计数纪律）。
- **计数**：每次 HTTP 请求（含重试、冒烟）都计入 hard cap，超限即停（FetchBudgetExceeded）。
- **限速**：默认 1 请求/秒；**超时**：每请求 urllib timeout（默认 10s），重试 ≤3 次指数退避。
- **响应校验（防静默退回复权）**：param 末位 fq 留空 → 只接受响应含 `day` 键且
  **不含** `qfqday`/`hfqday`；否则拒绝（RawResponseRejected），绝不静默用复权数据。
- 只实现请求单 §3 所需：单码一段日K（P-5..W_end 一次取齐）。

零网络测试见 test_independent_fetch_client.py（mock 传输层）。
"""
from __future__ import annotations

import json
import time
import urllib.parse
import urllib.request


class FetchBudgetExceeded(RuntimeError):
    pass


class RawResponseRejected(RuntimeError):
    pass


class FetchCounter:
    """每次 HTTP 请求计数；hard cap 含冒烟/重试/全部辅助请求。"""

    def __init__(self, hard_cap: int):
        self.hard_cap = hard_cap
        self.used = 0

    def consume(self, n: int = 1) -> None:
        if self.used + n > self.hard_cap:
            raise FetchBudgetExceeded(
                f'请求预算超限：将使用 {self.used + n} > hard_cap {self.hard_cap}；即停')
        self.used += n


def _validate_raw(payload: dict, symbol: str) -> list:
    data = payload.get('data') or {}
    node = data.get(symbol) or {}
    if 'qfqday' in node or 'hfqday' in node:
        raise RawResponseRejected(
            f'响应含复权键（qfqday/hfqday），拒绝：{symbol}（禁止静默退回复权）')
    raw = node.get('day')
    if not isinstance(raw, list) or not raw:
        raise RawResponseRejected(f'响应缺原始 day 序列：{symbol}')
    return raw


class TencentDailyClient:
    """受控腾讯日K客户端。transport 可注入以便零网络测试。"""

    BASE = 'https://web.ifzq.gtimg.cn/appstock/app/fqkline/get'

    def __init__(self, counter: FetchCounter, *, rate_limit_s: float = 1.0,
                 timeout_s: float = 10.0, max_retries: int = 3,
                 transport=None, sleep=time.sleep):
        self.counter = counter
        self.rate_limit_s = rate_limit_s
        self.timeout_s = timeout_s
        self.max_retries = max_retries
        self._transport = transport or self._default_transport
        self._sleep = sleep
        self._last_ts = 0.0

    def _default_transport(self, url: str, timeout: float) -> bytes:
        req = urllib.request.Request(url, headers={'Accept': 'application/json'})
        return urllib.request.build_opener(
            urllib.request.ProxyHandler({})).open(req, timeout=timeout).read()

    def fetch_raw_daily(self, code6: str, start: str, end: str) -> list:
        """单码一段日K（原始价）。一次本函数调用 = 至多 (1+max_retries) 次计数请求。"""
        prefix = 'sh' if code6.startswith('6') else 'sz'
        symbol = f'{prefix}{code6}'
        param = f'{symbol},day,{start},{end},640,'  # 末位留空 = 原始价
        url = self.BASE + '?' + urllib.parse.urlencode({'param': param})
        attempt = 0
        while True:
            self.counter.consume(1)
            self._throttle()
            try:
                raw = self._transport(url, self.timeout_s)
                return _validate_raw(json.loads(raw.decode('utf-8')), symbol)
            except (RawResponseRejected, FetchBudgetExceeded):
                raise
            except Exception:
                attempt += 1
                if attempt > self.max_retries:
                    raise
                self._sleep(2 ** attempt)

    def _throttle(self) -> None:
        wait = self.rate_limit_s - (time.time() - self._last_ts)
        if wait > 0:
            self._sleep(wait)
        self._last_ts = time.time()


def case_windows(cases: list) -> list:
    """把案例展开为 (code6, start, end)：取 P-5 自然日 .. W_end（接口按区间返回，一次取齐）。"""
    from datetime import datetime, timedelta
    out = []
    for c in cases:
        p = datetime.strptime(str(c['P']), '%Y%m%d')
        w = datetime.strptime(str(c['W_end']), '%Y%m%d')
        out.append((str(c['code']).zfill(6),
                    (p - timedelta(days=7)).strftime('%Y%m%d'),
                    w.strftime('%Y%m%d')))
    return out
