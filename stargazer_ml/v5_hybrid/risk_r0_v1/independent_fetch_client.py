"""independent_fetch_client.py — 腾讯日K独立取数受控客户端（请求单 REV3 执行件）。

设计约束（bounded_independent_verification_request_order.md 修订条款）：
- **直连** web.ifzq.gtimg.cn fqkline，不用 akshare 封装（隐藏起点查询+无超时）。
- **请求计数**：每次 HTTP（含重试/冒烟/辅助）入 FetchCounter hard_cap=25，超限即停。
- **字节预算真实生效**：默认传输按 read_cap 限量读取（绝不多读）；收到响应先按实际
  字节数计入 ByteCounter（**含重试与拒收响应**），累计触线即 ByteBudgetExceeded。
- **禁止隐藏重定向**：默认传输挂 _NoRedirect，遇 3xx 直接 RedirectRefused 停止，
  重定向产生的额外请求永远不进计数器（因为根本不发生）。
- **原始响应留证**：archive_dir 提供时，原始字节+请求参数+实际抓取时刻+SHA256+
  outcome（ok/rejected/error）全部落盘；**异常响应同样留证**（连接失败留尝试记录）。
  落盘字节另计 DiskWriteCounter（原始+sidecar+索引+报告+停机留因）。
- 冒烟 smoke=True → 该次调用零重试。

零网络测试见 test_independent_fetch_client.py（mock 传输层）。
"""
from __future__ import annotations

import hashlib
import json
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path


class FetchBudgetExceeded(RuntimeError):
    pass


class ByteBudgetExceeded(RuntimeError):
    pass


class RawResponseRejected(RuntimeError):
    pass


class RedirectRefused(RuntimeError):
    """遇 3xx 重定向即停，不自动跟随（隐藏请求不入计数器）。"""


class EvidenceCollision(RuntimeError):
    """证据文件身份碰撞：拒绝覆盖旧证据。"""


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


class ByteCounter:
    """累计响应/落盘字节；触线即停。"""

    def __init__(self, hard_cap_bytes: int):
        self.hard_cap_bytes = hard_cap_bytes
        self.used = 0

    def consume(self, n: int) -> None:
        if self.used + n > self.hard_cap_bytes:
            raise ByteBudgetExceeded(
                f'字节预算超限：累计 {self.used + n} > {self.hard_cap_bytes} B；即停')
        self.used += n

    @property
    def remaining(self) -> int:
        return self.hard_cap_bytes - self.used


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise RedirectRefused(f'重定向拒绝（HTTP {code} -> {newurl}）：不自动跟随')


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


def _default_transport(url: str, timeout: float, read_cap: int) -> bytes:
    """限量读取（最多 read_cap 字节，多读即抛 ByteBudgetExceeded）+ 禁重定向。"""
    opener = urllib.request.build_opener(
        urllib.request.ProxyHandler({}), _NoRedirect())
    req = urllib.request.Request(url, headers={'Accept': 'application/json'})
    with opener.open(req, timeout=timeout) as resp:
        data = resp.read(read_cap + 1)
    if len(data) > read_cap:
        raise ByteBudgetExceeded(f'响应超过剩余字节预算（>{read_cap} B 未读完即弃）')
    return data


class TencentDailyClient:
    BASE = 'https://web.ifzq.gtimg.cn/appstock/app/fqkline/get'

    def __init__(self, counter: FetchCounter, byte_counter: ByteCounter | None = None,
                 *, rate_limit_s: float = 1.0, timeout_s: float = 10.0,
                 max_retries: int = 3, transport=None, sleep=time.sleep,
                 disk_counter: ByteCounter | None = None):
        self.counter = counter
        self.bytes = byte_counter or ByteCounter(5 * 1024 * 1024)  # 5MB 响应硬限
        self.rate_limit_s = rate_limit_s
        self.timeout_s = timeout_s
        self.max_retries = max_retries
        self._transport = transport or _default_transport
        self._sleep = sleep
        self._last_ts = 0.0
        self.disk = disk_counter  # 提供时才落盘留证
        self._seq = 0
        self._run_dir = None      # 每次进程一个独占运行目录，重启不覆盖旧证据

    def fetch_raw_daily(self, code6: str, start: str, end: str, *,
                        archive_dir=None, smoke: bool = False) -> list:
        prefix = 'sh' if code6.startswith('6') else 'sz'
        symbol = f'{prefix}{code6}'
        param = f'{symbol},day,{start},{end},640,'  # 末位留空 = 原始价
        url = self.BASE + '?' + urllib.parse.urlencode({'param': param})
        attempt_cap = 0 if smoke else self.max_retries
        attempt = 0
        while True:
            self.counter.consume(1)
            self._throttle()
            fetched_at = datetime.now(timezone.utc).isoformat(timespec='seconds')
            try:
                read_cap = max(self.bytes.remaining, 0)
                raw = self._transport(url, self.timeout_s, read_cap)
                self.bytes.consume(len(raw))          # 含将被拒收的响应
                rows = _validate_raw(json.loads(raw.decode('utf-8')), symbol)
                self._archive(archive_dir, symbol, param, url, fetched_at, raw, 'ok')
                return rows
            except (RawResponseRejected,) as exc:
                # 拒收响应同样留证
                self._archive(archive_dir, symbol, param, url, fetched_at,
                              locals().get('raw', b''), f'rejected:{type(exc).__name__}')
                raise
            except (FetchBudgetExceeded, ByteBudgetExceeded,
                    RedirectRefused):
                raise                              # 预算/重定向：即停不重试
            except Exception as exc:
                self._archive(archive_dir, symbol, param, url, fetched_at, b'',
                              f'error:{type(exc).__name__}')
                attempt += 1
                if attempt > attempt_cap:
                    raise
                self._sleep(2 ** attempt)

    def _run_archive_dir(self, archive_dir):
        """每次运行一个独占目录（UTC 时间戳+pid，exist_ok=False），重启/并行互不覆盖。"""
        if self._run_dir is None:
            import os
            ts = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%f')
            d = Path(archive_dir) / f'run_{ts}_{os.getpid()}'
            d.mkdir(parents=True, exist_ok=False)
            self._run_dir = d
        return self._run_dir

    def _archive(self, archive_dir, symbol, param, url, fetched_at, raw: bytes,
                 outcome: str) -> None:
        if archive_dir is None:
            return
        self._seq += 1
        sha = hashlib.sha256(raw).hexdigest()
        base = f'{self._seq:04d}_{symbol}'
        meta_obj = dict(seq=self._seq, symbol=symbol, param=param,
                        url=url, fetched_at=fetched_at, sha256=sha,
                        bytes=len(raw), outcome=outcome)
        meta = (json.dumps(meta_obj, ensure_ascii=False, indent=1)
                + chr(10)).encode('utf-8')
        index_line = (json.dumps(meta_obj, ensure_ascii=False) + chr(10)).encode('utf-8')
        if self.disk is not None:
            # 先算后写：按实际增量预检通过后才创建目录/落盘（无估算、无预写）
            self.disk.consume(len(raw) + len(meta) + len(index_line))
        d = self._run_archive_dir(archive_dir)
        meta_obj['run_dir'] = d.name
        blob = d / f'{base}.bin'
        try:
            with blob.open('xb') as f:                # 独占创建，碰撞即拒
                f.write(raw)
            with (d / f'{base}.json').open('xb') as f:
                f.write(meta)
            with (d / 'index.jsonl').open('ab') as f:
                f.write(index_line)
        except FileExistsError as exc:
            raise EvidenceCollision(f'证据文件碰撞，拒绝覆盖旧证据：{exc.filename}')


    def _throttle(self) -> None:
        wait = self.rate_limit_s - (time.time() - self._last_ts)
        if wait > 0:
            self._sleep(wait)
        self._last_ts = time.time()


def case_windows(cases: list) -> list:
    from datetime import datetime, timedelta
    out = []
    for c in cases:
        p = datetime.strptime(str(c['P']), '%Y%m%d')
        w = datetime.strptime(str(c['W_end']), '%Y%m%d')
        out.append((str(c['code']).zfill(6),
                    (p - timedelta(days=7)).strftime('%Y%m%d'),
                    w.strftime('%Y%m%d')))
    return out
