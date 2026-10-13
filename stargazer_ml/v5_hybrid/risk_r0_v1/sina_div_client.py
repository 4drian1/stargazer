"""sina_div_client.py — 新浪"分红配股"页受控适配器（b 项执行件；不调用 akshare 封装）。

页面：vip.stock.finance.sina.com.cn/corp/go.php/vISSUE_ShareBonus/stockid/{code}.phtml
**单页同时含"分红"与"配股"两表**（用户更正①：一次请求覆盖两表，避免重复请求）。
纪律同 REV4：共享计数/限量读/禁重定向/预算异常绝不重试/原始留证/独占 run 目录。
"""
from __future__ import annotations

import hashlib
import json
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

from independent_fetch_client import (ByteBudgetExceeded, ByteCounter,
                                      FetchBudgetExceeded, FetchCounter,
                                      _NoRedirect, EvidenceCollision,
                                      RedirectRefused)

PAGE_URL = ('https://vip.stock.finance.sina.com.cn/corp/go.php/'
            'vISSUE_ShareBonus/stockid/{code}.phtml')


class SinaDivClient:
    def __init__(self, counter: FetchCounter, byte_counter: ByteCounter,
                 *, timeout_s: float = 10.0, rate_limit_s: float = 1.0,
                 max_retries: int = 2, transport=None, sleep=None,
                 disk_counter: ByteCounter | None = None):
        import time
        self.counter = counter
        self.bytes = byte_counter
        self.timeout_s = timeout_s
        self.rate_limit_s = rate_limit_s
        self.max_retries = max_retries
        self._sleep = sleep or time.sleep
        self.disk = disk_counter
        self._last = 0.0
        self._run_dir = None
        self._seq = 0
        # transport(url, timeout, read_cap) -> bytes（GET）
        self._transport = transport or self._default_transport

    def _default_transport(self, url, timeout, read_cap):
        opener = urllib.request.build_opener(
            urllib.request.ProxyHandler({}), _NoRedirect())
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with opener.open(req, timeout=timeout) as resp:
            body = resp.read(read_cap + 1)
        if len(body) > read_cap:
            raise ByteBudgetExceeded('新浪响应超剩余字节预算')
        return body

    def fetch_bonus_page(self, code6: str, archive_dir=None) -> bytes:
        """单代码分红配股页（1 次计数请求；含分红+配股两表）。"""
        url = PAGE_URL.format(code=code6)
        attempt = 0
        while True:
            self.counter.consume(1)
            wait = self.rate_limit_s - (__import__('time').time() - self._last)
            if wait > 0:
                self._sleep(wait)
            self._last = __import__('time').time()
            fetched_at = datetime.now(timezone.utc).isoformat(timespec='seconds')
            try:
                raw = self._transport(url, self.timeout_s,
                                      max(self.bytes.remaining, 0))
                self.bytes.consume(len(raw))
                self._archive(archive_dir, code6, url, fetched_at, raw, 'ok')
                return raw
            except (FetchBudgetExceeded, ByteBudgetExceeded, RedirectRefused):
                raise    # 预算/重定向即停，绝不重试
            except Exception:
                attempt += 1
                if attempt > self.max_retries:
                    raise
                self._sleep(2 ** attempt)

    def _archive(self, archive_dir, code6, url, fetched_at, raw, outcome):
        if archive_dir is None:
            return
        self._seq += 1
        sha = hashlib.sha256(raw).hexdigest()
        meta_obj = dict(seq=self._seq, code=code6, url=url, method='GET',
                        fetched_at=fetched_at, sha256=sha, bytes=len(raw),
                        outcome=outcome)
        meta = (json.dumps(meta_obj, ensure_ascii=False, indent=1) + '\n').encode('utf-8')
        index_line = (json.dumps(meta_obj, ensure_ascii=False) + '\n').encode('utf-8')
        if self.disk is not None:
            self.disk.consume(len(raw) + len(meta) + len(index_line))  # 先算后写
        import os
        if self._run_dir is None:
            ts = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%f')
            d = Path(archive_dir) / f'run_{ts}_{os.getpid()}'
            d.mkdir(parents=True, exist_ok=False)
            self._run_dir = d
        meta_obj['run_dir'] = self._run_dir.name
        base = f'{self._seq:04d}_{code6}'
        try:
            with (self._run_dir / f'{base}.bin').open('xb') as f:
                f.write(raw)
            with (self._run_dir / f'{base}.json').open('xb') as f:
                f.write(meta)
            with (self._run_dir / 'index.jsonl').open('ab') as f:
                f.write(index_line)
        except FileExistsError as exc:
            raise EvidenceCollision(f'证据碰撞拒绝覆盖：{exc.filename}')


def parse_bonus_page(html: bytes) -> dict:
    """解析分红配股页 → {'bonus': [...], 'allotment': [...], 'meta': {...}}。
    字段以页面实际表头为准原样保留（键=表头文本），不做语义重命名。"""
    import io
    import pandas as pd
    tables = pd.read_html(io.StringIO(html.decode('gbk', errors='replace')))
    out = {'tables_found': len(tables), 'bonus': None, 'allotment': None,
           'headers': []}
    for tb in tables:
        headers = [str(h) for h in tb.columns]
        out['headers'].append(headers)
        joined = '|'.join(headers)
        rows = tb.where(pd.notna(tb), None).to_dict('records')
        if '除权除息日' in joined and out['bonus'] is None:
            out['bonus'] = rows
        if '配股' in joined and out['allotment'] is None:
            out['allotment'] = rows
    return out
