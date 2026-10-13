"""cninfo_ann_client.py — 巨潮公告受控适配器（批次 a 执行件；不用 akshare 封装）。

与 akshare 封装的差异（用户裁决 #18.7）：无隐藏辅助查询（orgId 用 searchkey 直查）、
无自动分页（页码显式传入、每页=1 次计数请求）、无重复首页、每请求显式 timeout。
纪律与 REV4 腾讯客户端一致：FetchCounter/ByteCounter/DiskCounter 共享、限量读、
禁重定向（_NoRedirect 复用）、原始留证（含拒收/异常）、run 目录独占。
"""
from __future__ import annotations

import json
import urllib.parse
import urllib.request

from independent_fetch_client import (ByteBudgetExceeded, ByteCounter,
                                      FetchBudgetExceeded, FetchCounter, _NoRedirect,
                                      RedirectRefused)
LIST_URL = 'http://www.cninfo.com.cn/new/hisAnnouncement/query'


class CninfoAnnClient:
    def __init__(self, counter: FetchCounter, byte_counter: ByteCounter,
                 *, timeout_s: float = 10.0, rate_limit_s: float = 1.0,
                 max_retries: int = 1, transport=None, sleep=None,
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
        # transport(url, data_bytes, timeout, read_cap) -> bytes（POST）
        self._transport = transport or self._default_transport

    def _default_transport(self, url, data, timeout, read_cap):
        opener = urllib.request.build_opener(
            urllib.request.ProxyHandler({}), _NoRedirect())
        req = urllib.request.Request(url, data=data, headers={
            'Accept': 'application/json',
            'Content-Type': 'application/x-www-form-urlencoded; charset=UTF-8',
            'User-Agent': 'Mozilla/5.0'})
        with opener.open(req, timeout=timeout) as resp:
            body = resp.read(read_cap + 1)
        if len(body) > read_cap:
            from independent_fetch_client import ByteBudgetExceeded
            raise ByteBudgetExceeded('巨潮响应超剩余字节预算')
        return body

    def _request(self, url, data) -> bytes:
        attempt = 0
        while True:
            self.counter.consume(1)
            wait = self.rate_limit_s - (__import__('time').time() - self._last)
            if wait > 0:
                self._sleep(wait)
            self._last = __import__('time').time()
            try:
                raw = self._transport(url, data, self.timeout_s,
                                      max(self.bytes.remaining, 0))
                self.bytes.consume(len(raw))
                return raw
            except RedirectRefused:
                raise
            except (FetchBudgetExceeded, ByteBudgetExceeded):
                raise    # 预算触线即停，绝不重试
            except Exception:
                attempt += 1
                if attempt > self.max_retries:
                    raise
                self._sleep(2 ** attempt)

    def query_ann_list(self, code6: str, se_date: str, page_size: int = 30,
                       page_num: int = 1, archive_dir=None) -> dict:
        """se_date='YYYY-MM-DD~YYYY-MM-DD'。一次调用=1 次计数请求（无自动翻页）。"""
        column = 'sse' if code6.startswith('6') else 'szse'
        form = dict(pageNum=str(page_num), pageSize=str(page_size),
                    tabName='fullpage', column=column,
                    category='', plate='', stock='',
                    searchkey=code6, secid='', sortName='',
                    sortType='', isHLtitle='true', seDate=se_date)
        raw = self._request(LIST_URL,
                            urllib.parse.urlencode(form).encode('utf-8'))
        self._archive(archive_dir, f'list_{code6}_p{page_num}', LIST_URL, form, raw, 'ok')
        return json.loads(raw.decode('utf-8'))

    def fetch_ann_text(self, adjunct_url: str, archive_dir=None) -> bytes:
        """原文 PDF/文本下载（ adjunctUrl 相对路径补全 host 由调用方做）。1 请求。"""
        url = adjunct_url if adjunct_url.startswith('http') \
            else 'http://static.cninfo.com.cn/' + adjunct_url.lstrip('/')
        data = urllib.parse.urlencode({}).encode()
        raw = self._request(url, None)   # GET：data=None，与 method 标记一致
        self._archive(archive_dir, f'ann_{self._seq:03d}', url, {'method': 'GET'}, raw, 'ok')
        return raw

    def _archive(self, archive_dir, name, url, form, raw, outcome):
        if archive_dir is None:
            return
        import hashlib, os
        from datetime import datetime, timezone
        from pathlib import Path
        if self._run_dir is None:
            ts = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%f')
            d = Path(archive_dir) / f'run_{ts}_{os.getpid()}'
            d.mkdir(parents=True, exist_ok=False)
            self._run_dir = d
        self._seq += 1
        sha = hashlib.sha256(raw).hexdigest()
        meta_obj = dict(seq=self._seq, name=name, url=url, form=form,
                        fetched_at=datetime.now(timezone.utc).isoformat(timespec='seconds'),
                        sha256=sha, bytes=len(raw), outcome=outcome, run_dir=self._run_dir.name)
        meta = (json.dumps(meta_obj, ensure_ascii=False, indent=1) + '\n').encode('utf-8')
        index_line = (json.dumps(meta_obj, ensure_ascii=False) + '\n').encode('utf-8')
        if self.disk is not None:
            self.disk.consume(len(raw) + len(meta) + len(index_line))
        base = f'{self._seq:04d}_{name[:40]}'
        try:
            with (self._run_dir / f'{base}.bin').open('xb') as f:
                f.write(raw)
            with (self._run_dir / f'{base}.json').open('xb') as f:
                f.write(meta)
            with (self._run_dir / 'index.jsonl').open('ab') as f:
                f.write(index_line)
        except FileExistsError as exc:
            from independent_fetch_client import EvidenceCollision
            raise EvidenceCollision(f'证据碰撞拒绝覆盖：{exc.filename}')
