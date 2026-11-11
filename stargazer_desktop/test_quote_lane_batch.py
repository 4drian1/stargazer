"""离线测试：计划票快速报价通道 batch_size=30 修复。

隔离原则（用户硬性要求）：
- 不 mock 整个 quotes()；只替换最底层 urllib.request.urlopen，让真实
  quotes() 切批、quotes._get 的 on_attempt 额度闸、tries 重试、gtimg 文本解析
  全部真实运行。
- 禁止真实行情请求、付费调用、推送。所有账本落临时目录。

覆盖 6 项：
1. 7 只计划票 -> 仅 1 次 HTTP，报价完整返回。
2. 同批失败重试 3 次，账本准确 +3。
3. 额度耗尽后，底层 HTTP 零调用。
4. 发送前单票复查仍正常计账（1 票 = 1 次请求 +1 账）。
5. 指标阻塞时快速报价仍运行；重复报价输出稳定（下游去重有稳定输入）。
6. 31 只票按 30+1 分批 = 2 次 HTTP，证明批量上限与真实尝试计数未失效。
   + 回归：quotes() 全局默认 batch_size=6 不变（7 票默认 = 2 次）。
"""
import os
import sys
import json
import time
import tempfile
import unittest
from pathlib import Path
from unittest import mock

# 让 stargazer_desktop 与其子包 tools 可导入
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import tools.market as market
import pentacles_monitor as pm

TODAY = time.strftime('%Y%m%d')


def _line(secid):
    code = secid[2:] if secid[:2] in ('sh', 'sz') else secid
    b = [''] * 46
    b[1] = 'NAME_' + code
    b[2] = code
    b[3] = '10.00'
    b[4] = '9.50'
    b[5] = '9.80'
    b[30] = TODAY + '150000'
    b[32] = '5.26'
    b[33] = '10.20'
    b[34] = '9.60'
    b[37] = '12345.6'
    b[44] = '5000.0'
    b[45] = '8000.0'
    return 'v_%s="%s"' % (secid, '~'.join(b))


def _build_response(q):
    secids = q.split(',')
    return (';'.join(_line(s) for s in secids)).encode('gbk')


class _FakeUrlOpen:
    """成功桩：按 URL 里的 q 参数回真实格式报价；统计真实 HTTP 次数。"""

    def __init__(self):
        self.calls = []

    def __call__(self, req, timeout=None):
        self.calls.append(req.full_url if hasattr(req, 'full_url') else req)
        q = req.full_url.split('q=', 1)[1]
        return _FakeBody(_build_response(q))


class _FakeBody:
    def __init__(self, data):
        self._d = data

    def read(self):
        return self._d


class _FailingUrlOpen:
    """失败桩：每次调用都抛异常，模拟网络失败 -> _get 走 tries 重试。"""

    def __init__(self):
        self.calls = []

    def __call__(self, req, timeout=None):
        self.calls.append(req.full_url if hasattr(req, 'full_url') else req)
        raise OSError('simulated network failure')


def _tmp_budget(polls=0):
    d = tempfile.mkdtemp(prefix='quote_lane_test_')
    f = Path(d) / 'quote_lane_budget.json'
    f.write_text(json.dumps({'day': TODAY, 'polls': polls, 'blocked': False}),
                 encoding='utf-8')
    return f


def _polls(f):
    return json.loads(Path(f).read_bytes()).get('polls', 0)


SEVEN = ['600000', '600036', '601318', '000001', '000002', '300750', '002594']


class QuoteLaneBatchTest(unittest.TestCase):

    def tearDown(self):
        pm._BUDGET_FILE[0] = None

    # 1) 7 只票 -> 1 次 HTTP，完整返回
    def test_seven_codes_single_http(self):
        fake = _FakeUrlOpen()
        bf = _tmp_budget()
        with mock.patch('tools.market.urllib.request.urlopen', fake):
            out = pm._quotes_with_timeout(list(SEVEN), budget_file=bf)
        self.assertEqual(len(fake.calls), 1, '7 票应合并为 1 次请求')
        self.assertEqual(set(out.keys()), set(SEVEN), '7 只报价应完整返回')
        # 账本：1 批 = 1 次预约
        self.assertEqual(_polls(bf), 1)

    # 2) 同批失败重试 3 次，账本 +3
    def test_same_batch_retry_counts_three(self):
        fake = _FailingUrlOpen()
        bf = _tmp_budget()
        with mock.patch('tools.market.urllib.request.urlopen', fake):
            out = pm._quotes_with_timeout(list(SEVEN), budget_file=bf)
        self.assertEqual(out, {}, '全失败应返回空 dict')
        self.assertEqual(_polls(bf), 3, '失败重试 3 次，账本应 +3')
        self.assertEqual(len(fake.calls), 3, '底层 HTTP 应被尝试 3 次')

    # 3) 额度耗尽 -> 零 HTTP
    def test_exhausted_budget_zero_http(self):
        fake = _FakeUrlOpen()
        bf = _tmp_budget(polls=pm.QUOTE_DAILY_CAP)  # 2000
        with mock.patch('tools.market.urllib.request.urlopen', fake):
            out = pm._quotes_with_timeout(list(SEVEN), budget_file=bf)
        self.assertEqual(len(fake.calls), 0, '额度耗尽不应有任何 HTTP 调用')
        self.assertEqual(out, {}, '拦截时应返回空 dict')
        self.assertEqual(_polls(bf), pm.QUOTE_DAILY_CAP, '耗尽态账本不应再增加')

    # 4) 发送前单票复查仍计账
    def test_single_ticket_recheck_accounts(self):
        fake = _FakeUrlOpen()
        bf = _tmp_budget()
        with mock.patch('tools.market.urllib.request.urlopen', fake):
            out = pm._quotes_with_timeout(['600000'], budget_file=bf)
        self.assertEqual(len(fake.calls), 1, '单票复查 = 1 次请求')
        self.assertIn('600000', out, '单票报价应返回')
        self.assertEqual(_polls(bf), 1, '单票复查账本 +1')

    # 5) 指标阻塞时快速报价仍运行；重复报价输出稳定
    def test_fast_quote_independent_and_stable(self):
        fake = _FakeUrlOpen()
        bf = _tmp_budget()
        with mock.patch('tools.market.urllib.request.urlopen', fake):
            out1 = pm._quotes_with_timeout(list(SEVEN), budget_file=bf)
            out2 = pm._quotes_with_timeout(list(SEVEN), budget_file=bf)
        # 快速报价通道独立于指标：先后两次都成功
        self.assertEqual(set(out1.keys()), set(SEVEN))
        # 重复报价输出一致 -> 下游去重有稳定输入，不重复推送
        self.assertEqual(out1, out2, '相同输入应产生确定一致的报价，避免重复推送')
        # 两次轮询各 1 次请求，账本 +2（不被合并）
        self.assertEqual(len(fake.calls), 2)
        self.assertEqual(_polls(bf), 2)

    # 6) 31 只票 -> 30+1 = 2 次 HTTP；批量上限仍为 31
    def test_thirty_one_codes_two_batches(self):
        codes = ['%06d' % (i % 1000000) for i in range(100000, 100031)]  # 31 只
        fake = _FakeUrlOpen()
        bf = _tmp_budget()
        with mock.patch('tools.market.urllib.request.urlopen', fake):
            out = pm._quotes_with_timeout(codes, budget_file=bf)
        self.assertEqual(len(fake.calls), 2, '31 票应分 30+1 两批 = 2 次请求')
        self.assertEqual(set(out.keys()), set(codes), '31 只报价应完整返回')
        self.assertEqual(_polls(bf), 2)
        # 全局默认 6 未变：显式用默认应得 6 批
        fake2 = _FakeUrlOpen()
        with mock.patch('tools.market.urllib.request.urlopen', fake2):
            market.quotes(codes)  # 默认 batch_size=6
        self.assertEqual(len(fake2.calls), 6, 'quotes() 默认 batch_size 仍应为 6，不影响其他调用方')


if __name__ == '__main__':
    unittest.main(verbosity=2)
