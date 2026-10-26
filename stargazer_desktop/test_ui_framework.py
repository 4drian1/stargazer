"""test_ui_framework.py — Stargazer 1.0 UI 框架接线验证（零网络）。

覆盖：交易计划契约校验（合法/缺字段/区间颠倒）、落盘与读取回环、
/api/plan_check 规则分支（NO_PLAN/NOT_IN_PLAN/EXPIRED/NEAR/TRIGGERED/
BELOW_ZONE/WATCHING）、契约缺席时的 UI 状态（info_feed 未接入、
index_freshness 缺席、trade_plan 缺席）。
"""
import json
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import trade_plan_v1 as tp

PLAN = dict(version=1, created_ts='20261020223000', data_date='20261020',
            monitor_date='20261021',
            stocks=[dict(code='300639', name='普丽盛', buy_zone=[25.00, 25.60],
                         invalidations=['跌破25.00且无止跌'],
                         valid_until='20261021150000',
                         evidence_refs=['snap_v2'])],
            boards=[dict(name='共封装光学(CPO)', note='计划内板块')])


def test_validate_plan_ok():
    assert tp.validate_plan(PLAN) == []
    assert tp.validate_plan(json.loads(json.dumps(PLAN))) == []


def test_validate_plan_rejects_bad():
    bad = json.loads(json.dumps(PLAN))
    bad['version'] = 2
    assert 'VERSION_INVALID' in tp.validate_plan(bad)
    bad = json.loads(json.dumps(PLAN))
    bad['stocks'][0]['buy_zone'] = [25.60, 25.00]          # 区间颠倒
    assert 'STOCKS[0]_BUY_ZONE_ORDER' in tp.validate_plan(bad)
    bad = json.loads(json.dumps(PLAN))
    bad['stocks'][0]['buy_zone'] = None                    # 缺买入区
    assert 'STOCKS[0]_BUY_ZONE_INVALID' in tp.validate_plan(bad)
    bad = json.loads(json.dumps(PLAN))
    del bad['stocks']                                      # 空计划
    assert 'STOCKS_REQUIRED' in tp.validate_plan(bad)
    bad = json.loads(json.dumps(PLAN))
    bad['stocks'][0]['valid_until'] = '2026-10-21'         # 非14位
    assert 'STOCKS[0]_VALID_UNTIL_INVALID' in tp.validate_plan(bad)
    bad = json.loads(json.dumps(PLAN))
    bad['stocks'].append(dict(bad['stocks'][0]))           # 重复代码
    assert any('CODE_DUP' in e for e in tp.validate_plan(bad))
    assert tp.validate_plan({'hello': 'world'}) != []


def test_save_load_roundtrip(tmp: Path):
    f = tmp / 'trade_plan.json'
    tp.save_plan(PLAN, f)
    loaded = tp.load_plan(f)
    assert loaded == PLAN
    # 不合法计划拒绝落盘
    try:
        tp.save_plan({'version': 99}, f)
        raise AssertionError('应拒绝不合法计划')
    except ValueError:
        pass


def test_load_plan_absent(tmp: Path):
    assert tp.load_plan(tmp / 'nope.json') is None
    f = tmp / 'bad.json'
    f.write_text('not-json', encoding='utf-8')
    assert tp.load_plan(f) is None


def test_plan_check_branches():
    # 无计划 → NO_PLAN
    with patch.object(tp, 'load_plan', return_value=None):
        r = tp.plan_check('300639', 25.3, '20261021093000', plan=None)
    assert r['status'] == 'NO_PLAN' and r['decision_source'] == 'RULE'
    assert '模型决策待接入' in r['decision_note']
    # 不在计划
    r = tp.plan_check('000001', 25.3, '20261021093000', plan=PLAN)
    assert r['status'] == 'NOT_IN_PLAN'
    # 过期（时刻晚于 valid_until）
    r = tp.plan_check('300639', 25.3, '20261021160000', plan=PLAN)
    assert r['status'] == 'EXPIRED'
    # 触发（价格在买入区内）
    r = tp.plan_check('300639', 25.30, '20261021093000', plan=PLAN)
    assert r['status'] == 'TRIGGERED' and r['buy_zone'] == [25.0, 25.6]
    # 接近（上限上方 ≤1.5%）
    r = tp.plan_check('300639', 25.90, '20261021093000', plan=PLAN)
    assert r['status'] == 'NEAR', r
    # 高于上限且不够近 → 观察中
    r = tp.plan_check('300639', 27.00, '20261021093000', plan=PLAN)
    assert r['status'] == 'WATCHING'
    # 低于下限
    r = tp.plan_check('300639', 24.50, '20261021093000', plan=PLAN)
    assert r['status'] == 'BELOW_ZONE'
    # 价格非法
    r = tp.plan_check('300639', 'abc', '20261021093000', plan=PLAN)
    assert r['status'] == 'ERROR' and r['reason'] == 'PRICE_INVALID'


def test_server_routes_contract_absent(tmp: Path):
    """契约缺席时接口状态：trade_plan 无计划 / info_feed 未接入 / index_freshness 缺席。"""
    import server
    import trade_plan_v1 as tp2
    tp2.TRADE_PLAN_FILE = tmp / 'trade_plan.json'          # 缺席
    server.INFO_FEED_FILE = str(tmp / 'info_feed.jsonl')   # 缺席
    c = server.app.test_client()

    r = c.get('/api/trade_plan')
    assert r.status_code == 200
    body = r.get_json()
    assert body['status'] == 'NO_PLAN' and body['plan'] is None

    r = c.get('/api/info_feed')
    assert r.status_code == 200
    body = r.get_json()
    assert body['connected'] is False and body['items'] == []

    from unittest.mock import patch
    import index_minutes_v1
    with patch.object(index_minutes_v1, 'MIRROR', tmp / 'no_index'):
        r = c.get('/api/index_freshness')
        assert r.status_code == 200
        assert r.get_json()['available'] is False

    # info_feed 有数据 → 已接入（契约字段透出）
    (tmp / 'info_feed.jsonl').write_text(
        json.dumps(dict(ts='20261021090100', source='corvus', kind='announcement',
                        title='测试公告', summary='s', tags=['t'], refs=['300639']),
                   ensure_ascii=False) + '\n', encoding='utf-8')
    body = c.get('/api/info_feed').get_json()
    assert body['connected'] is True and body['items'][0]['title'] == '测试公告'

    # 计划写入：非法 400；合法 200 且可读回
    r = c.post('/api/trade_plan', data=json.dumps({'version': 2}),
               content_type='application/json')
    assert r.status_code == 400 and r.get_json()['errors']
    r = c.post('/api/trade_plan', data=json.dumps(PLAN, ensure_ascii=False),
               content_type='application/json')
    assert r.status_code == 200 and r.get_json()['ok'] is True
    body = c.get('/api/trade_plan').get_json()
    assert body['status'] == 'OK' and body['plan']['stocks'][0]['code'] == '300639'

    # plan_check 经路由：触发与缺席分支
    # Frozen fixture must not expire with the wall clock; no real notification.
    with patch.object(server.notifier, 'send_wechat', return_value=(False, 'offline test')):
        r = c.get('/api/plan_check?code=300639&price=25.30&ts=20261021100000')
    assert r.get_json()['status'] == 'TRIGGERED'
    r = c.get('/api/plan_check?price=25.30')               # 缺 code → 400 不 500
    assert r.status_code == 400
    tp2.TRADE_PLAN_FILE = tmp / 'gone.json'
    r = c.get('/api/plan_check?code=300639&price=25.30')
    assert r.get_json()['status'] == 'NO_PLAN'


if __name__ == '__main__':
    bad = 0
    fns = [(n, f) for n, f in sorted(globals().items())
           if n.startswith('test_')]
    for n, f in fns:
        try:
            if n in ('test_validate_plan_ok', 'test_validate_plan_rejects_bad',
                     'test_plan_check_branches'):
                f()
            else:
                with tempfile.TemporaryDirectory() as td:
                    f(Path(td))
            print('PASS', n)
        except Exception as exc:
            bad += 1
            print('FAIL', n, type(exc).__name__, str(exc)[:300])
    print(f'== {len(fns) - bad}/{len(fns)} 项 UI 框架接线验证通过 ==')
    sys.exit(1 if bad else 0)


def load_tests(loader, suite, pattern):
    """兼容 python -m unittest 发现。"""
    import unittest

    class FnCase(unittest.TestCase):
        def __init__(self, fn):
            super().__init__('runTest')
            self._fn = fn

        def runTest(self):
            if self._fn.__name__ in ('test_validate_plan_ok',
                                     'test_validate_plan_rejects_bad',
                                     'test_plan_check_branches'):
                self._fn()
            else:
                with tempfile.TemporaryDirectory() as td:
                    self._fn(Path(td))

        def shortDescription(self):
            return self._fn.__name__

    for n, f in sorted(globals().items()):
        if n.startswith('test_') and callable(f):
            suite.addTest(FnCase(f))
    return suite
