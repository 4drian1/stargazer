"""#18.188-W2 isolated acceptance: roster, manifest, version binding, backfill."""
import json
import sys
import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import postclose_prepared_v1 as P

CALENDAR = [20261026, 20261027, 20261028, 20261029, 20261030, 20261102, 20261103, 20261104]
DAY = 20261103


class FrozenDatetime(datetime):
    current = None

    @classmethod
    def now(cls, tz=None):
        return cls.current


class PreparedTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.risk = self.root / 'postclose_risk'
        self.risk.mkdir(parents=True)
        self.scheduled = []

        def fake_schedule(plan):
            self.scheduled.append(plan)
            return str(self.risk / ('prepared_fake_%d.json' % len(self.scheduled)))

        self.modules = mock.patch.dict(sys.modules, {
            'postclose_risk_runtime': SimpleNamespace(directory=lambda: self.risk),
            'minute_history_v1': SimpleNamespace(schedule=fake_schedule),
            'services': SimpleNamespace(wl_load=lambda cfg: self.wl)})
        self.modules.start()
        self.wl = {'positions': [], 'watch': []}
        self.calendar = mock.patch.object(P, '_calendar_days', return_value=CALENDAR)
        self.calendar.start()
        self.clock = mock.patch.object(P, 'datetime', FrozenDatetime)
        self.clock.start()
        FrozenDatetime.current = datetime(2026, 11, 3, 20, 0, 0)

    def tearDown(self):
        self.clock.stop()
        self.calendar.stop()
        self.modules.stop()
        self.temp.cleanup()

    # ---- roster -------------------------------------------------------------
    def test_roster_from_plan_kinds_and_dedupe(self):
        plan = dict(
            rows=[dict(code='300652', entry_rule=dict(history_days=[20261029, 20261030])),
                  dict(code='603028', entry_rule=dict())],
            manual_observations=[dict(code='000510'), dict(code='300652')],
            selected_manual_codes=['301132', '000510'])
        roster = P.roster_from_plan(plan)
        self.assertEqual(roster['300652']['kind'], 'PLAN')
        self.assertEqual(roster['300652']['history_days'], [20261029, 20261030])
        self.assertEqual(roster['000510']['kind'], 'MANUAL')  # dedupe: first wins
        self.assertEqual(roster['301132']['kind'], 'MANUAL')

    def test_roster_from_holdings_last_four_days(self):
        roster = P.roster_from_holdings([dict(code='510'), dict(code='xx1'), dict(code='600121')],
                                        CALENDAR, 20261102)
        self.assertEqual(sorted(roster), ['000510', '600121'])
        self.assertEqual(roster['600121']['history_days'], [20261028, 20261029, 20261030, 20261102])
        self.assertEqual(roster['600121']['kind'], 'HOLDINGS')

    # ---- incremental scheduling --------------------------------------------
    def test_schedule_roster_groups_by_shape(self):
        items = {'300652': dict(kind='PLAN', history_days=[1, 2], turnover_reference=None),
                 '603028': dict(kind='PLAN', history_days=[1, 2], turnover_reference=None),
                 '301132': dict(kind='MANUAL', history_days=[1, 2], turnover_reference=None),
                 '000510': dict(kind='HOLDINGS', history_days=[9], turnover_reference=None)}
        targets = P.schedule_roster(items, DAY, DAY + 1)
        self.assertEqual(len(self.scheduled), 2)  # two cohort shapes
        cohorts = [sorted(r['code'] for r in call['rows']) for call in self.scheduled]
        self.assertIn(['300652', '301132', '603028'], cohorts)
        self.assertIn(['000510'], cohorts)
        self.assertEqual(len(targets), 2)

    # ---- manifest ------------------------------------------------------------
    def _ready_target(self, name, codes, status='READY', reason=None,
                      P=DAY, E=None, receipts=True, days=(20261030,)):
        target = self.risk / name
        E = E if E is not None else DAY + 1
        rows = {c: dict(status='READY', days=list(days), minute_rows=96,
                        receipts=[dict(src='tdx')] if receipts else [],
                        previous_low_indicators={}) for c in codes}
        value = dict(status=status, data_date=P, monitor_date=E, rows=rows)
        if reason:
            value['reason'] = reason
        target.write_text(json.dumps(value), encoding='utf-8')
        return str(target)

    def test_refresh_manifest_status_and_open_items(self):
        t1 = self._ready_target('p1.json', ['300652'])
        t2 = self._ready_target('p2.json', ['000510'], status='WAIT_SOURCE', reason='TDX缺日')
        roster = {'300652': dict(kind='PLAN', history_days=[20261030]),
                  '000510': dict(kind='HOLDINGS', history_days=[20261030])}
        manifest = P.refresh_manifest(DAY, DAY + 1, roster,
                                      [(['300652'], [20261030], t1),
                                       (['000510'], [20261030], t2)])
        self.assertEqual(manifest['contract'], P.CONTRACT)
        self.assertEqual(manifest['load_version'], 'P1')
        self.assertEqual(manifest['observation_date'], DAY + 1)
        self.assertTrue(manifest['source_hash'])
        self.assertTrue(manifest['prepared_at'])
        self.assertEqual(manifest['items']['300652']['status'], 'READY')
        self.assertEqual(manifest['items']['000510']['status'], 'WAIT_SOURCE')
        self.assertIn('TDX缺日', manifest['items']['000510']['reason'])
        ids = [i['id'] for i in manifest['open_items']]
        self.assertEqual(ids.count('PYXIS_FULL_BOARD_SUPPLY'), 1)
        # 重复刷新不叠未完成项
        again = P.refresh_manifest(DAY, DAY + 1, roster,
                                   [(['300652'], [20261030], t1),
                                    (['000510'], [20261030], t2)],
                                   extra_items=manifest['open_items'])
        self.assertEqual([i['id'] for i in again['open_items']].count('PYXIS_FULL_BOARD_SUPPLY'), 1)

    def test_binding_freshness(self):
        self.assertIsNone(P.binding(day=DAY))
        roster = {'300652': dict(kind='PLAN', history_days=[1])}
        P.refresh_manifest(DAY, DAY + 1, roster, [])
        fresh = P.binding(day=DAY + 1)
        self.assertTrue(fresh['fresh'])
        self.assertEqual(fresh['load_version'], 'P1')
        stale = P.binding(day=DAY + 2)
        self.assertFalse(stale['fresh'])

    # ---- W2a on_plan_frozen ---------------------------------------------------
    def test_on_plan_frozen_syncs_stamps_and_prepares(self):
        export_dir = self.root / 'workspace' / 'data'
        export_dir.mkdir(parents=True)
        exported = export_dir / 'plan_frozen.json'
        exported.write_text('{"sessions": []}', encoding='utf-8')
        self.wl = {'positions': [dict(code='600121')], 'watch': []}
        result = P.on_plan_frozen({}, exported, DAY)
        synced = Path(result['stamp']['synced_to'])
        self.assertEqual(synced, self.risk.parent / 'plan_frozen.json')
        self.assertEqual(synced.read_bytes(), exported.read_bytes())
        meta = json.loads((self.risk.parent / 'plan_frozen.meta.json').read_bytes())
        self.assertEqual(meta['trading_date'], DAY)
        self.assertTrue(meta['source_sha256'])
        self.assertEqual(meta['load_version'], 'P1')
        manifest = result['manifest']
        self.assertEqual(manifest['observation_date'], 20261104)  # calendar 次日
        self.assertIn('600121', manifest['items'])
        self.assertEqual(manifest['items']['600121']['kind'], 'HOLDINGS')

    # ---- W2b watchlist incremental -------------------------------------------
    def test_watchlist_backfill_only_new_codes(self):
        t1 = self._ready_target('p1.json', ['300652'])
        roster = {'300652': dict(kind='PLAN', history_days=[20261030], target=t1, status='READY')}
        P.refresh_manifest(DAY, DAY + 1, roster, [(['300652'], [20261030], t1)])
        self.wl = {'positions': [dict(code='600121')], 'watch': []}
        self.scheduled.clear()
        P._watchlist_backfill({})
        self.assertEqual(len(self.scheduled), 1)
        self.assertEqual([r['code'] for r in self.scheduled[0]['rows']], ['600121'])
        manifest = P.current_manifest()
        self.assertIn('600121', manifest['items'])
        self.assertEqual(manifest['items']['300652']['status'], 'READY')  # A 不受影响

    def test_watchlist_backfill_skips_stale_manifest(self):
        t1 = self._ready_target('p1.json', ['300652'])
        roster = {'300652': dict(kind='PLAN', history_days=[20261030], target=t1, status='READY')}
        P.refresh_manifest(DAY - 1, DAY, roster, [(['300652'], [20261030], t1)])  # 观察日=今天之前? 用昨天数据日
        P.refresh_manifest(20261102, DAY, roster, [(['300652'], [20261030], t1)])
        FrozenDatetime.current = datetime(2026, 11, 4, 8, 0, 0)  # 观察日已过
        self.wl = {'positions': [dict(code='600121')], 'watch': []}
        self.scheduled.clear()
        P._watchlist_backfill({})
        self.assertEqual(self.scheduled, [])

    # ---- W2d verify_and_backfill ----------------------------------------------
    def test_backfill_reschedules_only_missing(self):
        t1 = self._ready_target('p1.json', ['300652'])
        t2 = self._ready_target('p2.json', ['000510'], status='WAIT_SOURCE', reason='缺源')
        roster = {'300652': dict(kind='PLAN', history_days=[20261030], target=t1, status='READY'),
                  '000510': dict(kind='HOLDINGS', history_days=[20261030], target=t2, status='READY')}
        P.refresh_manifest(DAY, DAY + 1, roster, [(['300652'], [20261030], t1), (['000510'], [20261030], t2)])
        self.scheduled.clear()
        report = P.verify_and_backfill({}, day=DAY + 1)
        self.assertEqual(report['status'], 'BACKFILLED')
        self.assertEqual(report['backlog'], ['000510'])
        self.assertEqual([r['code'] for r in self.scheduled[0]['rows']], ['000510'])

    def test_backfill_all_ready_no_reschedule(self):
        t1 = self._ready_target('p1.json', ['300652'])
        roster = {'300652': dict(kind='PLAN', history_days=[20261030], target=t1, status='READY')}
        P.refresh_manifest(DAY, DAY + 1, roster, [(['300652'], [20261030], t1)])
        self.scheduled.clear()
        report = P.verify_and_backfill({}, day=DAY + 1)
        self.assertEqual(report['status'], 'READY')
        self.assertEqual(self.scheduled, [])
        self.assertEqual(report['open_items'][0]['id'], 'PYXIS_FULL_BOARD_SUPPLY')

    def test_backfill_stale_or_absent_manifest(self):
        self.assertEqual(P.verify_and_backfill({}, day=DAY + 1)['status'], 'NO_MANIFEST')
        roster = {'300652': dict(kind='PLAN', history_days=[20261030])}
        P.refresh_manifest(DAY, DAY + 1, roster, [])
        self.assertEqual(P.verify_and_backfill({}, day=DAY + 5)['status'], 'STALE_MANIFEST')

    # ---- 收口验证组1：A就绪+新增B，A证据不变 --------------------------------
    def test_incremental_refresh_preserves_ready_evidence(self):
        t1 = self._ready_target('p1.json', ['300652'])
        roster = {'300652': dict(kind='PLAN', history_days=[20261030], target=t1, status='READY')}
        P.refresh_manifest(DAY, DAY + 1, roster, [(['300652'], [20261030], t1)])
        before = json.loads(Path(t1).read_bytes())
        self.wl = {'positions': [dict(code='600121')], 'watch': []}
        P._watchlist_backfill({})
        manifest = P.current_manifest()
        self.assertEqual(manifest['items']['300652']['status'], 'READY')
        self.assertEqual(manifest['items']['300652']['target'], t1)
        self.assertEqual(manifest['items']['600121']['status'], 'SCHEDULED')
        after = json.loads(Path(t1).read_bytes())
        self.assertEqual(before, after)  # A 的准备产物未被触碰

    # ---- 收口验证组2：需求变化与连续编辑都收敛 -------------------------------
    def test_requirement_change_reschedules_existing_code(self):
        t1 = self._ready_target('p1.json', ['600121'], days=(20261030,))
        # 名册里 600121 是 PLAN 且历史日只有 20261030；持仓口径要求近4日 → 需求变化
        roster = {'600121': dict(kind='PLAN', history_days=[20261030], target=t1, status='READY')}
        P.refresh_manifest(DAY, DAY + 1, roster, [(['600121'], [20261030], t1)])
        self.wl = {'positions': [dict(code='600121')], 'watch': []}
        self.scheduled.clear()
        P._watchlist_backfill({})
        self.assertEqual(len(self.scheduled), 1)
        new_days = self.scheduled[0]['rows'][0]['entry_rule']['history_days']
        self.assertEqual(new_days, [20261029, 20261030, 20261102, 20261103])
        manifest = P.current_manifest()
        self.assertEqual(manifest['items']['600121']['status'], 'SCHEDULED')
        self.assertNotEqual(manifest['items']['600121']['target'], t1)

    def test_burst_saves_coalesce_to_final_state(self):
        t1 = self._ready_target('p1.json', ['300652'])
        roster = {'300652': dict(kind='PLAN', history_days=[20261030], target=t1, status='READY')}
        P.refresh_manifest(DAY, DAY + 1, roster, [(['300652'], [20261030], t1)])
        # 模拟连续两次保存（防抖线程合并为最终状态处理）
        self.wl = {'positions': [dict(code='600121'), dict(code='000510')], 'watch': []}
        P._wl_state['pending'] = True
        P._wl_state['cfg'] = {}
        P._watchlist_worker()
        self.assertFalse(P._wl_state['pending'])
        manifest = P.current_manifest()
        self.assertEqual(sorted(manifest['items']), ['000510', '300652', '600121'])
        self.assertEqual(manifest['items']['300652']['status'], 'READY')

    # ---- 收口验证组3：错日期/缺股票/丢target不误报READY ----------------------
    def test_strict_readiness_rejects_bad_evidence(self):
        base = dict(kind='PLAN', history_days=[20261030], status='READY')
        good = self._ready_target('good.json', ['300652'])
        self.assertTrue(P.item_ready(dict(base, target=good, _code='300652'), DAY, DAY + 1))
        wrong_date = self._ready_target('wd.json', ['300652'], P=DAY - 1)
        self.assertFalse(P.item_ready(dict(base, target=wrong_date, _code='300652'), DAY, DAY + 1))
        missing_code = self._ready_target('mc.json', ['603028'])
        self.assertFalse(P.item_ready(dict(base, target=missing_code, _code='300652'), DAY, DAY + 1))
        top_ready_empty_rows = self.risk / 'empty.json'
        top_ready_empty_rows.write_text(json.dumps(
            dict(status='READY', data_date=DAY, monitor_date=DAY + 1, rows={})),
            encoding='utf-8')
        self.assertFalse(P.item_ready(dict(base, target=str(top_ready_empty_rows),
                                           _code='300652'), DAY, DAY + 1))
        no_receipts = self._ready_target('nr.json', ['300652'], receipts=False)
        self.assertFalse(P.item_ready(dict(base, target=no_receipts, _code='300652'), DAY, DAY + 1))
        self.assertFalse(P.item_ready(dict(base, target=None, _code='300652'), DAY, DAY + 1))
        self.assertFalse(P.item_ready(dict(base, target=str(self.risk / 'gone.json'),
                                           _code='300652'), DAY, DAY + 1))

    def test_verify_backfills_lost_target(self):
        t1 = self._ready_target('p1.json', ['300652'])
        roster = {'300652': dict(kind='PLAN', history_days=[20261030], target=t1, status='READY')}
        P.refresh_manifest(DAY, DAY + 1, roster, [(['300652'], [20261030], t1)])
        Path(t1).unlink()  # 证据丢失
        self.scheduled.clear()
        report = P.verify_and_backfill({}, day=DAY + 1)
        self.assertEqual(report['status'], 'BACKFILLED')
        self.assertEqual(report['backlog'], ['300652'])

    def test_turnover_dict_grouping_and_top_level(self):
        spec = dict(windows=[1, 5], field='amount')
        items = {'300652': dict(kind='PLAN', history_days=[20261030], turnover_reference=spec),
                 '603028': dict(kind='PLAN', history_days=[20261030], turnover_reference=spec),
                 '000510': dict(kind='PLAN', history_days=[20261030], turnover_reference=None)}
        targets = P.schedule_roster(items, DAY, DAY + 1)
        self.assertEqual(len(self.scheduled), 2)  # dict 规格可哈希分组
        with_spec = next(c for c in self.scheduled if c['rows'][0]['turnover_reference'])
        self.assertEqual(with_spec['rows'][0]['turnover_reference'], spec)  # 行顶层，schedule真实读取位
        self.assertEqual(len(with_spec['rows']), 2)
        # 换手规格纳入版本hash
        a = P._req_hash(dict(kind='PLAN', history_days=[20261030], turnover_reference=spec))
        b = P._req_hash(dict(kind='PLAN', history_days=[20261030], turnover_reference=None))
        self.assertNotEqual(a, b)

    def test_on_plan_frozen_reads_exported_candidates(self):
        export_dir = self.root / 'workspace' / 'data'
        export_dir.mkdir(parents=True)
        exported = export_dir / 'plan_frozen.json'
        exported.write_text(json.dumps(dict(
            sessions=[dict(version_no=2, candidates=[dict(code='301080'), dict(code='abc')])])),
            encoding='utf-8')
        self.wl = {'positions': [], 'watch': []}
        result = P.on_plan_frozen({}, exported, DAY)
        manifest = result['manifest']
        self.assertIn('301080', manifest['items'])
        self.assertEqual(manifest['items']['301080']['kind'], 'PLAN')
        self.assertEqual(len(result['stamp']['source_sha256']), 64)

    def test_on_plan_frozen_requires_calendar(self):
        export_dir = self.root / 'workspace' / 'data'
        export_dir.mkdir(parents=True)
        exported = export_dir / 'plan_frozen.json'
        exported.write_text('{}', encoding='utf-8')
        self.wl = {'positions': [], 'watch': []}
        with mock.patch.object(P, '_calendar_days', return_value=[]):
            with self.assertRaises(ValueError) as ctx:
                P.on_plan_frozen({}, exported, DAY)
        self.assertIn('CALENDAR_UNAVAILABLE', str(ctx.exception))  # 不缺日历退回E=P

    # ---- 收口验证组4：缺清单重建 + 失败有界续查（无需重启） --------------------
    def test_rebuild_from_authority_and_bounded_retry(self):
        import sys as _sys
        plan_dir = self.root / 'pentacles_demo'
        plan_dir.mkdir(parents=True)
        (plan_dir / 'assistant_plan.json').write_text(json.dumps(dict(
            data_date=DAY, monitor_date=DAY + 1,
            rows=[dict(code='300652', entry_rule=dict(history_days=[20261030]))],
            manual_observations=[dict(code='000510')], selected_manual_codes=[],
        )), encoding='utf-8')
        paths_stub = SimpleNamespace(data_file=lambda rel, seed_from_bundle=False:
                                     str(plan_dir / Path(rel).name), FROZEN=False)
        self.wl = {'positions': [dict(code='600121')], 'watch': []}
        with mock.patch.dict(sys.modules, {'paths': paths_stub}):
            manifest = P.rebuild_from_authority({})
        self.assertIsNotNone(manifest)
        self.assertEqual(sorted(manifest['items']), ['000510', '300652', '600121'])
        self.assertEqual(manifest['data_date'], DAY)

    def test_monitor_check_recovers_without_restart(self):
        import intraday_monitor
        import notifier
        monitor = intraday_monitor.IntradayMonitor(lambda: {})
        logs = []
        reports = iter([
            dict(status='NO_MANIFEST', day=DAY + 1),   # 首次：缺清单
            dict(status='BACKFILLED', day=DAY + 1, backlog=['300652']),  # 重建后仍缺
            dict(status='BACKFILLED', day=DAY + 1, backlog=['300652']),  # 8:46 续查：仍缺
            dict(status='READY', day=DAY + 1, items=3,
                 open_items=[dict(id='PYXIS_FULL_BOARD_SUPPLY')]),  # 9:02 数据恢复
        ])
        state = {'rebuilt': 0}
        stub = SimpleNamespace(
            verify_and_backfill=lambda cfg, day=None: next(reports),
            rebuild_from_authority=lambda cfg: state.__setitem__('rebuilt', state['rebuilt'] + 1) or {'ok': True})
        clock = [datetime(2026, 11, 4, 8, 30, 0)]
        with mock.patch.dict(sys.modules, {'postclose_prepared_v1': stub}), \
                mock.patch.object(notifier, 'log', side_effect=lambda msg: logs.append(msg)):
            monitor._check_prepared({}, clock[0])
            self.assertEqual(state['rebuilt'], 1)  # 缺清单已从权威名单重建
            monitor._check_prepared({}, clock[0])  # 15分钟内有界：跳过
            self.assertEqual(state['rebuilt'], 1)
            clock[0] = datetime(2026, 11, 4, 8, 46, 0)
            monitor._check_prepared({}, clock[0])  # 续查：仍缺
            self.assertIsNone(monitor._prepared_done_day)
            clock[0] = datetime(2026, 11, 4, 9, 2, 0)
            monitor._check_prepared({}, clock[0])  # 数据恢复：就绪
            self.assertEqual(monitor._prepared_done_day, '20261104')
            clock[0] = datetime(2026, 11, 4, 9, 30, 0)
            monitor._check_prepared({}, clock[0])  # 当日已完成：不再检查
        self.assertEqual(state['rebuilt'], 1)
        self.assertTrue(any('15分钟后复查' in m for m in logs))

    def test_restart_continuation_via_fresh_import(self):
        """模块重新加载测试（同一进程内）：状态全在磁盘，reload 后可续接。
        跨进程验收见 test_cross_process_restart_acceptance。"""
        import importlib
        t1 = self._ready_target('p1.json', ['300652'])
        roster = {'300652': dict(kind='PLAN', history_days=[20261030], target=t1, status='READY')}
        P.refresh_manifest(DAY, DAY + 1, roster, [(['300652'], [20261030], t1)])
        self.scheduled.clear()
        reloaded = importlib.reload(P)  # 模拟重启：模块级状态全部清空
        self.assertIsNotNone(reloaded.verify_and_backfill)
        report = reloaded.verify_and_backfill({}, day=DAY + 1)
        self.assertEqual(report['status'], 'READY')  # 重载后即可识别就绪
        Path(t1).write_text(json.dumps(dict(  # 证据损坏
            status='WAIT_SOURCE', data_date=DAY, monitor_date=DAY + 1,
            rows={'300652': dict(status='WAIT_SOURCE', reason='TDX缺日',
                                days=[20261030], receipts=[{}])})), encoding='utf-8')
        again = importlib.reload(reloaded)
        report = again.verify_and_backfill({}, day=DAY + 1)
        self.assertEqual(report['status'], 'BACKFILLED')  # 重载后自动补缺
        self.assertEqual(len(self.scheduled), 1)

    def test_three_rosters_union_same_code(self):
        """同票同时属于计划/持仓/观察：需求并集，不被后写名册覆盖。"""
        plan = dict(rows=[dict(code='300652', entry_rule=dict(
            history_days=[20261026, 20261027, 20261028, 20261029, 20261030, 20261102],
            turnover_reference=dict(windows=[1, 5])))])
        self.wl = {'positions': [dict(code='300652')],
                   'watch': [dict(code='300652')]}
        roster = P.full_roster({}, plan, DAY, CALENDAR)
        item = roster['300652']
        self.assertEqual(item['kind'], 'PLAN')  # 种类优先级
        self.assertEqual(item['history_days'],
                         [20261026, 20261027, 20261028, 20261029, 20261030,
                          20261102, 20261103])  # 计划6日 ∪ 持仓窗口
        self.assertEqual(item['turnover_reference'], dict(windows=[1, 5]))  # 换手不丢

    def test_on_plan_frozen_unions_three_rosters(self):
        export_dir = self.root / 'workspace' / 'data'
        export_dir.mkdir(parents=True)
        exported = export_dir / 'plan_frozen.json'
        exported.write_text(json.dumps(dict(
            sessions=[dict(candidates=[dict(code='300652')])])), encoding='utf-8')
        self.wl = {'positions': [dict(code='300652')], 'watch': [dict(code='300652')]}
        result = P.on_plan_frozen({}, exported, DAY)
        item = result['manifest']['items']['300652']
        self.assertEqual(item['kind'], 'PLAN')
        self.assertEqual(len(item['history_days']), 4)  # 导出默认窗口并入，不缩不回

    def test_cross_process_restart_acceptance(self):
        """真跨进程：父进程只落盘；独立子进程仅凭磁盘文件加载校验，
        证据丢失后自动触发增量补缺。"""
        import os
        import subprocess
        t1 = self._ready_target('p1.json', ['300652'])
        roster = {'300652': dict(kind='PLAN', history_days=[20261030], target=t1, status='READY')}
        P.refresh_manifest(DAY, DAY + 1, roster, [(['300652'], [20261030], t1)])
        child = self.root / 'child_check.py'
        child.write_text(_CHILD_SCRIPT, encoding='utf-8')
        env = {**os.environ, 'PYTHONIOENCODING': 'utf-8'}

        def run_child():
            proc = subprocess.run(
                [sys.executable, '-B', str(child), str(self.root), str(DAY + 1),
                 str(Path(__file__).resolve().parent)],
                capture_output=True, text=True, encoding='utf-8', timeout=120, env=env)
            line = next((l for l in (proc.stdout or '').splitlines()
                         if l.startswith('CHILD_REPORT:')), None)
            self.assertIsNotNone(line, (proc.stderr or '')[-500:])
            return json.loads(line[len('CHILD_REPORT:'):])

        self.assertEqual(run_child()['status'], 'READY')
        self.assertFalse((self.root / 'child_schedule.json').exists())  # 就绪不再准备
        Path(t1).unlink()  # 证据丢失
        report = run_child()
        self.assertEqual(report['status'], 'BACKFILLED')
        self.assertEqual(report['backlog'], ['300652'])
        marker = json.loads((self.root / 'child_schedule.json').read_bytes())
        self.assertEqual([r['code'] for r in marker['rows']], ['300652'])
        manifest = json.loads((self.risk / 'prepared_manifest.json').read_bytes())
        self.assertEqual(manifest['items']['300652']['status'], 'SCHEDULED')


_CHILD_SCRIPT = """\
import json, sys, types
from pathlib import Path
root = Path(sys.argv[1]); day = int(sys.argv[2])
sys.path.insert(0, sys.argv[3])  # stargazer_desktop（真实模块源码）
# 子进程内桩：仅代表外部服务（运行目录/调度worker/名单服务），
# 被测对象的全部输入均来自磁盘文件。
fake_rt = types.ModuleType('postclose_risk_runtime')
fake_rt.directory = lambda: root / 'postclose_risk'
fake_mh = types.ModuleType('minute_history_v1')
def fake_schedule(plan):
    (root / 'child_schedule.json').write_text(
        json.dumps(plan, ensure_ascii=False), encoding='utf-8')
    return str(root / 'postclose_risk' / 'child_target.json')
fake_mh.schedule = fake_schedule
fake_sv = types.ModuleType('services')
fake_sv.wl_load = lambda cfg: {'positions': [], 'watch': []}
sys.modules.update({'postclose_risk_runtime': fake_rt,
                    'minute_history_v1': fake_mh, 'services': fake_sv})
import postclose_prepared_v1 as P
report = P.verify_and_backfill({}, day=day)
print('CHILD_REPORT:' + json.dumps(report, ensure_ascii=False))
"""


if __name__ == '__main__':
    unittest.main()
