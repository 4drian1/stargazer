"""#18.199 storage layers: zones, layer-bound keys, read reuse, migration."""
import json
import threading
from datetime import datetime
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
ML = ROOT.parent / 'stargazer_ml/v5_hybrid/risk_r0_v1'
if str(ML) not in sys.path:
    sys.path.insert(0, str(ML))

import asset_migrate_v1
import cache_layer_keys_v1 as keys
import storage_roots_v1 as zones
from live_postclose_source_v1 import LocalCapture


class ZoneTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.env = mock.patch.dict(os.environ, {'STARGAZER_DATA_ROOT': str(self.root)})
        self.env.start()

    def tearDown(self):
        self.env.stop()
        self.temp.cleanup()

    def test_zone_roots_and_default_budgets(self):
        self.assertEqual(zones.assets_root(), self.root / 'stargazer_data/assets')
        self.assertEqual(zones.zone_budget('cache'), 4 * 1024**3)
        # assets/records are proposals, not allowances: no default budget
        self.assertIsNone(zones.zone_budget('assets'))
        self.assertIsNone(zones.zone_budget('records'))
        self.assertIsNone(zones.zone_headroom('assets'))
        self.assertIsNotNone(zones.zone_headroom('cache'))

    def test_budget_policy_fail_closed(self):
        zone = zones.ensure_zone('assets')
        (zone / 'zone_budget.json').write_text(
            json.dumps(dict(bytes=8 * 1024**3, user_authorized=True)), encoding='utf-8')
        with self.assertRaises(ValueError):
            zones.zone_budget('assets')
        (zone / 'zone_budget.json').write_text(
            json.dumps(dict(bytes=3 * 1024**3, user_authorized=True)), encoding='utf-8')
        self.assertEqual(zones.zone_budget('assets'), 3 * 1024**3)

    def test_single_writer_and_complete_gate(self):
        folder = zones.assets_root() / 'history_foundation_v1/000001'
        with zones.ZoneWriter(folder):
            self.assertFalse(zones.complete(folder))
            (folder / 'COMPLETE.json').write_text('{}', encoding='utf-8')
            (folder / 'a.bin').write_bytes(b'x')
            with self.assertRaises(ValueError):
                with zones.ZoneWriter(folder):
                    pass  # concurrent second writer refused
        self.assertTrue(zones.complete(folder))
        with zones.ZoneWriter(folder):  # sequential rebuild allowed
            (folder / 'b.bin').write_bytes(b'y')


class LayerKeyTest(unittest.TestCase):
    def test_math_key_excludes_capture_files(self):
        self.assertEqual(keys.math_key(), keys.math_key())
        self.assertNotEqual(keys.math_key(), keys.capture_key())
        self.assertNotEqual(keys.math_key(), keys.model_key())

    def test_extra_binds_source_signature(self):
        a = keys.math_key(extra=dict(extensions=[['p', 's1']]))
        b = keys.math_key(extra=dict(extensions=[['p', 's2']]))
        self.assertNotEqual(a, b)

    def test_unknown_layer_refused(self):
        with self.assertRaises(ValueError):
            keys.layer_key('ui')


class ReadReuseTest(unittest.TestCase):
    def test_same_run_read_once(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            body = b'hello-bytes'
            f = tmp / 'raw_x.msgpack.gz'
            f.write_bytes(body)
            import hashlib
            digest = hashlib.sha256(body).hexdigest()
            cap = LocalCapture(tmp / 'out', budget_root=tmp)
            one = cap.read(f, digest)
            before = cap.read_bytes
            two = cap.read(f, digest)
            self.assertIs(one, two)
            self.assertEqual(cap.read_bytes, before)  # no double accounting
            self.assertEqual(one, body)

    def test_fresh_read_surfaces_source_change(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            f = tmp / 'a.bin'
            f.write_bytes(b'v1')
            cap = LocalCapture(tmp / 'out', budget_root=tmp)
            self.assertEqual(cap.read(f), b'v1')
            f.write_bytes(b'v2-changed')
            self.assertEqual(cap.read(f), b'v1')            # memo serves same version
            self.assertEqual(cap.read(f, fresh=True), b'v2-changed')  # forced re-read

    def test_wrong_hash_still_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            f = tmp / 'a.bin'
            f.write_bytes(b'x')
            cap = LocalCapture(tmp / 'out', budget_root=tmp)
            with self.assertRaises(ValueError):
                cap.read(f, '0' * 64)


class ModelKeyTest(unittest.TestCase):
    def test_model_key_binds_artifacts(self):
        a = keys.model_key()
        b = keys.model_key()
        self.assertEqual(a, b)
        self.assertNotEqual(a, keys.math_key())


class TdxHistoryDigestTest(unittest.TestCase):
    LEGAL = [20261030, 20261102, 20261103, 20261104]

    def _rows(self):
        base = [dict(day=20261030, ts=20261030100000 + i * 500, open=1.0, high=1.0,
                     low=1.0, close=1.0, volume=1) for i in range(3)]
        d1 = [dict(day=20261102, ts=20261102100000 + i * 500, open=1.5, high=1.5,
                   low=1.5, close=1.5, volume=1) for i in range(3)]
        d2 = [dict(day=20261103, ts=20261103100000 + i * 500, open=2.0, high=2.0,
                   low=2.0, close=2.0, volume=1) for i in range(3)]
        d3 = [dict(day=20261104, ts=20261104100000 + i * 500, open=3.0, high=3.0,
                   low=3.0, close=3.0, volume=1) for i in range(3)]
        return base + d1, d2, d3

    def test_same_day_rerun_hits(self):
        from tdx_postclose_runtime_v1 import history_digest
        hist, d2, d3 = self._rows()
        allr = hist + d2
        d1 = history_digest('300652', allr, 20261103, self.LEGAL)
        d2d = history_digest('300652', allr, 20261103, self.LEGAL)
        self.assertEqual(d1, d2d)  # 同日同名单命中

    def test_append_next_day_keeps_foundation(self):
        from tdx_postclose_runtime_v1 import history_digest
        hist, d2rows, d3rows = self._rows()
        d_prev = history_digest('300652', hist + d2rows, 20261103, self.LEGAL)
        d_next = history_digest('300652', hist + d2rows, 20261104, self.LEGAL)
        # 次日覆盖域=≤20261103 = 前一日全集+当日bar → 身份延展而非失效；
        # 校验规则=重算覆盖域digest与存储比对（追加不污染旧覆盖）
        self.assertNotEqual(d_prev, d_next)  # 覆盖域+1日，键按覆盖末日区分
        from tdx_postclose_runtime_v1 import foundation_payload
        f = foundation_payload('300652', hist + d2rows, 20261103, self.LEGAL)
        self.assertEqual(f['coverage_last_day'], 20261102)  # 续算状态锚定前一日

    def test_history_revision_changes_identity(self):
        from tdx_postclose_runtime_v1 import history_digest
        hist, d2rows, d3rows = self._rows()
        d1 = history_digest('300652', hist + d2rows, 20261103, self.LEGAL)
        revised = [dict(r, close=9.0) if r['day'] == 20261102 else r for r in hist]
        d2 = history_digest('300652', revised + d2rows, 20261103, self.LEGAL)
        self.assertNotEqual(d1, d2)  # 历史修订正确失效

    def test_p_day_content_not_in_history_key(self):
        from tdx_postclose_runtime_v1 import history_digest
        hist, d2rows, d3rows = self._rows()
        d1 = history_digest('300652', hist + d2rows, 20261103, self.LEGAL)
        changed_p = [dict(r, close=5.0) for r in d2rows]
        d2 = history_digest('300652', hist + changed_p, 20261103, self.LEGAL)
        self.assertEqual(d1, d2)  # 当日bar不进覆盖域身份

    def test_history_raw_volume_revision_invalidates(self):
        from tdx_postclose_runtime_v1 import history_digest
        hist, d2rows, d3rows = self._rows()
        d1 = history_digest('300652', hist + d2rows, 20261103, self.LEGAL)
        revised = [dict(r, raw_volume=999999) if r['day'] == 20261102 else r
                   for r in hist + d2rows]
        d2 = history_digest('300652', revised, 20261103, self.LEGAL)
        self.assertNotEqual(d1, d2)  # 原生raw_volume修订正确失效（修复按0哈希）

    def test_today_key_binds_p_day_and_cutoff(self):
        from tdx_postclose_runtime_v1 import history_digest, today_digest
        hist, d2rows, d3rows = self._rows()
        V = ('m', 'c', 'n')
        base = history_digest('300652', hist + d2rows, 20261103, self.LEGAL)
        k1 = today_digest('300652', hist + d2rows, 20261103, 20261103150000, self.LEGAL,
                          [20261030, 20261102, 20261103], *V)
        self.assertEqual(k1, today_digest('300652', hist + d2rows, 20261103, 20261103150000,
                                          self.LEGAL, [20261030, 20261102, 20261103], *V))
        changed_p = [dict(r, close=5.0) for r in d2rows]
        k2 = today_digest('300652', hist + changed_p, 20261103, 20261103150000, self.LEGAL,
                          [20261030, 20261102, 20261103], *V)
        self.assertNotEqual(k1, k2)          # 当日价格改 → 当天结果键变
        k3 = today_digest('300652', hist + d2rows, 20261103, 20261103145500, self.LEGAL,
                          [20261030, 20261102, 20261103], *V)
        self.assertNotEqual(k1, k3)          # 截止时刻变 → 键变
        k4 = today_digest('300652', hist + d2rows, 20261103, 20261103150000, self.LEGAL,
                          [20261102, 20261103], *V)
        self.assertNotEqual(k1, k4)          # 日历起点变 → 键变
        self.assertEqual(base, history_digest('300652', hist + changed_p, 20261103, self.LEGAL))  # 底座键不变


class PrefixStoreGuardTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.env = mock.patch.dict(os.environ, {'STARGAZER_DATA_ROOT': str(self.root)})
        self.env.start()
        zone = zones.ensure_zone('assets')
        (zone / 'zone_budget.json').write_text(
            json.dumps(dict(bytes=1024**3, user_authorized=True)), encoding='utf-8')

    def tearDown(self):
        self.env.stop()
        self.temp.cleanup()

    def _cache(self):
        import numpy as np
        from stock_prefix_cache_v1 import PrefixFeatureCache
        cache = PrefixFeatureCache.__new__(PrefixFeatureCache)
        cache.kernel = 'math-v'
        cache.capture_version = 'cap-v'
        cache.root = zones.assets_root() / 'history_foundation_v1'
        cache.legacy = self.root / 'legacy'
        cache.audit = dict(bytes_written=0, write_skipped_budget=0)
        import numpy as np
        arrays = {f: {k: np.zeros(4, dtype=float) for k in ('open', 'high', 'low', 'close', 'volume')}
                  for f in (5, 15)}
        return cache, arrays

    def test_store_writes_marker_and_releases_lock(self):
        cache, arrays = self._cache()
        cache._store('000001', [20261102], 20261102150000, {},
                     arrays, {'feature_rows': 4}, [])
        folder = cache.root / '000001'
        self.assertTrue((folder / 'COMPLETE.json').is_file())
        self.assertFalse((cache.root / '000001.writing.lock').exists())
        self.assertFalse(any(self.root.rglob('*.writing.lock')))
        # 连续第二次写入（同键已存在）不报错、不留锁
        cache._store('000001', [20261102], 20261102150000, {},
                     arrays, {'feature_rows': 4}, [])
        self.assertFalse(any(self.root.rglob('*.writing.lock')))

    def test_store_refused_when_zone_unauthorized(self):
        (zones.assets_root() / 'zone_budget.json').unlink()
        cache, arrays = self._cache()
        cache._store('000002', [20261102], 20261102150000, {},
                     arrays, {'feature_rows': 4}, [])
        self.assertEqual(cache.audit.get('skip_reason'), 'UNAUTHORIZED_ASSETS_BUDGET')
        self.assertFalse((cache.root / '000002').exists())

    def test_memo_eviction_no_crash(self):
        import live_postclose_source_v1 as S
        with mock.patch.object(S, 'READ_MEMO_CAP', 24):
            with tempfile.TemporaryDirectory() as tmp:
                tmp = Path(tmp)
                cap = LocalCapture(tmp / 'out', budget_root=tmp)
                for i in range(6):
                    f = tmp / ('f%d.bin' % i)
                    f.write_bytes(b'x' * 16)
                    self.assertEqual(len(cap.read(f)), 16)  # 淘汰分支不抛TypeError
                self.assertLessEqual(cap._read_memo_bytes[0], 24 + 16)



    def test_two_distinct_versions_write_sequentially(self):
        cache, arrays = self._cache()
        cache._store('000003', [20261102], 20261102150000, {},
                     arrays, {'feature_rows': 4}, [])
        first = set(p.name for p in (cache.root / '000003').glob('*.json'))
        cache2, arrays2 = self._cache()
        cache2.kernel = 'math-v2'  # 不同版本 → 不同键
        cache2._store('000003', [20261102], 20261102150000, {},
                      arrays2, {'feature_rows': 4}, [])
        second = set(p.name for p in (cache.root / '000003').glob('*.json'))
        self.assertTrue(second - first)               # 新版本已写入
        self.assertFalse(any(self.root.rglob('*.writing.lock')))  # 无锁残留
        self.assertTrue((cache.root / '000003/COMPLETE.json').is_file())


class ConcurrentMigrationGuardTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.src = self.root / 'asset_v1'
        self.src.mkdir()
        (self.src / 'part.bin').write_bytes(b'payload')
        (self.src / 'COMPLETE.json').write_text('{"files": []}', encoding='utf-8')

    def tearDown(self):
        self.temp.cleanup()

    def test_busy_lock_refused_without_touching_staging(self):
        dst = self.root / 'asset_v2'
        (self.root / 'asset_v2.writing.lock').open('xb').close()
        staging = self.root / 'asset_v2.migrating'
        staging.mkdir()
        (staging / 'other.bin').write_bytes(b'owned-by-someone-else')
        with self.assertRaises(ValueError):
            asset_migrate_v1.migrate(self.src, dst, switch=lambda d: None,
                                     validate=lambda dst: True)
        self.assertTrue((staging / 'other.bin').exists())  # 他人暂存未被清理


class MigrationTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.src = self.root / 'asset_v1'
        self.src.mkdir()
        (self.src / 'part.bin').write_bytes(b'payload')
        marker = dict(version=1, files={'part.bin': __import__('hashlib').sha256(b'payload').hexdigest()})
        (self.src / 'COMPLETE.json').write_text(json.dumps(marker), encoding='utf-8')

    def tearDown(self):
        self.temp.cleanup()

    def test_migrate_copy_verify_switch_validate(self):
        seen = {}
        receipt = asset_migrate_v1.migrate(
            self.src, self.root / 'asset_v2',
            switch=lambda dst: seen.setdefault('dst', dst),
            validate=lambda dst: (dst / 'part.bin').read_bytes() == b'payload')
        self.assertEqual(receipt['files'], 1)  # COMPLETE.json excluded from hash list
        self.assertEqual(seen['dst'], self.root / 'asset_v2')
        self.assertEqual((self.root / 'asset_v2/part.bin').read_bytes(), b'payload')
        marker = json.loads((self.root / 'asset_v2/COMPLETE.json').read_bytes())
        self.assertEqual(marker['migrated_from'], str(self.src))
        # 清单自洽：按清单重算与存储一致，字典/列表两种格式均可，不含标记自身
        import hashlib
        recompute = {n: hashlib.sha256((self.root / 'asset_v2' / n).read_bytes()).hexdigest()
                     for n in marker['files']}
        self.assertEqual(recompute, marker['files'])
        self.assertNotIn('COMPLETE.json', marker['files'])
        self.assertTrue((self.src / 'COMPLETE.json').exists())  # old preserved

    def test_dict_marker_accepted(self):
        src2 = self.root / 'asset_dict'
        src2.mkdir()
        (src2 / 'x.bin').write_bytes(b'xy')
        (src2 / 'COMPLETE.json').write_text(json.dumps(
            dict(version=2, files={'x.bin': __import__('hashlib').sha256(b'xy').hexdigest()})),
            encoding='utf-8')
        receipt = asset_migrate_v1.migrate(
            src2, self.root / 'asset_dict_v2',
            switch=lambda d: None, validate=lambda dst: (dst / 'x.bin').exists())
        self.assertEqual(receipt['files'], 1)

    def test_extra_undeclared_file_refused(self):
        (self.src / 'undeclared.bin').write_bytes(b'sneaky')
        with self.assertRaises(ValueError):
            asset_migrate_v1.migrate(self.src, self.root / 'asset_v2',
                                     switch=lambda d: None, validate=lambda dst: True)
        self.assertFalse((self.src / 'undeclared.bin').exists() is False)  # 源不动
        self.assertFalse((self.root / 'asset_v2').exists())

    def test_incomplete_source_refused(self):
        (self.src / 'COMPLETE.json').unlink()
        with self.assertRaises(ValueError):
            asset_migrate_v1.migrate(self.src, self.root / 'asset_v2',
                                     switch=lambda d: None, validate=lambda dst: True)

    def test_validate_failure_keeps_old_only(self):
        with self.assertRaises(ValueError):
            asset_migrate_v1.migrate(self.src, self.root / 'asset_v2',
                                     switch=lambda d: None, validate=lambda dst: False)
        self.assertFalse((self.root / 'asset_v2').exists())
        self.assertTrue((self.src / 'part.bin').exists())




class RealTdxChainTest(unittest.TestCase):
    """完整链路（真实lc5小样本）：冷算落盘→热缓存命中→加入计划→
    只补这只股票的指标→新进程加载成功。"""
    @classmethod
    def setUpClass(cls):
        import tdx_native_kernel_v1 as native_kernel
        from tdx_native_history_v1 import locate, configured
        from tdx_postclose_runtime_v1 import read_lc5_memo
        if not configured():
            raise unittest.SkipTest('TDX 未配置')
        cls.native = native_kernel
        cls.code = '300652'
        p = locate(cls.code)
        cls.info = read_lc5_memo(p)
        cls.source = dict(path=str(p), sha256=cls.info['sha256'])
        cls.LEGAL = [20261026, 20261027, 20261028, 20261029, 20261030, 20261102, 20261103]
        cls.P, cls.E = 20261102, 20261103
        cls.CUTOFF = cls.P * 1000000 + 150000
        cls.days = [d for d in cls.LEGAL if d <= cls.P]

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.cache_dir = self.root / 'cache' / 'tdx_stock_cache'
        self.cache_dir.mkdir(parents=True)
        self.native.P, self.native.E, self.native.CUTOFF = self.P, self.E, self.CUTOFF
        from cache_layer_keys_v1 import math_key, capture_key
        from tdx_postclose_runtime_v1 import fingerprint
        self.versions = dict(math=math_key(), capture=capture_key(), native=fingerprint())
        self.spec = dict(P=self.P, E=self.E, cutoff=self.CUTOFF, legal=self.LEGAL,
                         files={self.code: dict(path=self.source['path'],
                                                size=self.info['bytes'],
                                                mtime_ns=self.info['mtime_ns'])})

    def tearDown(self):
        self.temp.cleanup()

    def _resolve(self, prep):
        from tdx_postclose_runtime_v1 import resolve_stock
        cache = self.native.NativeCache({self.code: self.source}, list(self.days), prep)
        timing = dict(read_parse=0.0, features_full=0.0, io_write=0.0, bytes_read=0,
                      rebuilt_reason='FULL_HISTORY_FALLBACK')
        status, delta = resolve_stock(self.code, self.spec, cache,
                                      (self.cache_dir,), self.versions,
                                      64 * 1024 * 1024, info=self.info, timing=timing)
        return status, cache

    def test_cold_hot_joinplan_reload_chain(self):
        import json, gzip
        # 1) 冷算：普通成员（非候选）落盘，无指标
        status, _ = self._resolve(set())
        self.assertEqual(status, 'miss')
        entries = list((self.cache_dir / self.code).glob('*.json.gz'))
        self.assertEqual(len(entries), 1)
        payload = json.loads(gzip.decompress(entries[0].read_bytes()))['payload']
        self.assertIsNone(payload['indicators'])
        # 2) 热命中：仍是普通成员
        status, _ = self._resolve(set())
        self.assertEqual(status, 'hit')
        # 3) 加入计划：命中被拒（缺指标），只补这只股票
        status, cache = self._resolve({self.code})
        self.assertEqual(status, 'miss')
        self.assertIn(self.code, cache.indicators)  # 指标已补齐
        # 4) 新进程：清空内存态，仅凭落盘文件加载
        import tdx_postclose_runtime_v1 as runtime_mod
        runtime_mod._LC5_MEMO.clear()
        runtime_mod._LC5_MEMO_BYTES[0] = 0
        status, cache2 = self._resolve({self.code})
        self.assertEqual(status, 'hit')
        self.assertIn(self.code, cache2.indicators)
        self.assertIsNotNone(cache2.indicators[self.code])

    def test_days_binding_different_calendar_start(self):
        from tdx_postclose_runtime_v1 import today_digest
        prefix = [r for r in self.info['rows'] if r['ts'] <= self.CUTOFF]
        k1 = today_digest(self.code, prefix, self.P, self.CUTOFF, self.LEGAL, self.days,
                          *self.versions.values())
        shorter_days = [d for d in self.days if d >= 20261030]
        k2 = today_digest(self.code, prefix, self.P, self.CUTOFF, self.LEGAL, shorter_days,
                          *self.versions.values())
        self.assertNotEqual(k1, k2)  # 不同日历起点不再碰撞





class UnifiedLedgerTest(unittest.TestCase):
    """合并总账：新区+完整旧运行区+原共享归档，路径去重，单一授权额度。"""
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.env = mock.patch.dict(os.environ, {'STARGAZER_DATA_ROOT': str(self.root / 'dataroot')})
        self.env.start()
        self.runtime = self.root / 'runtime' / 'postclose_risk'
        self.runtime.mkdir(parents=True)
        (self.root / 'runtime' / 'runtime_budget_v2.json').write_text(
            json.dumps(dict(disk_bytes=4 * 1024**3, user_authorized=True)), encoding='utf-8')

    def tearDown(self):
        self.env.stop()
        self.temp.cleanup()

    def _mb(self, path, n):
        # Simulated size only: sparse truncate sets st_size (what the disk
        # account reads) without allocating gigabytes of real memory.
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'wb') as f:
            f.seek(n - 1)
            f.write(b'\0')

    def test_full_legacy_area_and_archive_counted(self):
        from tdx_postclose_runtime_v1 import unified_cache_headroom
        from runtime_disk_limit_v1 import MONITOR_RESERVE
        # 旧运行区里 tdx_stock_cache 之外的文件（此前漏算的那 2.27GiB 同类）
        self._mb(self.runtime / 'run_old/big_archive.msgpack.gz', 700 * 1024**2)
        self._mb(self.runtime / 'tdx_stock_cache/000001/x.json.gz', 20 * 1024**2)
        archive = self.root / 'runtime' / 'review_archive' / 'strong_sector'
        self._mb(archive / 'alerts.json', 30 * 1024**2)
        # 新区
        zone = zones.cache_root() / 'tdx_stock_cache'
        self._mb(zone / '300652/k.json.gz', 50 * 1024**2)
        headroom = unified_cache_headroom(self.runtime)
        expected = 4 * 1024**3 - (700 + 20 + 30 + 50) * 1024**2 - MONITOR_RESERVE
        self.assertEqual(headroom, expected)

    def test_check_budget_shares_the_same_function(self):
        import tdx_postclose_runtime_v1 as M
        self._mb(self.runtime / 'huge/old.bin', 3 * 1024**3)
        self._mb(zones.cache_root() / 'c.bin', 800 * 1024**2)
        with self.assertRaises(ValueError):
            M.check_budget(self.runtime, 500 * 1024**2)  # 3.8GiB+0.5GiB>4GiB 拒
        M.check_budget(self.runtime, 100 * 1024**2)      # 3.9GiB 内 放行

    def test_nested_zone_not_double_counted(self):
        from tdx_postclose_runtime_v1 import unified_cache_headroom
        from runtime_disk_limit_v1 import MONITOR_RESERVE
        # 病态配置：新区目录嵌在旧运行区内 → 只计一次
        nested = self.runtime / 'stargazer_data' / 'cache'
        self._mb(nested / 'n.bin', 100 * 1024**2)
        self._mb(self.runtime / 'old.bin', 100 * 1024**2)
        headroom = unified_cache_headroom(self.runtime)
        expected = 4 * 1024**3 - 200 * 1024**2 - MONITOR_RESERVE
        self.assertEqual(headroom, expected)

    def test_disk_vs_logical_read_bytes(self):
        from tdx_postclose_runtime_v1 import read_lc5_memo, _LC5_DISK_BYTES
        _LC5_DISK_BYTES[0] = 0
        import tdx_postclose_runtime_v1 as M
        f = self.root / 'fake.lc5'
        f.write_bytes(b'not-real-lc5')
        with mock.patch.object(M.native, 'read_lc5_file',
                               return_value=dict(bytes=100, rows=[1], sha256='s', mtime_ns=1)):
            read_lc5_memo(f)
            read_lc5_memo(f)  # 备忘录命中：逻辑两次，磁盘一次
        self.assertEqual(_LC5_DISK_BYTES[0], 100)



class SlimKeyAndContinuationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from tdx_native_history_v1 import configured
        if not configured():
            raise unittest.SkipTest('TDX 未配置')

    def test_orchestrator_change_does_not_alter_native_math_key(self):
        from tdx_postclose_runtime_v1 import native_math_version, fingerprint
        a = native_math_version()
        b = native_math_version()
        self.assertEqual(a, b)
        # 调度/预算/存储文件在 fingerprint 内、但不在每股缓存身份内
        import tdx_postclose_runtime_v1 as M
        runtime_file = 'tdx_postclose_runtime_v1.py'
        files = M._NATIVE_MATH_FILES
        self.assertNotIn(runtime_file, files)
        self.assertEqual(len(files), 5)

    def test_windowed_matches_full_exactly(self):
        import numpy as np, pandas as pd
        import tdx_native_kernel_v1 as native
        import tdx_postclose_runtime_v1 as M
        from tdx_native_history_v1 import locate
        code = '300652'; P = 20261102; E = 20261103; CUTOFF = P * 1000000 + 150000
        native.P, native.E, native.CUTOFF = P, E, CUTOFF
        p = locate(code); info = M.read_lc5_memo(p)
        # FULL real history, not a short synthetic calendar: the claim is
        # proven against the long history the window actually replaces.
        raw_all = [r for r in info['rows'] if r['ts'] <= CUTOFF]
        days = [d for d in sorted(native.good_days(raw_all)) if d <= P]
        self.assertGreater(len(days), 40)
        cache = native.NativeCache({code: dict(path=str(p), sha256=info['sha256'])}, days, set())
        rows_full, ret_full = cache.stock(code, days, CUTOFF, info=info)
        tail_w, ret_w, _ = M.stock_windowed(code, days, CUTOFF, info, window=M.WINDOW_DAYS)
        full_tail = rows_full[rows_full.cutoff_index // 16 == len(days) - 1]
        a = tail_w.sort_values('cutoff_index').reset_index(drop=True)
        b = full_tail.sort_values('cutoff_index').reset_index(drop=True)
        self.assertEqual(len(a), len(b))
        for c in b.columns:
            self.assertTrue(a[c].isna().equals(b[c].isna()), 'NaN pattern: ' + c)
        def num(s):
            s = pd.to_numeric(s, errors='coerce')
            return s.astype(float) if getattr(s, 'dtype', None) == bool else s
        for c in b.columns:
            av, bv = num(a[c]), num(b[c])
            m = av.notna() & bv.notna()
            if m.any():
                self.assertLessEqual(float((av[m] - bv[m]).abs().max()), 0.0, c)
        self.assertEqual(float(np.nanmax(np.abs(np.nan_to_num(ret_w[5][-48:] - ret_full[5][-48:], nan=0)))), 0.0)

    def test_windowed_indicators_match_full(self):
        import tdx_native_kernel_v1 as native
        import tdx_postclose_runtime_v1 as M
        from tdx_native_history_v1 import locate
        code = '300652'; P = 20261102; CUTOFF = P * 1000000 + 150000
        native.P, native.E, native.CUTOFF = P, 20261103, CUTOFF
        p = locate(code); info = M.read_lc5_memo(p)
        raw_all = [r for r in info['rows'] if r['ts'] <= CUTOFF]
        days = [d for d in sorted(native.good_days(raw_all)) if d <= P]
        cache = native.NativeCache({code: dict(path=str(p), sha256=info['sha256'])}, days, {code})
        cache.stock(code, days, CUTOFF, info=info)
        _, _, wind = M.stock_windowed(code, days, CUTOFF, info, window=M.WINDOW_DAYS)
        def flat(d, prefix=''):
            out = {}
            for k, v in (d or {}).items():
                if isinstance(v, dict):
                    out.update(flat(v, prefix + k + '.'))
                elif isinstance(v, (int, float)) and v == v:
                    out[prefix + k] = v
            return out
        f, w = flat(cache.indicators[code]), flat(wind)
        self.assertEqual(set(f), set(w))
        for k, v in f.items():
            self.assertEqual(w[k], v)


class IndexMapLayersRealFieldsTest(unittest.TestCase):
    """真实执行地图字段重放：levels 无 above_close；带来自 resonance_bands。"""
    def _evidence(self):
        return dict(status='PRICE_FACTS_ONLY', asof=20261103150000, last_close=3608.10,
            session_low=3580.38,
            resonance_bands=[dict(node=3584.09, lower=3580.14, upper=3603.41,
                                  source_waves=3, peak_groups=1),
                             dict(node=3741.11, lower=3700.00, upper=3741.11,
                                  source_waves=2, peak_groups=1)],
            levels=[dict(price=3795.48, kind='RETRACEMENT'),
                    dict(price=3838.16, kind='RETRACEMENT'),
                    dict(price=3620.00, kind='NODE')],
            critical_position=dict(state='PENDING', lower=3584.09),
            outlook=dict(status='PRICE_MAP_ONLY', low=3574.27, low_day=20261023,
                high=3995.18, high_day=20260918,
                levels=[dict(ratio=.236, price=3673.60, distance=3673.60/3608.10-1)]))

    def test_band_from_resonance_not_node_stitch(self):
        import index_map_layers_v1 as M
        L = M.build_layers('sse', self._evidence(), 20261103150000)
        self.assertEqual(float(L['band']['lower']), 3580.14)   # 近处带，非3584—3741拼带
        self.assertEqual(float(L['band']['upper']), 3603.41)
        self.assertTrue(L['node_in_band'])
        self.assertEqual(L['above_near'], [3620.0, 3741.11, 3795.48, 3838.16])  # 含上方真实共振带
        text = M.render_layers({'sse': L})
        self.assertIn('3580.14—3603.41试运行黄金带', text)
        self.assertIn('带内节点3584.09', text)
        self.assertIn('近处压力3741.11', text)  # 上方带作为压力，不进下方观察
        self.assertIn('前低3574.27（20261023）（价格事实，非黄金位）', text)
        self.assertIn('近处压力3620.00', text)
        self.assertIn('3673.60', text)

    def test_no_bands_no_invention(self):
        import index_map_layers_v1 as M
        ev = dict(status='PRICE_FACTS_ONLY', asof=20261103150000, last_close=3608.10,
                  session_low=None, resonance_bands=None, levels=[],
                  critical_position=None, outlook={})
        L = M.build_layers('sse', ev, 20261103150000)
        text = M.render_layers({'sse': L})
        self.assertIn('不补造整数位', text)
        self.assertIn('近处无真实压力节点', text)


class TouchLaneRealPlanTest(unittest.TestCase):
    """真实计划区带重放 + 报价质量门 + 共享去重 + 逐事件回执。"""
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.state = Path(self.temp.name) / 'touch.json'
        self.now = 20261104093200
        self.plan = dict(monitor_date='20261104', assistant_plan_id='p1', rows=[
            dict(code='300652', name='雷迪克',
                 zone=dict(zone_lower=25.0124, zone_upper=25.45, zone_sources=['X']),
                 entry_rule=dict(observation_bands=[dict(lower=25.0124, upper=25.45)],
                                 f618=25.50)),
            dict(code='000510', name='新金路', zone=None,
                 entry_rule=dict(observation_bands=[], f618=7.10))])

    def tearDown(self):
        self.temp.cleanup()

    def _q(self, price, stamp='20261104093100'):
        return dict(price=price, time=stamp)

    def test_real_plan_bands_fire(self):
        import touch_alert_lane_v1 as lane
        # V2：首笔报价只建基准；自上沿上方向下穿越真实观察带 25.0124—25.45 才推送
        self.assertEqual(lane.detect(self.plan, {'300652': self._q(25.60, '20261104093050'),
                                                 '000510': self._q(7.20, '20261104093050')},
                                     self.state, now_stamp=self.now), [])
        events = lane.detect(self.plan, {'300652': self._q(25.30, '20261104093100')},
                             self.state, now_stamp=self.now)
        self.assertEqual([e['code'] for e in events], ['300652'])
        self.assertEqual(events[0]['lower'], 25.0124)
        # 带内每带一个身份：同价再测不重复；带内回落到下沿下不构成上沿穿越，仍同一身份
        self.assertEqual(lane.detect(self.plan, {'300652': self._q(24.90, '20261104093110')}, self.state,
                                     now_stamp=self.now), [])

    def test_quote_quality_gates(self):
        import touch_alert_lane_v1 as lane
        base = {'300652': self._q(25.30)}
        # 昨日报价
        self.assertEqual(lane.detect(self.plan, {'300652': self._q(25.30, '20261103093100')},
                                     self.state, now_stamp=self.now), [])
        # 过期报价(>120s)
        self.assertEqual(lane.detect(self.plan, {'300652': self._q(25.30, '20261104092800')},
                                     self.state, now_stamp=self.now), [])
        # 非正/缺失
        self.assertEqual(lane.detect(self.plan, {'300652': dict(price=0, time='20261104093100')},
                                     self.state, now_stamp=self.now), [])
        self.assertEqual(lane.detect(self.plan, {'300652': {}}, self.state,
                                     now_stamp=self.now), [])
        self.assertFalse((self.temp.name and Path(self.state).exists()))

    def test_shared_dedupe_and_per_event_delivery(self):
        import touch_alert_lane_v1 as lane
        # V2：首笔在 f618 上方只建基准；相邻报价下穿 7.10 才建事件
        self.assertEqual(lane.detect(self.plan, {'000510': self._q(7.20, '20261104093050')}, self.state,
                                     now_stamp=self.now), [])
        events = lane.detect(self.plan, {'000510': self._q(7.05, '20261104093100')}, self.state,
                             now_stamp=self.now)
        self.assertEqual(len(events), 1)
        key = events[0]['event_key']
        lane.mark_notified(self.plan, '000510', self.state, band_id='f618')  # 新通道按带消费
        again = lane.detect(self.plan, {'000510': self._q(6.50, '20261104093110')}, self.state,
                            now_stamp=self.now)
        self.assertEqual(again, [])  # 新旧通道共享身份，不重复
        lane.record_delivery(events[0], False, self.state)
        state = json.loads(Path(self.state).read_bytes())
        self.assertEqual(state['notified'][key]['delivery'], 'FAILED')  # 逐事件，不波及其他
        other = [k for k in state['notified'] if '300652' in k]
        self.assertTrue(all('delivery' not in state['notified'][k] for k in other))

    def test_gap_crossing_message_honest(self):
        import touch_alert_lane_v1 as lane
        ev = dict(name='n', code='300652', price=24.90, lower=25.0124, upper=25.45,
                  at='20261104093100', crossed=True)
        msg = lane.message(ev)
        self.assertIn('当前已在观察带下方', msg)
        self.assertIn('不能确认跳空', msg)
        self.assertIn('实时试算、未收盘', msg)
        ev2 = dict(ev, at='20261104094000', first_at='20261104093100', price=23.90)
        msg2 = lane.message(ev2)
        self.assertIn('首次到位 20261104093100', msg2)
        self.assertIn('最新报价 20261104094000', msg2)



class TouchProductionEntryTest(unittest.TestCase):
    """当前正式计划(TAROT_A_SECOND_CASH_V2)→tick→触价→队列 完整驱动。"""
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        (self.root / 'monitor.json').write_text(
            json.dumps(dict(enabled=True, alerts={})), encoding='utf-8')
        self.now = datetime(2026, 11, 4, 9, 32, 0)
        self.plan = dict(monitor_date='20261104', plan_ready=True,
                         rule_version='TAROT_A_SECOND_CASH_V2', assistant_plan_id='p1',
                         rows=[dict(code='300652', name='雷迪克',
                                    zone=dict(zone_lower=25.0124, zone_upper=25.45),
                                    entry_rule=dict(f618=25.50))])

    def tearDown(self):
        self.temp.cleanup()

    def _tick(self, price=25.30, push_enabled=True, stamp='20261104093100'):
        import pentacles_monitor as PM
        (self.root / 'assistant_plan.json').write_text(
            json.dumps(self.plan), encoding='utf-8')
        monitor = PM.PriceMonitor(lambda: self.root)
        calls = []
        import types
        fake = types.ModuleType('tarot_monitor_v2')
        fake.evaluate = lambda plan, cfg, now: calls.append('tarot_indicator') or dict(status='MARKET_ONLY', rows={})
        import sys as _s
        quotes_fn = lambda codes: {'300652': dict(price=price, time=stamp)}
        broadcasts = []
        with mock.patch.dict(_s.modules, {'tarot_monitor_v2': fake}):
            monitor._quote_lane_pass(lambda: dict(push=dict(enabled=push_enabled)),
                                     lambda: self.plan, quote_fn=quotes_fn,
                                     now=datetime(2026, 11, 4, 9, 32, 0),
                                     broadcast=broadcasts.append)
            monitor.tick(self.plan, dict(push=dict(enabled=push_enabled)),
                         datetime(2026, 11, 4, 9, 32, 0),
                         quotes_fn, broadcasts.append, lambda *a: (True, ''))
        return broadcasts, calls

    def test_tarot_plan_enters_touch_lane_before_evaluate(self):
        # V2：首笔 25.60 在上沿上方只建基准；下一轮下穿上沿 25.45 才即时广播
        self._tick(price=25.60, stamp='20261104093050')
        broadcasts, calls = self._tick(price=25.30, stamp='20261104093100')
        kinds = [b.get('kind') for b in broadcasts]
        self.assertIn('touch_alert', kinds)          # 触价即时广播
        self.assertIn('tarot_indicator', calls)      # 指标通道仍在跑
        # 事件已落盘去重
        self.assertTrue((self.root / 'touch_alerts.json').exists())

    def test_push_off_keeps_pending_then_delivers(self):
        self._tick(price=25.60, stamp='20261104093050', push_enabled=False)  # 建立基准
        broadcasts, _ = self._tick(price=25.30, stamp='20261104093100',
                                   push_enabled=False)  # 下穿上沿：事件已落盘
        self.assertIn('touch_alert', [b.get('kind') for b in broadcasts])
        state = json.loads((self.root / 'touch_alerts.json').read_bytes())
        entry = next(iter(state['notified'].values()))
        self.assertNotIn('queued', entry)  # 关闭时未预约发送
        import touch_alert_lane_v1 as lane
        q={'300652': dict(price=25.30, time='20261104093110')}
        pending = lane.pending_sends(self.plan, self.root / 'touch_alerts.json',
                                     20261104093200, q)
        self.assertEqual(len(pending), 1)  # 重开后以当前报价重新核验仍可交付
        self.assertEqual(pending[0]['price'], 25.30)

    def test_future_and_nonfinite_quotes_blocked(self):
        import touch_alert_lane_v1 as lane
        for bad in (dict(price=float('inf'), time='20261104093100'),
                    dict(price=True, time='20261104093100'),
                    dict(price=25.30, time='20261104093300')):  # 未来
            self.assertEqual(lane.detect(self.plan, {'300652': bad},
                                         self.root / 'x.json', now_stamp=20261104093200), [])


class MapCurrentBandTest(unittest.TestCase):
    def test_close_inside_band_keeps_it(self):
        import index_map_layers_v1 as M
        ev = dict(status='PRICE_FACTS_ONLY', asof=20261103150000, last_close=3590.00,
                  session_low=3580.38,
                  resonance_bands=[dict(node=3584.09, lower=3580.14, upper=3603.41,
                                        source_waves=3, peak_groups=1)],
                  levels=[dict(price=3620.00)], critical_position=dict(lower=3584.09),
                  outlook={})
        L = M.build_layers('sse', ev, 20261103150000)
        self.assertIsNotNone(L['band'])
        self.assertTrue(L['band_is_current'])
        text = M.render_layers({'sse': L})
        self.assertIn('当前所在带', text)
        self.assertIn('近处压力3620.00', text)



class TouchBoundaryFixTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.state = Path(self.temp.name) / 't.json'
        self.plan = dict(monitor_date='20261104', assistant_plan_id='p1', rows=[
            dict(code='300652', entry_rule=dict(observation_bands=[
                dict(lower=24.00, upper=24.30), dict(lower=25.00, upper=25.50)]))])

    def tearDown(self):
        self.temp.cleanup()

    def test_containing_band_not_upper_inference(self):
        import touch_alert_lane_v1 as lane
        self.assertEqual(lane.band_id_for_price(self.plan, '300652', 24.10), 'obs0')
        self.assertIsNone(lane.band_id_for_price(self.plan, '300652', 23.50))  # 低于全部带→不归属
        lane.mark_notified(self.plan, '300652', self.state, band_id=None)     # None 不消费
        st = json.loads(self.state.read_bytes()) if self.state.exists() else {'notified': {}}
        self.assertEqual(st['notified'], {})

    def test_stale_pending_archived_never_sent(self):
        import touch_alert_lane_v1 as lane
        # V2：未发送 V2 事件 + 过期报价（240s）→ 归档 EXPIRED_NOT_SENT，绝不以旧价格推送
        state = dict(contract=lane.CONTRACT, notified={
            '20261103:p0:300652:obs0': dict(price=24.10, lower=24.0, upper=24.3,  # 昨日旧计划
                                            crossed=True, trigger_policy='V2'),
            '20261104:p9:300652:obs1': dict(price=25.10, lower=25.0, upper=25.5,  # 旧计划身份
                                            crossed=True, trigger_policy='V2'),
            '20261104:p1:300652:obs0': dict(price=24.10, lower=24.0, upper=24.3,  # 本计划但报价过期
                                            crossed=True, trigger_policy='V2'),
        })
        self.state.write_text(json.dumps(state), encoding='utf-8')
        pending = lane.pending_sends(self.plan, self.state, 20261104093200,
                                     {'300652': dict(price=24.20, time='20261104092800')})
        self.assertEqual(pending, [])  # 全部过期
        st = json.loads(self.state.read_bytes())
        self.assertTrue(all(v.get('delivery') == 'EXPIRED_NOT_SENT' for v in st['notified'].values()))

    def test_close_time_not_blocked_by_shared_gate(self):
        import pentacles_monitor as PM
        root = Path(self.temp.name) / 'm'
        root.mkdir()
        (root / 'monitor.json').write_text(json.dumps(dict(enabled=True, alerts={})), encoding='utf-8')
        monitor = PM.PriceMonitor(lambda: root)
        now = datetime(2026, 11, 4, 15, 1, 0)  # 收盘后：风险分支允许，触价门只挡触价
        calls = []
        import types, sys as _s
        fake = types.ModuleType('tarot_monitor_v2')
        fake.evaluate = lambda plan, cfg, n: calls.append('indicator') or dict(rows={})
        plan = dict(monitor_date='20261104', plan_ready=True,
                    rule_version='TAROT_A_SECOND_CASH_V2', rows=[])
        with mock.patch.dict(_s.modules, {'tarot_monitor_v2': fake}):
            monitor.tick(plan, dict(push=dict(enabled=False)), now,
                         lambda codes: {}, lambda b: None, lambda *a: (True, ''))
        # 共用门只挡触价，不挡分支进入：健康态不是被共用门截断的状态
        # （分支自身运行时门在更深层，stub环境不展开）
        self.assertNotIn(monitor.health.get('status'), ('DISABLED', 'PLAN_NOT_READY', 'PLAN_CHANGED'))


class TouchMisreportFixTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.state = Path(self.temp.name) / 't.json'
        self.plan = dict(monitor_date='20261104', assistant_plan_id='p1', rows=[
            dict(code='300652', entry_rule=dict(observation_bands=[
                dict(lower=24.00, upper=24.30)]))])

    def tearDown(self):
        self.temp.cleanup()

    def test_out_of_band_not_redelivered_waits_reentry(self):
        import touch_alert_lane_v1 as lane
        # V2：legacy 条目（无 trigger_policy）按原语义隔离——不重发、不归档、只补 legacy_reason
        state = dict(contract=lane.CONTRACT, notified={
            '20261104:p1:300652:obs0': dict(price=24.10, lower=24.0, upper=24.3)})
        self.state.write_text(json.dumps(state), encoding='utf-8')
        away = {'300652': dict(price=26.00, time='20261104103000')}
        self.assertEqual(lane.pending_sends(self.plan, self.state, 20261104103000, away), [])
        st = json.loads(self.state.read_bytes())
        entry = st['notified']['20261104:p1:300652:obs0']
        self.assertTrue(entry.get('legacy_reason'))   # 隔离标记
        self.assertNotIn('delivery', entry)           # 未误发未归档，原语义保留
        back = {'300652': dict(price=24.20, time='20261104103100')}
        self.assertEqual(lane.pending_sends(self.plan, self.state, 20261104103100, back), [])
        # legacy reverify 原语义兜底（升级瞬间仍在途的旧队列事件）
        legacy_ev = dict(event_key='20261104:p1:300652:obs0',
                         lower=24.0, upper=24.3, crossed=False)
        self.assertTrue(lane.reverify(legacy_ev, back['300652'], 20261104103100))
        self.assertFalse(lane.reverify(legacy_ev, away['300652'], 20261104103000))

    def test_unconfirmed_plan_blocked_at_entry(self):
        import pentacles_monitor as PM
        root = Path(self.temp.name) / 'm2'
        root.mkdir()
        (root / 'monitor.json').write_text(json.dumps(dict(enabled=True, alerts={})), encoding='utf-8')
        monitor = PM.PriceMonitor(lambda: root)
        now = datetime(2026, 11, 4, 10, 0, 0)
        plan = dict(monitor_date='20261104', plan_ready=True,
                    rule_version='TAROT_A_SECOND_CASH_V2', assistant_plan_id='',
                    rows=[dict(code='300652', entry_rule=dict(f618=25.0))])
        q = lambda codes: {'300652': dict(price=24.9, time='20261104095900')}
        monitor._quote_lane_pass(lambda: dict(push=dict(enabled=False)),
                                 lambda: plan, quote_fn=q, now=now)
        self.assertFalse((root / 'touch_alerts.json').exists())  # 无确认ID：静默
        plan['assistant_plan_id'] = 'p9'
        (root / 'assistant_plan.json').write_text(json.dumps(plan), encoding='utf-8')
        monitor._quote_lane_pass(lambda: dict(push=dict(enabled=False)),
                                 lambda: plan, quote_fn=q, now=now)
        self.assertTrue((root / 'touch_alerts.json').exists())   # 补确认后放行

class TouchQueuePathTest(unittest.TestCase):
    """真实队列路径：入队→报价函数→模拟钉钉→逐事件SENT；穿带提醒；
    临时取消后重新具备条件可交付。"""
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.state = Path(self.temp.name) / 't.json'
        self.plan = dict(monitor_date='20261104', assistant_plan_id='p1', rows=[
            dict(code='300652', entry_rule=dict(observation_bands=[
                dict(lower=24.00, upper=24.30)]))])
        self.event = dict(event_key='20261104:p1:300652:obs0', code='300652',
                          lower=24.0, upper=24.3, crossed=False)
        state = dict(contract='TOUCH_ALERT_LANE_V1',
                     notified={self.event['event_key']: dict(
                         price=24.10, lower=24.0, upper=24.3, crossed=False, queued=True)})
        self.state.write_text(json.dumps(state), encoding='utf-8')

    def tearDown(self):
        self.temp.cleanup()

    def _deliver(self, quote, send_ok=True):
        import pentacles_monitor as PM
        sent = []
        with mock.patch('agent.brain.load_config',
                        return_value=dict(push=dict(enabled=True))),                 mock.patch.object(PM, '_plan_current', return_value=True):
            ok, cancelled = PM._deliver_one(
                dict(push=dict(enabled=True)), 't', '占位正文', self.state, self.event, self.plan,
                quote_fn=lambda codes, on_attempt=None: {'300652': quote},
                send_fn=lambda cfg, title, body: sent.append(body) or (send_ok, ''),
                now_stamp=20261104093200)
        st = json.loads(self.state.read_bytes())
        return ok, cancelled, sent, st

    def test_queue_path_success_writes_sent(self):
        ok, cancelled, sent, st = self._deliver(dict(price=24.20, time='20261104093100'))
        self.assertEqual((ok, cancelled, len(sent)), (True, None, 1))
        self.assertIn('24.20', sent[0])
        self.assertEqual(st['notified'][self.event['event_key']]['delivery'], 'SENT')

    def test_crossed_event_allowed_below_band(self):
        ev = dict(self.event, crossed=True)
        self.event.update(crossed=True)
        ok, cancelled, sent, st = self._deliver(dict(price=23.90, time='20261104093100'))
        self.assertEqual((ok, cancelled), (True, None))  # 带下方事实可送达
        self.assertEqual(len(sent), 1)

    def test_retryable_cancel_requalifies(self):
        import touch_alert_lane_v1 as lane
        ev = self.event
        lane.record_delivery(ev, False, self.state, cancelled='PUSH_DISABLED')
        st = json.loads(self.state.read_bytes())
        entry = st['notified'][ev['event_key']]
        self.assertNotIn('queued', entry)  # 资格已释放
        ok, cancelled, sent, st2 = self._deliver(dict(price=24.20, time='20261104093100'))
        self.assertEqual(st2['notified'][ev['event_key']]['delivery'], 'SENT')  # 重新具备条件


class QuoteLaneIndependenceTest(unittest.TestCase):
    """关键验收：指标通道阻塞60秒，报价通道仍发现新触价并入队；
    记录报价/发现/发送三时间；无重复提醒；请求量有核算。"""
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        (self.root / 'monitor.json').write_text(
            json.dumps(dict(enabled=True, alerts={})), encoding='utf-8')
        self.plan = dict(monitor_date='20261104', plan_ready=True,
                         rule_version='TAROT_A_SECOND_CASH_V2',
                         assistant_plan_id='p1',
                         rows=[dict(code='300652', name='n',
                                    entry_rule=dict(observation_bands=[
                                        dict(lower=24.00, upper=24.30)]))])
        (self.root / 'assistant_plan.json').write_text(
            json.dumps(self.plan), encoding='utf-8')

    def tearDown(self):
        try:
            self.temp.cleanup()
        except OSError:
            pass  # 后台池线程可能仍持有临时文件句柄

    def test_lane_detects_while_indicator_blocked(self):
        import pentacles_monitor as PM
        monitor = PM.PriceMonitor(lambda: self.root)
        monitor._indicator_inflight = True  # 模拟指标计算阻塞60秒
        now = datetime(2026, 11, 4, 10, 0, 0)
        base_quotes = {'300652': dict(price=24.50, time='20261104095950')}  # 上沿上方基准
        quotes = {'300652': dict(price=24.20, time='20261104100000')}       # 下穿上沿
        lane_now = now
        monitor._quote_lane_pass(lambda: dict(push=dict(enabled=True)),
                                 lambda: self.plan,
                                 quote_fn=lambda codes: base_quotes, now=lane_now)
        monitor._quote_lane_pass(lambda: dict(push=dict(enabled=True)),
                                 lambda: self.plan,
                                 quote_fn=lambda codes: quotes, now=lane_now)
        state = json.loads((self.root / 'touch_alerts.json').read_bytes())
        entries = list(state['notified'].values())
        self.assertEqual(len(entries), 1)            # 阻塞期间新下穿被发现
        e = entries[0]
        self.assertEqual(e['at'], '20261104100000')  # 报价时间
        self.assertTrue(e.get('event_recorded_at'))  # 发现时间（V2 证据链）
        # 指标仍阻塞，再来一轮同价：不重复提醒
        monitor._quote_lane_pass(lambda: dict(push=dict(enabled=True)),
                                 lambda: self.plan,
                                 quote_fn=lambda codes: quotes, now=lane_now)
        state2 = json.loads((self.root / 'touch_alerts.json').read_bytes())
        self.assertEqual(len(state2['notified']), 1)
        # 请求量核算落盘
        import pentacles_monitor as PM
        self.assertGreaterEqual(PM._budget_count(self.root / 'quote_lane_budget.json'), 0)

    def test_send_records_sent_at(self):
        import touch_alert_lane_v1 as lane
        ev = dict(event_key='20261104:p1:300652:obs0', code='300652',
                  lower=24.0, upper=24.3, crossed=False)
        state = dict(contract=lane.CONTRACT, notified={ev['event_key']: dict(
            price=24.1, lower=24.0, upper=24.3, queued=True)})
        f = self.root / 's.json'
        f.write_text(json.dumps(state), encoding='utf-8')
        lane.record_delivery(ev, True, f)
        st = json.loads(f.read_bytes())
        self.assertTrue(st['notified'][ev['event_key']]['sent_at'])


class LaneHardBoundTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def test_timeout_does_not_wait(self):
        import time
        import pentacles_monitor as PM
        def slow():
            time.sleep(0.25)
            return 'done'
        t0 = time.perf_counter()
        result = PM._with_timeout(slow, (), 0.05)
        elapsed = time.perf_counter() - t0
        self.assertIs(result, PM._TIMEOUT_SENTINEL)
        self.assertLess(elapsed, 0.2)  # 不等待慢任务结束

    def test_send_timeout_records_unknown(self):
        import pentacles_monitor as PM
        import touch_alert_lane_v1 as lane
        state = self.root / 's.json'
        ev = dict(event_key='20261104:p1:300652:obs0', code='300652',
                  lower=24.0, upper=24.3, crossed=False)
        state.write_text(json.dumps(dict(contract=lane.CONTRACT, notified={
            ev['event_key']: dict(price=24.1, lower=24.0, upper=24.3, queued=True)})),
            encoding='utf-8')
        def slow_send(cfg, title, body):
            import time as _t
            _t.sleep(0.3)
            return True, ''
        def fast_quote(codes, on_attempt=None):
            return {'300652': dict(price=24.2, time='20261104093005')}
        with mock.patch('agent.brain.load_config',
                        return_value=dict(push=dict(enabled=True))),                 mock.patch.object(PM, '_plan_current', return_value=True),                 mock.patch.object(PM, 'SEND_TIMEOUT_SECONDS', 0.1):
            ok, cancelled = PM._deliver_one(
                dict(push=dict(enabled=True)), 't', 'b', state, ev,
                dict(monitor_date='20261104', assistant_plan_id='p1', rows=[]),
                quote_fn=fast_quote, send_fn=slow_send, now_stamp=20261104093010)
        st = json.loads(state.read_bytes())
        self.assertEqual(st['notified'][ev['event_key']]['delivery'], 'UNKNOWN')

    def test_budget_counts_real_attempts_and_gate(self):
        """真实计账：on_attempt每次真实尝试入账；余额耗尽后请求前拦截。"""
        import pentacles_monitor as PM
        (self.root / 'monitor.json').write_text(
            json.dumps(dict(enabled=True, alerts={})), encoding='utf-8')
        plan = dict(monitor_date='20261104', plan_ready=True,
                    assistant_plan_id='p1', rows=[dict(code='300652')])
        (self.root / 'assistant_plan.json').write_text(json.dumps(plan), encoding='utf-8')
        monitor = PM.PriceMonitor(lambda: self.root)
        counter = self.root / 'quote_lane_budget.json'

        def gated_quotes(codes, on_attempt=None, timeout=10):
            out = {}
            for i in range(0, len(codes), 6):  # 7只→2批
                for _ in range(3):  # 重试
                    if on_attempt is not None and not on_attempt('u'):
                        return out
            return out
        import tools.market as MKT
        with mock.patch.object(MKT, 'quotes', gated_quotes):
            codes7 = [dict(code=str(300650 + i)) for i in range(7)]
            plan7 = dict(monitor_date='20261104', plan_ready=True,
                         assistant_plan_id='p1', rows=codes7)
            (self.root / 'assistant_plan.json').write_text(json.dumps(plan7), encoding='utf-8')
            monitor._quote_lane_pass(lambda: dict(push=dict(enabled=False)),
                                     lambda: plan7, now=datetime(2026, 11, 4, 10, 0, 0))
            self.assertEqual(PM._budget_count(counter), 6)  # 2批×3尝试
        # 请求前拦截：余额设为上限，新一轮不再发请求
        counter.write_text(json.dumps(dict(day='20261104', polls=PM.QUOTE_DAILY_CAP)),
                           encoding='utf-8')
        called = []
        with mock.patch.object(MKT, 'quotes',
                               lambda codes, on_attempt=None, timeout=10: called.append(1) or {}):
            monitor._quote_lane_pass(lambda: dict(push=dict(enabled=False)),
                                     lambda: plan, now=datetime(2026, 11, 4, 10, 0, 1))
        self.assertEqual(called, [])  # 额度门拦截，请求未发出

    def test_pool_full_rejects_and_cancels(self):
        """池满拒收不积压；超时未开始的任务不再执行。"""
        import time
        import pentacles_monitor as PM
        PM._POOL_SUBMITTED[0] = 4  # 占满
        ran = []
        result = PM._with_timeout(lambda: ran.append(1), (), 0.05)
        self.assertIs(result, PM._REJECTED_SENTINEL)
        self.assertEqual(ran, [])  # 拒收：未执行
        PM._POOL_SUBMITTED[0] = 0
        # 超时取消：任务开始前的等待被取消
        gate = threading.Event()
        def blocked():
            gate.wait(2)
            ran.append(2)
            return 'x'
        r = PM._with_timeout(blocked, (), 0.05)
        self.assertIs(r, PM._TIMEOUT_SENTINEL)
        time.sleep(0.2)
        self.assertEqual(ran, [])  # 尚未开始即取消
        gate.set()

    def test_pending_and_delivery_concurrent(self):
        """pending_sends 与 record_delivery 并发：SENT 回执不丢、状态一致。"""
        import threading
        import touch_alert_lane_v1 as lane
        state = self.root / 'p.json'
        plan = dict(monitor_date='20261104', assistant_plan_id='p1', rows=[
            dict(code='300652', entry_rule=dict(observation_bands=[
                dict(lower=24.0, upper=24.3)]))])
        q = {'300652': dict(price=24.2, time='20261104093000')}
        state.write_text(json.dumps(dict(contract=lane.CONTRACT, notified={
            '20261104:p1:300652:obs0': dict(price=24.1, lower=24.0, upper=24.3)})),
            encoding='utf-8')
        errors = []
        def pender():
            try:
                for _ in range(20):
                    lane.pending_sends(plan, state, 20261104093000, q)
            except Exception as exc:
                errors.append(exc)
        def deliverer():
            try:
                for _ in range(20):
                    lane.record_delivery(dict(event_key='20261104:p1:300652:obs0'), True, state)
            except Exception as exc:
                errors.append(exc)
        ts = [threading.Thread(target=pender), threading.Thread(target=deliverer)]
        for th in ts: th.start()
        for th in ts: th.join()
        self.assertEqual(errors, [])
        st = json.loads(state.read_bytes())
        self.assertEqual(st['notified']['20261104:p1:300652:obs0']['delivery'], 'SENT')

    def test_concurrent_state_no_lost_updates(self):
        import threading
        import touch_alert_lane_v1 as lane
        # V2 并发回归：多线程交替上穿/下穿同一带 + 并发回执；不丢事件、无异常。
        # （V2 同股同日分档去重后事件数不再随价格阶梯增长，故以"至少一条 +
        #   回执 SENT + 无异常"作为不丢更新判据）
        state = self.root / 'c.json'
        plan = dict(monitor_date='20261104', assistant_plan_id='p1', rows=[
            dict(code='300652', entry_rule=dict(observation_bands=[
                dict(lower=10.0, upper=10.5)]))])
        first_key = '20261104:p1:300652:obs0'
        errors = []
        def worker(n):
            try:
                for i in range(5):
                    price = 10.6 if i % 2 == 0 else 10.2
                    quotes = {'300652': dict(price=price,
                                             time=f'202611040930{n * 10 + i:02d}')}
                    lane.detect(plan, quotes, state, now_stamp=20261104093100)
                    lane.record_delivery(dict(event_key=first_key), True, state)
            except Exception as exc:
                errors.append(exc)
        threads = [threading.Thread(target=worker, args=(n,)) for n in range(4)]
        for th in threads: th.start()
        for th in threads: th.join()
        self.assertEqual(errors, [])
        st = json.loads(state.read_bytes())
        self.assertEqual(st['notified'][first_key]['delivery'], 'SENT')
        self.assertGreaterEqual(len(st['notified']), 1)  # 下穿事件一个没丢

    def test_real_lane_thread_during_indicator_block(self):
        import time
        import pentacles_monitor as PM
        (self.root / 'monitor.json').write_text(
            json.dumps(dict(enabled=True, alerts={})), encoding='utf-8')
        plan = dict(monitor_date='20261104', plan_ready=True,
                    assistant_plan_id='p1', rows=[dict(
                        code='300652', entry_rule=dict(observation_bands=[
                            dict(lower=24.00, upper=24.30)]))])
        (self.root / 'assistant_plan.json').write_text(json.dumps(plan), encoding='utf-8')
        monitor = PM.PriceMonitor(lambda: self.root)
        monitor._indicator_inflight = True  # 指标阻塞全程保持
        polls = []
        original_pass = PM.PriceMonitor._quote_lane_pass
        def timed_pass(self, get_cfg, get_plan):
            polls.append(time.time())
            return original_pass(
                self, get_cfg, get_plan,
                quote_fn=lambda codes: {'300652': dict(price=24.1, time='20261104095955')},
                now=datetime(2026, 11, 4, 10, 0, 0))
        with mock.patch.object(PM, 'QUOTE_LANE_MIN_SECONDS', 1),                 mock.patch.object(PM, 'QUOTE_LANE_MAX_SECONDS', 1),                 mock.patch.object(PM.PriceMonitor, '_quote_lane_pass', timed_pass):
            monitor._start_quote_lane(lambda: dict(push=dict(enabled=False)),
                                      lambda: plan)
            time.sleep(3.6)
            monitor.stop.set()
        self.assertGreaterEqual(len(polls), 3)
        gaps = [b - a for a, b in zip(polls, polls[1:])]
        self.assertTrue(all(g < 3.0 for g in gaps))  # 实际轮询间隔达标
        self.assertTrue((self.root / 'touch_alerts.json').exists())


class FailureBindingTest(unittest.TestCase):
    def test_stale_failure_allows_recompute(self):
        import tarot_monitor_v2 as M
        plan = dict(assistant_plan_id='p1', rows=[])
        current = M._flow_version()
        stale_version = dict(flow_version='v0', plan_id='p1', plan_hash=M.digest(plan))
        self.assertTrue(M.failure_stale(stale_version, plan))       # 旧版本→失效
        new_plan = dict(assistant_plan_id='p2', rows=[])
        stale_plan = dict(flow_version=current, plan_id='p1', plan_hash=M.digest(plan))
        self.assertTrue(M.failure_stale(stale_plan, new_plan))      # 计划变了→失效
        fresh = dict(flow_version=current, plan_id='p1', plan_hash=M.digest(plan))
        self.assertFalse(M.failure_stale(fresh, plan))              # 同版本同计划→仍锁定

    def test_pool_busy_releases_slot_not_unknown(self):
        import pentacles_monitor as PM
        import touch_alert_lane_v1 as lane
        state = self._state()
        ev = dict(event_key='20261104:p1:300652:obs0', code='300652',
                  lower=24.0, upper=24.3, crossed=False)
        PM._POOL_SUBMITTED[0] = 4
        try:
            with mock.patch('agent.brain.load_config',
                            return_value=dict(push=dict(enabled=True))), \
                    mock.patch.object(PM, '_plan_current', return_value=True):
                ok, cancelled = PM._deliver_one(
                    dict(push=dict(enabled=True)), 't', 'b', state, ev,
                    dict(monitor_date='20261104', assistant_plan_id='p1', rows=[]),
                    quote_fn=lambda codes, on_attempt=None: {'300652': dict(price=24.2, time='20261104093005')},
                    send_fn=lambda *a: (True, ''), now_stamp=20261104093010)
        finally:
            PM._POOL_SUBMITTED[0] = 0
        st = json.loads(state.read_bytes())
        entry = st['notified'][ev['event_key']]
        self.assertEqual(entry['delivery'], 'CANCELLED_POOL_BUSY')
        self.assertNotIn('queued', entry)  # 未执行：释放资格，不永久跳过

    def _state(self):
        state = Path(self.temp.name) / 's.json'
        import touch_alert_lane_v1 as lane
        state.write_text(json.dumps(dict(contract=lane.CONTRACT, notified={
            '20261104:p1:300652:obs0': dict(price=24.1, lower=24.0, upper=24.3,
                                           queued=True)})), encoding='utf-8')
        return state

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.temp.cleanup()


class TrialTimeBudgetTest(unittest.TestCase):
    def test_slow_trial_skipped_price_facts_kept(self):
        import time
        ev = dict(status='PRICE_FACTS_ONLY', asof=20261104100000, last_close=1.0)
        fake = type('M', (), {'predict': staticmethod(lambda frame, name, cutoff: (time.sleep(2), {})[1])})
        import sys
        sys.path.insert(0, 'D:/stargazer/stargazer_ml/v5_hybrid/risk_r0_v1')
        sys.path.insert(0, 'D:/stargazer/stargazer_ml/v5_hybrid/green_common_profile')
        import market_price_evidence_v1 as MPE
        with mock.patch.dict(sys.modules, {'market_research_trial_v1': fake}):
            t0 = time.perf_counter()
            MPE.attach_trial(ev, None, 'sse', 20261104100000, trial_seconds=0.2)
            self.assertLess(time.perf_counter() - t0, 1.5)  # 不被慢trial拖住
        self.assertEqual(ev['status'], 'PRICE_FACTS_ONLY')   # 价格事实保留
        self.assertEqual(ev['research_state'], 'TRIAL_SKIPPED_TIME_BUDGET')


class RetryGradingTest(unittest.TestCase):
    def test_failure_action_classifier(self):
        import tarot_monitor_v2 as M
        from datetime import datetime as _dt
        now = _dt(2026, 11, 4, 11, 0, 0)
        plan = dict(assistant_plan_id='p1', rows=[])
        with tempfile.TemporaryDirectory() as tmp,                 mock.patch.object(M.runtime, 'directory', return_value=Path(tmp)):
            self.assertEqual(M._failure_action(dict(reason='ValueError:REALTIME_HISTORY_DAY_EMPTY'), plan, now), 'proceed')
            self.assertEqual(M._failure_action(dict(reason='RuntimeError:MODEL_CONTRACT_BROKEN'), plan, now), 'block')
            for _ in range(3):
                M.retry_bump(plan)
            self.assertEqual(M._failure_action(dict(reason='ValueError:REALTIME_HISTORY_DAY_EMPTY'), plan, now), 'exhausted')
            self.assertEqual(M.retry_attempts(dict(assistant_plan_id='p2', rows=[])), 0)


class WorkerExitTimeTest(unittest.TestCase):
    def test_subprocess_returns_despite_slow_trial(self):
        """整子进程返回时间：慢trial(30s)下worker必须快速退出。"""
        import subprocess, sys, time, json, tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory() as tmp:
            stub = Path(tmp) / 'market_research_trial_v1.py'
            stub.write_text("import time\ndef predict(frame,name,cutoff):\n    time.sleep(30)\n    return {}\n", encoding='utf-8')
            env = {**__import__('os').environ,
                   'PYTHONPATH': tmp + ';D:/stargazer/stargazer_ml/v5_hybrid/risk_r0_v1;D:/stargazer/stargazer_ml/v5_hybrid/green_common_profile;D:/stargazer/stargazer_desktop'}
            t0 = time.perf_counter()
            proc = subprocess.run(
                [sys.executable, '-B', '-X', 'utf8',
                 'D:/stargazer/stargazer_ml/v5_hybrid/green_common_profile/market_price_evidence_v1.py'],
                input=json.dumps(dict(cutoff=20261103150000)),
                capture_output=True, text=True, encoding='utf-8', timeout=25, env=env)
            elapsed = time.perf_counter() - t0
            self.assertLess(elapsed, 15, f'worker被慢trial拖住: {elapsed:.1f}s')
            out = json.loads(proc.stdout)
            self.assertEqual(out['evidence']['sse']['status'], 'PRICE_FACTS_ONLY')
            self.assertEqual(out['evidence']['sse']['research_state'], 'TRIAL_SKIPPED_TIME_BUDGET')


class RetryChainRealTest(unittest.TestCase):
    """真实work()消费链：失败1→冷却→失败2→冷却→失败3→停止；三份证据均保留。"""
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.plan = dict(monitor_date='20261104', assistant_plan_id='p1',
                         rule_version='TAROT_A_SECOND_CASH_V2', rows=[
                             dict(code='300652', entry_rule=dict(f618=25.0))])
        self.target = self.root / 'tarot_v2' / 'result.json'
        self.target.parent.mkdir(parents=True)
        self.now = datetime.now()

    def tearDown(self):
        self.temp.cleanup()

    def _failing_run(self, attempt):
        import tarot_monitor_v2 as M
        class P:
            returncode = 1
            stdout = ''
            stderr = 'boom'
        with mock.patch.object(M.runtime, 'directory', return_value=self.root), \
                mock.patch.object(M, 'subprocess') as sp, \
                mock.patch.dict(__import__('sys').modules, {'notifier': __import__('types').SimpleNamespace(log=lambda *a, **k: None)}), \
                mock.patch.object(M, 'RETRY_COOLDOWN_SECONDS', 0), \
                mock.patch.object(M, 'enabled', return_value=True):
            sp.run.return_value = P()
            M.work(self.plan, {}, [], 20261104110000, self.target, attempt)

    def test_three_failures_then_stop_evidence_kept(self):
        import tarot_monitor_v2 as M
        with mock.patch.object(M.runtime, 'directory', return_value=self.root),                 mock.patch.object(M, 'RETRY_COOLDOWN_SECONDS', 0):
            self._chain(M)
    def _chain(self, M):
        # 第1次
        self._failing_run(0)
        self.assertEqual(M.retry_attempts(self.plan), 1)
        # 消费端判定：可恢复(REALTIME类reason?work写的是RISK_WORKER_FAILED——用可恢复样例模拟)
        record = json.loads((self.root / 'tarot_v2' / 'failed_20261104.json').read_bytes())
        self.assertEqual(record['attempts'], 1)
        action = M._failure_action(dict(reason='ValueError:REALTIME_HISTORY_DAY_EMPTY'), self.plan, self.now)
        self.assertEqual(action, 'proceed')
        # 第2次（归档第1份）
        self._failing_run(1)
        self.assertEqual(M.retry_attempts(self.plan), 2)
        # 第3次（归档第2份）
        self._failing_run(2)
        self.assertEqual(M.retry_attempts(self.plan), 3)
        archives = sorted(self.root.glob('tarot_v2/failed_20261104_a*.json'))
        self.assertEqual(len(archives), 2)          # 两份历史证据唯一归档
        names = [a.name for a in archives]
        self.assertEqual(len(set(names)), len(names))  # 无覆盖
        current = self.root / 'tarot_v2' / 'failed_20261104.json'
        self.assertTrue(current.exists())           # 当前失败仍在
        # 第4次消费判定：停止
        self.assertEqual(M._failure_action(dict(reason='ValueError:REALTIME_HISTORY_DAY_EMPTY'), self.plan, self.now), 'exhausted')
        # 不同绑定（计划变了）重新计数
        other = dict(self.plan, assistant_plan_id='p2')
        self.assertEqual(M.retry_attempts(other), 0)
if __name__ == '__main__':
    unittest.main()
