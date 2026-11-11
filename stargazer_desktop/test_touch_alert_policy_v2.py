"""Touch Alert Policy V2 单元测试（离线，全部状态文件写在 D 盘临时目录）。

覆盖任务书 19 项测试 + V2.1 破位事件 (测试20-24) + 603703 回放 + 抑制有界
+ 三态文案 + 健康统计。
核心语义：
  A. 真实向下穿越（相邻两次有效观察价，prev > upper 且 cur <= upper）；
  B. 同股同日分档去重（交易日+代码，跨 plan_id；SENT/UNKNOWN/consumed 计已通知，
     FAILED 保留证据不阻断；queued/可重试 CANCELLED 阻断其他档位）；
  C. 同轮多档只推最深一档；
  D. 带身份保持 zone/obsN/f618；
  E. reverify/message 新语义；旧事件隔离（legacy_reason，不重发）;
  G. (V2.1) 首次有效观察即破位（开盘缺口/监控启动前）→ 一次性 OPEN_GAP_BELOW_BAND
     事件，取最深破位档，占用该档当日身份，受规则 B/C 约束；首笔仅在带内不推送。
"""
import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import touch_alert_lane_v1 as lane

DAY = '20261106'
NEXT_DAY = '20261105'
NOW = 20261106093100
NEXT_NOW = 20261105093100
CODE = '600000'


def _t(sec, day=DAY):
    return f'{day}0930{sec:02d}'


class V2Base(unittest.TestCase):
    def setUp(self):
        # 状态文件全部落在 D 盘临时目录（随测试文件所在目录），不写 C 盘
        self.base = Path(tempfile.mkdtemp(dir=str(Path(__file__).resolve().parent)))
        self.state = self.base / 'touch_alerts.json'

    def tearDown(self):
        shutil.rmtree(self.base, ignore_errors=True)

    def plan(self, pid='p1', bands=None, code=CODE, day=DAY, zone=None):
        bands = bands if bands is not None else [
            dict(lower=10.0, upper=10.5), dict(lower=9.0, upper=9.4),
            dict(lower=8.0, upper=8.4)]
        row = dict(code=code, name='测试股', entry_rule=dict(observation_bands=bands))
        if zone:
            row['zone'] = zone
        return dict(monitor_date=day, assistant_plan_id=pid, rows=[row])

    @staticmethod
    def q(price, sec, day=DAY):
        return dict(price=price, time=_t(sec, day))

    def detect(self, plan, quote, now=NOW):
        return lane.detect(plan, {plan['rows'][0]['code']: quote}, self.state, now_stamp=now)

    def notified(self):
        if not self.state.exists():
            return {}
        return json.loads(self.state.read_bytes())['notified']

    def baseline(self, day=DAY):
        if not self.state.exists():
            return {}
        return json.loads(self.state.read_bytes()).get('last_observed_quotes', {}).get(day, {})


class RealDownwardCrossTest(V2Base):
    # 测试1：上方报价→下穿上沿产生一次事件
    def test_downward_cross_upper_fires_once(self):
        plan = self.plan(bands=[dict(lower=10.0, upper=10.5)])
        self.assertEqual(self.detect(plan, self.q(10.60, 1)), [])  # 上方：只建基准
        events = self.detect(plan, self.q(10.20, 2))
        self.assertEqual(len(events), 1)
        e = events[0]
        self.assertTrue(e['crossed'])
        self.assertEqual(e['trigger_policy'], 'V2')
        self.assertEqual(e['trigger_type'], 'DOWNWARD_CROSS_UPPER')
        self.assertEqual(e['prev_price'], 10.60)
        entry = self.notified()[e['event_key']]
        self.assertEqual(entry['direction_class'], 'DOWNWARD_ENTRY_OBSERVED')
        self.assertEqual(entry['direction_evidence'], 'last_observed_quotes')
        self.assertEqual(entry['direction_previous_price'], 10.60)
        # 同价再扫（时间更新）：同带已通知，不重复
        self.assertEqual(self.detect(plan, self.q(10.20, 3)), [])
        self.assertEqual(len(self.notified()), 1)

    # 测试2：第一笔报价在带内不推送（只建基准）
    def test_first_quote_in_band_alert_direction_unknown(self):
        plan = self.plan(bands=[dict(lower=10.0, upper=10.5)])
        events = self.detect(plan, self.q(10.20, 1))
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]['trigger_type'], 'FIRST_QUOTE_IN_BAND')
        self.assertEqual(events[0]['direction_class'], 'FIRST_QUOTE_IN_BAND_DIRECTION_UNKNOWN')
        self.assertIn('方向未知', lane.message(events[0]))
        self.assertFalse(events[0]['crossed'])
        self.assertFalse(lane.reverify(events[0], self.q(11.0, 2), NOW, self.state))
        lane.record_delivery(events[0], True, self.state)
        self.assertEqual(self.detect(plan, self.q(10.20, 2)), [])
        base = self.baseline()[CODE]
        self.assertEqual(base['price'], 10.20)
        self.assertEqual(base['quote_time'], _t(2))

    def test_first_inside_precedes_breached_upper_band(self):
        plan = self.plan(bands=[dict(lower=10.0,upper=10.5),dict(lower=9.0,upper=9.4)])
        events = self.detect(plan,self.q(9.2,1))
        self.assertEqual(len(events),1)
        self.assertEqual(events[0]['trigger_type'],'FIRST_QUOTE_IN_BAND')
        self.assertEqual(events[0]['lower'],9.0)

    def test_first_inside_waits_reentry_without_relabeling_cross(self):
        plan = self.plan(bands=[dict(lower=10.0,upper=10.5)])
        self.detect(plan,self.q(10.2,1))
        self.assertEqual(lane.pending_sends(plan,self.state,NOW,{CODE:self.q(10.8,2)}),[])
        pending=lane.pending_sends(plan,self.state,NOW,{CODE:self.q(10.3,3)})
        self.assertEqual(len(pending),1)
        self.assertFalse(pending[0]['crossed'])
        self.assertEqual(pending[0]['trigger_type'],'FIRST_QUOTE_IN_BAND')
        self.assertIn('方向未知',lane.message(pending[0]))

    # 测试3：首笔在带下 → V2.1 一次性破位事件 (OPEN_GAP_BELOW_BAND)
    def test_first_quote_below_band_gap_event(self):
        plan = self.plan(bands=[dict(lower=10.0, upper=10.5)])
        events = self.detect(plan, self.q(9.50, 1))
        self.assertEqual(len(events), 1)
        e = events[0]
        self.assertEqual(e['trigger_type'], 'OPEN_GAP_BELOW_BAND')
        self.assertEqual(e['trigger_policy'], 'V2')
        self.assertIsNone(e['prev_price'])
        self.assertTrue(e['crossed'])
        entry = self.notified()[e['event_key']]
        self.assertEqual(entry['direction_class'], 'GAP_BELOW_BAND_OBSERVED')
        self.assertEqual(entry['direction_evidence'], 'first_observed_quote')
        # 基准仍然建立
        self.assertEqual(self.baseline()[CODE]['price'], 9.50)
        # 同价再扫（时间更新）：破位档已占用，不重复
        self.assertEqual(self.detect(plan, self.q(9.50, 2)), [])
        self.assertEqual(len(self.notified()), 1)

    # 测试4：破位后从下方反弹进入带内不推送；升破上沿再下穿同档仍去重
    def test_rebound_from_below_no_push(self):
        plan = self.plan(bands=[dict(lower=10.0, upper=10.5)])
        gap = self.detect(plan, self.q(9.50, 1))   # 首笔带下 → 破位事件
        self.assertEqual(len(gap), 1)
        self.assertEqual(gap[0]['trigger_type'], 'OPEN_GAP_BELOW_BAND')
        lane.record_delivery(gap[0], True, self.state)
        self.assertEqual(self.detect(plan, self.q(10.20, 2)), [])  # 反弹入带
        self.assertEqual(self.baseline()[CODE]['price'], 10.20)
        self.assertEqual(self.detect(plan, self.q(9.80, 3)), [])   # 又落回带下：无上沿穿越
        self.assertEqual(self.detect(plan, self.q(10.60, 4)), [])  # 升破上沿（新基准）
        # 再次下穿同档 → 破位事件已占用该档当日身份，同档去重不再推
        self.assertEqual(self.detect(plan, self.q(10.20, 5)), [])
        self.assertEqual(len(self.notified()), 1)

    # 测试5：同带反复抖动只推一次
    def test_same_band_jitter_only_once(self):
        plan = self.plan(bands=[dict(lower=10.0, upper=10.5)])
        self.detect(plan, self.q(10.60, 1))
        events = self.detect(plan, self.q(10.20, 2))
        self.assertEqual(len(events), 1)
        lane.record_delivery(events[0], True, self.state)  # SENT
        self.assertEqual(self.detect(plan, self.q(10.10, 3)), [])
        self.assertEqual(self.detect(plan, self.q(10.60, 4)), [])  # 反弹回上沿上方（新基准）
        self.assertEqual(self.detect(plan, self.q(10.20, 5)), [])  # 再次下穿同带→抑制
        self.assertEqual(len(self.notified()), 1)
        snap = lane.health_snapshot(self.state, day=DAY)
        self.assertGreaterEqual(snap['suppressed_overlapping_band_count'], 1)


class TierDedupeTest(V2Base):
    # 测试6：zone 与 obs0 重叠只推一次（同轮多档→最深一档，其余 SUPPRESSED_SAME_SCAN）
    def test_zone_and_obs0_overlap_single_push(self):
        plan = self.plan(code='603703',
                         zone=dict(zone_lower=9.7449, zone_upper=9.8),
                         bands=[dict(lower=9.7449, upper=9.783)])
        self.assertEqual(self.detect(plan, self.q(9.90, 1)), [])   # 上方基准
        events = self.detect(plan, self.q(9.75, 2))                # 同轮穿越 zone+obs0
        self.assertEqual(len(events), 1)
        self.assertTrue(events[0]['event_key'].endswith(':obs0'))  # 更深的档（upper 更低）
        st = json.loads(self.state.read_bytes())
        reasons = [s['reason'] for s in st['suppression_events']]
        self.assertEqual(reasons.count('SUPPRESSED_SAME_SCAN'), 1)  # zone 被抑制
        self.assertEqual(self.detect(plan, self.q(9.75, 3)), [])
        self.assertEqual(len(self.notified()), 1)

    # 测试7：首档 SENT 后下穿更低独立档允许再推
    def test_deeper_independent_tier_after_sent(self):
        plan = self.plan(bands=[dict(lower=10.0, upper=10.5), dict(lower=9.0, upper=9.4)])
        self.detect(plan, self.q(10.60, 1))
        first = self.detect(plan, self.q(10.20, 2))
        self.assertEqual(len(first), 1)
        lane.record_delivery(first[0], True, self.state)
        second = self.detect(plan, self.q(9.20, 3))
        self.assertEqual(len(second), 1)
        self.assertTrue(second[0]['event_key'].endswith(':obs1'))
        self.assertEqual(len(self.notified()), 2)

    # 测试8：新档与旧档重叠禁止（SUPPRESSED_OVERLAPPING_BAND）
    def test_overlapping_new_tier_forbidden(self):
        plan = self.plan(bands=[dict(lower=10.0, upper=10.5), dict(lower=9.8, upper=10.2)])
        self.detect(plan, self.q(10.60, 1))
        first = self.detect(plan, self.q(10.35, 2))
        self.assertEqual(len(first), 1)
        lane.record_delivery(first[0], True, self.state)
        events = self.detect(plan, self.q(10.10, 3))  # 下穿重叠带
        self.assertEqual(events, [])
        self.assertEqual(len(self.notified()), 1)
        st = json.loads(self.state.read_bytes())
        self.assertEqual(st['suppression_events'][-1]['reason'], 'SUPPRESSED_OVERLAPPING_BAND')

    # 测试9：新档上沿未低于旧档下沿禁止（不相交但更浅 → SUPPRESSED_SAME_CODE）
    def test_upper_band_not_below_notified_lower_forbidden(self):
        plan = self.plan(bands=[dict(lower=9.0, upper=9.4), dict(lower=10.0, upper=10.5)])
        self.detect(plan, self.q(10.60, 1))           # 全部带上方 → 只建基准（无破位）
        first = self.detect(plan, self.q(9.20, 2))    # 同轮下穿两档 → 只推最深 obs0
        self.assertEqual(len(first), 1)
        self.assertTrue(first[0]['event_key'].endswith(':obs0'))
        lane.record_delivery(first[0], True, self.state)
        self.detect(plan, self.q(10.60, 3))           # 反弹到更高带上方（新基准）
        events = self.detect(plan, self.q(10.20, 4))  # 下穿更高带：obs1 未通知但比已通知档更浅
        self.assertEqual(events, [])
        self.assertEqual(len(self.notified()), 1)
        st = json.loads(self.state.read_bytes())
        self.assertEqual(st['suppression_events'][-1]['reason'], 'SUPPRESSED_SAME_CODE')
        self.assertEqual(st['suppression_events'][-1]['detail'], 'NOT_DEEPER_THAN_NOTIFIED_TIERS')

    # 测试10：同股换 plan_id 同日去重仍有效
    def test_cross_plan_same_day_dedup(self):
        p1 = self.plan(pid='p1', bands=[dict(lower=10.0, upper=10.5)])
        self.detect(p1, self.q(10.60, 1))
        first = self.detect(p1, self.q(10.20, 2))
        self.assertEqual(len(first), 1)
        lane.record_delivery(first[0], True, self.state)
        p2 = self.plan(pid='p2', bands=[dict(lower=10.0, upper=10.5)])
        self.detect(p2, self.q(10.60, 3))            # 跨计划共享基准
        events = self.detect(p2, self.q(10.20, 4))   # 跨计划同带下穿：仍去重
        self.assertEqual(events, [])
        self.assertEqual(list(self.notified()), [first[0]['event_key']])

    # 测试11：次日重新允许首档（自然重置 + last_observed_quotes 按日清理）
    def test_next_day_resets(self):
        plan1 = self.plan(day=DAY)
        self.detect(plan1, self.q(10.60, 1))
        first = self.detect(plan1, self.q(10.20, 2))
        lane.record_delivery(first[0], True, self.state)
        plan2 = self.plan(day=NEXT_DAY)
        self.assertEqual(self.detect(plan2, self.q(10.60, 1, NEXT_DAY), now=NEXT_NOW), [])
        events = self.detect(plan2, self.q(10.20, 2, NEXT_DAY), now=NEXT_NOW)
        self.assertEqual(len(events), 1)
        self.assertTrue(events[0]['event_key'].startswith(f'{NEXT_DAY}:'))
        self.assertEqual(len(self.notified()), 2)  # 两日各一条，历史保留
        self.assertEqual(list(json.loads(self.state.read_bytes())['last_observed_quotes']),
                         [NEXT_DAY])  # 旧日基准已清理


class SameScanAndQueueTest(V2Base):
    # 测试12：同轮跨多档只推最深一档
    def test_same_scan_multiple_bands_deepest_only(self):
        plan = self.plan(bands=[dict(lower=10.0, upper=10.5), dict(lower=9.0, upper=9.4),
                                dict(lower=8.0, upper=8.4)])
        self.detect(plan, self.q(10.60, 1))
        events = self.detect(plan, self.q(8.20, 2))  # 一轮同时穿越三个带
        self.assertEqual(len(events), 1)
        self.assertTrue(events[0]['event_key'].endswith(':obs2'))  # band.lower 最小
        st = json.loads(self.state.read_bytes())
        reasons = [s['reason'] for s in st['suppression_events']]
        self.assertEqual(reasons.count('SUPPRESSED_SAME_SCAN'), 2)
        self.assertEqual(len(self.notified()), 1)

    # 测试13：第一档 queued 时第二档不得入队
    def test_queued_first_tier_blocks_second_tier(self):
        plan = self.plan(bands=[dict(lower=10.0, upper=10.5), dict(lower=9.0, upper=9.4)])
        self.detect(plan, self.q(10.60, 1))
        first = self.detect(plan, self.q(10.20, 2))
        self.assertEqual(len(first), 1)
        lane.reserve_send(first[0], self.state)  # queued，未送达
        events = self.detect(plan, self.q(9.20, 3))  # 第二档穿越被阻断
        self.assertEqual(events, [])
        self.assertEqual(len(self.notified()), 1)
        st = json.loads(self.state.read_bytes())
        sup = st['suppression_events'][-1]
        self.assertEqual(sup['reason'], 'SUPPRESSED_SAME_CODE')
        self.assertEqual(sup['blocking_event_key'], first[0]['event_key'])

    # 测试14：UNKNOWN 计入已通知（阻止重叠/更浅档，允许更深独立档）
    def test_unknown_counts_as_notified(self):
        plan = self.plan(bands=[dict(lower=10.0, upper=10.5), dict(lower=9.0, upper=9.4)])
        self.detect(plan, self.q(10.60, 1))
        first = self.detect(plan, self.q(10.20, 2))
        lane.record_delivery(first[0], None, self.state)  # UNKNOWN
        self.assertEqual(self.notified()[first[0]['event_key']]['delivery'], 'UNKNOWN')
        self.detect(plan, self.q(10.60, 3))
        self.assertEqual(self.detect(plan, self.q(10.20, 4)), [])  # 同带重推被阻
        second = self.detect(plan, self.q(9.20, 5))  # 更深独立档允许
        self.assertEqual(len(second), 1)
        self.assertEqual(len(self.notified()), 2)

    # 测试15：FAILED 保留记录不阻断后续更低的独立档
    def test_failed_keeps_evidence_allows_deeper(self):
        plan = self.plan(bands=[dict(lower=10.0, upper=10.5), dict(lower=9.0, upper=9.4)])
        self.detect(plan, self.q(10.60, 1))
        first = self.detect(plan, self.q(10.20, 2))
        lane.record_delivery(first[0], False, self.state)  # FAILED
        entry = self.notified()[first[0]['event_key']]
        self.assertEqual(entry['delivery'], 'FAILED')
        self.assertEqual(entry['price'], 10.20)  # 证据保留
        self.assertEqual(entry['lower'], 10.0)
        second = self.detect(plan, self.q(9.20, 3))  # FAILED 不阻断更深档
        self.assertEqual(len(second), 1)
        self.assertEqual(len(self.notified()), 2)


class LegacyCompatibilityTest(V2Base):
    # 测试16：旧 state 文件正常加载
    def test_legacy_state_file_loads(self):
        legacy_key = f'{NEXT_DAY}:p9:{CODE}:obs0'
        state = dict(contract=lane.CONTRACT, notified={
            legacy_key: dict(price=10.2, lower=10.0, upper=10.5, crossed=True,
                             at=f'{NEXT_DAY}100000', event_key=legacy_key,
                             direction_class='DOWNWARD_ENTRY_OBSERVED',
                             direction_evidence='quote.prev', delivery='SENT'),
            f'{NEXT_DAY}:p9:{CODE}:obs1': dict(consumed_by='golden_channel',
                                               event_key=f'{NEXT_DAY}:p9:{CODE}:obs1'),
        }, health=dict(last_delivery_status='SENT', last_scan_event_count=1))
        self.state.write_text(json.dumps(state), encoding='utf-8')
        plan = self.plan()
        self.assertEqual(self.detect(plan, self.q(10.60, 1)), [])  # 不异常，只建基准
        st = json.loads(self.state.read_bytes())
        self.assertEqual(st['contract'], lane.CONTRACT)
        self.assertIn(legacy_key, st['notified'])
        legacy = st['notified'][legacy_key]
        # 历史字段不被重解释，仅新增 legacy_reason 标记
        self.assertEqual(legacy['price'], 10.2)
        self.assertEqual(legacy['delivery'], 'SENT')
        self.assertEqual(legacy['direction_evidence'], 'quote.prev')
        self.assertEqual(legacy['legacy_reason'], lane.LEGACY_REASON)
        self.assertEqual(st['health']['last_delivery_status'], 'SENT')
        self.assertEqual(st['last_observed_quotes'][DAY][CODE]['price'], 10.60)

    # 测试17：历史事件不被重解释/重发
    def test_legacy_not_reinterpreted_or_resent(self):
        key = f'{DAY}:p1:{CODE}:obs0'
        state = dict(contract=lane.CONTRACT, notified={
            key: dict(price=10.2, lower=10.0, upper=10.5, crossed=True,
                      at=_t(1), event_key=key)})  # 旧格式未发送事件
        self.state.write_text(json.dumps(state), encoding='utf-8')
        plan = self.plan(bands=[dict(lower=10.0, upper=10.5)])
        self.assertEqual(self.detect(plan, self.q(10.60, 2)), [])
        self.assertEqual(self.detect(plan, self.q(10.20, 3)), [])  # 同带穿越不再建事件
        self.assertEqual(list(self.notified()), [key])  # 历史保留，无新事件
        entry = self.notified()[key]
        self.assertEqual(entry['crossed'], True)      # 原语义字段不动
        self.assertNotIn('delivery', entry)           # 不重发、不归档、不重解释
        self.assertEqual(entry['legacy_reason'], lane.LEGACY_REASON)
        fresh = {CODE: dict(price=10.2, time=_t(4))}
        self.assertEqual(lane.pending_sends(plan, self.state, NOW, fresh), [])  # 不自动重发

    # 测试18：报价时间倒退或重复不产生虚假下穿（不更新基准）
    def test_quote_time_regression_no_false_cross(self):
        plan = self.plan(bands=[dict(lower=9.5, upper=9.6)])
        self.assertEqual(self.detect(plan, self.q(9.62, 21)), [])   # 首笔只建基准
        self.assertEqual(self.detect(plan, self.q(9.40, 20)), [])   # 时间倒退
        self.assertEqual(self.baseline()[CODE]['quote_time'], _t(21))
        self.assertEqual(self.baseline()[CODE]['price'], 9.62)
        self.assertEqual(self.detect(plan, self.q(9.40, 21)), [])   # 时间重复
        self.assertEqual(self.baseline()[CODE]['quote_time'], _t(21))
        self.assertEqual(self.notified(), {})
        events = self.detect(plan, self.q(9.40, 22))                # 时间前进→真实穿越
        self.assertEqual(len(events), 1)


class ReverifyAndMessageTest(V2Base):
    def _make_v2_event(self):
        plan = self.plan(bands=[dict(lower=10.0, upper=10.5)])
        self.detect(plan, self.q(10.60, 1))
        events = self.detect(plan, self.q(10.20, 2))
        return plan, events[0]

    # 测试19：reverify 与新语义一致
    def test_reverify_v2_semantics(self):
        plan, ev = self._make_v2_event()
        later = 20261106093159  # 报价 093040-093059 均在 120s 时效内
        # 现价三态均可发送（下穿是已观察到的事实）
        self.assertTrue(lane.reverify(ev, self.q(10.20, 40), later))       # 带内
        self.assertTrue(lane.reverify(ev, self.q(9.00, 41), later))        # 已低于下沿
        self.assertTrue(lane.reverify(ev, self.q(10.80, 42), later))       # 反弹离带
        # 报价无效/过期/跨日一律拒绝
        self.assertFalse(lane.reverify(ev, dict(price=10.2, time='20261106092500'), later))  # 过期>120s
        self.assertFalse(lane.reverify(ev, self.q(10.20, 40, NEXT_DAY), later))  # 跨日
        self.assertFalse(lane.reverify(ev, dict(price=10.2, time='bad'), later))
        # legacy 事件保持原语义
        legacy_crossed = dict(event_key=f'{DAY}:p1:{CODE}:obs0',
                              lower=10.0, upper=10.5, crossed=True)
        self.assertFalse(lane.reverify(legacy_crossed, self.q(10.20, 40), later))  # 带内≠带下
        self.assertTrue(lane.reverify(legacy_crossed, self.q(9.00, 41), later))
        legacy_inband = dict(legacy_crossed, crossed=False)
        self.assertTrue(lane.reverify(legacy_inband, self.q(10.20, 40), later))
        self.assertFalse(lane.reverify(legacy_inband, self.q(10.80, 42), later))
        # 已送达后经 state_file 复核拒绝（防重复送达）
        lane.record_delivery(ev, True, self.state)
        self.assertFalse(lane.reverify(ev, self.q(10.20, 43), later, state_file=self.state))

    # 文案三态：下穿仍在带内 / 已低于下沿 / 反弹离带；快照报价不得描述成连续逐笔
    def test_message_three_states_v2(self):
        ev = dict(name='n', code=CODE, price=10.20, lower=10.0, upper=10.5,
                  at=_t(2), crossed=True, trigger_policy='V2')
        m1 = lane.message(ev)
        self.assertIn('已观察到自上向下穿越观察带上沿 10.5000', m1)
        self.assertIn('仍在观察带 10.0000—10.5000', m1)
        self.assertIn('非连续逐笔成交', m1)
        m2 = lane.message(dict(ev, price=9.00))
        self.assertIn('已低于观察带下沿 10.0000', m2)
        m3 = lane.message(dict(ev, price=10.80))
        self.assertIn('反弹离带', m3)
        # legacy 文案保持不变
        legacy = dict(ev)
        legacy.pop('trigger_policy')
        m4 = lane.message(legacy)
        self.assertIn('当前已在观察带下方', m4)
        self.assertIn('不能确认跳空', m4)

    # pending_sends：V2 事件三态均可补发；报价过期归档，不排队不重发
    def test_pending_sends_v2_delivery_paths(self):
        plan, ev = self._make_v2_event()
        # 未发送的 V2 事件：带内现价可补发
        pend = lane.pending_sends(plan, self.state, NOW, {CODE: self.q(10.20, 3)})
        self.assertEqual([e['event_key'] for e in pend], [ev['event_key']])
        self.assertTrue(pend[0]['crossed'])
        self.assertEqual(pend[0]['trigger_policy'], 'V2')
        # 反弹离带现价同样可补发
        self.assertEqual(len(lane.pending_sends(plan, self.state, NOW,
                                                {CODE: self.q(10.80, 4)})), 1)
        # 报价过期：旧价格不能以「现价」名义推送 → 归档 EXPIRED_NOT_SENT
        self.assertEqual(lane.pending_sends(plan, self.state, 20261106094000,
                                            {CODE: dict(price=10.2, time=_t(1))}), [])
        self.assertEqual(self.notified()[ev['event_key']]['delivery'], 'EXPIRED_NOT_SENT')

    # 健康统计：三类抑制计入 health_snapshot；基准数可见
    def test_health_snapshot_counts_suppressions(self):
        plan = self.plan(bands=[dict(lower=10.0, upper=10.5)])
        self.detect(plan, self.q(10.60, 1))
        first = self.detect(plan, self.q(10.20, 2))
        lane.record_delivery(first[0], True, self.state)
        self.detect(plan, self.q(10.60, 3))
        self.detect(plan, self.q(10.20, 4))  # 同带重推→SUPPRESSED_OVERLAPPING_BAND
        snap = lane.health_snapshot(self.state, day=DAY)
        self.assertEqual(snap['suppressed_same_code_count'], 0)
        self.assertEqual(snap['suppressed_overlapping_band_count'], 1)
        self.assertEqual(snap['suppressed_same_scan_count'], 0)
        self.assertEqual(snap['suppression_count'], 1)
        self.assertEqual(snap['observed_quote_baseline_count'], 1)
        self.assertEqual(snap['total_events'], 1)
        self.assertEqual(snap['sent_count'], 1)


class BoundedSuppressionTest(V2Base):
    # suppression_events 有界：每票每日上限 + 总量上限，超出丢弃最旧
    def test_suppression_bounded(self):
        plan_a = self.plan(pid='p1', bands=[dict(lower=10.0, upper=10.5)])
        self.detect(plan_a, self.q(10.60, 1))
        first = self.detect(plan_a, self.q(10.20, 2))
        lane.record_delivery(first[0], True, self.state)
        plan_b = self.plan(pid='p1', bands=[dict(lower=10.0, upper=10.5)], code='600001')
        plan_c = self.plan(pid='p1', bands=[dict(lower=10.0, upper=10.5)], code='600002')
        with mock.patch.object(lane, 'SUPPRESSION_PER_CODE_DAILY_LIMIT', 5), \
                mock.patch.object(lane, 'SUPPRESSION_TOTAL_LIMIT', 7):
            sec = 10
            for _ in range(6):  # A/B/C 各制造 6 条抑制
                for p in (plan_a, plan_b, plan_c):
                    code = p['rows'][0]['code']
                    lane.detect(p, {code: self.q(10.60, sec)}, self.state, now_stamp=NOW)
                    sec += 1
                    lane.detect(p, {code: self.q(10.20, sec)}, self.state, now_stamp=NOW)
                    sec += 1
        st = json.loads(self.state.read_bytes())
        items = st['suppression_events']
        self.assertLessEqual(len(items), 7)  # 总量上限
        per = {}
        for s in items:
            per[s['code']] = per.get(s['code'], 0) + 1
        self.assertTrue(all(c <= 5 for c in per.values()))  # 每票每日上限
        self.assertEqual(set(per), {'600000', '600001', '600002'})  # 丢弃的是最旧（A 只剩 2 条）


class GapBelowBandTest(V2Base):
    """V2.1 破位事件 (OPEN_GAP_BELOW_BAND)：首次有效观察已在带下方时的一次性告知。"""

    # 测试20：破位事件完整交付流程；占用档位后同档不再推
    def test_gap_event_delivery_flow(self):
        plan = self.plan(bands=[dict(lower=10.0, upper=10.5)])
        events = self.detect(plan, self.q(9.50, 1))
        self.assertEqual(len(events), 1)
        ev = events[0]
        # reverify: V2 事件三态均可发送（破位是已观察到的事实）
        self.assertTrue(lane.reverify(ev, self.q(9.50, 40), 20261106093159))
        lane.record_delivery(ev, True, self.state)
        # 破位占用档位: 升破上沿 → 再下穿同档, 同档去重不再推
        self.assertEqual(self.detect(plan, self.q(10.60, 2)), [])
        self.assertEqual(self.detect(plan, self.q(10.20, 3)), [])
        self.assertEqual(len(self.notified()), 1)

    # 测试21：多档同时破位只推最深一档，其余 SUPPRESSED_SAME_SCAN
    def test_gap_multiple_bands_deepest_only(self):
        plan = self.plan(zone=dict(zone_lower=9.7449, zone_upper=9.8),
                         bands=[dict(lower=9.7449, upper=9.783),
                                dict(lower=9.5267, upper=9.6)])
        events = self.detect(plan, self.q(9.50, 1))  # 三档全部破位
        self.assertEqual(len(events), 1)
        self.assertTrue(events[0]['event_key'].endswith(':obs1'))  # 最深档
        self.assertEqual(events[0]['trigger_type'], 'OPEN_GAP_BELOW_BAND')
        st = json.loads(self.state.read_bytes())
        reasons = [s['reason'] for s in st['suppression_events']]
        self.assertEqual(reasons.count('SUPPRESSED_SAME_SCAN'), 2)
        self.assertTrue(all(s['detail'] == 'DEEPER_BAND_SELECTED_GAP'
                            for s in st['suppression_events']))
        # 更深无档，当日不再有任何推送
        self.assertEqual(self.detect(plan, self.q(9.40, 2)), [])
        self.assertEqual(len(self.notified()), 1)

    # 测试22：破位事件受规则 B 约束（在途事件阻断新档）
    def test_gap_blocked_by_alive_event(self):
        key = f'{DAY}:p1:{CODE}:obs0'
        state = dict(contract=lane.CONTRACT, notified={
            key: dict(price=10.2, lower=10.0, upper=10.5, at=_t(1), event_key=key,
                      crossed=True, trigger_policy='V2',
                      trigger_type='DOWNWARD_CROSS_UPPER')})  # 无 delivery → 在途
        self.state.write_text(json.dumps(state), encoding='utf-8')
        plan = self.plan(bands=[dict(lower=10.0, upper=10.5), dict(lower=9.0, upper=9.4)])
        events = self.detect(plan, self.q(8.50, 2))  # 首笔即全部破位
        self.assertEqual(events, [])  # 在途事件阻断
        st = json.loads(self.state.read_bytes())
        sup = st['suppression_events'][-1]
        self.assertEqual(sup['reason'], 'SUPPRESSED_SAME_CODE')
        self.assertEqual(sup['blocking_event_key'], key)

    # 测试23：f618 单档破位（对应 688498/300990/688661 实盘形态）
    def test_gap_f618_single_level(self):
        plan = self.plan(bands=[], code='688498')
        plan['rows'][0]['entry_rule']['f618'] = 628.52
        events = self.detect(plan, self.q(570.0, 1))
        self.assertEqual(len(events), 1)
        self.assertTrue(events[0]['event_key'].endswith(':f618'))
        self.assertEqual(events[0]['trigger_type'], 'OPEN_GAP_BELOW_BAND')
        self.assertEqual(self.detect(plan, self.q(569.0, 2)), [])  # 不重复

    # 测试24：破位文案只陈述破位事实，绝不描述成盘中穿越
    def test_gap_message_wording(self):
        ev = dict(name='n', code=CODE, price=570.0, lower=628.52, upper=628.52,
                  at=_t(2), crossed=True, trigger_policy='V2',
                  trigger_type='OPEN_GAP_BELOW_BAND')
        m1 = lane.message(ev)
        self.assertIn('首次有效观察已低于观察带下沿 628.5200', m1)
        self.assertIn('未观察到盘中穿越过程', m1)
        self.assertNotIn('自上向下穿越观察带上沿', m1)
        self.assertIn('仍在下沿下方', m1)
        m2 = lane.message(dict(ev, price=628.60))   # 回到上沿上方
        self.assertIn('已回到观察带上沿 628.5200 上方', m2)
        m3 = lane.message(dict(ev, price=628.52))   # 回到带内
        self.assertIn('已回到观察带 628.5200—628.5200 内', m3)
        # 破位事件计入健康统计
        plan = self.plan(bands=[dict(lower=10.0, upper=10.5)])
        events = self.detect(plan, self.q(9.50, 1))
        lane.record_delivery(events[0], True, self.state)
        snap = lane.health_snapshot(self.state, day=DAY)
        self.assertEqual(snap['gap_event_count'], 1)
        self.assertEqual(snap['trigger_type_distribution']['OPEN_GAP_BELOW_BAND'], 1)


class Replay603703Test(V2Base):
    """603703 盘中回放 (V2.1)：zone 9.7449–9.8 / obs0 9.7449–9.783 / obs1 9.5267–9.6 /
    obs2 9.4395–9.5101；报价 09:30:03→9.62、09:30:21→9.62、09:30:30→9.53、
    09:30:37→9.51。预期：首笔 9.62 已在 zone/obs0 下方 → 破位事件取更深 obs0
    (zone 记 SUPPRESSED_SAME_SCAN)；9.62→9.53 真实下穿 obs1 推一次；
    9.53→9.51 真实下穿 obs2 推一次；最终恰好 3 条事件。
    对照旧版同日生产实绩: 同价 9.62 重复推 zone+obs0, 共 4 条。"""

    def test_replay(self):
        plan = dict(monitor_date=DAY, assistant_plan_id='c4085ec3f2e5', rows=[dict(
            code='603703', name='盛洋科技',
            zone=dict(zone_lower=9.7449, zone_upper=9.8),
            entry_rule=dict(observation_bands=[
                dict(lower=9.7449, upper=9.783),
                dict(lower=9.5267, upper=9.6),
                dict(lower=9.4395, upper=9.5101)]))])
        state = self.state

        def scan(price, sec):
            return lane.detect(plan, {'603703': self.q(price, sec)}, state, now_stamp=NOW)

        # 09:30:03 → 9.62：已在 zone/obs0 下方 → V2.1 破位事件取更深 obs0，zone 记抑制
        events0 = scan(9.62, 3)
        self.assertEqual(len(events0), 1)
        self.assertTrue(events0[0]['event_key'].endswith(':603703:obs0'))
        self.assertEqual(events0[0]['trigger_type'], 'OPEN_GAP_BELOW_BAND')
        self.assertIsNone(events0[0]['prev_price'])
        lane.record_delivery(events0[0], True, state)
        base = json.loads(state.read_bytes())['last_observed_quotes'][DAY]['603703']
        self.assertEqual(base['price'], 9.62)
        self.assertEqual(base['quote_time'], _t(3))
        # 09:30:21 → 9.62：同价新报价，仍高于 obs1 上沿，不触发
        self.assertEqual(scan(9.62, 21), [])
        # 09:30:30 → 9.53：9.62 > 9.6 且 9.53 <= 9.6 → 真实下穿 obs1 上沿，推送一次
        events = scan(9.53, 30)
        self.assertEqual(len(events), 1)
        self.assertTrue(events[0]['event_key'].endswith(':603703:obs1'))
        self.assertEqual(events[0]['prev_price'], 9.62)
        self.assertTrue(events[0]['crossed'])
        self.assertEqual(events[0]['trigger_type'], 'DOWNWARD_CROSS_UPPER')
        lane.record_delivery(events[0], True, state)  # 模拟发送队列完成
        # 09:30:37 → 9.51：9.53 > 9.5101 且 9.51 <= 9.5101 → 真实下穿 obs2 上沿，推送一次
        events2 = scan(9.51, 37)
        self.assertEqual(len(events2), 1)
        self.assertTrue(events2[0]['event_key'].endswith(':603703:obs2'))
        self.assertEqual(events2[0]['prev_price'], 9.53)
        lane.record_delivery(events2[0], True, state)
        # 最终恰好 3 条事件：破位 obs0 + 下穿 obs1 + 下穿 obs2
        st = json.loads(state.read_bytes())
        self.assertEqual([k.split(':')[-1] for k in st['notified']], ['obs0', 'obs1', 'obs2'])
        # zone 仅在破位轮被抑制一次（同轮最深档原则），其余轮次无任何抑制
        sup = st['suppression_events']
        self.assertEqual(len(sup), 1)
        self.assertEqual(sup[0]['band_id'], 'zone')
        self.assertEqual(sup[0]['reason'], 'SUPPRESSED_SAME_SCAN')
        self.assertEqual(sup[0]['detail'], 'DEEPER_BAND_SELECTED_GAP')
        # 次日重扫同序列：自然重置，重新产生三条
        plan2 = dict(plan, monitor_date=NEXT_DAY)

        def scan2(price, sec):
            return lane.detect(plan2, {'603703': self.q(price, sec, NEXT_DAY)},
                               state, now_stamp=NEXT_NOW)

        ev_d0 = scan2(9.62, 3)
        self.assertEqual(len(ev_d0), 1)
        self.assertEqual(ev_d0[0]['trigger_type'], 'OPEN_GAP_BELOW_BAND')
        lane.record_delivery(ev_d0[0], True, state)
        scan2(9.62, 21)
        ev_d2 = scan2(9.53, 30)
        self.assertEqual(len(ev_d2), 1)
        lane.record_delivery(ev_d2[0], True, state)
        ev_d2b = scan2(9.51, 37)
        self.assertEqual(len(ev_d2b), 1)
        st2 = json.loads(state.read_bytes())
        day2_keys = [k for k in st2['notified'] if k.startswith(NEXT_DAY)]
        self.assertEqual(len(day2_keys), 3)


if __name__ == '__main__':
    unittest.main()
