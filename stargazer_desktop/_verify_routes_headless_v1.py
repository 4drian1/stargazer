# -*- coding: utf-8 -*-
"""无 GUI / 无外部调用的路由接线冒烟测试：验证 3 个新合同接口可响应且结构正确，
以及缺口③ _enrich_execution_status 在无真实交易库时全部 UNKNOWN（不伪造）。"""
import json
import sys

sys.path.insert(0, r'd:\stargazer\stargazer_desktop')

import server  # noqa: E402  (imports Flask app + ui_* modules)

app = server.app
app.config['TESTING'] = True
client = app.test_client()

fails = []


def check(cond, msg):
    print(('OK  ' if cond else 'FAIL') + ' ' + msg)
    if not cond:
        fails.append(msg)


# 1) 日历接口
r = client.get('/api/calendar/next_trading_day?date=20261109')
j = r.get_json()
print('calendar:', json.dumps(j, ensure_ascii=False)[:200])
check(r.status_code == 200, 'calendar route 200')
check(j.get('ok') in (True, False), 'calendar has ok flag')
if j.get('ok'):
    check(j.get('next_trading_day') is not None, 'calendar returns next_trading_day when available')

# 2) 计划版本接口
r = client.get('/api/plan/versions')
j = r.get_json()
print('plan_versions keys:', sorted(j.keys()))
check(r.status_code == 200, 'plan_versions route 200')
check('versions' in j and 'current' in j, 'plan_versions has versions/current')
check('status' in j.get('current', {}), 'plan_versions current has status field')

# 3) 全局概览接口（零网络；无快照时为 MISSING + decision_use=false）
r = client.get('/api/market/global_overview')
j = r.get_json()
print('global_overview:', json.dumps(j, ensure_ascii=False)[:200])
check(r.status_code == 200, 'global_overview route 200')
check(j.get('ok') is True, 'global_overview ok=true')
check(j.get('decision_use') is False, 'global_overview decision_use=false (no fake data)')
check(isinstance(j.get('items'), list) and len(j.get('items')) >= 1, 'global_overview has items list')

# 4) 缺口③ 执行状态富集：无 cfg → 全部 UNKNOWN，不伪造
plan = {'monitor_date': '20261109', 'stocks': [{'code': '600000'}, {'code': '000001'}]}
enriched = server._enrich_execution_status(plan)
for st in enriched['stocks']:
    det = st.get('execution_detail', {})
    check(det.get('price_touched') == 'UNKNOWN', 'price_touched UNKNOWN for %s' % st.get('code'))
    check(det.get('push_delivered') == 'UNKNOWN', 'push_delivered UNKNOWN for %s' % st.get('code'))
    check(det.get('manual_trade_registered') == 'UNKNOWN', 'manual_trade_registered UNKNOWN for %s' % st.get('code'))
    check(st.get('execution_status') == 'UNKNOWN', 'execution_status UNKNOWN for %s' % st.get('code'))

print('\nRESULT:', 'PASS' if not fails else 'FAIL')
print('failures:', fails)
sys.exit(1 if fails else 0)
