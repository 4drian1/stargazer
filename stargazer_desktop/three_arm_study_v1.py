"""three_arm_study_v1.py — 三臂对照（用户规格；2026-10-20）。

臂定义（同证据包/同选项语义/同硬门）：
  A1 代码规则自身：冻结映射规则 + 代码布尔输入（无供应商）——度量规则贡献；
  A2 JEV+同硬门：noul 评分 → 冻结映射；
  A3 DeepSeek+同硬门：自由判断。
案例 5 个（事先登记可接受集合与不可接受行为，不指定唯一答案）：
  C1 普通回调计划仍有效 / C2 领涨板块+领涨股同步转弱 / C3 部分板块转弱其他强势 /
  C4 已持仓证据不足退出 / C5 数据过期（本地硬门拦截，三家一致短路）。
输出：three_arm_v1.json + three_arm_report_v1.md。
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import decision_layer as dl
import jev_adapter_v1 as jev
from jev_live_provider import JEVJudge
from oracles_v1 import EvidencePack, pyxis_context, tarot_candidate

ASOF = '20261021110000'
QUERY = '20261021111200'
VU = '20261021120000'

ACCEPTABLE = {
    'C1_pullback_ok': {'UNPOSITIONED': {'READY_BUY_CONFIRM_HUMAN',
                                        'CONTINUE_OBSERVE'}},
    'C2_leader_both_weak': {'UNPOSITIONED': {'DEFER', 'CONTINUE_OBSERVE'}},
    'C3_partial_weak': {'UNPOSITIONED': {'CONTINUE_OBSERVE', 'DEFER'}},
    'C4_hold_no_exit': {'POSITIONED': {'HOLD_OBSERVE', 'HEIGHTEN_RISK_WATCH'}},
    'C5_stale': {'UNPOSITIONED': {'EVIDENCE_INSUFFICIENT'}},
}
PREFERRED = {'C2_leader_both_weak': 'DEFER', 'C3_partial_weak': 'CONTINUE_OBSERVE'}


def pack(case: str) -> EvidencePack:
    trig = dict(t='20261021111000', close=25.4)
    ds = dict(approach=True, stop=True, divergence=True)
    if case == 'C1_pullback_ok':
        boards = [dict(board='semi', strength='stable', weakening=False)]
        leaders = [dict(code='688012', weakening=False, note='normal_pullback')]
    elif case == 'C2_leader_both_weak':
        boards = [dict(board='semi', strength='narrowing', weakening=True)]
        leaders = [dict(code='688012', weakening=True)]
    elif case == 'C3_partial_weak':
        boards = [dict(board='semi', strength='narrowing', weakening=True),
                  dict(board='ai_compute', strength='strong', weakening=False)]
        leaders = [dict(code='688012', weakening=False)]
    else:
        boards = [dict(board='semi', strength='stable')]
        leaders = [dict(code='688012', weakening=False)]
    stale = case == 'C5_stale'
    risk = pyxis_context(
        version='232主版', amp_232=0.014, amp_266=0.013,
        market_state=dict(index='sse', trend='flat'),
        board_states=boards, leader_states=leaders,
        missing=[], asof_ts='20261020110000' if stale else ASOF)
    cand = tarot_candidate(
        '300639', p_close_score=0.55, golden_zone=-0.12, f618=25.43,
        plan_dist_atr=0.31, atr_ref=0.407, downside_state=ds,
        trigger=trig if case != 'C4_hold_no_exit' else None,
        evidence_refs=dict(snapshot='snap'))
    return jev.EvidencePack(
        evidence_id=case, asof_ts=ASOF, market_date='20261021',
        query_ts=QUERY, candidates=[cand], risk_context=risk,
        risk_missing=[]).bind()


def code_policy(p: EvidencePack, scenario: str) -> dict:
    """A1：冻结映射规则 + 代码布尔（无供应商）。度量规则自身贡献。"""
    rc = p.risk_context or {}
    weak = any(b.get('weakening') for b in rc.get('board_states') or []) or \
        any(l.get('weakening') for l in rc.get('leader_states') or [])
    trig = bool(p.candidates and p.candidates[0].get('trigger'))
    stale = str(rc.get('asof_ts', ''))[:8] != str(p.asof_ts)[:8]
    ins = 1.0 if stale else 0.0
    if scenario == 'UNPOSITIONED':
        scores = dict(insufficient=ins,
                      buy_ready=1.0 if trig and not weak else 0.0,
                      defer=1.0 if trig and weak else 0.0)
    else:
        scores = dict(insufficient=ins, reduce=1.0 if weak else 0.0,
                      heighten=0.0)
    from jev_live_provider import _derive_option
    option, conf = _derive_option(scenario, scores)
    return dict(arm='A1_code_rules', option=option, confidence=conf,
                scores=scores)


def main(repeats: int = 3) -> None:
    arms = [('A2_JEV', JEVJudge), ('A3_DeepSeek', dl.DeepSeekJudge)]
    results = []
    for case, scens in ACCEPTABLE.items():
        scenario = list(scens.keys())[0]
        p = pack(case)
        for k in range(repeats):
            results.append(dict(case=case, run=k, scenario=scenario,
                                **code_policy(p, scenario)))
            for arm, cls in arms:
                prov = cls()
                t0 = time.perf_counter()
                try:
                    advice = jev.judge(
                        p, scenario, valid_until_ts=VU,
                        api_caller=lambda req, pr=prov: dl._vendor_payload(pr, req))
                    results.append(dict(
                        case=case, run=k, scenario=scenario, arm=arm,
                        option=advice.option, confidence=advice.confidence,
                        gate='HARD' if advice.notes.startswith('HARD_GATE') else None,
                        latency_s=round(time.perf_counter() - t0, 3)))
                except ValueError as exc:
                    results.append(dict(case=case, run=k, scenario=scenario,
                                        arm=arm, gate_error=str(exc),
                                        latency_s=round(time.perf_counter() - t0, 3)))
                except Exception as exc:
                    results.append(dict(case=case, run=k, scenario=scenario,
                                        arm=arm, error=type(exc).__name__,
                                        latency_s=round(time.perf_counter() - t0, 3)))
    Path('three_arm_v1.json').write_text(
        json.dumps(results, ensure_ascii=False, indent=1), encoding='utf-8')

    # 汇总：可接受集合命中 + 中位延迟
    from collections import defaultdict
    hit = defaultdict(lambda: defaultdict(int))
    tot = defaultdict(lambda: defaultdict(int))
    lat = defaultdict(list)
    for r in results:
        arm = r.get('arm', '?')
        if r.get('gate_error') or r.get('error'):
            continue
        acc = ACCEPTABLE[r['case']][r['scenario']]
        hit[r['case']][arm] += int(r['option'] in acc)
        tot[r['case']][arm] += 1
        if arm != 'A1_code_rules' and r.get('latency_s') is not None:
            lat[arm].append(r['latency_s'])
    L = ['# 三臂对照 V1（同证据包/同硬门/同选项语义；可接受集合事前登记）', '']
    for case in ACCEPTABLE:
        row = {a: f"{hit[case][a]}/{tot[case][a]}" for a in
               ('A1_code_rules', 'A2_JEV', 'A3_DeepSeek')}
        pref = PREFERRED.get(case)
        L.append(f"- {case}（优先参考={pref or '—'}）: {row}")
    for arm, v in lat.items():
        s = sorted(v)
        L.append(f"- {arm} 延迟中位 {s[len(s)//2]}s（n={len(v)}，真实调用）")
    L.append('- A1 为本地规则臂，无供应商调用；C5 为三臂共同本地硬门拦截。')
    Path('three_arm_report_v1.md').write_text('\n'.join(L), encoding='utf-8')
    print('\n'.join(L))


if __name__ == '__main__':
    main()
