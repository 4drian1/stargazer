"""jev_live_provider.py — JEV（TypeSafe systemone）适配器（第三适配器位；2026-10-20）。

协议实测：POST https://api.typesafe.ai/v1/systemone，Bearer key，
body={model, state, questions:{名:{type:'noul', instructions}}}；
响应={model, answers:{名:{type:'noul', noul:0..1}}, usage}。**noul=0..1 评分**，
延迟 ~0.5s、极廉价——它是结构化判断引擎，不是聊天模型。

映射（与另两家同一把尺）：state=同一 judge 请求 JSON；按场景问一组 noul 问题；
**选项由代码按冻结优先级从分数推导**，输出标准响应 JSON 走同一 judge() 门禁：
- UNPOSITIONED: 证据缺失>0.5→EVIDENCE_INSUFFICIENT；
  否则 买入就绪>0.5 且 暂缓≤0.5→READY_BUY_CONFIRM_HUMAN；
  否则 暂缓>0.5→DEFER；否则 CONTINUE_OBSERVE。
- POSITIONED: 证据缺失>0.5→EVIDENCE_INSUFFICIENT；
  减仓>0.5→REDUCE_EXIT_CONFIRM_HUMAN；加强关注>0.5→HEIGHTEN_RISK_WATCH；
  否则 HOLD_OBSERVE。
置信度=所选维度分数（**分数≠胜率**，同列展示口径不变）；理由维度 JEV 不提供
（如实标注：质量维度只比选项/一致性/延迟费用，不比理由有据性）。
"""
from __future__ import annotations

import base64
import json
import os
import urllib.request
from pathlib import Path

from decision_layer import JudgeProvider

ENDPOINT = 'https://api.typesafe.ai/v1/systemone'
MODEL = 'jev-latest'
_JEV_KEY_FILE = '.secrets/jev.dpapi'

_Q_UNPOSITIONED = {
    'insufficient': '风险或机会证据是否缺失到无法给出判断？',
    'buy_ready': '机会侧触发条件是否全部成立、且不存在反对买入的证据？',
    'defer': '是否已经出现了反对原计划的证据、应当暂缓推进？',
}
_Q_POSITIONED = {
    'insufficient': '风险或机会证据是否缺失到无法给出判断？',
    'reduce': '是否出现了应当提交人工减仓或退出确认的证据？',
    'heighten': '是否需要加强风险关注（尚未到减仓确认级）？',
}


def get_jev_key() -> str:
    from deepseek_api import crypt
    import paths as _p
    key = os.environ.get('JEV_API_KEY')
    if key:
        return key
    file = Path(_p.data_file(_JEV_KEY_FILE, seed_from_bundle=False))
    if not file.exists():
        raise ValueError('请保存 JEV API Key')
    return crypt(base64.b64decode(file.read_text(encoding='ascii')), True).decode()


def _derive_option(scenario: str, scores: dict) -> tuple[str, float]:
    ins = scores.get('insufficient', 0.0)
    if ins > 0.5:
        return 'EVIDENCE_INSUFFICIENT', ins
    if scenario == 'UNPOSITIONED':
        buy, defer = scores.get('buy_ready', 0.0), scores.get('defer', 0.0)
        if buy > 0.5 and defer <= 0.5:
            return 'READY_BUY_CONFIRM_HUMAN', buy
        if defer > 0.5:
            return 'DEFER', defer
        return 'CONTINUE_OBSERVE', max(buy, defer, 1.0 - ins)
    red, hig = scores.get('reduce', 0.0), scores.get('heighten', 0.0)
    if red > 0.5:
        return 'REDUCE_EXIT_CONFIRM_HUMAN', red
    if hig > 0.5:
        return 'HEIGHTEN_RISK_WATCH', hig
    return 'HOLD_OBSERVE', max(1.0 - red, 1.0 - hig) / 2


class JEVJudge(JudgeProvider):
    name = 'JEV'
    MAX_ATTEMPTS = 1

    def judge(self, request: dict, *, timeout_s: float = 30.0) -> dict:
        if 'hard_gate' in request:
            return dict(hard_gate=request['hard_gate'], provider=self.name)
        self.last_extra_attempts = 0
        scenario = request.get('scenario', 'UNPOSITIONED')
        questions = _Q_UNPOSITIONED if scenario == 'UNPOSITIONED' \
            else _Q_POSITIONED
        body = json.dumps(dict(
            model=MODEL, state=json.dumps(request, ensure_ascii=False),
            questions={k: dict(type='noul', instructions=v)
                       for k, v in questions.items()})).encode('utf-8')
        req = urllib.request.Request(
            ENDPOINT, data=body,
            headers={'Authorization': 'Bearer ' + get_jev_key(),
                     'Content-Type': 'application/json'})
        import time
        t0 = time.perf_counter()
        with urllib.request.urlopen(req, timeout=timeout_s) as r:
            resp = json.loads(r.read().decode('utf-8'))
        latency = round(time.perf_counter() - t0, 3)
        scores = {k: float(v.get('noul', 0.0))
                  for k, v in (resp.get('answers') or {}).items()}
        option, conf = _derive_option(scenario, scores)
        usage = resp.get('usage') or {}
        payload = dict(
            option=option, confidence=round(conf, 4),
            supporting=[f'JEV noul scores: {json.dumps(scores)}'],
            opposing=[], watch=[], invalidations=[],
            evidence_refs=[request.get('evidence_id', 'pack')],
            notes='derived_from_noul_scores')
        return dict(provider=self.name, raw=json.dumps(payload, ensure_ascii=False),
                    latency_s=latency, usage=usage, scores=scores)
