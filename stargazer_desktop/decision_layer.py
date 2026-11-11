"""decision_layer.py — Stargazer 1.0 可替换决策层（双供应商对照；2026-10-20）。

架构（用户定）：统一证据包 → DeepSeek / Kimi / (后续JEV) → **同一代码校验**
（jev_adapter._validate_response：选项合法、置信度有限[0,1]、证据引用、
三时刻、机会门）→ 建议与人工确认。星币侍从与 Pyxis 固定，不随供应商更换。

实验纪律（用户定）：
- 同输入同任务：两家读同一个代码生成的证据包，语义一致的提示/选项/结构；
  直接读原始证据，不经任一供应商摘要中转；
- 先过正确性门（硬违规不可被其他项高分抵消）；
- 再比判断质量（固定案例、隐藏供应商、重复运行看稳定性）；
- 前瞻影子对照后置；费用与延迟同录；只有差异足够明确才选胜者。

Kimi 密钥走 DPAPI 本机加密（复用 deepseek_api 的 crypt），不入源码。
"""
from __future__ import annotations

import base64
import json
import time
import urllib.request
import urllib.error
from pathlib import Path

import paths
import jev_adapter_v1 as jev
from oracles_v1 import EvidencePack

HERE = Path(__file__).resolve().parent

KIMI_BASE = 'https://api.kimi.com/coding/v1'
KIMI_KEY_FILE = '.secrets/kimi.dpapi'
PROBE_MODEL_CANDIDATES = ('kimi-for-coding', 'kimi-for-coding-highspeed')


# ---------- Kimi 密钥（DPAPI） ----------

def save_kimi_key(key: str) -> None:
    from deepseek_api import crypt
    import paths as _p  # noqa
    if not isinstance(key, str) or not key.startswith('sk-') or len(key) < 20:
        raise ValueError('Kimi 密钥格式不正确')
    file = Path(_p.data_file(KIMI_KEY_FILE, seed_from_bundle=False))
    file.parent.mkdir(parents=True, exist_ok=True)
    raw = base64.b64encode(crypt(key.strip().encode())).decode('ascii')
    temp = file.with_suffix('.tmp')
    temp.write_text(raw, encoding='ascii')
    temp.replace(file)


def get_kimi_key() -> str:
    from deepseek_api import crypt
    import paths as _p  # noqa
    import os
    key = os.environ.get('KIMI_API_KEY')
    if key:
        return key
    file = Path(_p.data_file(KIMI_KEY_FILE, seed_from_bundle=False))
    if not file.exists():
        raise ValueError('请保存 Kimi API Key')
    return crypt(base64.b64decode(file.read_text(encoding='ascii')), True).decode()


# ---------- 供应商判断提示（供应商中性，同一提示两家共用） ----------

JUDGE_SYSTEM = """你是 Stargazer 1.0 的综合判断层。你只依据下方 JSON 证据包判断，
禁止改写或补造任何数字、时间或状态。
铁律：
1) Pyxis 幅度是未来120交易分钟最大不利下探幅度（amplitude），不是概率；
2) 风险证据缺失或占位 → 只能选 EVIDENCE_INSUFFICIENT（未知≠安全）；
3) Corvus 信息未接入——信息缺口必须显式保留，不得当作"无利空"；
4) 只能输出给定选项之一，置信度为0到1的小数（你的自信度，不是股票胜率）；
5) 支持/反对理由必须简短并绑定证据编号；每条建议必须有失效条件与有效期；
6) evidence_refs 只能取自请求中 legal_refs 列表里的编号，不得自创编号、
字段路径或自由文本当引用；引用不确定时用章节级锚点（如 risk_context）；
7) 选项语义（V1.4 冻结）：CONTINUE_OBSERVE=条件尚未齐、无足够依据推进买入，
必须在 watch 列出等待条件；DEFER=已出现反对原计划的证据、暂缓推进，
必须在 opposing 列出反证并在 watch 给出重新评估条件；两者当前都不买，
但下一步不同；证据缺失选 EVIDENCE_INSUFFICIENT，不得混入上两类。
输出必须是 JSON，结构：
{"option","confidence","supporting":[..],"opposing":[..],"watch":[..],
"invalidations":[..],"evidence_refs":[..],"notes":""}"""


def build_judge_request(pack: EvidencePack, scenario: str, opp, risk,
                        valid_until_ts: str) -> dict:
    """供应商中性的判断请求（同一提示、同一结构、直接读原始证据）。"""
    gate = jev.gate_checks(pack, scenario)
    if gate is not None:
        return dict(hard_gate=gate.as_dict())
    return jev._build_request(pack, scenario, opp, risk, valid_until_ts)


# ---------- 供应商适配器 ----------

class JudgeProvider:
    name = 'base'

    def judge(self, request: dict, *, timeout_s: float = 30.0) -> dict:
        raise NotImplementedError


def _openai_chat(base_url: str, key: str, model: str, messages: list,
                 timeout_s: float, temperature: float = 0.2,
                 max_attempts: int = 1) -> tuple[str, float, dict, int]:
    """OpenAI 兼容接口；返回 (文本, 延迟秒, 用量, 实际尝试数)。
    首轮关闭自动重试：max_attempts=1；代理回退仅在显式允许时二次尝试。"""
    import openai
    last_exc = None
    attempts = 0
    for attempt in range(max_attempts):
        try:
            client = openai.OpenAI(api_key=key, base_url=base_url,
                                   timeout=timeout_s,
                                   max_retries=0)
            t0 = time.perf_counter()
            resp = client.chat.completions.create(
                model=model, messages=messages,
                response_format={'type': 'json_object'},
                temperature=temperature)
            latency = time.perf_counter() - t0
            usage = {}
            if getattr(resp, 'usage', None):
                usage = dict(prompt_tokens=resp.usage.prompt_tokens,
                             completion_tokens=resp.usage.completion_tokens,
                             total_tokens=resp.usage.total_tokens)
            return resp.choices[0].message.content, latency, usage, attempts + 1
        except Exception as exc:
            last_exc = exc
            attempts += 1
            if attempt + 1 < max_attempts and _is_network_error(exc):
                import os
                os.environ.setdefault('https_proxy', 'http://127.0.0.1:7897')
                os.environ.setdefault('http_proxy', 'http://127.0.0.1:7897')
                continue
            raise
    raise last_exc


def _is_network_error(exc) -> bool:
    name = type(exc).__name__.lower()
    return any(k in name for k in ('connection', 'timeout', 'apiconnection'))


class DeepSeekJudge(JudgeProvider):
    name = 'DeepSeek'
    MODEL = 'deepseek-flash'
    BASE = 'https://api.deepseek.com'

    def judge(self, request: dict, *, timeout_s: float = 30.0) -> dict:
        if 'hard_gate' in request:
            return dict(hard_gate=request['hard_gate'], provider=self.name)
        from deepseek_api import get_key
        messages = [dict(role='system', content=JUDGE_SYSTEM),
                    dict(role='user', content=json.dumps(
                        request, ensure_ascii=False))]
        text, latency, usage, attempts = _openai_chat(
            self.BASE, get_key(), self.MODEL, messages, timeout_s)
        self.last_extra_attempts = max(0, attempts - 1)
        return dict(provider=self.name, raw=text, latency_s=round(latency, 3),
                    usage=usage)


class KimiJudge(JudgeProvider):
    name = 'Kimi'
    model = None          # 探测后缓存到文件（每日至多一次）
    TEMPERATURE = 1.0     # kimi-for-coding 仅允许 temperature=1
    # Persist the probed Kimi model name next to the EXE (or source tree under
    # dev) via paths.data_file, NOT under __file__: a frozen one-file bundle has
    # no writable location beside the module, and __file__ resolves into the
    # read-only _MEIPASS temp dir. Config/holdings/plan/info-panel data already
    # use this same helper.
    _MODEL_CACHE = Path(paths.data_file('data/assistant/kimi_model.txt'))

    def judge(self, request: dict, *, timeout_s: float = 30.0) -> dict:
        if 'hard_gate' in request:
            return dict(hard_gate=request['hard_gate'], provider=self.name)
        self.last_extra_attempts = 0
        if self.model is None:
            if self._MODEL_CACHE.exists():
                self.model = self._MODEL_CACHE.read_text(encoding='ascii').strip()
            else:
                self.model = probe_kimi_model(timeout_s=timeout_s)
                self._MODEL_CACHE.parent.mkdir(parents=True, exist_ok=True)
                self._MODEL_CACHE.write_text(self.model, encoding='ascii')
                self.last_extra_attempts += 1   # 探测计入尝试
        messages = [dict(role='system', content=JUDGE_SYSTEM),
                    dict(role='user', content=json.dumps(
                        request, ensure_ascii=False))]
        text, latency, usage, attempts = _openai_chat(
            KIMI_BASE, get_kimi_key(), self.model, messages, timeout_s,
            temperature=self.TEMPERATURE)
        self.last_extra_attempts = max(0, attempts - 1)
        return dict(provider=self.name, model=self.model, raw=text,
                    latency_s=round(latency, 3), usage=usage)


def probe_kimi_model(*, timeout_s: float = 15.0) -> str:
    """列出可用模型并选首个 kimi 系；失败抛错（UNAVAILABLE 路径处理）。"""
    req = urllib.request.Request(
        KIMI_BASE + '/models',
        headers={'Authorization': 'Bearer ' + get_kimi_key()})
    with urllib.request.urlopen(req, timeout=timeout_s) as r:
        data = json.loads(r.read().decode('utf-8'))
    ids = [m.get('id', '') for m in data.get('data', [])]
    for cand in PROBE_MODEL_CANDIDATES:
        if cand in ids:
            return cand
    for mid in ids:
        if 'kimi' in mid:
            return mid
    raise ValueError(f'Kimi 无可用模型: {ids[:10]}')


# ---------- 对照执行（统一走 jev.judge：请求前+返回后同一套校验） ----------

def _vendor_payload(prov: JudgeProvider, request: dict) -> dict:
    """供应商调用+JSON解析；仅在 judge() 前置校验（硬门+三时刻）通过后执行。"""
    import re as _re
    out = prov.judge(request)
    if 'raw' not in out:
        raise ValueError('VENDOR_NO_RAW')
    m = _re.search(r'\{.*\}', out['raw'], _re.S)
    payload = json.loads(m.group(0)) if m else None
    if not isinstance(payload, dict):
        raise ValueError('VENDOR_PARSE_FAIL')
    prov._last_out = out     # 延迟/token 记账（消费侧只取 validated advice）
    return payload


def run_case(case_pack: EvidencePack, scenario: str,
             providers: list[JudgeProvider], *, valid_until_ts: str,
             opp=None, risk=None, repeats: int = 1) -> list[dict]:
    """同一证据包跑所有供应商；统一走 jev.judge()：请求前校验（硬门+三时刻）
    与返回后校验（选项/置信度/引用/机会门）同一条路径，不另写一套。
    硬门短路=不发送供应商请求（归因：本地拦截，非供应商判断）。"""
    results = []
    for prov in providers:
        for k in range(repeats):
            t0 = time.perf_counter()
            rec = dict(case=case_pack.evidence_id, scenario=scenario,
                       run=k, provider=prov.name,
                       wall_s=round(time.perf_counter() - t0, 3))
            try:
                advice = jev.judge(
                    case_pack, scenario, opportunity_explain=opp,
                    risk_explain=risk, valid_until_ts=valid_until_ts,
                    api_caller=lambda req, p=prov: _vendor_payload(p, req))
                rec['advice'] = advice.as_dict()
                rec['correctness'] = 'PASS'
                if advice.notes.startswith('HARD_GATE'):
                    rec['gate'] = 'HARD'      # 本地硬门短路，未发送供应商请求
                else:
                    _lo = getattr(prov, '_last_out', None) or {}
                    rec['latency_s'] = _lo.get('latency_s')
                    rec['usage'] = _lo.get('usage')
            except ValueError as exc:
                rec['correctness'] = 'FAIL'
                rec['gate_error'] = str(exc)
            except jev.JEVNotConfigured:
                rec['correctness'] = 'ERROR'
                rec['error'] = 'JEVNotConfigured'
            except Exception as exc:
                rec['correctness'] = 'ERROR'
                rec['error'] = type(exc).__name__
                rec['msg'] = str(exc)[:200]
            results.append(rec)
    return results
