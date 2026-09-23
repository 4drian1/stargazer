"""Qwen3-4B-GGUF 事实抽取封装（事实抽取层，规格：LLM 只出事实，不判断涨跌）。

已实测要点（勿改）：
- llama-cpp-python 的 create_chat_completion 不支持 chat_template_kwargs，
  关思考模式靠 prompt 末尾加 /no_think。
- 输出清洗：去掉 <think>...</think> 残留，用"第一个 { 到其配对的 }"括号匹配截取 JSON。
- 解析失败重试一次（temperature 仍 0.1，换措辞要求只输出 JSON）。
- 单次调用 120 秒超时保护（线程实现）。
- 模型懒加载单例；n_gpu_layers 默认 -1 全量上 GPU，传 0 纯 CPU 兜底（3050 6GB 降级）。
"""
import json
import re
import threading
import time
from pathlib import Path

DEFAULT_MODEL_PATH = "d:/stargazer/models/Qwen3-4B-GGUF/Qwen3-4B-Q4_K_M.gguf"

SYSTEM_PROMPT = "你是金融信息结构化抽取工具，只抽取事实，不判断涨跌"

# 事实层字段严格限定（规格 4.0①），影响推断层字段禁止出现在抽取输出中
FACT_FIELDS = ["event_type", "entities", "action", "amount", "official_scope", "effective_date"]

_PROMPT_TEMPLATE = """请从下面的新闻文本中抽取事实信息，只输出一个 JSON 对象，不要输出任何其他文字。
JSON 字段：event_type（事件类型）、entities（涉及实体列表）、action（动作描述）、
amount（金额/比例数值，没有则填 null）、official_scope（官方适用范围）、effective_date（生效日期，没有则填 null）。

严格规则（违反则输出无效）：
1. effective_date：只有原文明确写出完整年月日（如"2026年9月11日"）才填该日期；
   原文没有明确完整日期时填 null。禁止根据当前日期、发布时间或常识补充年份。
2. amount：只准复制原文中直接出现的金额数字；禁止乘除、汇总、换算（如"40,000万元"
   照抄，禁止自己算成"400000000"）；原文没有金额填 null。

新闻文本：
{text}

/no_think"""

_RETRY_PROMPT_TEMPLATE = """上一轮的输出无法解析。请只输出一个合法的 JSON 对象（不要 markdown 代码块，不要解释文字），字段同上。

新闻文本：
{text}

/no_think"""

_llm = None          # 懒加载单例
_llm_lock = threading.Lock()


def _get_llm(model_path=DEFAULT_MODEL_PATH, n_gpu_layers=-1, n_ctx=4096):
    global _llm
    if _llm is None:
        with _llm_lock:
            if _llm is None:
                from llama_cpp import Llama  # 延迟导入，避免无 GPU 环境 import 即失败
                _llm = Llama(model_path=str(model_path), n_gpu_layers=n_gpu_layers,
                             n_ctx=n_ctx, verbose=False)
    return _llm


def unload_llm():
    """释放模型（切换 GPU/CPU 模式或退出前调用）。"""
    global _llm
    with _llm_lock:
        _llm = None


def _call_model(prompt, model_path, n_gpu_layers, temperature=0.1, max_tokens=400):
    llm = _get_llm(model_path, n_gpu_layers)
    resp = llm.create_chat_completion(
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return resp["choices"][0]["message"]["content"]


def _call_with_timeout(prompt, model_path, n_gpu_layers, timeout_s=120):
    """线程实现超时保护：超时返回 None。"""
    result = {}

    def _worker():
        try:
            result["text"] = _call_model(prompt, model_path, n_gpu_layers)
        except Exception as e:  # 模型调用异常也要带出来，不能静默
            result["error"] = e

    t = threading.Thread(target=_worker, daemon=True)
    t.start()
    t.join(timeout_s)
    if t.is_alive():
        return None, TimeoutError(f"LLM 调用超过 {timeout_s} 秒")
    if "error" in result:
        return None, result["error"]
    return result.get("text"), None


def _clean_and_parse(text):
    """清洗 + 括号匹配截取 JSON。返回 (parsed_dict 或 None, cleaned_text)。"""
    # 去掉 <think>...</think> 残留（含跨行）
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
    # 找第一个 {，然后做配对括号截取（容忍模型多吐一个 }）
    start = cleaned.find("{")
    if start < 0:
        return None, cleaned
    depth = 0
    end = -1
    for i, ch in enumerate(cleaned[start:], start):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                end = i
                break
    if end < 0:
        return None, cleaned
    candidate = cleaned[start:end + 1]
    try:
        return json.loads(candidate), cleaned
    except json.JSONDecodeError:
        return None, cleaned


def extract_facts(text, model_path=DEFAULT_MODEL_PATH, n_gpu_layers=-1, timeout_s=120):
    """抽取事实。返回 dict：parsed / raw_text / parse_ok / latency_s。

    parse_ok=False 时 parsed 为 None；解析失败已自动重试一次。
    """
    t0 = time.time()
    raw_text = ""
    for attempt, prompt in enumerate((_PROMPT_TEMPLATE, _RETRY_PROMPT_TEMPLATE)):
        out, err = _call_with_timeout(prompt.format(text=text), model_path, n_gpu_layers, timeout_s)
        if err is not None:
            raw_text = f"<ERROR: {err}>"
            continue  # 超时/异常也走一次重试
        raw_text = out or ""
        parsed, _ = _clean_and_parse(raw_text)
        if parsed is not None:
            # 只保留事实层字段，防止模型越权输出推断层字段
            parsed = {k: parsed.get(k) for k in FACT_FIELDS}
            return {
                "parsed": parsed,
                "raw_text": raw_text,
                "parse_ok": True,
                "latency_s": round(time.time() - t0, 2),
            }
    return {
        "parsed": None,
        "raw_text": raw_text,
        "parse_ok": False,
        "latency_s": round(time.time() - t0, 2),
    }


if __name__ == "__main__":
    # 冒烟：一条监管政策新闻
    news = ("2026年9月10日，证监会发布《关于修改〈上市公司减持股份管理暂行办法〉的决定》，"
            "自2026年10月1日起施行，进一步限制大股东通过大宗交易减持的比例。")
    r = extract_facts(news)
    print(json.dumps(r, ensure_ascii=False, indent=2))
