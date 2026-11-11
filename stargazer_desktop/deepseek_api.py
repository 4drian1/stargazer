"""DeepSeek only. Local DPAPI credential and usage ledger; never expose API errors/keys."""
import base64
import ctypes
from contextlib import closing
from ctypes import wintypes
from datetime import datetime
import json
import os
from pathlib import Path
import sqlite3
import threading

import paths

_lock = threading.Lock()
MODEL = 'deepseek-flash'


class Blob(ctypes.Structure):
    _fields_ = [('size', wintypes.DWORD), ('data', ctypes.POINTER(ctypes.c_ubyte))]


def crypt(raw, decrypt=False):
    if os.name != 'nt':
        raise ValueError('本机密钥文件需要 Windows DPAPI；其他平台请用 DEEPSEEK_API_KEY')
    buf = ctypes.create_string_buffer(raw)
    src = Blob(len(raw), ctypes.cast(buf, ctypes.POINTER(ctypes.c_ubyte)))
    dst = Blob()
    lib = ctypes.WinDLL('crypt32', use_last_error=True)
    name = 'CryptUnprotectData' if decrypt else 'CryptProtectData'
    fn = getattr(lib, name)
    fn.argtypes = [ctypes.POINTER(Blob), ctypes.c_void_p, ctypes.c_void_p,
                  ctypes.c_void_p, ctypes.c_void_p, wintypes.DWORD, ctypes.POINTER(Blob)]
    fn.restype = wintypes.BOOL
    if not fn(ctypes.byref(src), None, None, None, None, 1, ctypes.byref(dst)):
        raise ValueError('Windows 密钥保护失败；迁移电脑后请在设置中重新填写')
    try:
        return ctypes.string_at(dst.data, dst.size)
    finally:
        kernel = ctypes.WinDLL('kernel32')
        kernel.LocalFree.argtypes = [ctypes.c_void_p]
        kernel.LocalFree(ctypes.cast(dst.data, ctypes.c_void_p))


def key_path():
    return Path(paths.data_file('.secrets/deepseek.dpapi', seed_from_bundle=False))


def encrypted_key(key):
    if not isinstance(key, str) or not key.startswith('sk-') or len(key) < 20:
        raise ValueError('DeepSeek 密钥格式不正确')
    return base64.b64encode(crypt(key.strip().encode())).decode('ascii')


def save_key(key):
    file = key_path()
    raw = encrypted_key(key)
    file.parent.mkdir(parents=True, exist_ok=True)
    temp = file.with_suffix('.tmp')
    temp.write_text(raw, encoding='ascii')
    temp.replace(file)


def get_key():
    key = os.environ.get('DEEPSEEK_API_KEY')
    if key:
        return key
    file = key_path()
    if not file.exists():
        raise ValueError('请在设置中填写 DeepSeek API Key')
    return crypt(base64.b64decode(file.read_text(encoding='ascii')), True).decode()


def status():
    return {'provider': 'DeepSeek', 'model': MODEL,
            'configured': bool(os.environ.get('DEEPSEEK_API_KEY') or key_path().exists())}


def record_result(db, model, purpose, max_tokens, *, finish_reason=None, usage=None,
                  outcome='OK', error_type=None, error_details=None):
    """Metadata only: no prompt, response text, credentials, or raw SDK exception strings."""
    def count(name):
        value = getattr(usage, name, None)
        return value if type(value) is int and value >= 0 else None
    allowed = {'stop', 'length', 'tool_calls', 'content_filter', 'insufficient_system_resource'}
    reason = finish_reason if finish_reason in allowed else 'unknown'
    with _lock, closing(sqlite3.connect(db)) as conn:
        conn.execute('CREATE TABLE IF NOT EXISTS results('
                     'created_at TEXT,model TEXT,purpose TEXT,max_tokens INTEGER,finish_reason TEXT,'
                     'prompt_tokens INTEGER,completion_tokens INTEGER,total_tokens INTEGER,'
                     'outcome TEXT,error_type TEXT)')
        conn.execute('INSERT INTO results VALUES(?,?,?,?,?,?,?,?,?,?)',
                     (datetime.now().isoformat(), str(model)[:80], purpose, max_tokens, reason,
                      count('prompt_tokens'), count('completion_tokens'), count('total_tokens'),
                      outcome, error_type))
        if error_details is not None:
            conn.execute('CREATE TABLE IF NOT EXISTS request_errors('
                         'created_at TEXT,purpose TEXT,details TEXT)')
            conn.execute('INSERT INTO request_errors VALUES(?,?,?)',
                         (datetime.now().isoformat(), purpose, json.dumps(error_details)))
        conn.commit()


def request_failure(exc):
    """Only exception class names/codes; never message, URL, headers or body."""
    chain, seen = [], set()
    current = exc
    for _ in range(6):
        if current is None or id(current) in seen:
            break
        seen.add(id(current))
        chain.append(type(current).__name__)
        current = current.__cause__
    status = getattr(exc, 'status_code', None)
    status = status if type(status) is int else None
    kind = 'REQUEST_ERROR'
    if status == 401:
        kind, note = 'AUTH_ERROR', '服务端拒绝鉴权，请核对本机密钥；不是余额判断'
    elif status == 402:
        kind, note = 'BALANCE_ERROR', '服务端返回余额不足，请核对该密钥所属账户'
    elif status == 429:
        kind, note = 'RATE_LIMIT', '服务端限流，请稍后手动再试；不是余额判断'
    elif any('Timeout' in name for name in chain):
        kind, note = 'TIMEOUT', '等待服务端响应超时，不能据此判断余额不足'
    elif 'APIConnectionError' in chain:
        kind, note = 'CONNECTION_ERROR', '连接中断或未取得完整响应，不能据此判断余额不足；请检查网络或代理'
    else:
        note = '服务请求未完成，请查看本地错误分类；不能据此判断余额不足'
    return (f'DeepSeek 请求失败（{type(exc).__name__}）；{note}；未自动重试',
            dict(kind=kind, cause_types=chain, http_status=status))


def complete(cfg, messages, *, tools=None, structured=False, purpose='chat'):
    from openai import OpenAI
    settings = cfg.get('deepseek') or {}
    if purpose not in ('chat', 'plan', 'review'):
        raise ValueError('DeepSeek调用用途无效')
    key = get_key()
    max_tokens = min(int(settings.get('max_tokens', 2000)), 4000)
    if max_tokens < 1:
        raise ValueError('DeepSeek 输出Token上限必须为正数')
    # Account before sending without a daily cap. Failed requests count; no hidden retries.
    db = Path(paths.data_file('data/assistant/deepseek_usage.sqlite', seed_from_bundle=False))
    reserve = len(json.dumps(messages, ensure_ascii=False).encode('utf-8')) + max_tokens + 1000
    if tools:
        reserve += len(json.dumps(tools, ensure_ascii=False).encode('utf-8'))
    with _lock:
        db.parent.mkdir(parents=True, exist_ok=True)
        with closing(sqlite3.connect(db)) as conn:
            conn.execute('CREATE TABLE IF NOT EXISTS usage(day TEXT PRIMARY KEY,calls INTEGER,tokens INTEGER)')
            day = datetime.now().strftime('%Y%m%d')
            conn.execute('INSERT OR IGNORE INTO usage VALUES(?,0,0)', (day,))
            conn.execute('UPDATE usage SET calls=calls+1,tokens=tokens+? WHERE day=?', (reserve, day))
            conn.commit()
    kwargs = dict(model=settings.get('model', MODEL), messages=messages,
                  max_tokens=max_tokens, extra_body={'thinking': {'type': 'disabled'}})
    if tools:
        kwargs['tools'] = tools
    if structured:
        kwargs['response_format'] = {'type': 'json_object'}
    try:
        response = OpenAI(api_key=key, base_url='https://api.deepseek.com',
                          timeout=60.0, max_retries=0).chat.completions.create(**kwargs)
    except Exception as exc:
        # SDK error text can include request contents; never forward it.
        message, details = request_failure(exc)
        record_result(db, kwargs['model'], purpose, max_tokens,
                      outcome='REQUEST_ERROR', error_type=type(exc).__name__, error_details=details)
        raise ValueError(message) from None
    with _lock, closing(sqlite3.connect(db)) as conn:
        actual = getattr(response.usage, 'total_tokens', None)
        if isinstance(actual, int) and actual >= 0:
            conn.execute('UPDATE usage SET tokens=tokens-?+? WHERE day=?', (reserve, actual, day))
            conn.commit()
    choice = response.choices[0] if response.choices else None
    reason = getattr(choice, 'finish_reason', None)
    outcome = ('NO_CHOICES' if choice is None else 'TRUNCATED' if reason == 'length' else
               'OK' if reason == 'stop' or (tools and reason == 'tool_calls') else 'INCOMPLETE')
    if outcome == 'OK' and not tools and not (getattr(choice.message, 'content', None) or '').strip():
        outcome = 'EMPTY_CONTENT'
    record_result(db, kwargs['model'], purpose, max_tokens, finish_reason=reason,
                  usage=response.usage, outcome=outcome)
    if choice is None:
        raise ValueError('DeepSeek未返回有效回复（NO_CHOICES）；本次未保存，请重新生成')
    if reason == 'length':
        raise ValueError(f'DeepSeek输出被截断（finish_reason=length，本次预算{max_tokens} tokens）；'
                         '未保存不完整内容，请重新生成简短回复')
    if outcome == 'EMPTY_CONTENT':
        raise ValueError('DeepSeek返回了空正文（EMPTY_CONTENT）；本次未保存，请重新生成')
    if outcome != 'OK':
        label = '内容被服务端拦截' if reason == 'content_filter' else '回复未正常结束'
        raise ValueError(f'DeepSeek{label}；本次未保存，请稍后重新生成')
    return response
