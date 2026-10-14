"""DeepSeek only. Local DPAPI credential; bounded calls; never expose API errors/keys."""
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


def complete(cfg, messages, *, tools=None, structured=False):
    from openai import OpenAI
    settings = cfg.get('deepseek') or {}
    key = get_key()
    max_tokens = min(int(settings.get('max_tokens', 2000)), 4000)
    if max_tokens < 1:
        raise ValueError('DeepSeek 输出Token上限必须为正数')
    if len(json.dumps(messages, ensure_ascii=False)) > 60000:
        raise ValueError('单次输入打包超过本地安全预算；请拆分问题或新建会话，原记录保留（非DeepSeek上下文已满）')
    # Reserve before sending. Failed requests count too; retries are never hidden.
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
            calls, tokens = conn.execute('SELECT calls,tokens FROM usage WHERE day=?', (day,)).fetchone()
            if calls >= 40 or tokens + reserve > 120000:
                raise ValueError('DeepSeek 当日调用/Token预算已达到上限，提醒监控不受影响')
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
        raise ValueError(f'DeepSeek 请求失败（{type(exc).__name__}）；请检查密钥、余额或网络') from None
    with _lock, closing(sqlite3.connect(db)) as conn:
        actual = getattr(response.usage, 'total_tokens', None)
        if isinstance(actual, int) and actual >= 0:
            conn.execute('UPDATE usage SET tokens=tokens-?+? WHERE day=?', (reserve, actual, day))
            conn.commit()
    if not response.choices or response.choices[0].finish_reason == 'length':
        raise ValueError('DeepSeek 输出不完整，本次不保存计划/复盘')
    return response
