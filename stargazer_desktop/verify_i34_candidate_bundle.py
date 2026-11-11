#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Verify I34 against the REAL candidate EXE (no GUI launched, no side effects).

Reads dist_candidate/Stargazer.exe with PyInstaller's archive reader (same approach as
accept_handoff_artifacts.py) and checks:
  1. every I34 resource the runtime needs is actually bundled
  2. each bundled file is BYTE-IDENTICAL to its whitelisted source (sha256 compare)
  3. i34_bridge is compiled into the PYZ (so the server can import it)
  4. acceptance: using ONLY the bytes extracted from the EXE, run the whole chain
     import -> classify -> panel shows it -> DeepSeek request contains it

Usage: python verify_i34_candidate_bundle.py [path/to/Stargazer.exe]
"""
import gzip
import hashlib
import json
import shutil
import sys
import tempfile
from pathlib import Path

from PyInstaller.archive.readers import CArchiveReader

HERE = Path(__file__).resolve().parent
DEFAULT_EXE = HERE / 'dist_candidate' / 'Stargazer.exe'
CORVUS = HERE.parent / 'c3' / 'corvus' / 'impact_v1_1'

# essential runtime assets (whitelist from Stargazer_demo_safe.spec)
EXPECTED = [
    'i34_demo/i34_infer.py',
    'i34_demo/i34_desktop_integration.py',
    'i34_demo/i34_vocab.json',
    'i34_demo/i34_receipt_index.json',
    'i34_demo/i34_deepseek_input.json',
    'i25_quantity_repair_fixed_fit_20261029_v1/common.py',
    'i25_quantity_repair_fixed_fit_20261029_v1/extractor_v3.py',
    'i24_1_zero_fit_quantity_contract_20261029_v1/extractor_v2.py',
    'i28_time_clause_role_repair_20261029_v1/common.py',
    'i28_time_clause_role_repair_20261029_v1/extractor_v4_1.py',
    'i28_time_clause_role_repair_20261029_v1/extractor_v4_2.py',
    'i29_bound_clause_time_encoding_20261029_v1/time_encoding.py',
    'i33_clause_role_fit_20261031_v1/model33.pkl.gz',
]

NEW_TITLE = '候选EXE验收：关于签订重大合同的公告'
NEW_BODY = (
    '关于签订重大合同的公告\n'
    '证券代码：600519\n'
    '本公司于2026年10月30日与交易对手签订重大采购合同，合同金额为人民币1.20亿元，'
    '占公司最近一个会计年度经审计营业收入的15%。合同尚未生效，需经双方董事会审议通过后生效。\n'
    '本次交易不构成重大资产重组，亦不构成关联交易。\n'
)


def sha_bytes(b):
    return hashlib.sha256(b).hexdigest()


def _norm(k):
    return k.replace('\\', '/')


def check_bundle(exe: Path, outdir: Path):
    reader = CArchiveReader(str(exe))
    toc = {_norm(k): k for k in reader.toc}
    missing = [e for e in EXPECTED if e not in toc]
    assert not missing, 'bundled resources missing: %s' % missing

    # Reconstruct the WHOLE I34 subtree exactly as the EXE ships it (not just the
    # whitelist subset) — otherwise helper modules like i29/bound_terms.py are absent
    # here even though the real bundle contains them, giving a false failure.
    PREFIXES = ('i34_demo/', 'i25_quantity_repair_fixed_fit_20261029_v1/',
                'i24_1_zero_fit_quantity_contract_20261029_v1/',
                'i28_time_clause_role_repair_20261029_v1/',
                'i29_bound_clause_time_encoding_20261029_v1/',
                'i33_clause_role_fit_20261031_v1/')
    relevant = [n for n in toc if n.startswith(PREFIXES)]
    for norm_key in relevant:
        data = reader.extract(toc[norm_key])
        target = outdir / norm_key
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)

    # byte-identity check for the essential assets only
    mismatched = []
    for e in EXPECTED:
        data = reader.extract(toc[e])
        src = CORVUS / e
        if not src.exists() or sha_bytes(src.read_bytes()) != sha_bytes(data):
            mismatched.append(e)
    assert not mismatched, 'bundled bytes differ from source: %s' % mismatched
    print('[bundle] %d/%d resources packed in the exe; %d entries in the I34 subtree'
          % (len(EXPECTED), len(EXPECTED), len(relevant)))

    pyz_key = [k for k in reader.toc if 'pyz' in _norm(k).lower()]
    assert pyz_key, 'no PYZ found'
    pyz = reader.open_embedded_archive(pyz_key[0])
    assert 'i34_bridge' in {_norm(k) for k in pyz.toc}, 'i34_bridge not compiled into the bundle'
    print('[bundle] %d/%d resources present, byte-identical to source' % (len(EXPECTED), len(EXPECTED)))
    print('[bundle] i34_bridge present in PYZ: OK')
    return outdir


def _resp_cls():
    class Msg:
        def __init__(self, c):
            self.content = c

    class Choice:
        def __init__(self, c):
            self.message = Msg(c)
            self.finish_reason = 'stop'

    class Usage:
        prompt_tokens, completion_tokens, total_tokens = 10, 20, 30

    class Resp:
        def __init__(self, c):
            self.choices = [Choice(c)]
            self.model = 'deepseek-chat'
            self.usage = Usage()

    return Resp


def chain_acceptance(bundle_root: Path):
    """Run import -> classify -> panel -> DeepSeek request using ONLY extracted bytes."""
    from flask import Flask
    sys.path.insert(0, str(bundle_root / 'i34_demo'))
    for mod in ('i34_infer', 'i34_desktop_integration', 'i34_bridge'):
        sys.modules.pop(mod, None)

    import i34_infer as I
    # inside the EXE i34_bridge does this injection; mirrored here explicitly
    I.configure(impact_dir=str(bundle_root), artifacts_dir=str(bundle_root / 'i34_demo'))
    assert I.CLF is not None and I.VOCAB_E, 'model/vocab not loaded from extracted bundle'
    print('[load] model+dict loaded from EXE bytes: OK (%d event types)' % len(I.VOCAB_E))

    import i34_desktop_integration as INT
    INT.set_desktop_dir(str(HERE))
    store = bundle_root / 'runtime_records.json'
    INT.set_runtime_store_path(store)
    # 隔离写入：队列与信息影子库也必须指向临时目录，否则 i34_desktop_integration
    # 内的 import paths 会命中真实 stargazer_desktop，把测试公告写进生产路径。
    INT.set_d2_queue_path(str(bundle_root / 'i34_d2_pending_queue.json'))
    INT.set_info_shadow_store_path(str(bundle_root / 'i34_info_shadow_records.json'))
    INT.reset_runtime_records()

    Resp = _resp_cls()
    import deepseek_api
    captured = {}

    def fake_complete(cfg, messages, purpose='review'):
        captured['messages'] = messages
        return Resp('【模拟回复】政策/美股/金油本次未提供材料 → 未知。')

    deepseek_api.complete = fake_complete

    app = Flask(__name__)
    app.config['TESTING'] = True
    INT.register_i34_routes(app, lambda: {})
    c = app.test_client()

    # step 1: import one new announcement
    j = c.post('/api/i34/classify',
               json={'body': NEW_BODY, 'code': '600519', 'title': NEW_TITLE}).get_json()
    assert j['ok'], j
    rec, sha = j['record'], j['record']['input_sha']
    print('[step1] imported -> importance=%s on_chain=%s' % (rec['importance'], j['on_chain']))

    # step 2: it must show in the panel
    items = c.get('/api/i34/panel').get_json()['items']
    assert sha in [i.get('input_sha') for i in items], 'not shown in panel'
    assert any(i.get('title') == NEW_TITLE for i in items), 'title not shown'
    print('[step2] panel shows it (%d items total)' % len(items))

    # step 3: it must be in the actual DeepSeek request
    r = c.post('/api/i34/deepseek').get_json()
    assert r['ok'] and isinstance(r['result'], dict)
    ctx = '\n'.join(m.get('content', '') for m in captured['messages'])
    assert sha in ctx and NEW_TITLE in ctx, 'not present in the DeepSeek request'
    assert '未知' in ctx, 'policy/US/gold-oil must be flagged 未知'
    print('[step3] DeepSeek request contains it; reply=plain dict keys=%s' % list(r['result']))
    print('[step4] mock reply rendered:', (r['result']['content'] or '')[:40] + '...')
    return True


def main():
    exe = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_EXE
    assert exe.exists(), 'candidate exe not found: %s' % exe
    print('candidate:', exe, '(%.1f MB)' % (exe.stat().st_size / 1e6))
    tmp = Path(tempfile.mkdtemp(prefix='i34_bundle_'))
    try:
        bundle_root = check_bundle(exe, tmp)
        chain_acceptance(bundle_root)
        print('I34 CANDIDATE-EXE ACCEPTANCE PASSED')
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == '__main__':
    main()
