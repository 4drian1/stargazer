"""Compare shipped model-tool and plan code to the tested sources."""
import hashlib
import json
from pathlib import Path
from PyInstaller.archive.readers import CArchiveReader
from verify_risk_display_bundle import normalized


def main():
    desktop = Path(__file__).resolve().parent
    exe = desktop/'manual_plan_build_20261021/Stargazer.exe'
    archive = CArchiveReader(str(exe))
    pyz = archive.open_embedded_archive('PYZ.pyz')
    modules = ('plan_evidence', 'assistant_workflow', 'services', 'agent.brain',
               'postclose_risk_runtime', 'golden_rules', 'golden_monitor', 'server', 'paths', 'trade_plan_v1')
    for name in modules:
        file = desktop/(name.replace('.', '/')+'.py')
        assert normalized(pyz.extract(name)) == normalized(compile(file.read_bytes(), str(file), 'exec')), name
    for name in ('app.js', 'style.css', 'pentacles.js'):
        assert archive.extract('web\\'+name) == (desktop/'web'/name).read_bytes(), name
    assert not any('.secrets' in name or name.endswith('config.yaml') for name in archive.toc)
    result = dict(status='PASS', modules=list(modules), web_resources=3, embedded_secrets=False,
                  exe_sha256=hashlib.sha256(exe.read_bytes()).hexdigest())
    with (desktop/'manual_plan_acceptance_20261021_v2/bundle_verification.json').open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2)
    print(json.dumps(result))


if __name__ == '__main__':
    main()
