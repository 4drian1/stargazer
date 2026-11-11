# Compatibility entry: every build uses the same maintained whitelist.
from pathlib import Path

_safe_spec = Path(SPEC).with_name('Stargazer_demo_safe.spec')
exec(compile(_safe_spec.read_text(encoding='utf-8'), str(_safe_spec), 'exec'), globals())
