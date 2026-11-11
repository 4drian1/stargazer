import difflib, zipfile, hashlib
from pathlib import Path

DESK = Path(r'd:/stargazer/stargazer_desktop')
ZIP = Path(r'd:/stargazer/ui_next/artifacts/newui_sources_independent_v1.zip')

with zipfile.ZipFile(ZIP) as z:
    newui = z.read('server.py').decode('utf-8').splitlines()
cur = DESK.joinpath('server.py').read_text(encoding='utf-8').splitlines()

# unified diff new-ui -> current (shows what my revert removed/changed)
print('=== new-UI backup -> current reverted (unified) ===')
for line in difflib.unified_diff(newui, cur, 'newui_backup', 'reverted', lineterm=''):
    if line.startswith('+'):
        print(line)  # only show added-to-current (should be empty if clean removal)
