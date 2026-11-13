"""只读扫描编辑器本地历史，寻找与 v158 期望 SHA 完全一致的文件副本。"""
import hashlib
from pathlib import Path

TARGETS = {
    'touch_alert_lane_v1.py': 'acd7917ad6f5',
    'self_check_realtime_v1.py': '0840e8bd8b69',
    'joint_shadow_context_v1.py': 'f5757a85e58f',
}
HISTORY = Path(r'C:\Users\22107\AppData\Roaming\Trae CN\User\History')
HISTORY2 = Path(r'C:\Users\22107\AppData\Roaming\Code\User\History')

found = {name: [] for name in TARGETS}
scanned = 0
for root in (HISTORY, HISTORY2):
    if not root.is_dir():
        continue
    for path in root.rglob('*'):
        if not path.is_file():
            continue
        if path.suffix == '.json':
            continue
        name = path.name
        # 历史条目可能带时间戳后缀, 用前缀匹配
        match = next((t for t in TARGETS if name.startswith(t[:18])), None)
        if not match:
            continue
        scanned += 1
        try:
            body = path.read_bytes()
        except OSError:
            continue
        digest = hashlib.sha256(body).hexdigest()
        found[match].append((digest, len(body), str(path)))

for name, hits in found.items():
    want = TARGETS[name]
    print('=== %s (want %s) candidates=%d' % (name, want, len(hits)))
    for digest, size, path in sorted(hits, key=lambda x: -x[1]):
        flag = '  *** MATCH ***' if digest.startswith(want) else ''
        print('  %s  size=%-8d %s%s' % (digest[:16], size, path, flag))
    print()
print('scanned candidates:', scanned)
