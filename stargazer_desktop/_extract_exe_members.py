"""只读解析 PyInstaller onefile 归档，查找并可选提取指定 .py 成员。

不写入 C 盘；输出目录由调用方指定（默认 D 盘）。
"""
import argparse
import struct
import sys
from pathlib import Path

MAGIC = b'MEI\014\013\012\013\016'


def find_cookie(data):
    tail = data[-8192:]
    idx = tail.rfind(MAGIC)
    if idx < 0:
        raise SystemExit('cookie magic not found')
    base = len(data) - len(tail) + idx
    (pkg_len, toc_off, toc_len, pyvers) = struct.unpack('!IIII', data[base + 8: base + 24])
    return base, pkg_len, toc_off, toc_len, pyvers


def read_toc(data, cookie):
    _, _, toc_off, toc_len, _ = cookie
    end = toc_off + toc_len
    entries = []
    pos = toc_off
    while pos < end:
        entry_len, offset, length, ulen, typecode = struct.unpack('!IIIIB', data[pos: pos + 17])
        name = data[pos + 18: pos + entry_len].split(b'\0', 1)[0].decode('utf8', 'replace')
        entries.append({'name': name, 'offset': offset, 'length': length,
                        'ulen': ulen, 'typecode': typecode})
        pos += entry_len
    return entries


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('exe')
    ap.add_argument('--grep', default=None)
    ap.add_argument('--extract', default=None, help='output dir (D drive)')
    args = ap.parse_args()

    path = Path(args.exe)
    data = path.read_bytes()
    cookie = find_cookie(data)
    print('exe:', path, 'size:', len(data))
    print('cookie: pkg_len=%d toc_off=%d toc_len=%d pyvers=%d' % cookie[1:])
    toc = read_toc(data, cookie)
    print('toc entries:', len(toc))
    print('typecode histogram:', {chr(t): sum(1 for e in toc if e['typecode'] == t)
                                  for t in {e['typecode'] for e in toc}})

    needle = (args.grep or '').lower()
    hits = [e for e in toc if needle and needle in e['name'].lower()]
    if needle:
        print('\n--- matches for %r ---' % args.grep)
        for e in hits:
            print('  [%s] %-70s len=%d ulen=%d' % (chr(e['typecode']), e['name'],
                                                   e['length'], e['ulen']))

    if args.extract and hits:
        out = Path(args.extract)
        out.mkdir(exist_ok=True)
        for e in hits:
            body = data[e['offset']: e['offset'] + e['length']]
            target = out / e['name']
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(body)
            print('extracted ->', target, len(body))


if __name__ == '__main__':
    sys.exit(main())
