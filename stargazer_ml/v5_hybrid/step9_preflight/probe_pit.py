"""PIT Sector 探活 (①②③ 入口)。

用 SDK 探查:
- bk.get 返回记录是否含任何 as_of / effective_date 字段 (我们之前只看到 sample_rows=1)
- bk 是否支持 as_of 参数
- bk / rd 可用方法清单 (发现历史快照表 / 分区)
"""
import sys
import os
import data_availability_audit as a

ok = a._init_sdk()
print("sdk_init", ok)
if not ok:
    raise SystemExit(1)
sdk = a._SDK

print("== bk methods ==", [m for m in dir(sdk.bk) if not m.startswith("_")])
print("== rd methods ==", [m for m in dir(sdk.rd) if not m.startswith("_")])

print("== bk.get(600000, category=1) full record ==")
try:
    info = sdk.bk.get(x="600000", category=1)
    print("type", type(info))
    if isinstance(info, list):
        for it in info[:3]:
            print("  elem", (list(it.keys()) if isinstance(it, dict) else it), "->", str(it)[:240])
    elif isinstance(info, dict):
        print("  keys", list(info.keys()), "->", str(info)[:400])
except Exception as e:  # noqa
    print("err", e)

print("== try bk.get with as_of ==")
for ao in ("20200101", "20250101", "20260916"):
    try:
        info2 = sdk.bk.get(x="600000", category=1, as_of=ao)
        print(f"  as_of={ao}: type={type(info2)} {str(info2)[:160]}")
    except Exception as e:  # noqa
        print(f"  as_of={ao}: ERR {e}")
