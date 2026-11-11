"""通达信盘后日线适配器（迁移第2步，只读/规范化层）。

读取 TDX vipdoc 日线(.day) + T0002/hq_cache/base.dbf(股本/行业，单位:万股)
+ stockdb 日线 parquet 的名称/ST(只读参照，有效证券名称表)，
输出与现有 stockdb 日线 parquet 同契约的规范日线（raw 视图可用；qfq 视图待 权息资料解析）。

设计纪律（来自 PLAN 第2步）：
- 不覆盖旧 data_daily_parts；新数据走独立缓存版本 CACHE_VERSION。
- 原始日线与复权视图分开；qfq 必须由真实公司行为资料生成，禁止用当日收盘比例补整段历史。
- 基础资料（股本/名称/ST）缺失时记录原因，不默认非ST或用0市值。
- 单位：price/100=>元；volume=>股；amount=>元；base.dbf 股本字段=>万股=>×10000=>股。
"""
import os
import struct
import glob
import datetime as dt
import pandas as pd

def _tdx_root_default():
    """TDX vipdoc 路径兜底：本机私有路径不入库。
    环境变量 > 模块旁 data/tdx_root.txt（已 gitignore）> 通用占位符。"""
    override = Path(__file__).resolve().parent / 'data' / 'tdx_root.txt'
    if override.is_file():
        text = override.read_text(encoding='utf-8').strip()
        if text:
            return text
    return os.environ.get('STARGAZER_TDX_DEFAULT') or 'D:/tdx/vipdoc'


DEFAULT_VIPDOC = None  # 本机私有路径不入库：环境变量或 data/tdx_root.txt 提供
TDX_ROOT = os.environ.get("STARGAZER_TDX_ROOT") or _tdx_root_default()
INSTALL = os.path.dirname(TDX_ROOT)
T0002 = os.path.join(INSTALL, "T0002")
BASE_DBF = os.path.join(T0002, "hq_cache", "base.dbf")
GBBQ = os.path.join(T0002, "hq_cache", "gbbq")
GBBQ_MAP = os.path.join(T0002, "hq_cache", "gbbq.map")
SDB_PARTS = os.environ.get("STARGAZER_STOCKDB_PARTS", r"D:/a股数据/data_daily_parts")
CACHE_VERSION = "tdx_daily_v1"

DAY_FMT = "<iiiiifii"  # date,o,h,l,c,amount(f),volume(i),reserved  (32B)

# 我们的内部代码(去点大写) -> TDX 文件代码；指数需要映射（沪指文件为 sh999999）
INDEX_MAP = {"000001SH": "sh999999", "399006SZ": "sz399006", "000300SH": "sh000300"}


def to_tdx_file(our_code):
    k = our_code.upper().replace(".", "")
    if k in INDEX_MAP:
        return INDEX_MAP[k]
    mkt = "sh" if k[0] in "69" else ("bj" if k[0] in "48" else "sz")
    return mkt + k


# ---------------------------------------------------------------------------
# 1) TDX 原始日线（实测解码，price×100、volume=股、amount=元）
# ---------------------------------------------------------------------------
def read_day_raw(tdx_file):
    """tdx_file 如 'sh600000'；返回 DataFrame[date,open,high,low,close,volume,amount] 或 None。"""
    mkt, code = tdx_file[:2], tdx_file[2:]
    p = os.path.join(TDX_ROOT, mkt, "lday", tdx_file + ".day")
    if not os.path.exists(p):
        return None
    with open(p, "rb") as f:
        blob = f.read()
    n = len(blob) // 32
    if n == 0:
        return None
    rows = []
    for i in range(n):
        d, o, h, l, c, amt, vol, _ = struct.unpack(DAY_FMT, blob[i * 32:(i + 1) * 32])
        rows.append((d, o / 100.0, h / 100.0, l / 100.0, c / 100.0, int(vol), float(amt)))
    df = pd.DataFrame(rows, columns=["date", "open", "high", "low", "close", "volume", "amount"])
    return df


# ---------------------------------------------------------------------------
# 2) base.dbf：股本/行业（单位 万股），保留删除标记、偏移+1
# ---------------------------------------------------------------------------
def load_base():
    out = {}
    if not os.path.exists(BASE_DBF):
        return out
    with open(BASE_DBF, "rb") as f:
        head = f.read(32)
        hlen = int.from_bytes(head[8:10], "little")
        rlen = int.from_bytes(head[10:12], "little")
        data = f.read(hlen - 32)
    fields, off = [], 0
    while off + 32 <= len(data) and data[off] != 0x0D:
        nm = data[off:off + 11].split(b"\x00")[0].decode("gbk", "replace").strip()
        fields.append((nm, data[off + 11], data[off + 16]))
        off += 32
    fmap = {}
    cur = 0
    for nm, typ, flen in fields:
        fmap[nm] = (cur, flen)
        cur += flen
    gp, sc = fmap.get("GPDM"), fmap.get("SC")
    if gp is None:
        return out
    need = {x: fmap.get(x) for x in ("ZGB", "LTAG", "HY", "DY", "SSDATE")}

    def num(fld, rec):
        if fld is None:
            return None
        s = rec[fld[0]:fld[0] + fld[1]].decode("ascii", "replace").strip()
        try:
            return float(s) if s not in ("", "-") else None
        except ValueError:
            return None

    with open(BASE_DBF, "rb") as f:
        f.seek(hlen)
        blob = f.read()
    n = len(blob) // rlen if rlen else 0
    for i in range(n):
        rec = blob[i * rlen:(i + 1) * rlen][1:]   # 去删除标记
        code = rec[gp[0]:gp[0] + 6].decode("ascii", "replace").strip()
        if not code.isdigit():
            continue
        sc_v = rec[sc[0]:sc[0] + 1].decode("ascii", "replace").strip() if sc else ""
        mkt = "sh" if sc_v == "1" else "sz"
        out[f"{mkt}{code}"] = {
            "zgb_wan": num(need["ZGB"], rec), "ltag_wan": num(need["LTAG"], rec),
            "hy": (rec[need["HY"][0]:need["HY"][0] + need["HY"][1]].decode("gbk", "replace").strip() if need["HY"] else ""),
            "dy": (rec[need["DY"][0]:need["DY"][0] + need["DY"][1]].decode("gbk", "replace").strip() if need["DY"] else ""),
            "ssdate": (rec[need["SSDATE"][0]:need["SSDATE"][0] + need["SSDATE"][1]].decode("ascii", "replace").strip() if need["SSDATE"] else ""),
        }
    return out


# ---------------------------------------------------------------------------
# 3) 名称/ST：stockdb 日线 parquet（有效证券名称表，只读参照）
# ---------------------------------------------------------------------------
def load_stockdb_names_st():
    out = {}
    if not os.path.isdir(SDB_PARTS):
        return out
    try:
        files = sorted(glob.glob(os.path.join(SDB_PARTS, "part_*.parquet")))
        if not files:
            return out
        cols = pd.read_parquet(files[0]).columns
        if "name" not in cols:
            return out
        use = ["code", "name", "is_st"] if "is_st" in cols else ["code", "name"]
        df = pd.concat([pd.read_parquet(f, columns=use) for f in files], ignore_index=True)
        df["code"] = df["code"].astype(str).str.zfill(6)
        g = df.dropna(subset=["name"]).groupby("code")
        for c, sub in g:
            out[str(c)] = (str(sub["name"].iloc[-1]), bool(sub.get("is_st", pd.Series([False])).iloc[-1]))
    except Exception:
        pass
    return out


# ---------------------------------------------------------------------------
# 4) 规范化日线：raw + 派生字段（float_share/float_mv/pre_close）
#    返回 (df, meta)；qfq 视图占位（待 权息资料解析）。
# ---------------------------------------------------------------------------
def get_daily(our_code, start=None, end=None):
    tdx_file = to_tdx_file(our_code)
    raw = read_day_raw(tdx_file)
    meta = {"our_code": our_code, "tdx_file": tdx_file, "raw_rows": 0,
            "qfq_status": "pending_gbbq", "float_mv_source": None,
            "name_source": "stockdb_parquet"}
    if raw is None:
        meta["error"] = "raw day file missing"
        return None, meta
    meta["raw_rows"] = len(raw)
    if start:
        raw = raw[raw["date"] >= int(start)]
    if end:
        raw = raw[raw["date"] <= int(end)]
    raw = raw.sort_values("date").reset_index(drop=True)
    raw["pre_close"] = raw["close"].shift(1)
    raw["pct_chg"] = (raw["close"] / raw["pre_close"] - 1) * 100
    # 股本/市值（base.dbf 单位 万股 => 股）
    base = load_base()
    b = base.get(tdx_file, {})
    fs_wan = b.get("ltag_wan")
    ts_wan = b.get("zgb_wan")
    raw["float_share"] = (fs_wan * 10000.0) if fs_wan is not None else None   # 股
    raw["total_share"] = (ts_wan * 10000.0) if ts_wan is not None else None   # 股
    raw["float_mv"] = (raw["float_share"] * raw["close"]) if fs_wan is not None else None
    raw["total_mv"] = (raw["total_share"] * raw["close"]) if ts_wan is not None else None
    raw["turnover"] = (raw["volume"] / raw["float_share"] * 100.0) if fs_wan else None
    # 名称/ST（参照）
    names = load_stockdb_names_st()
    key6 = our_code.upper().replace(".", "")
    nm, st = names.get(key6, (None, False))
    raw["code"] = key6
    raw["name"] = nm
    raw["is_st"] = st
    meta["float_mv_source"] = "base.dbf(LTAG×10000)×close" if fs_wan is not None else "MISSING"
    return raw, meta


if __name__ == "__main__":
    import json
    for c in ["600000", "000001.SH", "300845", "399006.SZ", "002313"]:
        df, meta = get_daily(c, "20261001", "20261030")
        print(f"\n=== {c} -> {meta['tdx_file']} rows={meta['raw_rows']} qfq={meta['qfq_status']} mv={meta['float_mv_source']}")
        if df is not None:
            print(df.tail(3).to_string(index=False))
            print("float_share=", df['float_share'].iloc[-1], "float_mv=", df['float_mv'].iloc[-1],
                  "name=", df['name'].iloc[-1], "is_st=", df['is_st'].iloc[-1])
