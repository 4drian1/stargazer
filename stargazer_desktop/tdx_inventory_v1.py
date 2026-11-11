"""通达信盘后数据迁移 — 第1步只读盘点（不改任何生产数据/配置）。

读取 STARGAZER_TDX_ROOT（默认回退到 index_minutes_v1 的 vipdoc 路径），
盘 daily/5分钟 覆盖、每股首末日期、条数、最新完整日、文件指纹，并交叉 base.dbf
确认漏下载；同时登记 股本/除权/板块 资料文件是否齐全。

输出：migration_support/tdx_inventory_20261030.json + 同名 .md 缺口表。
"""
import os
import sys
import json
import glob
import struct
import hashlib
import datetime as dt

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

def _tdx_root_default():
    """TDX vipdoc 路径兜底：本机私有路径不入库。
    环境变量 > 模块旁 data/tdx_root.txt（已 gitignore）> 通用占位符。"""
    from pathlib import Path as _Path
    override = _Path(__file__).resolve().parent / 'data' / 'tdx_root.txt'
    if override.is_file():
        text = override.read_text(encoding='utf-8').strip()
        if text:
            return text
    return os.environ.get('STARGAZER_TDX_DEFAULT') or 'D:/tdx/vipdoc'


DEFAULT_VIPDOC = None  # 本机私有路径不入库：环境变量或 data/tdx_root.txt 提供

VIPDOC = os.environ.get("STARGAZER_TDX_ROOT") or _tdx_root_default()
INSTALL = os.path.dirname(VIPDOC)            # 安装根（含 vipdoc/T0002）
T0002 = os.path.join(INSTALL, "T0002")
# 真表在 T0002/hq_cache/base.dbf（含股本/行业）；安装根 base.dbf 是空壳，仅回退
BASE_DBF_PRIMARY = os.path.join(T0002, "hq_cache", "base.dbf")
BASE_DBF_FALLBACK = os.path.join(INSTALL, "base.dbf")
# 只读参照：有效证券名称表（stockdb 日线 parquet 含 name 列），用于 ST 状态，不写入生产
SDB_PARTS = os.environ.get("STARGAZER_STOCKDB_PARTS", r"D:/a股数据/data_daily_parts")

DAY_FMT = "<iiiiifii"   # date,o,h,l,c,amount(float),volume,reserved  (32B)
LC5_FMT = "<iiiiifii"   # date,int time,o,h,l,c,amount,volume,reserved (32B)

SPOT = {  # 定点抽查（market, code）— 规则登记在报告里，先看结果前锁定
    "sh600000": ("sh", "600000"), "sz002313": ("sz", "002313"),
    "sz300845": ("sz", "300845"), "sh688651": ("sh", "688651"),
    "sz002933": ("sz", "002933"), "sz000001": ("sz", "000001"),
    "sh999999(沪指)": ("sh", "999999"), "sz399006": ("sz", "399006"),
}
RANDOM_SEED = 20261030
RANDOM_PER_MARKET = 8


def is_index(code):
    m = code[:2]
    d = code[2:]
    if m == "sh":
        return d[:3] in ("000", "999")
    if m == "sz":
        return d[:3] == "399"
    return False


def read_day_dates(path):
    sz = os.path.getsize(path)
    if sz == 0 or sz % 32 != 0:
        return None, None, sz, (sz != 0 and sz % 32 != 0)
    with open(path, "rb") as f:
        first = struct.unpack(DAY_FMT, f.read(32))
        if sz >= 32:
            f.seek(sz - 32)
            last = struct.unpack(DAY_FMT, f.read(32))
    return first[0], last[0], sz, False


def ymd(v):
    return dt.datetime.strptime(str(v), "%Y%m%d").date()


def parse_base_dbf(path):
    """返回 {market+code: {"zgb","ltag","hy","dy","ssdate"}} 股本/行业字段。
    注：本 DBF 无证券名称/ST，名称与 ST 须另从有效证券名称表读取（见 load_stockdb_names）。"""
    out = {}
    if not os.path.exists(path):
        return out
    with open(path, "rb") as f:
        head = f.read(32)
        hlen = int.from_bytes(head[8:10], "little")
        rlen = int.from_bytes(head[10:12], "little")
        data = f.read(hlen - 32)
    fields = []
    off = 0
    while off + 32 <= len(data) and data[off] != 0x0D:
        nm = data[off:off + 11].split(b"\x00")[0].decode("gbk", "replace").strip()
        typ = chr(data[off + 11])
        flen = data[off + 16]
        fields.append((nm, typ, flen))
        off += 32
    fmap = {}
    cur = 0
    for nm, typ, flen in fields:
        fmap[nm] = (cur, flen, typ)
        cur += flen
    sc = fmap.get("SC")
    gp = fmap.get("GPDM")
    if gp is None:
        return out

    def _num(fld, rec):
        if fld is None:
            return None
        s = rec[fld[0]:fld[0] + fld[1]].decode("ascii", "replace").strip()
        if s in ("", "-"):
            return None
        try:
            return float(s)
        except ValueError:
            return None

    need = {k: fmap.get(k) for k in ("ZGB", "LTAG", "HY", "DY", "SSDATE")}
    with open(path, "rb") as f:
        f.seek(hlen)
        blob = f.read()
    n = len(blob) // rlen if rlen else 0
    for i in range(n):
        raw = blob[i * rlen:(i + 1) * rlen]   # 含 1 字节删除标记
        rec = raw[1:]                          # 去掉标记后字段偏移与描述符一致
        code = rec[gp[0]:gp[0] + 6].decode("ascii", "replace").strip()
        if not code.isdigit():
            continue
        sc_v = rec[sc[0]:sc[0] + 1].decode("ascii", "replace").strip() if sc else ""
        mkt = "sh" if sc_v == "1" else "sz"
        out[f"{mkt}{code}"] = {
            "zgb": _num(need["ZGB"], rec),      # 总股本，单位：万股（已与 stockdb total_share 核对）
            "ltag": _num(need["LTAG"], rec),     # 流通A股，单位：万股
            "hy": (rec[need["HY"][0]:need["HY"][0] + need["HY"][1]].decode("gbk", "replace").strip() if need["HY"] else ""),
            "dy": (rec[need["DY"][0]:need["DY"][0] + need["DY"][1]].decode("gbk", "replace").strip() if need["DY"] else ""),
            "ssdate": (rec[need["SSDATE"][0]:need["SSDATE"][0] + need["SSDATE"][1]].decode("ascii", "replace").strip() if need["SSDATE"] else ""),
        }
    return out


def load_stockdb_names(parts_dir):
    """只读参照：从 stockdb 日线 parquet 的 name 列构建 code->name（有效证券名称表）。
    base.dbf 无名称字段，绝不在本脚本伪造名称；ST 由调用方据 name 判定。"""
    names = {}
    if not os.path.isdir(parts_dir):
        return names
    try:
        import pandas as pd, glob
        files = sorted(glob.glob(os.path.join(parts_dir, "part_*.parquet")))
        if not files:
            return names
        cols = pd.read_parquet(files[0]).columns
        if "name" not in cols:
            return names
        use = ["code", "name"] if "code" in cols else ["name"]
        df = pd.concat([pd.read_parquet(f, columns=use) for f in files], ignore_index=True)
        if "code" in df.columns:
            df["code"] = df["code"].astype(str).str.zfill(6)
            for c, nm in df.dropna(subset=["name"]).groupby("code")["name"].last().items():
                names[str(c)] = str(nm)
        else:
            names["*"] = str(df["name"].dropna().iloc[-1]) if len(df) else ""
    except Exception:
        pass
    return names


def main():
    started = dt.datetime.now()
    rep = {"meta": {}, "markets": {}, "spot_checks": {}, "random_sample": {},
           "gaps": {}, "aux_files": {}, "minute_5": {}, "base_dbf": {}}
    rep["meta"] = {
        "tdx_root_env": os.environ.get("STARGAZER_TDX_ROOT"),
        "vipdoc": VIPDOC, "install_root": INSTALL,
        "vipdoc_exists": os.path.isdir(VIPDOC),
        "t0002_exists": os.path.isdir(T0002),
        "base_dbf_exists": os.path.exists(BASE_DBF_PRIMARY),
        "scanned_at": started.isoformat(timespec="seconds"),
        "verdict": "PENDING",
    }
    if not os.path.isdir(VIPDOC):
        print("VIPDOC 不存在：", VIPDOC)
        return rep

    # ---- daily ----
    daily_files = []  # (key, mkt, code, first, last, n, corrupt)
    per_mkt = {"sh": {"stock": 0, "index": 0, "recs": 0},
               "sz": {"stock": 0, "index": 0, "recs": 0},
               "bj": {"stock": 0, "index": 0, "recs": 0}}
    latest_set = set()
    fingerprint_src = []
    total = 0
    for mkt in ("sh", "sz", "bj"):
        d = os.path.join(VIPDOC, mkt, "lday")
        if not os.path.isdir(d):
            continue
        for fn in os.listdir(d):
            if not fn.endswith(".day"):
                continue
            total += 1
            code = fn[:-4]                 # 已含 sh/sz/bj 前缀，如 sh600000
            path = os.path.join(d, fn)
            first, last, sz, corrupt = read_day_dates(path)
            n = sz // 32 if sz else 0
            mkt = code[:2]
            idx = is_index(code)
            kind = "index" if idx else "stock"
            per_mkt[mkt][kind] += 1
            per_mkt[mkt]["recs"] += n
            if first and last:
                latest_set.add(last)
                fingerprint_src.append(f"{code}:{n}:{last}")
            daily_files.append((code, mkt, code, first, last, n, corrupt, sz))

    latest = max(latest_set) if latest_set else 0
    # 分类
    gaps = {"corrupt": [], "short_history": [], "new_listing": [],
            "stale_suspect": [], "missing_daily_vs_base": []}
    LATEST_D = ymd(latest) if latest else None
    for key, mkt, code, first, last, n, corrupt, sz in daily_files:
        if corrupt or n == 0:
            gaps["corrupt"].append(key); continue
        fd = ymd(first); ld = ymd(last)
        if n < 60:
            gaps["short_history"].append((key, str(fd), str(ld), n))
        if LATEST_D and (LATEST_D - fd).days < 365 and n < 200:
            gaps["new_listing"].append((key, str(fd), str(ld), n))
        if LATEST_D and (LATEST_D - ld).days > 14:
            gaps["stale_suspect"].append((key, str(ld), n))

    rep["markets"] = {m: {**v, "files": per_mkt[m]["stock"] + per_mkt[m]["index"]} for m, v in per_mkt.items()}
    rep["meta"]["total_day_files"] = total
    rep["meta"]["latest_complete_date"] = str(LATEST_D) if LATEST_D else None
    rep["meta"]["global_first_seen"] = str(min(ymd(r[3]) for r in daily_files if r[3]))
    fp = hashlib.sha256("\n".join(sorted(fingerprint_src)).encode("utf-8")).hexdigest()[:16]
    rep["meta"]["daily_set_fingerprint"] = fp

    # ---- base.dbf（T0002/hq_cache 真表）交叉 + stockdb 名称表（只读参照，用于 ST） ----
    b_primary = parse_base_dbf(BASE_DBF_PRIMARY)
    base = b_primary or parse_base_dbf(BASE_DBF_FALLBACK)
    used_path = BASE_DBF_PRIMARY if b_primary else (BASE_DBF_FALLBACK if os.path.exists(BASE_DBF_FALLBACK) else None)
    rep["base_dbf"] = {"exists": bool(base), "count": len(base), "empty": len(base) == 0,
                       "path_used": used_path, "name_source": "stockdb_parquet"}
    day_keys = {r[0] for r in daily_files}
    names = load_stockdb_names(SDB_PARTS)
    sdb_codes = set(names.keys())

    def to_tdx_code(c):
        c = str(c).strip()
        if c.startswith(("6", "9")):
            return "sh" + c
        if c.startswith(("4", "8")):
            return "bj" + c
        return "sz" + c  # 0/2/3 开头为深市

    sdb_codes_tdx = {to_tdx_code(c) for c in sdb_codes}
    rep["base_dbf"]["name_source_available"] = bool(sdb_codes)
    # 漏下载：以 stockdb 股票池（纯 A 股权益）为应下清单，标准化为 TDX 代码后缺日线者
    missing = sorted(sdb_codes_tdx - day_keys)
    gaps["missing_daily_vs_base"] = missing
    rep["base_dbf"]["missing_daily_count"] = len(missing)
    # 同时保留 base.dbf 全量交叉（含基金/债券/指数/港股等非 A 股，仅作参考）
    rep["base_dbf"]["base_total"] = len(base)
    rep["base_dbf"]["base_not_in_daily_raw"] = len([k for k in base if k not in day_keys])
    st = [k for k, v in names.items() if any(x in v for x in ("ST", "退"))]
    rep["base_dbf"]["st_count"] = len(st)
    rep["base_dbf"]["st_sample"] = st[:20]

    # ---- 5分钟 ----
    min5 = {}
    for mkt in ("sh", "sz"):
        d = os.path.join(VIPDOC, mkt, "fzline")
        cnt = 0; bytes_ = 0; half = 0
        if os.path.isdir(d):
            for fn in os.listdir(d):
                if not fn.endswith(".lc5"):
                    continue
                cnt += 1
                sz = os.path.getsize(os.path.join(d, fn))
                bytes_ += sz
                if sz % 32 != 0:
                    half += 1
        min5[mkt] = {"files": cnt, "bytes": bytes_, "half_written": half}
    rep["minute_5"] = min5
    rep["minute_5"]["note"] = ("下载中：未校验48槽完整性；half_written>0 表示可能半写。"
                               "与日线文件数对比可见5分钟覆盖进度。")

    # ---- 辅助资料文件：股本/除权/板块 ----
    aux = {}
    patterns = ["*weight*", "*gbbq*", "*bonus*", "*right*", "*cq*", "*pg*",
                "*block*", "*blk*", "*hy_*", "*gn_*", "*fg_*", "*cs_*"]
    for label, pat in [("dividend_rights", "*weight*"), ("dividend_rights2", "*gbbq*"),
                       ("block_sector", "*block*"), ("block_sector2", "*blk*"),
                       ("sector_hy", "*hy_*"), ("sector_gn", "*gn_*"),
                       ("sector_fg", "*fg_*")]:
        hits = []
        for root, _, files in os.walk(T0002):
            for fn in files:
                if any(pat.replace("*", "") in fn.lower() for pat in [pat]):
                    hits.append(os.path.join(root, fn))
        aux[label] = {"pattern": pat, "count": len(hits),
                      "sample": [os.path.relpath(h, INSTALL) for h in hits[:5]]}
    aux["base_dbf"] = {"path": BASE_DBF_PRIMARY, "exists": os.path.exists(BASE_DBF_PRIMARY)}
    rep["aux_files"] = aux

    # ---- 定点抽查 + 随机小样本 ----
    import random
    rnd = random.Random(RANDOM_SEED)
    for key, (mkt, code) in SPOT.items():
        path = os.path.join(VIPDOC, mkt, "lday", f"{mkt}{code}.day")
        item = {"exists": os.path.exists(path)}
        if os.path.exists(path):
            with open(path, "rb") as f:
                first = struct.unpack(DAY_FMT, f.read(32))
                sz = os.path.getsize(path)
                f.seek(sz - 32)
                last = struct.unpack(DAY_FMT, f.read(32))
            item.update({"first_date": str(ymd(first[0])), "last_date": str(ymd(last[0])),
                         "bars": sz // 32, "last_ohlc_x100": [last[1], last[2], last[3], last[4]],
                         "last_amount_yuan": last[5], "last_volume_shares": last[6]})
        rep["spot_checks"][key] = item
    # 随机样本（看结果前锁定种子）
    day_by_mkt = {"sh": [], "sz": []}
    for r in daily_files:
        if r[1] in day_by_mkt and not is_index(r[2]):
            day_by_mkt[r[1]].append(r[0])
    for mkt, keys in day_by_mkt.items():
        keys.sort()
        sample = rnd.sample(keys, min(RANDOM_PER_MARKET, len(keys)))
        for k in sorted(sample):
            rec = next(x for x in daily_files if x[0] == k)
            rep["random_sample"].setdefault(mkt, {})[k] = {
                "first": str(ymd(rec[3])) if rec[3] else None,
                "last": str(ymd(rec[4])) if rec[4] else None, "bars": rec[5]}

    # ---- 结论 ----
    if LATEST_D:
        age = (dt.date.today() - LATEST_D).days
        rep["meta"]["verdict"] = ("CURRENT" if age <= 2 else f"STALE({age}d)")
    rep["gaps"]["counts"] = {k: (len(v) if isinstance(v, list) else v) for k, v in gaps.items()}

    # ---- 写盘 ----
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "migration_support")
    out_dir = os.path.abspath(out_dir)
    os.makedirs(out_dir, exist_ok=True)
    jp = os.path.join(out_dir, "tdx_inventory_20261030.json")
    mp = os.path.join(out_dir, "tdx_inventory_20261030.md")
    with open(jp, "w", encoding="utf-8") as f:
        json.dump(rep, f, ensure_ascii=False, indent=2)
    # 可读缺口表
    lines = []
    m = rep["meta"]
    lines.append(f"# 通达信盘后数据盘点（只读，第1步） {m['scanned_at']}\n")
    lines.append(f"- vipdoc: `{m['vipdoc']}`")
    lines.append(f"- STARGAZER_TDX_ROOT 环境变量: {m['tdx_root_env'] or '未设置（用默认回退）'}")
    lines.append(f"- 最新完整日: **{m['latest_complete_date']}**  判定: **{m['verdict']}**")
    lines.append(f"- 全市场日线文件数: {m['total_day_files']}；日线集合指纹: {m['daily_set_fingerprint']}\n")
    lines.append("## 市场覆盖")
    for mk, v in rep["markets"].items():
        lines.append(f"- {mk}: 股票 {v['stock']} / 指数 {v['index']} / 日线条数 {v['recs']}")
    b = rep["base_dbf"]
    lines.append(f"\n## base.dbf（证券基础/股本/行业；名称/ST 来自 stockdb 名称表）")
    lines.append(f"- 真表路径: `{b.get('path_used')}`；条目: {b['count']}；ST/退(据名称): {b['st_count']}")
    lines.append(f"- 名称/ST 来源: stockdb 日线 parquet（有效证券名称表，只读参照）；base.dbf 本身无名称字段，不伪造。")
    if b.get("empty"):
        lines.append("  - **空文件（0条）**：证券基础/股本/ST 来源未下载（TDX 基本资料缺失），"
                     "漏下载交叉验证不可用；需补 TDX 基本资料/财务数据后再做漏下载核验。")
    lines.append(f"- 与日线交叉：base 有但缺日线文件 **{b['missing_daily_count']}** 条（漏下载候选）")
    if b['missing_daily_count'] and b['missing_daily_count'] <= 60:
        lines.append("  - " + ", ".join(b['missing_daily_count'] and gaps['missing_daily_vs_base']))
    lines.append(f"\n## 5分钟（下载中，未校验48槽）")
    for mk, v in rep["minute_5"].items():
        if isinstance(v, dict) and "files" in v:
            lines.append(f"- {mk}: 文件 {v['files']}；半写 {v['half_written']}")
    g = rep["gaps"]["counts"]
    lines.append(f"\n## 缺口分类（疑似，需交易日历确认）")
    lines.append(f"- 损坏/空文件: {g['corrupt']}")
    lines.append(f"- 历史不足/新上市候选: {g['short_history']}")
    lines.append(f"- 陈旧疑似停牌(落后>LATEST 14天): {g['stale_suspect']}")
    lines.append(f"- 漏下载(基dict缺日线): {g['missing_daily_vs_base']}")
    lines.append(f"\n## 定点抽查")
    for k, v in rep["spot_checks"].items():
        if v.get("exists"):
            lines.append(f"- {k}: {v['first_date']}→{v['last_date']} 共{v['bars']}条 "
                         f"末收{v['last_ohlc_x100'][3]/100:.2f} 量{v['last_volume_shares']}股")
        else:
            lines.append(f"- {k}: **文件缺失**")
    lines.append(f"\n## 随机小样本（seed={RANDOM_SEED}）")
    for mk, items in rep["random_sample"].items():
        lines.append(f"- {mk}: " + ", ".join(f"{k}({v['first']}→{v['last']},{v['bars']})" for k, v in items.items()))
    with open(mp, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    # stdout 摘要
    print("VIPDOC:", VIPDOC)
    print("最新完整日:", m["latest_complete_date"], "判定:", m["verdict"])
    print("日线文件数:", m["total_day_files"], "指纹:", m["daily_set_fingerprint"])
    print("市场:", {k: (v["stock"], v["index"]) for k, v in rep["markets"].items()})
    print("base.dbf 条目:", b["count"], "ST/退:", b["st_count"], "漏下载候选:", b["missing_daily_count"])
    print("5分钟:", {k: v.get("files") for k, v in rep["minute_5"].items() if isinstance(v, dict)})
    print("缺口计数:", g)
    print("写入:", jp)
    print("写入:", mp)
    return rep


if __name__ == "__main__":
    main()
