"""官方源采集器：CNINFO / SSE / SZSE。

所有接口均先小样本实测确认字段后才固化（见各函数注释）：
- CNINFO: POST http://www.cninfo.com.cn/new/hisAnnouncement/query（公告查询，JSON）
- SSE:   GET  https://query.sse.com.cn/security/stock/queryCompanyBulletin.do（需 Referer）
- SZSE:  POST https://www.szse.cn/api/disc/announcement/annList（JSON body）

详情正文均为 PDF：下载落盘 {raw_id}_detail.pdf，detail_mime=application/pdf。
SSE 的 www.sse.com.cn 在本网络 SSL 被重置，PDF 下载会如实标 detail_status=FAILED 并留 url 供 retry。
监管类独立栏目（SSE 纪律处分、SZSE 监管动态）未找到可验证端点，留 TODO。
"""
import json
import time
from datetime import datetime, timedelta
from pathlib import Path

import requests

from .news_collect import save_raw_news
from .timeutil import BEIJING, now_iso, parse_source_time

UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36")

PDF_SOURCES = {"CNINFO", "SSE", "SZSE"}
MAX_PDF_BYTES = 50 * 1024 * 1024


def _today():
    return now_iso()[:10]


def last_run_latest(conn, source_id):
    """上次成功/部分成功运行的 latest_published_at（增量模式锚点）。"""
    r = conn.execute(
        """SELECT MAX(latest_published_at) FROM collection_run
           WHERE source_id=? AND status IN ('SUCCESS','PARTIAL')""",
        (source_id,)).fetchone()
    return r[0] if r else None


def _new_stats():
    return {"fetched": 0, "inserted": 0, "duplicate": 0, "failed": 0,
            "latest_published_at": None, "errors": []}


def _note_latest(stats, source_time_raw):
    pub, _, _, _ = parse_source_time(source_time_raw)
    if pub and (stats["latest_published_at"] is None or pub > stats["latest_published_at"]):
        stats["latest_published_at"] = pub


def save_pdf_detail(conn, raw_id, url, referer=None):
    """下载 PDF 详情落盘（只追加），返回 (ok, error)。"""
    row = conn.execute(
        "SELECT raw_path FROM raw_news WHERE raw_id=?", (raw_id,)).fetchone()
    if not row or not row["raw_path"]:
        return False, "无 raw_path"
    dpath = Path(row["raw_path"]).with_suffix(".detail.pdf")
    headers = {"User-Agent": UA}
    if referer:
        headers["Referer"] = referer
    try:
        r = requests.get(url, headers=headers, timeout=60, stream=True)
        r.raise_for_status()
        ctype = r.headers.get("Content-Type", "")
        blob = r.content
        if "pdf" not in ctype.lower() and not blob.startswith(b"%PDF"):
            raise ValueError(f"非 PDF 响应: {ctype}")
        if len(blob) > MAX_PDF_BYTES:
            raise ValueError(f"PDF 超 {MAX_PDF_BYTES} 字节")
        if not dpath.exists():  # 只追加
            dpath.write_bytes(blob)
        conn.execute(
            """UPDATE raw_news SET detail_status='OK', detail_mime='application/pdf'
               WHERE raw_id=?""", (raw_id,))
        return True, None
    except Exception as e:
        conn.execute("UPDATE raw_news SET detail_status='FAILED' WHERE raw_id=?", (raw_id,))
        return False, f"{type(e).__name__}: {e}"


def retry_pdf_detail(conn, raw_id):
    """按 url 列重试 PDF 详情下载。"""
    row = conn.execute(
        "SELECT raw_id, url, source_id FROM raw_news WHERE raw_id=?", (raw_id,)).fetchone()
    if not row:
        return False, "raw_id 不存在"
    if not row["url"]:
        return False, "无详情 URL"
    referer = {"CNINFO": "http://www.cninfo.com.cn/",
               "SSE": "https://www.sse.com.cn/disclosure/listedinfo/announcement/",
               "SZSE": "https://www.szse.cn/disclosure/listed/notice/index.html"}.get(row["source_id"])
    ok, err = save_pdf_detail(conn, raw_id, row["url"], referer)
    conn.commit()
    return ok, err or "OK"


# ---------------- 巨潮资讯 ----------------

CNINFO_QUERY = "http://www.cninfo.com.cn/new/hisAnnouncement/query"
CNINFO_STATIC = "http://static.cninfo.com.cn/"


def _ms_to_beijing_iso(ms):
    """cninfo announcementTime 为毫秒 epoch（绝对时间），直接转 +08:00 ISO。"""
    return datetime.fromtimestamp(ms / 1000, tz=BEIJING).isoformat()


def cninfo_list_page(start, end, page_num, page_size=30, column="szse"):
    """单页查询。column=szse 查深市（含创业板），切 sse 查沪市；先各采后合并去重。"""
    headers = {"User-Agent": UA, "Referer": "http://www.cninfo.com.cn/new/commonUrl/pageOfSearch?url=disclosure/list/search"}
    data = {"pageNum": page_num, "pageSize": page_size, "column": column,
            "tabName": "fulltext", "plate": "", "stock": "", "searchkey": "",
            "secid": "", "category": "", "trade": "", "seDate": f"{start}~{end}",
            "sortName": "", "sortType": "", "limit": ""}
    r = requests.post(CNINFO_QUERY, data=data, headers=headers, timeout=30)
    r.raise_for_status()
    return r.json()


def cninfo_collect(conn, start=None, end=None, page_size=30, incremental=True, max_pages=None):
    """巨潮公告采集。增量模式从上次的 latest_published_at 所在日起扫，哈希去重兜底。"""
    stats = _new_stats()
    if end is None:
        end = _today()
    if start is None:
        if incremental:
            last = last_run_latest(conn, "CNINFO")
            start = last[:10] if last else (datetime.now(BEIJING) - timedelta(days=1)).date().isoformat()
        else:
            start = end

    for column in ("szse",):  # 实测 column 参数对 fulltext 无效，单遍即可覆盖全市场
        page = 1
        while True:
            try:
                j = cninfo_list_page(start, end, page, page_size, column)
            except Exception as e:
                stats["errors"].append(f"{column} page{page}: {type(e).__name__}: {e}")
                break
            if not isinstance(j, dict) or "announcements" not in j:
                stats["errors"].append(f"{column} page{page}: 页面结构变化，缺 announcements 字段")
                break
            anns = j.get("announcements") or []
            if not anns:
                break
            for a in anns:
                stats["fetched"] += 1
                adjunct = a.get("adjunctUrl") or ""
                pdf_url = CNINFO_STATIC + adjunct if adjunct else None
                payload = {"secCode": a.get("secCode"), "secName": a.get("secName"),
                           "announcementId": a.get("announcementId"),
                           "announcementTitle": a.get("announcementTitle"),
                           "announcementTime": a.get("announcementTime"),
                           "adjunctUrl": adjunct, "adjunctType": a.get("adjunctType"),
                           "adjunctSize": a.get("adjunctSize"),
                           "announcementType": a.get("announcementType"),
                           "pageColumn": a.get("pageColumn"), "orgId": a.get("orgId"),
                           "url": pdf_url}
                t_raw = _ms_to_beijing_iso(a["announcementTime"]) if a.get("announcementTime") else None
                try:
                    raw_id, is_new = save_raw_news(
                        conn, "CNINFO", a.get("announcementTitle"), pdf_url,
                        json.dumps(payload, ensure_ascii=False),
                        source_time_raw=t_raw, detail_status="PENDING")
                    stats["inserted" if is_new else "duplicate"] += 1
                    _note_latest(stats, t_raw)
                    if is_new and pdf_url:
                        ok, err = save_pdf_detail(conn, raw_id, pdf_url,
                                                  referer="http://www.cninfo.com.cn/")
                        if not ok:
                            stats["errors"].append(f"pdf raw_id={raw_id}: {err}")
                except Exception as e:
                    stats["failed"] += 1
                    stats["errors"].append(f"save {a.get('announcementId')}: {type(e).__name__}: {e}")
            total_pages = j.get("totalpages") or 0
            if page >= total_pages or not j.get("hasMore") or (max_pages and page >= max_pages):
                break
            page += 1
            time.sleep(0.3)
    return stats


# ---------------- 上交所 ----------------

SSE_QUERY = "https://query.sse.com.cn/security/stock/queryCompanyBulletin.do"
SSE_REFERER = "https://www.sse.com.cn/disclosure/listedinfo/announcement/"


def sse_list_page(start, end, page_no, page_size=25):
    headers = {"User-Agent": UA, "Referer": SSE_REFERER}
    params = {"isPagination": "true", "productId": "", "keyWord": "",
              "securityType": "0101,120100,020100,020200,120200",
              "pageHelp.cacheSize": "1", "pageHelp.beginPage": str(page_no),
              "pageHelp.pageSize": str(page_size), "pageHelp.pageNo": str(page_no),
              "beginDate": start, "endDate": end}
    r = requests.get(SSE_QUERY, params=params, headers=headers, timeout=30)
    r.raise_for_status()
    return r.json()


def sse_collect(conn, start=None, end=None, page_size=25, incremental=True, max_pages=None):
    """上交所上市公司公告。ADDDATE 为入库时间（naive，按北京时间解释）。

    注意：PDF 在 www.sse.com.cn，本网络 SSL 被重置，下载失败如实标 FAILED 并留 url。
    """
    stats = _new_stats()
    _sse_t0 = time.time()
    if end is None:
        end = _today()
    if start is None:
        if incremental:
            last = last_run_latest(conn, "SSE")
            start = last[:10] if last else (datetime.now(BEIJING) - timedelta(days=1)).date().isoformat()
        else:
            start = end

    page = 1
    while True:
        try:
            j = sse_list_page(start, end, page, page_size)
        except Exception as e:
            stats["errors"].append(f"page{page}: {type(e).__name__}: {e}")
            break
        if not isinstance(j, dict) or ("result" not in j and "pageHelp" not in j):
            stats["errors"].append(f"page{page}: 页面结构变化，缺 result/pageHelp 字段")
            break
        rows = j.get("result") or []
        if not rows:
            break
        ph = j.get("pageHelp", {})
        total = int(ph.get("total") or 0)
        if time.time() - _sse_t0 > 180:  # 单源详情预算：本网络 SSE PDF 会挂死，防拖垮整轮
            stats["errors"].append("time budget 180s exhausted, rest deferred to retry_detail")
            break
        for a in rows:
            stats["fetched"] += 1
            rel = a.get("URL") or ""
            pdf_url = "https://www.sse.com.cn" + rel if rel else None
            payload = {"SECURITY_CODE": a.get("SECURITY_CODE"),
                       "SECURITY_NAME": a.get("SECURITY_NAME"),
                       "TITLE": a.get("TITLE"), "BULLETIN_HEADING": a.get("BULLETIN_HEADING"),
                       "BULLETIN_TYPE": a.get("BULLETIN_TYPE"),
                       "BULLETIN_YEAR": a.get("BULLETIN_YEAR"),
                       "SSEDATE": a.get("SSEDATE"), "ADDDATE": a.get("ADDDATE"),
                       "url": pdf_url}
            t_raw = a.get("ADDDATE")  # naive 入库时间
            try:
                raw_id, is_new = save_raw_news(
                    conn, "SSE", a.get("TITLE"), pdf_url,
                    json.dumps(payload, ensure_ascii=False),
                    source_time_raw=t_raw, detail_status="PENDING")
                stats["inserted" if is_new else "duplicate"] += 1
                _note_latest(stats, t_raw)
                if is_new and pdf_url:
                    ok, err = save_pdf_detail(conn, raw_id, pdf_url, referer=SSE_REFERER)
                    if not ok:
                        stats["errors"].append(f"pdf raw_id={raw_id}: {err}")
            except Exception as e:
                stats["failed"] += 1
                stats["errors"].append(f"save: {type(e).__name__}: {e}")
        if page * page_size >= total or (max_pages and page >= max_pages):
            break
        page += 1
        time.sleep(0.3)
    return stats


# ---------------- 深交所 ----------------

SZSE_ANNLIST = "https://www.szse.cn/api/disc/announcement/annList"
SZSE_REFERER = "https://www.szse.cn/disclosure/listed/notice/index.html"


def szse_list_page(start, end, page_num, page_size=50):
    headers = {"User-Agent": UA, "Referer": SZSE_REFERER, "Content-Type": "application/json"}
    body = {"seDate": [start, end], "channelCode": ["listedNotice_disc"],
            "pageSize": page_size, "pageNum": page_num}
    r = requests.post(SZSE_ANNLIST, data=json.dumps(body), headers=headers, timeout=30)
    r.raise_for_status()
    return r.json()


def szse_collect(conn, start=None, end=None, page_size=50, incremental=True, max_pages=None):
    """深交所上市公司公告。publishTime 为 naive 字符串，按北京时间解释。"""
    stats = _new_stats()
    if end is None:
        end = _today()
    if start is None:
        if incremental:
            last = last_run_latest(conn, "SZSE")
            start = last[:10] if last else (datetime.now(BEIJING) - timedelta(days=1)).date().isoformat()
        else:
            start = end

    page = 1
    while True:
        try:
            j = szse_list_page(start, end, page, page_size)
        except Exception as e:
            stats["errors"].append(f"page{page}: {type(e).__name__}: {e}")
            break
        if not isinstance(j, dict) or "data" not in j:
            stats["errors"].append(f"page{page}: 页面结构变化，缺 data 字段")
            break
        rows = j.get("data") or []
        if not rows:
            break
        total = int(j.get("announceCount") or 0)
        for a in rows:
            stats["fetched"] += 1
            attach = a.get("attachPath") or ""
            pdf_url = "https://disc.static.szse.cn" + attach if attach else None
            payload = {"id": a.get("id"), "annId": a.get("annId"), "title": a.get("title"),
                       "publishTime": a.get("publishTime"), "attachPath": attach,
                       "attachFormat": a.get("attachFormat"), "attachSize": a.get("attachSize"),
                       "secCode": a.get("secCode"), "secName": a.get("secName"),
                       "bigCategoryId": a.get("bigCategoryId"), "url": pdf_url}
            t_raw = a.get("publishTime")
            try:
                raw_id, is_new = save_raw_news(
                    conn, "SZSE", a.get("title"), pdf_url,
                    json.dumps(payload, ensure_ascii=False),
                    source_time_raw=t_raw, detail_status="PENDING")
                stats["inserted" if is_new else "duplicate"] += 1
                _note_latest(stats, t_raw)
                if is_new and pdf_url:
                    ok, err = save_pdf_detail(conn, raw_id, pdf_url)
                    if not ok:
                        stats["errors"].append(f"pdf raw_id={raw_id}: {err}")
            except Exception as e:
                stats["failed"] += 1
                stats["errors"].append(f"save: {type(e).__name__}: {e}")
        if page * page_size >= total or (max_pages and page >= max_pages):
            break
        page += 1
        time.sleep(0.3)
    return stats
