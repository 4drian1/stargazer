"""新闻快讯：华尔街见闻7x24。公开接口。"""
import json
import time
import urllib.request

UA = {"User-Agent": "Mozilla/5.0"}


def flash(pages=2, limit=100):
    """拉取最新快讯。返回 [{time, title, content}]，新的在前。"""
    out, cursor = [], ""
    for _ in range(pages):
        url = ("https://api-one-wscn.awtmt.com/apiv1/content/lives?"
               f"channel=global-channel&limit={limit}" + (f"&cursor={cursor}" if cursor else ""))
        try:
            req = urllib.request.Request(url, headers=UA)
            d = json.loads(urllib.request.urlopen(req, timeout=10).read())
        except Exception:
            break
        items = d.get("data", {}).get("items", [])
        if not items:
            break
        for it in items:
            c = it.get("content_text") or it.get("content") or ""
            out.append({"time": it.get("display_time") or it.get("created_at"),
                        "title": it.get("title") or "", "content": c})
        cursor = d.get("data", {}).get("next_cursor")
        if not cursor:
            break
        time.sleep(0.2)
    return out


def sector_news_density(news, keywords):
    """板块新闻密度：命中关键词的快讯条数与占比。
    keywords: 列表，如 ["算力","CPO","光模块"]"""
    if not news:
        return 0, 0.0, []
    hits = [n for n in news if any(k in (n["title"] + n["content"]) for k in keywords)]
    ratio = len(hits) / len(news)
    return len(hits), round(ratio, 3), hits[:10]
