"""Tavily 深度搜索：投行研报、券商观点、公司公告求证（照妖镜）、海外消息。
REST API，无需 MCP/Node。文档: https://docs.tavily.com"""
import json
import time
import urllib.request

API = "https://api.tavily.com/search"


def search(api_key, query, max_results=8, days=None):
    """返回 {answer, results:[{title,url,content,published}]}；失败返回 None。"""
    body = {"query": query, "max_results": min(int(max_results), 10),
            "search_depth": "advanced", "include_answer": True}
    if days:
        body["days"] = int(days)
    for i in range(3):
        try:
            req = urllib.request.Request(
                API, data=json.dumps(body).encode(),
                headers={"Authorization": f"Bearer {api_key}",
                         "Content-Type": "application/json"})
            r = json.loads(urllib.request.urlopen(req, timeout=30).read().decode())
            return {
                "answer": r.get("answer") or "",
                "results": [{"title": x.get("title", ""), "url": x.get("url", ""),
                             "published": x.get("published_date") or "",
                             "content": (x.get("content") or "")[:500]}
                            for x in r.get("results", [])],
            }
        except Exception:
            time.sleep(0.5 * (i + 1))
    return None
