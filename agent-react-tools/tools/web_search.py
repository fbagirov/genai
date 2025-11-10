
import os, requests

def web_search(query: str):
    provider = os.getenv("SEARCH_PROVIDER", os.getenv("search_provider","tavily"))
    if provider == "stub":
        return [
            {"title":"Stub result 1","url":"https://example.com/1","snippet":f"Info about {query}."},
            {"title":"Stub result 2","url":"https://example.com/2","snippet":f"More about {query}."},
        ]
    api = os.getenv("TAVILY_API_KEY")
    if not api:
        return [{"title":"No API key","url":"","snippet":"Set TAVILY_API_KEY or use SEARCH_PROVIDER=stub"}]
    resp = requests.post("https://api.tavily.com/search",
                         json={"api_key": api, "query": query, "max_results": 5, "include_answer": True},
                         timeout=20)
    resp.raise_for_status()
    data = resp.json()
    results = []
    for r in data.get("results", []):
        results.append({"title": r.get("title",""), "url": r.get("url",""), "snippet": r.get("content","")[:280]})
    if data.get("answer"):
        results.insert(0, {"title":"Direct answer","url":"","snippet": data["answer"]})
    return results
