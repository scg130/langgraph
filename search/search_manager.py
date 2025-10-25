from googlesearch import search


def web_search(query: str, num_results: int = 5):
    try:
        results = []
        for url in search(query, num_results=num_results, lang='zh-cn'):
            results.append(url)
        return results
    except Exception as e:
        return [f"搜索出错: {e}"]
