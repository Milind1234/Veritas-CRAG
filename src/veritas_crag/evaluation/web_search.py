"""Active Web Search Fallback Engine using DuckDuckGo and Tavily."""

from typing import List, Optional
import httpx
from ddgs import DDGS
from loguru import logger

from veritas_crag.config.settings import settings
from veritas_crag.models.crag import WebSearchResult


class WebSearchEngine:
    """External knowledge retriever for Corrective RAG fallback."""

    def __init__(
        self,
        provider: Optional[str] = None,
        max_results: Optional[int] = None,
        tavily_api_key: Optional[str] = None,
    ):
        self.provider = provider or settings.WEB_SEARCH_PROVIDER
        self.max_results = max_results or settings.MAX_WEB_RESULTS
        self.tavily_api_key = tavily_api_key or settings.TAVILY_API_KEY

    def search(
        self,
        query: str,
        max_results: Optional[int] = None,
    ) -> List[WebSearchResult]:
        """Executes web search query and returns structured results."""
        k = max_results or self.max_results
        logger.info(f"Executing web fallback search ({self.provider}) for: '{query}'")

        if self.provider == "tavily" and self.tavily_api_key:
            return self._search_tavily(query, k)
        else:
            return self._search_duckduckgo(query, k)

    def _search_duckduckgo(self, query: str, max_results: int) -> List[WebSearchResult]:
        """Free, zero-config web search using DuckDuckGo."""
        results: List[WebSearchResult] = []
        try:
            ddgs = DDGS()
            raw_results = list(ddgs.text(query, max_results=max_results))
            for item in raw_results:
                title = item.get("title", "")
                url = item.get("href", "")
                snippet = item.get("body", "")
                if snippet:
                    results.append(
                        WebSearchResult(
                            title=title,
                            url=url,
                            snippet=snippet,
                        )
                    )
            logger.info(f"DuckDuckGo returned {len(results)} search results.")
        except Exception as e:
            logger.error(f"DuckDuckGo search error for '{query}': {e}")

        return results

    def _search_tavily(self, query: str, max_results: int) -> List[WebSearchResult]:
        """Search using Tavily API if configured."""
        results: List[WebSearchResult] = []
        try:
            url = "https://api.tavily.com/search"
            payload = {
                "api_key": self.tavily_api_key,
                "query": query,
                "max_results": max_results,
                "include_answer": False,
            }
            with httpx.Client(timeout=10.0) as client:
                resp = client.post(url, json=payload)
                resp.raise_for_status()
                data = resp.json()
                for item in data.get("results", []):
                    results.append(
                        WebSearchResult(
                            title=item.get("title", ""),
                            url=item.get("url", ""),
                            snippet=item.get("content", ""),
                        )
                    )
            logger.info(f"Tavily returned {len(results)} search results.")
        except Exception as e:
            logger.error(f"Tavily search error for '{query}': {e}. Falling back to DuckDuckGo.")
            return self._search_duckduckgo(query, max_results)

        return results
