"""
Tavily Search Provider.

Provides web search using Tavily's AI-native search API.
Tavily is designed specifically for AI agents and RAG applications.
"""

import aiohttp
from typing import Optional, Literal
from datetime import datetime

from .base import SearchProvider, SearchResult, SearchResponse


class TavilySearchProvider(SearchProvider):
    """Tavily search provider implementation."""

    def __init__(self, search_depth: Literal["basic", "advanced"] = "basic"):
        """
        Initialize Tavily provider.

        Args:
            search_depth: Search depth - "basic" (faster) or "advanced" (more thorough)
        """
        self.search_depth = search_depth

    @property
    def name(self) -> str:
        return "tavily"

    @property
    def api_key_env_var(self) -> str:
        return "TAVILY_API_KEY"

    async def search(self, query: str) -> SearchResponse:
        """
        Search using Tavily AI and return structured results.

        Args:
            query: Search term to query

        Returns:
            SearchResponse: Unified search results

        Raises:
            ValueError: If TAVILY_API_KEY environment variable is not set
            aiohttp.ClientError: If the API request fails
        """
        api_key = self.get_api_key()
        if not api_key:
            raise ValueError(f"{self.api_key_env_var} environment variable is required")

        url = "https://api.tavily.com/search"
        headers = {
            "Content-Type": "application/json"
        }
        payload = {
            "api_key": api_key,
            "query": query,
            "search_depth": self.search_depth,
            "include_answer": False,
            "include_raw_content": False,
            "max_results": 10
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(url, headers=headers, json=payload) as response:
                response.raise_for_status()
                data = await response.json()

                results = []
                for item in data.get("results", []):
                    # Parse published date if available
                    published_time = None
                    if item.get("published_date"):
                        try:
                            published_time = datetime.fromisoformat(
                                item["published_date"].replace("Z", "+00:00")
                            )
                        except (ValueError, TypeError):
                            pass

                    results.append(SearchResult(
                        title=item.get("title", ""),
                        url=item.get("url", ""),
                        content=item.get("content", ""),
                        description=item.get("snippet"),
                        published_time=published_time
                    ))

                return SearchResponse(
                    results=results,
                    total_results=len(results),
                    query_used=query,
                    provider=self.name
                )
