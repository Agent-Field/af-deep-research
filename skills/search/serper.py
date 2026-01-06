"""
Serper Search Provider.

Provides web search using Serper's Google SERP API.
Serper is a fast, affordable Google search API.
"""

import aiohttp
from typing import Optional, Literal

from .base import SearchProvider, SearchResult, SearchResponse


class SerperSearchProvider(SearchProvider):
    """Serper (Google SERP) search provider implementation."""

    def __init__(
        self,
        search_type: Literal["search", "news", "images"] = "search",
        num_results: int = 10,
        country: str = "us",
        locale: str = "en"
    ):
        """
        Initialize Serper provider.

        Args:
            search_type: Type of search - "search", "news", or "images"
            num_results: Number of results to return
            country: Country code for localized results
            locale: Language locale
        """
        self.search_type = search_type
        self.num_results = num_results
        self.country = country
        self.locale = locale

    @property
    def name(self) -> str:
        return "serper"

    @property
    def api_key_env_var(self) -> str:
        return "SERPER_API_KEY"

    async def search(self, query: str) -> SearchResponse:
        """
        Search using Serper (Google SERP) and return structured results.

        Args:
            query: Search term to query

        Returns:
            SearchResponse: Unified search results

        Raises:
            ValueError: If SERPER_API_KEY environment variable is not set
            aiohttp.ClientError: If the API request fails
        """
        api_key = self.get_api_key()
        if not api_key:
            raise ValueError(f"{self.api_key_env_var} environment variable is required")

        url = f"https://google.serper.dev/{self.search_type}"
        headers = {
            "Content-Type": "application/json",
            "X-API-KEY": api_key
        }
        payload = {
            "q": query,
            "num": self.num_results,
            "gl": self.country,
            "hl": self.locale
        }

        async with aiohttp.ClientSession() as session:
            async with session.post(url, headers=headers, json=payload) as response:
                response.raise_for_status()
                data = await response.json()

                results = []

                # Handle different result types based on search_type
                if self.search_type == "news":
                    items = data.get("news", [])
                elif self.search_type == "images":
                    items = data.get("images", [])
                else:
                    items = data.get("organic", [])

                for item in items:
                    # For images, use imageUrl as url
                    url_field = item.get("link") or item.get("imageUrl", "")

                    # Content varies by type
                    content = item.get("snippet", "")
                    if self.search_type == "news":
                        content = item.get("snippet") or item.get("description", "")
                    elif self.search_type == "images":
                        content = item.get("title", "")

                    results.append(SearchResult(
                        title=item.get("title", ""),
                        url=url_field,
                        content=content,
                        description=item.get("snippet"),
                        published_time=None  # Serper doesn't provide structured dates
                    ))

                return SearchResponse(
                    results=results,
                    total_results=len(results),
                    query_used=query,
                    provider=self.name
                )
