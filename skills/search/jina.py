"""
Jina AI Search Provider.

Provides web search using Jina AI's search API.
"""

import aiohttp
from typing import Optional
from datetime import datetime

from .base import SearchProvider, SearchResult, SearchResponse


class JinaSearchProvider(SearchProvider):
    """Jina AI search provider implementation."""

    @property
    def name(self) -> str:
        return "jina"

    @property
    def api_key_env_var(self) -> str:
        return "JINA_API_KEY"

    async def search(self, query: str) -> SearchResponse:
        """
        Search using Jina AI and return structured results.

        Args:
            query: Search term to query

        Returns:
            SearchResponse: Unified search results

        Raises:
            ValueError: If JINA_API_KEY environment variable is not set
            aiohttp.ClientError: If the API request fails
        """
        api_key = self.get_api_key()
        if not api_key:
            raise ValueError(f"{self.api_key_env_var} environment variable is required")

        url = "https://s.jina.ai/"
        headers = {
            "Accept": "application/json",
            "Authorization": f"Bearer {api_key}",
            "X-Engine": "browser"
        }
        params = {"q": query}

        async with aiohttp.ClientSession() as session:
            async with session.get(url, headers=headers, params=params) as response:
                response.raise_for_status()
                data = await response.json()

                results = []
                for item in data.get("data", []):
                    # Parse published time if available
                    published_time = None
                    if item.get("publishedTime"):
                        try:
                            published_time = datetime.fromisoformat(
                                item["publishedTime"].replace("Z", "+00:00")
                            )
                        except (ValueError, TypeError):
                            pass

                    results.append(SearchResult(
                        title=item.get("title", ""),
                        url=item.get("url", ""),
                        content=item.get("content", ""),
                        description=item.get("description"),
                        published_time=published_time
                    ))

                return SearchResponse(
                    results=results,
                    total_results=len(results),
                    query_used=query,
                    provider=self.name
                )
