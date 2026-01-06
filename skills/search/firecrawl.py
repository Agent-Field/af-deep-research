"""
Firecrawl Search Provider.

Provides web search using Firecrawl's search API.
Firecrawl combines search with optional content scraping in one call.
"""

import aiohttp
from typing import Optional, List, Literal
from datetime import datetime

from .base import SearchProvider, SearchResult, SearchResponse


class FirecrawlSearchProvider(SearchProvider):
    """Firecrawl search provider implementation."""

    def __init__(
        self,
        limit: int = 10,
        scrape_results: bool = False,
        source_filter: Optional[Literal["web", "news", "github", "research", "pdf"]] = None
    ):
        """
        Initialize Firecrawl provider.

        Args:
            limit: Maximum number of results to return
            scrape_results: Whether to scrape full content from results
            source_filter: Filter results by source type
        """
        self.limit = limit
        self.scrape_results = scrape_results
        self.source_filter = source_filter

    @property
    def name(self) -> str:
        return "firecrawl"

    @property
    def api_key_env_var(self) -> str:
        return "FIRECRAWL_API_KEY"

    async def search(self, query: str) -> SearchResponse:
        """
        Search using Firecrawl and return structured results.

        Args:
            query: Search term to query

        Returns:
            SearchResponse: Unified search results

        Raises:
            ValueError: If FIRECRAWL_API_KEY environment variable is not set
            aiohttp.ClientError: If the API request fails
        """
        api_key = self.get_api_key()
        if not api_key:
            raise ValueError(f"{self.api_key_env_var} environment variable is required")

        url = "https://api.firecrawl.dev/v1/search"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }
        payload = {
            "query": query,
            "limit": self.limit
        }

        # Add source filter if specified
        if self.source_filter:
            payload["filter"] = {"sourceType": self.source_filter}

        # Add scrape options if enabled
        if self.scrape_results:
            payload["scrapeOptions"] = {
                "formats": ["markdown"]
            }

        async with aiohttp.ClientSession() as session:
            async with session.post(url, headers=headers, json=payload) as response:
                response.raise_for_status()
                data = await response.json()

                results = []
                for item in data.get("data", []):
                    # Handle both search-only and scrape results
                    content = item.get("markdown") or item.get("description") or item.get("snippet", "")

                    # Parse date if available
                    published_time = None
                    if item.get("publishedDate"):
                        try:
                            published_time = datetime.fromisoformat(
                                item["publishedDate"].replace("Z", "+00:00")
                            )
                        except (ValueError, TypeError):
                            pass

                    results.append(SearchResult(
                        title=item.get("title", ""),
                        url=item.get("url", ""),
                        content=content,
                        description=item.get("description"),
                        published_time=published_time
                    ))

                return SearchResponse(
                    results=results,
                    total_results=len(results),
                    query_used=query,
                    provider=self.name
                )
