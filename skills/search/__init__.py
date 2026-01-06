"""
Multi-Provider Web Search Package.

Provides a unified interface for web search across multiple providers.
Automatically detects available providers and uses the best available option.

Supported Providers:
- Jina AI (JINA_API_KEY)
- Tavily (TAVILY_API_KEY)
- Firecrawl (FIRECRAWL_API_KEY)
- Serper (SERPER_API_KEY)

Usage:
    # Auto-detect provider
    from skills.search import search, parallel_search
    results = await search("AI agents 2025")

    # Explicit provider
    from skills.search import search_with_provider
    results = await search_with_provider("AI agents", provider="tavily")

    # Check available providers
    from skills.search import list_provider_status
    status = list_provider_status()
"""

import asyncio
from typing import List, Optional
from datetime import datetime

from .base import SearchProvider, SearchResult, SearchResponse
from .jina import JinaSearchProvider
from .tavily import TavilySearchProvider
from .firecrawl import FirecrawlSearchProvider
from .serper import SerperSearchProvider
from .registry import (
    get_default_provider,
    get_provider,
    get_available_providers,
    get_all_providers,
    list_provider_status,
    register_provider,
    PROVIDER_CLASSES,
    DEFAULT_PROVIDER_PRIORITY,
)


async def search(query: str) -> SearchResponse:
    """
    Search using the default available provider.

    Args:
        query: Search term to query

    Returns:
        SearchResponse: Unified search results

    Raises:
        RuntimeError: If no providers are available
    """
    provider = get_default_provider()
    if not provider:
        raise RuntimeError(
            "No search providers available. Please configure at least one of: "
            "JINA_API_KEY, TAVILY_API_KEY, FIRECRAWL_API_KEY, or SERPER_API_KEY"
        )
    return await provider.search(query)


async def search_with_provider(query: str, provider: str) -> SearchResponse:
    """
    Search using a specific provider.

    Args:
        query: Search term to query
        provider: Provider name (jina, tavily, firecrawl, serper)

    Returns:
        SearchResponse: Unified search results

    Raises:
        ValueError: If provider is not found or not available
    """
    provider_instance = get_provider(provider)
    if not provider_instance:
        raise ValueError(f"Unknown provider: {provider}")
    if not provider_instance.is_available():
        raise ValueError(f"Provider '{provider}' is not available (API key not configured)")
    return await provider_instance.search(query)


async def parallel_search(queries: List[str], provider: Optional[str] = None) -> List[SearchResponse]:
    """
    Execute multiple searches in parallel.

    Args:
        queries: List of search queries
        provider: Optional specific provider to use

    Returns:
        List of SearchResponse objects
    """
    if not queries:
        return []

    # Get provider
    if provider:
        provider_instance = get_provider(provider)
        if not provider_instance or not provider_instance.is_available():
            raise ValueError(f"Provider '{provider}' is not available")
    else:
        provider_instance = get_default_provider()
        if not provider_instance:
            raise RuntimeError("No search providers available")

    # Execute searches in parallel
    tasks = [provider_instance.search(query) for query in queries]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Handle exceptions gracefully
    successful_results = []
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            print(f"Search failed for query '{queries[i]}': {result}")
            successful_results.append(SearchResponse(
                results=[],
                total_results=0,
                query_used=queries[i],
                provider=provider_instance.name
            ))
        else:
            successful_results.append(result)

    return successful_results


def extract_search_content(search_responses: List[SearchResponse], max_content_per_result: int = 1000) -> str:
    """
    Extract and combine content from search responses for AI analysis.

    Args:
        search_responses: List of SearchResponse objects
        max_content_per_result: Maximum content length per result

    Returns:
        Combined content string for AI analysis
    """
    combined_content = ""

    for response in search_responses:
        for result in response.results:
            title = result.title
            content = result.content
            url = result.url

            # Truncate content if too long
            if len(content) > max_content_per_result:
                content = content[:max_content_per_result] + "..."

            # Format for AI consumption
            result_text = f"Title: {title}\nURL: {url}\nContent: {content}\n\n"
            combined_content += result_text

    return combined_content


def generate_search_variations(base_query: str) -> List[str]:
    """
    Generate search query variations for comprehensive coverage.

    Args:
        base_query: Base search query

    Returns:
        List of query variations
    """
    variations = [base_query]

    # Add temporal variations
    current_year = datetime.now().year
    variations.extend([
        f"{base_query} {current_year}",
        f"{base_query} latest",
        f"{base_query} recent",
    ])

    # Add perspective variations
    variations.extend([
        f"{base_query} analysis",
        f"{base_query} research",
        f"{base_query} study",
        f"{base_query} report",
    ])

    return variations[:8]  # Limit to 8 variations


def filter_results_by_relevance(results: List[SearchResult], min_content_length: int = 100) -> List[SearchResult]:
    """Filter search results by relevance criteria."""
    filtered = []
    for result in results:
        if len(result.content) >= min_content_length and result.title and result.url:
            filtered.append(result)
    return filtered


def deduplicate_results(results: List[SearchResult]) -> List[SearchResult]:
    """Remove duplicate results based on URL."""
    seen_urls = set()
    deduplicated = []

    for result in results:
        if result.url not in seen_urls:
            seen_urls.add(result.url)
            deduplicated.append(result)

    return deduplicated


def rank_results_by_content_quality(results: List[SearchResult]) -> List[SearchResult]:
    """Rank results by content quality heuristics."""
    def quality_score(result: SearchResult) -> float:
        score = 0.0

        # Content length (longer is generally better, up to a point)
        content_length = len(result.content)
        if content_length > 500:
            score += 1.0
        elif content_length > 200:
            score += 0.5

        # Title quality (presence and length)
        if result.title and len(result.title) > 10:
            score += 0.5

        # URL quality (avoid certain patterns)
        if result.url:
            if any(domain in result.url for domain in ['.edu', '.gov', '.org']):
                score += 0.3
            if 'wikipedia.org' in result.url:
                score += 0.2

        # Recent publication (if available)
        if result.published_time:
            days_old = (datetime.now() - result.published_time).days
            if days_old < 30:
                score += 0.3
            elif days_old < 365:
                score += 0.1

        return score

    return sorted(results, key=quality_score, reverse=True)


# Synchronous wrapper for backward compatibility
def search_sync(query: str) -> SearchResponse:
    """
    Synchronous wrapper for search.

    Args:
        query: Search term to query

    Returns:
        SearchResponse: Unified search results
    """
    try:
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(search(query))
    except RuntimeError:
        return asyncio.run(search(query))


__all__ = [
    # Main search functions
    "search",
    "search_with_provider",
    "parallel_search",
    "search_sync",
    # Content processing
    "extract_search_content",
    "generate_search_variations",
    "filter_results_by_relevance",
    "deduplicate_results",
    "rank_results_by_content_quality",
    # Models
    "SearchResult",
    "SearchResponse",
    "SearchProvider",
    # Provider classes
    "JinaSearchProvider",
    "TavilySearchProvider",
    "FirecrawlSearchProvider",
    "SerperSearchProvider",
    # Registry functions
    "get_default_provider",
    "get_provider",
    "get_available_providers",
    "get_all_providers",
    "list_provider_status",
    "register_provider",
]
