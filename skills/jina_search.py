"""
Jina AI Search Integration - Deep Research Agent

BACKWARD COMPATIBILITY MODULE
This module maintains backward compatibility with existing code that imports
from skills.jina_search. New code should use skills.search instead.

Provides internet search capabilities using Jina AI API for the dynamic research system.
Optimized for parallel execution and structured data extraction.
"""

import os
import asyncio
from typing import List, Optional
from datetime import datetime
import aiohttp
from pydantic import BaseModel, Field


# Keep original models for backward compatibility
class SearchResult(BaseModel):
    """Individual search result from Jina AI"""
    title: str = Field(description="Result title")
    url: str = Field(description="Result URL")
    content: str = Field(description="Result content")
    description: Optional[str] = Field(None, description="Result description")
    published_time: Optional[datetime] = Field(None, alias="publishedTime", description="Publication time")

    class Config:
        populate_by_name = True


class JinaSearchResponse(BaseModel):
    """Complete response from Jina AI search"""
    results: List[SearchResult] = Field(default_factory=list, alias="data", description="Search results")
    total_results: int = Field(0, description="Total number of results")
    query_used: str = Field("", description="Query that was executed")

    def __init__(self, **data):
        # Handle the response structure where results are in 'data' field
        if 'data' in data and isinstance(data['data'], list):
            data['total_results'] = len(data['data'])
        super().__init__(**data)

    class Config:
        populate_by_name = True


async def search_jina_ai(query: str) -> JinaSearchResponse:
    """
    Search using Jina AI and return structured results.

    Args:
        query: Search term to query

    Returns:
        JinaSearchResponse: Structured search results

    Raises:
        ValueError: If JINA_API_KEY environment variable is not set
        aiohttp.ClientError: If the API request fails
    """
    api_key = os.getenv("JINA_API_KEY")
    if not api_key:
        raise ValueError("JINA_API_KEY environment variable is required")

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

            # Transform the data to match our Pydantic model
            transformed_data = {
                "data": data.get("data", []),
                "query_used": query
            }

            return JinaSearchResponse(**transformed_data)


async def parallel_search(queries: List[str]) -> List[JinaSearchResponse]:
    """
    Execute multiple searches in parallel for maximum efficiency.

    Args:
        queries: List of search queries to execute

    Returns:
        List of JinaSearchResponse objects
    """
    if not queries:
        return []

    # Create tasks for parallel execution
    tasks = [search_jina_ai(query) for query in queries]

    # Execute all searches in parallel
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Filter out exceptions and return successful results
    successful_results = []
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            print(f"Search failed for query '{queries[i]}': {result}")
            # Add empty result for failed searches
            successful_results.append(JinaSearchResponse(data=[], query_used=queries[i]))
        else:
            successful_results.append(result)

    return successful_results


def extract_search_content(search_responses: List[JinaSearchResponse], max_content_per_result: int = 1000) -> str:
    """
    Extract and combine content from search responses for AI analysis.

    Args:
        search_responses: List of JinaSearchResponse objects
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

    return variations[:8]  # Limit to 8 variations to avoid overwhelming


# Synchronous wrapper for backward compatibility
def search_jina_sync(query: str) -> JinaSearchResponse:
    """
    Synchronous wrapper for Jina AI search.

    Args:
        query: Search term to query

    Returns:
        JinaSearchResponse: Structured search results
    """
    try:
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(search_jina_ai(query))
    except RuntimeError:
        # If no event loop is running, create a new one
        return asyncio.run(search_jina_ai(query))


# Utility functions for search result processing
def filter_results_by_relevance(results: List[SearchResult], min_content_length: int = 100) -> List[SearchResult]:
    """Filter search results by relevance criteria"""
    filtered = []
    for result in results:
        if len(result.content) >= min_content_length and result.title and result.url:
            filtered.append(result)
    return filtered


def deduplicate_results(results: List[SearchResult]) -> List[SearchResult]:
    """Remove duplicate results based on URL"""
    seen_urls = set()
    deduplicated = []

    for result in results:
        if result.url not in seen_urls:
            seen_urls.add(result.url)
            deduplicated.append(result)

    return deduplicated


def rank_results_by_content_quality(results: List[SearchResult]) -> List[SearchResult]:
    """Rank results by content quality heuristics"""
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
