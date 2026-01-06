#!/usr/bin/env python3
"""
Utility functions for the Deep Research Agent

This module contains shared utility functions that can be imported
by multiple modules without creating circular dependencies.
"""

import asyncio
import logging
import os
from datetime import datetime
from typing import List, Optional

import aiohttp
from pydantic import BaseModel

# Setup logging
logger = logging.getLogger(__name__)


class SearchResult(BaseModel):
    """Individual search result from Jina AI"""

    title: str
    url: str
    content: str
    description: Optional[str] = None
    published_time: Optional[datetime] = None
    relevance: float = 0.0  # Add relevance field for compatibility


class JinaSearchResponse(BaseModel):
    """Complete response from Jina AI search"""

    results: List[SearchResult] = []
    total_results: int = 0
    query_used: str = ""


async def search_jina_ai(query: str) -> JinaSearchResponse:
    """Search using Jina AI and return structured results."""
    try:
        api_key = os.getenv("JINA_API_KEY")
        if not api_key:
            logger.warning("JINA_API_KEY not found, returning empty results")
            return JinaSearchResponse(query_used=query)

        url = "https://s.jina.ai/"
        headers = {"Authorization": f"Bearer {api_key}", "Accept": "application/json"}

        async with aiohttp.ClientSession() as session:
            async with session.get(f"{url}{query}", headers=headers) as response:
                if response.status == 200:
                    data = await response.json()
                    results = []
                    for item in data.get("data", [])[:10]:  # Limit to 10 results
                        results.append(
                            SearchResult(
                                title=item.get("title", ""),
                                url=item.get("url", ""),
                                content=item.get("content", ""),
                                description=item.get("description", ""),
                                relevance=0.8,  # Default relevance score
                            )
                        )
                    return JinaSearchResponse(
                        results=results, total_results=len(results), query_used=query
                    )
                else:
                    logger.error(f"Jina AI search failed: {response.status}")
                    return JinaSearchResponse(query_used=query)
    except Exception as e:
        logger.error(f"Jina AI search error: {e}")
        return JinaSearchResponse(query_used=query)