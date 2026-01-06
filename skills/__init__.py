"""
Skills Package - deep-research-agent

This package contains custom utility functions for your Brain agent.
Skills are utility functions that provide reusable capabilities.

Created: 2025-07-09 16:20:06 EDT
Author: Brain Research Team
"""

from .example_skill import example_utility, helper_function

# Backward compatible imports from jina_search
from .jina_search import (
    search_jina_ai, parallel_search, extract_search_content,
    generate_search_variations, SearchResult, JinaSearchResponse,
    filter_results_by_relevance, deduplicate_results, rank_results_by_content_quality
)

# New multi-provider search imports
from .search import (
    search,
    search_with_provider,
    search_sync,
    SearchResponse,
    SearchProvider,
    JinaSearchProvider,
    TavilySearchProvider,
    FirecrawlSearchProvider,
    SerperSearchProvider,
    get_default_provider,
    get_provider,
    get_available_providers,
    list_provider_status,
)

__all__ = [
    "example_utility",
    "helper_function",
    # Backward compatible Jina AI Search functions
    "search_jina_ai",
    "parallel_search",
    "extract_search_content",
    "generate_search_variations",
    "SearchResult",
    "JinaSearchResponse",
    "filter_results_by_relevance",
    "deduplicate_results",
    "rank_results_by_content_quality",
    # New multi-provider search
    "search",
    "search_with_provider",
    "search_sync",
    "SearchResponse",
    "SearchProvider",
    "JinaSearchProvider",
    "TavilySearchProvider",
    "FirecrawlSearchProvider",
    "SerperSearchProvider",
    "get_default_provider",
    "get_provider",
    "get_available_providers",
    "list_provider_status",
]
