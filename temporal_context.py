"""
Temporal Context Helper - Deep Research Agent

Simple, static temporal context injection for reasoners.
Provides current date/time context without AI-based detection.
"""

import datetime
from typing import Dict, Optional


def get_temporal_context(context_type: str = "standard") -> str:
    """
    Get temporal context for injection into reasoner prompts.

    Args:
        context_type: Type of temporal context needed
            - "search": For search query generation and web content retrieval
            - "evidence": For evidence evaluation and extraction
            - "synthesis": For synthesis and discovery generation
            - "classification": For query classification
            - "standard": Default temporal context

    Returns:
        Formatted temporal context string for prompt injection
    """
    now = datetime.datetime.now()
    current_date = now.strftime("%B %d, %Y")
    current_year = now.year
    prev_year = current_year - 1

    contexts = {
        "search": f"""
<temporal_context>
Current Date: {current_date}
Recent Period: {prev_year}-{current_year}
Search Optimization: For recent topics, prioritize content from {current_year} and late {prev_year}
</temporal_context>""",
        "evidence": f"""
<temporal_relevance>
Analysis Date: {current_date}
Recency Weighting: {current_year} content (highest relevance) > {prev_year} content (high relevance) > older content (contextual relevance)
</temporal_relevance>""",
        "synthesis": f"""
<temporal_framework>
Current Context: {current_date}
Recent Developments: Focus on {prev_year}-{current_year} trends and developments
Temporal Perspective: Consider recency when identifying patterns and implications
</temporal_framework>""",
        "classification": f"""
<temporal_awareness>
Current Date: {current_date}
Recent Period Definition: Last 12 months ({prev_year}-{current_year})
Temporal Keywords: "recent", "current", "latest", "trending", "new", "emerging"
</temporal_awareness>""",
        "standard": f"""
<temporal_context>
Current Date: {current_date}
Analysis Context: Consider temporal relevance where applicable
</temporal_context>""",
    }

    return contexts.get(context_type, contexts["standard"])


def get_current_date_simple() -> str:
    """Get simple current date string for basic temporal reference."""
    return datetime.datetime.now().strftime("%B %d, %Y")


def get_current_year() -> int:
    """Get current year for temporal calculations."""
    return datetime.datetime.now().year


def enhance_search_queries_with_temporal_context(
    queries: list, current_year: Optional[int] = None
) -> list:
    """
    Enhance search queries with temporal context for recent information.

    Args:
        queries: List of search query strings
        current_year: Current year (auto-detected if not provided)

    Returns:
        Enhanced queries with temporal indicators
    """
    if current_year is None:
        current_year = get_current_year()

    enhanced = []
    temporal_indicators = [f"{current_year}", "recent", "latest"]

    for query in queries:
        # Add original query
        enhanced.append(query)

        # Add temporally enhanced versions for market/trend queries
        if any(
            word in query.lower()
            for word in ["market", "trends", "analysis", "report", "development"]
        ):
            enhanced.append(f"{query} {current_year}")
            enhanced.append(f"{query} recent developments")

    return enhanced
