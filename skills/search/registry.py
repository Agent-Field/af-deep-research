"""
Search Provider Registry.

Handles auto-detection of available providers and provider selection.
"""

import os
from typing import List, Optional, Dict, Type

from .base import SearchProvider
from .jina import JinaSearchProvider
from .tavily import TavilySearchProvider
from .firecrawl import FirecrawlSearchProvider
from .serper import SerperSearchProvider


# Default priority order for providers
DEFAULT_PROVIDER_PRIORITY = ["jina", "tavily", "firecrawl", "serper"]

# Registry of all available provider classes
PROVIDER_CLASSES: Dict[str, Type[SearchProvider]] = {
    "jina": JinaSearchProvider,
    "tavily": TavilySearchProvider,
    "firecrawl": FirecrawlSearchProvider,
    "serper": SerperSearchProvider,
}


def get_all_providers() -> Dict[str, SearchProvider]:
    """
    Get instances of all registered providers.

    Returns:
        Dict mapping provider names to provider instances
    """
    return {name: cls() for name, cls in PROVIDER_CLASSES.items()}


def get_available_providers() -> List[SearchProvider]:
    """
    Get list of providers that have API keys configured.

    Returns:
        List of available provider instances
    """
    available = []
    for name in DEFAULT_PROVIDER_PRIORITY:
        if name in PROVIDER_CLASSES:
            provider = PROVIDER_CLASSES[name]()
            if provider.is_available():
                available.append(provider)
    return available


def get_provider(name: str) -> Optional[SearchProvider]:
    """
    Get a specific provider by name.

    Args:
        name: Provider name (jina, tavily, firecrawl, serper)

    Returns:
        Provider instance or None if not found
    """
    if name in PROVIDER_CLASSES:
        return PROVIDER_CLASSES[name]()
    return None


def get_default_provider() -> Optional[SearchProvider]:
    """
    Get the default provider based on availability and priority.

    Checks for SEARCH_PROVIDER env var first, then falls back to
    the first available provider in priority order.

    Returns:
        Default provider instance or None if no providers available
    """
    # Check if user has explicitly set a preferred provider
    preferred = os.getenv("SEARCH_PROVIDER", "").lower().strip()
    if preferred:
        provider = get_provider(preferred)
        if provider and provider.is_available():
            return provider
        # If preferred provider isn't available, log a warning and fall through
        print(f"Warning: Preferred provider '{preferred}' is not available, trying alternatives")

    # Fall back to first available provider in priority order
    available = get_available_providers()
    if available:
        return available[0]

    return None


def list_provider_status() -> Dict[str, bool]:
    """
    Get availability status of all providers.

    Returns:
        Dict mapping provider names to availability status
    """
    return {name: PROVIDER_CLASSES[name]().is_available() for name in DEFAULT_PROVIDER_PRIORITY}


def register_provider(name: str, provider_class: Type[SearchProvider]) -> None:
    """
    Register a custom provider.

    Args:
        name: Provider name identifier
        provider_class: Provider class (must inherit from SearchProvider)
    """
    if not issubclass(provider_class, SearchProvider):
        raise TypeError(f"Provider must inherit from SearchProvider")
    PROVIDER_CLASSES[name] = provider_class
    if name not in DEFAULT_PROVIDER_PRIORITY:
        DEFAULT_PROVIDER_PRIORITY.append(name)
