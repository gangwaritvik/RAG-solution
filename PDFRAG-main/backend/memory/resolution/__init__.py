"""Resolution module - Query context resolution and classification."""

from .context_resolver import (
    ContextResolver,
    QueryContext,
    DependencyType,
    RetrievalIntent
)

__all__ = [
    "ContextResolver",
    "QueryContext",
    "DependencyType",
    "RetrievalIntent"
]
