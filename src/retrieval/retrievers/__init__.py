# retrieval/retrievers/__init__.py

"""
Retrievers module: provides flexible and composable retrieval strategies.

This module supports:
- Single-hop retrieval via vector similarity (`VectorStoreRetriever`)
- Multi-hop retrieval with entity-aware query expansion (`MultiHopRetriever`)
- Easy integration with custom embedders, rerankers, and vector stores

All retrievers implement the `BaseRetriever` interface for uniform usage.
"""

from .base_retriever import BaseRetriever, VectorStoreRetriever
from .multi_hop_retriever import (
    MultiHopRetriever,
    EntityExtractor,
    QueryExpander,
    KeywordBasedEntityExtractor,
    TemplateBasedQueryExpander,
)

# Public API — intended for external use
__all__ = [
    # Core interfaces
    "BaseRetriever",

    # Concrete retrievers
    "VectorStoreRetriever",
    "MultiHopRetriever",

    # Multi-hop components (for customization)
    "EntityExtractor",
    "QueryExpander",
    "KeywordBasedEntityExtractor",
    "TemplateBasedQueryExpander",
]