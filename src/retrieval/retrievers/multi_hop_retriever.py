# retrieval/retrievers/multi_hop_retriever.py

from abc import ABC, abstractmethod
from typing import List, Protocol, Optional, Set, Tuple, Dict, Any
from src.retrieval.schemas import RetrievedChunk, TextChunk
from retrieval.retrievers.base_retriever import BaseRetriever, VectorStoreRetriever
from retrieval.vectorstores.base_vectorstore import BaseVectorStore
from retrieval.rerankers.base_reranker import BaseReranker


# =============== 接口定义 ===============

class EntityExtractor(ABC):
    """Extract structured entities from raw query text."""
    @abstractmethod
    def extract_entities(self, query: str) -> List[str]:
        pass


class QueryExpander(ABC):
    """Expand an entity into multiple semantic sub-queries."""
    @abstractmethod
    def expand(self, entity: str) -> List[str]:
        pass


# =============== 默认实现（通用、无硬编码） ===============

class KeywordBasedEntityExtractor(EntityExtractor):
    """
    Generic entity extractor using a provided keyword set.
    Suitable for domain-specific terms (e.g., drugs, genes, diseases).
    """
    def __init__(self, keywords: List[str], case_sensitive: bool = False):
        self.case_sensitive = case_sensitive
        if not case_sensitive:
            self.keywords = [k.lower() for k in keywords]
            self._original_keywords = {k.lower(): k for k in keywords}
        else:
            self.keywords = keywords
            self._original_keywords = {k: k for k in keywords}

    def extract_entities(self, query: str) -> List[str]:
        text = query if self.case_sensitive else query.lower()
        found = []
        seen: Set[str] = set()

        # Sort by length (longest first) to avoid partial matches
        sorted_keywords = sorted(self.keywords, key=len, reverse=True)

        for kw in sorted_keywords:
            if kw in text:
                orig_kw = self._original_keywords[kw]
                if orig_kw not in seen:
                    found.append(orig_kw)
                    seen.add(orig_kw)

        return found


class TemplateBasedQueryExpander(QueryExpander):
    """
    Expand entity using configurable templates.
    Example: templates = ["{entity} side effects", "{entity} contraindications"]
    """
    def __init__(self, templates: List[str]):
        self.templates = templates

    def expand(self, entity: str) -> List[str]:
        return [template.format(entity=entity) for template in self.templates]


# =============== 主检索器 ===============

class MultiHopRetriever(BaseRetriever):
    """
    General-purpose multi-hop retriever.

    Works for any domain where:
      - You can define a set of relevant entities (e.g., drugs, proteins, diseases)
      - You can define expansion templates for those entities

    No hard-coded logic. Fully configurable via constructor.
    """

    def __init__(
        self,
        vectorstore: BaseVectorStore,
        reranker: BaseReranker,
        base_retriever: VectorStoreRetriever,
        embedder,
        entity_extractor: EntityExtractor,
        query_expander: QueryExpander,
        single_hop_threshold: int = 1,
        top_k_per_subquery: int = 3
    ):
        self.vectorstore = vectorstore
        self.reranker = reranker
        self.base_retriever = base_retriever
        self.embedder = embedder
        self.entity_extractor = entity_extractor
        self.query_expander = query_expander
        self.single_hop_threshold = single_hop_threshold
        self.top_k_per_subquery = top_k_per_subquery

    def _retrieve_for_entity(self, entity: str) -> List[Tuple[TextChunk, float]]:
        """Retrieve chunks relevant to a single entity."""
        sub_queries = self.query_expander.expand(entity)
        all_results: List[Tuple[TextChunk, float]] = []
        seen_texts: Set[str] = set()

        for sq in sub_queries:
            sq_embedding = self.embedder.encode(sq)
            raw_results = self.vectorstore.search(sq_embedding, k=self.top_k_per_subquery)

            for item in raw_results:
                if not (hasattr(item, 'text') and hasattr(item, 'metadata')):
                    continue

                text = item.text
                if text in seen_texts:
                    continue
                seen_texts.add(text)

                metadata = item.metadata
                if hasattr(metadata, 'dict'):
                    metadata = metadata.dict()

                score = getattr(item, 'score', 0.0)
                chunk = TextChunk(
                    text=text,
                    metadata=metadata,
                    source_id=getattr(item, 'source_id', None),
                    embedding=None
                )
                all_results.append((chunk, float(score)))

        return all_results

    def retrieve(self, query: str, top_k: int = 6) -> List[RetrievedChunk]:
        entities = self.entity_extractor.extract_entities(query)

        # Single-hop mode
        if len(entities) <= self.single_hop_threshold:
            query_emb = self.embedder.encode(query)
            return self.base_retriever.retrieve(
                query_text=query,
                query_embedding=query_emb,
                top_k=top_k
            )

        # Multi-hop mode
        all_chunks_with_scores: List[Tuple[TextChunk, float]] = []
        for entity in entities:
            results = self._retrieve_for_entity(entity)
            all_chunks_with_scores.extend(results)

        if not all_chunks_with_scores:
            return []

        retrieved_chunks = [
            RetrievedChunk(
                text=chunk.text,
                metadata=chunk.metadata,
                score=score
            )
            for chunk, score in all_chunks_with_scores
        ]

        return self.reranker.rerank(query, retrieved_chunks, top_k=top_k)