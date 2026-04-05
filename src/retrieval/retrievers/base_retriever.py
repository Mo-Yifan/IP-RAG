# retrieval/retrievers/base_retriever.py

from abc import ABC, abstractmethod
from typing import List, Optional
from src.retrieval.schemas import RetrievedChunk
from retrieval.vectorstores.base_vectorstore import BaseVectorStore
from retrieval.rerankers.base_reranker import BaseReranker


class BaseRetriever(ABC):
    @abstractmethod
    def retrieve(
        self,
        query_text: str,
        query_embedding: List[float],
        top_k: int = 5
    ) -> List[RetrievedChunk]:
        pass


class VectorStoreRetriever(BaseRetriever):
    def __init__(
        self,
        vectorstore: BaseVectorStore,
        reranker: Optional[BaseReranker] = None
    ):
        self.vectorstore = vectorstore
        self.reranker = reranker

    def retrieve(
        self,
        query_text: str,
        query_embedding: List[float],
        top_k: int = 5
    ) -> List[RetrievedChunk]:
        # Step 1: Vector search (uses embedding)
        raw_results = self.vectorstore.search(query_embedding, top_k=top_k * 2)

        if not raw_results:
            return []

        # Step 2: Normalize to RetrievedChunk
        normalized = []
        for item in raw_results:
            if isinstance(item, RetrievedChunk):
                normalized.append(item)
            else:
                # Assume it's a chunk-like object with .text and .metadata
                text = getattr(item, 'text', "")
                metadata = getattr(item, 'metadata', {})
                if hasattr(metadata, 'dict'):
                    metadata = metadata.dict()
                normalized.append(RetrievedChunk(
                    text=text,
                    metadata=metadata,
                    score=0.0  # placeholder
                ))

        # Step 3: Rerank using query TEXT
        if self.reranker is not None:
            ranked = self.reranker.rerank(query_text, normalized, top_k=top_k)
            return ranked
        else:
            return normalized[:top_k]