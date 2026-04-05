# src/retrieval/vectorstores/faiss_store.py

import os
import pickle
import logging
import faiss
import numpy as np
from typing import List, Optional
from .base_vectorstore import BaseVectorStore
from src.data import DocumentChunk

logger = logging.getLogger(__name__)

class FAISSVectorStore(BaseVectorStore):
    """
    使用 FAISS 作为向量存储后端。
    注意：FAISS 不存储元数据，需额外用列表维护。
    """

    def __init__(self, dimension: int):
        self.dimension = dimension
        self.index = faiss.IndexFlatIP(dimension)  # 内积（需先归一化）
        self.chunks: List[DocumentChunk] = []      # 按添加顺序存储 chunks

    def _normalize(self, vectors: np.ndarray) -> np.ndarray:
        """L2 归一化，使内积 = 余弦相似度"""
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        return vectors / norms

    def add_documents(self, chunks: List[DocumentChunk], embeddings: List[List[float]]) -> None:
        if not chunks:
            return
        embeddings_np = np.array(embeddings, dtype=np.float32)
        embeddings_np = self._normalize(embeddings_np)
        self.index.add(embeddings_np)
        self.chunks.extend(chunks)
        logger.info(f"Added {len(chunks)} documents to FAISS. Total: {self.index.ntotal}")

    def search(
        self,
        query_embedding: List[float],
        k: int = 5,
        filter_metadata: Optional[dict] = None
    ) -> List[DocumentChunk]:
        if self.index.ntotal == 0:
            return []

        # 归一化查询向量
        query_vec = np.array([query_embedding], dtype=np.float32)
        query_vec = self._normalize(query_vec)

        # 初步检索（FAISS 不支持过滤，先取更多结果）
        expanded_k = min(k * 5, self.index.ntotal)
        distances, indices = self.index.search(query_vec, expanded_k)

        results = []
        for idx in indices[0]:
            if idx == -1:
                continue
            chunk = self.chunks[idx]

            # 手动应用元数据过滤
            if filter_metadata:
                match = True
                for key, value in filter_metadata.items():
                    if chunk.metadata.dict().get(key) != value:
                        match = False
                        break
                if not match:
                    continue

            results.append(chunk)
            if len(results) >= k:
                break

        return results

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        faiss.write_index(self.index, f"{path}.faiss")
        with open(f"{path}.pkl", "wb") as f:
            pickle.dump(self.chunks, f)
        logger.info(f"FAISS store saved to {path}")

    def load(self, path: str) -> None:
        self.index = faiss.read_index(f"{path}.faiss")
        with open(f"{path}.pkl", "rb") as f:
            self.chunks = pickle.load(f)
        logger.info(f"FAISS store loaded from {path}")