# src/retrieval/rerankers/base_reranker.py

from abc import ABC, abstractmethod
from typing import List
from src.retrieval.schemas import RetrievedChunk


class BaseReranker(ABC):
    """
    检索结果重排序器基类。
    
    所有实现必须：
      - 输入：List[RetrievedChunk]（每个 chunk 应有 text + metadata）
      - 输出：List[RetrievedChunk]（按 score 降序，长度 ≤ top_k）
    """

    @abstractmethod
    def rerank(
        self,
        query: str,
        chunks: List[RetrievedChunk],
        top_k: int = 5
    ) -> List[RetrievedChunk]:
        """
        对检索结果进行重排序。

        Args:
            query: 用户原始查询文本
            chunks: 原始检索结果（通常来自 vector retriever）
            top_k: 返回前 K 个结果

        Returns:
            重排序后的结果列表，按 score 降序排列
        """
        pass