# src/retrieval/rerankers/none_reranker.py

from typing import List
from .base_reranker import BaseReranker
from src.retrieval.schemas import RetrievedChunk


class NoneReranker(BaseReranker):
    """
    空重排序器：保持原始顺序，赋予默认分数。
    用于性能测试或禁用重排序的场景。
    """

    def rerank(
        self,
        query: str,
        chunks: List[RetrievedChunk],
        top_k: int = 5
    ) -> List[RetrievedChunk]:
        # 为每个 chunk 赋予默认分数 1.0（表示“未排序”）
        for chunk in chunks[:top_k]:
            chunk.score = 1.0
        return chunks[:top_k]