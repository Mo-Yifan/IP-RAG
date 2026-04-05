# src/retrieval/vectorstores/base_vectorstore.py

from abc import ABC, abstractmethod
from typing import List, Optional
from ..schemas import TextChunk, RetrievedChunk


class BaseVectorStore(ABC):
    """
    向量数据库基类
    负责存储和检索 TextChunk
    """

    @abstractmethod
    def add_chunks(self, chunks: List[TextChunk]) -> None:
        """
        将文本块及其嵌入存入向量库
        
        Args:
            chunks: 要存储的文本块列表
        """
        pass

    @abstractmethod
    def search(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        filter_criteria: Optional[dict] = None
    ) -> List[RetrievedChunk]:
        """
        根据查询向量检索相似文本块
        
        Args:
            query_embedding: 查询的嵌入向量
            top_k: 返回前 K 个结果
            filter_criteria: 元数据过滤条件（如 {"fda_approved": True}）
            
        Returns:
            List[RetrievedChunk]: 检索结果（含 text, metadata, score）
        """
        pass

    def delete_collection(self) -> None:
        """删除整个集合（可选）"""
        raise NotImplementedError("子类需实现 delete_collection 方法")