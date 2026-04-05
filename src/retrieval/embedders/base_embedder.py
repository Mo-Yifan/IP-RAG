# src/retrieval/embedders/base_embedder.py

from abc import ABC, abstractmethod
from typing import List, Optional
from ..schemas import EmbeddingResult


class BaseEmbedder(ABC):
    """
    文本嵌入模型基类
    输入：文本列表
    输出：嵌入向量 + 元数据（通过 EmbeddingResult 封装）
    """

    @abstractmethod
    def embed(
        self,
        texts: List[str],
        batch_size: int = 32,
        show_progress: bool = False
    ) -> EmbeddingResult:
        """
        为文本列表生成嵌入向量
        
        Args:
            texts: 待嵌入的文本列表
            batch_size: 批处理大小
            show_progress: 是否显示进度条
            
        Returns:
            EmbeddingResult: 包含 texts, embeddings, metadatas
        """
        pass