# src/data/chunkers/base_chunker.py

from abc import ABC, abstractmethod
from typing import List
from src.data.loaders import DocumentChunk

class BaseChunker(ABC):
    """所有文本切分器的抽象基类"""

    @abstractmethod
    def split(self, chunks: List[DocumentChunk]) -> List[DocumentChunk]:
        """
        接收原始 chunks，返回优化后的 chunks 列表。
        可能会将一个 chunk 拆成多个，也可能保留原样。
        """
        pass