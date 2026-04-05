# src/data/loaders/base_loader.py

from abc import ABC, abstractmethod
from typing import List

# ✅ 关键修改：从 schemas 导入标准的 DocumentChunk
# 这样这个文件就不再自己定义类，而是复用 schemas.py 里的“精装修”版本
from ..schemas import DocumentChunk 

class BaseLoader(ABC):
    """所有数据加载器的抽象基类"""

    @abstractmethod
    def load(self, file_path: str) -> List[DocumentChunk]:
        """
        从指定路径加载数据，并返回 DocumentChunk 列表
        
        Args:
            file_path: 数据文件的路径
            
        Returns:
            DocumentChunk 对象列表
        """
        pass