# src/data/__init__.py

# 导入核心数据结构
from .schemas import DocumentChunk, IPMetadata

# 导入加载器
from .loaders import BaseLoader, IPJsonLoader, DocumentChunk as _DocChunk  # 避免重复名

# 导入切分器
from .chunkers import BaseChunker, SemanticChunker, IPChunker

# 统一暴露 DocumentChunk（以 schemas 中的为准）
DocumentChunk = _DocChunk

__all__ = [
    "DocumentChunk",
    "IPMetadata",
    "BaseLoader",
    "IPJsonLoader",
    "BaseChunker",
    "SemanticChunker",
    "IPChunker",
]