# src/retrieval/__init__.py

"""
Retrieval 模块公共接口
=====================

负责从原始数据（DrugBank XML 等）到可检索文本块的全过程：
1. 解析 → 2. 切分 → 3. 嵌入 → 4. 存入向量库
"""

# 数据结构（核心输出）
from .parsers import Drug, PubmedArticle

# 解析器（Parser）
from .parsers import (
    BaseParser,
    DrugBankXMLParser,
    DrugBankJSONParser,
    PubmedXMLParser
)

# 嵌入模型（Embedder）
from .embedders import (
    BaseEmbedder,
    HuggingFaceEmbedder,
    QwenEmbedder
)

# 向量数据库（Vector Store）
from .vectorstores import (
    BaseVectorStore,
    ChromaVectorStore,
    FAISSVectorStore
)

# 重排序器（Reranker，如果已实现）
try:
    from .rerankers import (
        BaseReranker,
        BGEReranker,
        QwenReranker
    )
except ImportError:
    # 兼容未实现 reranker 的情况
    BaseReranker = None
    BGEReranker = None

# 📦 统一导出
__all__ = [
    # 数据结构
    "Drug",
    "PubmedArticle",
    
    # Parser
    "BaseParser",
    "DrugBankXMLParser",
    "DrugBankJSONParser",
    "PubmedXMLParser",
    
    # Chunker
    "BaseChunker",
    "SemanticChunker",
    "MedicalSectionChunker",
    
    # Embedder
    "BaseEmbedder",
    "HuggingFaceEmbedder",
    "QwenEmbedder",
    
    # Vector Store
    "BaseVectorStore",
    "ChromaVectorStore",
    "FAISSVectorStore",
]

# 条件性添加 Reranker
if BaseReranker is not None:
    __all__.extend(["BaseReranker", "BGEReranker", "QwenReranker"])