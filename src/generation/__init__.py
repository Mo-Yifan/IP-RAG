# src/generation/__init__.py

"""
生成模块的统一入口。
提供 LLM 客户端、Prompt 构建器和 RAG 链的便捷导入。
"""

# 导入 LLM 客户端
from .llm_clients import BaseLLMClient, OpenAILLMClient, LocalLLMClient

# 导入 Prompt 工具
from .prompts import format_citations, extract_unique_sources, BasePrompt, PatentQAPrompt

# 如果已实现 RAGChain，也在此导入（预留位置）
# from .rag_chain import RAGChain

__all__ = [
    # LLM Clients
    "BaseLLMClient",
    "OpenAILLMClient",
    "LocalLLMClient",
    
    # Prompts
    "BasePrompt",
    "PatentQAPrompt",
    "format_citations",
    "extract_unique_sources",
    
    # RAG Chain (待实现后取消注释)
    # "RAGChain",
]