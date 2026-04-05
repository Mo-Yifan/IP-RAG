# src/__init__.py

"""
DrugBank Clinical RAG System
============================

一个基于 DrugBank 数据库的临床药物问答系统，
结合嵌入检索（Retrieval）、重排序（Reranking）与大语言模型（LLM）生成权威回答。

核心模块：
- config:     全局配置管理
- retrieval:  嵌入、向量存储与重排序
- generation: 提示构建与 LLM 集成
- api:        Web API 与用户界面

版本: 0.1.0
"""

__version__ = "0.1.0"
__author__ = "DrugBank RAG Team"
__license__ = "Apache-2.0"

# 可选：暴露高层 API（按需启用）
# from src.api.app import app
# from src.config import settings
# 
# __all__ = ["app", "settings"]