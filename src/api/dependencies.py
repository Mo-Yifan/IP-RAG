# src/api/dependencies.py
import os
import logging
from typing import Optional
from src.config import settings

# Retrieval 组件
from src.retrieval import QwenEmbedder, ChromaVectorStore

# Generation 组件
from src.generation.prompts.patent_prompt import PatentQAPrompt 
from src.generation.llm_clients import OpenAILLMClient, LocalLLMClient
from src.generation.rag_chain import RAGChain

# ✅ 新增导入：BGE Reranker
from src.retrieval.rerankers.qwen_reranker import QwenReranker

logger = logging.getLogger(__name__)

# ✅ 1. 定义全局变量，初始化为 None
_rag_chain_instance: Optional[RAGChain] = None

def get_rag_chain() -> RAGChain:
    """
    获取 RAG Chain 单例
    """
    global _rag_chain_instance
    
    # ✅ 2. 如果已经初始化过，直接返回，不再执行下面的代码
    if _rag_chain_instance is not None:
        # 可以加一个调试日志，看看是否命中缓存
        # logger.debug("♻️ 复用已存在的 RAG Chain 实例")
        return _rag_chain_instance

    # ✅ 3. 第一次执行时才进入这里
    logger.info("🔧 初始化 IP-RAG Chain (专利检索)... (这是第一次也是唯一一次)")

    # 1. Embedder
    embedder = QwenEmbedder(
        model_path=settings.EMBEDDING_MODEL,
        device=settings.EMBEDDING_DEVICE
    )

    # 2. Vector Store
    vectorstore = ChromaVectorStore(
        collection_name=settings.CHROMA_COLLECTION, 
        persist_directory=str(settings.CHROMA_PERSIST_DIR)
    )

    # 3. LLM
    if settings.LLM_TYPE == "openai":
        llm_client = OpenAILLMClient(
            model_name=settings.OPENAI_MODEL,
            api_key=settings.OPENAI_API_KEY
        )
    elif settings.LLM_TYPE == "local":
        llm_client = LocalLLMClient(
            model_name=settings.LOCAL_LLM_MODEL,
            device=settings.LLM_DEVICE
        )
    else:
        raise ValueError(f"Unsupported LLM_TYPE: {settings.LLM_TYPE}")

    # 4. Prompt
    prompt_builder = PatentQAPrompt()

    # 5. Reranker
    logger.info("🚀 正在加载 BGE Reranker 模型...")
    try:
        reranker = QwenReranker(model_path=settings.RERANKER_MODEL)
        logger.info("✅ BGE Reranker 加载成功!")
    except Exception as e:
        logger.error(f"❌ 加载 BGE Reranker 失败: {e}")
        reranker = None

    # 6. 组装 Chain
    _rag_chain_instance = RAGChain(
        embedder=embedder,
        vectorstore=vectorstore,
        llm=llm_client,
        prompt_builder=prompt_builder,
        reranker=reranker,
        top_k=settings.FINAL_ANSWER_TOP_K
    )

    logger.info("✅ IP-RAG Chain 初始化完成!")
    return _rag_chain_instance