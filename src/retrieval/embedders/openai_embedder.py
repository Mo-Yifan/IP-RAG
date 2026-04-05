# src/retrieval/embedders/openai_embedder.py

import os
import logging
from typing import List
from openai import OpenAI
from .base_embedder import BaseEmbedder

logger = logging.getLogger(__name__)

class OpenAIEmbedder(BaseEmbedder):
    """
    使用 OpenAI Embedding API 的嵌入器。
    """

    SUPPORTED_MODELS = {
        "text-embedding-3-small": 1536,
        "text-embedding-3-large": 3072,
        "text-embedding-ada-002": 1536,
    }

    def __init__(
        self,
        model: str = "text-embedding-3-small",
        api_key: str = None,
        base_url: str = None
    ):
        if model not in self.SUPPORTED_MODELS:
            raise ValueError(f"Unsupported OpenAI embedding model: {model}")
        
        api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OpenAI API key is required. Set OPENAI_API_KEY env var.")
        
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model
        self._dimension = self.SUPPORTED_MODELS[model]
        logger.info(f"Initialized OpenAIEmbedder with model: {model}")

    def embed(self, texts: List[str]) -> List[List[float]]:
        try:
            response = self.client.embeddings.create(
                input=texts,
                model=self.model
            )
            # 提取嵌入向量
            embeddings = [item.embedding for item in response.data]
            return embeddings
        except Exception as e:
            logger.error(f"OpenAI embedding error: {e}")
            raise RuntimeError(f"Embedding failed: {e}")

    @property
    def dimension(self) -> int:
        return self._dimension