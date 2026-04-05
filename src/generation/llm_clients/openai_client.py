# src/generation/llm_clients/openai_client.py

import os
import logging
from typing import List, Dict, Any
from openai import OpenAI
from .base_client import BaseLLMClient

logger = logging.getLogger(__name__)

class OpenAILLMClient(BaseLLMClient):
    """
    使用 OpenAI 兼容 API 的客户端（支持 OpenAI 官方或 Azure）
    """

    def __init__(
        self,
        api_key: str = None,
        base_url: str = None,
        model: str = "gpt-4o-mini"
    ):
        api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OpenAI API key is required. Set OPENAI_API_KEY env var.")

        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model
        logger.info(f"Initialized OpenAI client with model: {model}")

    def generate(self, messages: List[Dict[str, str]], **kwargs) -> str:
        # 设置默认参数
        params = {
            "model": self.model,
            "messages": messages,
            "temperature": kwargs.get("temperature", 0.0),  # 临床场景建议低 temperature
            "max_tokens": kwargs.get("max_tokens", 512),
            "top_p": kwargs.get("top_p", 1.0),
        }

        try:
            response = self.client.chat.completions.create(**params)
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            raise RuntimeError(f"LLM generation failed: {e}")