# src/generation/llm_clients/base_client.py

from abc import ABC, abstractmethod
from typing import List, Dict, Any

class BaseLLMClient(ABC):
    """
    所有 LLM 客户端的抽象基类。
    统一输入：messages (List[Dict])，统一输出：str
    """

    @abstractmethod
    def generate(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """
        生成回答。
        
        Args:
            messages: 对话历史，格式如 [{"role": "user", "content": "What is aspirin?"}]
            **kwargs: 模型参数（temperature, max_tokens 等）
            
        Returns:
            生成的文本回答
        """
        pass