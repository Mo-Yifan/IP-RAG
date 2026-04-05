# src/generation/prompts/base_prompt.py

import os
from pathlib import Path
from typing import List
from abc import ABC, abstractmethod

# 假设你的 DocumentChunk 定义在 src/data/chunk.py 或类似位置
from src.data import DocumentChunk 

class BasePrompt(ABC):
    """
    Prompt 基类。
    定义了从文件加载模板的通用逻辑。
    子类必须实现 build 方法来定义具体的格式化规则。
    """

    def __init__(self, template_path: str = None):
        """
        初始化 Prompt。
        
        Args:
            template_path: 模板文件的绝对路径或相对路径。
                          如果为 None，则使用子类中定义的默认路径。
        """
        if template_path is None:
            # 子类必须覆盖这个属性，或者在构造函数中传入
            raise ValueError("template_path 不能为空，请在子类中指定默认路径或传入参数")
            
        if not os.path.exists(template_path):
            raise FileNotFoundError(f"Prompt 模板文件未找到: {template_path}")
            
        with open(template_path, "r", encoding="utf-8") as f:
            self.template = f.read().strip()

    @abstractmethod
    def build(self, question: str, retrieved_chunks: List[DocumentChunk]) -> str:
        """
        抽象方法：子类必须实现此方法。
        用于将用户问题和检索到的上下文片段组合成最终的 Prompt 字符串。
        
        Args:
            question: 用户输入的问题。
            retrieved_chunks: 检索到的文档块列表。
            
        Returns:
            格式化后的完整 Prompt 字符串。
        """
        pass