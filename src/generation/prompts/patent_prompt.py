# src/generation/prompts/patent_prompt.py

from .base_prompt import BasePrompt
from src.data import DocumentChunk
from typing import List
from .citation_utils import format_citations 
from pathlib import Path

class PatentQAPrompt(BasePrompt):
    """
    专利智能检索系统 Prompt。
    继承自 BasePrompt，专门用于处理技术专利相关的问答。
    """

    def __init__(self, template_path: str = None):
        # ✅ 关键点：定义专利系统专用的默认模板路径
        if template_path is None:
            template_path = Path(__file__).parent / "IP_qa_prompt.txt"
        
        # 调用父类的 __init__ 来加载文件
        super().__init__(template_path)

    def build(self, question: str, retrieved_chunks: List[DocumentChunk]) -> str:
        """
        构建专利问答 Prompt。
        将检索到的专利片段格式化，并插入到预定义的模板中。
        
        Args:
            question: 用户的技术问题。
            retrieved_chunks: 检索到的专利片段列表。
            
        Returns:
            最终发送给 LLM 的字符串。
        """
        # 使用工具函数将 DocumentChunk 列表转换为可读的文本
        # 例如：[1] CN123456A - 一种新型电池... [2] US987654B...
        context = format_citations(retrieved_chunks)
        
        # 假设你的模板文件中有 {context} 和 {question} 两个占位符
        return self.template.format(context=context, question=question)