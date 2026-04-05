"""
src/generation/prompts/citation_utils.py
"""

from typing import List
from src.data.schemas import DocumentChunk

def format_citations(chunks: List[DocumentChunk]) -> str:
    """
    将检索到的 chunks 转换为带引用标记的上下文字符串。
    修改点：同时注入 专利号 和 标题，让 LLM 知道它引用的到底是什么。
    """
    cited_texts = []
    for chunk in chunks:
        # 1. 获取元数据
        patent_id = getattr(chunk.metadata, "patent_id", "UNKNOWN")
        title = getattr(chunk.metadata, "title", "无标题")
        
        # 2. 构建引用标记
        # 我们给 LLM 的提示更明确一点：[ID: xxx | Title: yyy]
        # 这样 LLM 在生成答案时，就能“看”到标题了
        citation_tag = f"[Source: {patent_id} | Title: {title}]"
        
        # 3. 拼接文本
        cited_text = f"{chunk.text} {citation_tag}"
        cited_texts.append(cited_text)
    
    return "\n\n".join(cited_texts)

def extract_unique_sources(chunks: List[DocumentChunk]) -> List[dict]:
    """
    提取所有唯一的专利来源信息。
    修改点：返回字典列表，包含 ID 和 标题，方便前端直接渲染，不用再去查库。
    """
    seen = set()
    unique_sources = []
    
    for chunk in chunks:
        patent_id = chunk.metadata.get("patent_id", "UNKNOWN")
        title = chunk.metadata.get("title", "无标题")
        
        # 去重
        if patent_id not in seen:
            seen.add(patent_id)
            unique_sources.append({
                "patent_id": patent_id,
                "title": title
            })
            
    return unique_sources