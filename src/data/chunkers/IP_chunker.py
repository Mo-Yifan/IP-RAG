import re
from typing import List, Dict, Any
# ✅ 确保导入正确的 Schema
from src.retrieval.schemas import TextChunk, PatentMetadata 
from src.data.schemas import DocumentChunk
import logging

logger = logging.getLogger(__name__)

class IPChunker:
    """
    专门用于处理专利 (IP) JSON 数据的分块器。
    """

    def __init__(self, 
                 description_chunk_size: int = 1024, 
                 description_chunk_overlap: int = 128):
        self.description_chunk_size = description_chunk_size
        self.description_chunk_overlap = description_chunk_overlap

    def _split_description_by_paragraph(self, text: str) -> List[str]:
        """
        专门针对专利 Description 的分段逻辑。
        """
        text = text.strip()
        # 正则匹配：换行符 + [ + 数字/字母 + ]
        pattern = r'(\n\[\d{4}\]|\n\[\d+\]|\n\[[A-Za-z]\])'
        parts = re.split(pattern, text)
        
        paragraphs = []
        for i in range(1, len(parts), 2):
            try:
                para = parts[i] + parts[i+1]
                para = para.strip()
                if para:
                    paragraphs.append(para)
            except IndexError:
                break
        
        # 备用逻辑：如果没有匹配到，按字符切分
        if not paragraphs:
            paragraphs = [text[i:i+self.description_chunk_size] for i in range(0, len(text), self.description_chunk_size)]
        
        return paragraphs

    def split(self, documents: List[DocumentChunk]) -> List[TextChunk]:
        """
        既然 loader 已经把数据切分好了（text 有内容，source_field 有标记），
        我们只需要直接转换格式即可。
        """
        processed_chunks = []
        
        for doc in documents:
            try:
                # 1. 获取内容和来源标记
                content = doc.text
                source = doc.metadata.source_field
                
                # 2. 简单校验：如果没有内容，跳过
                if not content:
                    continue
                
                # 3. 直接构建 TextChunk
                # 注意：这里不需要再按段落切分了，因为 loader 已经切好了（看日志 Description 已经是切片了）
                # 我们只需要给每个块分配一个唯一的 chunk_index
                # 这里简单使用列表长度作为索引，或者你可以根据 patent_id 分组计数
                chunk_index = len([c for c in processed_chunks if c.metadata.patent_id == doc.metadata.patent_id])

                chunk = self._make_chunk(
                    content=content,
                    metadata=doc.metadata,
                    source_field=source,
                    patent_id=doc.metadata.patent_id,
                    chunk_index=chunk_index
                )
                processed_chunks.append(chunk)

            except Exception as e:
                logger.error(f"处理专利数据时出错: {e}")
                continue
            
        return processed_chunks

    def _make_chunk(
        self, 
        content: str, 
        metadata: PatentMetadata, 
        source_field: str, 
        patent_id: str,
        chunk_index: int
    ) -> TextChunk:
        """
        辅助函数：构建标准的 TextChunk 对象
        """
        source_id = f"{patent_id}_{source_field}_{chunk_index}"
        
        # 1. 先转字典
        metadata_dict = metadata.model_dump()
        
        # 🔴 关键修复：显式映射关键字段，确保向量库能看懂
        # 假设你的 PatentMetadata 里标题叫 invention_title
        # 我们强制把它复制一份给 'title'，这样无论哪里查都能查到
        if hasattr(metadata, 'invention_title'):
            metadata_dict['title'] = metadata.invention_title
            
        # 确保 patent_id 也在顶层（防止 model_dump 里的字段名是 patent_number 之类的）
        metadata_dict['patent_id'] = patent_id 
        
        # 2. 补充分块信息
        metadata_dict["source_field"] = source_field
        metadata_dict["chunk_index"] = chunk_index

        return TextChunk(
            text=content,
            metadata=metadata_dict,
            source_id=source_id
        )