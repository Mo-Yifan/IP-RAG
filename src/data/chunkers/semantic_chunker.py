# src/data/chunkers/semantic_chunker.py
import logging
import json
from typing import List, Dict, Any
from .base_chunker import BaseChunker
from src.retrieval.schemas import PatentMetadata, TextChunk

logger = logging.getLogger(__name__)

class SemanticChunker(BaseChunker):
    def __init__(self, max_chunk_size: int = 512, overlap: int = 50):
        self.max_tokens = max_chunk_size
        self.overlap = overlap
        # 粗略估算：1 token ≈ 4 个字符 (英文)
        self.chars_per_token = 4

    def split(self, patents: List[PatentMetadata]) -> List[TextChunk]:
        """ 兼容基类接口：接收 List[PatentMetadata]，返回 List[TextChunk] """
        all_chunks = []
        for patent in patents:
            chunks = self.chunk_patent(patent)
            all_chunks.extend(chunks)
        logger.info(f"SemanticChunker: {len(patents)} patents → {len(all_chunks)} chunks")
        return all_chunks

    def chunk_patent(self, patent: PatentMetadata) -> List[TextChunk]:
        """ 核心逻辑：将单个 PatentMetadata 对象的各个字段切分为 TextChunk """
        chunks = []
        # 定义要处理的字段映射：(字段名, 显示名称)
        fields_to_process = [
            ("description", "description"),
            ("indications", "indications"),
            ("pharmacodynamics", "pharmacodynamics"),
            ("mechanism_of_action", "mechanism_of_action"),
            ("toxicity", "toxicity"),
            ("metabolism", "metabolism"),
            ("half_life", "half_life"),
        ]
        for field_attr, field_name in fields_to_process:
            # 1. 获取文本内容
            text = getattr(patent, field_attr, None)
            if not text or not isinstance(text, str) or not text.strip():
                continue
            # 2. 简单的按长度切分 (如果文本太长)
            sub_chunks = self._split_text_by_length(text, field_name, patent)
            chunks.extend(sub_chunks)

        # 如果没有切分出任何块，至少保留一个包含名字和ID的块（可选）
        if not chunks:
            summary_text = f"Patent: {patent.title} ({patent.patent_id}). No detailed clinical text available."
            chunks.append(TextChunk(
                text=summary_text,
                metadata=self._create_metadata(patent, "summary"),
                source_id=f"{patent.patent_id}_summary"
            ))
        return chunks

    def _split_text_by_length(self, text: str, field_name: str, patent: PatentMetadata) -> List[TextChunk]:
        """ 将长文本按 max_tokens 切分 """
        sub_chunks = []
        # 简单按字符切分（生产环境建议用 nltk 或 tiktoken 按 token 切分）
        max_chars = self.max_tokens * self.chars_per_token
        start = 0
        text_len = len(text)
        chunk_idx = 0
        while start < text_len:
            end = start + max_chars
            # 如果不是最后一段，尝试在句子边界切断，避免截断单词
            if end < text_len:
                last_period = text.rfind('.', start, end)
                if last_period > start:
                    end = last_period + 1
            segment = text[start:end].strip()
            if segment:
                source_id = f"{patent.patent_id}_{field_name}_{chunk_idx}"
                # 👇 关键修复：在 _create_metadata 中传入 source_id
                metadata = self._create_metadata(patent, field_name, source_id=source_id)
                sub_chunks.append(TextChunk(
                    text=segment,
                    metadata=metadata,
                    source_id=source_id
                ))
                chunk_idx += 1
            start = end
            # 处理 overlap (简单跳过一部分字符)
            if start < text_len and self.overlap > 0:
                pass
        return sub_chunks

    def _create_metadata(self, patent: PatentMetadata, field: str, source_id: str = None) -> Dict[str, Any]:
        """创造元数据"""
        # 🔑 关键修改：将 synonyms 列表序列化为 JSON 字符串，以兼容 Chroma
        synonyms_list = getattr(patent, 'synonyms', [])
        synonyms_str = json.dumps(synonyms_list, ensure_ascii=False)
        metadata = {
            "patent_id": patent.patent_id,
            "title": patent.title,
            "field": field
        }
        # 👇 新增：如果传入了 source_id，就加到 metadata 里
        if source_id is not None:
            metadata["source_id"] = source_id
        return metadata