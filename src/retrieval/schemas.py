"""
统一数据模型（IP RAG 适配版 - 英文变量名）
=========================================
使用英文变量名以符合编程规范，并增加对中文 JSON 字段的映射支持。
"""

from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field


# ================
# 🧾 专利元数据 (英文变量名)
# ================

class PatentMetadata(BaseModel):
    # --- 基础信息 ---
    # ✅ 修改点：类型改为 Optional[str]，默认值设为 None
    seq_no: Optional[str] = Field(default=None, description="序号")
    patent_id: str = Field(..., description="专利号") # 这个保持必填
    pdf_url: Optional[str] = Field(default=None, description="PDF链接")
    text_url: Optional[str] = Field(default=None, description="Text链接")
    title: str = Field(default="未知专利", description="专利标题")
    inventor: Optional[str] = Field(default=None, description="发明人")
    pub_date: Optional[str] = Field(default=None, description="公开日期")
    page_count: Optional[str] = Field(default=None, description="页数")

    # --- 专利内容字段 ---
    # ✅ 修改点：内容字段允许为空
    abstract: Optional[str] = Field(default=None, description="专利摘要")
    background: Optional[str] = Field(default=None, description="背景技术")
    description: Optional[str] = Field(default=None, description="详细描述")
    claims: Optional[str] = Field(default=None, description="权利要求书")

    # --- 追踪信息 ---
    # ✅ 修改点：这两个字段在初始化时也允许为空，因为我们在 split 函数里没传它们
    # 然后在 _make_chunk 里再赋值
    source_field: Optional[str] = Field(default=None, description="该文本块来自哪个字段")
    chunk_index: Optional[int] = Field(default=None, description="该块在所属字段中的序号")


# ================
# ✂️ 文本块（Chunk）
# ================

class TextChunk(BaseModel):
    """
    切分后的文本单元（对应 Chunker 的输出 & VectorStore 的输入）
    """
    text: str
    metadata: PatentMetadata  # 👈 使用上面定义的 PatentMetadata 类
    source_id: Optional[str] = None  # 唯一标识符，如 "US-20250246757-A1_claims_0"
    embedding: Optional[List[float]] = None  # 向量数据


# ================
# 🔍 检索结果
# ================

class RetrievedChunk(BaseModel):
    """
    检索返回的结果单元
    """
    text: str
    metadata: Dict[str, Any]
    score: float  # 相似度分数 (0-1)


# ================
# 🧠 嵌入批处理结果
# ================

class EmbeddingResult(BaseModel):
    """
    批量嵌入的返回结构
    """
    texts: List[str]
    embeddings: List[List[float]]
    metadatas: List[Dict[str, Any]]