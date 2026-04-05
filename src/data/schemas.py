"""
src/data/schemas.py

专利领域（Intellectual Property）数据标准化 Schema。
该文件定义了数据在加载器（Loader）和分块器（Chunker）之间流转的统一格式。
"""

from typing import Dict, Any, Optional, List
from pydantic import BaseModel, Field, ConfigDict


class IPMetadata(BaseModel):
    """
    专利（IP）数据块的标准化元数据。
    适配你提供的 JSON 数据结构（中文键 -> 英文字段）。
    """
    model_config = {
        "extra": "allow"  # 允许 JSON 中存在但模型未定义的字段（如 "序号", "PDF链接"）
    }

    # --- 核心标识字段 ---
    patent_id: str = Field(
        ..., 
        description="专利号，如 US-20250246757-A1",
        json_schema_extra={"alias": "专利号"}  # 映射 JSON 中的 "专利号"
    )
    invention_title: str = Field(
        ..., 
        description="专利标题",
        json_schema_extra={"alias": "专利标题"}  # 映射 JSON 中的 "专利标题"
    )
    publication_date: str = Field(
        ..., 
        description="公开日期，格式如 2025-07-31",
        json_schema_extra={"alias": "公开日期"}  # 映射 JSON 中的 "公开日期"
    )

    # --- 内容与来源字段 ---
    Abstract: Optional[str] = Field(
        None, 
        description="专利摘要文本",
        json_schema_extra={"alias": "Abstract"}  # 保持与 JSON 一致
    )

    Classification: Optional[str] = Field(
        None, 
        description="国际专利分类号 (IPC/ CPC)",
        json_schema_extra={"alias": "Classification"}  # 保持与 JSON 一致
    )

    Background: Optional[str] = Field(
        None, 
        description="背景技术描述",
        json_schema_extra={"alias": "Background"}  # 保持与 JSON 一致
    )

    Description: Optional[str] = Field(
        None, 
        description="专利详细描述文本",
        json_schema_extra={"alias": "Description"}  # 保持与 JSON 一致
    )

    Claims: Optional[str] = Field(
        None, 
        description="专利权利要求文本",
        json_schema_extra={"alias": "Claims"}  # 保持与 JSON 一致
    )

    source_field: str = Field(
        ..., 
        description="来源字段名，用于区分块来源"
    )
    data_source: str = Field(
        default="USPTO", 
        description="原始数据来源"
    )

    # --- 可选/扩展字段 ---
    inventor: Optional[str] = Field(
        None, 
        description="发明人",
        json_schema_extra={"alias": "发明人"}  # 映射 JSON 中的 "发明人"
    )
    pdf_link: Optional[str] = Field(
        None, 
        description="PDF 下载链接",
        json_schema_extra={"alias": "PDF链接"}  # 映射 JSON 中的 "PDF链接"
    )
    text_link: Optional[str] = Field(
        None, 
        description="文本详情链接",
        json_schema_extra={"alias": "Text链接"}  # 映射 JSON 中的 "Text链接"
    )
    pages: Optional[str] = Field(
        None, 
        description="页数",
        json_schema_extra={"alias": "页数"}  # 映射 JSON 中的 "页数"
    )

    # 用于存储 JSON 中存在但模型未显式定义的字段（如 "序号", "Text链接" 等）
    extra_fields: Dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def from_json_dict(cls, data: Dict[str, Any]) -> "IPMetadata":
        """
        从 JSON 字典（键可能是中文）创建实例。
        """
        # Pydantic 会自动处理 alias 映射
        return cls(**data)

class DocumentChunk(BaseModel):
    """
    标准化的文本块，用于 RAG 系统的输入单元。
    """
    text: str = Field(
        ..., 
        min_length=1, 
        description="文本内容"
    )
    
    # ✅ 修正：将类型从 Dict[str, Any] 改回 IPMetadata
    # 这样可以利用 Pydantic 的校验功能，确保元数据符合专利格式
    metadata: IPMetadata = Field(
        ..., 
        description="结构化元数据"
    )
    
    # ✅ 新增：显式声明 embedding 字段
    # 使用 Optional 包裹，因为刚切分文本时还没有向量
    embedding: Optional[List[float]] = Field(
        default=None,
        description="文本的向量嵌入 (可选)"
    )

    # ✅ Pydantic V2 标准配置写法
    model_config = ConfigDict(
        arbitrary_types_allowed=False,
        # V2 中对应 exclude_unset 的参数，确保默认值（如 None）也被序列化
        # 这对于 Chroma 存储很重要，否则 embedding=None 会被忽略
        exclude_defaults=False 
    )

    def to_dict(self) -> Dict[str, Any]:
        """
        转换为字典，便于存储或日志。
        特别处理了 metadata 对象，确保其被正确序列化。
        """
        # ✅ Pydantic V2: 使用 model_dump() 替代 dict()
        data = self.model_dump()
        
        # 如果 metadata 还是 IPMetadata 对象（虽然在 model_dump 中通常会自动处理，但为了保险起见）
        # 或者如果我们在其他地方手动构建了对象，这里确保它是纯字典
        if hasattr(data['metadata'], 'model_dump'):
             data['metadata'] = data['metadata'].model_dump()
             
        return data