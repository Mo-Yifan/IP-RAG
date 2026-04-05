"""
src/data/loaders/IP_json_loader.py
针对专利数据（JSON格式）的专用数据加载器。
该加载器借鉴了 DrugBank 加载器的健壮性设计，包含详细的日志记录和字段处理逻辑。
"""
import json
import logging
from pathlib import Path
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
from .base_loader import BaseLoader, DocumentChunk
from ..schemas import IPMetadata  # 导入 IPMetadata

logger = logging.getLogger(__name__)

class IPJsonLoader(BaseLoader):
    """
    专利 (Intellectual Property) JSON 数据加载器。
    专门用于解析包含专利元数据（如专利号、标题）和全文内容（如摘要、权利要求、说明书）的 JSON 文件。
    该加载器将长文本切分为较小的块，以便于 RAG 系统进行向量化存储和检索。
    """

    def __init__(self, chunk_size: int = 1024, chunk_overlap: int = 100):
        """
        初始化加载器。

        Args:
            chunk_size (int): 每个文本块的目标字符数。
            chunk_overlap (int): 相邻文本块之间的重叠字符数，用于保持语义连贯。
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def load(self, file_path: Path) -> List[DocumentChunk]:
        """
        从指定的 JSON 文件路径加载专利数据。
        注意：移除了外层的大 try...except，以便暴露 _parse_patent 内部的致命错误（如 Pydantic 验证错误）。

        Args:
            file_path (Path): JSON 文件的路径对象。

        Returns:
            List[DocumentChunk]: 解析后的文档块列表。
        """
        logger.info(f"📂 正在读取专利文件: {file_path}")
        chunks = []

        # 1. 读取文件
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except Exception as e:
            logger.error(f"❌ 读取文件 {file_path} 失败: {e}")
            return []

        success_count = 0
        skip_count = 0

        # 2. 处理数据（核心修改点：适配清洗后的 Dict 结构）
        # 清洗后的数据结构：{"专利号1": {...}, "专利号2": {...}}
        # 直接遍历字典的 values()
        if isinstance(data, dict):
            for item in data.values():
                try:
                    # 调用你原有的解析逻辑，无需改动
                    sub_chunks = self._parse_patent(item, source_file=file_path.name)
                    if sub_chunks:
                        chunks.extend(sub_chunks)
                        success_count += 1
                    else:
                        skip_count += 1
                except Exception as e:
                    logger.error(f"❌ 单条专利解析异常被跳过: {e}")
                    skip_count += 1
                    continue  # 继续处理下一条

        # 3. 兼容性处理（保留原有逻辑，以防万一遇到 List 结构）
        elif isinstance(data, list):
            for item in data:
                try:
                    sub_chunks = self._parse_patent(item, source_file=file_path.name)
                    if sub_chunks:
                        chunks.extend(sub_chunks)
                        success_count += 1
                    else:
                        skip_count += 1
                except Exception as e:
                    logger.error(f"❌ 单条专利解析异常被跳过: {e}")
                    skip_count += 1
                    continue

        else:
            logger.error(f"❌ 错误：文件 {file_path} 的根节点格式不支持 (非列表非字典)")
            return chunks

        logger.info(f"🎉 专利文件解析完成！成功处理: {success_count}, 跳过: {skip_count}, 生成文本块: {len(chunks)}")
        return chunks

    def _parse_patent(self, item: Dict[str, Any], source_file: str) -> List[DocumentChunk]:
        """解析单个专利条目。"""
        chunks = []
        
        try:
            # 1. 提取核心标识符
            patent_id = self._get_text(item, "专利号")
            if not patent_id:
                logger.warning("⚠️ 跳过条目：未找到专利号")
                return []

            # 2. 提取基础元数据
            # ✅ 关键修改：必须在这里提供 source_field，因为它是 IPMetadata 的必填项
            # 我们暂时给它一个 "Header" 的值，表示这是专利头部信息
            metadata = IPMetadata(
                source_file=source_file,
                patent_id=str(patent_id),
                invention_title=self._get_text(item, "专利标题"),
                publication_date=self._get_text(item, "公开日期"),
                inventor=self._get_text(item, "发明人"),
                assignee=self._get_text(item, "申请人"),
                legal_status=self._get_text(item, "法律状态"),
                classification=self._get_text(item, "Classification"),
                data_source="USPTO",
                source_field="Header"  # ✅ 补上这个必填字段！
            )

            # 3. 定义文本字段映射
            text_fields = [
                ("Abstract", "Abstract", "专利摘要"),
                ("Claims", "Claims", "权利要求书"),
                ("Description", "Description", "说明书"),
                ("Background", "Background", "背景技术"),
            ]

            # 4. 遍历并处理每个文本字段
            for key, section_type, section_name in text_fields:
                raw_text = item.get(key, "").strip()
                if raw_text:
                    try:
                        # 注意：这里 _create_chunks 会复制 metadata 并覆盖 source_field
                        field_chunks = self._create_chunks(
                            raw_text,
                            metadata, 
                            section_type,
                            section_title=section_name
                        )
                        chunks.extend(field_chunks)
                    except Exception as e:
                        logger.warning(f"⚠️ 处理字段 {key} 时出错: {e}")
                    continue

            # 5. 如果没有提取到正文，生成元数据块作为回退
            if not chunks:
                logger.info(f"ℹ️ 专利 {patent_id} 无正文内容，生成元数据块。")
                
                title = getattr(metadata, 'invention_title', 'No Title')
                fallback_text = f"Metadata: {patent_id} | {title}"

                fallback_chunk = DocumentChunk(
                    text=fallback_text,
                    metadata={
                        "source_file": source_file,
                        "patent_id": str(patent_id),
                        "invention_title": title,
                        "source_field": "Metadata_Fallback",
                        "data_source": "USPTO"
                    }
                )
                chunks.append(fallback_chunk)

        except Exception as e:
            logger.error(f"❌ 解析专利时发生错误: {e}")
            return []
        
        return chunks

    def _create_chunks(
        self, 
        text: str, 
        base_metadata: IPMetadata,  # ✅ 修改1: 类型改为 BaseModel 或 IPMetadata
        section_type: str, 
        section_title: str
    ) -> List[DocumentChunk]:
        """
        将长文本切分为 DocumentChunk 列表。
        """
        chunks = []
        start = 0
        text_len = len(text)

        # 确保 text 是字符串
        if not isinstance(text, str):
            text = str(text)

        while start < text_len:
            end = start + self.chunk_size
            chunk_text = text[start:end]

            # ✅ 修改2: 使用 Pydantic 的 .copy(update={}) 方法
            # 这里我们更新 source_field 和 content_type
            # 注意：content_type 如果不在 IPMetadata 定义中，会被 Pydantic 自动忽略或报错，建议确保定义了 extra_fields
            chunk_metadata = base_metadata.copy(update={
                "source_field": section_type,
                "content_type": f"Patent_{section_type}",
                "start_pos": start,
                "end_pos": end
            })

            try:
                # 尝试创建 Chunk 对象
                # 注意：DocumentChunk 的 metadata 参数需要是 dict 还是 IPMetadata？
                # 如果是 dict，需要加 .model_dump()；如果是对象，直接传
                chunk = DocumentChunk(text=chunk_text, metadata=chunk_metadata)
                chunks.append(chunk)
            except Exception as e:
                # 如果 DocumentChunk 构造失败，记录并跳过
                logger.error(f"❌ 创建 DocumentChunk 失败 (Section: {section_type}): {e}")

            # 滑动窗口
            if end >= text_len:
                break
            start += self.chunk_size - self.chunk_overlap

        return chunks

    @staticmethod
    def _get_text(data_dict: Dict[str, Any], *keys) -> Optional[str]:
        """
        静态工具方法：安全地从字典中提取文本值。
        参考了 DrugBank 加载器中的健壮提取逻辑。
        """
        for key in keys:
            if key in data_dict:
                value = data_dict[key]
                if value is None:
                    continue
                if isinstance(value, str):
                    return value.strip()
                elif isinstance(value, (list, dict)):
                    # 将复杂结构转换为字符串，避免丢弃数据
                    str_val = str(value)
                    return str_val.strip() if str_val else None
                else:
                    return str(value).strip()
        return None