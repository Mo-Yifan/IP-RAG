# src/retrieval/vectorstores/chroma_store.py
import logging
import chromadb
from typing import List, Optional
from .base_vectorstore import BaseVectorStore
from ..schemas import TextChunk, RetrievedChunk  # ✅ 使用 retrieval.schemas

logger = logging.getLogger(__name__)


class ChromaVectorStore(BaseVectorStore):
    """
    使用 ChromaDB 作为向量存储后端。
    优势：原生元数据过滤、自动 ID、持久化简单。
    """
    def __init__(
        self, 
        collection_name: str = "ip_rag",  # ✅ 修改1: 改名为 ip_rag
        persist_directory: str = "./artifacts/vectors/chroma"
    ):
        self.client = chromadb.PersistentClient(path=persist_directory)
        # 注意：我们自己提供 embedding，所以禁用内部 embedding function
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=None
        )
        logger.info(f"ChromaDB collection '{collection_name}' ready.")

    def add_chunks(self, chunks: List[TextChunk]) -> None:
        """
        实现基类抽象方法：将 TextChunk 列表存入 Chroma
        【新增】支持自动分批插入，避免超过最大 Batch Size 限制
        """
        if not chunks:
            return

        # 验证所有 chunk 都有 embedding
        for i, chunk in enumerate(chunks):
            if chunk.embedding is None:
                raise ValueError(
                    f"TextChunk[{i}] missing embedding. "
                    "Did you forget to call embedder and assign chunk.embedding?"
                )

        # === 关键修改：设置批次大小 ===
        # ChromaDB 默认限制约为 5000-6000，为了安全设为 1000
        BATCH_SIZE = 1000
        total_count = len(chunks)
        logger.info(f"💾 开始存入 {total_count} 个文档 (分批处理，每批 {BATCH_SIZE} 条)...")

        for i in range(0, total_count, BATCH_SIZE):
            batch_chunks = chunks[i: i + BATCH_SIZE]
            
            ids = []
            texts = []
            metadatas = [] # 用于存储清洗后的元数据
            embeddings = []

            for idx, chunk in enumerate(batch_chunks):
                global_idx = i + idx
                
                # 1. 修复 ID 生成 (使用 getattr 访问 Pydantic 对象)
                patent_id = getattr(chunk.metadata, 'patent_id', 'unk') 
                source_field = getattr(chunk.metadata, 'source_field', 'txt')
                chunk_id = f"{patent_id}_{source_field}_{global_idx}"
                ids.append(chunk_id)
                
                texts.append(chunk.text)
                embeddings.append(chunk.embedding)

                # 2. ✅ 新增：清洗元数据 (Sanitize Metadata)
                raw_metadata = chunk.metadata.model_dump()
                
                # 定义一个清洗函数
                def sanitize_value(v):
                    # 如果是基本类型，直接返回
                    if isinstance(v, (str, int, float, bool)) or v is None:
                        return v
                    # 如果是列表，转换为字符串 (例如: "Claim 1, Claim 2")
                    elif isinstance(v, list):
                        return ", ".join(str(item) for item in v)
                    # 如果是字典，转换为字符串 (或者跳过)
                    elif isinstance(v, dict):
                        # 如果是空字典 {}, Chroma 会报错，所以跳过或转字符串
                        if not v:
                            return None # 或者返回 "Empty" 字符串
                        return str(v)
                    # 其他复杂类型 (datetime, 自定义对象等) 都转为字符串
                    else:
                        return str(v)

                # 清洗整个元数据字典
                clean_metadata = {}
                for key, value in raw_metadata.items():
                    # 确保 key 是字符串
                    clean_key = str(key)
                    # 清洗 value
                    clean_value = sanitize_value(value)
                    # 只有当值合法时才加入 (过滤掉那些清洗后为 None 的空项)
                    if clean_value is not None:
                        clean_metadata[clean_key] = clean_value

                metadatas.append(clean_metadata)
            
            # 👇👇👇 [新增] 真正的写入操作：将这一批次的数据推送到 ChromaDB 👇👇👇
            self.collection.add(
                ids=ids,
                embeddings=embeddings,
                documents=texts,
                metadatas=metadatas
            )
            logger.info(f"   - 批次 {i//BATCH_SIZE + 1} (索引 {i} 到 {i+len(batch_chunks)}) 写入成功")

        logger.info(f"🎉 所有 {total_count} 个文档已成功存入 ChromaDB!")

    def search(
        self, 
        query_embedding: List[float], 
        top_k: int = 5, 
        filter_criteria: Optional[dict] = None
    ) -> List[RetrievedChunk]:
        """
        实现基类 search 方法（参数名已对齐）
        """
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=filter_criteria  # Chroma 原生支持
        )
        
        retrieved_chunks = []
        # 注意：Chroma 返回的是列表的列表，需要处理索引 [0]
        if results["ids"] and results["ids"][0]:
            for i in range(len(results["ids"][0])):
                text = results["documents"][0][i]
                metadata = results["metadatas"][0][i]
                distance = results["distances"][0][i]  # Chroma 返回的是距离（越小越相似）
                
                # 转换为相似度分数 [0,1]（可选，也可直接返回 distance）
                score = 1.0 / (1.0 + distance) 
                
                retrieved_chunks.append(
                    RetrievedChunk(
                        text=text, 
                        metadata=metadata, 
                        score=score
                    )
                )
        return retrieved_chunks

    def save(self) -> None:
        # PersistentClient 自动持久化
        logger.info("ChromaDB is automatically persisted.")

    def load(self) -> None:
        pass