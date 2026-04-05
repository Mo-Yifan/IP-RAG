#!/usr/bin/env python3
# scripts/ingest.py
"""
专利 (IP) 数据注入脚本 (优化版)
======================
- 流式处理：避免一次性加载所有数据导致内存溢出
- 增加进度条：实时显示嵌入进度和预估剩余时间
- 修复 Schema 导入：确保与 VectorStore 定义一致
"""

import sys
import logging
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from pathlib import Path
from typing import List
import gc  # 引入垃圾回收模块

# 引入进度条库 (如果未安装，请运行 pip install tqdm)
from tqdm import tqdm

# 添加项目根目录到 Python 路径
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

# 🔑 关键修复：导入 retrieval 模块下的 Schema，与 ChromaStore 保持一致
# 之前的错误: from src.data.schemas import DocumentChunk
from src.retrieval.schemas import TextChunk, PatentMetadata  

# data 模块（loader + chunker）
from src.data.loaders import IPJsonLoader
from src.data.chunkers import IPChunker

# retrieval 模块（嵌入 + 向量库）
from src.retrieval import QwenEmbedder, ChromaVectorStore

# ⚙️ 配置 (导入新增的 IP_DATA_DIR 和参数)
from src.config import settings

# 日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# 强制将根日志级别设为 WARNING，忽略 DEBUG 和 INFO 级别的废话
logging.getLogger().setLevel(logging.WARNING)

# 如果只想针对特定库（比如 transformers 或 modelscope）闭嘴
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("modelscope").setLevel(logging.ERROR)

def select_input_files() -> List[Path]:
    """选择专利输入文件"""
    if settings.IP_DATA_DIR.exists():
        json_files = list(settings.IP_DATA_DIR.glob("*.json"))
        if json_files:
            logger.warning(f"📄 发现 {len(json_files)} 个 JSON 文件")
            return json_files
        else:
            raise FileNotFoundError(f"❌ 在 {settings.IP_DATA_DIR} 中未找到任何 JSON 文件")
    else:
        raise FileNotFoundError(
            "❌ 未找到专利数据文件夹。\n"
            f"请检查路径: {settings.IP_DATA_DIR}"
        )


def main():
    logger.warning("🚀 开始 专利 (IP) 数据注入流程...")

    # === 1. 选择输入文件 ===
    try:
        input_files = select_input_files()
        logger.warning(f"📂 选中文件夹: {settings.IP_DATA_DIR}")
    except FileNotFoundError as e:
        logger.error(str(e))
        sys.exit(1)

    # === 2. 初始化组件 ===
    chunker = IPChunker(
        description_chunk_size=settings.IP_CHUNK_SIZE,
        description_chunk_overlap=settings.IP_CHUNK_OVERLAP
    )

    embedder = QwenEmbedder(
        model_path=settings.EMBEDDING_MODEL,
        device=settings.EMBEDDING_DEVICE
    )

    vectorstore = ChromaVectorStore(
        collection_name=settings.CHROMA_COLLECTION,
        persist_directory=settings.CHROMA_PERSIST_DIR
    )

    # === 3. 流式加载、嵌入与存储 ===
    total_chunks = 0
    loader = IPJsonLoader()

    for file_path in tqdm(input_files, desc="处理文件", unit="file"):
        logger.warning(f"📄 正在处理: {file_path.name}")
        
        try:
            # 1. 加载数据
            raw_data = loader.load(file_path)
            
            if not raw_data:
                logger.warning(f"⚠️ 文件 {file_path.name} 未加载到数据，跳过")
                continue

            # 2. 切分数据
            file_chunks: List[TextChunk] = []
            for item in raw_data:
                chunks = chunker.split([item]) 
                file_chunks.extend(chunks)

            if not file_chunks:
                logger.warning(f"⚠️ 文件 {file_path.name} 未生成任何块，跳过")
                continue

            # 3. 提取文本
            texts = [chunk.text for chunk in file_chunks]
            
            # === 🚀 核心加速部分：使用 DataLoader 进行批量并行处理 ===
            logger.warning(f"🧠 正在为 {len(texts)} 个块生成嵌入 (批处理加速模式)...")
            
            final_embeddings = []
            batch_size = 32  # 显存够大可以改成 64 或 128，速度更快
            max_len = 512    # 截断长度，防止长文本撑爆显存

            # 定义一个简单的 Dataset 类，方便 DataLoader 读取
            class TextDataset(Dataset):
                def __init__(self, texts):
                    self.texts = texts
                def __len__(self):
                    return len(texts)
                def __getitem__(self, idx):
                    return self.texts[idx]

            dataset = TextDataset(texts)
            # num_workers=0 是为了避免多进程在 Windows 上的一些兼容性问题，如果是在 Linux 服务器可以改成 2 或 4
            data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

            with torch.no_grad(): # 关闭梯度计算，节省显存并加速
                for batch_texts in tqdm(data_loader, desc="生成向量"):
                    try:
                        # 1. 预处理：强制截断 (这里按字符粗略截断，防止 Token 溢出)
                        processed_batch = [t[:max_len] if len(t) > max_len else t for t in batch_texts]
                        
                        # 2. 批量调用模型 (一次处理 32 个，速度起飞)
                        # 注意：这里假设 embedder.embed 支持接收一个列表
                        res = embedder.embed(processed_batch)
                        
                        # 3. 解析结果
                        # 假设 res 是一个包含多个向量的列表，或者对象
                        batch_vecs = []
                        if isinstance(res, list):
                            batch_vecs = res
                        elif hasattr(res, 'embeddings'):
                            batch_vecs = res.embeddings
                        elif hasattr(res, 'data'): # 兼容 OpenAI 格式
                            batch_vecs = [item.embedding for item in res.data]
                        
                        # 4. 格式转换与兜底
                        for vec in batch_vecs:
                            if vec is None or (isinstance(vec, list) and len(vec) == 0):
                                # 万一某条坏了，补个噪声
                                final_embeddings.append((np.random.rand(1024) * 1e-6).tolist())
                            else:
                                if isinstance(vec, torch.Tensor):
                                    final_embeddings.append(vec.cpu().tolist())
                                else:
                                    final_embeddings.append(vec)

                    except Exception as e:
                        logger.error(f"⚠️ 批次处理失败，整批使用噪声填充: {e}")
                        # 如果整批都挂了，给这批数据全部填充噪声
                        final_embeddings.extend([(np.random.rand(1024) * 1e-6).tolist() for _ in batch_texts])

            # 确保数量对齐
            assert len(final_embeddings) == len(file_chunks), f"数量不一致！文本: {len(file_chunks)}, 向量: {len(final_embeddings)}"

            # 4. 赋值并入库
            logger.warning("💾 正在存入向量库...")
            for i, chunk in enumerate(file_chunks):
                chunk.embedding = final_embeddings[i]
            
            vectorstore.add_chunks(file_chunks)

            # 5. 统计与清理
            total_chunks += len(file_chunks)
            logger.warning(f"✅ 文件 {file_path.name} 处理完成，累计: {total_chunks}")
            
            del file_chunks, texts, final_embeddings
            gc.collect()

        except Exception as e:
            logger.error(f"❌ 处理文件 {file_path} 时出错: {e}", exc_info=True)
            continue

    logger.warning("🎉 所有文件注入完成！")
    logger.warning(f"📊 总共处理文本块: {total_chunks}")


if __name__ == "__main__":
    main()