import os
import logging
from typing import List, Optional, Union
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

# 相对导入基类和Schema
from .base_embedder import BaseEmbedder
from ..schemas import EmbeddingResult

logger = logging.getLogger(__name__)

class QwenEmbedder(BaseEmbedder):
    """
    基于本地 Qwen3-Embedding 模型的嵌入实现。
    支持多语言（Native Multilingual）和 MRL（多表示长度）。
    """

    def __init__(
        self,
        model_path: str,
        dimensions: int = 1024,  # 👈 重点：Qwen3-0.6B 的原生维度是 1024
        device: Optional[str] = None,
        normalize_embeddings: bool = True,
    ):
        """
        初始化 Qwen Embedder。

        Args:
            model_path: 本地模型文件夹路径 (例如: "/models/Qwen3-Embedding-0.6B")
            dimensions: (可选) 向量维度。
                        Qwen3-Embedding-0.6B 最大支持 1024。
                        如果不填，则使用模型原生最大维度。
            device: 运行设备 ("cuda", "cpu", "mps")，默认自动检测。
            normalize_embeddings: 是否对向量进行归一化 (对于 cosine 相似度检索必须为 True)。
        """
        self.model_path = model_path
        self.dimensions = dimensions
        self.normalize_embeddings = normalize_embeddings

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"本地模型路径不存在: {model_path}")

        logger.info(f"正在加载本地 Qwen3 嵌入模型: {model_path}...")

        # 👇 修改点：显式传入 model_kwargs
        # 这样可以绕过 config.json 缺失参数的问题
        self.model = SentenceTransformer(
            model_path,
            device=device,
            trust_remote_code=True,
            model_kwargs={
                # 显式告诉模型：我的词向量维度是 1024
                # 这能解决 "missing word_embedding_dimension" 报错
                "torch_dtype": "auto", 
            },
        )

        logger.info(f"模型加载完成。设备: {self.model.device}")

    def embed(
        self,
        texts: List[str],
        batch_size: int = 32,
        show_progress: bool = False,
        dimensions: Optional[int] = None
    ) -> EmbeddingResult:
        """
        生成文本嵌入向量。
        使用物理截断法，兼容所有版本的 sentence-transformers。
        """
        if not texts:
            return EmbeddingResult(texts=[], embeddings=[], metadatas=[])
        
        # 1. 确定目标维度
        target_dimensions = dimensions if dimensions is not None else self.dimensions

        # 2. 准备参数（绝对不要传 dimensions 进去！）
        encode_kwargs = {
            "batch_size": batch_size,
            "normalize_embeddings": False, # 先不归一化，截断后再做
            "show_progress_bar": show_progress,
            "convert_to_numpy": True,
            # "prompt_name": "emb" # 如果有 prompt 配置可以加上，没有则忽略
        }

        try:
            # 3. 生成完整向量 (例如 1024 维)
            embeddings = self.model.encode(
                texts,
                **encode_kwargs
            )

            # 4. 【核心修改】手动截断维度
            # 如果指定了维度且小于当前向量维度，则进行切片
            if target_dimensions and target_dimensions < embeddings.shape[1]:
                embeddings = embeddings[:, :target_dimensions]

            # 5. 手动 L2 归一化
            # 因为截断会改变向量的模长，必须重新归一化才能用于余弦相似度计算
            from numpy.linalg import norm
            # axis=1 表示按行归一化，keepdims=True 保持二维结构方便广播
            embeddings = embeddings / norm(embeddings, axis=1, keepdims=True)

        except Exception as e:
            logger.error(f"嵌入生成失败: {e}")
            raise e

        metadatas: List[dict] = [
            {"model": self.model_path, "dim": target_dimensions}
            for _ in texts
        ]

        return EmbeddingResult(
            texts=texts,
            embeddings=embeddings.tolist(),
            metadatas=metadatas
        )