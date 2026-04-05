import os
import logging
from typing import List, Dict, Any

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# 导入基类和 Schema
from .base_reranker import BaseReranker
from ..schemas import RetrievedChunk

logger = logging.getLogger(__name__)

class QwenReranker(BaseReranker):
    """
    基于本地 Qwen3-Reranker 模型的重排序器。
    适用于 Qwen3-Reranker-0.6B/4B/8B。
    """

    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
        max_length: int = 8192,  # 根据显存调整，0.6B 支持到 32k，但通常 8k 够用且快
        batch_size: int = 16,
    ):
        """
        初始化重排序器。

        Args:
            model_path: 本地模型路径
            device: 运行设备 ("cuda" 或 "cpu")
            max_length: 模型处理的最大 Token 长度
            batch_size: 推理批大小
        """
        self.device = device
        self.max_length = max_length
        self.batch_size = batch_size

        logger.info(f"🔄 正在加载本地重排序模型: {model_path} -> 设备: {device}")

        # 1. 加载分词器
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True
        )

        # 2. 加载模型
        # 使用 bf16 或 fp16 加速推理并节省显存
        model_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=model_dtype,
            device_map=device
        )
        self.model.eval()

        logger.info("✅ 重排序模型加载完成")

    def _get_score(self, query: str, text: str) -> float:
        """
        对单对 query-text 进行打分。
        """
        try:
            # Qwen Reranker 通常接受 pairs 格式输入
            inputs = self.tokenizer(
                [query],
                [text],
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt"
            ).to(self.device)

            with torch.no_grad():
                # 模型输出 logits
                logits = self.model(**inputs).logits
                # Sigmoid 将 logits 转为 0-1 之间的概率分数
                score = torch.sigmoid(logits).cpu().float().numpy()[0][0]
                return float(score)

        except Exception as e:
            logger.error(f"❌ 重排序推理错误: {e}")
            return 0.0

    def rerank(
        self,
        query: str,
        chunks: List[RetrievedChunk],
        top_k: int = 5
    ) -> List[RetrievedChunk]:
        """
        对检索结果进行重排序。

        逻辑：
        1. 批量计算 query 与每个 chunk 的相关性分数。
        2. 更新 chunk 的 score 属性。
        3. 按 score 降序排序。
        4. 返回 Top-K。
        """
        if not chunks:
            return []

        logger.debug(f"开始对 {len(chunks)} 个文档块进行重排序...")

        # --- 批量推理优化 ---
        # 如果 chunks 很多，建议分批处理以避免 OOM (Out Of Memory)
        scores: List[float] = []

        # 简单的循环推理，实际生产中可使用更高效的批处理
        for i in range(0, len(chunks), self.batch_size):
            batch_chunks = chunks[i : i + self.batch_size]
            # 这里为了演示清晰使用了单条循环，若需极致性能可改为全批量化张量输入
            for chunk in batch_chunks:
                score = self._get_score(query, chunk.text)
                scores.append(score)

        # --- 关联分数与 Chunk ---
        for i, chunk in enumerate(chunks):
            # 将计算出的分数写回 chunk 对象
            # 注意：确保 RetrievedChunk 对象允许修改 score 属性
            chunk.score = scores[i]

        # --- 排序与截断 ---
        # 按 score 降序排序
        sorted_chunks = sorted(chunks, key=lambda x: x.score, reverse=True)

        # 返回 Top-K
        return sorted_chunks[:top_k]