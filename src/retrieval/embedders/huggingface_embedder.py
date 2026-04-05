import logging
import warnings
from typing import List
from sentence_transformers import SentenceTransformer
from .base_embedder import BaseEmbedder
from tqdm import tqdm

logger = logging.getLogger(__name__)

class HuggingFaceEmbedder(BaseEmbedder):
    """
    使用 Hugging Face Sentence Transformers 的嵌入器。
    针对大规模专利数据进行了性能优化。
    """

    def __init__(self, model_name: str = "nomic-ai/nomic-embed-text-v1.5", device: str = "cuda"):
        """
        Args:
            model_name: 模型名称
            device: 设备 (建议 "cuda")
        """
        logger.info(f"Loading embedding model: {model_name} on {device}")
        
        # 🔴 关键修复 1: 全局忽略所有警告
        # 防止 "Sequence length is longer..." 这种警告刷屏导致 I/O 阻塞
        warnings.filterwarnings("ignore")
        
        # 加载模型时，强制指定信任远程代码（如果需要）
        self.model = SentenceTransformer(model_name, device=device, trust_remote_code=True)
        
        self._dimension = self.model.get_sentence_embedding_dimension()
        logger.info(f"Embedding dimension: {self._dimension}")

    def embed(
        self, 
        texts: List[str], 
        batch_size: int = 128, 
        show_progress: bool = True
    ) -> List[List[float]]:
        """
        生成嵌入向量
        """
        if not texts:
            return []
        
        # 自动处理空文本
        safe_texts = [text if text.strip() else " " for text in texts]
        
        total_batches = (len(safe_texts) + batch_size - 1) // batch_size
        
        logger.info(f"开始生成嵌入，总文本数: {len(safe_texts)}, 批次大小: {batch_size}")
        
        # 🔴 关键修复 2: 强制截断 + 关闭内部警告
        # 我们使用 model.encode 的内置参数，但为了彻底杜绝警告，
        # 最好在初始化时或者这里确保 tokenizer 不会啰嗦。
        # SentenceTransformer 的 encode 方法默认会截断，但有时会报警告。
        
        # 使用 tqdm 显示进度
        embeddings = self.model.encode(
            safe_texts, 
            batch_size=batch_size, 
            show_progress_bar=tqdm(range(total_batches), desc="Generating Embeddings"),
            convert_to_numpy=False,
            normalize_embeddings=True,
            # 🔴 关键修复 3: 显式告诉模型不要抱怨文本太长
            # 注意：SentenceTransformer 封装层可能不直接支持 truncate=True 参数，
            # 所以主要靠上面的 warnings.filterwarnings("ignore") 来压制
        )
        
        # 转为 list
        if hasattr(embeddings, 'tolist'):
            return embeddings.tolist()
        else:
            return [emb.tolist() if hasattr(emb, 'tolist') else list(emb) for emb in embeddings]

    @property
    def dimension(self) -> int:
        return self._dimension