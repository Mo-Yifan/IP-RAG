import logging
import re
from typing import List, Set, Dict
from .base_reranker import BaseReranker
from src.retrieval.schemas import RetrievedChunk
from FlagEmbedding import FlagReranker

logger = logging.getLogger(__name__)

class BGEReranker(BaseReranker):
    """
    专为 IP RAG API (知识产权/专利检索) 设计的重排序器。
    
    设计逻辑：
    1. 语义匹配：使用 BGE 模型计算基础相关性。
    2. 结构增强：权利要求(Claims) > 实施例(Examples) > 说明书。
    3. 元数据路由：IPC分类号、发明领域匹配奖励。
    """

    # 定义专利特有的字段奖励权重
    BOOST_WEIGHTS = {
        'claims': 1.5,       # 权利要求书：法律保护范围，权重最高
        'example': 1.2,      # 实施例：具体数据支持
        'description': 0.8,  # 详细说明
        'abstract': 0.5,     # 摘要
        'title': 1.0         # 标题
    }

    # 常见的专利技术术语映射 (可扩展)
    TECH_TERMS = {
        'anode': ['negative electrode', 'anode'],
        'cathode': ['positive electrode', 'cathode'],
        'separator': ['membrane', 'separator'],
        'dendrite': ['dendrite', 'branching', 'short circuit']
    }

    def __init__(self, model_name: str = 'BAAI/bge-reranker-v2-m3'):
        logger.info(f"🚀 Loading IP Reranker: {model_name}")
        self.reranker = FlagReranker(model_name, use_fp16=True)
        logger.info("IP Reranker (BGE) loaded successfully.")

    def _extract_technical_keywords(self, query: str) -> Set[str]:
        """
        提取查询中的技术关键词。
        过滤停用词，并进行基础的术语归一化。
        """
        # 移除标点并转小写
        cleaned = re.sub(r'[^\w\s-]', ' ', query.lower())
        words = cleaned.split()
        
        # 过滤掉常见停用词（可根据专利词典扩展）
        stop_words = {'the', 'and', 'or', 'of', 'to', 'in', 'a', 'an', 'for', 'on', 'with', 'by', 'is', 'are', 'be', 'as'}
        return {word for word in words if len(word) > 2 and word not in stop_words}

    def _calculate_metadata_boost(self, query_keywords: Set[str], chunk: RetrievedChunk) -> float:
        """
        根据专利元数据计算奖励分数。
        """
        boost = 0.0
        metadata = chunk.metadata

        # 1. 字段类型奖励 (Claims 最重要)
        field_type = metadata.get('field', '').lower()
        if 'claim' in field_type:
            boost += self.BOOST_WEIGHTS['claims']
        elif 'example' in field_type:
            boost += self.BOOST_WEIGHTS['example']
        elif 'abstract' in field_type:
            boost += self.BOOST_WEIGHTS['abstract']
        elif 'description' in field_type:
            boost += self.BOOST_WEIGHTS['description']
        elif 'title' in field_type:
            boost += self.BOOST_WEIGHTS['title']

        # 2. IPC 分类号匹配 (如果查询包含特定分类号)
        # 假设 metadata 中包含 'ipc_class' 字段
        ipc_class = metadata.get('ipc_class', '').upper()
        query_ipc = next((word for word in query_keywords if re.match(r'[A-Z]\d', word.upper())), None)
        if query_ipc and query_ipc in ipc_class:
            boost += 0.5

        # 3. 发明领域匹配
        field_of_invention = metadata.get('field_of_invention', '').lower()
        for keyword in query_keywords:
            if keyword in field_of_invention:
                boost += 0.3
                break

        return boost

    def _calculate_term_expansion_boost(self, query: str, chunk_text: str) -> float:
        """
        检查技术术语的同义词匹配。
        例如：查询 "negative electrode" 应该匹配文本中的 "anode"。
        """
        boost = 0.0
        lower_text = chunk_text.lower()
        
        for standard_term, variations in self.TECH_TERMS.items():
            if standard_term in query.lower():
                # 如果查询中包含标准术语，检查文本中是否包含其变体
                if any(variation in lower_text for variation in variations):
                    boost += 0.2
                    break
        return boost

    def rerank(
        self, 
        query: str, 
        chunks: List[RetrievedChunk], 
        top_k: int = 5
    ) -> List[RetrievedChunk]:
        """
        执行 IP 专用的重排序逻辑。
        """
        if not chunks:
            return []

        logger.info(f"🔍 开始对 {len(chunks)} 个候选块进行重排序 (Query: {query[:30]}...)")

        # 1. 提取关键词
        query_keywords = self._extract_technical_keywords(query)

        # 2. BGE 基础语义打分
        # 注意：FlagReranker v2 支持批量推理，效率更高
        pairs = [[query, chunk.text] for chunk in chunks]
        
        try:
            # compute_score 返回的是 List[float] 或 Tuple[List[float], ...]
            raw_scores = self.reranker.compute_score(pairs, normalize=True)
            
            # 处理返回值 (兼容不同版本的 FlagEmbedding)
            if isinstance(raw_scores, list):
                scores = raw_scores
            elif isinstance(raw_scores, tuple):
                scores = raw_scores[0]
            else:
                scores = [raw_scores] * len(pairs)
                
        except Exception as e:
            logger.error(f" Reranker 推理错误: {e}")
            # 出错时回退到原始顺序
            scores = [1.0] * len(chunks)

        # 3. 计算增强分数并赋值
        final_results = []
        for i, chunk in enumerate(chunks):
            if i >= len(scores):
                break
                
            base_score = float(scores[i])
            metadata_boost = self._calculate_metadata_boost(query_keywords, chunk)
            term_boost = self._calculate_term_expansion_boost(query, chunk.text)
            
            # 综合分数
            final_score = base_score + metadata_boost + term_boost
            
            # 创建新对象或修改副本 (避免污染缓存)
            chunk.score = final_score
            final_results.append(chunk)

        # 4. 按分数降序排序
        final_results.sort(key=lambda x: x.score, reverse=True)

        # 5. 调试日志：打印 Top 结果分析
        if logger.isEnabledFor(logging.INFO):
            logger.info("🏆━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
            logger.info(f"🏆 IP RAG 重排序结果 (Query: {query[:40]}...)")
            logger.info("🏆━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
            for i, chunk in enumerate(final_results[:top_k]):

                meta = chunk.metadata
                field = meta.get('source_field', 'Unknown')
                title = meta.get('title', 'No Title')[:30]
                ipc = meta.get('classification', 'N/A') # 注意：根据你的日志，这里大概率会是 N/A，因为没存进去
                score = chunk.score
                # 标记高分原因
                marker = ""
                if 'claim' in field.lower():
                    marker = "⚡[Claims]"
                elif 'example' in field.lower():
                    marker = "📊[Example]"
                    
                logger.info(f"#{i+1} {marker} | Score: {score:.3f} | {field} | {title}... | IPC: {ipc}")

        return final_results[:top_k]