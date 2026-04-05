import logging
import re
import jieba
from typing import List, Optional, Dict, Any, Set
from src.retrieval.schemas import TextChunk, RetrievedChunk
from src.retrieval.embedders.base_embedder import BaseEmbedder
from src.retrieval.vectorstores.base_vectorstore import BaseVectorStore
from src.retrieval.rerankers.base_reranker import BaseReranker
from src.generation.prompts.patent_prompt import PatentQAPrompt
from src.generation.llm_clients.base_client import BaseLLMClient

logger = logging.getLogger(__name__)


class RAGChain:
    """ IP RAG 核心链路：专用于专利/IP检索
    Embedding -> Retrieval -> [Rerank] -> Prompt Building -> LLM Generation
    """

    def __init__(self, embedder: BaseEmbedder, vectorstore: BaseVectorStore, llm: Any, prompt_builder: PatentQAPrompt, reranker: Optional[BaseReranker] = None, top_k: int = 10):
        self.embedder = embedder
        self.vectorstore = vectorstore
        self.llm = llm
        self.prompt_builder = prompt_builder
        self.reranker = reranker
        self.top_k = top_k

        # 1. 设置检索倍数 (保持不变)
        if self.reranker and not isinstance(self.reranker, type(None)):
            self.retrieve_multiplier = 5
            logger.info(f"🚀 Reranker 已启用: {type(self.reranker).__name__}")
        else:
            self.retrieve_multiplier = 1
            logger.info("⚠️ Reranker 未启用 (或为 NoneReranker)，将直接使用向量检索结果。")
        logger.info(f"RAGChain initialized with top_k={self.top_k}, retrieve_multiplier={self.retrieve_multiplier}")

    def _extract_query_keywords(self, query: str) -> Set[str]:
        """
        提取专利查询中的关键词。
        使用 jieba 进行中文语义分词，替代简单的正则切分。
        """
        keywords = set()
        
        # ==========================================
        # 1. 第一优先级：提取专利号 (保持正则逻辑不变)
        # ==========================================
        patent_pattern = r'(?<![A-Z])(?:US|EP|WO|JP|CN|KR|DE|GB|FR|AU)[-\s]*\d{8,15}[-\s]*[A-Z]\d?(?![A-Z0-9])'
        patent_matches = re.findall(patent_pattern, query, re.IGNORECASE)
        clean_patents = [p.upper().replace(" ", "-") for p in patent_matches]
        keywords.update(clean_patents)

        # 如果已经匹配到专利号，其实可以直接返回了
        # 但为了防止用户输入 "US12345678 和 锂离子电池" 这种混合查询，我们继续提取
        if clean_patents:
            print(f"🔍 捕获到专利号: {clean_patents}")

        # ==========================================
        # 2. 第二优先级：使用 jieba 提取核心术语
        # ==========================================
        
        # B. 使用 jieba 进行分词 (精准模式)
        # jieba.cut 返回的是一个生成器，我们转成列表
        words = list(jieba.cut(query))
        
        # C. 定义停用词表 (Stop Words)
        # 这些词在专利检索中通常没有意义，反而会干扰召回
        stop_words = {
            # 中文虚词/疑问词
            '的', '了', '和', '或', '是', '在', '有', '哪些', '什么', '怎么', '如何', '一种', '及其', 
            '以及', '吗', '呢', '吧', '啊', '呀', '为', '对', '等', '该', '此', '其', '即', '就',
            # 英文虚词 (虽然中文多，但为了保险)
            'the', 'and', 'or', 'of', 'to', 'in', 'for', 'on', 'with', 'by', 'is', 'are', 'it', 'this', 'that',
            'be', 'an', 'as', 'at', 'so', 'if', 'we', 'do', 'will', 'would', 'can', 'could', 'may', 'might'
        }
        
        # D. 过滤并收集关键词
        for word in words:
            word = word.strip().lower() # 统一转小写
            # 过滤条件：
            # 1. 长度 > 1 (去除单字)
            # 2. 不在停用词表中
            # 3. 不是纯数字 (年份提取逻辑单独处理)
            # 4. 不是专利号 (避免重复)
            if (len(word) > 1 
                and word not in stop_words 
                and not word.isdigit() 
                and not re.match(r'^(us|ep|wo|cn)\w+', word)):
                keywords.add(word)

        # ==========================================
        # 3. 【可选】保留特定技术词 (如权利要求相关)
        # ==========================================
        specific_terms = ['claim', 'structure', 'method', 'apparatus', 'system', 'circuit', 'embodiment']
        for term in specific_terms:
            if term in query.lower():
                keywords.add(term)

        print(f"✅ jieba 分词后提取到的关键词: {keywords}")
        return keywords

    def invoke(self, question: str, valid_only: bool = False) -> Dict[str, Any]:
        logger.info(f"🔍 处理专利问题: {question[:50]}...")
        final_query = question

        try:
            query_keywords = self._extract_query_keywords(final_query)
            logger.info(f"🔍 多路检测：提取到关键词 {query_keywords}")

            # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
            # 🔴🔴🔴 【终极红绿灯机制】开始 🔴🔴🔴
            # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
            retrieved_chunks = [] # 初始化为空
            is_pure_patent_query = False
            extracted_patent_id = None

            # 1. 修正点：只检查前缀，不检查横杠和数字
            for kw in query_keywords:
                # 修改正则：只匹配开头的 US/EP/WO/CN，不管后面有没有横杠
                if re.match(r'^(US|EP|WO|CN)', kw):
                    is_pure_patent_query = True
                    extracted_patent_id = kw
                    break

            # 2. 如果是纯专利查询
            if is_pure_patent_query:
                logger.info(f"🛑 红灯！检测到纯专利号查询: {extracted_patent_id}。跳过多路检索。")

                # 构造过滤条件
                # 注意：这里假设你的 Chroma metadata 字段叫 'id' 或 'patent_id'
                # 根据你的日志 `legal_status`，我猜测可能是 'id'，请根据实际情况调整
                filter_criteria = {"patent_id": extracted_patent_id} 
                
                if valid_only:
                    filter_criteria["legal_status"] = "Valid"

                # 直接搜索
                # 注意：这里 top_k 设为 1，因为我们只需要那个特定的专利
                retrieved_chunks = self.vectorstore.search(
                    query_embedding=[0.0] * 1024, 
                    top_k=20, # 只取 1 个，因为 ID 是唯一的
                    filter_criteria=filter_criteria
                )

                # 3. 【关键修复】防御性检查：如果没查到，直接报错，不降级！
                if not retrieved_chunks:
                    logger.error(f"❌ 红灯机制生效，但数据库中未找到 ID 为 {extracted_patent_id} 的专利！")
                    # 这里我们不降级，直接返回空结果，让你看到底是没数据还是逻辑错了
                    return {
                        "answer": f"抱歉，数据库中未找到专利 {extracted_patent_id}。",
                        "citations": [],
                        "question": final_query
                    }
                else:
                    logger.info(f"✅ 红灯机制生效！精确检索成功，找到 {len(retrieved_chunks)} 个结果。")

            # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
            # 🟢 【红绿灯机制】结束：如果不是纯专利查询，走原来的多路检索
            # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
            else:
                logger.info(f"🔄 启动 IP 多路检索，关键词: {list(query_keywords)[:5]}")
                all_retrieved_chunks: List[RetrievedChunk] = []
                seen_texts: Set[str] = set()

                for kw in query_keywords:
                    if re.match(r'\b(?:US|EP|WO|CN)\w+', kw):
                        sub_query = kw
                    else:
                        sub_query = kw

                    logger.debug(f" → 子查询: {sub_query}")
                    sub_embedding = self.embedder.embed(texts=[sub_query], dimensions=1024).embeddings[0]
                    search_top_k = self.top_k * self.retrieve_multiplier
                    filter_criteria = {"legal_status": "Valid"} if valid_only else None

                    chunks = self.vectorstore.search(
                        query_embedding=sub_embedding,
                        top_k=search_top_k,
                        filter_criteria=filter_criteria
                    )

                    for chunk in chunks:
                        if chunk.text not in seen_texts:
                            seen_texts.add(chunk.text)
                            all_retrieved_chunks.append(chunk)

                retrieved_chunks = all_retrieved_chunks
                logger.info(f"✅ 专利多路检索完成，共召回 {len(retrieved_chunks)} 个唯一片段")

            # ==================================================================
            # ✅ 后续流程：重排序、生成回答
            # ==================================================================
            # 注意：如果上面红灯查到了数据，这里会直接跳过重排序，或者只对那 1 个结果做形式上的重排序
            if not retrieved_chunks:
                return {
                    "answer": "未在专利数据库中找到相关信息。",
                    "citations": [],
                    "question": final_query
                }

            # 重排序 (如果只有一个结果，这一步其实很快)
            if self.reranker:
                logger.info(f"🛑 调试：当前 self.top_k 的值是 [{self.top_k}], 内存地址是 {id(self)}")
                logger.info(f"🔄 启动重排序 ({type(self.reranker).__name__}): {len(retrieved_chunks)} -> {self.top_k} ...")
                try:
                    instruction_query = f"Retrieve this sentence: {final_query}"
                    retrieved_chunks = self.reranker.rerank(
                        query=instruction_query,
                        chunks=retrieved_chunks,
                        top_k=self.top_k
                    )
                    if retrieved_chunks:
                        logger.info(f"✅ 重排序完成。新 Top 1: {retrieved_chunks[0].metadata.get('patent_id', 'Unknown')} (Score: {retrieved_chunks[0].score:.4f})")
                    else:
                        logger.warning("⚠️ 重排序后结果为空！")
                except Exception as e:
                    logger.error(f"❌ Rerank 过程出错: {e}", exc_info=True)
                    retrieved_chunks = retrieved_chunks[:self.top_k]
            else:
                retrieved_chunks = retrieved_chunks[:self.top_k]

            # 构建 Prompt 和生成回答 (保持不变)
            context_chunks = [
                TextChunk(text=c.text, metadata=c.metadata)
                for c in retrieved_chunks
            ]
            prompt = self.prompt_builder.build(final_query, context_chunks)
            answer = self.llm.generate(prompt)

            citations = [
                {
                    "patent_id": c.metadata.get("patent_id", "Unknown"),
                    "title": c.metadata.get("title", "Unknown Title"),
                    "applicant": c.metadata.get("applicant", "Unknown Applicant"),
                    "score": float(getattr(c, 'score', 0.0))
                }
                for c in retrieved_chunks
            ]

            return {
                "answer": answer,
                "citations": citations,
                "question": final_query
            }

        except Exception as e:
            logger.error(f"RAGChain invoke 发生严重错误: {e}", exc_info=True)
            return {
                "answer": f"抱歉，处理您的请求时发生错误: {str(e)}",
                "citations": [],
                "question": final_query
            }