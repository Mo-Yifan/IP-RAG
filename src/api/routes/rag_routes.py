# src/api/routes/rag_routes.py
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
import logging

from src.api.dependencies import get_rag_chain
from src.generation.rag_chain import RAGChain

logger = logging.getLogger(__name__)
router = APIRouter()

# ✅ 修改 1: 定义新的请求模型
# 去掉了 fda_only，增加了 valid_only (假设专利有有效/失效状态)
class QueryRequest(BaseModel):
    question: str = Field(..., description="用户提出的技术或专利问题")
    top_k: int = Field(default=3, description="检索返回的专利数量")
    valid_only: bool = Field(default=False, description="是否仅检索有效的专利") # 可选逻辑

# ✅ 修改 2: 定义响应模型 (匹配前端期望)
class PatentCitation(BaseModel):
    patent_id: str = Field(..., description="专利公开号")
    title: str = Field(..., description="专利标题")
    abstract_snippet: str = Field(..., description="摘要片段")
    score: float = Field(..., description="相关度分数")

class QueryResponse(BaseModel):
    answer: str = Field(..., description="生成的回答")
    citations: List[PatentCitation] = Field(..., description="引用的专利列表")
    question: str = Field(..., description="原始问题")

@router.post("/query", response_model=QueryResponse)
async def query_patent(
    request: QueryRequest,
    rag_chain: RAGChain = Depends(get_rag_chain)
):
    """
    专利智能检索接口
    -----------------
    接收技术问题，检索专利库，并生成带引用的回答。
    """
    if rag_chain is None:
        logger.error("❌ 致命错误：RAG Chain 未初始化")
        raise HTTPException(status_code=500, detail="RAG chain not initialized.")

    try:
        logger.info(f"🔍 收到专利查询: [Q='{request.question}', TopK={request.top_k}, ValidOnly={request.valid_only}]")

        # ✅ 修改 3: 调用 RAG Chain
        # 注意：这里需要确保你的 RAGChain.invoke 方法支持 valid_only 参数
        # 如果不支持，直接传入 question 即可
        result = rag_chain.invoke(
            question=request.question,
            valid_only=request.valid_only # 这里需要你在 RAGChain 里做处理
        )

        # ✅ 修改 4: 数据格式化 (确保返回给前端的字段名是 patent_id 而不是 source)
        # 假设 RAGChain 返回的 citations 是一个字典列表
        formatted_citations = []
        for cit in result.get("citations", []):
            formatted_citations.append({
                "patent_id": cit.get("patent_id", "未知"),
                "title": cit.get("title", "无标题"),
                "abstract_snippet": cit.get("content", "")[:200] + "...", # 截取片段
                "score": cit.get("score", 0.0)
            })

        final_response = {
            "answer": result.get("answer", "未找到相关信息"),
            "citations": formatted_citations,
            "question": request.question
        }

        return final_response

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ 查询失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"服务器内部错误: {str(e)}")