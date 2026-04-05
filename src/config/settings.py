# src/config/settings.py
import os
from pathlib import Path
from typing import Literal, Optional
from pydantic_settings import BaseSettings
from pydantic import Field, validator

class Settings(BaseSettings):
    """
    应用全局配置类。
    优先级：环境变量 > .env 文件 > 默认值
    """

    # ======================
    # 通用设置
    # ======================
    PROJECT_NAME: str = "Laomo IP RAG"
    API_V1_STR: str = "/api"
    DEBUG: bool = Field(default=False, env="DEBUG")

    # ======================
    # 路径设置（可被 paths.py 覆盖）
    # ======================
    BASE_DIR: Path = Path(__file__).resolve().parent.parent.parent  # 项目根目录
    DATA_DIR: Path = Field(default_factory=lambda: Path(os.getenv("DATA_DIR", "./data")))
    ARTIFACTS_DIR: Path = Field(default_factory=lambda: Path(os.getenv("ARTIFACTS_DIR", "./artifacts")))

    # ✅ 新增：指向包含所有专利 JSON 的文件夹（对应你截图中的 July 文件夹）
    IP_DATA_DIR: Path = Field(default_factory=lambda: Path(os.getenv("IP_DATA_DIR", "./data")))

    # ======================
    # Embedding 配置
    # ======================
    EMBEDDING_MODEL: str = Field(default="BAAI/bge-medical-v1.5", env="EMBEDDING_MODEL")
    EMBEDDING_DEVICE: str = Field(default="cuda", env="EMBEDDING_DEVICE")
    EMBEDDING_BATCH_SIZE: int = Field(default=32, env="EMBEDDING_BATCH_SIZE")

    # ======================
    # 向量数据库 (Chroma)
    # ======================
    CHROMA_COLLECTION: str = Field(default="patent_rag", env="CHROMA_COLLECTION")
    CHROMA_PERSIST_DIR: Path = Field(
        default=Path(r"D:\myf\IPRAG\artifacts\vectors\chroma") 
    )

    # ======================
    # 重排序器 (Reranker)
    # ======================
    RERANKER_MODEL: str = Field(default="BAAI/bge-reranker-v2-m3", env="RERANKER_MODEL")
    RERANKER_DEVICE: str = Field(default="cuda", env="RERANKER_DEVICE")
    RERANKER_TOP_K: int = Field(default=5, env="RERANKER_TOP_K")

    # ======================
    # LLM 配置
    # ======================
    LLM_TYPE: Literal["openai", "local"] = Field(default="openai", env="LLM_TYPE")

    # OpenAI 特有
    OPENAI_API_KEY: Optional[str] = Field(default=None, env="OPENAI_API_KEY")
    OPENAI_MODEL: str = Field(default="gpt-4o-mini", env="OPENAI_MODEL")
    OPENAI_MAX_TOKENS: int = Field(default=1024, env="OPENAI_MAX_TOKENS")
    OPENAI_TEMPERATURE: float = Field(default=0.0, env="OPENAI_TEMPERATURE")

    # 本地 LLM 特有
    LOCAL_LLM_MODEL: str = Field(default="Qwen/Qwen2-1.5B-Instruct", env="LOCAL_LLM_MODEL")
    LLM_DEVICE: str = Field(default="cuda", env="LLM_DEVICE")
    LLM_MAX_NEW_TOKENS: int = Field(default=512, env="LLM_MAX_NEW_TOKENS")
    LLM_TEMPERATURE: float = Field(default=0.1, env="LLM_TEMPERATURE")

    # ======================
    # RAG 超参数
    # ======================
    RETRIEVER_TOP_K: int = Field(default=10, env="RETRIEVER_TOP_K")  # 初检数量
    FINAL_ANSWER_TOP_K: int = Field(default=5, env="FINAL_ANSWER_TOP_K")  # 最终引用数

    # ======================
    # ✅ 新增：IP分块器专用参数
    # ======================
    # 注意：IPChunker 类定义中使用的是 description_chunk_size
    IP_CHUNK_SIZE: int = Field(default=1024, env="IP_CHUNK_SIZE")
    IP_CHUNK_OVERLAP: int = Field(default=128, env="IP_CHUNK_OVERLAP")

    # ======================
    # 验证逻辑
    # ======================
    @validator("OPENAI_API_KEY", always=True)
    def validate_openai_key(cls, v, values):
        if values.get("LLM_TYPE") == "openai" and not v:
            raise ValueError("OPENAI_API_KEY is required when LLM_TYPE=openai")
        return v

    @validator("CHROMA_PERSIST_DIR", always=True)
    def create_chroma_dir(cls, v):
        v.mkdir(parents=True, exist_ok=True)
        return v

    class Config:
        env_file = ".env"  # 自动加载 .env
        env_file_encoding = "utf-8"
        case_sensitive = False  # 环境变量不区分大小写（如 embedding_model 也有效）

# 单例实例
settings = Settings()