# src/config/paths.py

from pathlib import Path
from .settings import settings

# 项目根目录
ROOT_DIR: Path = settings.BASE_DIR

# 数据目录
DATA_DIR: Path = settings.DATA_DIR
DRUGBANK_JSON_PATH: Path = DATA_DIR / "drugbank_small_molecule.json"
DRUGBANK_XML_PATH: Path = DATA_DIR / "drugbank.xml"

# 向量存储
VECTOR_STORE_DIR: Path = settings.ARTIFACTS_DIR / "vectors"
CHROMA_PERSIST_DIR: Path = settings.CHROMA_PERSIST_DIR

# 缓存目录
CACHE_DIR: Path = settings.ARTIFACTS_DIR / "cache"
EMBEDDING_CACHE_DIR: Path = CACHE_DIR / "embeddings"

# 日志目录
LOG_DIR: Path = settings.ARTIFACTS_DIR / "logs"

# 确保关键目录存在
for dir_path in [DATA_DIR, VECTOR_STORE_DIR, CACHE_DIR, LOG_DIR, CHROMA_PERSIST_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)