#  IPRAG: 智能专利检索增强生成系统

一个基于 **Qwen3-Embedding** 和 **ChromaDB** 的高性能专利数据分析与检索系统。本项目旨在处理海量专利文本（JSON格式），通过高效的向量化流水线，实现对专利技术细节的精准问答与洞察。

---

###  核心特性

- **海量数据处理**: 专为处理 10万+ 级专利文本块设计，支持断点续传与内存优化。
- **极速向量化**: 采用 PyTorch `DataLoader` 批处理机制，相比单条处理提速 10-20 倍。
- **智能容错**: 内置异常捕获与噪声填充机制，确保长尾数据（如超长文本、特殊字符）不导致程序崩溃。
- **精准溯源**: 检索结果自动关联专利号（如 `US-20250244724-A1`），确保回答有据可依。
- **现代化技术栈**:
    - **Embedding**: Qwen3-Embedding (Alibaba)
    - **VectorDB**: ChromaDB
    - **Processing**: PyTorch, Python 3.10+

---

### ️ 环境准备

在开始之前，请确保你的环境满足以下要求：

- **Python**: 3.9 或更高版本
- **GPU**: 推荐 NVIDIA GPU (用于加速 Embedding 生成)
- **依赖库**: 见 `requirements.txt`

#### 安装依赖

```bash
pip install -r requirements.txt
```

---

### ️ 配置说明

在运行注入脚本前，请检查 `settings.py`（或相应的配置文件）：

| 配置项 | 说明 | 推荐值 |
| :--- | :--- | :--- |
| `IP_DATA_DIR` | 专利 JSON 数据所在文件夹路径 | `data/patents` |
| `EMBEDDING_MODEL` | 使用的 Embedding 模型路径或名称 | `Alibaba-NLP/gte-Qwen3-Embedding` |
| `CHROMA_PERSIST_DIR` | 向量库持久化存储路径 | `./chroma_db` |
| `EMBEDDING_BATCH_SIZE` | 批处理大小 (根据显存调整) | `32` 或 `64` |

---

### ‍♂️ 快速开始

#### 1. 数据注入

运行 `ingest.py` 将专利数据加载、切分并转化为向量存入数据库。

```bash
python scripts/ingest.py
```

**运行日志示例：**
```text
 开始 专利 (IP) 数据注入流程...
 选中文件夹: data/patents
 正在处理: patents_10.json
 正在为 1170000 个块生成嵌入 (批处理加速模式)...
生成向量: 100%|██████████| 36563/36563 [05:23<00:00, 113.01it/s]
 正在存入向量库...
 文件 patents_10.json 处理完成，累计: 1170000
 所有文件注入完成！
```

#### 2. 启动 API 服务

(假设你有一个 `main.py` 或 `api.py` 来启动 FastAPI/Flask)

```bash
python main.py
```

#### 3. 提问测试

向 API 发送请求：

> **问题**: "自动驾驶汽车的激光雷达传感器系统是如何工作的？"

**预期回答**: 系统会检索相关专利（如 `US-12360222-B2`），总结 ToF 原理及点云生成技术，并给出引用来源。

---

###  项目结构

```text
D:.
│  .env     
│  README.md        
│  requirements.txt 
│
├─artifacts
│  ├─cache
│  ├─logs
│  └─vectors
│      ├─chroma
│      └─faiss
├─data
│
├─models
│  
├─scripts
│      ingest.py
│      serve_api.py
│
├─src
│  │  __init__.py
│  │
│  ├─api
│  │  │  app.py
│  │  │  dependencies.py
│  │  │  __init__.py
│  │  │
│  │  ├─routes
│  │  │      rag_routes.py
│  │  │      __init__.py
│  │  │
│  │  ├─static
│  │  │  ├─css
│  │  │  │      style.css
│  │  │  │
│  │  │  └─js
│  │  │          main.js
│  │  │
│  │  └─templates
│  │          base.html
│  │          index.html
│  │
│  ├─config
│  │      paths.py
│  │      settings.py
│  │      __init__.py
│  │
│  ├─data
│  │  │  schemas.py
│  │  │  __init__.py
│  │  │
│  │  ├─chunkers
│  │  │      base_chunker.py
│  │  │      IP_chunker.py
│  │  │      semantic_chunker.py
│  │  │      __init__.py
│  │  │
│  │  └─loaders
│  │          base_loader.py
│  │          IP_json_loader.py
│  │          __init__.py
│  │
│  ├─generation
│  │  │  rag_chain.py
│  │  │  __init__.py
│  │  │
│  │  ├─llm_clients
│  │  │      base_client.py
│  │  │      local_llm_client.py
│  │  │      openai_client.py
│  │  │      __init__.py
│  │  │
│  │  └─prompts
│  │          base_prompt.py
│  │          citation_utils.py
│  │          IP_qa_prompt.txt
│  │          patent_prompt.py
│  │          __init__.py
│  │
│  └─retrieval
│      │  schemas.py
│      │  __init__.py
│      │
│      ├─embedders
│      │      base_embedder.py
│      │      huggingface_embedder.py
│      │      openai_embedder.py
│      │      qwen_embedder.py
│      │      __init__.py
│      │
│      ├─parsers
│      │      base_parser.py
│      │      drugbank_json.py
│      │      drugbank_xml.py
│      │      pubmed_parser.py
│      │      utils.py
│      │      __init__.py
│      │
│      ├─rerankers
│      │      base_reranker.py
│      │      bge_reranker.py
│      │      none_reranker.py
│      │      qwen_reranker.py
│      │      __init__.py
│      │
│      ├─retrievers
│      │      base_retriever.py
│      │      multi_hop_retriever.py
│      │      __init__.py
│      │
│      └─vectorstores
│              base_vectorstore.py
│              chroma_store.py
│              faiss_store.py
│              __init__.py

```

---

###  技术亮点：如何处理 100万+ 数据块？

本项目在 `ingest.py` 中解决了传统 RAG 系统处理大规模数据时的痛点：

1. **变量隔离**: 严格区分 `IPJsonLoader` (文件读取) 与 PyTorch `DataLoader` (批量计算)，避免命名空间污染。
2. **显存保护**:
    - 使用 `torch.no_grad()` 关闭梯度计算。
    - 动态截断 (`max_len=512`) 防止超长专利文本撑爆显存。
3. **鲁棒性设计**:
    - 遇到解析失败的文本块，自动注入随机噪声向量，保证索引对齐，防止整个流水线中断。

---

###  许可证

MIT License

---

###  贡献

欢迎提交 Issue 或 Pull Request 来优化专利切分算法或检索精度！
你觉得这份 README 的技术亮点总结到位吗？特别是关于“海量数据处理”的部分，是否符合你的预期？
