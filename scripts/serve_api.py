#!/usr/bin/env python3
# scripts/serve_api.py

"""
启动 IP-RAG 专利检索 API 服务
============================
提供：
- Web 用户界面 (http://localhost:8000)
- RESTful API (http://localhost:8000/api/query)
- 自动 Swagger 文档 (http://localhost:8000/docs)
"""

import os
import sys
import logging
from pathlib import Path
import argparse

# 将项目根目录加入 Python 路径
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src.api.app import app

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="启动 IP-RAG 专利检索 API 服务")
    
    # ✅ 保持变量名不变 (host, port, reload)
    parser.add_argument("--host", type=str, default="0.0.0.0", help="服务绑定地址")
    parser.add_argument("--port", type=int, default=8000, help="服务端口")
    parser.add_argument("--reload", action="store_true", help="开发模式：自动重载")
    
    args = parser.parse_args()
    
    # ✅ 修改打印信息：从 DrugBank 改为 IP-RAG
    logger.info(f"🚀 启动 IP-RAG 专利检索 API 服务")
    logger.info(f" 主机: {args.host}")
    logger.info(f" 端口: {args.port}")
    logger.info(f" 自动重载: {'是' if args.reload else '否'}")
    logger.info(f" Web UI: http://localhost:{args.port}")
    logger.info(f" API 文档: http://localhost:{args.port}/docs")

    try:
        import uvicorn
    except ImportError:
        logger.error("❌ 未安装 uvicorn，请运行: pip install uvicorn")
        sys.exit(1)

    # ✅ 保持 Uvicorn 运行参数不变，只改了日志里的名字
    uvicorn.run(
        "src.api.app:app", # 保持导入路径不变
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="info"
    )

if __name__ == "__main__":
    main()