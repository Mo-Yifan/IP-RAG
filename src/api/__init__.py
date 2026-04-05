# src/api/__init__.py

"""
DrugBank 临床问答系统的 FastAPI 应用入口模块。

该模块暴露 `app` 实例，供 ASGI 服务器（如 uvicorn）直接加载。
"""

from .app import app

__all__ = ["app"]