# src/api/app.py
import logging
from pathlib import Path
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

# 导入路由器
from src.api.routes.rag_routes import router as rag_router

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ✅ 修改：应用标题和描述
app = FastAPI(
    title="IP-RAG 专利智能检索系统", 
    description="基于海量专利数据的语义搜索与问答系统",
    version="1.0.0"
)

# 配置 CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 注册 API 路由
# ✅ 修改：API 路由前缀保持 /api，但 Tag 改为 Patent/IP
app.include_router(rag_router, prefix="/api", tags=["IP Search"]) 

# ✅ 关键修复：配置模板和静态文件目录
BASE_DIR = Path(__file__).resolve().parent
templates_path = BASE_DIR / "templates"
static_path = BASE_DIR / "static"

# 初始化 Jinja2 模板引擎
if templates_path.exists():
    templates = Jinja2Templates(directory=str(templates_path))
    logger.info(f"✅ 模板目录已加载：{templates_path}")
else:
    logger.error(f"❌ 模板目录不存在：{templates_path}")
    templates = None

# 挂载静态文件 (CSS, JS, images 等)
if static_path.exists():
    app.mount("/static", StaticFiles(directory=str(static_path)), name="static")
    logger.info(f"✅ 静态文件目录已挂载：{static_path}")
else:
    logger.warning(f"⚠️ 静态文件目录不存在：{static_path}")

# ✅ 关键修复：添加根路径路由，返回 index.html
@app.get("/")
async def read_root(request: Request):
    if templates is None:
        return {"error": "Template engine not initialized. Please check logs."}
    return templates.TemplateResponse("index.html", {"request": request})

# 健康检查接口
@app.get("/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)