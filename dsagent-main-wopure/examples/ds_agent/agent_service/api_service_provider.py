import logging
import os
import sys
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path as PathLib

# 添加项目根目录到Python路径
project_root = PathLib(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from fastapi import FastAPI, Request, HTTPException, UploadFile, File, Form, Path, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, FileResponse

from examples.ds_agent.agent_service.agent_service import AgentServiceProvider
from examples.ds_agent.agent_service.translate_service import translate_cn_text
from examples.ds_agent.agent_service.file_service import FileService

# 配置日志
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

# 全局service对象
service = None

# ------------------ FastAPI生命周期管理 ------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    global service
    try:
        print("=" * 60)
        print("Service starting...")
        print("=" * 60)
        logger.info("=" * 60)
        logger.info("Service starting...")
        logger.info("=" * 60)
        INITIAL_AGENT_COUNT = int(os.getenv("INITIAL_AGENT_COUNT", 1))
        print(f"Initializing AgentServiceProvider with {INITIAL_AGENT_COUNT} agents per mode...")
        logger.info(f"Initializing AgentServiceProvider with {INITIAL_AGENT_COUNT} agents per mode...")
        service = AgentServiceProvider(initial_agent_counts={"ds": INITIAL_AGENT_COUNT, "lats": INITIAL_AGENT_COUNT})
        print("=" * 60)
        print("✅ Service initialized successfully!")
        print("=" * 60)
        logger.info("=" * 60)
        logger.info("✅ Service initialized successfully!")
        logger.info("=" * 60)
    except Exception as e:
        print(f"❌ Failed to initialize service: {e}")
        logger.error(f"❌ Failed to initialize service: {e}")
        import traceback
        traceback.print_exc()
        raise
    
    yield
    
    try:
        print("Cleaning up agents...")
        logger.info("Cleaning up agents...")
        if service:
            for mode, agents in service.agents_pool.items():
                for agent_id, agent in agents.items():
                    agent.release()
        print("Cleanup complete")
        logger.info("Cleanup complete")
    except Exception as e:
        print(f"Error during cleanup: {e}")
        logger.error(f"Error during cleanup: {e}")


app = FastAPI(
    title="DSAgent API Service",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# 使用与前端统一的上传目录
UPLOAD_DIR = os.getenv("UPLOAD_DIR", str(project_root / "DSassistant" / "uploads"))
file_service = FileService(upload_dir=UPLOAD_DIR)

logger.info(f"上传目录设置为: {UPLOAD_DIR}")


# ------------------ 接口端点 ------------------
@app.post("/v1/dsagent/stream")
async def chat_stream(request: Request):
    try:
        data = await request.json()
        
        # 处理query可能是列表或字符串的情况
        query_raw = data.get("query", "")
        if isinstance(query_raw, list):
            query = " ".join(str(item) for item in query_raw).strip()
        else:
            query = str(query_raw).strip()
        
        agent_mode = data.get("agent_mode", "").strip()
        session_id = data.get("session_id", "").strip()
        if not query:
            raise HTTPException(400, detail="Query parameter required")

        query = translate_cn_text(query)

        agent = await service.get_idle_agent(mode=agent_mode)
        logger.info(f"Agent {agent.agent_id} handling: {query[:50]}...")

        return StreamingResponse(
            service.stream_generator(agent, query, session_id),
            media_type="text/event-stream",
            headers={
                "X-Agent-ID": agent.agent_id,
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Event-Stream-Version": "2.0"
            }
        )
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.exception("Unexpected error")
        raise HTTPException(500, detail="Internal server error")


@app.post("/v1/files/upload")
async def upload_file(file: UploadFile = File(...), session_id: str = Form(...)):
    """
    上传文件API
    """
    try:
        if not file:
            raise HTTPException(status_code=400, detail="No file provided")
        if not session_id:
            raise HTTPException(status_code=400, detail="Session ID is required")
        file_info = await file_service.save_file(file, session_id)
        return {
            "file_id": file_info.file_id,
            "file_name": file_info.file_name,
            "file_size": file_info.file_size,
            "mime_type": file_info.mime_type,
            "upload_time": file_info.upload_time
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("File upload error")
        raise HTTPException(status_code=500, detail=f"File upload failed: {str(e)}")


@app.get("/v1/files")
async def get_session_files(session_id: str = Query(..., description="会话ID")):
    """
    获取会话的文件列表
    """
    try:
        if not session_id:
            raise HTTPException(status_code=400, detail="Session ID is required")

        files = file_service.get_session_files(session_id)
        return {"files": files}
    except Exception as e:
        logger.exception("Get files error")
        raise HTTPException(status_code=500, detail=f"Failed to get files: {str(e)}")


@app.get("/v1/files/{file_id}")
async def download_file(file_id: str = Path(..., description="文件ID"), session_id: str = Query(..., description="会话ID")):
    """
    下载文件
    """
    try:
        file_path = file_service.get_file_path(file_id, session_id)
        if not file_path:
            raise HTTPException(status_code=404, detail="File not found or access denied")

        file_info = file_service.get_file_info(file_id)
        return FileResponse(
            path=file_path,
            filename=file_info.file_name,
            media_type=file_info.mime_type
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("File download error")
        raise HTTPException(status_code=500, detail=f"Failed to download file: {str(e)}")


@app.delete("/v1/files/{file_id}")
async def delete_file(file_id: str = Path(..., description="文件ID"), session_id: str = Query(..., description="会话ID")):
    """
    删除文件
    """
    try:
        success = file_service.delete_file(file_id, session_id)
        if not success:
            raise HTTPException(status_code=404, detail="File not found or access denied")

        return {"success": True, "message": "File deleted successfully"}
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("File deletion error")
        raise HTTPException(status_code=500, detail=f"Failed to delete file: {str(e)}")


@app.delete("/v1/files")
async def cleanup_session_files(session_id: str = Query(..., description="会话ID")):
    """
    清理会话的所有文件
    """
    try:
        deleted_count = file_service.cleanup_session(session_id)
        return {
            "success": True,
            "message": f"Cleaned up {deleted_count} files from session",
            "deleted_count": deleted_count
        }
    except Exception as e:
        logger.exception("Session cleanup error")
        raise HTTPException(status_code=500, detail=f"Failed to cleanup session: {str(e)}")


@app.get("/health")
async def health_check():
    return {
        "status": "ok",
        "timestamp": datetime.now().isoformat(),
        "agents": {
            "total": len(service.agents_pool),
            "active": sum(1 for a in service.agents_pool.values() if a._active)
        }
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        timeout_keep_alive=300
    )