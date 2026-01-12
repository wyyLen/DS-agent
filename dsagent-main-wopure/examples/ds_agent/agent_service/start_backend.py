"""
后端API服务启动入口
"""
import sys
import os
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

# 现在导入FastAPI应用
from examples.ds_agent.agent_service import api_service_provider

# 获取app对象
app = api_service_provider.app

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)
