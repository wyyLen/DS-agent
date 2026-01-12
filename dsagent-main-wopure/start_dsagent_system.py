"""
DSAgent 完整系统启动脚本
启动后端API服务（FastAPI）和前端UI服务（Gradio）
"""
import sys
import subprocess
import time
import os
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def start_backend():
    """启动后端FastAPI服务"""
    print("=" * 80)
    print("启动后端 API 服务 (FastAPI)...")
    print("=" * 80)
    
    backend_script = project_root / "examples" / "ds_agent" / "agent_service" / "start_backend.py"
    
    # 使用uvicorn启动FastAPI
    backend_cmd = [
        sys.executable,
        "-m",
        "uvicorn",
        "examples.ds_agent.agent_service.api_service_provider:app",
        "--host", "0.0.0.0",
        "--port", "8000",
        "--reload"
    ]
    
    # 设置环境变量 - 统一上传目录
    env = os.environ.copy()
    upload_dir = str(project_root / "DSassistant" / "uploads")
    env["UPLOAD_DIR"] = upload_dir
    env["INITIAL_AGENT_COUNT"] = "1"
    
    print(f"上传目录: {upload_dir}")
    print()
    
    backend_process = subprocess.Popen(
        backend_cmd,
        cwd=str(project_root),
        env=env
    )
    
    print(f"✓ 后端服务正在启动...")
    print(f"  URL: http://localhost:8000")
    print(f"  Docs: http://localhost:8000/docs")
    
    return backend_process


def start_frontend():
    """启动前端Gradio服务"""
    print("\n" + "=" * 80)
    print("启动前端 UI 服务 (Gradio)...")
    print("=" * 80)
    
    frontend_cmd = [
        sys.executable,
        "DSassistant/main.py"
    ]
    
    frontend_process = subprocess.Popen(
        frontend_cmd,
        cwd=str(project_root),
        env=os.environ.copy()
    )
    
    print(f"✓ 前端服务正在启动...")
    print(f"  URL: http://localhost:7860")
    
    return frontend_process


def main():
    print("""
╔════════════════════════════════════════════════════════════════════════════╗
║                                                                            ║
║                   DSAgent 完整系统启动                                      ║
║                   (基于 DSAgent Core 框架无关架构)                          ║
║                                                                            ║
╚════════════════════════════════════════════════════════════════════════════╝
    """)
    
    try:
        # 启动后端
        backend_process = start_backend()
        
        # 等待后端启动
        print("\n等待后端服务就绪...")
        time.sleep(5)
        
        # 启动前端
        frontend_process = start_frontend()
        
        # 等待前端启动
        print("\n等待前端服务就绪...")
        time.sleep(3)
        
        print("\n" + "=" * 80)
        print("✓ 系统启动完成！")
        print("=" * 80)
        print("""
访问地址:
  - 前端界面: http://localhost:7860
  - 后端API:  http://localhost:8000
  - API文档:  http://localhost:8000/docs

特性:
  ✓ 文本经验检索 (240条经验，BM25算法)
  ✓ 工作流经验检索 (577个工作流，图匹配)  
  ✓ 树搜索自主探索 (MCTS算法)
  ✓ 实时代码执行与反馈
  ✓ 文件上传与会话管理

按 Ctrl+C 停止服务...
        """)
        
        # 等待用户中断
        try:
            backend_process.wait()
            frontend_process.wait()
        except KeyboardInterrupt:
            print("\n正在停止服务...")
            backend_process.terminate()
            frontend_process.terminate()
            backend_process.wait()
            frontend_process.wait()
            print("服务已停止。")
            
    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
