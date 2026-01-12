"""
测试后端初始化问题
"""
import sys
import os
import traceback
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# 设置环境变量
os.environ['AGENT_FRAMEWORK'] = 'metagpt'

print("=" * 60)
print("Step 1: Testing imports...")
print("=" * 60)

try:
    from examples.ds_agent.agent_service import api_service_provider
    print("✅ Successfully imported api_service_provider")
except Exception as e:
    print(f"❌ Failed to import api_service_provider: {e}")
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 60)
print("Step 2: Testing app creation...")
print("=" * 60)

try:
    app = api_service_provider.app
    print(f"✅ App created successfully: {app}")
except Exception as e:
    print(f"❌ Failed to create app: {e}")
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 60)
print("Step 3: Testing service initialization...")
print("=" * 60)

try:
    service = api_service_provider.service
    print(f"✅ Service initialized successfully")
    print(f"   Framework: {service.framework}")
    print(f"   Agent pools: {list(service.agents_pool.keys())}")
    for mode, agents in service.agents_pool.items():
        print(f"   - {mode}: {len(agents)} agents")
except Exception as e:
    print(f"❌ Failed to initialize service: {e}")
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 60)
print("Step 4: Testing uvicorn server...")
print("=" * 60)

try:
    import uvicorn
    print("✅ Starting uvicorn server...")
    print("   Server will start on http://0.0.0.0:8000")
    print("   Press CTRL+C to stop")
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)
except KeyboardInterrupt:
    print("\n✅ Server stopped by user")
except Exception as e:
    print(f"❌ Failed to start server: {e}")
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 60)
print("All tests passed!")
print("=" * 60)
