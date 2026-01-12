import traceback
from examples.ds_agent.agent_service.agent_service import DSAgent

try:
    # 测试不使用RAG的Agent初始化
    print("正在创建Agent(不使用RAG)...")
    from metagpt.roles.ds_agent.ds_agent_stream import DSAgentStream
    agent = DSAgentStream(use_reflection=True, use_rag=False, use_kaggle_exp=True, use_exp_extractor=False)
    print("✓ Agent创建成功（不使用RAG）!")
    
except Exception as e:
    print(f"✗ 错误: {e}")
    traceback.print_exc()
