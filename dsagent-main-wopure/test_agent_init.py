import traceback
from examples.ds_agent.agent_service.agent_service import DSAgent

try:
    agent = DSAgent('test')
    print("Agent created successfully!")
except Exception as e:
    print(f"Error: {e}")
    traceback.print_exc()
