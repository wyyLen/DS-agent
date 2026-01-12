import sys
sys.path.insert(0, 'e:\\dsagent-main')

from metagpt.actions.ds_agent.write_ds_plan import update_plan_from_rsp
from metagpt.schema import Plan
import json
import traceback

test_json = '[{"task_id": "1", "dependent_task_ids": [], "instruction": "test", "task_type": "pda"}]'

try:
    plan = Plan(goal='test')
    update_plan_from_rsp(test_json, plan)
    print("✓ 测试成功")
    print(f"Plan tasks: {plan.tasks}")
except Exception as e:
    print(f"❌ 测试失败: {e}")
    traceback.print_exc()
