import asyncio

from metagpt.actions.ds_agent.fixed_plan_for_test import get_fixed_plan
from metagpt.actions.ds_agent.write_ds_plan import RefinePlan


async def tey_refine_task_type():
    plan = get_fixed_plan(181)
    await RefinePlan().refine_ds_scenarios_in_plan(plan)
    print("refined result for test")
    for task in plan.tasks:
        print(task.task_type + " ")


if __name__ == '__main__':
    asyncio.run(tey_refine_task_type())
