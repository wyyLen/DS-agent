from metagpt.roles.ds_agent.ds_agent import DSAgent

ds = DSAgent(use_reflection=True, use_rag=False)
ds.planner.set_fixed_plan()

print(ds.planner.plan)
ds.planner.clear_plan()

print(ds.planner.plan)
