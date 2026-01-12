import json

from metagpt.actions.ds_agent.extract_thought import ThoughtExtract
from metagpt.actions.ds_agent.fixed_plan_for_test import get_fixed_plan
from metagpt.config2 import Config
from metagpt.const import METAGPT_ROOT, EXAMPLE_DATA_PATH
from dsagent_core.roles.ds_agent import add_to_exp_bank

gpt4turbo_config_path = METAGPT_ROOT / "config" / "gpt-4-turbo.yaml"
gpt4t_config = Config.from_yaml_file(gpt4turbo_config_path)
EXP_PLAN = EXAMPLE_DATA_PATH / "exp_bank/plan_exp.json"


async def main():
    plan = get_fixed_plan(181)
    print(plan.goal)
    exp = await ThoughtExtract(config=gpt4t_config).extract_thought2(get_fixed_plan(181))
    await add_to_exp_bank(plan.goal, exp, EXP_PLAN)
    print(exp)


if __name__ == '__main__':
    import asyncio
    asyncio.run(main())
