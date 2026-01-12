import asyncio

from dsagent_core.roles.ds_agent import DSAgent


async def main(requirement: str):
    role = DSAgent(use_reflection=False, tools=["<all>"], use_plan=True, use_rag=False, react_mode="plan_and_act")
    await role.run(requirement)


if __name__ == "__main__":
    requirement = "今天四川省绵阳市的天气怎么样？"
    asyncio.run(main(requirement))
