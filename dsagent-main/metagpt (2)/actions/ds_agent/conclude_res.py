from metagpt.actions import Action
from metagpt.prompts.ds_agent.conclusion_final_result import CONCLUSION_GOAL_RESULTS


class Conclusion(Action):
    async def run(self, final_goal: str, tasks_res: list[dict]):
        prompt = CONCLUSION_GOAL_RESULTS.format(final_goal=final_goal, tasks_res=tasks_res)
        final_ans = await self._aask(prompt)
        return final_ans

