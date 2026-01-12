from metagpt.actions import Action
from metagpt.logs import logger
from dsagent_core.prompts.ds_agent.query import GET_QUESTION_TYPE_PROMPT, GET_QA_TYPE_PROMPT_TEMPLATE
from dsagent_core.strategy.ds_task_type import TaskType
from metagpt.utils.common import CodeParser


class QueryUtils(Action):
    async def getQuestionType(self, question: str) -> str:
        task_type_desc = "\n".join([f"- **{tt.type_name}**: {tt.value.desc}" for tt in TaskType])
        prompt = GET_QUESTION_TYPE_PROMPT.format(question=question, task_type_desc=task_type_desc)
        rsp = await self._aask(prompt)
        rsp = CodeParser.parse_code(block=None, text=rsp)
        rsp2list = rsp[1:-1].split(", ")
        if len(rsp2list) == 0:
            logger.exception(f"Current question does not belong to any known task type ")
        return rsp

    async def getQAType(self, question: str, solution: str) -> str:
        task_type_desc = "\n".join([f"- **{tt.type_name}**: {tt.value.desc}" for tt in TaskType])
        prompt = GET_QA_TYPE_PROMPT_TEMPLATE.format(question=question, solution=solution, task_type_desc=task_type_desc)
        rsp = await self._aask(prompt)
        rsp = CodeParser.parse_code(block=None, text=rsp)
        rsp2list = rsp[1:-1].split(", ")
        if len(rsp2list) == 0:
            logger.exception(f"Current question does not belong to any known task type ")
        return rsp
