from __future__ import annotations

import json
import re
from copy import deepcopy
from typing import Tuple

from dsagent_core.actions.ds_agent.fixed_plan_for_test import FixedPlan
from metagpt.config2 import Config
from metagpt.actions import Action
from metagpt.logs import logger
from dsagent_core.prompts.ds_agent.extract_thinking import (
    EXTRACT_SEQ_FROM_PLAN, EXTRACT_DETAIL_FROM_PLAN, EXTRACT_FROM_PLAN_TEMPLATE,
    COMBINED_STRUCTURED_PLAN_PROMPT_TEMPLATE, EXTRACT_FROM_CODE_TEMPLATE, EXTRACT_WORKFLOW_FROM_EXP_TEMPLATE,
    workflow_example, EXTRACT_WORKFLOW_FROM_PLAN_TEMPLATE, EXTRACT_WORKFLOW_FROM_PLAN_TEMPLATE_COT,
    EXTRACT_WORKFLOW_FROM_EXP_TEMPLATE_COT, EXTRACT_WORKFLOW_FROM_CODE_TEMPLATE, EXTRACT_GOAL_FROM_COMP_DESC,
    problem_example, EXTRACT_EXP_FROM_WORKING_MEMORY_TEMPLATE, EXTRACT_WORKFLOW_FROM_CODE_IMPROVEMENT
)
from metagpt.schema import Message, Plan, Task
from dsagent_core.strategy.ds_task_type import TaskType
from metagpt.utils.common import CodeParser


class ThoughtExtract(Action):
    async def _extract_seq_from_plan(self, plan: Plan) -> str:
        print("---------------------------start extracting sequential characteristics from plan-----------------------")
        print("cur plan:\n", plan.tasks)
        task_list = [
            {"task_id": task.task_id, "dependent_task_ids": task.dependent_task_ids, "task_type": task.task_type,
             "instruction": task.instruction} for task in plan.tasks]
        # task_list = FixedPlan(181).tasks
        print("task_list:\n", task_list)
        prompt = EXTRACT_SEQ_FROM_PLAN.format(plan=json.dumps(task_list))
        rsp = await self._aask(prompt)
        seq = CodeParser.parse_code(block=None, text=rsp)
        return seq

    async def _extract_detail_from_plan(self, plan: Plan) -> str:
        print("----------------------------start extracting detailed characteristics from plan-----------------------")
        task_list = [{"task_id": task.task_id, "dependent_task_ids": task.dependent_task_ids,
                      "task_type": task.task_type, "instruction": task.instruction} for task in plan.tasks]
        task_type_desc = "\n".join([f"- **{tt.type_name}**: {tt.value.desc}" for tt in TaskType])
        prompt = EXTRACT_DETAIL_FROM_PLAN.format(plan=json.dumps(task_list), task_type_desc=task_type_desc)
        rsp = await self._aask(prompt)
        detail = CodeParser.parse_code(block=None, text=rsp)
        return detail

    async def extract_structured_thought_from_plan(self, plan: Plan):
        """
        func 两阶段提取结构化思维，然后进行归纳
            阶段1：从plan中提取任务顺序特征
            阶段2：从plan中提取任务细节特征
        """
        seq = await self._extract_seq_from_plan(plan)
        detail = await self._extract_detail_from_plan(plan)
        prompt = COMBINED_STRUCTURED_PLAN_PROMPT_TEMPLATE.format(seq=seq, detail=detail)
        rsp = await self._aask(prompt)
        expr = CodeParser.parse_code(block=None, text=rsp)
        return expr

    async def extract_thought_from_plan(self, plan: Plan):
        """
        func 直接从完整的计划中，根据提示归纳计划思路
        """
        task_list = [{"task_id": task.task_id, "dependent_task_ids": task.dependent_task_ids,
                      "task_type": task.task_type, "instruction": task.instruction} for task in plan.tasks]
        prompt = EXTRACT_FROM_PLAN_TEMPLATE.format(plan=json.dumps(task_list), question=plan.goal)
        rsp = await self._aask(prompt)
        return rsp

    async def extract_thought_from_code(self, goal: str, code: str):
        """
        func 从完整的代码解决方案中，根据提示归纳解决思路
        """
        task_type_desc = "\n".join([f"- **{tt.type_name}**: {tt.value.desc}" for tt in TaskType])
        prompt = EXTRACT_FROM_CODE_TEMPLATE.format(question=goal, code=code, task_type_desc=task_type_desc)
        rsp = await self._aask(prompt)
        return rsp

    async def extract_workflow_from_code(self, goal: str, code: str):
        """
        func 从完整的代码解决方案中，根据提示归纳工作流
        """
        task_type_desc = "\n".join([f"- **{tt.type_name}**: {tt.value.desc}" for tt in TaskType])
        prompt = EXTRACT_WORKFLOW_FROM_CODE_IMPROVEMENT.format(question=goal, code=code, task_type_desc=task_type_desc)
        rsp = await self._aask(prompt)
        rsp = CodeParser.parse_code(block=None, text=rsp)
        return rsp

    async def extract_workflow_from_exp(self, exp: str):
        """
        func 从现有的经验知识中，归纳工作流范式 ( not used yet )
        """
        task_type_desc = "\n".join([f"- **{tt.type_name}**: {tt.value.desc}" for tt in TaskType])
        prompt = EXTRACT_WORKFLOW_FROM_EXP_TEMPLATE_COT.format(
            exp=exp, task_type_desc=task_type_desc, workflow_example=workflow_example)
        rsp = await self._aask(prompt)
        workflow_exp = CodeParser.parse_code(block=None, text=rsp)
        return workflow_exp

    async def extract_workflow_from_plan(self, plan: Plan):
        """
        func 从完成任务的计划中，归纳工作流范式
        """
        task_list = [{"task_id": task.task_id, "dependent_task_ids": task.dependent_task_ids,
                      "task_type": task.task_type, "instruction": task.instruction} for task in plan.tasks]
        prompt = EXTRACT_WORKFLOW_FROM_PLAN_TEMPLATE_COT.format(
            plan=json.dumps(task_list), question=plan.goal, workflow_example=workflow_example)
        rsp = await self._aask(prompt)
        workflow_exp = CodeParser.parse_code(block=None, text=rsp)
        return workflow_exp

    async def extract_goal_from_competition_description(self, description: str, files: list):
        prompt = EXTRACT_GOAL_FROM_COMP_DESC.format(desc=description, problem_example=problem_example)
        rsp = await self._aask(prompt)
        goal = CodeParser.parse_code(block=None, text=rsp)
        if files is not None:
            goal = re.sub(r'File:.*', f'File: {files}', goal)
        return goal

    async def extract_goal_from_working_memory(self, goal: str, working_memory: list[Message]):
        # func 从解决问题的完整过程中，总结解决问题的经验知识，包括问题解决方法、注意事项等，保证其他agent能够根据你提供的经验知识准确生成计划。
        format_working_memory = [{"role": message.role, "content": message.content} for message in working_memory]
        prompt = EXTRACT_EXP_FROM_WORKING_MEMORY_TEMPLATE.format(question=goal, working_memory=format_working_memory)
        rsp = await self._aask(prompt)
        return rsp
