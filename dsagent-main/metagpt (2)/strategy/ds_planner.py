from __future__ import annotations

import json
import os
import re

import yaml
from openai import OpenAI
from pydantic import BaseModel, Field

from dsagent_core.actions.ds_agent.ask_review import AskReview, ReviewConst
from dsagent_core.actions.ds_agent.fixed_plan_for_test import get_fixed_plan
from dsagent_core.actions.ds_agent.retrieval_exp import RetrievalExp
from dsagent_core.actions.ds_agent.write_ds_plan import (
    WritePlan,
    precheck_update_plan_from_rsp,
    update_plan_from_rsp,
)
from metagpt.config2 import Config
from metagpt.configs.llm_config import LLMConfig
from metagpt.const import METAGPT_ROOT, EXAMPLE_DATA_PATH
from metagpt.logs import logger
from metagpt.memory import Memory
from metagpt.rag.engines import CustomEngine
from metagpt.rag.schema import FAISSRetrieverConfig
from metagpt.schema import Message, Plan, Task, TaskResult
from metagpt.strategy.ds_task_type import TaskType
from metagpt.utils.common import remove_comments


EXP_PLAN = EXAMPLE_DATA_PATH / "exp_bank/plan_exp.json"

STRUCTURAL_CONTEXT = """
## User Requirement
{user_requirement}
## Context
{context}
## Current Plan
{tasks}
## Current Task
{current_task}
"""

PLAN_STATUS = """
## Finished Tasks
### code
```python
{code_written}
```

### execution result
{task_results}

## Current Task
{current_task}

## Task Guidance
Write complete code for 'Current Task'. And avoid duplicating code from 'Finished Tasks', such as repeated import of packages, reading data, etc.
Specifically, {guidance}
"""

PROMPT_KEYWORDS = """
# Text
{text}

# Available Keywords
{keywords}

Based on the above user messages, summarize and extract the relevant data science task types.
Output a list of jsons following the format:
```json
[
    {{
        "keyword": str = "keyword extracted from text, should be one of Available Keywords",
    }},
    ...
]
```
# Suffix:
- Keep in mind that Your response MUST follow the valid format above.
"""

PROMPT_KEYWORDS_PUBLIC = (
    "A question is provided below. Given the question, extract up to {max_keywords} "
    "keywords from the text. Focus on extracting the keywords that we can use "
    "to best lookup answers to the question. Avoid stopwords.\n"
    "Available Keywords: {keywords}\n"
    "---------------------\n"
    "{text}\n"
    "---------------------\n"
    "Provide keywords in the following comma-separated format: 'KEYWORDS: <keywords>'\n"
    "keyword extracted from text, should be one of Available Keywords\n"
)


async def extract_keywords(text) -> list[str]:
    config_path = os.path.join("D:\\Dev\\DSAgent", "config/config2.yaml")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"config not found in {config_path}.")
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    client = OpenAI(api_key=config['llm']['api_key'])
    task_type_desc = "\n".join([f"- **{tt.type_name}**: {tt.value.desc}" for tt in TaskType])

    # note： 自定义keyword格式， list[str]
    rsp = client.chat.completions.create(
        model="gpt-3.5-turbo-0125",
        messages=[{"role": "user", "content": PROMPT_KEYWORDS.format(text=text, keywords=task_type_desc)}],
    )
    print(rsp.choices[0].message.content)
    pattern = rf"```json.*?\s+(.*?)```"
    match = re.search(pattern, rsp.choices[0].message.content, re.DOTALL)
    keywords = [tt["keyword"] for tt in json.loads(match.group(1))]
    return keywords

    # note： llamaindex中 示例的格式 KEYWORDS: correlation analysis, feature engineering, machine learning
    # rsp = client.chat.completions.create(
    #     model="gpt-3.5-turbo-0125",
    #     messages=[{"role": "user", "content": PROMPT_KEYWORDS_PUBLIC.format(text=text, keywords=task_type_desc, max_keywords=10)}],
    # )
    # print(rsp.choices[0].message.content)
    # return None


def repair_json(json_str: str) -> str:
    # desc:将生成的json中不合理的单引号修正为双引号（针对task_id和dependent_task_ids）
    data = re.sub(r'"task_id": \'(\d+)\'', r'"task_id": "\1"', json_str)
    def replace_dependent_task_ids(match):
        content = match.group(0).replace("'", '"')
        return content
    data = re.sub(r'"dependent_task_ids": \[(\'\d+\'(,\s*\'\d+\')*)?\]', replace_dependent_task_ids, data)
    return data


class Planner(BaseModel):
    plan: Plan
    working_memory: Memory = Field(
        default_factory=Memory
    )  # memory for working on each task, discarded each time a task is done
    auto_run: bool = False

    def __init__(self, goal: str = "", plan: Plan = None, **kwargs):
        plan = plan or Plan(goal=goal)
        super().__init__(plan=plan, **kwargs)

    @property
    def current_task(self):
        return self.plan.current_task

    @property
    def current_task_id(self):
        return self.plan.current_task_id

    async def update_plan(self, goal: str = "", max_tasks: int = 5, max_retries: int = 3):
        if goal:
            self.plan = Plan(goal=goal)

        plan_confirmed = False
        while not plan_confirmed:
            context = self.get_useful_memories()
            rsp = await WritePlan().run(context, max_tasks=max_tasks)
            rsp = repair_json(rsp)
            self.working_memory.add(Message(content=rsp, role="assistant", cause_by=WritePlan))

            # precheck plan before asking reviews
            is_plan_valid, error = precheck_update_plan_from_rsp(rsp, self.plan)
            if not is_plan_valid and max_retries > 0:
                error_msg = f"The generated plan is not valid with error: {error}, try regenerating, remember to generate either the whole plan or the single changed task only"
                logger.warning(error_msg)
                logger.warning(f"Debug JSON: {rsp}")
                self.working_memory.add(Message(content=error_msg, role="assistant", cause_by=WritePlan))
                max_retries -= 1
                continue

            # 当设置自动运行的时候，这里会直接返回true。这里写成函数可以为后续用户交互或者LLM检查提供入口。
            _, plan_confirmed = await self.ask_review(trigger=ReviewConst.TASK_REVIEW_TRIGGER)

        try:
            update_plan_from_rsp(rsp=rsp, current_plan=self.plan)
        except Exception as e:
            logger.error(f"Failed to update plan from response")
            logger.error(f"Response: {rsp}")
            import traceback
            logger.error(traceback.format_exc())
            raise e

        self.working_memory.clear()

    async def process_task_result(self, task_result: TaskResult):
        # ask for acceptance, users can other refuse and change tasks in the plan
        review, task_result_confirmed = await self.ask_review(task_result)

        if task_result_confirmed:
            # tick off this task and record progress
            await self.confirm_task(self.current_task, task_result, review)

        elif "redo" in review:
            # Ask the Role to redo this task with help of review feedback,
            # useful when the code run is successful but the procedure or result is not what we want
            pass  # simply pass, not confirming the result

        else:
            # update plan according to user's feedback and to take on changed tasks
            await self.update_plan()

    async def ask_review(self, task_result: TaskResult = None, auto_run: bool = None,
                         trigger: str = ReviewConst.TASK_REVIEW_TRIGGER, review_context_len: int = 5):
        """
        Ask to review the task result, reviewer needs to provide confirmation or request change.
        If human confirms the task result, then we deem the task completed, regardless of whether the code run succeeds;
        if auto mode, then the code run has to succeed for the task to be considered completed.
        """
        # note: useful when set "auto_run=False"
        # 当设置自动运行的时候，这里会直接返回true。这里写成函数可以为后续用户交互或者LLM检查提供入口。
        auto_run = auto_run or self.auto_run
        if not auto_run:
            context = self.get_useful_memories()
            review, confirmed = await AskReview().run(
                context=context[-review_context_len:], plan=self.plan, trigger=trigger
            )
            if not confirmed:
                self.working_memory.add(Message(content=review, role="user", cause_by=AskReview))
            return review, confirmed
        confirmed = task_result.is_success if task_result else True
        return "", confirmed

    async def confirm_task(self, task: Task, task_result: TaskResult, review: str):
        task.update_task_result(task_result=task_result)
        self.plan.finish_current_task()
        self.working_memory.clear()

        # note: useful when set "auto_run=False"
        #  基于review的内容。若除了confirm之外还有其他更新计划的描述，再次调用update_plan方法
        confirmed_and_more = (
                ReviewConst.CONTINUE_WORDS[0] in review.lower() and review.lower() not in ReviewConst.CONTINUE_WORDS[0]
        )  # "confirm, ... (more content, such as changing downstream tasks)"
        if confirmed_and_more:
            self.working_memory.add(Message(content=review, role="user", cause_by=AskReview))
            await self.update_plan()

    def get_useful_memories(self, task_exclude_field=None) -> list[Message]:
        """find useful memories only to reduce context length and improve performance"""
        user_requirement = self.plan.goal
        context = self.plan.context
        tasks = [task.dict(exclude=task_exclude_field) for task in self.plan.tasks]
        tasks = json.dumps(tasks, indent=4, ensure_ascii=False)
        current_task = self.plan.current_task.json() if self.plan.current_task else {}
        context = STRUCTURAL_CONTEXT.format(
            user_requirement=user_requirement, context=context, tasks=tasks, current_task=current_task
        )
        context_msg = [Message(content=context, role="user")]

        return context_msg + self.working_memory.get()

    def get_plan_status(self) -> str:
        # prepare components of a plan status
        finished_tasks = self.plan.get_finished_tasks()
        code_written = [remove_comments(task.code) for task in finished_tasks]
        code_written = "\n\n".join(code_written)
        task_results = [task.result for task in finished_tasks]
        task_results = "\n\n".join(task_results)
        task_type_name = self.current_task.task_type
        task_type = TaskType.get_type(task_type_name)
        guidance = task_type.guidance if task_type else ""

        # combine components in a prompt
        prompt = PLAN_STATUS.format(
            code_written=code_written,
            task_results=task_results,
            current_task=self.current_task.instruction,
            guidance=guidance,
        )

        return prompt

    def get_all_tasks_results(self) -> tuple[str, list[dict]]:
        user_requirement = self.plan.goal
        tasks_with_res = [
            {"task_instr": task.instruction, "task_res": task.result}
            for idx, task in enumerate(self.plan.tasks)
            if idx != 0  # Exclude the first task
        ]
        return user_requirement, tasks_with_res

    def clear_plan(self):
        self.plan = Plan(goal="")

    def set_fixed_plan(self):
        self.plan = get_fixed_plan(181)

