"""
@Date    :   2024/6/28 15:47:31
@Author  :   wbq
@File    :   write_ds_plan.py
"""
from __future__ import annotations

import json
import re
from copy import deepcopy
from typing import Tuple

from metagpt.actions import Action
from metagpt.config2 import Config
from metagpt.const import METAGPT_ROOT
from metagpt.logs import logger
from metagpt.schema import Message, Plan, Task
from metagpt.strategy.ds_task_type import TaskType
from metagpt.utils.common import CodeParser


class WritePlan(Action):
    llm_config: Config = None
    PROMPT_TEMPLATE: str = """
    # Context:
    {context}
    # Available Task Types:
    {task_type_desc}
    # Task:
    Based on the context, write a plan or modify an existing plan of what you should do to achieve the goal. A plan consists of one to {max_tasks} tasks.
    If you are modifying an existing plan, carefully follow the instruction, don't make unnecessary changes. Give the whole plan unless instructed to modify only one task of the plan.
    If you encounter errors on the current task, revise and output the current single task only.
    Output a list of jsons following the format:
    ```json
    [
        {{
            "task_id": "1",
            "dependent_task_ids": [],
            "instruction": "Load the dataset, inspect its structure, and display basic information, including column names, data types, missing values, and sample data for each column.",
            "task_type": "pda"
        }},
        {{
            "task_id": str = "unique identifier for a task in plan, can be an ordinal",
            "dependent_task_ids": list[str] = "ids of tasks prerequisite to this task. The dependent task here must be a necessary condition for the current task. If there is no required dependent task, directly rely on the data pre-analysis of task 1",
            "instruction": "Detailed description of what you should do in this task. The 'instruction' field should avoid any double quotes (\") inside its text. Use single quotes (') to highlight phrases or specific terms to prevent conflict with the required outer double quotes.",
            "task_type": "type of this task, should be one of Available Task Types",
        }},
        ...
    ]
    ```
    # Suffix:
    - Keep in mind that Your response MUST follow the valid format above.
    - The `instruction` field should avoid using double quotes. Use single quotes instead.
    - The first task should thoroughly analyze and present the basic structure of the dataset, including columns, data types, and any missing values or anomalies.
    - The number of tasks should not exceed {max_tasks}. Don't add unnecessary tasks or over-complicate the plan. You can merge two simple tasks if necessary.
    """

    async def run(self, context: list[Message], max_tasks: int = 5) -> str:
        task_type_desc = "\n".join([f"- **{tt.type_name}**: {tt.value.desc}" for tt in TaskType])
        prompt = self.PROMPT_TEMPLATE.format(
            context="\n".join([str(ct) for ct in context]), max_tasks=max_tasks, task_type_desc=task_type_desc
        )
        rsp = await self._aask(prompt)
        rsp = CodeParser.parse_code(block=None, text=rsp)
        return rsp


def update_plan_from_rsp(rsp: str, current_plan: Plan):
    try:
        rsp = json.loads(rsp)
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse JSON response: {rsp}")
        raise e
    
    # Filter task_config to only include valid Task fields
    valid_fields = {"task_id", "dependent_task_ids", "instruction", "task_type", "code", "result", "is_success", "is_finished"}
    try:
        tasks = [Task(**{k: v for k, v in task_config.items() if k in valid_fields}) for task_config in rsp]
    except Exception as e:
        logger.error(f"Failed to create Task objects from response: {rsp}")
        logger.error(f"Error: {e}")
        raise e

    if len(tasks) == 1 or tasks[0].dependent_task_ids:
        if tasks[0].dependent_task_ids and len(tasks) > 1:
            # tasks[0].dependent_task_ids means the generated tasks are not a complete plan
            # for they depend on tasks in the current plan, in this case, we only support updating one task each time
            logger.warning(
                "Current plan will take only the first generated task if the generated tasks are not a complete plan"
            )
        # handle a single task
        if current_plan.has_task_id(tasks[0].task_id):
            # replace an existing task
            current_plan.replace_task(tasks[0])
        else:
            # append one task
            current_plan.append_task(tasks[0])

    else:
        # add tasks in general
        current_plan.add_tasks(tasks)


def precheck_update_plan_from_rsp(rsp: str, current_plan: Plan) -> Tuple[bool, str]:
    temp_plan = deepcopy(current_plan)
    try:
        update_plan_from_rsp(rsp, temp_plan)
        return True, ""
    except Exception as e:
        return False, e


class RefinePlan(Action):
    llm_config: Config = None
    PROMPT_TEMPLATE: str = """
    You are a data scientist and need to determine which specific detailed task type the current task belongs to.
    Output an Available Refined Task Type, WITHOUT any additional description.
    # Current Task
    {current_task}
    # Available Refined Task Types:
    {task_type_desc}
    - Keep in mind that Your response MUST NOT include any additional content.
    - Avoid appearing special symbols in your response, such as '-' and '*'.
    """
    FREEDOM_REFINE_PROMPT_TEMPLATE: str = """
    You are a data scientist tasked with classifying the current task into a specific refined task type.
    If the task involves summarizing or integrating results, always classify it as "Results Integration".
    For all other tasks, determine the most appropriate refined task type based on the description.

    - Your response must contain ONLY the refined task type.
    - Do not include any additional content, explanations, or symbols.
    - Format the task type with each word capitalized and separated by a single space.

    # Current Task
    {current_task}
    """

    async def refine_ds_scenarios_in_plan(self, current_plan: Plan):
        task_list = [{"task_id": task.task_id, "dependent_task_ids": task.dependent_task_ids,
                      "task_type": task.task_type, "instruction": task.instruction} for task in current_plan.tasks]
        refine_target = {
            "machine learning": {
                "Linear Regression": "The linear regression algorithm describes the linear relationship between the independent variable and the dependent variable by fitting a straight line and predicting new data points.",
                "Logistic Regression": "Logistic regression is a classification algorithm that models the probability of a binary outcome using a logistic function to map input variables to a value between 0 and 1.",
                "Linear Discriminant Analysis": "Linear Discriminant Analysis (LDA) is a classification algorithm that finds a linear combination of features to separate two or more classes by maximizing the ratio of between-class variance to within-class variance.",
                "Decision Tree": "A decision tree is a machine learning algorithm that splits data into branches based on feature values, recursively partitioning the dataset to create a model that predicts outcomes by following decision paths from root to leaf nodes.",
                "Naive Bayes classifier": "The Naive Bayes classifier is a probabilistic algorithm that applies Bayes' theorem with the assumption of independence between features to classify data based on the likelihood of different outcomes.",
                "KNN": "The K-Nearest Neighbors (KNN) algorithm classifies data points based on the majority label of the K closest points in the feature space, using a distance metric to identify neighbors.",
                "SVM": "Support Vector Machine (SVM) is a classification algorithm that finds the optimal hyperplane to separate data points from different classes with the maximum margin between them.",
                "model evaluation": "Use appropriate methods to evaluate model performance based on task requirements",
            },
        }
        for task in task_list:
            if task["task_type"] in refine_target:
                refine_task_type_desc = "\n".join([f"- **{type_name}**: {desc}" for type_name, desc in refine_target[task["task_type"]].items()])
                prompt = self.PROMPT_TEMPLATE.format(current_task=task, task_type_desc=refine_task_type_desc)
                rsp = await self._aask(prompt)
                if rsp in refine_target[task["task_type"]]:
                    task["task_type"] = task["task_type"] + "-" + rsp
                else:
                    print(f"Invalid refined response: {rsp}")
            elif task["task_type"] == "other":
                prompt = self.FREEDOM_REFINE_PROMPT_TEMPLATE.format(current_task=task)
                rsp = await self._aask(prompt)
                pattern = r'[^a-zA-Z\s]'
                if bool(re.search(pattern, rsp)):
                    print(f"Invalid refined response: {rsp}")
                else:
                    task["task_type"] = task["task_type"] + "-" + rsp
        for i, task in enumerate(current_plan.tasks):
            task.task_type = task_list[i]["task_type"]
