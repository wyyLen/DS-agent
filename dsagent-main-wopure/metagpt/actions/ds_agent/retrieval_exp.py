import json
import re

from llama_index.core import PromptTemplate

from metagpt.actions import Action
from metagpt.actions.ds_agent.write_ds_plan import update_plan_from_rsp
from metagpt.const import EXAMPLE_DATA_PATH
from metagpt.rag.engines import CustomEngine
from metagpt.rag.schema import FAISSRetrieverConfig
from metagpt.schema import Message, Plan
from metagpt.strategy.ds_task_type import TaskType
from metagpt.utils.common import CodeParser


class RetrievalExp(Action):
    def __init__(self):
        super().__init__()

    async def run(self, exp_path: str, goal: str) -> str:
        engine = CustomEngine.from_docs(input_files=[exp_path], retriever_configs=[FAISSRetrieverConfig()])
        retrieval_res = await engine.aquery(goal)
        return retrieval_res.response


class GenerateQuery(Action):
    def __init__(self):
        super().__init__()

    async def run(self, query: str, num_queries: int) -> list[str]:
        query_gen_prompt_str = (
            "You are a helpful assistant that generates multiple search queries based on a "
            "single input query. Generate {num_queries} search queries, one on each line, "
            "related to the following input query:\n"
            "Query: {query}\n"
            "Queries:\n"
        )
        query_gen_prompt = PromptTemplate(query_gen_prompt_str)
        fmt_prompt = query_gen_prompt.format(num_queries=num_queries, query=query)
        response = await self._aask(fmt_prompt)
        queries = response.split("\n")
        return queries


class AdjustPlanFromWorkflow(Action):
    PROMPT_TEMPLATE: str = """
        # Current plan:
        {current_plan}
        # Similar workflow experience:
        {workflow_exp}
        # Available Task Types:
        {task_type_desc}
        # Task:
        Based on the planning experience from similar successful workflows, examine whether the current plan needs further refinement. 
        If you are modifying an existing plan, carefully follow the instructions and don't make unnecessary changes. Provide the whole plan unless instructed to modify only one task of the plan.
        Ensure that the number of tasks does not exceed {max_tasks} to avoid over-complicating the plan.

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
                "dependent_task_ids": list[str] = "ids of tasks prerequisite to this task",
                "instruction": "what you should do in this task, one short phrase or sentence",
                "task_type": "type of this task, should be one of Available Task Types",
            }},
            ...
        ]
        ```
        # Suffix:
        - Keep in mind that your response MUST follow the valid format above.
        - The first task should thoroughly analyze and present the basic structure of the dataset, including columns, data types, and any missing values or anomalies.
        - The number of tasks should not exceed {max_tasks}. **Don't add unnecessary tasks or over-complicate the plan**. You can merge two simple tasks if necessary.
        - Each task's instruction should be detailed and aligned with best practices from the similar workflow experience. Tailor each task based on those insights to ensure it fits within the overall plan.
        - If any task involves similar processes to those in the workflow experience, explicitly reference the learned strategies to improve task effectiveness.
        """

    async def run(self, plan: Plan, workflow_exp: str, max_tasks: int = 7):
        task_type_desc = "\n".join([f"- **{tt.type_name}**: {tt.value.desc}" for tt in TaskType])
        task_list = [{"task_id": task.task_id, "dependent_task_ids": task.dependent_task_ids,
                      "task_type": task.task_type, "instruction": task.instruction} for task in plan.tasks]
        prompt = self.PROMPT_TEMPLATE.format(current_plan=task_list, workflow_exp=workflow_exp,
                                             task_type_desc=task_type_desc, max_tasks=max_tasks)
        rsp = await self._aask(prompt)
        rsp = CodeParser.parse_code(block=None, text=rsp)
        rsp = self.repair_json(rsp)
        update_plan_from_rsp(rsp, plan)


    def repair_json(self, json_str: str) -> str:
        # desc:将生成的json中不合理的单引号修正为双引号（针对task_id和dependent_task_ids）
        data = re.sub(r'"task_id": \'(\d+)\'', r'"task_id": "\1"', json_str)
        def replace_dependent_task_ids(match):
            content = match.group(0).replace("'", '"')
            return content
        data = re.sub(r'"dependent_task_ids": \[(\'\d+\'(,\s*\'\d+\')*)?\]', replace_dependent_task_ids, data)
        return data
