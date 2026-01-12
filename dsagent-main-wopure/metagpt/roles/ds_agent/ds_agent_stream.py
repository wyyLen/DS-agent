from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Literal, Any, Union, AsyncGenerator

from llama_index.core import QueryBundle
from llama_index.core.query_engine import RetrieverQueryEngine
from pydantic import Field, model_validator

from examples.ds_agent.reformat import reformat
from examples.experiment.da_bench.util.common import check_file_exist, record_token_cost
from metagpt.actions import ExecuteNbCode, UserRequirement
from metagpt.actions.ds_agent.ask_review import ReviewConst
from metagpt.actions.ds_agent.conclude_res import Conclusion
from metagpt.actions.ds_agent.extract_thought import ThoughtExtract
from metagpt.actions.ds_agent.query_utils import QueryUtils
from metagpt.actions.ds_agent.retrieval_exp import RetrievalExp, GenerateQuery, AdjustPlanFromWorkflow
from metagpt.actions.ds_agent.write_ds_code import WriteDsCode, CheckData
from metagpt.actions.ds_agent.write_ds_plan import RefinePlan
from metagpt.config2 import Config
from metagpt.const import METAGPT_ROOT, EXP_PLAN, WORKFLOW_EXP, EXAMPLE_DATA_PATH, DA_EVAL_RES_PATH
from metagpt.llm import LLM
from metagpt.logs import logger
from metagpt.memory import Memory
from metagpt.prompts.ds_agent.write_ds_code import DATA_INFO
from metagpt.provider import OpenAILLM
from metagpt.provider.base_llm import BaseLLM
from metagpt.rag.engines import CustomEngine
from metagpt.rag.engines.customMixture import CustomMixtureEngine
from metagpt.rag.engines.customWorkflowGM import CustomWorkflowGMEngine
from metagpt.rag.factories.llm import RAGLLM
from metagpt.rag.schema import FAISSRetrieverConfig, MixtureRetrieverConfig
from metagpt.roles import Role
from metagpt.schema import Message, Task, TaskResult, Plan
from metagpt.strategy import ds_planner
from metagpt.strategy.ds_task_type import TaskType
from metagpt.strategy.lats_react import LanguageAgentTreeSearch, Node, collect_trajectory
from metagpt.tools.tool_recommend import ToolRecommender, BM25ToolRecommender

gpt4o_config_path = METAGPT_ROOT / "config" / "gpt-4o.yaml"
gpt4o_config = Config.from_yaml_file(gpt4o_config_path)
gpt4o_mini_config_path = METAGPT_ROOT / "config" / "gpt-4o-mini.yaml"
gpt4o_mini_config = Config.from_yaml_file(gpt4o_mini_config_path)

# desc 要验证 exp_extractor 的能力，需要保证经验池为空，使用 custom系列的经验池
custom_plan_exp = EXP_PLAN
custom_workflow_exp = WORKFLOW_EXP
# custom_plan_exp = EXAMPLE_DATA_PATH / "exp_bank/custom_plan_exp.json"
# custom_workflow_exp = EXAMPLE_DATA_PATH / "exp_bank/custom_workflow_exp.json"

check_file_exist(custom_plan_exp)
check_file_exist(custom_workflow_exp)


def get_rag_engine_llm(model_infer: BaseLLM = None) -> RAGLLM:
    # Use config with explicit context_window to avoid context size errors
    llm = model_infer or LLM()
    # Qwen models typically support 8192 tokens context
    return RAGLLM(model_infer=llm, context_window=8192, num_output=4096)


async def add_to_exp_bank(goal: str, thought: str, plan_exp: Path):
    if not plan_exp.exists():
        raise ValueError(f"{plan_exp} does not exist.")
    with open(plan_exp, 'r', encoding='utf-8') as file:
        exp_data = json.load(file)
    exp_data.append({
        "task": goal,
        "solution": thought
    })
    with open(plan_exp, 'w', encoding='utf-8') as file:
        json.dump(exp_data, file, ensure_ascii=False, indent=4)


async def add_to_exp_bank_with_metadata(goal: str, thought: str, metadata: str, plan_exp: Path):
    if not plan_exp.exists():
        raise ValueError(f"{plan_exp} does not exist.")
    with open(plan_exp, 'r', encoding='utf-8') as file:
        exp_data = json.load(file)
    exp_data.append({
        "task": goal,
        "solution": thought,
        "metadata": metadata
    })
    with open(plan_exp, 'w', encoding='utf-8') as file:
        json.dump(exp_data, file, ensure_ascii=False, indent=4)


def remove_last_item_from_exp_bank(plan_exp: Path):
    if not plan_exp.exists():
        raise ValueError(f"{plan_exp} does not exist.")
    with open(plan_exp, 'r', encoding='utf-8') as file:
        exp_data = json.load(file)
    if isinstance(exp_data, list) and exp_data:
        exp_data.pop()
    with open(plan_exp, 'w', encoding='utf-8') as file:
        json.dump(exp_data, file, ensure_ascii=False, indent=4)


async def add_to_workflow_exp_bank(plan: Plan, exp: str, workflow_exp_path: Path):
    if not workflow_exp_path.exists():
        raise ValueError(f"{workflow_exp_path} does not exist.")
    with open(workflow_exp_path, 'r', encoding='utf-8') as file:
        workflow_exp_data = json.load(file)
    tasks = [{
        "task_id": task.task_id,
        "dependent_task_ids": task.dependent_task_ids,
        "instruction": task.instruction,
        "task_type": task.task_type
    } for task in plan.tasks]
    workflow_exp_data.append({
        "task": plan.goal,
        "workflow": tasks,
        "exp": exp
    })
    with open(workflow_exp_path, 'w', encoding='utf-8') as file:
        json.dump(workflow_exp_data, file, ensure_ascii=False, indent=4)


class DSAgentStream(Role):
    name: str = "wbq"
    profile: str = "DSAgent"
    auto_run: bool = True
    use_plan: bool = True
    use_reflection: bool = True
    model_infer: Union[BaseLLM, None] = Field(default=None, description="Optional model inference LLM")
    execute_code: ExecuteNbCode = Field(default_factory=ExecuteNbCode, exclude=True)
    react_mode: Literal["plan_and_act", "react"] = "plan_and_act"
    max_react_loop: int = 10  # used for react mode
    use_rag: bool = True
    use_kaggle_exp: bool = True
    use_exp_extractor: bool = False
    rag_engine: Union[CustomEngine, CustomMixtureEngine] = None
    rag_exp_similarity_threshold: float = 0.33
    tools: Union[str, list[str]] = []  # Use special symbol ["<all>"] to indicate use of all registered tools
    tool_recommender: ToolRecommender = None
    workflow_engine: CustomWorkflowGMEngine = None

    @model_validator(mode="after")
    def set_plan_and_tool(self):
        self._set_react_mode(react_mode=self.react_mode, max_react_loop=self.max_react_loop, auto_run=self.auto_run)
        self.use_plan = (
                self.react_mode == "plan_and_act"
        )  # create a flag for convenience, overwrite any passed-in value
        self.set_actions([WriteDsCode])
        self._set_state(0)
        if self.use_rag:
            with open(custom_plan_exp, 'r', encoding='utf-8') as file:
                exp_data = json.load(file)
            print(f"exp_bank size: {len(exp_data)}")
            # 暂时使用BM25检索器以避免FAISS兼容性问题
            from metagpt.rag.schema import BM25RetrieverConfig
            self.rag_engine = CustomEngine.from_docs(
                input_files=[custom_plan_exp],
                retriever_configs=[BM25RetrieverConfig(similarity_top_k=2)],
                ranker_configs=[],
                llm=get_rag_engine_llm(),
            )
        # BM25不需要调整阈值
        if self.tools:
            self.tool_recommender = BM25ToolRecommender(tools=self.tools)
        self.workflow_engine = CustomWorkflowGMEngine(custom_workflow_exp)
        return self

    @property
    def working_memory(self):
        return self.rc.working_memory

    async def stream_run(self, with_message=None) -> AsyncGenerator[str, None]:
        try:
            if with_message:
                msg = self._format_input_message(with_message)
                self.put_message(msg)
                if not await self._observe():
                    # If there is no new information, suspend and wait
                    logger.debug(f"{self._setting}: no news. waiting.")
                    return
                yield f"🚀 任务接收成功：{msg.content[:100]}...\n"
            async for chunk in self._plan_and_act_stream():
                yield chunk
        except Exception as e:
            yield f"❌ 执行异常：{str(e)}"
        finally:
            self._set_state(state=-1)
            self.set_todo(None)
            yield "\n🔚 流程执行结束"

    def _format_input_message(self, input_data) -> Message:
        if isinstance(input_data, str):
            return Message(content=input_data, cause_by=UserRequirement)
        elif isinstance(input_data, Message):
            return input_data
        elif isinstance(input_data, list):
            return Message(content="\n".join(input_data))
        raise ValueError("不支持的输入类型")

    async def _plan_and_act_stream(self) -> AsyncGenerator[str, None]:
        try:
            goal = self.rc.memory.get()[-1].content
            self.planner = ds_planner.Planner(goal=self.goal, working_memory=self.rc.working_memory, auto_run=self.auto_run)
            yield f"【think stage】\n"
            # yield f"【think stage】\n 当前任务目标：{goal}\n"
            current_use_rag = self.use_rag
            if current_use_rag and self.use_kaggle_exp and not self.use_exp_extractor:
                yield "🤔 分析任务类型...\n"
                task_types = await QueryUtils().getQuestionType(question=goal)
                yield f"📊 任务类型分析结果：{task_types}\n"
                if 'machine learning' not in task_types:
                    current_use_rag = False

            if self.use_rag and current_use_rag:
                yield "🔍 启动经验检索流程...\n"
                async for rag_chunk in self._rag_stream(goal):
                    yield rag_chunk
            else:
                yield "⏩ 跳过经验检索阶段\n"
            yield "📝 正在生成执行计划...\n"
            await self.planner.update_plan(goal=goal)
            yield f"📋 初步计划已完成\n"
            if len(self.planner.plan.tasks) > 3:
                yield "🔍 启动工作流检索流程...\n"
                workflow_rag_res = self.workflow_engine.retrieval(self.planner.plan)
                # yield f"workflow_rag_res: \n{workflow_rag_res}\n"
                if workflow_rag_res is not None and current_use_rag:
                    await AdjustPlanFromWorkflow().run(plan=self.planner.plan, workflow_exp=workflow_rag_res)
                    yield f"📋 工作流检索计划优化完成\n"
                    self.working_memory.add(Message(content=workflow_rag_res, role="user", cause_by=AdjustPlanFromWorkflow))
            self.planner.plan.tasks[0].instruction = "Load the dataset, inspect its structure, and display basic information, including column names, data types, missing values, and sample data for each column."

            def format_tasks(tasks: list[Task]) -> str:
                if not tasks:
                    return "暂无任务详情"
                tasks_str = ""
                for i, task in enumerate(tasks):
                    tasks_str += f"{task.task_id}. {task.task_type} - {task.instruction}\n"
                return tasks_str


            tasks = format_tasks(self.planner.plan.tasks)
            yield f"📋 最终计划：\n```\n{tasks}\n``` \n ️进入执行阶段...\n"

            while self.planner.current_task:
                task = self.planner.current_task
                yield f"ready to take on task {task}\n\n"
                task_result = await self._act_on_task(task)
                yield (f"```\n{task_result.code}\n``` \n "
                       f"execute result: \n```\n{task_result.result}\n```\n "
                       f"execute status: {task_result.is_success}\n")
                if not task_result.is_success:
                    yield "❌ 执行失败"
                    break
                await self.planner.process_task_result(task_result)

            await self.execute_code.terminate()
            conclusion = await self._conclude()
            yield f"【/think stage】\n{conclusion}"
            self.working_memory.add(conclusion)
        except Exception as e:
            import traceback
            error_detail = traceback.format_exc()
            logger.error(f"智能体执行异常详情:\n{error_detail}")
            yield f"❌ 智能体执行异常：{str(e)}"
            raise e

    async def _rag_stream(self, goal: str) -> AsyncGenerator[str, None]:
        """RAG流程流式化"""
        try:
            yield "🔎 正在检索经验库...\n"
            nodes_with_score = await self.rag_engine.aretrieve(goal)
            yield f"📚 检索到{len(nodes_with_score)}条相关经验\n"
            
            # 输出检索到的经验详情
            for idx, node_with_score in enumerate(nodes_with_score, 1):
                logger.info(f"检索经验 {idx}:")
                logger.info(f"  匹配分数: {node_with_score.score:.4f}")
                logger.info(f"  内容预览: {node_with_score.get_content()[:200]}...")
            
            if nodes_with_score and len(nodes_with_score) > 0:
                top_score = nodes_with_score[0].score
                # yield f"📈 最高匹配度：{top_score:.2f}\n"
                if top_score > self.rag_exp_similarity_threshold:
                    yield "🧠 正在分析经验内容...\n"
                    try:
                        retrieval_res = await self.rag_engine.get_synthesizer_response(nodes_with_score, QueryBundle(goal))
                        self.working_memory.add(Message(content=retrieval_res.response, role="user", cause_by=RetrievalExp))
                        yield "✅ 经验整合完成\n"
                    except IndexError as ie:
                        if "pop from empty list" in str(ie):
                            logger.warning(f"llama-index callback stack error (known issue), but synthesis completed: {ie}")
                            # 尝试直接使用节点内容
                            exp_texts = [node.get_content() for node in nodes_with_score]
                            combined_exp = "\n\n".join(exp_texts[:2])  # 只取前2个
                            self.working_memory.add(Message(content=f"相关经验：\n{combined_exp}", role="user", cause_by=RetrievalExp))
                            yield "✅ 经验整合完成（绕过回调错误）\n"
                        else:
                            raise
                else:
                    pass
                    # yield "❌ 未找到高匹配度经验\n"
            else:
                yield "⚠️ 未检索到相关经验\n"
        except Exception as e:
            import traceback
            logger.error(f"RAG检索失败: {str(e)}\n{traceback.format_exc()}")
            yield f"❌ 经验检索失败：{str(e)}"

    async def _plan_and_act(self) -> Message:
        try:
            goal = self.rc.memory.get()[-1].content
            self.planner = ds_planner.Planner(goal=self.goal, working_memory=self.rc.working_memory,
                                              auto_run=self.auto_run)

            # desc 0. 判断是否需要 RAG
            #    如果只使用kaggle的经验知识，但是并非机器学习任务，就不使用 rag
            current_use_rag = self.use_rag
            if current_use_rag and self.use_kaggle_exp and not self.use_exp_extractor:
                task_types = await QueryUtils().getQuestionType(question=goal)
                if 'machine learning' not in task_types:
                    current_use_rag = False

            # desc 1. retrieve from exp_bank base on custom mixture/vector retrieval engine
            nodes_with_score = []
            if self.use_rag and current_use_rag:
                nodes_with_score = await self.rag_engine.aretrieve(goal)
                logger.info(f"nodes_with_score: {nodes_with_score}")
                if nodes_with_score and len(nodes_with_score) > 0:
                    nodes_with_score = nodes_with_score[:1]
                    # todo: Maybe we need an action to determine whether this experience is effective.
                    if nodes_with_score[0].score > self.rag_exp_similarity_threshold:
                        print(f"original retrieval_res: {nodes_with_score}\n")
                        retrieval_res = await self.rag_engine.get_synthesizer_response(nodes_with_score, QueryBundle(goal))
                        print(f"summary retrieval_res: {retrieval_res}\n")
                        self.working_memory.add(Message(content=retrieval_res.response, role="user", cause_by=RetrievalExp))
                    else:
                        logger.info(f"no similar experience found. recording rag failed case.")
                        self.record_rag_failed_cases()
                else:
                    logger.info(f"no nodes retrieved from RAG engine.")
                    self.record_rag_failed_cases()

            # desc 2. make a plan and adjust based on custom workflow retrieval engine
            await self.planner.update_plan(goal=goal)
            workflow_rag_res = None
            if len(self.planner.plan.tasks) > 3:
                workflow_rag_res = self.workflow_engine.retrieval(self.planner.plan)
                print(f"workflow_rag_res: {workflow_rag_res}\n")
                if workflow_rag_res is not None and current_use_rag:
                    await AdjustPlanFromWorkflow().run(plan=self.planner.plan, workflow_exp=workflow_rag_res)
                    self.working_memory.add(Message(content=workflow_rag_res, role="user", cause_by=AdjustPlanFromWorkflow))
            self.planner.plan.tasks[0].instruction = "Load the dataset, inspect its structure, and display basic information, including column names, data types, missing values, and sample data for each column."

            # desc 2.1  distinguish different data science scenarios
            await RefinePlan().refine_ds_scenarios_in_plan(self.planner.plan)

            use_lats = False

            # desc 3. act on each task
            while self.planner.current_task:
                task = self.planner.current_task
                logger.info(f"ready to take on task {task}")
                task_result = await self._act_on_task(task)
                if not task_result.is_success:
                    use_lats = True
                    self.save_failed_cases()
                    break
                await self.planner.process_task_result(task_result)

            if use_lats:
                # desc:
                #     1. 当 ds_planner多次尝试无法解决时，采用lats，需要将 ds_planner 失败的路径传递给 lats
                #     2. 注意最后需要将lats的成功路径转移到DSAgent中，以便后续经验提取
                failed_child_node = self._ds_planner_to_node_trajectory()
                failed_trajectory = [{'trajectory': collect_trajectory(failed_child_node), 'final_answer': f"{failed_child_node.state['observation']}"}]
                lats = LanguageAgentTreeSearch(goal=goal, failed_trajectory=failed_trajectory)
                conclusion, best_child = await lats.enhance_run(iterations=10)
                self.add_node_trajectory_to_working_memory_and_planner(best_child)
                prompt_token, completion_token = lats.calculate_total_cost()
                lats_token_record_path = DA_EVAL_RES_PATH / f"lats_token_record.md"
                record_token_cost(lats_token_record_path, prompt_token, completion_token)
                self.llm.cost_manager.update_cost(prompt_token, completion_token, self.llm.model)
            else:
                # desc 4. close nb terminal & conclude
                # conclusion = self.planner.get_useful_memories()[0]
                await self.execute_code.terminate()
                conclusion = await self._conclude()
                self.working_memory.add(conclusion)

            # desc 5. extract thought & update exp_bank
            extracted_thought = None
            is_completed = self.planner.plan.tasks[-1].is_success  # 只有当前任务完成了才有经验价值
            if self.use_rag and self.use_exp_extractor and is_completed and len(nodes_with_score) > 0:
                if (self.rag_engine is CustomMixtureEngine and nodes_with_score[0].get_score() < self.rag_exp_similarity_threshold or
                        self.rag_engine is not CustomMixtureEngine and nodes_with_score[0].get_score() > self.rag_exp_similarity_threshold):
                    logger.info("Current task exp knowledge should be added to exp_bank")
                    extracted_thought = await ThoughtExtract(config=gpt4o_config).extract_goal_from_working_memory(self.goal, self.working_memory.get())
                    metadata = await QueryUtils().getQAType(goal, extracted_thought)
                    await self._dynamic_add_exp_with_metadata(goal, extracted_thought, metadata)
                    await add_to_exp_bank_with_metadata(goal, extracted_thought, metadata, custom_plan_exp)

            # desc 6. update workflow_bank
            if self.use_exp_extractor and is_completed and len(self.planner.plan.tasks) > 3 and workflow_rag_res is None:
                logger.info("Current task exp knowledge should be added to workflow_bank")
                if extracted_thought is None:
                    extracted_thought = await ThoughtExtract(config=gpt4o_config).extract_goal_from_working_memory(self.goal, self.working_memory.get())
                await self._dynamic_add_workflow_exp(self.planner.plan, extracted_thought)
                await add_to_workflow_exp_bank(self.planner.plan, extracted_thought, custom_workflow_exp)

            return conclusion
        except Exception as e:
            await self.execute_code.terminate()
            raise e

    async def _act_on_task(self, current_task: Task) -> TaskResult:
        code, result, is_success = await self._write_and_exec_code()
        task_result = TaskResult(code=code, result=result, is_success=is_success)
        print(task_result.result)
        return task_result

    async def _write_and_exec_code(self, max_retry: int = 10, max_replanning_times: int = 2):
        counter, success, replanning_counter = 0, False, 0
        code, result = "", ""
        plan_status = self.planner.get_plan_status() if self.use_plan else ""

        if self.tools:
            context = self.working_memory.get()[-1].content if self.working_memory.get() else ""
            plan = self.planner.plan if self.use_plan else None
            tool_info = await self.tool_recommender.get_recommended_tool_info(context=context, plan=plan)
        else:
            tool_info = ""
        logger.info(f"recommended tools: {tool_info}")

        # await self._check_data()

        while not success and counter <= max_retry:
            code, cause_by = await self._write_ds_code(counter, plan_status, tool_info)
            self.working_memory.add(Message(content=code, role="assistant", cause_by=cause_by))
            result, success = await self.execute_code.run(code)
            self.working_memory.add(Message(content=result, role="user", cause_by=ExecuteNbCode))
            counter += 1
            if not success and counter >= max_retry:
                logger.info("awesome, coding failed!")
                replanning_counter += 1
                if replanning_counter >= max_replanning_times:
                    logger.error(f"re-planning too many times, coding failed!, re-planning counter: {replanning_counter}")
                    break
                review, _ = await self.planner.ask_review(auto_run=False, trigger=ReviewConst.CODE_REVIEW_TRIGGER)
                if ReviewConst.CHANGE_WORDS[0] in review:
                    counter = 0
        return code, result, success

    async def _write_ds_code(self, counter: int, plan_status: str = "", tool_info: str = ""):
        todo = self.rc.todo  # todo is WriteDSCode
        logger.info(f"ready to {todo.name}")
        use_reflection = counter > 0 and self.use_reflection
        user_requirement = self.get_memories()[0].content
        code = await todo.run(user_requirement=user_requirement, plan_status=plan_status, tool_info=tool_info,
                              working_memory=self.working_memory.get(), use_reflection=use_reflection)
        return code, todo

    async def _conclude(self) -> Message:
        question, tasks_res = self.planner.get_all_tasks_results()
        final_res = await Conclusion().run(final_goal=question, tasks_res=tasks_res)
        conclude_msg = Message(content=final_res, role="assistant", cause_by=Conclusion)
        return conclude_msg

    async def _dynamic_add_exp(self, goal: str, thought: str):
        new_docs = json.dumps({
            "task": goal,
            "plan_output": thought
        })
        self.rag_engine.add_exp(new_docs)

    async def _dynamic_add_exp_with_metadata(self, goal: str, thought: str, metadata: str):
        new_docs = json.dumps({
            "task": goal,
            "plan_output": thought,
            "metadata": metadata
        })
        self.rag_engine.add_exp(new_docs)

    def clear_content(self):
        self.working_memory.clear()
        self.rc.memory.clear()
        self._set_react_mode(react_mode="plan_and_act", max_react_loop=10, auto_run=True)
        self.use_plan = (self.react_mode == "plan_and_act")
        self.set_actions([WriteDsCode])
        self._set_state(0)

    async def generate_queries(self, query: str, num_queries: int = 3):
        queries = [query]
        gen_queries = await GenerateQuery().run(query=query, num_queries=num_queries - 1)
        queries.extend(gen_queries)
        return queries

    async def _dynamic_add_workflow_exp(self, plan: Plan, exp: str):
        self.workflow_engine.add(plan, exp)

    def save_failed_cases(self):
        failed_cases_path = DA_EVAL_RES_PATH / "failed_cases/DSAInterpreter_failed_cases_glm4.json"
        check_file_exist(failed_cases_path)
        workflow = [{"task_id": task.task_id, "dependent_task_ids": task.dependent_task_ids,
                     "task_type": task.task_type, "instruction": task.instruction, "code": task.code,
                     "result": task.result} for task in self.planner.plan.tasks]
        memory = [{"role": event.role, "content": event.content, "cause_by": event.cause_by}
                  for event in self.working_memory.get()]
        new_case = {
            "goal": self.planner.plan.goal,
            "plan": workflow,
            "working_memory": memory,
        }
        with open(failed_cases_path, 'a', encoding='utf-8') as f:
            json.dump(new_case, f, ensure_ascii=False)
            f.write('\n')

    def record_rag_failed_cases(self):
        failed_cases_path = DA_EVAL_RES_PATH / "failed_cases/our_rag_failed_cases.md"
        check_file_exist(failed_cases_path)
        with open(failed_cases_path, 'a', encoding='utf-8') as f:
            json.dump({"goal": self.planner.plan.goal or self.goal or self.rc.memory.get()[-1].content}, f, ensure_ascii=False)
            f.write('\n')

    def add_node_trajectory_to_working_memory_and_planner(self, node):
        self._clear_ds_planner_working_memory()
        self._node_trajectory_to_ds_planner(node)
        msg_list = self._node_trajectory_to_msg_list(node)
        for msg in msg_list:
            self.working_memory.add(msg)

    def _node_trajectory_to_msg_list(self, node) -> list[Message]:
        msg_list = []
        question = node.question
        while node:
            if node.state['observation']:
                msg_list.append(Message(content=node.state['observation'], role="assistant"))
            if node.state['action']:
                msg_list.append(Message(content=node.state['action'], role="assistant"))
            if node.state['thought']:
                msg_list.append(Message(content=node.state['thought']['thought'], role="assistant"))
            node = node.parent
        msg_list.append(Message(content=question, role="user"))
        msg_list.reverse()
        return msg_list

    def _node_trajectory_to_ds_planner(self, node):
        new_plan = Plan(goal=self.planner.plan.goal)
        task_list = []
        while node:
            if node.state['observation'] and node.state['action'] and node.state['thought']:
                task_list.append(Task(instruction=node.state['thought']['thought'], task_type=node.state['thought']['task_type'],
                                      code=node.state['action'], result=node.state['observation']))
            node = node.parent
        task_list.reverse()
        for i, task in enumerate(task_list, 1):
            task.task_id = str(i)
            if i > 1:
                task.dependent_task_ids.append(str(i - 1))
            task.is_success = True
            if i == len(task_list):
                task.is_finished = True
        new_plan.tasks = task_list
        self.planner.plan = new_plan

    def _ds_planner_to_node_trajectory(self):
        task_list, goal = self.planner.plan.tasks, self.planner.plan.goal
        if not task_list:
            logger.error("error in _ds_planner_to_node_trajectory function: tasks in ds_planner is empty")
            return None
        root = Node(question=goal, state={'thought': task_list[0].instruction, 'action': task_list[0].code, 'observation': task_list[0].result})
        current_node = root
        for task in task_list[1:]:
            node = Node(question=goal, parent=current_node, state={'thought': task.instruction, 'action': task.code, 'observation': task.result})
            current_node = node
        return current_node

    def _clear_ds_planner_working_memory(self):
        user_messages = self.working_memory.get_by_role("user")
        self.working_memory.clear()
        for msg in user_messages:
            self.working_memory.add(msg)

    async def _check_data(self):
        if (
            not self.use_plan
            or not self.planner.plan.get_finished_tasks()
            or self.planner.plan.current_task.task_type
            not in [
                TaskType.DATA_PREPROCESS.type_name,
                TaskType.FEATURE_ENGINEERING.type_name,
                TaskType.MACHINE_LEARNING.type_name,
            ]
        ):
            return
        logger.info("Check updated data")
        code = await CheckData().run(self.planner.plan)
        if not code.strip():
            return
        result, success = await self.execute_code.run(code)
        if success:
            print(result)
            data_info = DATA_INFO.format(info=result)
            self.working_memory.add(Message(content=data_info, role="user", cause_by=CheckData))
