import json
import re

import json5

from metagpt.actions import Action, ExecuteNbCode
from metagpt.config2 import Config
from metagpt.const import METAGPT_ROOT
from metagpt.logs import logger
from metagpt.prompts.ds_agent.write_ds_code import DS_AGENT_SYSTEM_MSG, REFLECTION_PROMPT2, REFLECTION_SYSTEM_MSG, \
    FAILURE_TRAJECTORY_ANALYSIS_PROMPT, FAILURE_NODE_ANALYSIS_PROMPT, LATS_REFLECTION_PROMPT
from metagpt.prompts.lats.lats_react_prompt import cot_prompt_feedback, cot_prompt_feedback_short, cot_prompt, \
    cot_prompt_short, value_prompt_reasoning_feedback, value_prompt_reasoning_feedback_short, value_prompt_reasoning, \
    cot_generate_solution_space_prompt, generate_solution_space_case1, trajectory_value_prompt_reasoning, \
    experience_driven_generate_solution_space_prompt, terminal_trajectory_value_prompt_reasoning
from metagpt.schema import Message
from metagpt.utils.common import CodeParser
from metagpt.utils.custom_utils import fix_json, extract_final_score, extract_evaluation_scores, try_parse_json_object, \
    extract_last_thought_json
from metagpt.utils.token_counter import count_message_tokens


def get_unique_trajectories(failed_trajectories, num=5):
    unique_trajectories = []
    seen_final_answers = set()
    for traj in failed_trajectories:
        final_answer = traj.get('final_answer')
        if final_answer not in seen_final_answers:
            unique_trajectories.append(node_trajectory_to_text(traj['trajectory']))
            seen_final_answers.add(final_answer)
        if len(unique_trajectories) >= num:
            break
    return unique_trajectories


def node_trajectory_to_text(node_string):
    lines = node_string.split('\n')
    formatted_lines = []
    for line in lines:
        try:
            depth = int(line.split(",")[0].split("=")[1].strip())
            thought = line.split(", thought=")[1].split(", action=")[0].strip()
            action = line.split(", action=")[1].split(", observation=")[0].strip()
            observation = line.split(", observation=")[1].split(")")[0].strip()
        except IndexError:
            continue

        if depth != 0:
            if thought:
                formatted_lines.append(f"Thought {depth}: {thought}")
            if action:
                formatted_lines.append(f"Action {depth}: {action}")
            if observation:
                formatted_lines.append(f"Observation {depth}: {observation}")

    return '\n'.join(formatted_lines)


def node_trajectory_to_msg_list_short(node):
    msg_list = []
    question = node.question
    while node:
        if node.state['observation']:
            msg_list.append(Message(content=node.state['observation'], role="assistant"))
        if node.state['action']:
            msg_list.append(Message(content=node.state['action'], role="assistant"))
        node = node.parent
    msg_list.reverse()
    return question, msg_list


def node_trajectory_to_msg_list(node) -> list[Message]:
    msg_list = []
    question = node.question
    while node:
        if node.state['observation']:
            msg_list.append(Message(content=node.state['observation'], role="assistant"))
        if node.state['action']:
            msg_list.append(Message(content=node.state['action'], role="assistant"))
        if node.state['thought']:
            msg_list.append(Message(content=json.dumps(node.state['thought']), role="assistant"))
        node = node.parent
    msg_list.append(Message(content=question, role="user"))
    msg_list.reverse()
    return msg_list[-10:]


def format_reflections(reflection_map):
    reflections_str = ""
    for index, value in enumerate(reflection_map.values(), start=1):
        reflections_str += f"Reflection {index}:\n{value}\n"
    return reflections_str


class GenerateAction(Action):
    async def reflect_failed_node(self, context):
        logger.info(f"Reflecting failed node...")
        failure_trajectory_analysis_prompt = FAILURE_NODE_ANALYSIS_PROMPT.format(context=context)
        summary = await self._aask(failure_trajectory_analysis_prompt)
        return summary

    async def reflect_failed_trajectory(self, node):
        logger.info(f"Reflecting failed trajectory...")
        goal, context = node_trajectory_to_msg_list_short(node)
        failure_trajectory_analysis_prompt = FAILURE_TRAJECTORY_ANALYSIS_PROMPT.format(goal=goal, context=context)
        summary = await self._aask(failure_trajectory_analysis_prompt)
        return summary

    async def debug_with_reflection(self, node, current_thought, current_action_code, error_msg, layer_reflections) -> tuple[str, str]:
        context: list[Message] = node_trajectory_to_msg_list(node)
        context.append(Message(content=json.dumps(current_thought), role="assistant"))
        previous_impl: list[Message] = [Message(content=current_action_code, role="assistant"),
                                        Message(content=error_msg, role="assistant")]
        reflection_prompt = REFLECTION_PROMPT2.format(context=context, previous_impl=previous_impl) \
            if not layer_reflections else LATS_REFLECTION_PROMPT.format(context=context, previous_impl=previous_impl, layer_reflections=layer_reflections)
        rsp = await self._aask(reflection_prompt, system_msgs=[REFLECTION_SYSTEM_MSG])

        def extract_before_code_block(s):
            index = s.find('```')
            return s[:index] if index != -1 else s

        thought = extract_before_code_block(rsp)
        improved_impl = CodeParser.parse_code(block=None, text=rsp)
        return thought, improved_impl

    async def generate_pda(self, goal) -> tuple[str, str]:
        instruction = "First, I will load the dataset, inspect its structure, and display basic information, including column names, data types, missing values, and sample data for each column."
        WRITE_PDA_CODE_PROMPT = """
        # Task Description
        {user_requirement}
        # Current Step Requirement
        {instruction}
        # Constraints
        - Focus on the Current Step Requirement to generate the necessary code.
        - Ensure the output code is executable in the same Jupyter notebook as the previous executed code.
        - The code should be concise, well-commented, and follow best practices for data analysis.
        - Use pandas for data manipulation and analysis unless otherwise specified.
        # Output
        While some concise thoughts are helpful, code is absolutely required. Always output one and only one code block in your response. Output code in the following format:
        ```python
        your code
        ```
        """
        prompt = WRITE_PDA_CODE_PROMPT.format(user_requirement=goal, instruction=instruction)
        rsp = await self._aask(prompt)
        code = CodeParser.parse_code(block=None, text=rsp)
        return instruction, code

    async def generate_solution_space(self, previous_trajectory, next_tip, n_generate_sample, workflow_exp: list,  reflection_map, failed_trajectories, stop="Observation", max_token_length=16000):
        # func 解空间生成
        input = previous_trajectory + next_tip
        solution_space, sampled_cases = [], []
        reflections = format_reflections(reflection_map)
        max_using_workflow_count = min(n_generate_sample - 1, len(workflow_exp))  # 最多的经验驱动数量，保证有自由探索的空间
        for count in range(n_generate_sample):
            previous_samples = "\n".join(sampled_cases) + "\n" if sampled_cases else ""
            if count < max_using_workflow_count:
                prompt = experience_driven_generate_solution_space_prompt.format(case=generate_solution_space_case1, workflow_exp=workflow_exp.pop(0), input=input, failed_reflections=reflections)
            else:
                prompt = cot_generate_solution_space_prompt.format(case=generate_solution_space_case1, previous_samples=previous_samples, input=input, failed_reflections=reflections)
            sample = await self._aask_json_format(prompt, system_msgs=[DS_AGENT_SYSTEM_MSG], stop=stop)
            print(f"sample {count} is {sample}")
            # 由于 GLM4flashx模型可能会输出之前的一些步骤，因此这里需要检查Thought次数，并过滤之前的思考过程
            response = extract_last_thought_json(sample[0])
            # 尝试提取json代码块，如果没有```结构，则通过{}提取
            thought_line = CodeParser.parse_code(block=None, text=response, lang='json')
            print(f"CodeParser extracted thought_line is {thought_line}")
            if thought_line == response:
                input_str, json_dict = try_parse_json_object(response)
                thought = json_dict
                print(f"try_parse_json_object extracted thought_json is {thought}, type is {type(thought)}")
            else:
                try:
                    thought = json5.loads(thought_line)
                except Exception:
                    thought = json5.loads(fix_json(thought_line))
                print(f"CodeParser extracted thought_json is {thought}, type is {type(thought)}")
            sampled_cases.append(thought.get("thought", ""))
            solution_space.append({
                "thought": thought,
                "response": response
            })
            if thought.get("task_type", "other") == "finish":
                break
        return solution_space

    def calculate_final_score(self, goal_score: float, code_score: float, prospect_score: float, d: int) -> float:
        clamped_depth = max(0, min(d, 10))
        goal_weight = 0.2 + 0.3 * (clamped_depth / 10)      # 0.2->0.5
        code_weight = 0.3 + 0.1 * (clamped_depth / 10)      # 0.3->0.4
        prospect_weight = 0.5 - 0.4 * (clamped_depth / 10)  # 0.5->0.1
        weighted_sum = (
                goal_score * goal_weight
                + code_score * code_weight
                + prospect_score * prospect_weight
        )
        print(f"final_score: {weighted_sum}, goal_score: {goal_score}, code_score: {code_score}, prospect_score: {prospect_score}, depth: {d}")
        return weighted_sum

    async def evaluate_current_trajectory(self, goal, explore_trajectories, n_evaluate_sample, depth: int):
        # func 根据整体目标goal和当前探索的路径explore_trajectories 评估路径价值
        print(f"For node evaluation, Current explore_trajectories: {explore_trajectories}")
        prompt = trajectory_value_prompt_reasoning.format(goal=goal, trajectory=explore_trajectories)
        values = []
        for count in range(n_evaluate_sample):
            rsp = await self._aask_with_config(prompt)
            rsp = rsp[0]
            print(f"evaluate_current_trajectory rsp: {rsp}\n")
            try:
                value_dict = extract_evaluation_scores(rsp)
                value = self.calculate_final_score(value_dict['goal_score'], value_dict['code_score'], value_dict['prospect_score'], depth)
            except Exception as e:
                value_dict = extract_evaluation_scores(rsp)
                value = value_dict['correctness_score']
            # 保险措施
            if (value < 1 and value != 0) or value > 10:
                value_dict = extract_evaluation_scores(rsp)
                value = value_dict['correctness_score']
            # try:
            #     value = extract_final_score(rsp)
            # except Exception as e:
            #     value_dict = extract_evaluation_scores(rsp)
            #     value = value_dict['correctness_score']
            values.append(value)
        print(f"For node evaluation, VALUES: {values}")
        return sum(values) / len(values)

    async def evaluate_terminal_trajectory(self, goal, explore_trajectories, n_evaluate_sample, reflection_map, failed_trajectories):
        # func 根据整体目标goal和当前探索的路径explore_trajectories 评估路径价值
        prompt = terminal_trajectory_value_prompt_reasoning.format(goal=goal, trajectory=explore_trajectories)
        values = []
        for count in range(n_evaluate_sample):
            rsp = await self._aask_with_config(prompt)
            rsp = rsp[0]
            print(f"evaluate_terminal_trajectory rsp: {rsp}\n")
            try:
                value = extract_final_score(rsp)
            except Exception as e:
                value_dict = extract_evaluation_scores(rsp)
                value = value_dict['correctness_score']
            values.append(value)
        print(f"Terminal node VALUES: {values}")
        return sum(values) / len(values)

    async def get_samples(self, goal, x, y, n_generate_sample, prompt_sample, reflection_map, failed_trajectories, stop="Observation"):
        unique_trajectories = get_unique_trajectories(failed_trajectories)
        if len(unique_trajectories) > len(reflection_map) and len(unique_trajectories) < 4:
            print("generating reflections")
            reflection_map = goal.generate_self_reflection(unique_trajectories, x)
        if prompt_sample == 'cot':
            prompt = self._cot_prompt_wrap(x, y, reflection_map)
        else:
            raise ValueError(f'prompt_sample {prompt_sample} not recognized')

        # print(f"tips prompt: {prompt}")
        samples = await self._aask_with_config(prompt, system_msgs=[DS_AGENT_SYSTEM_MSG], n=n_generate_sample, stop=stop)

        return [y + sample for sample in samples]

    async def get_value(self, goal, x, y, n_evaluate_sample, reflection_map, failed_trajectories):
        unique_trajectories = get_unique_trajectories(failed_trajectories)
        logger.info(f"Current x: {x}")
        logger.info(f"Current y: {y}")
        value_prompt = self._value_prompt_wrap(x, y, unique_trajectories, reflection_map)
        # logger.info(f"VALUE PROMPT: {value_prompt}")
        value_outputs = await self._aask_with_config(value_prompt, system_msgs=[DS_AGENT_SYSTEM_MSG], n=n_evaluate_sample)
        logger.info(f"VALUE OUTPUTS: {value_outputs}")
        values = [self._value_outputs_unwrap(value) for value in value_outputs]
        logger.info(f"VALUES: {values}")
        return sum(values) / len(values)

    def _cot_prompt_wrap(self, x: str, y: str = '', reflection_mapping_list=[], max_token_length=16000):
        question = x
        input = x + y
        trajectories = ""
        if reflection_mapping_list:
            # fixme: 重写带 reflection 的 prompt
            for reflection_mapping in reflection_mapping_list:
                traj_with_reflection = reflection_mapping['trajectory'] + "FAILED TRAJECTORY\nReflection: " + \
                                       reflection_mapping['reflection'] + "\n\n"
                trajectories += traj_with_reflection
            prompt = cot_prompt_feedback.format(trajectories=trajectories, input=input)
            if count_message_tokens(prompt) > max_token_length:
                print("Too long")
                trajectories = ""
                for reflection_mapping in reflection_mapping_list[:3]:
                    traj_with_reflection = reflection_mapping['trajectory'] + "FAILED TRAJECTORY\nReflection: " + \
                                           reflection_mapping['reflection'] + "\n\n"
                    trajectories += traj_with_reflection
                prompt = cot_prompt_feedback_short.format(trajectories=trajectories, input=input)
        else:
            # fixed: prompt已重写
            prompt = cot_prompt.format(input=input)
            messages = [{
                "role": "user",
                "content": prompt
            }]
            if count_message_tokens(messages) > max_token_length:
                prompt = cot_prompt_short.format(input=input)
        return prompt

    def _value_prompt_wrap(self, x: str, y: str, z: list = [], reflections: list = [], max_token_length=16000):
        question = x
        if len(z) != 0:
            failed_trajectories = ""
            # Combine the trajectories with their corresponding reflections
            for traj, ref in zip(z, reflections):
                failed_trajectories += f"{question}\n{traj}\nThis trajectory is incorrect as {ref['reflection']}\nThus the correctness score is 1\n"
            inp = x + y + "\nThis trajectory is "
            prompt = value_prompt_reasoning_feedback.format(s="", trajectories=failed_trajectories, input=inp)
            if count_message_tokens(prompt) > max_token_length:
                prompt = value_prompt_reasoning_feedback_short.format(s="", trajectories=failed_trajectories, input=inp)
        inp = y + "\nThis trajectory is "
        prompt = value_prompt_reasoning.format(s="", input=inp)
        return prompt

    def _value_outputs_unwrap(self, evaluate_prompt):
        if '10' in evaluate_prompt:
            return 1.0
        elif '9' in evaluate_prompt:
            return 0.9
        elif '8' in evaluate_prompt:
            return 0.8
        elif '7' in evaluate_prompt:
            return 0.7
        elif '6' in evaluate_prompt:
            return 0.6
        elif '5' in evaluate_prompt:
            return 0.5
        elif '4' in evaluate_prompt:
            return 0.4
        elif '3' in evaluate_prompt:
            return 0.3
        elif '2' in evaluate_prompt:
            return 0.2
        elif '1' in evaluate_prompt:
            return 0.1
        else:
            return -1


class ExecuteAction(Action):
    async def step(self, goal, thought, code, executor: ExecuteNbCode):
        result, success = await executor.run(code)
        # todo: 结合当前节点的执行结果，评估整体代码的完成效果 (原先这里是根据测试用例来计算分数)
        observation = result           # "当前执行状态——观察"
        reward = 1 if success else 0   # 当前执行状态的奖励分数
        done = success                 # 当前节点是否完成了所有的任务
        info = {}                      # 其他信息
        return observation, reward, done, info
